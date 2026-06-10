"""
Async transport built on `asyncio.open_connection`.

Mirror of _sync.py but async end-to-end.  Key differences:

- `asyncio.open_connection` gives us a (reader, writer) pair, not a raw socket.
- Stale-connection detection: we peek at the reader's transport socket with
  select (fd-level), same trick as the sync side.
- Cancellation correctness: if the caller's task is cancelled mid-stream,
  we must close the writer (not return it to the pool), then re-raise
  CancelledError without awaiting anything that could itself be cancelled.
- Timeouts use `asyncio.wait_for`.  We avoid `asyncio.timeout` (3.11+) to
  stay Python 3.10-compatible.
"""
from __future__ import annotations

import asyncio
import select
import socket
import ssl
from typing import AsyncIterator

from ._exceptions import (
    ConnectError,
    ConnectTimeout,
    ProtocolError,
    ReadError,
    ReadTimeout,
    TransportError,
    WriteError,
    WriteTimeout,
)
from ._http11 import (
    ResponseHeadParser,
    build_request_head,
)
from ._ssl import make_ssl_context
from ._types import AsyncTransportResponse, TransportRequest
from ._url import ParsedURL, parse_url


_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 60.0
_DEFAULT_WRITE_TIMEOUT = 60.0
_READ_CHUNK = 64 * 1024


# ─── Async connection wrapper ────────────────────────────────────────


class _AsyncConnection:
    __slots__ = ("origin", "reader", "writer", "closed")

    def __init__(
        self,
        origin: tuple[str, str, int],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.origin = origin
        self.reader = reader
        self.writer = writer
        self.closed = False

    def is_stale(self) -> bool:
        if self.closed:
            return True
        # Look at the underlying socket's FD for readability
        try:
            sock = self.writer.get_extra_info("socket")
        except Exception:
            return True
        if sock is None:
            return True
        try:
            r, _, _ = select.select([sock], [], [], 0)
            if not r:
                return False
            # Readable on an idle keepalive means EOF/close_notify or
            # unexpected unread bytes, so the connection is not safe to reuse.
            # Avoid MSG_PEEK: TLS sockets/proxies may reject non-zero recv()
            # flags, and peeking is unnecessary once select() reports readable.
            return True
        except (OSError, ValueError):
            return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.writer.close()
        except Exception:
            pass


# ─── Async pool ──────────────────────────────────────────────────────


class _AsyncConnectionPool:
    def __init__(self, max_connections: int) -> None:
        self._max = max_connections
        self._idle: dict[tuple[str, str, int], list[_AsyncConnection]] = {}
        self._in_use: set[_AsyncConnection] = set()
        self._lock = asyncio.Lock()
        self._slot = asyncio.Semaphore(max_connections)
        self._total_opened = 0
        self._closed = False

    async def acquire_slot(self, timeout: float | None = None) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        if timeout is None:
            await self._slot.acquire()
            return
        try:
            await asyncio.wait_for(self._slot.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TransportError("timed out waiting for connection pool slot")

    def release_slot(self) -> None:
        self._slot.release()

    async def checkout(self, origin: tuple[str, str, int]) -> _AsyncConnection | None:
        async with self._lock:
            q = self._idle.get(origin)
            while q:
                conn = q.pop()
                if conn.is_stale():
                    conn.close()
                    continue
                self._in_use.add(conn)
                return conn
            return None

    async def checkin(self, conn: _AsyncConnection) -> None:
        async with self._lock:
            self._in_use.discard(conn)
            if self._closed or conn.closed:
                conn.close()
                return
            self._idle.setdefault(conn.origin, []).append(conn)

    async def discard(self, conn: _AsyncConnection) -> None:
        async with self._lock:
            self._in_use.discard(conn)
        conn.close()

    async def register_new(self, conn: _AsyncConnection) -> None:
        async with self._lock:
            self._in_use.add(conn)
            self._total_opened += 1

    def stats(self) -> dict:
        return {
            "idle": sum(len(q) for q in self._idle.values()),
            "in_use": len(self._in_use),
            "total_opened": self._total_opened,
        }

    async def close_all(self) -> None:
        async with self._lock:
            self._closed = True
            for q in self._idle.values():
                for conn in q:
                    conn.close()
            self._idle.clear()
            for conn in list(self._in_use):
                conn.close()
            self._in_use.clear()


# ─── Transport ───────────────────────────────────────────────────────


class StdlibAsyncTransport:
    def __init__(
        self,
        *,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        write_timeout: float = _DEFAULT_WRITE_TIMEOUT,
        max_connections: int = 10,
        verify: bool = True,
        ca_bundle: str | None = None,
        user_agent: str = "lm15/stdlib",
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._user_agent = user_agent
        self._verify = verify
        self._ca_bundle = ca_bundle
        self._ssl_ctx: ssl.SSLContext | None = None
        self._pool = _AsyncConnectionPool(max_connections)
        self._closed = False

    def _get_ssl_ctx(self) -> ssl.SSLContext:
        if self._ssl_ctx is None:
            self._ssl_ctx = make_ssl_context(
                verify=self._verify, ca_bundle=self._ca_bundle
            )
        return self._ssl_ctx

    def pool_stats(self) -> dict:
        return self._pool.stats()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._pool.close_all()

    async def __aenter__(self) -> "StdlibAsyncTransport":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ─── Main entry point ───

    def stream(self, request: TransportRequest) -> "_AsyncStreamCM":
        """Return an async context manager that produces an AsyncTransportResponse."""
        return _AsyncStreamCM(self, request)

    async def _do_stream(self, request: TransportRequest) -> AsyncTransportResponse:
        if self._closed:
            raise TransportError("transport is closed")

        parsed = parse_url(request.url)
        connect_timeout = request.connect_timeout or self._connect_timeout
        read_timeout = request.read_timeout or self._read_timeout
        write_timeout = request.write_timeout or self._write_timeout

        await self._pool.acquire_slot(timeout=connect_timeout * 5)
        slot_released = {"done": False}

        def release_slot_once() -> None:
            if not slot_released["done"]:
                slot_released["done"] = True
                self._pool.release_slot()

        conn: _AsyncConnection | None = None
        try:
            attempt = 0
            while True:
                conn = (
                    await self._pool.checkout(parsed.origin()) if attempt == 0 else None
                )
                reused = conn is not None
                if conn is None:
                    conn = await self._open(parsed, connect_timeout=connect_timeout)
                    await self._pool.register_new(conn)

                try:
                    await self._send_request(
                        conn, request, parsed, write_timeout=write_timeout
                    )
                    break
                except (WriteError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                    if not reused or attempt > 0:
                        await self._pool.discard(conn)
                        if isinstance(exc, WriteError):
                            raise
                        raise WriteError(str(exc)) from exc
                    await self._pool.discard(conn)
                    conn = None
                    attempt += 1
                    continue

            head = await self._read_head(conn, read_timeout=read_timeout)
            decoder = head.body_decoder(request.method)
            keep_alive = head.keep_alive()

            release_conn = self._make_release(conn, keep_alive)

            async def chunks_gen() -> AsyncIterator[bytes]:
                aborted = False
                try:
                    if head.leftover:
                        for out in decoder.feed(head.leftover):
                            if out:
                                yield out
                        if decoder.complete:
                            return
                    while not decoder.complete:
                        try:
                            data = await asyncio.wait_for(
                                conn.reader.read(_READ_CHUNK),
                                timeout=read_timeout,
                            )
                        except asyncio.TimeoutError as exc:
                            aborted = True
                            raise ReadTimeout("read timed out mid-body") from exc
                        except asyncio.CancelledError:
                            aborted = True
                            raise
                        except OSError as exc:
                            aborted = True
                            raise ReadError(f"read failed: {exc}") from exc
                        if not data:
                            try:
                                decoder.eof()
                            except ProtocolError:
                                aborted = True
                                raise
                            break
                        try:
                            for out in decoder.feed(data):
                                if out:
                                    yield out
                        except ProtocolError:
                            aborted = True
                            raise
                finally:
                    if aborted:
                        release_slot_once()

            async def release_with_slot(body_consumed: bool) -> None:
                try:
                    await release_conn(body_consumed)
                finally:
                    release_slot_once()

            return AsyncTransportResponse(
                status=head.status,
                reason=head.reason,
                headers=head.headers,
                http_version=head.http_version,
                chunks=chunks_gen(),
                release=release_with_slot,
            )
        except BaseException:
            if conn is not None:
                await self._pool.discard(conn)
            release_slot_once()
            raise

    # ─── Connection I/O ───

    async def _open(
        self, parsed: ParsedURL, *, connect_timeout: float
    ) -> _AsyncConnection:
        ctx = self._get_ssl_ctx() if parsed.is_tls else None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host=parsed.host,
                    port=parsed.port,
                    ssl=ctx,
                    server_hostname=parsed.host if ctx else None,
                ),
                timeout=connect_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ConnectTimeout(
                f"timed out connecting to {parsed.host}:{parsed.port}"
            ) from exc
        except ssl.SSLError as exc:
            raise ConnectError(f"TLS handshake failed: {exc}") from exc
        except OSError as exc:
            raise ConnectError(
                f"failed to connect to {parsed.host}:{parsed.port}: {exc}"
            ) from exc

        # TCP_NODELAY on the underlying socket
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        return _AsyncConnection(parsed.origin(), reader, writer)

    async def _send_request(
        self,
        conn: _AsyncConnection,
        request: TransportRequest,
        parsed: ParsedURL,
        *,
        write_timeout: float,
    ) -> None:
        body = request.body or b""
        has_body = request.method.upper() not in ("GET", "HEAD", "DELETE") or bool(body)
        body_length = len(body) if has_body else None
        head = build_request_head(
            method=request.method,
            target=parsed.target,
            host=parsed.host,
            port=parsed.port,
            is_tls=parsed.is_tls,
            headers=request.headers,
            body_length=body_length,
            user_agent=self._user_agent,
        )
        try:
            conn.writer.write(head)
            if body_length is not None and body:
                conn.writer.write(body)
            await asyncio.wait_for(conn.writer.drain(), timeout=write_timeout)
        except asyncio.TimeoutError as exc:
            raise WriteTimeout(f"write timed out: {exc}") from exc
        except (BrokenPipeError, ConnectionResetError):
            raise
        except OSError as exc:
            raise WriteError(f"write failed: {exc}") from exc

    async def _read_head(
        self, conn: _AsyncConnection, *, read_timeout: float
    ) -> ResponseHeadParser:
        parser = ResponseHeadParser()
        while not parser.complete:
            try:
                data = await asyncio.wait_for(
                    conn.reader.read(_READ_CHUNK), timeout=read_timeout
                )
            except asyncio.TimeoutError as exc:
                raise ReadTimeout("read timed out waiting for headers") from exc
            except OSError as exc:
                raise ReadError(f"read failed: {exc}") from exc
            if not data:
                raise ReadError("server closed connection before sending response")
            parser.feed(data)
        return parser

    def _make_release(self, conn: _AsyncConnection, keep_alive: bool):
        async def release(body_consumed: bool) -> None:
            if body_consumed and keep_alive and not conn.closed:
                await self._pool.checkin(conn)
            else:
                await self._pool.discard(conn)
        return release


class _AsyncStreamCM:
    """Async context manager wrapper so callers can write:

        async with transport.stream(req) as resp:
            async for chunk in resp: ...
    """

    def __init__(self, transport: StdlibAsyncTransport, request: TransportRequest) -> None:
        self._transport = transport
        self._request = request
        self._response: AsyncTransportResponse | None = None

    def __await__(self):
        return self._transport._do_stream(self._request).__await__()

    async def __aenter__(self) -> AsyncTransportResponse:
        self._response = await self._transport._do_stream(self._request)
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._response is not None:
            await self._response.aclose()
