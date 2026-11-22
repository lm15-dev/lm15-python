"""Proxy support: absolute-URI forwarding, CONNECT tunnels, env pickup.

The stub proxy here plays both roles a real forward proxy has:
    - plain HTTP: it receives the absolute-URI request and answers itself
      (no upstream needed — the assertions are about what the transport sent);
    - CONNECT: it replies 200 and then pumps bytes to a real upstream, so
      the TLS handshake genuinely runs end-to-end through the tunnel.
"""
from __future__ import annotations

import asyncio
import base64
import socket
import threading

import pytest

from lm15.transports import (
    ConnectError,
    StdlibAsyncTransport,
    StdlibTransport,
    TransportRequest,
)


# ─── stub forward proxy ──────────────────────────────────────────────


class _StubProxy:
    """Single-purpose forward proxy for tests. Records every request head."""

    def __init__(self, connect_upstream: tuple[str, int] | None = None, connect_status: int = 200) -> None:
        self.request_lines: list[str] = []
        self.headers: list[list[tuple[str, str]]] = []
        self._connect_upstream = connect_upstream
        self._connect_status = connect_status
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._sock.getsockname()[1]}"

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def _read_head(self, client: socket.socket) -> bytes:
        buf = b""
        while b"\r\n\r\n" not in buf:
            data = client.recv(65536)
            if not data:
                return buf
            buf += data
        return buf

    def _record(self, head: bytes) -> tuple[str, dict[str, str]]:
        lines = head.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
        self.request_lines.append(lines[0])
        parsed = [tuple(line.split(": ", 1)) for line in lines[1:] if ": " in line]
        self.headers.append(parsed)  # type: ignore[arg-type]
        return lines[0], {k.lower(): v for k, v in parsed}

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        try:
            head = self._read_head(client)
            if not head:
                return
            request_line, _ = self._record(head)
            if request_line.startswith("CONNECT "):
                if self._connect_status != 200:
                    client.sendall(
                        f"HTTP/1.1 {self._connect_status} refused\r\ncontent-length: 0\r\n\r\n".encode()
                    )
                    return
                client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                assert self._connect_upstream is not None, "CONNECT test needs an upstream"
                upstream = socket.create_connection(self._connect_upstream, timeout=10)
                self._pump(client, upstream)
                return
            # Plain HTTP: answer in place.
            body = b"via-proxy"
            client.sendall(
                b"HTTP/1.1 200 OK\r\ncontent-length: %d\r\nconnection: close\r\n\r\n%s"
                % (len(body), body)
            )
        except OSError:
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _pump(self, a: socket.socket, b: socket.socket) -> None:
        def one_way(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                for s in (src, dst):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        t = threading.Thread(target=one_way, args=(b, a), daemon=True)
        t.start()
        one_way(a, b)
        t.join(timeout=5)


@pytest.fixture()
def proxy():
    p = _StubProxy()
    try:
        yield p
    finally:
        p.stop()


# ─── plain HTTP through the proxy ────────────────────────────────────


def test_explicit_proxy_sends_absolute_uri(proxy) -> None:
    t = StdlibTransport(proxy=proxy.url)
    try:
        req = TransportRequest(method="GET", url="http://upstream.invalid:8213/v1/x?q=1")
        with t.stream(req) as resp:
            assert resp.status == 200
            assert resp.read() == b"via-proxy"
    finally:
        t.close()
    assert proxy.request_lines == ["GET http://upstream.invalid:8213/v1/x?q=1 HTTP/1.1"]
    headers = {k.lower(): v for k, v in proxy.headers[0]}
    assert headers["host"] == "upstream.invalid:8213"
    assert "proxy-authorization" not in headers


def test_proxy_credentials_become_basic_auth(proxy) -> None:
    authed = proxy.url.replace("http://", "http://user:pw%40x@")
    t = StdlibTransport(proxy=authed)
    try:
        req = TransportRequest(method="GET", url="http://upstream.invalid/x")
        with t.stream(req) as resp:
            resp.read()
    finally:
        t.close()
    headers = {k.lower(): v for k, v in proxy.headers[0]}
    expected = "Basic " + base64.b64encode(b"user:pw@x").decode()
    assert headers["proxy-authorization"] == expected


def test_env_proxy_is_honored(proxy, monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    t = StdlibTransport()
    try:
        with t.stream(TransportRequest(method="GET", url="http://upstream.invalid/x")) as resp:
            assert resp.read() == b"via-proxy"
    finally:
        t.close()
    assert proxy.request_lines, "request should have gone through the proxy"


def test_no_proxy_bypasses(server, proxy, monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    t = StdlibTransport()
    try:
        with t.stream(TransportRequest(method="GET", url=f"{server.base_url()}/hello")) as resp:
            assert resp.status == 200
    finally:
        t.close()
    assert proxy.request_lines == []
    assert server.ctx.request_count == 1


def test_trust_env_false_ignores_proxy_vars(server, proxy, monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    t = StdlibTransport(trust_env=False)
    try:
        with t.stream(TransportRequest(method="GET", url=f"{server.base_url()}/hello")) as resp:
            assert resp.status == 200
    finally:
        t.close()
    assert proxy.request_lines == []


def test_non_http_proxy_scheme_refuses() -> None:
    t = StdlibTransport(proxy="socks5://127.0.0.1:1080")
    try:
        with pytest.raises(ValueError, match="plain-HTTP proxies only"):
            t.stream(TransportRequest(method="GET", url="http://x.invalid/"))
    finally:
        t.close()


# ─── CONNECT tunnels for TLS targets ─────────────────────────────────


def test_tls_target_tunnels_through_connect(tls_server) -> None:
    host, port = tls_server._sock.getsockname()[:2]
    p = _StubProxy(connect_upstream=(host, port))
    t = StdlibTransport(proxy=p.url, ca_bundle=tls_server.ca_bundle_path())
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/hello")
        with t.stream(req) as resp:
            assert resp.status == 200
            resp.read()
    finally:
        t.close()
        p.stop()
    assert p.request_lines == [f"CONNECT {host}:{port} HTTP/1.1"]
    assert tls_server.ctx.request_count == 1
    # The tunneled request itself is origin-form, invisible to the proxy.
    (req_seen,) = tls_server.ctx.requests
    assert req_seen.target == "/hello"


def test_proxy_refusing_connect_raises_connect_error(tls_server) -> None:
    p = _StubProxy(connect_status=403)
    t = StdlibTransport(proxy=p.url, ca_bundle=tls_server.ca_bundle_path())
    try:
        with pytest.raises(ConnectError, match="refused CONNECT"):
            t.stream(TransportRequest(method="GET", url=f"{tls_server.base_url()}/hello"))
    finally:
        t.close()
        p.stop()


# ─── async mirror ────────────────────────────────────────────────────


def test_async_explicit_proxy_sends_absolute_uri(proxy) -> None:
    async def run() -> bytes:
        t = StdlibAsyncTransport(proxy=proxy.url)
        try:
            req = TransportRequest(method="GET", url="http://upstream.invalid:8213/v1/x")
            async with t.stream(req) as resp:
                assert resp.status == 200
                return await resp.read()
        finally:
            await t.aclose()

    body = asyncio.run(run())
    assert body == b"via-proxy"
    assert proxy.request_lines == ["GET http://upstream.invalid:8213/v1/x HTTP/1.1"]


def test_async_tls_target_tunnels_through_connect(tls_server) -> None:
    host, port = tls_server._sock.getsockname()[:2]
    p = _StubProxy(connect_upstream=(host, port))

    async def run() -> int:
        t = StdlibAsyncTransport(proxy=p.url, ca_bundle=tls_server.ca_bundle_path())
        try:
            req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/hello")
            async with t.stream(req) as resp:
                await resp.read()
                return resp.status
        finally:
            await t.aclose()

    try:
        status = asyncio.run(run())
    finally:
        p.stop()
    assert status == 200
    assert p.request_lines == [f"CONNECT {host}:{port} HTTP/1.1"]


def test_async_env_proxy_is_honored(proxy, monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)
    monkeypatch.delenv("NO_PROXY", raising=False)

    async def run() -> bytes:
        t = StdlibAsyncTransport()
        try:
            async with t.stream(TransportRequest(method="GET", url="http://upstream.invalid/x")) as resp:
                return await resp.read()
        finally:
            await t.aclose()

    assert asyncio.run(run()) == b"via-proxy"


def test_proxy_auth_secrecy() -> None:
    """AUTH-5: ProxyRoute repr hides basic_auth and parse_proxy_url redacts userinfo on error."""
    from lm15.transports._proxy import parse_proxy_url

    secret = "SECRET-PROXY-PASS-999"
    route = parse_proxy_url(f"http://alice:{secret}@proxy.example.com:8080")
    assert route.basic_auth is not None
    assert secret not in repr(route)
    assert route.basic_auth not in repr(route)

    with pytest.raises(ValueError) as excinfo:
        parse_proxy_url(f"https://alice:{secret}@proxy.example.com:8443/path")
    assert secret not in str(excinfo.value)
    assert "<redacted>@proxy.example.com:8443" in str(excinfo.value)
