"""Edge cases and corner conditions that LLM APIs actually hit in production."""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from lm15.transports import (
    ProtocolError,
    ReadError,
    TransportRequest,
    StdlibAsyncTransport,
    StdlibTransport,
    TransportError,
)

from .conftest import reply_bytes, reply_chunked


# ─── No-body responses (204/304/HEAD) ────────────────────────────────


def test_204_no_body(server) -> None:
    def handler(req, client):
        client.sendall(b"HTTP/1.1 204 No Content\r\nX-Foo: bar\r\n\r\n")

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            assert resp.status == 204
            assert b"".join(resp) == b""
        # 204 is keep-alive — connection should be reused
        assert t.pool_stats()["idle"] == 1
        assert t.pool_stats()["total_opened"] == 1

        with t.stream(req) as resp:
            b"".join(resp)
        assert t.pool_stats()["total_opened"] == 1
    finally:
        t.close()


def test_304_no_body(server) -> None:
    def handler(req, client):
        client.sendall(b"HTTP/1.1 304 Not Modified\r\nETag: \"x\"\r\n\r\n")

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            assert resp.status == 304
            assert resp.header("etag") == '"x"'
            assert b"".join(resp) == b""
    finally:
        t.close()


def test_head_request_no_body(server) -> None:
    """HEAD responses can include a Content-Length header describing what
    a GET would return, but no body is sent."""
    def handler(req, client):
        assert req.method == "HEAD"
        client.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nContent-Type: text/plain\r\n\r\n"
        )

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="HEAD", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            assert resp.status == 200
            assert resp.header("content-length") == "100"
            assert b"".join(resp) == b""
        # Connection reusable
        assert t.pool_stats()["idle"] == 1
    finally:
        t.close()


# ─── Connection: close ───────────────────────────────────────────────


def test_server_connection_close_not_pooled(server) -> None:
    def handler(req, client):
        reply_bytes(client, 200, b"ok", headers=[("Connection", "close")])

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            assert b"".join(resp) == b"ok"
        assert t.pool_stats()["idle"] == 0
    finally:
        t.close()


# ─── Large chunked responses (SSE-sized) ─────────────────────────────


def test_large_chunked_response(server) -> None:
    payload = [b"event: x\ndata: " + (b"y" * 1000) + b"\n\n" for _ in range(200)]

    def handler(req, client):
        reply_chunked(client, payload, headers=[("Content-Type", "text/event-stream")])

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            body = b"".join(resp)
        assert body == b"".join(payload)
    finally:
        t.close()


# ─── Pool slot limit ─────────────────────────────────────────────────


def test_sync_pool_slot_limit_queues_requests(server) -> None:
    """With max_connections=2, 4 concurrent requests should serialize 2x2."""
    start_times: list[float] = []
    lock = threading.Lock()

    def handler(req, client):
        with lock:
            start_times.append(time.monotonic())
        time.sleep(0.2)
        reply_bytes(client, 200, b"ok")

    server.ctx.handler = handler
    t = StdlibTransport(max_connections=2)
    try:
        results: list[bytes] = []
        res_lock = threading.Lock()

        def worker():
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            with t.stream(req) as r:
                out = b"".join(r)
            with res_lock:
                results.append(out)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        t0 = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        elapsed = time.monotonic() - t0

        assert len(results) == 4
        # With 2 slots and 0.2s/req, total time must be >= 0.4s
        assert elapsed >= 0.35, f"elapsed={elapsed}"
    finally:
        t.close()


# ─── TransportRequest body: POST with explicit Content-Length ─────────────────


def test_post_with_body_host_header_correct(server) -> None:
    captured = []

    def handler(req, client):
        captured.append((req.method, req.target, req.body, req.header("host")))
        reply_bytes(client, 200, b"ok")

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(
            method="POST", url=f"{server.base_url()}/v1/messages?stream=true",
            headers=[("Content-Type", "application/json")],
            body=b'{"model":"x"}',
        )
        with t.stream(req) as r:
            b"".join(r)
        assert captured[0] == ("POST", "/v1/messages?stream=true", b'{"model":"x"}',
                               f"127.0.0.1:{server.port}")
    finally:
        t.close()


# ─── Server-sent events: realistic stream ────────────────────────────


def test_realistic_sse_stream(server) -> None:
    """Simulates an Anthropic-style SSE stream."""
    sse_events = [
        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
        b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n",
    ]
    for token in ["Hello", ", ", "world", "!"]:
        sse_events.append(
            f"event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{token}\"}}}}\n\n".encode()
        )
    sse_events.append(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
    sse_events.append(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
    sse_events.append(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")

    def handler(req, client):
        reply_chunked(client, sse_events, headers=[("Content-Type", "text/event-stream")],
                      chunk_delay=0.01)

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="POST", url=f"{server.base_url()}/v1/messages",
                      body=b'{"x":1}')
        with t.stream(req) as resp:
            assert resp.header("content-type") == "text/event-stream"
            body = b"".join(resp)
        assert b"message_start" in body
        assert b"Hello" in body
        assert b"message_stop" in body
        # Connection should be returned to the pool
        assert t.pool_stats()["idle"] == 1
    finally:
        t.close()


# ─── Async concurrent stream ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_realistic_sse_stream(server):
    events = [f"data: token{i}\n\n".encode() for i in range(50)]

    def handler(req, client):
        reply_chunked(client, events, headers=[("Content-Type", "text/event-stream")],
                      chunk_delay=0.005)

    server.ctx.handler = handler
    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="POST", url=f"{server.base_url()}/v1/stream",
                      body=b"{}")
        received: list[bytes] = []
        async with t.stream(req) as resp:
            assert resp.status == 200
            async for chunk in resp:
                received.append(chunk)
        body = b"".join(received)
        for i in range(50):
            assert f"token{i}".encode() in body
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_async_100_concurrent_sse_streams(server):
    """True async concurrency: 100 in-flight SSE streams on 10 pool slots."""
    def handler(req, client):
        events = [f"data: token{i}\n\n".encode() for i in range(3)]
        reply_chunked(client, events, headers=[("Content-Type", "text/event-stream")],
                      chunk_delay=0.01)

    server.ctx.handler = handler
    t = StdlibAsyncTransport(max_connections=10)
    try:
        async def one() -> int:
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            total = 0
            async with t.stream(req) as resp:
                async for chunk in resp:
                    total += len(chunk)
            return total

        results = await asyncio.gather(*[one() for _ in range(100)])
        assert len(results) == 100
        # Each response is 3 events — count should be uniform
        assert all(r > 0 for r in results)
        # Only opened at most 10 connections
        assert t.pool_stats()["total_opened"] <= 10
    finally:
        await t.aclose()


# ─── Headers without Content-Length or Transfer-Encoding ─────────────


def test_body_until_close(server) -> None:
    """HTTP/1.0-style: no framing headers; body ends on close."""
    def handler(req, client):
        client.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b"hello world"
        )
        # Don't close yet — let the client try to read
        time.sleep(0.05)
        # server closes its side
        try:
            client.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            body = b"".join(resp)
        assert body == b"hello world"
    finally:
        t.close()


# ─── User-provided headers override defaults ─────────────────────────


def test_user_agent_override(server) -> None:
    captured = []

    def handler(req, client):
        captured.append(req.header("user-agent"))
        reply_bytes(client, 200, b"ok")

    server.ctx.handler = handler
    t = StdlibTransport(user_agent="lm15-default/1.0")
    try:
        # Default UA
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as r:
            b"".join(r)

        # User override
        req = TransportRequest(
            method="GET", url=f"{server.base_url()}/",
            headers=[("User-Agent", "my-app/2.0")],
        )
        with t.stream(req) as r:
            b"".join(r)

        assert captured == ["lm15-default/1.0", "my-app/2.0"]
    finally:
        t.close()


# ─── Protocol error surfacing ────────────────────────────────────────


def test_malformed_server_response_raises_protocol_error(server) -> None:
    def handler(req, client):
        client.sendall(b"this is not HTTP\r\n\r\n")

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises((ProtocolError, TransportError)):
            with t.stream(req) as resp:
                b"".join(resp)
    finally:
        t.close()


def test_truncated_chunked_response(server) -> None:
    def handler(req, client):
        client.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"5\r\nhel"  # promise 5 bytes, deliver 3, then close
        )
        client.close()

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises((ProtocolError, ReadError, TransportError)):
            with t.stream(req) as resp:
                b"".join(resp)
    finally:
        t.close()
