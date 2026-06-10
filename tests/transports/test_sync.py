"""Integration tests for the sync transport against an in-process HTTP server."""
from __future__ import annotations

import socket
import threading
import time
import json
from typing import Iterator

import pytest

from lm15.transports import (
    ConnectError,
    ConnectTimeout,
    ProtocolError,
    ReadError,
    ReadTimeout,
    TransportRequest,
    StdlibTransport,
    TransportError,
)

from .conftest import reply_bytes, reply_chunked


def test_simple_get(server) -> None:
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/hello")
        with t.stream(req) as resp:
            assert resp.status == 200
            body = b"".join(resp)
            assert body == b"ok"
        assert server.ctx.requests[0].method == "GET"
        assert server.ctx.requests[0].target == "/hello"
    finally:
        t.close()


def test_post_with_json_body(server) -> None:
    captured = {}

    def handler(req, client):
        captured["method"] = req.method
        captured["body"] = req.body
        captured["content_type"] = req.header("content-type")
        reply_bytes(client, 200, b'{"ok":true}',
                    headers=[("Content-Type", "application/json")])
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        body = json.dumps({"x": 1}).encode()
        req = TransportRequest(
            method="POST", url=f"{server.base_url()}/v1/x",
            headers=[("Content-Type", "application/json")],
            body=body,
        )
        with t.stream(req) as resp:
            assert resp.status == 200
            out = b"".join(resp)
        assert json.loads(out) == {"ok": True}
        assert captured["method"] == "POST"
        assert captured["body"] == body
        assert captured["content_type"] == "application/json"
    finally:
        t.close()


def test_host_header_includes_port_when_non_default(server) -> None:
    def handler(req, client):
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            b"".join(resp)
        host = server.ctx.requests[0].header("host")
        assert host == f"127.0.0.1:{server.port}"
    finally:
        t.close()


def test_chunked_response(server) -> None:
    def handler(req, client):
        reply_chunked(client, [b"foo", b"bar", b"baz"])
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            assert resp.status == 200
            chunks = list(resp)
        joined = b"".join(chunks)
        assert joined == b"foobarbaz"
    finally:
        t.close()


def test_streaming_yields_chunks_as_they_arrive(server) -> None:
    """Key property: streaming must actually stream, not buffer."""
    arrival_times: list[float] = []

    def handler(req, client):
        reply_chunked(client, [b"first", b"second", b"third"], chunk_delay=0.1)
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        start = time.monotonic()
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            for chunk in resp:
                arrival_times.append(time.monotonic() - start)
        # Three chunks 100ms apart — arrivals should be spaced, not all at end
        assert len(arrival_times) >= 3
        assert arrival_times[-1] - arrival_times[0] >= 0.15
    finally:
        t.close()


def test_sse_style_response(server) -> None:
    """SSE is just chunked text/event-stream — should work transparently."""
    def handler(req, client):
        reply_chunked(
            client,
            [
                b"event: message\ndata: hello\n\n",
                b"event: message\ndata: world\n\n",
                b"data: [DONE]\n\n",
            ],
            headers=[("Content-Type", "text/event-stream")],
            chunk_delay=0.02,
        )
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/v1/stream")
        with t.stream(req) as resp:
            assert resp.header("content-type") == "text/event-stream"
            body = b"".join(resp)
    finally:
        t.close()
    assert b"event: message" in body
    assert b"[DONE]" in body


def test_error_status_still_delivers_body(server) -> None:
    def handler(req, client):
        reply_bytes(client, 429, b'{"error":"rate limit"}',
                    headers=[("Content-Type", "application/json")])
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            assert resp.status == 429
            body = b"".join(resp)
        assert b"rate limit" in body
    finally:
        t.close()


def test_content_length_response(server) -> None:
    payload = b"x" * 2048

    def handler(req, client):
        reply_bytes(client, 200, payload)
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            received = b"".join(resp)
        assert received == payload
    finally:
        t.close()


def test_connect_refused_raises_connect_error() -> None:
    t = StdlibTransport()
    try:
        # port 1 is reliably closed
        req = TransportRequest(method="GET", url="http://127.0.0.1:1/")
        with pytest.raises(ConnectError):
            t.stream(req).__enter__()
    finally:
        t.close()


def test_connect_timeout() -> None:
    t = StdlibTransport(connect_timeout=0.2)
    try:
        # TEST-NET-1 address — non-routable, connect will hang
        req = TransportRequest(method="GET", url="http://192.0.2.1/")
        with pytest.raises((ConnectTimeout, ConnectError)):
            t.stream(req).__enter__()
    finally:
        t.close()


def test_read_timeout(server) -> None:
    def handler(req, client):
        # Send headers, then hang
        client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n")
        time.sleep(3.0)
    server.ctx.handler = handler

    t = StdlibTransport(read_timeout=0.3)
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises((ReadTimeout, ReadError, TransportError)):
            with t.stream(req) as resp:
                b"".join(resp)
    finally:
        t.close()


def test_response_closed_on_early_break(server) -> None:
    """If the caller breaks out mid-body, the connection must be closed
    (not returned to the pool) since the body is half-consumed."""
    def handler(req, client):
        reply_chunked(client, [b"one"] * 100, chunk_delay=0.01)
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            for i, chunk in enumerate(resp):
                if i >= 3:
                    break
        # Now the pool must NOT have this connection cached as idle
        assert t.pool_stats()["idle"] == 0
        # But a follow-up request should still work
        req2 = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req2) as resp2:
            b"".join(resp2)
    finally:
        t.close()


def test_keepalive_reuses_connection(server) -> None:
    def handler(req, client):
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        for _ in range(3):
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            with t.stream(req) as resp:
                b"".join(resp)
        # Three requests, one connection
        assert server.ctx.request_count == 3
        assert t.pool_stats()["total_opened"] == 1
    finally:
        t.close()


def test_pool_detects_server_close_and_reconnects(server) -> None:
    """If the server closes the keepalive connection after a response,
    the transport must notice and open a new connection."""
    call_count = [0]

    def handler(req, client):
        call_count[0] += 1
        # First request: respond normally then close
        reply_bytes(
            client, 200, b"one",
            headers=[("Connection", "close")],
        )
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as r:
            b"".join(r)
        # Server closed; idle pool should be empty
        assert t.pool_stats()["idle"] == 0

        with t.stream(req) as r:
            b"".join(r)
        assert call_count[0] == 2
        assert t.pool_stats()["total_opened"] == 2
    finally:
        t.close()


def test_multiple_hosts_pool_separately(server) -> None:
    def handler(req, client):
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        # Same server, but address 127.0.0.1 vs localhost resolve differently
        for host in ("127.0.0.1", "localhost"):
            req = TransportRequest(method="GET", url=f"http://{host}:{server.port}/")
            try:
                with t.stream(req) as r:
                    b"".join(r)
            except ConnectError:
                # localhost may not resolve on some CI — skip if so
                pass
    finally:
        t.close()


def test_concurrent_requests_from_threads(server) -> None:
    """Pool should handle concurrent sync requests from multiple threads."""
    def handler(req, client):
        time.sleep(0.05)
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibTransport(max_connections=5)
    results: list[bytes] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker():
        try:
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            with t.stream(req) as r:
                out = b"".join(r)
            with lock:
                results.append(out)
        except Exception as e:
            with lock:
                errors.append(e)

    try:
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert results == [b"ok"] * 10
    finally:
        t.close()


def test_auth_header_passed_through(server) -> None:
    captured = []

    def handler(req, client):
        captured.append(req.header("authorization"))
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibTransport()
    try:
        req = TransportRequest(
            method="GET", url=f"{server.base_url()}/",
            headers=[("Authorization", "Bearer sk-test-xxx")],
        )
        with t.stream(req) as r:
            b"".join(r)
        assert captured == ["Bearer sk-test-xxx"]
    finally:
        t.close()


def test_error_then_recovery(server) -> None:
    """A request that errors mid-body must not leave the pool broken."""
    def handler_fail(req, client):
        client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
        client.close()

    def handler_ok(req, client):
        reply_bytes(client, 200, b"ok")

    server.ctx.handler = handler_fail
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises((ProtocolError, ReadError, TransportError)):
            with t.stream(req) as r:
                b"".join(r)

        # Pool must be clean — follow-up request succeeds
        server.ctx.handler = handler_ok
        with t.stream(req) as r:
            out = b"".join(r)
        assert out == b"ok"
    finally:
        t.close()


def test_read_returns_single_response_not_partial_stream(server) -> None:
    """After consuming one response, the connection must be positioned
    exactly at the next response boundary (for keepalive to work)."""
    def handler(req, client):
        reply_bytes(client, 200, b"first")

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        for expected in (b"first", b"first"):
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            with t.stream(req) as r:
                out = b"".join(r)
            assert out == expected
        # Second request should have reused the connection
        assert t.pool_stats()["total_opened"] == 1
    finally:
        t.close()
