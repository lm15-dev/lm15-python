"""Integration tests for the async transport."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from lm15.transports import (
    ConnectError,
    ConnectTimeout,
    ProtocolError,
    ReadError,
    ReadTimeout,
    TransportRequest,
    StdlibAsyncTransport,
    TransportError,
)

from .conftest import reply_bytes, reply_chunked


@pytest.mark.asyncio
async def test_simple_get(server):
    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/hello")
        async with t.stream(req) as resp:
            assert resp.status == 200
            body = b""
            async for chunk in resp:
                body += chunk
        assert body == b"ok"
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_post_with_json_body(server):
    captured = {}

    def handler(req, client):
        captured["method"] = req.method
        captured["body"] = req.body
        reply_bytes(client, 200, b'{"ok":true}',
                    headers=[("Content-Type", "application/json")])
    server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        body = json.dumps({"x": 1}).encode()
        req = TransportRequest(
            method="POST", url=f"{server.base_url()}/v1/x",
            headers=[("Content-Type", "application/json")], body=body,
        )
        async with t.stream(req) as resp:
            assert resp.status == 200
            out = b""
            async for chunk in resp:
                out += chunk
        assert json.loads(out) == {"ok": True}
        assert captured["body"] == body
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_chunked_response(server):
    def handler(req, client):
        reply_chunked(client, [b"foo", b"bar", b"baz"])
    server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        async with t.stream(req) as resp:
            chunks = []
            async for c in resp:
                chunks.append(c)
        assert b"".join(chunks) == b"foobarbaz"
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_streaming_yields_chunks_as_they_arrive(server):
    arrival_times: list[float] = []

    def handler(req, client):
        reply_chunked(client, [b"a", b"b", b"c"], chunk_delay=0.1)
    server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        start = time.monotonic()
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        async with t.stream(req) as resp:
            async for _ in resp:
                arrival_times.append(time.monotonic() - start)
        assert len(arrival_times) >= 3
        assert arrival_times[-1] - arrival_times[0] >= 0.15
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_sse_style_response(server):
    def handler(req, client):
        reply_chunked(
            client,
            [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"],
            headers=[("Content-Type", "text/event-stream")],
            chunk_delay=0.02,
        )
    server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/v1/stream")
        async with t.stream(req) as resp:
            assert resp.header("content-type") == "text/event-stream"
            body = b""
            async for c in resp:
                body += c
        assert b"[DONE]" in body
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_connect_refused():
    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url="http://127.0.0.1:1/")
        with pytest.raises(ConnectError):
            async with t.stream(req):
                pass
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_connect_timeout():
    t = StdlibAsyncTransport(connect_timeout=0.2)
    try:
        req = TransportRequest(method="GET", url="http://192.0.2.1/")
        with pytest.raises((ConnectTimeout, ConnectError)):
            async with t.stream(req):
                pass
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_read_timeout(server):
    def handler(req, client):
        client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n")
        time.sleep(3.0)
    server.ctx.handler = handler

    t = StdlibAsyncTransport(read_timeout=0.3)
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises((ReadTimeout, ReadError, TransportError)):
            async with t.stream(req) as resp:
                async for _ in resp:
                    pass
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_cancellation_closes_connection(server):
    """Cancelling the task mid-stream must close the connection cleanly."""
    def handler(req, client):
        reply_chunked(client, [b"x"] * 1000, chunk_delay=0.05)
    server.ctx.handler = handler

    t = StdlibAsyncTransport()

    async def run():
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        async with t.stream(req) as resp:
            async for _ in resp:
                await asyncio.sleep(0)  # yield to allow cancellation

    try:
        task = asyncio.create_task(run())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Pool must not hold this half-consumed connection
        assert t.pool_stats()["idle"] == 0

        # And a fresh request still works
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        # Replace the handler with one that completes
        server.ctx.handler = lambda req, client: reply_bytes(client, 200, b"ok")
        async with t.stream(req) as resp:
            out = b""
            async for c in resp:
                out += c
        assert out == b"ok"
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_keepalive_reuses_connection(server):
    def handler(req, client):
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        for _ in range(3):
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            async with t.stream(req) as resp:
                async for _ in resp:
                    pass
        assert server.ctx.request_count == 3
        assert t.pool_stats()["total_opened"] == 1
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_concurrent_requests(server):
    """Many concurrent async requests, pool caps connections."""
    def handler(req, client):
        time.sleep(0.05)
        reply_bytes(client, 200, b"ok")
    server.ctx.handler = handler

    t = StdlibAsyncTransport(max_connections=5)
    try:
        async def one():
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            async with t.stream(req) as resp:
                out = b""
                async for c in resp:
                    out += c
            return out

        results = await asyncio.gather(*[one() for _ in range(20)])
        assert results == [b"ok"] * 20
        # Pool must have opened at most `max_connections`
        assert t.pool_stats()["total_opened"] <= 5
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_early_break_closes_connection(server):
    def handler(req, client):
        reply_chunked(client, [b"x"] * 1000, chunk_delay=0.01)
    server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        async with t.stream(req) as resp:
            i = 0
            async for _ in resp:
                i += 1
                if i >= 3:
                    break
        assert t.pool_stats()["idle"] == 0
    finally:
        await t.aclose()
