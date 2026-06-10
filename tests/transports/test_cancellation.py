"""Cancellation and early-termination correctness.

These tests verify the "stop the stream mid-flight" contract that lm15
needs for the 'wrong-answer detection' use case.  They ensure that:
  - half-consumed connections are never pooled
  - cancelling an async task doesn't leak connections
  - exceptions during iteration close the connection
"""
from __future__ import annotations

import asyncio
import time

import pytest

from lm15.transports import TransportRequest, StdlibAsyncTransport, StdlibTransport
from .conftest import reply_chunked, reply_bytes


# ─── Sync early-break does not pool connection ───────────────────────


def test_break_mid_stream_closes_connection(server) -> None:
    def handler(req, client):
        reply_chunked(client, [b"x"] * 100, chunk_delay=0.01)

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with t.stream(req) as resp:
            for i, _ in enumerate(resp):
                if i >= 2:
                    break
        assert t.pool_stats()["idle"] == 0
        assert t.pool_stats()["in_use"] == 0
    finally:
        t.close()


def test_exception_in_iteration_closes_connection(server) -> None:
    def handler(req, client):
        reply_chunked(client, [b"x"] * 100, chunk_delay=0.01)

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises(RuntimeError):
            with t.stream(req) as resp:
                for i, _ in enumerate(resp):
                    if i >= 2:
                        raise RuntimeError("boom")
        assert t.pool_stats()["idle"] == 0
        assert t.pool_stats()["in_use"] == 0
    finally:
        t.close()


def test_break_then_new_request_works(server) -> None:
    def handler(req, client):
        reply_chunked(client, [b"y"] * 50, chunk_delay=0.01)

    server.ctx.handler = handler
    t = StdlibTransport()
    try:
        for _ in range(3):
            req = TransportRequest(method="GET", url=f"{server.base_url()}/")
            with t.stream(req) as resp:
                for i, _ in enumerate(resp):
                    if i >= 2:
                        break
        # Each request opens fresh (never reused since we always broke early)
        assert t.pool_stats()["total_opened"] >= 3
        assert t.pool_stats()["idle"] == 0
    finally:
        t.close()


# ─── Async cancellation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_cancelled_mid_stream_closes_connection(server):
    def handler(req, client):
        reply_chunked(client, [b"z"] * 1000, chunk_delay=0.05)

    server.ctx.handler = handler
    t = StdlibAsyncTransport()

    async def run():
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        async with t.stream(req) as resp:
            async for _ in resp:
                await asyncio.sleep(0)

    try:
        task = asyncio.create_task(run())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert t.pool_stats()["idle"] == 0
        assert t.pool_stats()["in_use"] == 0
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_async_exception_closes_connection(server):
    def handler(req, client):
        reply_chunked(client, [b"x"] * 100, chunk_delay=0.01)

    server.ctx.handler = handler
    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        with pytest.raises(RuntimeError):
            async with t.stream(req) as resp:
                i = 0
                async for _ in resp:
                    i += 1
                    if i >= 3:
                        raise RuntimeError("boom")
        assert t.pool_stats()["idle"] == 0
        assert t.pool_stats()["in_use"] == 0
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_gather_cancellation_cleans_up_all(server):
    """Cancelling a gather while N streams are running must clean up all."""
    def handler(req, client):
        reply_chunked(client, [b"x"] * 500, chunk_delay=0.02)

    server.ctx.handler = handler
    t = StdlibAsyncTransport(max_connections=5)

    async def one():
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        async with t.stream(req) as resp:
            async for _ in resp:
                await asyncio.sleep(0)

    try:
        group = asyncio.gather(*[one() for _ in range(5)], return_exceptions=True)
        await asyncio.sleep(0.1)
        group.cancel()
        await asyncio.gather(group, return_exceptions=True)
        # Allow any background cleanup
        await asyncio.sleep(0.05)
        assert t.pool_stats()["idle"] == 0
        assert t.pool_stats()["in_use"] == 0
    finally:
        await t.aclose()


# ─── "Wrong answer" scenario ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_early_on_detected_wrong_answer(server):
    """The lm15 use case: stream, detect bad content, abort.
    Verify we saved tokens (got far less than full response)."""
    full_response = [b"data: token_" + str(i).encode() + b"\n\n"
                     for i in range(1000)]

    def handler(req, client):
        reply_chunked(client, full_response, chunk_delay=0.005,
                      headers=[("Content-Type", "text/event-stream")])

    server.ctx.handler = handler
    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{server.base_url()}/")
        received_events = 0
        async with t.stream(req) as resp:
            async for chunk in resp:
                received_events += chunk.count(b"data:")
                if received_events >= 3:
                    break  # "wrong answer" detected
        assert received_events >= 3
        assert received_events < 100  # We stopped way early
        assert t.pool_stats()["idle"] == 0
    finally:
        await t.aclose()
