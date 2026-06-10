"""TLS integration tests.

Uses a self-signed cert produced by the fixture.  We pass the cert as a CA
bundle to verify it succeeds, and verify an untrusted bundle fails.
"""
from __future__ import annotations

import asyncio
import shutil
import socket
import time

import pytest

from lm15.transports import (
    ConnectError,
    TransportRequest,
    StdlibAsyncTransport,
    StdlibTransport,
)

from .conftest import reply_bytes


pytestmark = pytest.mark.skipif(
    shutil.which("openssl") is None,
    reason="openssl CLI required to provision self-signed cert",
)


def test_sync_tls_with_trusted_cert(tls_server):
    def handler(req, client):
        reply_bytes(client, 200, b"secure")
    tls_server.ctx.handler = handler

    t = StdlibTransport(ca_bundle=tls_server.ca_bundle_path())
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        with t.stream(req) as resp:
            body = b"".join(resp)
        assert body == b"secure"
    finally:
        t.close()


def test_sync_tls_with_untrusted_cert_rejected(tls_server):
    def handler(req, client):
        reply_bytes(client, 200, b"secure")
    tls_server.ctx.handler = handler

    # No CA bundle → system default, which won't trust our self-signed cert
    t = StdlibTransport()
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        with pytest.raises(ConnectError):
            with t.stream(req) as resp:
                b"".join(resp)
    finally:
        t.close()


def test_sync_tls_verify_false_accepts_self_signed(tls_server):
    def handler(req, client):
        reply_bytes(client, 200, b"secure")
    tls_server.ctx.handler = handler

    t = StdlibTransport(verify=False)
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        with t.stream(req) as resp:
            assert b"".join(resp) == b"secure"
    finally:
        t.close()


def test_sync_tls_stale_keepalive_reconnects_without_peeking(tls_server):
    calls = [0]

    def handler(req, client):
        calls[0] += 1
        reply_bytes(client, 200, f"secure-{calls[0]}".encode("ascii"))
        if calls[0] == 1:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()

    tls_server.ctx.handler = handler

    t = StdlibTransport(ca_bundle=tls_server.ca_bundle_path())
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        with t.stream(req) as resp:
            assert b"".join(resp) == b"secure-1"

        # Give the TLS close_notify/EOF a moment to become readable before the
        # pooled connection is checked out again.
        time.sleep(0.05)

        with t.stream(req) as resp:
            assert b"".join(resp) == b"secure-2"

        assert calls[0] == 2
        assert t.pool_stats()["total_opened"] == 2
    finally:
        t.close()


@pytest.mark.asyncio
async def test_async_tls_with_trusted_cert(tls_server):
    def handler(req, client):
        reply_bytes(client, 200, b"secure")
    tls_server.ctx.handler = handler

    t = StdlibAsyncTransport(ca_bundle=tls_server.ca_bundle_path())
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        async with t.stream(req) as resp:
            body = b""
            async for c in resp:
                body += c
        assert body == b"secure"
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_async_tls_untrusted_rejected(tls_server):
    def handler(req, client):
        reply_bytes(client, 200, b"secure")
    tls_server.ctx.handler = handler

    t = StdlibAsyncTransport()
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        with pytest.raises(ConnectError):
            async with t.stream(req) as resp:
                async for _ in resp:
                    pass
    finally:
        await t.aclose()


@pytest.mark.asyncio
async def test_async_tls_stale_keepalive_reconnects_without_peeking(tls_server):
    calls = [0]

    def handler(req, client):
        calls[0] += 1
        reply_bytes(client, 200, f"secure-{calls[0]}".encode("ascii"))
        if calls[0] == 1:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()

    tls_server.ctx.handler = handler

    t = StdlibAsyncTransport(ca_bundle=tls_server.ca_bundle_path())
    try:
        req = TransportRequest(method="GET", url=f"{tls_server.base_url()}/")
        async with t.stream(req) as resp:
            body = b""
            async for chunk in resp:
                body += chunk
        assert body == b"secure-1"

        await asyncio.sleep(0.05)

        async with t.stream(req) as resp:
            body = b""
            async for chunk in resp:
                body += chunk
        assert body == b"secure-2"

        assert calls[0] == 2
        assert t.pool_stats()["total_opened"] == 2
    finally:
        await t.aclose()
