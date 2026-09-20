"""Fetch bridge lifecycle tests with host doubles; real Pyodide remains separate."""
import asyncio
from types import SimpleNamespace

import pytest

from lm15.transports import FetchTransport, ProtocolError, ReadTimeout, TransportError, TransportRequest
from lm15.transports import _fetch


class Headers:
    def __init__(self, pairs=()):
        self.pairs = list(pairs)

    def append(self, name, value):
        self.pairs.append((name, value))

    def entries(self):
        return iter(self.pairs)


class Controller:
    def __init__(self):
        self.signal = self
        self.aborted = False
        self.event = asyncio.Event()

    def abort(self):
        self.aborted = True
        self.event.set()


class Reader:
    def __init__(self, chunks=(), *, blocked=False, fail=False):
        self.chunks = list(chunks)
        self.blocked = blocked
        self.fail = fail
        self.started = asyncio.Event()
        self.cancelled = False
        self.released = False

    async def read(self):
        self.started.set()
        if self.blocked:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("host body failed")
        if not self.chunks:
            return SimpleNamespace(done=True)
        chunk = self.chunks.pop(0)
        return SimpleNamespace(done=False, value=SimpleNamespace(to_py=lambda: chunk))

    async def cancel(self):
        self.cancelled = True

    def releaseLock(self):
        self.released = True


def host_response(reader=None, headers=()):
    return SimpleNamespace(status=200, statusText="OK", headers=Headers(headers),
                           body=SimpleNamespace(getReader=lambda: reader) if reader is not None else None)


@pytest.fixture
def bridge(monkeypatch):
    controllers = []

    def controller():
        c = Controller()
        controllers.append(c)
        return c

    monkeypatch.setattr(_fetch, "_js", SimpleNamespace(AbortController=SimpleNamespace(new=controller),
        Headers=SimpleNamespace(new=Headers), Object=SimpleNamespace(fromEntries=lambda x: x)))
    monkeypatch.setattr(_fetch, "to_js", lambda value, **kwargs: value)
    return controllers


def request(**kwargs):
    return TransportRequest(method="GET", url="https://example.test/stream", **kwargs)


@pytest.mark.asyncio
async def test_header_timeout_aborts_before_draining_host_cancellation(bridge):
    aborted_during_cleanup = []

    async def fetch(url, options):
        try:
            await asyncio.Event().wait()
        finally:
            aborted_during_cleanup.append(options["signal"].aborted)

    transport = FetchTransport(fetch=fetch, read_timeout=0.01)
    with pytest.raises(ReadTimeout, match="headers timed out.*lm15's read timeout"):
        await transport.stream(request())
    assert aborted_during_cleanup == [True]
    assert bridge[0].aborted and transport._controllers == {}


@pytest.mark.asyncio
async def test_cancel_before_headers_aborts_the_in_flight_request(bridge):
    started = asyncio.Event()

    async def fetch(url, options):
        started.set()
        await asyncio.Event().wait()

    transport = FetchTransport(fetch=fetch)
    task = asyncio.ensure_future(transport.stream(request()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge[0].aborted and transport._controllers == {}


@pytest.mark.asyncio
async def test_cancel_concurrent_with_header_acquisition_is_not_swallowed(bridge):
    started = asyncio.Event()
    ready = asyncio.get_running_loop().create_future()

    async def fetch(url, options):
        started.set()
        return await ready

    transport = FetchTransport(fetch=fetch)
    task = asyncio.ensure_future(transport.stream(request()))
    await started.wait()
    ready.set_result(host_response(Reader([b"body"])))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge[0].aborted and not transport._controllers


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [0.0, 0.01])
async def test_explicit_header_deadline_overrides_transport_default_even_zero(bridge, override):
    async def fetch(url, options):
        await asyncio.Event().wait()

    transport = FetchTransport(fetch=fetch, read_timeout=600)
    with pytest.raises(ReadTimeout):
        await transport.stream(request(read_timeout=override))
    assert bridge[0].aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["gzip", "x-gzip", "deflate", "gzip, deflate", "identity", None])
async def test_fetch_decoded_bytes_are_never_inflated_again(bridge, header):
    reader = Reader([b'{"answer":', b'42}'])
    options_seen = []

    async def fetch(url, options):
        options_seen.append(options)
        return host_response(reader, [("Content-Encoding", header)] if header else [])

    transport = FetchTransport(fetch=fetch)
    async with transport.stream(request()) as response:
        assert await response.read() == b'{"answer":42}'
    assert reader.released and not reader.cancelled and not bridge[0].aborted
    assert ("Accept-Encoding", "identity") in options_seen[0]["headers"].pairs
    assert not transport._controllers


@pytest.mark.asyncio
@pytest.mark.parametrize("coding", ["br", "zstd", "unknown", "gzip, br"])
async def test_visible_unsupported_coding_aborts_before_body_acquisition(bridge, coding):
    reader = Reader([b"already decoded"])

    async def fetch(url, options):
        return host_response(reader, [("Content-Encoding", coding)])

    transport = FetchTransport(fetch=fetch)
    with pytest.raises(ProtocolError, match=coding.split(", ")[-1]):
        await transport.stream(request())
    assert bridge[0].aborted and not reader.started.is_set() and not transport._controllers


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "cancel", "failure", "early_close"])
async def test_body_faults_and_early_close_abort_and_release_reader(bridge, mode):
    reader = Reader(blocked=mode in ("timeout", "cancel"), fail=mode == "failure")

    async def fetch(url, options):
        return host_response(reader)

    transport = FetchTransport(fetch=fetch, read_timeout=0.01 if mode == "timeout" else 600)
    response = await transport.stream(request())
    if mode == "early_close":
        await response.aclose()
    elif mode == "cancel":
        task = asyncio.create_task(response.read())
        await reader.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ReadTimeout if mode == "timeout" else TransportError):
            await response.read()
    await response.aclose()  # idempotent after an error, too
    assert bridge[0].aborted and reader.cancelled and reader.released
    assert not transport._controllers


@pytest.mark.asyncio
async def test_no_body_is_complete_not_an_abandoned_request(bridge):
    async def fetch(url, options):
        return host_response()

    transport = FetchTransport(fetch=fetch)
    async with transport.stream(request()) as response:
        assert await response.read() == b""
    assert not bridge[0].aborted and not transport._controllers


@pytest.mark.asyncio
async def test_transport_close_aborts_pending_headers_and_open_bodies(bridge):
    started = asyncio.Event()
    reader = Reader([b"pending"])

    async def fetch(url, options):
        if len(bridge) == 1:
            return host_response(reader)
        started.set()
        await options["signal"].event.wait()
        raise RuntimeError("aborted by host")

    transport = FetchTransport(fetch=fetch)
    response = await transport.stream(request())
    pending = asyncio.ensure_future(transport.stream(request()))
    await started.wait()
    await transport.aclose()
    assert reader.released and not transport._responses
    with pytest.raises(TransportError):
        await pending
    await response.aclose()
    assert all(c.aborted for c in bridge) and reader.released and not transport._controllers
    with pytest.raises(TransportError, match="closed"):
        await transport.stream(request())


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["connect_timeout", "write_timeout"])
async def test_unimplementable_socket_deadlines_are_not_silently_ignored(bridge, field):
    async def fetch(url, options):
        raise AssertionError("must refuse before fetch")

    with pytest.raises(TransportError, match=field):
        await FetchTransport(fetch=fetch).stream(request(**{field: 1}))
    assert bridge == []
