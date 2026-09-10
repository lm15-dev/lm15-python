"""Cancellation must win a race with completion, including on Python 3.10/3.11."""

import asyncio
from types import SimpleNamespace

import pytest

from lm15.transports._async import StdlibAsyncTransport, _AsyncConnectionPool
from lm15.transports._timeouts import wait_for
from lm15.transports._types import TransportRequest
from lm15.transports._url import parse_url


@pytest.mark.asyncio
async def test_simultaneous_completion_and_caller_cancellation():
    caller = asyncio.current_task()
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def finish_and_cancel():
        future.set_result("completed")
        caller.cancel()

    loop.call_soon(finish_and_cancel)
    with pytest.raises(asyncio.CancelledError):
        await wait_for(future, 1)


@pytest.mark.asyncio
async def test_resource_completed_during_cancellation_is_released():
    caller = asyncio.current_task()
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    resource = object()
    released = []

    def finish_and_cancel():
        future.set_result(resource)
        caller.cancel()

    loop.call_soon(finish_and_cancel)
    with pytest.raises(asyncio.CancelledError):
        await wait_for(future, 1, cancel_result=released.append)
    assert released == [resource]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0, -1, 1])
async def test_already_completed_result(timeout):
    future = asyncio.get_running_loop().create_future()
    future.set_result("ok")
    released = []
    assert await wait_for(future, timeout, cancel_result=released.append) == "ok"
    assert released == []


@pytest.mark.asyncio
async def test_timeout_cancels_and_drains_operation():
    cleaned = []

    async def operation():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    with pytest.raises(asyncio.TimeoutError):
        await wait_for(operation(), .01)
    assert cleaned == [True]


@pytest.mark.asyncio
async def test_zero_timeout_does_not_start_coroutine():
    started = []

    async def operation():
        started.append(True)

    with pytest.raises(asyncio.TimeoutError):
        await wait_for(operation(), 0)
    assert not started


@pytest.mark.asyncio
async def test_exception_preserved():
    error = ValueError("original")

    async def operation():
        raise error

    with pytest.raises(ValueError) as caught:
        await wait_for(operation(), 1)
    assert caught.value is error


@pytest.mark.asyncio
async def test_operation_can_suppress_timeout_cancellation():
    async def operation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "salvaged"

    assert await wait_for(operation(), .01) == "salvaged"


@pytest.mark.asyncio
async def test_caller_cancellation_during_timeout_cleanup_stays_cancellation():
    caller = asyncio.current_task()
    loop = asyncio.get_running_loop()

    async def operation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            loop.call_soon(caller.cancel)
            await asyncio.sleep(0)
            raise

    with pytest.raises(asyncio.CancelledError):
        await wait_for(operation(), .01)


@pytest.mark.asyncio
async def test_write_drain_completion_cannot_swallow_cancellation():
    caller = asyncio.current_task()
    loop = asyncio.get_running_loop()

    class Writer:
        def write(self, data):
            pass

        def drain(self):
            future = loop.create_future()

            def finish_and_cancel():
                future.set_result(None)
                caller.cancel()

            loop.call_soon(finish_and_cancel)
            return future

    transport = StdlibAsyncTransport(trust_env=False)
    request = TransportRequest(method="POST", url="http://localhost/v1", body=b"{}")
    try:
        with pytest.raises(asyncio.CancelledError):
            await transport._send_request(SimpleNamespace(writer=Writer()), request, parse_url(request.url),
                                          proxy=None, write_timeout=1)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_pool_permit_acquired_during_cancellation_is_returned(monkeypatch):
    pool = _AsyncConnectionPool(1)
    caller = asyncio.current_task()
    original = pool._slot.acquire

    async def acquire():
        result = await original()
        asyncio.get_running_loop().call_soon(caller.cancel)
        return result

    monkeypatch.setattr(pool._slot, "acquire", acquire)
    with pytest.raises(asyncio.CancelledError):
        await pool.acquire_slot(timeout=1)
    assert pool._slot._value == 1
