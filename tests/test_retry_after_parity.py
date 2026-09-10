"""HTTP retry hints survive both synchronous and asynchronous inference errors."""
import asyncio
import datetime
import json

import pytest

from lm15 import Message, Request, RateLimitError
from lm15.providers import OpenAIChatLM
from lm15.providers.async_base import AsyncOpenAIChatLM
from lm15.testing import FakeResponse, FakeTransport
from lm15.transports import AsyncTransportResponse

REQUEST = Request(model="example", messages=(Message.user("hello"),))
BODY = json.dumps({"error": {
    "type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "try later",
}}).encode()


class AsyncTransport:
    def __init__(self, headers, *, cancel=False):
        self.headers = headers
        self.cancel = cancel
        self.requests = []
        self.released = False

    def stream(self, request):
        self.requests.append(request)

        async def chunks():
            if self.cancel:
                raise asyncio.CancelledError()
            yield BODY

        async def release(body_consumed):
            self.released = True

        return AsyncTransportResponse(
            status=429, reason="Too Many Requests", http_version="HTTP/1.1",
            headers=self.headers, chunks=chunks(), release=release,
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("headers", "expected"), [
    ([("Retry-After", "9")], 9.0),
    ([("rEtRy-AfTeR", "9")], 9.0),
    ([("Retry-After", "Wed, 01 Jan 2025 00:00:09 GMT")], 9.0),
    ([], None),
    ([("Retry-After", "soonish")], None),
    ([("Retry-After", "-1")], None),
])
def test_retry_after_parity(monkeypatch, stream, asynchronous, headers, expected):
    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2025, 1, 1, tzinfo=datetime.timezone.utc)

    monkeypatch.setattr(datetime, "datetime", Clock)
    if asynchronous:
        transport = AsyncTransport(headers)
        client = AsyncOpenAIChatLM(api_key="fake", transport=transport)

        async def run():
            try:
                if stream:
                    async for _ in client.stream(REQUEST):
                        pytest.fail("An HTTP error must not emit stream events")
                else:
                    await client.complete(REQUEST)
            finally:
                await client.aclose()

        invoke = lambda: asyncio.run(run())
    else:
        transport = FakeTransport([FakeResponse(status=429, body=BODY, headers=headers)])
        client = OpenAIChatLM(api_key="fake", transport=transport)
        invoke = lambda: list(client.stream(REQUEST)) if stream else client.complete(REQUEST)

    with pytest.raises(RateLimitError) as caught:
        invoke()
    error = caught.value
    assert error.retry_after == expected
    assert error.status == 429
    assert error.provider_code == "rate_limit_exceeded"
    assert len(transport.requests) == 1  # LM15 does not retry.
    if asynchronous:
        assert transport.released


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("delay", [0.0, 3.0])
def test_provider_delay_wins(monkeypatch, stream, delay):
    error = RateLimitError("try later", status=429, retry_after=delay)
    monkeypatch.setattr(OpenAIChatLM, "normalize_error", lambda *args: error)
    transport = AsyncTransport([("Retry-After", "9")])
    client = AsyncOpenAIChatLM(api_key="fake", transport=transport)

    async def run():
        try:
            with pytest.raises(RateLimitError) as caught:
                if stream:
                    async for _ in client.stream(REQUEST):
                        pytest.fail("Unexpected event")
                else:
                    await client.complete(REQUEST)
            assert caught.value is error
            assert error.retry_after == delay
        finally:
            await client.aclose()

    asyncio.run(run())
    assert transport.released
    assert len(transport.requests) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_cancellation_is_not_normalized(stream):
    transport = AsyncTransport([("Retry-After", "9")], cancel=True)
    client = AsyncOpenAIChatLM(api_key="fake", transport=transport)

    async def run():
        try:
            with pytest.raises(asyncio.CancelledError):
                if stream:
                    async for _ in client.stream(REQUEST):
                        pytest.fail("Unexpected event")
                else:
                    await client.complete(REQUEST)
        finally:
            await client.aclose()

    asyncio.run(run())
    assert transport.released
    assert len(transport.requests) == 1
