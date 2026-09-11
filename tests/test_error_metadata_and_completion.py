"""HTTP diagnostics and terminal stream failures remain truthful in every driver."""

import asyncio

import pytest

from lm15 import (
    AuthError, Message, OpenAIChatLM, RateLimitError, Request, StreamAssemblyError,
    StreamDeltaEvent, StreamEndEvent, StreamErrorEvent, StreamStartEvent,
    TextDelta, TransportError, Usage,
)
from lm15.providers.async_base import AsyncOpenAIChatLM
from lm15.providers.base import _attach_error_metadata, _retry_after_seconds
from lm15.result import (
    AsyncResponseStream, ResponseStream, acoalesce_stream, amaterialize_response,
    coalesce_stream, materialize_response,
)
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import ErrorDetail
from tests.test_auxiliary_retry_after import AsyncTransport

REQUEST = Request(model="test", messages=(Message.user("hello"),))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["complete", "stream", "list_models"])
@pytest.mark.parametrize("header", ["X-Request-ID", "request-id", "x-amzn-requestid", "X-MS-Request-ID"])
def test_http_diagnostics_survive_sync_async_and_streaming(asynchronous, operation, header):
    response = FakeResponse(status=429, body=b'{"error":{"type":"rate_limit_error","message":"wait"}}',
                            headers=[(header, "request-1"), ("rEtRy-AfTeR", "3")])
    transport = AsyncTransport([response]) if asynchronous else FakeTransport([response])
    client = (AsyncOpenAIChatLM if asynchronous else OpenAIChatLM)(api_key="fake", transport=transport)

    async def run():
        if operation == "stream":
            return [event async for event in client.stream(REQUEST)]
        return await (client.list_models() if operation == "list_models" else client.complete(REQUEST))

    with pytest.raises(RateLimitError) as caught:
        if asynchronous:
            asyncio.run(run())
        elif operation == "stream":
            list(client.stream(REQUEST))
        elif operation == "list_models":
            client.list_models()
        else:
            client.complete(REQUEST)
    assert caught.value.request_id == "request-1"
    assert caught.value.retry_after == 3.0
    assert caught.value.status == 429


@pytest.mark.parametrize("invalid", ["nan", "inf", "-1", "bad", float("inf"), True])
def test_invalid_retry_hints_are_not_timer_inputs(invalid):
    assert _retry_after_seconds(invalid) is None
    error = RateLimitError("wait", retry_after=invalid)
    _attach_error_metadata(error, [("Retry-After", "3")])
    assert error.retry_after == 3.0


def test_every_wire_error_code_has_a_canonical_class():
    from lm15.errors import canonical_error_code, error_class_for_code
    from lm15.types import ERROR_CODES

    for code in ERROR_CODES:
        assert canonical_error_code(error_class_for_code(code)) == code


def test_body_metadata_wins_and_absent_fields_stay_absent():
    error = RateLimitError("wait", retry_after=0, request_id="body-id")
    _attach_error_metadata(error, [("Retry-After", "3"), ("X-Request-ID", "header-id")])
    assert error.retry_after == 0.0
    assert error.request_id == "body-id"
    empty = RateLimitError("wait")
    _attach_error_metadata(empty, [])
    assert empty.retry_after is None and empty.request_id is None


class Source:
    def __init__(self, events, *, error=None, cleanup=None):
        self.events = iter(events)
        self.error = error
        self.cleanup = cleanup
        self.closed = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.error is not None:
                raise self.error
            raise

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration from None

    def close(self):
        self.closed += 1
        if self.cleanup is not None:
            raise self.cleanup

    async def aclose(self):
        self.close()


def consume(source, asynchronous, wrapper):
    async def run():
        # Inspect the library's await boundary: Python 3.10's Task.result()
        # creates a new CancelledError when it escapes asyncio.run().
        try:
            events = acoalesce_stream(source) if wrapper == "coalesced" else source
            if wrapper == "response-stream":
                return await AsyncResponseStream(events, REQUEST).response()
            return await amaterialize_response(events, REQUEST)
        except BaseException as exc:
            return exc

    if asynchronous:
        result = asyncio.run(run())
        if isinstance(result, BaseException):
            raise result
        return result
    events = coalesce_stream(source) if wrapper == "coalesced" else source
    if wrapper == "response-stream":
        return ResponseStream(events, REQUEST).response
    return materialize_response(events, REQUEST)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapper", ["one-shot", "response-stream", "coalesced"])
def test_missing_end_is_not_materialized_as_success(asynchronous, wrapper):
    source = Source([StreamStartEvent(), StreamDeltaEvent(TextDelta("partial"))])
    with pytest.raises(StreamAssemblyError, match="without a completion event") as caught:
        consume(source, asynchronous, wrapper)
    assert caught.value.partial.text == "partial"
    assert source.closed == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapper", ["one-shot", "response-stream", "coalesced"])
@pytest.mark.parametrize("kind", ["exception", "event", "cancellation"])
def test_cleanup_does_not_replace_primary_failure(asynchronous, wrapper, kind):
    primary = asyncio.CancelledError() if kind == "cancellation" else AuthError("denied")
    cleanup = RuntimeError("close failed")
    source = Source(
        [StreamErrorEvent(ErrorDetail(code="auth", message="denied"))] if kind == "event" else [],
        error=None if kind == "event" else primary, cleanup=cleanup,
    )
    with pytest.raises(type(primary)) as caught:
        consume(source, asynchronous, wrapper)
    if kind != "event":
        assert caught.value is primary
    assert caught.value.cleanup_errors == (cleanup,)
    assert source.closed == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapper", ["one-shot", "response-stream", "coalesced"])
def test_cleanup_after_completion_is_nonretryable_and_carries_response(asynchronous, wrapper):
    cleanup = TransportError("close failed")
    source = Source([StreamStartEvent(), StreamDeltaEvent(TextDelta("ok")),
                     StreamEndEvent(finish_reason="stop", usage=Usage(input_tokens=2, output_tokens=1))], cleanup=cleanup)
    with pytest.raises(StreamAssemblyError) as caught:
        consume(source, asynchronous, wrapper)
    assert caught.value.__cause__ is cleanup
    assert caught.value.partial.text == "ok"
    assert caught.value.partial.usage.total_tokens == 3
    assert source.closed == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapper", ["one-shot", "response-stream"])
def test_event_after_end_is_rejected(asynchronous, wrapper):
    source = Source([StreamStartEvent(), StreamEndEvent(finish_reason="stop"), StreamDeltaEvent(TextDelta("late"))])
    with pytest.raises(StreamAssemblyError, match="after completion"):
        consume(source, asynchronous, wrapper)
    assert source.closed == 1
