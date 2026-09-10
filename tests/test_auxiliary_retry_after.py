"""Exercise every shared auxiliary HTTP-error branch through real adapters."""
import asyncio
import datetime
import json
from collections import deque

import pytest

from lm15 import (
    BatchRequest, FileUploadRequest, ImageGenerationRequest, Message, Request,
    RateLimitError, SpeechGenerationRequest, VideoGenerationRequest,
)
from lm15.providers import GeminiLM, OpenAILM
from lm15.providers.async_base import AsyncGeminiLM, AsyncOpenAILM
from lm15.testing import FakeResponse, FakeTransport
from lm15.transports import AsyncTransportResponse

PREFIX = Request(model="gemini-2.5-flash", messages=(Message.user("hello"),))
BATCH = BatchRequest(requests=(Request(model="gpt-4.1-mini", messages=PREFIX.messages),))
UPLOAD = FileUploadRequest(filename="test.txt", bytes_data=b"hello", media_type="text/plain")
BATCH_DONE = {"id": "batch_1", "status": "completed", "output_file_id": "file_output"}
VIDEO_DONE = {"id": "video_1", "status": "completed", "model": "sora-2"}

# name, arguments, keyword arguments, successful responses before the failure.
CASES = [
    ("list_models", (), {}, []),
    ("file_upload", (UPLOAD,), {}, []),
    ("file_get", ("file_1",), {}, []),
    ("file_list", (), {}, []),
    ("file_delete", ("file_1",), {}, []),
    ("file_download", ("file_1",), {}, []),
    ("batch_submit", (BATCH,), {}, []),  # upload fails
    ("batch_submit", (BATCH,), {}, [{"id": "file_input"}]),  # submission fails
    ("batch_status", ("batch_1",), {}, []),
    ("batch_results", ("batch_1",), {}, []),
    ("batch_results", ("batch_1",), {}, [BATCH_DONE]),  # download fails
    ("batch_cancel", ("batch_1",), {}, []),
    ("batch_list", (), {}, []),
    ("cache_create", (PREFIX,), {"ttl_seconds": 60}, []),
    ("cache_get", ("cachedContents/test",), {}, []),
    ("cache_list", (), {}, []),
    ("cache_delete", ("cachedContents/test",), {}, []),
    ("cache_update", ("cachedContents/test",), {"ttl_seconds": 60}, []),
    ("video_submit", (VideoGenerationRequest(model="sora-2", prompt="hello"),), {}, []),
    ("video_status", ("video_1",), {}, []),
    ("video_result", ("video_1",), {}, []),
    ("video_result", ("video_1",), {}, [VIDEO_DONE]),  # download fails
    ("video_list", (), {}, []),
    ("image_generate", (ImageGenerationRequest(model="gpt-image-1", prompt="hello"),), {}, []),
    ("speech_generate", (SpeechGenerationRequest(model="gpt-4o-mini-tts", prompt="hello"),), {}, []),
]


class AsyncTransport:
    def __init__(self, responses, cancel=False):
        self.responses = deque(responses)
        self.cancel = cancel
        self.requests = []
        self.releases = []

    def stream(self, request):
        self.requests.append(request)
        response = self.responses.popleft()

        async def chunks():
            if self.cancel and response.status >= 400:
                raise asyncio.CancelledError()
            yield response.body

        async def release(consumed):
            self.releases.append(consumed)

        return AsyncTransportResponse(
            status=response.status, reason="scripted", headers=response.headers,
            http_version="HTTP/1.1", chunks=chunks(), release=release,
        )


def make_client(name, successful, headers, asynchronous, *, cancel=False):
    gemini = name.startswith("cache_")
    payload = ({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "try later"}}
               if gemini else {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "try later"}})
    responses = [FakeResponse(status=200, body=json.dumps(body).encode(),
                              headers=[("Retry-After", "77")]) for body in successful]
    responses.append(FakeResponse(status=429, body=json.dumps(payload).encode(), headers=headers))
    if asynchronous:
        transport = AsyncTransport(responses, cancel=cancel)
        cls = AsyncGeminiLM if gemini else AsyncOpenAILM
    else:
        transport = FakeTransport(responses)
        cls = GeminiLM if gemini else OpenAILM
    return cls(api_key="fake", transport=transport), transport


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-step{len(c[3])+1}" for c in CASES])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("headers", "expected"), [
    ([("Retry-After", "9")], 9.0),
    ([("rEtRy-AfTeR", "Wed, 01 Jan 2025 00:00:09 GMT")], 9.0),
    ([], None),
    ([("Retry-After", "soonish")], None),
])
def test_auxiliary_headers(monkeypatch, case, asynchronous, headers, expected):
    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2025, 1, 1, tzinfo=datetime.timezone.utc)

    monkeypatch.setattr(datetime, "datetime", Clock)
    name, args, kwargs, successful = case
    client, transport = make_client(name, successful, headers, asynchronous)

    async def run():
        try:
            return await getattr(client, name)(*args, **kwargs)
        finally:
            await client.aclose()

    with pytest.raises(RateLimitError) as caught:
        if asynchronous:
            asyncio.run(run())
        else:
            with client:
                getattr(client, name)(*args, **kwargs)
    assert caught.value.retry_after == expected
    assert caught.value.status == 429
    assert caught.value.provider_code == ("RESOURCE_EXHAUSTED" if name.startswith("cache_") else "rate_limit_exceeded")
    assert len(transport.requests) == len(successful) + 1
    if asynchronous:
        assert len(transport.releases) == len(transport.requests)


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-step{len(c[3])+1}" for c in CASES])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("delay", [0.0, 3.0])
def test_auxiliary_provider_delay_wins(monkeypatch, case, asynchronous, delay):
    name, args, kwargs, successful = case
    client, transport = make_client(name, successful, [("Retry-After", "9")], asynchronous)
    cls = GeminiLM if name.startswith("cache_") else OpenAILM
    error = RateLimitError("try later", status=429, retry_after=delay)
    monkeypatch.setattr(cls, "normalize_error", lambda *args: error)

    async def run():
        try:
            await getattr(client, name)(*args, **kwargs)
        finally:
            await client.aclose()

    with pytest.raises(RateLimitError) as caught:
        if asynchronous:
            asyncio.run(run())
        else:
            with client:
                getattr(client, name)(*args, **kwargs)
    assert caught.value is error
    assert error.retry_after == delay
    assert len(transport.requests) == len(successful) + 1
    if asynchronous:
        assert len(transport.releases) == len(transport.requests)


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-step{len(c[3])+1}" for c in CASES])
def test_auxiliary_cancellation(case):
    name, args, kwargs, successful = case
    client, transport = make_client(name, successful, [("Retry-After", "9")], True, cancel=True)

    async def run():
        try:
            with pytest.raises(asyncio.CancelledError):
                await getattr(client, name)(*args, **kwargs)
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(transport.requests) == len(successful) + 1
    assert len(transport.releases) == len(transport.requests)
