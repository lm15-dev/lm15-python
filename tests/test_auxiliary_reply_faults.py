"""INV-054 auxiliary reply boundaries; offline sync/async regression matrix.

Source added during the code-before-tests repair pass; not executed in that pass.
"""
import inspect
import json

import pytest

from lm15 import Message, Request
from lm15.errors import ProviderError, RETRYABLE_ERRORS, UnsupportedFeatureError
from lm15.providers import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM, TypeSafeLM, XaiLM
from lm15.providers.async_base import (
    AsyncAnthropicLM, AsyncGeminiLM, AsyncOpenAIChatLM, AsyncOpenAILM,
    AsyncTypeSafeLM, AsyncXaiLM,
)
from lm15.providers.base import HttpResponse, _parse_reply
from lm15.testing import FakeResponse, FakeTransport
from lm15.transports import AsyncTransportResponse
from lm15.types import (
    BatchRequest, FileUploadRequest, ImageGenerationRequest, SpeechGenerationRequest,
    VideoGenerationRequest,
)


CLASSES = {
    "openai": (OpenAILM, AsyncOpenAILM),
    "openai-chat": (OpenAIChatLM, AsyncOpenAIChatLM),
    "anthropic": (AnthropicLM, AsyncAnthropicLM),
    "gemini": (GeminiLM, AsyncGeminiLM),
    "typesafe": (TypeSafeLM, AsyncTypeSafeLM),
    "xai": (XaiLM, AsyncXaiLM),
}
REQUEST = Request(model="gpt-5-mini", messages=(Message.user("hello"),))
PREFIX = Request(model="gemini-2.5-flash", messages=(Message.user("prefix"),))
BATCH = BatchRequest(requests=(REQUEST,))
UPLOAD = FileUploadRequest(filename="test.txt", bytes_data=b"hello")
VIDEO = VideoGenerationRequest(model="sora-2", prompt="a bird")
IMAGE = ImageGenerationRequest(model="gpt-image-1", prompt="a bird")
SPEECH = SpeechGenerationRequest(model="tts-1", prompt="hello")


class AsyncFakeTransport(FakeTransport):
    def stream(self, request):
        scripted = super().stream(request)

        async def chunks():
            yield scripted.body

        async def release(consumed):
            pass

        return AsyncTransportResponse(
            status=scripted.status, reason=scripted.reason, headers=scripted.headers,
            http_version=scripted.http_version, chunks=chunks(), release=release,
        )


def headers(request_id="aux-id", content_type="application/json"):
    return [
        ("Content-Type", content_type), ("APIM-Request-ID", request_id),
        ("Retry-After", "invalid"), ("X-MS-Retry-After-MS", "1250"),
        ("X-RateLimit-Remaining-Requests", "3"),
        ("x-ratelimit-remaining-requests", "2"), ("Authorization", "must-not-copy"),
    ]


def wire(body, *, request_id="aux-id", status=200, content_type="application/json"):
    return FakeResponse(status=status, body=body, headers=headers(request_id, content_type))


def make_lm(provider, asynchronous, replies):
    transport = (AsyncFakeTransport if asynchronous else FakeTransport)(replies)
    return CLASSES[provider][int(asynchronous)](api_key="k", transport=transport), transport


async def invoke(lm, method, *args, **kwargs):
    result = getattr(lm, method)(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def assert_evidence(error, provider, *, request_id="aux-id", status=200):
    assert type(error) is ProviderError
    assert error.code == "provider" and not isinstance(error, RETRYABLE_ERRORS)
    assert error.provider == provider and error.status == status
    assert error.request_id == request_id and error.retry_after == 1.25
    assert dict(error.rate_limit_headers) == {
        "retry-after": ("invalid",), "x-ms-retry-after-ms": ("1250",),
        "x-ratelimit-remaining-requests": ("3", "2"),
    }
    with pytest.raises(TypeError):
        error.rate_limit_headers["retry-after"] = ("99",)


# Every public auxiliary parsing driver, plus each provider's model hook.
CASES = [
    *((p, "list_models", (), {}, ()) for p in CLASSES),
    ("openai", "file_upload", (UPLOAD,), {}, ()),
    ("openai", "file_get", ("file-1",), {}, ()),
    ("openai", "file_list", (), {}, ()),
    ("anthropic", "file_get", ("file-1",), {}, ()),
    ("gemini", "file_get", ("files/1",), {}, ()),
    ("openai", "batch_submit", (BATCH,), {}, (b'{"id":"input-file"}',)),
    ("anthropic", "batch_submit", (BATCH,), {}, ()),
    ("gemini", "batch_submit", (BatchRequest(requests=(PREFIX,)),), {}, ()),
    ("openai", "batch_status", ("batch-1",), {}, ()),
    ("openai", "batch_results", ("batch-1",), {}, ()),
    ("openai", "batch_cancel", ("batch-1",), {}, ()),
    ("openai", "batch_list", (), {}, ()),
    ("gemini", "cache_create", (PREFIX,), {}, ()),
    ("gemini", "cache_get", ("cachedContents/1",), {}, ()),
    ("gemini", "cache_list", (), {}, ()),
    ("gemini", "cache_update", ("cachedContents/1",), {"ttl_seconds": 60}, ()),
    ("openai", "video_submit", (VIDEO,), {}, ()),
    ("openai", "video_status", ("video-1",), {}, ()),
    ("openai", "video_result", ("video-1",), {}, ()),
    ("openai", "video_list", (), {}, ()),
    ("gemini", "video_status", ("operations/1",), {}, ()),
    ("xai", "video_status", ("video-1",), {}, ()),
    ("openai", "image_generate", (IMAGE,), {}, ()),
    ("gemini", "speech_generate", (SpeechGenerationRequest(model="gemini-2.5-flash-preview-tts", prompt="hi"),), {}, ()),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("provider,method,args,kwargs,prior", CASES)
@pytest.mark.parametrize("body,content_type", [
    (b"<html>gateway broke</html>", "text/html"),
    (b'{"truncated":', "application/json"),
    (b'\xff', "application/json"),
])
async def test_auxiliary_non_json_matrix(asynchronous, provider, method, args, kwargs, prior, body, content_type):
    lm, transport = make_lm(provider, asynchronous, [
        *(wire(b, request_id="prior-id") for b in prior), wire(body, content_type=content_type),
    ])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, method, *args, **kwargs)
    assert_evidence(caught.value, lm.provider)
    assert "not JSON" in caught.value.message and content_type in caught.value.message
    assert len(transport.requests) == len(prior) + 1  # no retry


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider,method,arg,body", [
    ("openai", "file_get", "file-1", b"[]"),  # field access on wrong shape
    ("anthropic", "file_get", "file-1", b"{}"),  # decoder ProviderError
    ("gemini", "cache_get", "cachedContents/1", b"{}"),
    ("openai", "batch_status", "batch-1", b"null"),
    ("openai", "video_status", "video-1", b'{"id":"video-1","status":"completed","progress":101}'),
    ("xai", "video_result", "video-1", b'{"status":"done"}'),
])
async def test_existing_provider_errors_and_shape_faults_gain_evidence(asynchronous, provider, method, arg, body):
    lm, _ = make_lm(provider, asynchronous, [wire(body)])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, method, arg)
    assert_evidence(caught.value, lm.provider)


@pytest.mark.parametrize("exc", [TypeError("bad shape"), ValueError("bad value"), KeyError("field")])
def test_pure_decoder_shape_errors_are_provider_faults(exc):
    def parse(reply):
        raise exc

    response = HttpResponse(200, "OK", headers(), b"{}", provider="test")
    with pytest.raises(ProviderError) as caught:
        _parse_reply(response, parse)
    assert_evidence(caught.value, "test")
    assert caught.value.__cause__ is exc
    assert "malformed provider reply" in caught.value.message


@pytest.mark.parametrize("exc", [RuntimeError("result not ready"), UnsupportedFeatureError("unsupported")])
def test_local_hook_control_errors_are_not_reclassified(exc):
    def parse(reply):
        raise exc

    with pytest.raises(type(exc)) as caught:
        _parse_reply(HttpResponse(200, "OK", [], b"{}"), parse)
    assert caught.value is exc


def test_existing_provider_error_identity_and_body_fields_are_preserved():
    error = ProviderError("decoder refused", provider="body-provider", status=202, request_id="body-id", retry_after=2)

    def parse(reply):
        raise error

    with pytest.raises(ProviderError) as caught:
        _parse_reply(HttpResponse(200, "OK", headers(), b"{}", provider="header-provider"), parse)
    assert caught.value is error
    assert (error.provider, error.status, error.request_id, error.retry_after) == ("body-provider", 202, "body-id", 2)
    assert error.rate_limit_headers["x-ratelimit-remaining-requests"] == ("3", "2")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("body", [b"<html>bad upload</html>", b"[]", b"{}"])
async def test_batch_upload_reply_faults_keep_upload_evidence(asynchronous, body):
    lm, transport = make_lm("openai", asynchronous, [wire(body, request_id="upload-id")])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, "batch_submit", BATCH)
    assert_evidence(caught.value, lm.provider, request_id="upload-id")
    assert len(transport.requests) == 1


BATCH_STATUS = json.dumps({
    "id": "batch-1", "status": "completed", "output_file_id": "output-file",
    "error_file_id": "error-file", "request_counts": {"total": 2},
}).encode()
ENTRY0 = b'{"custom_id":"0","response":{"status_code":400,"body":{"error":{"message":"first entry failed"}}}}'
ENTRY1 = b'{"custom_id":"1","response":{"status_code":400,"body":{"error":{"message":"second entry failed"}}}}'


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("bad", [b"<html>broken results</html>", b'{"custom_id":', b"[]", b'{"custom_id":"not-an-index"}', b"\xff"])
async def test_jsonl_faults_belong_to_the_offending_fetch_not_job_status(asynchronous, bad):
    lm, transport = make_lm("openai", asynchronous, [
        wire(BATCH_STATUS, request_id="status-id"),
        wire(b"\n" + ENTRY0 + b"\r\n\n", request_id="output-id"),
        wire(bad, request_id="error-file-id", status=206, content_type="application/jsonl"),
    ])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, "batch_results", "batch-1")
    assert_evidence(caught.value, lm.provider, request_id="error-file-id", status=206)
    assert "application/jsonl" in caught.value.message
    assert "status-id" not in str(caught.value) and "output-id" not in str(caught.value)
    assert "Body starts:" in caught.value.message
    assert len(transport.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_jsonl_blank_lines_and_multi_file_combination_are_preserved(asynchronous):
    lm, transport = make_lm("openai", asynchronous, [
        wire(BATCH_STATUS), wire(b"\n \r\n" + ENTRY0 + b"\n\n"), wire(b"\r\n" + ENTRY1 + b"\n\t\n"),
    ])
    entries = await invoke(lm, "batch_results", "batch-1")
    assert [entry.index for entry in entries] == [0, 1]
    assert [entry.error.message for entry in entries] == ["first entry failed", "second entry failed"]
    assert len(transport.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["batch_results", "video_result"])
async def test_pending_job_remains_a_local_value_error(asynchronous, method):
    lm, transport = make_lm("openai", asynchronous, [wire(b'{"id":"job-1","status":"in_progress"}')])
    with pytest.raises(ValueError, match="not finished"):
        await invoke(lm, method, "job-1")
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_user_credential_exception_outside_parser_is_not_masked(asynchronous):
    failure = ValueError("user credential failed")

    def credential():
        raise failure

    transport = (AsyncFakeTransport if asynchronous else FakeTransport)([])
    lm = CLASSES["openai"][int(asynchronous)](api_key=credential, transport=transport)
    with pytest.raises(ValueError) as caught:
        await invoke(lm, "list_models")
    assert caught.value is failure and not transport.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_binary_download_speech_video_and_empty_deletes_never_preparse_json(asynchronous, monkeypatch):
    def no_json(self):
        pytest.fail("binary/empty reply was parsed as JSON")

    lm, _ = make_lm("openai", asynchronous, [wire(b"\xff\x00raw"), wire(b"", status=204), wire(b"\xff\x00audio", content_type="audio/mpeg")])
    cache, _ = make_lm("gemini", asynchronous, [wire(b"", status=204)])
    with monkeypatch.context() as patch:
        patch.setattr(HttpResponse, "json", no_json)
        assert await invoke(lm, "file_download", "file-1") == b"\xff\x00raw"
        assert await invoke(lm, "file_delete", "file-1") is None
        assert await invoke(cache, "cache_delete", "cachedContents/1") is None
        speech = await invoke(lm, "speech_generate", SPEECH)
        assert speech.audio.bytes == b"\xff\x00audio"
    video, _ = make_lm("openai", asynchronous, [
        wire(b'{"id":"video-1","status":"completed"}'), wire(b"\xff\x00video", content_type="video/mp4"),
    ])
    part = await invoke(video, "video_result", "video-1")
    assert part.bytes == b"\xff\x00video"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_binary_video_decoder_failure_uses_download_headers(asynchronous):
    lm, _ = make_lm("openai", asynchronous, [
        wire(b'{"id":"video-1","status":"completed"}', request_id="status-id"),
        FakeResponse(status=206, body=b"video", headers=[h for h in headers("download-id") if h[0] != "Content-Type"]),
    ])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, "video_result", "video-1")
    assert_evidence(caught.value, lm.provider, request_id="download-id", status=206)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_jsonl_nested_decoder_fault_uses_real_fetch_status(asynchronous, monkeypatch):
    def broken_entry(self, request, response):
        raise ProviderError("invalid entry response", provider=self.provider, status=response.status)

    monkeypatch.setattr(OpenAILM, "parse_response", broken_entry)
    entry = b'{"custom_id":"0","response":{"status_code":200,"body":{"model":"m"}}}'
    lm, _ = make_lm("openai", asynchronous, [
        wire(BATCH_STATUS, request_id="status-id"), wire(entry, request_id="entry-id", status=206),
    ])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, "batch_results", "batch-1")
    assert_evidence(caught.value, lm.provider, request_id="entry-id", status=206)
    assert caught.value.message == "invalid entry response"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider,method,arg,body", [
    ("anthropic", "batch_results", "batch-1", b'{"id":"batch-1","processing_status":"ended","request_counts":{"succeeded":1}}'),
    ("gemini", "video_result", "operations/1", b'{"name":"operations/1","done":true}'),
])
async def test_missing_result_location_keeps_status_reply_evidence(asynchronous, provider, method, arg, body):
    lm, transport = make_lm(provider, asynchronous, [wire(body)])
    with pytest.raises(ProviderError) as caught:
        await invoke(lm, method, arg)
    assert_evidence(caught.value, lm.provider)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unsupported_and_local_validation_do_not_consume_a_reply(asynchronous):
    lm, transport = make_lm("xai", asynchronous, [])
    with pytest.raises(UnsupportedFeatureError):
        await invoke(lm, "video_list")
    cache, cache_transport = make_lm("gemini", asynchronous, [])
    with pytest.raises(ValueError, match="positive int"):
        await invoke(cache, "cache_update", "cachedContents/1", ttl_seconds=0)
    assert not transport.requests and not cache_transport.requests


def test_non_json_excerpt_and_header_snapshot_are_bounded_and_detached():
    raw_headers = headers()
    response = HttpResponse(200, "OK", raw_headers, b"<html>" + b"x" * 300 + b"SECRET-TAIL", provider="test")
    with pytest.raises(ProviderError) as caught:
        _parse_reply(response, lambda reply: reply.json())
    raw_headers.clear()
    assert_evidence(caught.value, "test")
    assert "<html>" in caught.value.message and "SECRET-TAIL" not in caught.value.message
