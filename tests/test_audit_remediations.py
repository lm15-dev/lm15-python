"""Regression tests for audit remediations (Workstreams 1 & 2):
- AUTH-5: RouterConfig.env hidden from repr
- MAP-13: Responses reasoning_format="none" dropped/refused; qwen/zai/qwen_chat_template substituted
- MAP-14: Anthropic structured_output="reject" still refuses probabilities="required"
- Gemini trailing usage-only SSE chunk preserves earlier finish_reason
- AsyncBaseProviderLM enforces _require gates and batch_submit adaptation policy
- StdlibTransport / StdlibAsyncTransport never retry WriteTimeout on reused pooled sockets
"""
from __future__ import annotations

import asyncio
import json

import pytest

from lm15 import (
    AnthropicLM,
    AsyncOpenAIChatLM,
    BatchRequest,
    Config,
    FileUploadRequest,
    GeminiLM,
    ImageGenerationRequest,
    Message,
    OpenAIChatLM,
    OpenAILM,
    Reasoning,
    Request,
    RouterConfig,
    SpeechGenerationRequest,
    UnsupportedFeatureError,
    VideoGenerationRequest,
)
from lm15.result import coalesce_stream
from lm15.sse import SSEEvent
from lm15.transports import (
    StdlibAsyncTransport,
    StdlibTransport,
    TransportRequest,
    WriteError,
    WriteTimeout,
)


def test_router_config_env_not_in_repr() -> None:
    secret = "SECRET-ENV-TOKEN-42"
    rc = RouterConfig(env={"OPENAI_API_KEY": secret})
    assert secret not in repr(rc)
    assert secret not in str(rc)


@pytest.mark.parametrize("effort", ["low", "off"])
def test_responses_reasoning_format_none_records_and_refuses(effort: str) -> None:
    req = Request(
        model="llama3.2",
        messages=(Message.user("hi"),),
        config=Config(reasoning=Reasoning(effort=effort)),  # type: ignore[arg-type]
    )
    lm = OpenAILM(api_key="k", compat="ollama")
    planned = lm.plan(req)
    assert len(planned) == 1
    assert planned[0].field == "config.reasoning"
    assert planned[0].action == "dropped"
    assert planned[0].asked == {"effort": effort}

    strict = OpenAILM(api_key="k", compat="ollama", adaptations="refuse")
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        strict.complete(req)
    assert excinfo.value.feature == "config.reasoning"


@pytest.mark.parametrize("compat", ["qwen", "zai"])
def test_responses_boolean_thinking_switch_records_substituted(compat: str) -> None:
    req = Request(
        model="qwen3",
        messages=(Message.user("hi"),),
        config=Config(reasoning=Reasoning(effort="high")),
    )
    lm = OpenAILM(api_key="k", compat=compat, base_url="https://example.invalid/v1")
    planned = lm.plan(req)
    assert any(
        a.field == "config.reasoning.effort" and a.action == "substituted" and a.asked == "high" and a.applied == "on"
        for a in planned
    )


def test_chat_qwen_thinking_switch_records_substituted() -> None:
    req = Request(
        model="qwen3",
        messages=(Message.user("hi"),),
        config=Config(reasoning=Reasoning(effort="low")),
    )
    lm = OpenAIChatLM(api_key="k", compat="qwen", base_url="https://example.invalid/v1")
    planned = lm.plan(req)
    assert any(
        a.field == "config.reasoning.effort" and a.action == "substituted" and a.asked == "low" and a.applied == "on"
        for a in planned
    )


def test_anthropic_structured_output_reject_still_refuses_required_probabilities() -> None:
    lm = AnthropicLM(api_key="k", compat="deepseek")
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    req = Request(
        model="deepseek-chat",
        messages=(Message.user("hi"),),
        config=Config(
            response_format={"type": "json_schema", "name": "j", "schema": schema, "strict": True},
            probabilities="required",
        ),
    )
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        lm.build_request(req, stream=False)
    assert excinfo.value.feature == "config.probabilities"


def test_gemini_trailing_usage_only_chunk_preserves_finish_reason() -> None:
    lm = GeminiLM(api_key="k")
    req = Request(model="gemini-2.5-flash", messages=(Message.user("hi"),))
    ev1 = SSEEvent(
        event="message",
        data=json.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "cut off"}]}, "finishReason": "MAX_TOKENS"}]
        }),
    )
    ev2 = SSEEvent(
        event="message",
        data=json.dumps({"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}),
    )
    raw = list(lm.parse_stream_events(req, ev1)) + list(lm.parse_stream_events(req, ev2))
    coalesced = list(coalesce_stream(iter(raw), model="gemini-2.5-flash"))
    end = next(e for e in coalesced if e.type == "end")
    assert end.finish_reason == "length"
    assert end.usage is not None
    assert end.usage.input_tokens == 10
    assert end.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_async_adapter_enforces_require_gates_and_batch_refuse() -> None:
    # OpenAI chat compat without files/batches/caches/images/speech/video must fail before network.
    from lm15.access import AccessPolicy
    from lm15.features import EndpointSupport

    bare_access = AccessPolicy(
        provider="bare-test",
        supports=EndpointSupport(
            complete=True, stream=True, models=False, files=False,
            batches=False, caches=False, images=False, speech=False, video=False, live=False,
        ),
    )
    lm = AsyncOpenAIChatLM(api_key="k", access=bare_access)
    req = Request(model="m", messages=(Message.user("hi"),))
    with pytest.raises(UnsupportedFeatureError):
        await lm.list_models()
    with pytest.raises(UnsupportedFeatureError):
        await lm.batch_list()
    with pytest.raises(UnsupportedFeatureError):
        await lm.cache_list()
    with pytest.raises(UnsupportedFeatureError):
        await lm.file_list()
    with pytest.raises(UnsupportedFeatureError):
        await lm.file_upload(FileUploadRequest(filename="a.txt", bytes_data=b"x"))
    with pytest.raises(UnsupportedFeatureError):
        await lm.video_submit(VideoGenerationRequest(model="v", prompt="p"))
    with pytest.raises(UnsupportedFeatureError):
        await lm.image_generate(ImageGenerationRequest(model="i", prompt="p"))
    with pytest.raises(UnsupportedFeatureError):
        await lm.speech_generate(SpeechGenerationRequest(model="s", prompt="hi"))

    # Async batch_submit respects adaptations="refuse".
    from lm15 import AsyncAnthropicLM

    strict_anthropic = AsyncAnthropicLM(api_key="k", adaptations="refuse")
    batch_req = BatchRequest(
        model="claude-sonnet-4-5",
        requests=(Request(model="claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config(seed=42)),),
    )
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        await strict_anthropic.batch_submit(batch_req)
    assert excinfo.value.feature == "config.seed"


@pytest.mark.parametrize("failure", [WriteTimeout, WriteError, BrokenPipeError, ConnectionResetError, OSError])
def test_sync_transport_never_retries_failed_write_on_reused_socket(monkeypatch, failure) -> None:
    """A failed write may have reached the peer; do not replay it."""
    import socket

    s1, s2 = socket.socketpair()
    t = StdlibTransport(trust_env=False)
    try:
        origin = ("http", "example.invalid", 80)
        from lm15.transports._sync import _SyncConnection

        conn = _SyncConnection(origin, s1)
        t._pool.register_new(conn)
        t._pool.checkin(conn)

        send_calls = 0

        def fake_send_request(*args, **kwargs):
            nonlocal send_calls
            send_calls += 1
            raise failure("write failed mid-POST")

        def fail_open(*args, **kwargs):
            raise AssertionError("must not open a fresh connection to retry WriteTimeout")

        monkeypatch.setattr(t, "_send_request", fake_send_request)
        monkeypatch.setattr(t, "_open", fail_open)

        with pytest.raises(WriteTimeout if failure is WriteTimeout else WriteError):
            t.stream(TransportRequest(method="POST", url="http://example.invalid/v1/responses", body=b"{}"))
        assert send_calls == 1
    finally:
        t.close()
        s1.close()
        s2.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [WriteTimeout, WriteError, BrokenPipeError, ConnectionResetError, OSError])
async def test_async_transport_never_retries_failed_write_on_reused_socket(monkeypatch, failure) -> None:
    import socket

    s1, s2 = socket.socketpair()
    t = StdlibAsyncTransport(trust_env=False)
    try:
        origin = ("http", "example.invalid", 80)

        class _FakeWriter:
            def is_closing(self) -> bool:
                return False

            def get_extra_info(self, name: str):
                return s1 if name == "socket" else None

            def close(self) -> None:
                pass

            async def wait_closed(self) -> None:
                pass

        from lm15.transports._async import _AsyncConnection

        conn = _AsyncConnection(origin, object(), _FakeWriter())  # type: ignore[arg-type]
        await t._pool.register_new(conn)
        await t._pool.checkin(conn)

        send_calls = 0

        async def fake_send_request(*args, **kwargs):
            nonlocal send_calls
            send_calls += 1
            raise failure("write failed mid-POST")

        async def fail_open(*args, **kwargs):
            raise AssertionError("must not open a fresh connection to retry WriteTimeout")

        monkeypatch.setattr(t, "_send_request", fake_send_request)
        monkeypatch.setattr(t, "_open", fail_open)

        with pytest.raises(WriteTimeout if failure is WriteTimeout else WriteError):
            await t.stream(TransportRequest(method="POST", url="http://example.invalid/v1/responses", body=b"{}"))
        assert send_calls == 1
    finally:
        await t.aclose()
        s1.close()
        s2.close()
