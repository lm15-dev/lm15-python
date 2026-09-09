"""Migration ergonomics: real routing and dialects, no credentials or network."""
import asyncio
import json

import pytest

from lm15 import (
    LMRouter, AsyncLMRouter, RouterConfig, Response, ResponseStream,
    AsyncResponseStream, NotConfiguredError,
)
from lm15.doctor import explain_auth
from lm15.testing import FakeResponse, FakeTransport
from .test_async_adapters import FakeAsyncTransport

MESSAGES = [{"role": "user", "content": "Hello"}]
CHAT = {"model": "gpt-4o-mini", "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
SSE = b''.join(b'data: ' + json.dumps(frame).encode() + b'\n\n' for frame in [
    {"choices": [{"delta": {"content": "Hi"}}]},
    {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
]) + b'data: [DONE]\n\n'


@pytest.mark.parametrize("router_cls", [LMRouter, AsyncLMRouter])
@pytest.mark.parametrize("source,target", [
    ("openai", "openai-chat"), ("openai_chat", "openai"),
    ("meta", "meta-anthropic"), ("meta-chat", "meta"),
    ("moonshotai", "moonshotai-responses"), ("moonshotai-responses", "moonshotai-anthropic"),
    ("deepseek", "deepseek-anthropic"),
])
def test_one_key_serves_sibling_endpoints(router_cls, source, target):
    config = RouterConfig(env={}, api_keys={source: "explicit-test-key"}, transport=FakeTransport())
    router = router_cls(config)
    where = router.resolve(f"{target}:test-model")
    assert where.env_key is None
    assert router.lm(f"{target}:test-model").api_key == "explicit-test-key"
    report = explain_auth(where, config=config)
    assert report.selected.kind == "api_keys"
    assert source in report.selected.source
    assert "explicit-test-key" not in report.describe() + repr(report)


def test_exact_entry_wins_over_shared_entry_and_env():
    config = RouterConfig(env={"OPENAI_API_KEY": "env-test-key"},
                          api_keys={"openai": "family-test-key", "openai-chat": "exact-test-key"})
    router = LMRouter(config)
    assert router.lm("openai-chat:x").api_key == "exact-test-key"
    assert router.lm("openai:x").api_key == "family-test-key"
    report = explain_auth(router.resolve_openai_chat("gpt-4o-mini"), config=config)
    assert report.selected.source == "explicit api_keys entry"
    assert next(s for s in report.steps if s.kind == "env:OPENAI_API_KEY").state == "shadowed"


@pytest.mark.parametrize("source,target", [("gemini", "vertex-express"), ("vertex-express", "gemini"),
                                           ("openai", "anthropic"), ("ollama", "vllm"),
                                           ("vertex", "vertex-anthropic")])
def test_empty_or_overlapping_env_lists_do_not_share(source, target):
    config = RouterConfig(env={}, api_keys={source: "test-key"})
    report = explain_auth(target, config=config, files={})
    assert not any(s.kind == "api_keys" and s.state == "selected" for s in report.steps)


def test_cloud_shared_key_keeps_host_settings_separate():
    config = RouterConfig(env={}, api_keys={"azure": "test-key"},
                          settings={"azure-chat": {"resource": "test-resource"}})
    router = LMRouter(config)
    lm = router.lm("azure-chat:deployment")
    assert lm.api_key == "test-key"
    assert "test-resource.openai.azure.com" in lm.base_url
    report = explain_auth(router.resolve("azure-chat:deployment"), config=config, files={})
    assert "azure" in report.selected.source
    assert report.selected.kind == "api_keys"


def test_ambiguous_shared_entries_refuse_without_comparing_or_invoking_values():
    def credential():
        pytest.fail("lookup must not invoke credentials")
    config = RouterConfig(env={"META_API_KEY": "ambient-test-key"},
                          api_keys={"meta": credential, "meta-chat": credential})
    router = LMRouter(config)
    for lookup in (lambda: router.lm("meta-anthropic:x"),
                   lambda: explain_auth("meta-anthropic", config=config)):
        with pytest.raises(NotConfiguredError, match="ambiguous") as exc:
            lookup()
        assert "ambient-test-key" not in str(exc.value)
    # Exact target entries resolve even if the family has several entries.
    assert router.lm("meta:x").api_key is credential


@pytest.mark.parametrize("value", [None, ""])
def test_invalid_explicit_key_never_falls_back_to_environment(value):
    config = RouterConfig(env={"OPENAI_API_KEY": "ambient-test-key"}, api_keys={"openai": value})
    with pytest.raises(NotConfiguredError):
        LMRouter(config).lm("openai-chat:x")
    with pytest.raises(NotConfiguredError):
        explain_auth("openai-chat", config=config)


def test_duplicate_provider_aliases_are_not_order_dependent():
    with pytest.raises(NotConfiguredError, match="duplicate"):
        LMRouter(RouterConfig(env={}, api_keys={"openai_chat": "a", "openai-chat": "b"}))


def test_shared_callable_remains_lazy_and_rotates_per_request():
    calls = []
    def credential():
        calls.append(1)
        return f"test-key-{len(calls)}"
    transport = FakeTransport([FakeResponse(200, json.dumps(CHAT).encode()) for _ in range(2)])
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": credential}, transport=transport))
    explain_auth(router.resolve_openai_chat("gpt-4o-mini"), config=router.config)
    assert not calls
    for _ in range(2):
        router.complete_from_openai_chat("gpt-4o-mini", MESSAGES)
    assert len(calls) == 2
    assert [dict(r.headers)["Authorization"] for r in transport.requests] == ["Bearer test-key-1", "Bearer test-key-2"]


def test_tutorial_stream_and_auth_blocks_run_offline(capsys):
    """Execute the actual public examples, not a separately maintained copy."""
    import re
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs/migrating-from-openai-chat.md").read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", doc, re.M | re.S)
    transport = FakeTransport([FakeResponse(200, SSE)])
    keyed_router = LMRouter(RouterConfig(env={"OPENAI_API_KEY": "ambient-test-key"},
                                        api_keys={"openai": "explicit-test-key"}, transport=transport))
    scope = {"router": keyed_router, "keyed_router": keyed_router, "messages": MESSAGES, "explain_auth": explain_auth}
    for block in blocks:
        if block.startswith('where = keyed_router.resolve_openai_chat') or block.startswith('result = router.complete_from_openai_chat'):
            exec(block, scope)
    shown = capsys.readouterr().out
    assert "via 'openai'" in shown and "Hi!" in shown
    assert "ambient-test-key" not in shown and "explicit-test-key" not in shown


def test_migration_stream_is_lazy_and_assembles_usage():
    transport = FakeTransport([FakeResponse(200, SSE)])
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=transport))
    result = router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
    assert isinstance(result, ResponseStream)
    assert not transport.requests
    assert list(result) == ["Hi", "!"]
    assert result.response.text == "Hi!"
    assert result.response.usage.output_tokens == 2
    assert json.loads(transport.requests[0].body)["stream"] is True
    assert list(result) == []  # no replay, no second paid call


@pytest.mark.parametrize("kwargs", [{}, {"stream": False}])
def test_nonstreaming_return_type_stays_response(kwargs):
    transport = FakeTransport([FakeResponse(200, json.dumps(CHAT).encode())])
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=transport))
    response = router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, **kwargs)
    assert isinstance(response, Response)
    assert response.text == "Hi"


@pytest.mark.parametrize("bad", [None, 0, 1, "false", "true", [], {}])
def test_stream_requires_a_real_boolean_before_credentials_or_network(bad):
    with pytest.raises(TypeError, match="stream"):
        LMRouter(RouterConfig(env={})).complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=bad)


def test_async_stream_mirrors_sync():
    async def run():
        empty = AsyncLMRouter(RouterConfig(env={}))
        for invalid in (None, "false", 0, 1):
            with pytest.raises(TypeError, match="stream"):
                await empty.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=invalid)
        for opts in ({}, {"stream": False}):
            completed = AsyncLMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"},
                                                   transport=FakeAsyncTransport(json.dumps(CHAT).encode())))
            assert isinstance(await completed.complete_from_openai_chat("gpt-4o-mini", MESSAGES, **opts), Response)
        transport = FakeAsyncTransport(SSE)
        router = AsyncLMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=transport))
        result = await router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
        assert isinstance(result, AsyncResponseStream)
        assert not transport.requests
        assert [text async for text in result] == ["Hi", "!"]
        response = await result.response()
        assert response.text == "Hi!" and response.usage.output_tokens == 2
    asyncio.run(run())


@pytest.mark.parametrize("model,key,frames,url_suffix", [
    ("anthropic/claude-sonnet-4-5", "anthropic", [
        {"type": "message_start", "message": {"id": "msg-test", "model": "claude-sonnet-4-5"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ], "/messages"),
    ("gemini/gemini-2.5-flash", "gemini", [
        {"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1}},
    ], ":streamGenerateContent"),
])
def test_stream_migration_keeps_native_provider_dialect(model, key, frames, url_suffix):
    body = b''.join(b'data: ' + json.dumps(frame).encode() + b'\n\n' for frame in frames)
    transport = FakeTransport([FakeResponse(200, body)])
    router = LMRouter(RouterConfig(env={}, api_keys={key: "test-key"}, transport=transport))
    with router.complete_from_openai_chat(model, MESSAGES, stream=True) as result:
        assert list(result) == ["Hi"]
    assert result.response.text == "Hi"
    assert result.response.usage.output_tokens == 1
    assert url_suffix in transport.requests[0].url


def test_stream_keeps_tools_and_reasoning_while_iterating_text():
    frames = [
        {"choices": [{"delta": {"reasoning_content": "Think", "content": "Wait"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call-test", "function": {"name": "lookup", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
    ]
    body = b''.join(b'data: ' + json.dumps(frame).encode() + b'\n\n' for frame in frames)
    transport = FakeTransport([FakeResponse(200, body)])
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=transport))
    result = router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
    assert list(result) == ["Wait"]
    assert [p.type for p in result.response.message.parts] == ["thinking", "text", "tool_call"]
    assert result.response.tool_calls[0].name == "lookup"


def test_stream_error_stays_typed_and_never_returns_success():
    from lm15 import RateLimitError
    body = b'data: {"error":{"code":"rate_limit_exceeded","message":"slow down"}}\n\n'
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"},
                                  transport=FakeTransport([FakeResponse(200, body)])))
    result = router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
    with pytest.raises(RateLimitError):
        list(result)
    with pytest.raises(RateLimitError):
        result.response


def test_close_releases_retained_source_without_draining():
    from lm15 import Request, Message, StreamDeltaEvent, TextDelta, StreamEndEvent
    from lm15.result import coalesce_stream
    closed = []
    def source():
        try:
            yield StreamDeltaEvent(TextDelta("first"))
            pytest.fail("early close must not drain")
        finally:
            closed.append(True)
    raw = source()  # retain a reference: GC must not be responsible for cleanup
    result = ResponseStream(coalesce_stream(raw), Request("x", (Message.user("x"),)))
    with result:
        assert next(iter(result)) == "first"
    assert closed == [True]
    result.close()
    with pytest.raises(RuntimeError, match="closed"):
        result.response
    complete = ResponseStream(iter([StreamEndEvent(finish_reason="stop")]), Request("x", (Message.user("x"),)))
    next(complete.events())
    complete.close()
    assert complete.response.finish_reason == "stop"


def test_router_early_close_releases_http_response_sync_and_async():
    released = []
    class TrackedResponse(FakeResponse):
        def __exit__(self, *args):
            released.append("sync")
    transport = FakeTransport([TrackedResponse(200, SSE)])
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=transport))
    result = router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
    with result:
        assert next(iter(result)) == "Hi"
    assert released == ["sync"]

    from lm15.transports import AsyncTransportResponse
    class TrackedAsyncTransport:
        def stream(self, request):
            async def chunks():
                yield SSE
            async def release(consumed):
                released.append("async")
            return AsyncTransportResponse(status=200, reason="OK", headers=[], http_version="HTTP/1.1",
                                          chunks=chunks(), release=release)
    async def run():
        router = AsyncLMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=TrackedAsyncTransport()))
        result = await router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
        async with result:
            assert await anext(result.__aiter__()) == "Hi"
        assert released == ["sync", "async"]
    asyncio.run(run())


def test_close_before_iteration_never_starts_request():
    transport = FakeTransport()
    router = LMRouter(RouterConfig(env={}, api_keys={"openai": "test-key"}, transport=transport))
    result = router.complete_from_openai_chat("gpt-4o-mini", MESSAGES, stream=True)
    result.close()
    assert not transport.requests
    assert list(result) == []
    with pytest.raises(RuntimeError, match="closed"):
        result.response


def test_async_close_and_cancel_release_source():
    from lm15 import Request, Message, StreamDeltaEvent, TextDelta
    from lm15.result import acoalesce_stream
    async def run():
        for cancel in (False, True):
            closed = []
            waiting = asyncio.Event()
            async def source():
                try:
                    yield StreamDeltaEvent(TextDelta("first"))
                    waiting.set()
                    await asyncio.Event().wait()
                finally:
                    closed.append(True)
            raw = source()
            result = AsyncResponseStream(acoalesce_stream(raw), Request("x", (Message.user("x"),)))
            iterator = result.__aiter__()
            assert await anext(iterator) == "first"
            if cancel:
                task = asyncio.create_task(anext(iterator))
                await waiting.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                with pytest.raises(asyncio.CancelledError):
                    await result.response()
            await result.aclose()
            assert closed == [True]
            if not cancel:
                with pytest.raises(RuntimeError, match="closed"):
                    await result.response()
    asyncio.run(run())
