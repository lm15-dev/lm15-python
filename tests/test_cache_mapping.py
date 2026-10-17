"""MAP-6 caching: adapter mapping table + the cache-resource surface.

Wire bytes for the pinned shapes are the contract's job (cache cases and
the `cache` harness direction); these tests pin the mapping table, the
raises, the fallback-sends-nothing cells, and the surface drivers.
Receipts: lm15-contract/research/caching/ (2026-09-01).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from lm15 import (
    AnthropicLM, AsyncGeminiLM, CacheConfig, CachedPrefix, CacheInfo, Config, FunctionTool, GeminiLM,
    LMRouter, Message, OpenAIChatLM, OpenAILM, Request, RouterConfig, UnsupportedFeatureError, XaiLM,
)
from lm15.providers.openai import openai_model_has_cache_options
from lm15.testing import FakeResponse, FakeTransport


def _req(model: str, cache: CacheConfig | None, *, system: str | None = "SYS", n: int = 3) -> Request:
    msgs = [Message.user("doc")]
    for i in range(n - 1):
        msgs.append(Message.assistant("a") if i % 2 == 0 else Message.user("q"))
    return Request(model=model, system=system, messages=msgs, config=Config(cache=cache) if cache else Config())


def _body(lm, request: Request) -> dict:
    return json.loads(lm.build_request(request, stream=False).body)


# ─── model class table ───────────────────────────────────────────────

def test_openai_model_class_for_cache_options() -> None:
    assert openai_model_has_cache_options("gpt-6")
    assert openai_model_has_cache_options("gpt-5.6-sol")
    assert openai_model_has_cache_options("gpt-6.0")
    assert openai_model_has_cache_options("gpt-6-luna")
    assert openai_model_has_cache_options("GPT-5.10")
    assert not openai_model_has_cache_options("gpt-5")
    assert not openai_model_has_cache_options("gpt-5-mini")
    assert not openai_model_has_cache_options("gpt-5.5")
    assert not openai_model_has_cache_options("gpt-4.1-mini")
    assert not openai_model_has_cache_options("gpt-3.5-turbo")
    assert not openai_model_has_cache_options("gpt-35-turbo")
    assert not openai_model_has_cache_options("gpt-35-turbo-16k")
    assert not openai_model_has_cache_options("o3")            # unknown pattern -> no switch (stated trade-off)
    assert not openai_model_has_cache_options("chat-latest")


# ─── OpenAI Responses ────────────────────────────────────────────────

def test_openai_off_sends_the_real_switch_only_on_5_6_class() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    assert _body(lm, _req("gpt-5.6-sol", CacheConfig(mode="off")))["prompt_cache_options"] == {"mode": "explicit"}
    assert "prompt_cache_options" not in _body(lm, _req("gpt-5.4-mini", CacheConfig(mode="off")))
    assert "prompt_cache_breakpoint" not in json.dumps(_body(lm, _req("gpt-5.6-sol", CacheConfig(mode="off"))))


def test_openai_stable_renders_system_as_marked_developer_message() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = _body(lm, _req("gpt-5.6-sol", CacheConfig(prefix="stable")))
    assert "instructions" not in body
    assert body["input"][0] == {"role": "developer", "content": [
        {"type": "input_text", "text": "SYS", "prompt_cache_breakpoint": {"mode": "explicit"}}]}
    assert body["input"][1]["role"] == "user"
    # The mark and explicit mode go together on 5.6+: without the mode the
    # warm call still writes the suffix at 1.25x (review probe 3, 2026-09-02).
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    # No system: nothing to mark, nothing sent (automatic tier) — and no
    # explicit mode either, which with no mark would cache nothing at all.
    bare = _body(lm, _req("gpt-5.6-sol", CacheConfig(prefix="stable"), system=None))
    assert "prompt_cache_breakpoint" not in json.dumps(bare) and "prompt_cache_options" not in bare
    # Pre-5.6 has no prompt_cache_options; the mark alone.
    old = _body(lm, _req("gpt-5.4-mini", CacheConfig(prefix="stable")))
    assert "prompt_cache_options" not in old and "prompt_cache_breakpoint" in json.dumps(old)


def test_openai_history_sends_nothing_implicit_mode_already_trails() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = _body(lm, _req("gpt-5.6-sol", CacheConfig(prefix="history")))
    assert "prompt_cache_breakpoint" not in json.dumps(body) and "prompt_cache_options" not in body
    assert body["instructions"] == "SYS"


def test_openai_long_retention_every_model_class() -> None:
    # 5.6 used to raise on a doc line about a different field; live
    # 2026-09-02 (review probe 2) gpt-5.6-sol accepts and echoes 24h.
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    assert _body(lm, _req("gpt-5.4-mini", CacheConfig(retention="long")))["prompt_cache_retention"] == "24h"
    assert _body(lm, _req("gpt-5.6-sol", CacheConfig(retention="long")))["prompt_cache_retention"] == "24h"


def test_openai_key_maps_resource_raises() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    assert _body(lm, _req("gpt-5.6-sol", CacheConfig(key="thread-1")))["prompt_cache_key"] == "thread-1"
    with pytest.raises(UnsupportedFeatureError, match="cache.resource"):
        _body(lm, _req("gpt-5.6-sol", CacheConfig(resource="caches/x")))


# ─── OpenAI Chat dialect ─────────────────────────────────────────────

def test_chat_dialect_off_stable_and_compat_gate() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    assert _body(lm, _req("gpt-5.6-sol", CacheConfig(mode="off")))["prompt_cache_options"] == {"mode": "explicit"}
    body = _body(lm, _req("gpt-5.6-sol", CacheConfig(prefix="stable")))
    assert body["messages"][0] == {"role": "system", "content": [
        {"type": "text", "text": "SYS", "prompt_cache_breakpoint": {"mode": "explicit"}}]}
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    groq = OpenAIChatLM(api_key="k", transport=FakeTransport([]), compat="groq")
    body = _body(groq, _req("openai/gpt-oss-20b", CacheConfig(prefix="stable", key="k")))
    assert body["messages"][0]["content"] == "SYS" and "prompt_cache_key" not in body  # cache_control none -> nothing


# ─── Anthropic ───────────────────────────────────────────────────────

def test_anthropic_history_marks_last_block_stable_marks_system() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    body = _body(lm, _req("claude-sonnet-5", CacheConfig(prefix="history")))
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    body = _body(lm, _req("claude-sonnet-5", CacheConfig(prefix="stable")))
    assert "cache_control" not in json.dumps(body["messages"])
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    body = _body(lm, _req("claude-sonnet-5", CacheConfig(prefix="history", retention="long")))
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_off_sends_nothing_key_is_dropped_resource_raises() -> None:
    from tests._adapt import adapted
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    assert "cache_control" not in json.dumps(_body(lm, _req("claude-sonnet-5", CacheConfig(mode="off"))))
    # MAP-13: a best-effort affinity hint with no home → dropped, recorded.
    out = adapted(lm, _req("claude-sonnet-5", CacheConfig(key="k")))
    assert out["config.cache.key"].action == "dropped"
    # A stored object that does not exist here → still a refusal (rule 4b), naming its field.
    with pytest.raises(UnsupportedFeatureError, match="cache.resource") as err:
        _body(lm, _req("claude-sonnet-5", CacheConfig(resource="r")))
    assert err.value.feature == "config.cache.resource"


# ─── Gemini ──────────────────────────────────────────────────────────

def test_gemini_prefix_intents_fall_back_to_automatic_tier() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    plain = _body(lm, _req("gemini-2.5-flash", None))
    for cfg in (CacheConfig(), CacheConfig(prefix="stable"), CacheConfig(prefix="history"), CacheConfig(prefix_until_index=0), CacheConfig(mode="off")):
        assert _body(lm, _req("gemini-2.5-flash", cfg)) == plain  # nothing added, nothing hidden


def test_gemini_key_and_retention_are_dropped_and_recorded() -> None:
    from tests._adapt import adapted
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    plain = _body(lm, _req("gemini-2.5-flash", None))
    out = adapted(lm, _req("gemini-2.5-flash", CacheConfig(key="k")))
    assert out["config.cache.key"].action == "dropped" and out["__body__"] == plain
    out = adapted(lm, _req("gemini-2.5-flash", CacheConfig(retention="long")))
    assert out["config.cache.retention"].action == "dropped" and out["__body__"] == plain


def test_gemini_resource_sends_suffix_only_and_no_system_or_tools() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    prefix = Request(model="gemini-2.5-flash", system="SYS", tools=(FunctionTool(name="f"),), messages=[Message.user("DOC")])
    cached = CachedPrefix(prefix, CacheInfo(id="cachedContents/abc", model="gemini-2.5-flash"))
    body = _body(lm, cached + "q?")
    assert body == {"contents": [{"role": "user", "parts": [{"text": "q?"}]}], "cachedContent": "cachedContents/abc"}
    # resource without an index: the object is assumed to hold system/tools only; all messages go
    body = _body(lm, Request(model="gemini-2.5-flash", system="SYS", messages=[Message.user("a"), Message.user("b")],
                             config=Config(cache=CacheConfig(resource="abc"))))
    assert len(body["contents"]) == 2 and body["cachedContent"] == "cachedContents/abc" and "systemInstruction" not in body
    with pytest.raises(ValueError, match="at least one message after the prefix"):
        _body(lm, Request(model="gemini-2.5-flash", messages=[Message.user("only")], config=Config(cache=CacheConfig(resource="abc", prefix_until_index=0))))


def test_gemini_cache_hooks_build_the_documented_wire() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    prefix = Request(model="gemini-2.5-flash", system="SYS", messages=[Message.user("DOC")])
    tr = lm._cache_create_request(prefix, 300, "docs")
    assert tr.method == "POST" and tr.url.endswith("/cachedContents")
    assert json.loads(tr.body) == {"model": "models/gemini-2.5-flash", "contents": [{"role": "user", "parts": [{"text": "DOC"}]}],
                                   "systemInstruction": {"parts": [{"text": "SYS"}]}, "ttl": "300s", "displayName": "docs"}
    assert lm._cache_get_request("cachedContents/x").url.endswith("/cachedContents/x")
    assert lm._cache_get_request("x").url.endswith("/cachedContents/x")
    assert lm._cache_delete_request("x").method == "DELETE"
    assert json.loads(lm._cache_update_request("x", 600).body) == {"ttl": "600s"}
    assert "pageToken=tok" in lm._cache_list_request(10, "tok").url


_CACHE_BODY = {"name": "cachedContents/abc", "model": "models/gemini-2.5-flash", "createTime": "2026-09-01T10:49:45.6Z",
               "expireTime": "2026-09-01T10:54:45.649467125Z", "displayName": "docs", "usageMetadata": {"totalTokenCount": 3574}}


def test_gemini_cache_info_parse_and_drivers() -> None:
    t = FakeTransport([FakeResponse(200, json.dumps(_CACHE_BODY).encode()),
                       FakeResponse(200, json.dumps({"cachedContents": [_CACHE_BODY], "nextPageToken": "n"}).encode()),
                       FakeResponse(200, b"{}")])
    lm = GeminiLM(api_key="k", transport=t)
    prefix = Request(model="gemini-2.5-flash", system="SYS", messages=[Message.user("DOC")])
    cached = lm.cache(prefix, ttl_seconds=300, label="docs")
    assert cached.id == "cachedContents/abc" and cached.resource.tokens == 3574
    assert cached.resource.expires_at == "2026-09-01T10:54:45Z" and cached.resource.model == "gemini-2.5-flash"
    page = lm.cache_list()
    assert page.items[0].id == "cachedContents/abc" and page.next_cursor == "n"
    lm.cache_delete(cached.id)
    assert [r.method for r in t.requests] == ["POST", "GET", "DELETE"]
    with pytest.raises(ValueError, match="default Config"):
        lm.cache(Request(model="gemini-2.5-flash", messages=[Message.user("x")], config=Config(temperature=0.1)))


def test_cache_is_pure_on_marker_and_automatic_providers() -> None:
    for lm in (AnthropicLM(api_key="k", transport=FakeTransport([])), OpenAILM(api_key="k", transport=FakeTransport([])),
               XaiLM(api_key="k", transport=FakeTransport([]))):
        cached = lm.cache(Request(model="m", messages=[Message.user("DOC")]))
        assert cached.resource is None and lm.transport.requests == []
        with pytest.raises(UnsupportedFeatureError):
            lm.cache_list()


def test_async_gemini_cache_mirror() -> None:
    from tests.test_async_adapters import FakeAsyncTransport

    alm = AsyncGeminiLM(api_key="k", transport=FakeAsyncTransport(json.dumps(_CACHE_BODY).encode()))
    cached = asyncio.run(alm.cache(Request(model="gemini-2.5-flash", messages=[Message.user("DOC")]), ttl_seconds=60))
    assert cached.id == "cachedContents/abc" and cached.resource.tokens == 3574


def test_router_cache_routes_by_prefix_model() -> None:
    t = FakeTransport([FakeResponse(200, json.dumps(_CACHE_BODY).encode())])
    router = LMRouter(RouterConfig(api_keys={"gemini": "k", "anthropic": "k"}, transport=t))
    cached = router.cache(Request(model="gemini-2.5-flash", messages=[Message.user("DOC")]))
    assert cached.id == "cachedContents/abc"
    cached = router.cache(Request(model="anthropic:claude-sonnet-5", messages=[Message.user("DOC")]))
    assert cached.resource is None and cached.prefix.model == "claude-sonnet-5"
