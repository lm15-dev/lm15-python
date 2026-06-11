"""Tests for lm15.router — LMRouter / AsyncLMRouter.

Hermetic: env is always injected via RouterConfig(env=...) and keys via
RouterConfig(api_keys=...); no live network (FakeTransport patterns).
"""

from __future__ import annotations

import asyncio
import json

import pytest

import lm15
from lm15.models import ModelInfo, ModelRegistry
from lm15.providers import (
    AnthropicLM,
    AsyncAnthropicLM,
    AsyncOpenAIChatLM,
    GeminiLM,
    OpenAIChatLM,
    OpenAILM,
)
from lm15.router import (
    ADAPTERS,
    ASYNC_ADAPTERS,
    DEFAULT_RULES,
    AmbiguousModelError,
    AsyncLMRouter,
    LMRouter,
    MissingCredentialError,
    Resolution,
    RouteRule,
    RouterConfig,
    RouterError,
    UnknownModelError,
)
from lm15.types import Message, Request, TextPart

from .test_providers import _FakeResponse, _FakeTransport

_ENV = {
    "OPENAI_API_KEY": "sk-openai",
    "ANTHROPIC_API_KEY": "sk-ant",
    "GEMINI_API_KEY": "sk-gem",
}


def _router(**kwargs) -> LMRouter:
    kwargs.setdefault("env", dict(_ENV))
    return LMRouter(config=RouterConfig(**kwargs))


def _registry(*infos: ModelInfo) -> ModelRegistry:
    registry = ModelRegistry()
    for info in infos:
        registry.add(info)
    return registry


def _info(model_id: str, provider: str, aliases: tuple[str, ...] = ()) -> ModelInfo:
    return ModelInfo(id=model_id, provider=provider, api_family=provider, aliases=aliases)


# ─── resolve: ladder ─────────────────────────────────────────────────


class TestResolvePrefix:
    def test_explicit_prefix_wins(self) -> None:
        res = _router().resolve("openai:gpt-4.1-mini")
        assert res == Resolution(
            requested="openai:gpt-4.1-mini",
            model="gpt-4.1-mini",
            provider="openai",
            adapter="OpenAILM",
            source="prefix",
            env_key="OPENAI_API_KEY",
        )

    def test_prefix_beats_catalog_and_rules(self) -> None:
        registry = _registry(_info("claude-x", "anthropic"))
        res = _router(registry=registry).resolve("openai_chat:claude-x")
        assert res.provider == "openai_chat"
        assert res.source == "prefix"
        assert res.model == "claude-x"

    def test_split_on_first_colon_preserves_rest(self) -> None:
        res = _router().resolve("openai:ft:gpt-4.1:org")
        assert res.model == "ft:gpt-4.1:org"
        assert res.provider == "openai"

    def test_unknown_head_falls_through_to_rules(self) -> None:
        # "ft" is not a provider; the whole string is treated as a bare id.
        with pytest.raises(UnknownModelError):
            _router().resolve("ft:gpt-4.1:org")

    def test_requested_preserves_verbatim_input(self) -> None:
        res = _router().resolve("anthropic:claude-opus-4")
        assert res.requested == "anthropic:claude-opus-4"
        assert res.model == "claude-opus-4"


class TestResolveCatalog:
    def test_unique_catalog_match(self) -> None:
        info = _info("llama3.3-70b", "openai_chat")
        res = _router(registry=_registry(info)).resolve("llama3.3-70b")
        assert res.source == "catalog"
        assert res.provider == "openai_chat"
        assert res.adapter == "OpenAIChatLM"
        assert res.model_info == info

    def test_alias_resolves_to_canonical_id(self) -> None:
        info = _info("llama-3.3-70b-instruct", "openai_chat", aliases=("llama3.3",))
        res = _router(registry=_registry(info)).resolve("llama3.3")
        assert res.model == "llama-3.3-70b-instruct"
        assert res.requested == "llama3.3"

    def test_catalog_beats_rules(self) -> None:
        # A catalog claiming a claude- model lives on openai_chat wins
        # over the built-in claude- -> anthropic rule.
        info = _info("claude-mirror", "openai_chat")
        res = _router(registry=_registry(info)).resolve("claude-mirror")
        assert res.provider == "openai_chat"
        assert res.source == "catalog"

    def test_ambiguous_catalog_match_raises(self) -> None:
        registry = _registry(_info("shared-model", "openai_chat"), _info("shared-model", "gemini"))
        with pytest.raises(AmbiguousModelError) as exc_info:
            _router(registry=registry).resolve("shared-model")
        err = exc_info.value
        assert err.model == "shared-model"
        assert set(err.providers) == {"openai_chat", "gemini"}
        assert err.candidates == err.providers
        # Error message prints the exact fix line.
        assert "Request(model=" in str(err)
        assert ":shared-model" in str(err)

    def test_exact_id_match_beats_alias_match(self) -> None:
        # An alias on another entry must never shadow an entry whose
        # canonical id IS the requested string — in either insertion order.
        dated = _info("gpt-4.1-2025-04-14", "openai", aliases=("gpt-4.1",))
        plain = _info("gpt-4.1", "openai")
        for ordering in ((dated, plain), (plain, dated)):
            res = _router(registry=_registry(*ordering)).resolve("gpt-4.1")
            assert res.model == "gpt-4.1"
            assert res.model_info == plain

    def test_same_provider_duplicate_alias_match_is_ambiguous(self) -> None:
        registry = _registry(
            _info("model-a", "openai", aliases=("fast",)),
            _info("model-b", "openai", aliases=("fast",)),
        )
        with pytest.raises(AmbiguousModelError) as exc_info:
            _router(registry=registry).resolve("fast")
        assert "model-a" in str(exc_info.value)
        assert "model-b" in str(exc_info.value)

    def test_catalog_provider_without_adapter_is_clear_error(self) -> None:
        registry = _registry(_info("router-model", "openrouter"))
        with pytest.raises(UnknownModelError) as exc_info:
            _router(registry=registry).resolve("router-model")
        assert "openrouter" in str(exc_info.value)
        assert exc_info.value.catalog_searched is True

    def test_no_registry_means_no_catalog_rung(self) -> None:
        with pytest.raises(UnknownModelError) as exc_info:
            _router().resolve("llama3.3-70b")
        assert exc_info.value.catalog_searched is False


class TestResolveRules:
    @pytest.mark.parametrize(
        ("model", "provider", "adapter"),
        [
            ("claude-sonnet-4-5", "anthropic", "AnthropicLM"),
            ("gpt-4.1-mini", "openai", "OpenAILM"),
            ("o1-preview", "openai", "OpenAILM"),
            ("o3-mini", "openai", "OpenAILM"),
            ("o4-mini", "openai", "OpenAILM"),
            ("gemini-2.5-pro", "gemini", "GeminiLM"),
        ],
    )
    def test_default_rules(self, model: str, provider: str, adapter: str) -> None:
        res = _router().resolve(model)
        assert (res.provider, res.adapter, res.source) == (provider, adapter, "rule")
        assert res.rule is not None and model.startswith(res.rule.prefix)
        assert res.model == model

    def test_custom_rules_replace_defaults(self) -> None:
        rules = (RouteRule("my-", "openai_chat", note="local vllm naming"),)
        res = _router(rules=rules).resolve("my-model")
        assert res.provider == "openai_chat"
        assert res.rule == rules[0]
        with pytest.raises(UnknownModelError):
            _router(rules=rules).resolve("claude-sonnet-4-5")

    def test_first_match_wins(self) -> None:
        rules = (RouteRule("g", "gemini"), RouteRule("gpt-", "openai"))
        res = _router(rules=rules).resolve("gpt-4.1")
        assert res.provider == "gemini"

    def test_unknown_model_error_carries_ladder_state(self) -> None:
        with pytest.raises(UnknownModelError) as exc_info:
            _router().resolve("mystery-model")
        err = exc_info.value
        assert err.model == "mystery-model"
        assert err.rules_tried == DEFAULT_RULES
        assert err.catalog_searched is False
        # Suggests the explicit prefix and the catalog package.
        assert "anthropic:mystery-model" in str(err)
        assert "aimo" in str(err)

    def test_unknown_model_with_registry_omits_catalog_hint(self) -> None:
        with pytest.raises(UnknownModelError) as exc_info:
            _router(registry=_registry()).resolve("mystery-model")
        assert exc_info.value.catalog_searched is True
        assert "aimo" not in str(exc_info.value)


# ─── resolve: purity and explanation ─────────────────────────────────


class TestResolutionRecord:
    def test_resolve_is_pure_no_key_needed(self) -> None:
        # Empty env: resolve still works; only lm() needs the key.
        res = _router(env={}).resolve("claude-sonnet-4-5")
        assert res.provider == "anthropic"
        assert res.env_key == "ANTHROPIC_API_KEY"  # first declared key

    def test_env_key_records_first_hit(self) -> None:
        res = _router(env={"GOOGLE_API_KEY": "g"}).resolve("gemini-2.5-pro")
        assert res.env_key == "GOOGLE_API_KEY"
        res = _router(env={"GEMINI_API_KEY": "g", "GOOGLE_API_KEY": "g"}).resolve("gemini-2.5-pro")
        assert res.env_key == "GEMINI_API_KEY"

    def test_env_key_none_when_api_keys_override(self) -> None:
        res = _router(env={}, api_keys={"anthropic": "sk"}).resolve("claude-sonnet-4-5")
        assert res.env_key is None

    def test_env_key_none_for_oauth_providers(self) -> None:
        res = _router().resolve("claude-code:claude-sonnet-4-5")
        assert res.env_key is None
        assert res.adapter == "ClaudeCodeLM"

    def test_resolution_is_frozen_and_slotted(self) -> None:
        res = _router().resolve("gpt-4.1")
        with pytest.raises(AttributeError):
            res.provider = "gemini"  # type: ignore[misc]
        assert not hasattr(res, "__dict__")

    def test_describe_renders_explanation(self) -> None:
        res = _router().resolve("claude-sonnet-4-5")
        text = res.describe()
        assert "anthropic" in text
        assert "AnthropicLM" in text
        assert "claude-" in text  # the rule prefix
        assert "ANTHROPIC_API_KEY" in text
        assert str(res) == text

    def test_rule_note_surfaced_in_describe(self) -> None:
        res = _router().resolve("gpt-4.1")
        assert "Responses API" in res.describe()


# ─── config ──────────────────────────────────────────────────────────


class TestRouterConfig:
    def test_config_is_frozen(self) -> None:
        config = RouterConfig()
        with pytest.raises(AttributeError):
            config.rules = ()  # type: ignore[misc]

    def test_api_keys_repr_suppressed(self) -> None:
        config = RouterConfig(api_keys={"openai": "sk-secret"})
        assert "sk-secret" not in repr(config)

    def test_default_rules_is_inspectable_tuple(self) -> None:
        assert isinstance(DEFAULT_RULES, tuple)
        assert all(isinstance(rule, RouteRule) for rule in DEFAULT_RULES)
        assert set(ADAPTERS) == set(ASYNC_ADAPTERS)


# ─── lm(): construction, caching, credentials ────────────────────────


class TestLM:
    def test_lm_constructs_provider_lm_from_env(self) -> None:
        router = _router()
        lm = router.lm("claude-sonnet-4-5")
        try:
            assert isinstance(lm, AnthropicLM)
            assert lm.api_key == "sk-ant"
        finally:
            lm.close()

    def test_lm_cached_per_provider(self) -> None:
        router = _router()
        lm1 = router.lm("gpt-4.1")
        lm2 = router.lm("openai:gpt-4.1-mini")
        try:
            assert lm1 is lm2
            assert isinstance(lm1, OpenAILM)
        finally:
            lm1.close()

    def test_lm_distinct_per_provider(self) -> None:
        router = _router()
        lms = [router.lm("gemini-2.5-pro"), router.lm("claude-opus-4")]
        try:
            assert isinstance(lms[0], GeminiLM)
            assert isinstance(lms[1], AnthropicLM)
        finally:
            for lm in lms:
                lm.close()

    def test_api_keys_beat_env(self) -> None:
        router = _router(api_keys={"anthropic": "sk-explicit"})
        lm = router.lm("claude-sonnet-4-5")
        try:
            assert lm.api_key == "sk-explicit"
        finally:
            lm.close()

    def test_hermetic_no_env_at_all(self) -> None:
        router = _router(env={}, api_keys={"openai_chat": "sk-h"})
        lm = router.lm("openai_chat:local-model")
        try:
            assert isinstance(lm, OpenAIChatLM)
            assert lm.api_key == "sk-h"
        finally:
            lm.close()

    def test_missing_credential_error(self) -> None:
        with pytest.raises(MissingCredentialError) as exc_info:
            _router(env={}).lm("claude-sonnet-4-5")
        err = exc_info.value
        assert err.provider == "anthropic"
        assert err.env_keys == ("ANTHROPIC_API_KEY",)
        assert "ANTHROPIC_API_KEY" in str(err)
        assert "api_keys" in str(err)

    def test_missing_credential_is_not_configured(self) -> None:
        # MissingCredentialError hooks the existing taxonomy.
        assert issubclass(MissingCredentialError, lm15.NotConfiguredError)
        assert issubclass(MissingCredentialError, RouterError)
        assert issubclass(RouterError, lm15.LM15Error)

    def test_gemini_second_env_key(self) -> None:
        router = _router(env={"GOOGLE_API_KEY": "sk-g"})
        lm = router.lm("gemini-2.5-pro")
        try:
            assert lm.api_key == "sk-g"
        finally:
            lm.close()


# ─── complete / stream routing ───────────────────────────────────────


_CHAT_BODY = json.dumps(
    {
        "id": "chatcmpl-1",
        "model": "m-live",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
).encode()


def _request(model: str) -> Request:
    return Request(model=model, messages=(Message.user("Hi"),))


class TestCompleteStream:
    def test_complete_routes_and_strips_prefix(self) -> None:
        router = _router(api_keys={"openai_chat": "sk-h"}, env={})
        transport = _FakeTransport([_FakeResponse(status=200, body=_CHAT_BODY)])
        router.lm("openai_chat:local-model").transport = transport
        response = router.complete(_request("openai_chat:local-model"))
        assert response.message.parts == (TextPart(text="Hello!"),)
        # The wire request carried the stripped model id.
        sent = json.loads(transport.requests[0].body)
        assert sent["model"] == "local-model"

    def test_complete_bare_model_unchanged(self) -> None:
        router = _router(api_keys={"openai_chat": "sk-h"}, env={})
        transport = _FakeTransport([_FakeResponse(status=200, body=_CHAT_BODY)])
        router.lm("openai_chat:x").transport = transport
        router.complete(_request("openai_chat:llama3.3-70b"))
        sent = json.loads(transport.requests[0].body)
        assert sent["model"] == "llama3.3-70b"

    def test_stream_routes_through_provider_lm(self) -> None:
        sse = (
            b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}\n\n'
            b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        router = _router(api_keys={"openai_chat": "sk-h"}, env={})
        transport = _FakeTransport(
            [_FakeResponse(status=200, body=sse, headers=[("content-type", "text/event-stream")])]
        )
        router.lm("openai_chat:x").transport = transport
        events = list(router.stream(_request("openai_chat:local-model")))
        assert events, "expected stream events"
        sent = json.loads(transport.requests[0].body)
        assert sent["model"] == "local-model"

    def test_complete_unknown_model_raises_before_network(self) -> None:
        with pytest.raises(UnknownModelError):
            _router().complete(_request("mystery-model"))


# ─── async mirror ────────────────────────────────────────────────────


class FakeAsyncResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.reason = "OK"
        self.headers = [("content-type", "application/json")]
        self.http_version = "HTTP/1.1"
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def read(self) -> bytes:
        return self._body

    def __aiter__(self):
        async def gen():
            yield self._body

        return gen()


class FakeAsyncTransport:
    def __init__(self, responses: list[FakeAsyncResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class TestAsyncRouter:
    def _arouter(self, **kwargs) -> AsyncLMRouter:
        kwargs.setdefault("env", dict(_ENV))
        return AsyncLMRouter(config=RouterConfig(**kwargs))

    def test_resolve_is_shared_and_sync(self) -> None:
        res = self._arouter().resolve("claude-sonnet-4-5")
        assert res.provider == "anthropic"
        assert res.adapter == "AsyncAnthropicLM"
        assert res.source == "rule"

    def test_lm_returns_async_mirror_and_caches(self) -> None:
        router = self._arouter()
        lm1 = router.lm("claude-sonnet-4-5")
        lm2 = router.lm("anthropic:claude-opus-4")
        assert isinstance(lm1, AsyncAnthropicLM)
        assert lm1 is lm2
        assert lm1.api_key == "sk-ant"

    def test_missing_credential_error_async(self) -> None:
        with pytest.raises(MissingCredentialError):
            self._arouter(env={}).lm("gpt-4.1")

    def test_async_complete_routes_and_strips_prefix(self) -> None:
        router = self._arouter(api_keys={"openai_chat": "sk-h"}, env={})
        lm = router.lm("openai_chat:x")
        assert isinstance(lm, AsyncOpenAIChatLM)
        transport = FakeAsyncTransport([FakeAsyncResponse(200, _CHAT_BODY)])
        lm.transport = transport

        async def run():
            return await router.complete(_request("openai_chat:local-model"))

        response = asyncio.run(run())
        assert response.message.parts == (TextPart(text="Hello!"),)
        sent = json.loads(transport.requests[0].body)
        assert sent["model"] == "local-model"

    def test_async_stream_routes(self) -> None:
        sse = (
            b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}\n\n'
            b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        router = self._arouter(api_keys={"openai_chat": "sk-h"}, env={})
        lm = router.lm("openai_chat:x")
        lm.transport = FakeAsyncTransport([FakeAsyncResponse(200, sse)])

        async def run():
            return [event async for event in router.stream(_request("openai_chat:local-model"))]

        events = asyncio.run(run())
        assert events, "expected stream events"


# ─── exports ─────────────────────────────────────────────────────────


def test_router_surface_exported_from_lm15() -> None:
    for name in (
        "LMRouter",
        "AsyncLMRouter",
        "RouterConfig",
        "RouteRule",
        "Resolution",
        "DEFAULT_RULES",
        "RouterError",
        "UnknownModelError",
        "AmbiguousModelError",
        "MissingCredentialError",
        "ADAPTERS",
        "ASYNC_ADAPTERS",
    ):
        assert hasattr(lm15, name), name
        assert name in lm15.__all__


def test_router_error_code_is_bare_noun() -> None:
    # Taxonomy convention: codes are bare nouns ('transport', 'provider',
    # ...), never suffixed with '_error'.
    assert RouterError("boom").code == "router"
