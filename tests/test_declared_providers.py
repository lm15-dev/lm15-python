"""RouterConfig(providers=...): a provider the registry does not list,
declared by the caller as the same pure-data triple a registry entry is
(access policy + dialect + compat), routable only through routers built
with that config, and honest about carrying no lm15 receipt.

Hermetic: env via RouterConfig(env=...), keys via api_keys, FakeTransport.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from lm15.compat import AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat
from lm15.errors import NotConfiguredError
from lm15.features import AccessPolicy, EndpointSupport
from lm15.providers import AnthropicLM, AsyncOpenAIChatLM, OpenAIChatLM, OpenAILM
from lm15.registry import PROVIDERS, ProviderDefinition
from lm15.router import (
    AsyncLMRouter,
    LMRouter,
    MissingCredentialError,
    RouterConfig,
    UnknownModelError,
    openai_chat_model_string,
)
from lm15.types import Message, Request, TextPart

from .test_providers import _FakeResponse, _FakeTransport
from .test_router import _CHAT_BODY

BASE = "https://api.hostco.test/inference/v1"


def _access(provider: str = "hostco", **overrides) -> AccessPolicy:
    fields = {
        "provider": provider,
        "supports": EndpointSupport(complete=True, stream=True, models=True),
        "auth_modes": ("bearer",),
        "env_keys": ("HOSTCO_API_KEY",),
        "base_url": BASE,
    }
    fields.update(overrides)
    return AccessPolicy(**fields)


HOSTCO_COMPAT = OpenAIChatCompat(max_tokens_field="max_tokens", thinking_format="reasoning_effort")
HOSTCO = ProviderDefinition.chat(_access(), compat=HOSTCO_COMPAT, aliases=("hostco-ai",), note="HostCo (declared)")
MODEL = "accounts/hostco/models/deepseek-v4p1-flash"


def _request(model: str) -> Request:
    return Request(model=model, messages=(Message.user("Hi"),))


def _router(**kwargs) -> LMRouter:
    kwargs.setdefault("providers", (HOSTCO,))
    kwargs.setdefault("env", {})
    return LMRouter(config=RouterConfig(**kwargs))


class TestDefinition:
    def test_chat_constructor_names_the_dialect_classes(self) -> None:
        assert HOSTCO.id == "hostco"
        assert HOSTCO.dialect == "openai-chat"
        assert HOSTCO.adapter is OpenAIChatLM and HOSTCO.async_adapter is AsyncOpenAIChatLM
        assert HOSTCO.bound and not HOSTCO.hosted
        assert HOSTCO.compat is HOSTCO_COMPAT
        assert HOSTCO.spellings == ("hostco", "hostco-ai")

    def test_responses_and_anthropic_constructors(self) -> None:
        responses = ProviderDefinition.responses(_access("gw-responses"), compat=OpenAIResponsesCompat())
        anthropic = ProviderDefinition.anthropic(_access("gw-anthropic"), compat=AnthropicCompat())
        assert responses.adapter is OpenAILM and responses.dialect == "openai-responses"
        assert anthropic.adapter is AnthropicLM and anthropic.dialect == "anthropic"

    def test_registry_entries_still_build_through_the_constructors(self) -> None:
        # The private helpers delegate; the receipted table is unchanged.
        groq = PROVIDERS["groq"]
        assert groq.compat == "groq" and groq.aliases == () and groq.bound

    def test_compat_object_must_match_the_dialect(self) -> None:
        with pytest.raises(TypeError, match="OpenAIChatCompat"):
            ProviderDefinition.chat(_access(), compat=AnthropicCompat())  # type: ignore[arg-type]

    def test_declared_entry_needs_a_base_url(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            ProviderDefinition.chat(_access(base_url=None), compat=OpenAIChatCompat())

    def test_declared_entry_is_key_based(self) -> None:
        with pytest.raises(ValueError, match="key-based"):
            ProviderDefinition.chat(_access(credential_policy="oauth", env_keys=()), compat=OpenAIChatCompat())
        with pytest.raises(ValueError, match="key-based"):
            ProviderDefinition.chat(_access(credential_policy="oauth-unless-explicit"), compat=OpenAIChatCompat())

    def test_aliases_are_canonical_unique_and_not_the_id(self) -> None:
        with pytest.raises(ValueError, match="hyphenated"):
            ProviderDefinition.chat(_access(), compat=OpenAIChatCompat(), aliases=("hostco_ai",))
        with pytest.raises(ValueError, match="repeat"):
            ProviderDefinition.chat(_access(), compat=OpenAIChatCompat(), aliases=("hostco",))
        with pytest.raises(ValueError, match="repeat"):
            ProviderDefinition.chat(_access(), compat=OpenAIChatCompat(), aliases=("a", "a"))
        # The constructor coerces; the dataclass itself insists on a tuple.
        assert ProviderDefinition.chat(_access(), compat=OpenAIChatCompat(), aliases=["a"]).aliases == ("a",)
        with pytest.raises(TypeError, match="tuple"):
            dataclasses.replace(HOSTCO, aliases=["a"])  # type: ignore[arg-type]

    def test_preset_names_keep_the_table_rule(self) -> None:
        # A preset NAME still has to agree with the compat table's URL: a
        # declared provider that reuses a preset name is a registry
        # drift, not a new door.
        with pytest.raises(ValueError, match="compat table"):
            ProviderDefinition.chat(_access(), compat="groq")

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            HOSTCO.note = "x"  # type: ignore[misc]


class TestConfig:
    def test_type_checked(self) -> None:
        with pytest.raises(TypeError, match="tuple"):
            RouterConfig(providers=[HOSTCO])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ProviderDefinition"):
            RouterConfig(providers=("hostco",))  # type: ignore[arg-type]

    @pytest.mark.parametrize("spelling", ["groq", "deepseek", "openai-chat", "ollama-chat", "hosted-vllm"])
    def test_a_spelling_lm15_already_routes_is_refused(self, spelling: str) -> None:
        definition = ProviderDefinition.chat(_access("new-door"), compat=OpenAIChatCompat(), aliases=(spelling,))
        with pytest.raises(NotConfiguredError, match=f"{spelling!r} already names"):
            RouterConfig(providers=(definition,))
        # As an id too.
        with pytest.raises(NotConfiguredError, match="already names"):
            RouterConfig(providers=(ProviderDefinition.chat(_access(spelling), compat=OpenAIChatCompat()),))

    def test_two_declared_providers_cannot_share_a_spelling(self) -> None:
        other = ProviderDefinition.chat(_access("other"), compat=OpenAIChatCompat(), aliases=("hostco-ai",))
        with pytest.raises(NotConfiguredError, match="spelled by both"):
            RouterConfig(providers=(HOSTCO, other))

    def test_api_keys_and_base_urls_accept_the_declared_id(self) -> None:
        router = _router(api_keys={"hostco": "k"}, base_urls={"hostco": "http://127.0.0.1:9/v1"})
        assert router.lm(f"hostco:{MODEL}").base_url == "http://127.0.0.1:9/v1"
        # An alias is an input spelling for model strings; config is keyed
        # by the id, and a near miss is named like any other.
        with pytest.raises(NotConfiguredError, match="Did you mean 'hostco'"):
            _router(api_keys={"hostco-ai": "k"})

    def test_unknown_key_still_refused_with_declared_in_the_known_list(self) -> None:
        with pytest.raises(NotConfiguredError, match="hostco"):
            _router(api_keys={"firewrks": "k"})


class TestResolve:
    def test_prefix_id_alias_and_litellm_spellings(self) -> None:
        router = _router()
        for spelling in (f"hostco:{MODEL}", f"hostco-ai:{MODEL}", f"hostco_ai:{MODEL}"):
            res = router.resolve(spelling)
            assert (res.provider, res.model, res.source) == ("hostco", MODEL, "prefix"), spelling
        for spelling in (f"hostco/{MODEL}", f"hostco_ai/{MODEL}", f"hostco-ai/{MODEL}"):
            res = router.resolve_openai_chat(spelling)
            assert (res.provider, res.model) == ("hostco", MODEL), spelling

    def test_resolution_says_declared_and_carries_the_compat_object(self) -> None:
        res = _router().resolve(f"hostco:{MODEL}")
        assert res.declared is True
        assert res.compat is HOSTCO_COMPAT
        assert res.adapter == "OpenAIChatLM"
        text = res.describe()
        assert "declared by RouterConfig(providers=...)" in text and "no lm15 receipts" in text
        assert "compat OpenAIChatCompat object" in text

    def test_describe_reads_the_declaration_not_the_registry(self) -> None:
        # A declared keyless server: the resolution carries the placeholder
        # and describe() says so (greptile on dspy#10440).
        local = ProviderDefinition.chat(
            _access("gpu-box", env_keys=(), base_url="http://gpu-box:8000/v1"),
            compat=OpenAIChatCompat(), placeholder_key="EMPTY",
        )
        res = LMRouter(config=RouterConfig(providers=(local,), env={})).resolve("gpu-box:m")
        assert res.placeholder_key == "EMPTY" and res.credential_policy == "key" and res.env_key is None
        assert "local-server default" in res.describe()
        # Registry entries carry the same facts.
        ollama = _router().resolve("ollama:m")
        assert ollama.placeholder_key == "ollama" and "local-server default" in ollama.describe()
        xai = _router().resolve("xai:grok-4")
        assert xai.credential_policy == "oauth-unless-explicit" and "stored subscription" in xai.describe()

    def test_registry_entries_are_not_declared(self) -> None:
        res = _router(env={"GROQ_API_KEY": "k"}).resolve("groq:llama")
        assert res.declared is False and res.compat == "groq"

    def test_env_key_is_the_declared_policy_s(self) -> None:
        res = _router(env={"HOSTCO_API_KEY": "k"}).resolve(f"hostco:{MODEL}")
        assert res.env_key == "HOSTCO_API_KEY"
        assert _router(api_keys={"hostco": "k"}).resolve(f"hostco:{MODEL}").env_key is None

    def test_a_router_without_the_declaration_does_not_know_it(self) -> None:
        with pytest.raises(UnknownModelError):
            LMRouter(config=RouterConfig(env={})).resolve(f"hostco:{MODEL}")
        with pytest.raises(UnknownModelError, match="hostco"):
            LMRouter(config=RouterConfig(env={})).resolve_openai_chat(f"hostco/{MODEL}")

    def test_declared_appears_in_known_providers_hint(self) -> None:
        with pytest.raises(UnknownModelError, match="hostco"):
            _router().resolve("nope:model")

    def test_openai_chat_model_string_lists_declared_spellings(self) -> None:
        assert openai_chat_model_string(f"hostco_ai/{MODEL}", providers=(HOSTCO,)) == f"hostco:{MODEL}"
        with pytest.raises(UnknownModelError, match="hostco-ai"):
            openai_chat_model_string("nope/x", providers=(HOSTCO,))

    def test_object_provider_alias(self) -> None:
        class Tagged(str):
            provider = "hostco_ai"

        res = _router().resolve(Tagged(MODEL))
        assert (res.provider, res.source) == ("hostco", "object")

    def test_rule_may_name_a_declared_provider(self) -> None:
        from lm15.router import DEFAULT_RULES, RouteRule

        rules = (RouteRule("accounts/", "hostco", note="HostCo ids"), *DEFAULT_RULES)
        res = _router(rules=rules).resolve(MODEL)
        assert (res.provider, res.source) == ("hostco", "rule")


class TestBuild:
    def test_lm_is_the_dialect_class_bound_to_the_declared_policy(self) -> None:
        router = _router(api_keys={"hostco": "k-1"})
        lm = router.lm(f"hostco:{MODEL}")
        assert type(lm) is OpenAIChatLM
        assert lm.provider == "hostco"
        assert lm.base_url == BASE
        assert lm.access is HOSTCO.access
        assert lm._compat_partial is HOSTCO_COMPAT
        assert router.lm(f"hostco_ai:{MODEL}") is lm  # one LM per provider

    def test_complete_sends_the_key_and_the_stripped_model(self) -> None:
        router = _router(api_keys={"hostco": "k-1"})
        transport = _FakeTransport([_FakeResponse(status=200, body=_CHAT_BODY)])
        router.lm(f"hostco:{MODEL}").transport = transport
        response = router.complete(_request(f"hostco_ai:{MODEL}"))
        assert response.message.parts == (TextPart(text="Hello!"),)
        sent = transport.requests[0]
        assert sent.url == f"{BASE}/chat/completions"
        assert dict(sent.headers)["Authorization"] == "Bearer k-1"
        assert json.loads(sent.body)["model"] == MODEL

    def test_compat_object_shapes_the_wire(self) -> None:
        router = _router(api_keys={"hostco": "k-1"})
        transport = _FakeTransport([_FakeResponse(status=200, body=_CHAT_BODY)])
        router.lm(f"hostco:{MODEL}").transport = transport
        request = dataclasses.replace(_request(f"hostco:{MODEL}"), config=lm15_config(max_tokens=7))
        router.complete(request)
        body = json.loads(transport.requests[0].body)
        assert body["max_tokens"] == 7 and "max_completion_tokens" not in body

    def test_env_key_then_missing_credential(self) -> None:
        lm = _router(env={"HOSTCO_API_KEY": "from-env"}).lm(f"hostco:{MODEL}")
        assert lm.api_key == "from-env"
        with pytest.raises(MissingCredentialError, match="HOSTCO_API_KEY") as info:
            _router().lm(f"hostco:{MODEL}")
        assert info.value.provider == "hostco"

    def test_plan_needs_no_key(self) -> None:
        assert _router().plan(_request(f"hostco:{MODEL}")) == ()

    def test_async_router_builds_the_async_mirror(self) -> None:
        async def go():
            router = AsyncLMRouter(config=RouterConfig(providers=(HOSTCO,), api_keys={"hostco": "k"}, env={}))
            res = router.resolve_openai_chat(f"hostco_ai/{MODEL}")
            assert res.adapter == "AsyncOpenAIChatLM" and res.declared
            lm = router.lm(f"hostco:{MODEL}")
            assert type(lm) is AsyncOpenAIChatLM and lm.provider == "hostco"
            await router.aclose()

        asyncio.run(go())

    def test_module_views_are_untouched(self) -> None:
        from lm15.router import ADAPTERS, ASYNC_ADAPTERS, CHAT_PRESET_ROUTES

        _router(api_keys={"hostco": "k"})
        assert "hostco" not in ADAPTERS and "hostco" not in ASYNC_ADAPTERS
        assert "hostco" not in CHAT_PRESET_ROUTES and "hostco" not in PROVIDERS
        # A router with nothing declared keeps the module view by identity
        # (tests and callers that compare against ADAPTERS rely on it).
        assert LMRouter(config=RouterConfig(env={}))._adapters is ADAPTERS


def lm15_config(**kwargs):
    from lm15.types import Config

    return Config(**kwargs)
