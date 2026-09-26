"""DeepInfra, Together AI, Fireworks AI and Parasail: the registry entries and
the compat rules each one carries (lm15-contract changes/2026-09-26-inference-
hosts-live.md; every rule here has a receipt under receipts/2026-09-26-<host>/).
"""
from __future__ import annotations

import json

import pytest

from lm15 import Config, FunctionTool, ImagePart, LMRouter, Message, OpenAIChatLM, Request, RouterConfig, ToolChoice
from lm15.errors import ProviderError, UnsupportedFeatureError
from lm15.providers.openai_chat import _usage_from_chat
from lm15.registry import PROVIDERS
from lm15.router import LITELLM_PROVIDER_PREFIXES, openai_chat_model_string
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import CacheConfig, Reasoning, ThinkingPart, ToolCallPart, ToolResultPart

HOSTS = {
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "deepinfra"),
    "together": ("https://api.together.ai/v1", "TOGETHER_API_KEY", "together_ai"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "fireworks_ai"),
    "parasail": ("https://api.parasail.io/v1", "PARASAIL_API_KEY", "parasail"),
}
WEATHER = FunctionTool(name="get_weather", description="Get weather.",
                       parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})


def body(provider: str, request: Request) -> dict:
    return json.loads(OpenAIChatLM(api_key="k", compat=provider).build_request(request, stream=False).body)


def build(provider: str, request: Request):
    return OpenAIChatLM(api_key="k", compat=provider)._build(request, False)


def ask(model: str, **config) -> Request:
    return Request(model=model, messages=(Message.user("hi"),), config=Config(**config))


# ─── registry, router, credentials ─────────────────────────────────


@pytest.mark.parametrize("provider", HOSTS)
def test_registry_entry(provider: str) -> None:
    base_url, env_key, _ = HOSTS[provider]
    entry = PROVIDERS[provider]
    assert entry.dialect == "openai-chat" and entry.adapter is OpenAIChatLM
    assert entry.base_url == base_url
    assert entry.env_keys == (env_key,)
    assert entry.console_url and entry.console_url.startswith("https://")
    supports = entry.supports
    assert (supports.complete, supports.stream, supports.models) == (True, True, True)
    assert not (supports.files or supports.batches or supports.images or supports.responses_api)


@pytest.mark.parametrize("provider", HOSTS)
def test_router_prefix_and_env_key(provider: str) -> None:
    base_url, env_key, litellm = HOSTS[provider]
    router = LMRouter(config=RouterConfig(env={env_key: "k"}))
    resolution = router.resolve(f"{provider}:vendor/some-model")
    assert (resolution.provider, resolution.model, resolution.env_key) == (provider, "vendor/some-model", env_key)
    lm = router.lm(f"{provider}:vendor/some-model")
    assert lm.provider == provider and lm.base_url == base_url
    # litellm's spelling routes to the same door
    assert LITELLM_PROVIDER_PREFIXES[litellm] == provider
    assert openai_chat_model_string(f"{litellm}/vendor/some-model") == f"{provider}:vendor/some-model"


# ─── one policy for the four: dials and replay ─────────────────────


@pytest.mark.parametrize("provider", HOSTS)
def test_effort_rides_reasoning_effort_and_the_cap_is_max_completion_tokens(provider: str) -> None:
    sent = body(provider, ask("vendor/reasoner", max_tokens=50, reasoning=Reasoning(effort="low")))
    assert sent["reasoning_effort"] == "low" and sent["max_completion_tokens"] == 50
    assert "reasoning" not in sent and "thinking" not in sent  # Fireworks answers 400 to the object


@pytest.mark.parametrize("provider", HOSTS)
def test_reasoning_is_replayed_as_reasoning_content(provider: str) -> None:
    turn = Message.assistant((ThinkingPart(text="Need the tool."),
                              ToolCallPart(id="call_1", name="get_weather", input={"city": "Paris"})))
    request = Request(model="vendor/model", tools=(WEATHER,),
                      messages=(Message.user("Weather?"), turn, Message.tool("call_1", "Sunny")))
    assistant = body(provider, request)["messages"][1]
    assert assistant["reasoning_content"] == "Need the tool."
    assert assistant["content"] is None  # never pasted into the visible text


@pytest.mark.parametrize("provider", HOSTS)
def test_cache_key_is_dropped_with_a_record(provider: str) -> None:
    transport, adaptations = build(provider, ask("vendor/model", cache=CacheConfig(key="session-1")))
    assert "prompt_cache_key" not in json.loads(transport.body)
    assert [(a.field, a.action) for a in adaptations] == [("config.cache.key", "dropped")]


# ─── per-host, per-model rules with receipts ───────────────────────


def test_deepinfra_refuses_a_forced_tool_choice_except_on_deepseek_v4() -> None:
    forced = Config(tool_choice=ToolChoice(mode="required"))
    with pytest.raises(UnsupportedFeatureError):
        body("deepinfra", Request(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=(Message.user("hi"),),
                                  tools=(WEATHER,), config=forced))
    sent = body("deepinfra", Request(model="deepseek-ai/DeepSeek-V4.1-Flash", messages=(Message.user("hi"),),
                                     tools=(WEATHER,), config=forced))
    assert sent["tool_choice"] == "required"


def test_together_refuses_a_forced_tool_choice_on_gpt_oss_only() -> None:
    forced = Config(tool_choice=ToolChoice(mode="required"))
    with pytest.raises(UnsupportedFeatureError):
        body("together", Request(model="openai/gpt-oss-120b", messages=(Message.user("hi"),), tools=(WEATHER,), config=forced))
    for model in ("meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V4.1-Flash"):
        sent = body("together", Request(model=model, messages=(Message.user("hi"),), tools=(WEATHER,), config=forced))
        assert sent["tool_choice"] == "required"


@pytest.mark.parametrize(("asked", "applied"), [("max", "high"), ("xhigh", "high"), ("minimal", "low")])
def test_together_gpt_oss_effort_is_clamped_to_its_levels(asked: str, applied: str) -> None:
    transport, adaptations = build("together", ask("openai/gpt-oss-120b", reasoning=Reasoning(effort=asked)))
    assert json.loads(transport.body)["reasoning_effort"] == applied
    assert [(a.field, a.action, a.asked, a.applied) for a in adaptations] == [
        ("config.reasoning.effort", "clamped", asked, applied)]
    # DeepSeek on Together refuses unknown words itself: sent verbatim
    assert body("together", ask("deepseek-ai/DeepSeek-V4.1-Flash", reasoning=Reasoning(effort=asked)))["reasoning_effort"] == asked


@pytest.mark.parametrize(("provider", "model"), [
    ("together", "openai/gpt-oss-120b"),
    ("together", "zai-org/GLM-5.3-Flash"),
    ("deepinfra", "openai/gpt-oss-120b"),
])
def test_off_becomes_the_lowest_level_where_the_server_ignores_none(provider: str, model: str) -> None:
    transport, adaptations = build(provider, ask(model, reasoning=Reasoning(effort="off")))
    assert json.loads(transport.body)["reasoning_effort"] == "low"
    assert [(a.field, a.action, a.asked, a.applied) for a in adaptations] == [
        ("config.reasoning.effort", "substituted", "off", "low")]


@pytest.mark.parametrize(("provider", "model"), [
    ("together", "deepseek-ai/DeepSeek-V4.1-Flash"),        # honours none
    ("fireworks", "accounts/fireworks/models/gpt-oss-120b"),  # refuses none loudly
    ("parasail", "openai/gpt-oss-20b"),                      # refuses none loudly
])
def test_off_is_sent_where_it_is_honoured_or_refused_loudly(provider: str, model: str) -> None:
    transport, adaptations = build(provider, ask(model, reasoning=Reasoning(effort="off")))
    assert json.loads(transport.body)["reasoning_effort"] == "none" and adaptations == ()


@pytest.mark.parametrize(("provider", "native"), [("fireworks", True), ("parasail", True), ("deepinfra", False), ("together", False)])
def test_images_in_tool_results(provider: str, native: bool) -> None:
    result = ToolResultPart(id="call_1", content=(ImagePart(media_type="image/png", data="iVBORw0KGgo="),))
    request = Request(model="vendor/vl", tools=(WEATHER,), messages=(
        Message.user("Look."),
        Message.assistant((ToolCallPart(id="call_1", name="get_weather", input={"city": "Paris"}),)),
        Message(role="tool", parts=(result,)),
    ))
    if native:
        row = body(provider, request)["messages"][2]
        assert row["role"] == "tool" and row["content"][0]["type"] == "image_url"
    else:
        with pytest.raises(UnsupportedFeatureError):
            body(provider, request)


# ─── dialect-wide fixes found on these hosts ───────────────────────


def test_models_listing_reads_a_bare_array_and_refuses_an_unknown_shape() -> None:
    lm = OpenAIChatLM(api_key="k", compat="together")
    assert [m.id for m in lm._models_from_body('[{"id": "a"}, {"id": "b"}]')] == ["a", "b"]
    assert [m.id for m in lm._models_from_body('{"object": "list", "data": [{"id": "c"}]}')] == ["c"]
    with pytest.raises(ValueError):
        lm._models_from_body('{"models": []}')


def test_a_malformed_catalog_is_a_provider_error_not_an_empty_list() -> None:
    lm = OpenAIChatLM(api_key="k", compat="together", transport=FakeTransport([FakeResponse(status=200, body=b'{"models": [{"id": "x"}]}')]))
    with pytest.raises(ProviderError, match="malformed provider reply"):
        lm.list_models()


def test_cached_tokens_are_read_nested_first_then_flat() -> None:
    assert _usage_from_chat({"prompt_tokens": 9, "cached_tokens": 0}).cache_read_tokens == 0  # Together, Llama
    assert _usage_from_chat({"prompt_tokens_details": {"cached_tokens": 3}, "cached_tokens": 9}).cache_read_tokens == 3
    assert _usage_from_chat({"prompt_tokens": 9}).cache_read_tokens is None

