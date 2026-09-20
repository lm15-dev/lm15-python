"""Source-only cached destination regressions; no live credentials or network."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from lm15 import CacheInfo, CachedPrefix, Message, Request
from lm15.access import OPENAI_CHAT_API
from lm15.compat import OpenAIChatCompat
from lm15.providers import OpenAIChatLM
from lm15.registry import ProviderDefinition
from lm15.router import AsyncLMRouter, LMRouter, RouterConfig
from lm15.serde import cached_prefix_from_dict, cached_prefix_to_dict, request_to_dict
from lm15.testing import FakeResponse, FakeTransport
from .test_async_adapters import FakeAsyncTransport


def req(model: str) -> Request:
    return Request(model=model, messages=(Message.user("prefix"),))


def reply(body: dict) -> FakeResponse:
    return FakeResponse(200, json.dumps(body).encode())


CHAT = {"id": "r", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
RESPONSES = {"id": "r", "status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]}
PRIVATE = ProviderDefinition.chat(
    replace(OPENAI_CHAT_API, provider="private-chat", base_url="https://private.invalid/v1", env_keys=()),
    compat=OpenAIChatCompat(), aliases=("my-gateway",),
)


@pytest.mark.parametrize("destination", ["azure", "private-chat"])
def test_automatic_cache_roundtrip_routes_suffix_and_direct_binding(destination):
    body = RESPONSES if destination == "azure" else CHAT
    transport = FakeTransport([reply(body), reply(body)])
    router = LMRouter(config=RouterConfig(
        env={}, providers=(PRIVATE,), api_keys={destination: "test-key"},
        base_urls={"azure": "https://account.services.ai.azure.com"}, transport=transport,
    ))
    input_provider = "my_gateway" if destination == "private-chat" else destination
    cached = router.cache(req(f"{input_provider}:gpt-4.1-mini"))
    assert not transport.requests
    assert cached.provider == destination
    assert cached.prefix.model == "gpt-4.1-mini"
    restored = cached_prefix_from_dict(cached_prefix_to_dict(cached))
    suffix = restored.request("question")
    assert suffix.model == f"{destination}:gpt-4.1-mini"
    assert router.resolve(suffix.model).provider == destination
    router.complete(suffix)
    direct = router.lm(suffix.model)
    direct.plan(suffix)
    direct.complete(suffix)
    assert len(transport.requests) == 2
    for wire in transport.requests:
        assert json.loads(wire.body)["model"] == "gpt-4.1-mini"
        assert ("account.services.ai.azure.com" if destination == "azure" else "private.invalid") in wire.url


def test_async_router_cache_keeps_destination_and_bound_adapter_accepts_it():
    async def run():
        transport = FakeAsyncTransport(json.dumps(RESPONSES).encode())
        router = AsyncLMRouter(config=RouterConfig(
            env={}, api_keys={"azure": "test-key"},
            base_urls={"azure": "https://account.services.ai.azure.com"}, transport=transport,
        ))
        cached = await router.cache(req("azure:gpt-4.1-mini"))
        assert cached.provider == "azure" and not transport.requests
        suffix = cached_prefix_from_dict(cached_prefix_to_dict(cached)).request("question")
        await router.complete(suffix)
        await router.lm(suffix.model).complete(suffix)
        assert all(json.loads(w.body)["model"] == "gpt-4.1-mini" for w in transport.requests)
        assert all("account.services.ai.azure.com" in w.url for w in transport.requests)
    asyncio.run(run())


def test_resource_model_remains_provider_fact_across_roundtrip():
    transport = FakeTransport([reply({"name": "cachedContents/c", "model": "models/gemini-2.5-flash"})])
    router = LMRouter(config=RouterConfig(env={}, api_keys={"gemini": "key"}, transport=transport))
    cached = router.cache(req("gemini:gemini-2.5-flash"), ttl_seconds=60)
    assert cached.resource.model == cached.prefix.model == "gemini-2.5-flash"
    assert cached.provider == "gemini"
    restored = cached_prefix_from_dict(cached_prefix_to_dict(cached))
    suffix = restored.request("question")
    assert suffix.model == "gemini:gemini-2.5-flash"
    assert suffix.config.cache.resource == "cachedContents/c"
    assert restored.resource == cached.resource
    assert router.resolve(suffix.model).provider == "gemini"


def test_suffix_identity_validation_and_legacy_canonical_shape():
    legacy = CachedPrefix(req("m"))
    assert cached_prefix_to_dict(legacy) == {"prefix": request_to_dict(req("m"))}
    assert legacy.request("q").model == "m"
    cached = CachedPrefix(req("m"), CacheInfo(id="c", model="m"), provider="private_chat")
    assert cached.provider == "private-chat"
    for model in ("m", "private-chat:m", "private_chat:m"):
        assert cached.request(req(model)).model == "private-chat:m"
    for model in ("azure:m", "openai:m", "private-chat:other"):
        with pytest.raises(ValueError):
            cached.request(req(model))
    for provider in ("", "a:b", "a/b", "a b", "\t"):
        with pytest.raises(ValueError):
            CachedPrefix(req("m"), provider=provider)
    with pytest.raises(TypeError):
        CachedPrefix(req("m"), provider=42)
    with pytest.raises(ValueError):
        CachedPrefix(req("m"), CacheInfo(id="c", model="azure:m"), provider="azure")


def test_direct_cache_no_invented_destination_and_strip_only_own_prefix_once():
    lm = OpenAIChatLM(api_key="k", base_url="https://custom.invalid/v1")
    assert lm.cache(req("m")).provider is None
    routed = lm.cache(req("openai_chat:m"))
    assert routed.provider == "openai-chat" and routed.prefix.model == "m"
    assert routed.request("q").model == "openai-chat:m"
    for model, expected in (("openai_chat:m", "m"), ("openai-chat:openai-chat:m", "openai-chat:m"), ("other:m", "other:m"), ("arn:aws:bedrock:model", "arn:aws:bedrock:model")):
        wire = lm.build_request(req(model), stream=False)
        assert json.loads(wire.body)["model"] == expected
        lm.plan(req(model))
