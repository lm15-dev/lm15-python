"""Candidate-scoring preflight and generated-stream boundaries (offline)."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lm15 import Config, Message, Request, judgments, yes_no
from lm15.errors import ProviderError, UnsupportedFeatureError
from lm15.providers import OpenAIChatLM
from lm15.providers.async_base import AsyncOpenAIChatLM
from lm15.result import materialize_response
from lm15.testing import FakeResponse, FakeTransport
from lm15.transports import AsyncTransportResponse
from lm15.types import CacheConfig, FunctionTool, ToolChoice


FORMAT = judgments(ok=yes_no("Is it fine?"))
BASE = Request(model="m", messages=(Message.user("Fine."),),
               config=Config(response_format=FORMAT, probabilities="required"))


def boundary_request(case):
    if case == "mixed":
        fmt = deepcopy(FORMAT)
        fmt["schema"]["properties"]["explanation"] = {"type": "string"}
        fmt["schema"]["required"].append("explanation")
        return replace(BASE, config=replace(BASE.config, response_format=fmt)), "config.response_format"
    if case == "tools":
        return replace(BASE, tools=(FunctionTool(name="lookup", parameters={"type": "object"}),)), "tools"
    if case == "choice":
        return replace(BASE, config=replace(BASE.config, tool_choice=ToolChoice(mode="none"))), "config.tool_choice"
    if case == "cache":
        return replace(BASE, config=replace(BASE.config, cache=CacheConfig(resource="cached/prompt"))), "config.cache.resource"
    return replace(BASE, config=replace(BASE.config, extensions={"n": 2})), "config.extensions.n"


def forbidden(*args, **kwargs):
    raise AssertionError("credential/tokenization/transport must not run")


@pytest.mark.parametrize("case", ["tools", "choice", "cache", "multiple"])
@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
@pytest.mark.parametrize("probabilities", ["required", "if_available"])
def test_harmful_scoring_omissions_refuse_before_credentials(case, policy, probabilities):
    request, feature = boundary_request(case)
    request = replace(request, config=replace(request.config, probabilities=probabilities))
    transport = FakeTransport([])
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    async_lm = AsyncOpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    for invoke in (lambda: lm.plan(request), lambda: lm.complete(request),
                   lambda: async_lm.plan(request), lambda: asyncio.run(async_lm.complete(request))):
        with pytest.raises(UnsupportedFeatureError) as caught:
            invoke()
        assert caught.value.feature == feature
        assert "generated JSON" in str(caught.value) and "separate scoring" in str(caught.value)
    assert transport.requests == []


def test_strict_plan_and_complete_refuse_before_even_building_tokenizations(monkeypatch):
    monkeypatch.setattr(OpenAIChatLM, "_judgment_plan", forbidden)
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", adaptations="refuse")
    async_lm = AsyncOpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", adaptations="refuse")
    mixed, _ = boundary_request("mixed")
    for invoke in (lambda: lm.plan(mixed), lambda: lm.complete(mixed), lambda: async_lm.plan(mixed),
                   lambda: asyncio.run(async_lm.complete(mixed))):
        with pytest.raises(UnsupportedFeatureError) as caught:
            invoke()
        assert caught.value.feature == "config.response_format"
    # Per-call policy override, including the async mirror's policy.
    note = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1")
    with pytest.raises(UnsupportedFeatureError):
        note.plan(mixed, policy="refuse")


class ScoringTransport(FakeTransport):
    missing_ids = False
    scoring_usage = None
    generated_usage = None
    generated_content = '{"ok": false, "explanation": "ordinary answer", "extra": {"keep": [1, 2]}}'
    generated_finish = "stop"

    def reply(self, request):
        self.requests.append(request)
        payload = json.loads(request.body)
        if request.url.endswith("/tokenize"):
            answer = payload["messages"][-1]["content"]
            tokens = [1]
            if answer != "Answer:":
                tokens.append(10 if answer.endswith("true") else 11)
                if not payload["continue_final_message"]:
                    tokens.append(99)
            return json.dumps({"tokens": tokens}).encode()
        if request.url.endswith("/chat/completions"):
            return json.dumps({"id": "generated", "model": "m", "usage": self.generated_usage,
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": self.generated_content},
                                            "finish_reason": self.generated_finish}]}).encode()
        assert request.url.endswith("/completions")
        top = {} if self.missing_ids else {f"token_id:{i}": -1.0 for i in payload["logprob_token_ids"]}
        return json.dumps({"model": "m", "usage": self.scoring_usage, "choices": [
            {"index": i, "logprobs": {"top_logprobs": [top]}}
            for i in range(len(payload["prompt"]))
        ]}).encode()

    def stream(self, request):
        return FakeResponse(status=200, body=self.reply(request), headers=[("x-request-id", "scoring-wire")])


class AsyncScoringTransport(ScoringTransport):
    def stream(self, request):
        body = self.reply(request)

        async def chunks():
            yield body

        async def release(body_consumed):
            pass

        return AsyncTransportResponse(status=200, reason="OK", headers=[("x-request-id", "scoring-wire")], http_version="HTTP/1.1",
                                      chunks=chunks(), release=release)


@pytest.mark.parametrize("model,wire_model", [("openai_chat:m", "m"), ("openai-chat:openai-chat:m", "openai-chat:m")])
@pytest.mark.parametrize("mixed,missing", [(False, False), (True, False), (True, True)])
def test_scoring_qualified_models_are_stripped_once_across_all_exchanges(model, wire_model, mixed, missing):
    original = boundary_request("mixed")[0] if mixed else BASE
    request = replace(original, model=model, config=replace(original.config, probabilities="if_available" if missing else "required"))
    for asynchronous in (False, True):
        transport = AsyncScoringTransport([]) if asynchronous else ScoringTransport([])
        transport.missing_ids = missing
        cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
        lm = cls(api_key="synthetic-key", compat="vllm", base_url="http://scoring/v1", transport=transport)
        lm.plan(request)
        assert transport.requests == []
        if asynchronous:
            asyncio.run(lm.complete(request))
        else:
            lm.complete(request)
        assert len(transport.requests) == (7 if mixed or missing else 6)
        assert {json.loads(wire.body)["model"] for wire in transport.requests} == {wire_model}


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
def test_pure_plan_and_native_complete_have_no_adaptation(policy):
    calls = []

    def credential():
        calls.append("credential")
        return "k"

    transport = ScoringTransport()
    lm = OpenAIChatLM(api_key=credential, compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    plan = lm.plan(BASE)
    assert calls == [] and transport.requests == []
    assert plan == ()
    response = lm.complete(BASE)
    assert response.adaptations == (plan if policy == "note" else ())
    assert response.method == "candidate_sequence_likelihood"
    assert len(transport.requests) == 6
    assert lm.plan(BASE) == plan  # records do not accumulate

    async_transport = AsyncScoringTransport()
    async_lm = AsyncOpenAIChatLM(api_key=credential, compat="vllm", base_url="http://scoring/v1", transport=async_transport, adaptations=policy)
    assert async_lm.plan(BASE) == plan
    result = asyncio.run(async_lm.complete(BASE))
    assert result.adaptations == response.adaptations and result.probabilities == response.probabilities
    assert len(async_transport.requests) == 6


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
def test_required_stream_refuses_before_wire_sync_and_async(policy):
    transport = FakeTransport([])
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    async_lm = AsyncOpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)

    async def consume():
        return [event async for event in async_lm.stream(BASE)]

    for invoke in (lambda: list(lm.stream(BASE)), lambda: asyncio.run(consume())):
        with pytest.raises(UnsupportedFeatureError) as caught:
            invoke()
        assert caught.value.feature == "config.probabilities"
        assert "complete()" in str(caught.value)
    assert transport.requests == []


def stream_body():
    frames = [
        {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": None}]},
        {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return ("".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n").encode()


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
def test_if_available_stream_is_unmeasured_and_records_drop(policy):
    request = replace(BASE, config=replace(BASE.config, probabilities="if_available"))
    transport = FakeTransport([FakeResponse(status=200, body=stream_body())])
    lm = OpenAIChatLM(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    if policy == "refuse":
        with pytest.raises(UnsupportedFeatureError) as caught:
            list(lm.stream(request))
        assert caught.value.feature == "config.probabilities" and transport.requests == []
        return
    events = list(lm.stream(request))
    start = next(e for e in events if e.type == "start")
    assert [(a.field, a.action) for a in start.adaptations] == ([] if policy == "silent" else [("config.probabilities", "dropped")])
    assert len(transport.requests) == 1 and transport.requests[0].url.endswith("/chat/completions")
    response = materialize_response(iter(events), request)
    assert response.data == {"ok": True}
    assert response.probabilities is None and response.method is None


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
def test_async_if_available_stream_records_the_same_drop(policy):
    class AsyncStreamTransport(AsyncScoringTransport):
        def reply(self, request):
            self.requests.append(request)
            return stream_body()

    request = replace(BASE, config=replace(BASE.config, probabilities="if_available"))
    transport = AsyncStreamTransport()
    lm = AsyncOpenAIChatLM(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)

    async def consume():
        return [event async for event in lm.stream(request)]

    if policy == "refuse":
        with pytest.raises(UnsupportedFeatureError) as caught:
            asyncio.run(consume())
        assert caught.value.feature == "config.probabilities" and transport.requests == []
        return
    events = asyncio.run(consume())
    start = next(e for e in events if e.type == "start")
    assert [(a.field, a.action) for a in start.adaptations] == ([] if policy == "silent" else [("config.probabilities", "dropped")])
    assert len(transport.requests) == 1 and transport.requests[0].url.endswith("/chat/completions")
    response = materialize_response(iter(events), request)
    assert response.data == {"ok": True}
    assert response.probabilities is None and response.method is None


def test_generated_json_paths_do_not_acquire_scoring_restrictions():
    request, _ = boundary_request("mixed")
    request = replace(request, config=replace(request.config, probabilities="off"))
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1")
    assert lm.plan(request) == ()
    # A non-scoring preset retains the full mixed schema, and ordinary
    # extensions keep their documented canonical override behavior.
    plain = OpenAIChatLM(api_key="k")
    request = replace(request, config=replace(request.config, probabilities="if_available", extensions={"temperature": 0.25}))
    payload = plain._payload(request, stream=False)
    assert payload["response_format"]["json_schema"]["schema"] == request.config.response_format["schema"]
    assert payload["temperature"] == 0.25
    assert [(a.field, a.action) for a in plain.plan(request)] == [("config.probabilities", "dropped")]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("policy", ["note", "silent"])
@pytest.mark.parametrize("missing_ids", [False, True])
def test_mixed_preserves_ordinary_fields_one_generation_and_usage(asynchronous, policy, missing_ids):
    request, _ = boundary_request("mixed")
    request = replace(request, config=replace(request.config, probabilities="if_available", temperature=0.25))
    transport = AsyncScoringTransport() if asynchronous else ScoringTransport()
    transport.missing_ids = missing_ids
    transport.scoring_usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 99,
                               "completion_tokens_details": {"reasoning_tokens": 2}}
    transport.generated_usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 30}
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    plan = lm.plan(request)
    assert [(a.field, a.action) for a in plan] == [("config.response_format", "client_side"), ("config.temperature", "dropped")]
    response = asyncio.run(lm.complete(request)) if asynchronous else lm.complete(request)
    assert response.data == {"ok": not missing_ids, "explanation": "ordinary answer", "extra": {"keep": [1, 2]}}
    assert response.id == "generated" and response.finish_reason == "stop"
    assert (response.method is None) == missing_ids
    assert (response.probabilities is None) == missing_ids
    if not missing_ids:
        assert list(response.probabilities) == ["ok"]
    records = [(a.field, a.action) for a in response.adaptations]
    expected = [(a.field, a.action) for a in plan] + ([("config.probabilities", "dropped")] if missing_ids else [])
    assert records == ([] if policy == "silent" else expected)
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (18, 8, 129)
    assert response.usage.reasoning_tokens is None and response.usage.cache_read_tokens is None
    assert response.provider_data["scoring_usage"]["total_tokens"] == 99
    assert len(transport.requests) == 7
    assert transport.requests[-2].url.endswith("/completions")
    assert transport.requests[-1].url.endswith("/chat/completions")
    payload = json.loads(transport.requests[-1].body)
    assert payload["response_format"]["json_schema"]["schema"] == request.config.response_format["schema"]
    assert payload["temperature"] == 0.25
    assert json.loads(transport.requests[-2].body)["temperature"] == 1.0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("missing_ids", [False, True])
def test_mixed_missing_scoring_usage_stays_unknown(asynchronous, missing_ids):
    request, _ = boundary_request("mixed")
    request = replace(request, config=replace(request.config, probabilities="if_available"))
    transport = AsyncScoringTransport() if asynchronous else ScoringTransport()
    transport.missing_ids = missing_ids
    transport.generated_usage = {"prompt_tokens": 11, "completion_tokens": 5}
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    response = asyncio.run(lm.complete(request)) if asynchronous else lm.complete(request)
    assert response.usage.input_tokens is None and response.usage.output_tokens is None and response.usage.total_tokens is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("content,finish", [('{"ok": false}', "stop"), ('{"ok":', "stop"),
                                           ('[]', "stop"), ('{"ok": false, "explanation": "cut"}', "length"),
                                           ('{"ok": false, "explanation": NaN}', "stop")])
def test_mixed_incomplete_or_malformed_is_provider_error(asynchronous, content, finish):
    request, _ = boundary_request("mixed")
    transport = AsyncScoringTransport() if asynchronous else ScoringTransport()
    transport.generated_content, transport.generated_finish = content, finish
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    with pytest.raises(ProviderError) as caught:
        asyncio.run(lm.complete(request)) if asynchronous else lm.complete(request)
    assert caught.value.provider == "openai-chat" and caught.value.status == 200
    assert caught.value.request_id == "scoring-wire"
    assert len(transport.requests) == 7


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
@pytest.mark.parametrize("changes,feature", [({"store": False}, "config.store"), ({"user_id": "safety"}, "config.user_id"),
    ({"service_tier": "priority"}, "config.service_tier"), ({"cache": CacheConfig(mode="off")}, "config.cache.mode"),
    ({"cache": CacheConfig(retention="long")}, "config.cache.retention"),
    ({"extensions": {"store": False}}, "config.extensions.store"), ({"extensions": {"mystery": 1}}, "config.extensions.mystery")])
def test_unknown_or_harmful_measurement_controls_refuse_prewire(policy, changes, feature):
    request = replace(BASE, config=replace(BASE.config, **changes))
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", adaptations=policy)
    async_lm = AsyncOpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1", adaptations=policy)
    for invoke in (lambda: lm.plan(request), lambda: lm.complete(request), lambda: async_lm.plan(request),
                   lambda: asyncio.run(async_lm.complete(request))):
        with pytest.raises(UnsupportedFeatureError) as caught:
            invoke()
        assert caught.value.feature == feature


def test_ignored_measurement_hints_are_visible_and_strict_rejects_prewire():
    request = replace(BASE, config=replace(BASE.config, temperature=0.2, max_tokens=10, stop=("end",), seed=0,
                                          extensions={"top_p": 0.9}))
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1")
    assert [(a.field, a.action) for a in lm.plan(request)] == [(f"config.{name}", "dropped") for name in
        ("max_tokens", "temperature", "stop", "seed", "extensions")]
    with pytest.raises(UnsupportedFeatureError) as caught:
        lm.plan(request, policy="refuse")
    assert caught.value.feature == "config.max_tokens"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("probabilities,policy", [("required", "note"), ("if_available", "refuse")])
def test_missing_ids_never_generate_when_required_or_strict(asynchronous, probabilities, policy):
    request = replace(BASE, config=replace(BASE.config, probabilities=probabilities))
    transport = AsyncScoringTransport() if asynchronous else ScoringTransport()
    transport.missing_ids = True
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport, adaptations=policy)
    with pytest.raises(UnsupportedFeatureError) as caught:
        asyncio.run(lm.complete(request)) if asynchronous else lm.complete(request)
    assert caught.value.feature == "config.probabilities"
    assert len(transport.requests) == 6
    assert not any(r.url.endswith("/chat/completions") for r in transport.requests)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_fallback_cannot_omit_a_required_judgment(asynchronous):
    request, _ = boundary_request("mixed")
    request = replace(request, config=replace(request.config, probabilities="if_available"))
    transport = AsyncScoringTransport() if asynchronous else ScoringTransport()
    transport.missing_ids = True
    transport.generated_content = '{"explanation": "ordinary only"}'
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    with pytest.raises(ProviderError) as caught:
        asyncio.run(lm.complete(request)) if asynchronous else lm.complete(request)
    assert caught.value.status == 200 and caught.value.request_id == "scoring-wire"
    assert len(transport.requests) == 7


@pytest.mark.parametrize("asynchronous", [False, True])
def test_multiple_judgments_still_use_one_batch_and_one_generation(asynchronous):
    fmt = judgments(ok=yes_no("Fine?"), another=yes_no("Again?"), explanation={"type": "string"})
    request = replace(BASE, config=replace(BASE.config, response_format=fmt))
    transport = AsyncScoringTransport() if asynchronous else ScoringTransport()
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    response = asyncio.run(lm.complete(request)) if asynchronous else lm.complete(request)
    assert response.data["ok"] is True and response.data["another"] is True
    assert response.data["explanation"] == "ordinary answer"
    assert set(response.probabilities) == {"ok", "another"}
    assert len(transport.requests) == 12
    assert len(json.loads(transport.requests[-2].body)["prompt"]) == 6
    assert response.provider_data["judgments"]["tokenize_calls"] == 10


def test_single_output_extension_is_not_a_multiple_output_refusal():
    lm = OpenAIChatLM(api_key=forbidden, compat="vllm", base_url="http://scoring/v1")
    request = replace(BASE, config=replace(BASE.config, extensions={"n": 1}))
    assert [(a.field, a.action) for a in lm.plan(request)] == [("config.extensions", "dropped")]
