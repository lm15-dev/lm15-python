"""Tinker (Thinking Machines): the two doors, the reasoning switch per base
family, fine-tunes that follow their base model, URI model names, the
FastAPI error body, the stop sequence returned in the text, and the catalog's
fine-tune origin (lm15-contract changes/2026-11-09-tinker-live.md; every rule
has a receipt under receipts/2026-11-09-tinker-survey/).
"""
from __future__ import annotations

import json

import pytest

from lm15 import AnthropicLM, Config, LMRouter, Message, OpenAIChatLM, Request, RouterConfig
from lm15.errors import AuthError, InvalidRequestError, NotConfiguredError, UnknownModelError, UnsupportedModelError
from lm15.registry import PROVIDERS, ProviderDefinition
from lm15.access import AccessPolicy
from lm15.compat import OpenAIChatCompat
from lm15.errors import detail_message
from lm15.router import openai_chat_model_string
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import Reasoning, TextPart

CHECKPOINT = "tinker://e68146d2-1a2e-56e8-b42f-1f5957705501:train:0/sampler_weights/lm15-untrained"


def ask(model: str, **config) -> Request:
    return Request(model=model, messages=(Message.user("hi"),), config=Config(**config))


def chat(**kw) -> OpenAIChatLM:
    return OpenAIChatLM(api_key="k", compat="tinker", access=PROVIDERS["tinker"].access, **kw)


def wire(lm, request: Request) -> tuple[dict, tuple]:
    treq, records = lm._build(request, False)
    return json.loads(treq.body), records


# ─── registry ────────────────────────────────────────────────────────


def test_two_doors_one_key():
    chat_door, messages_door = PROVIDERS["tinker"], PROVIDERS["tinker-anthropic"]
    assert chat_door.access.env_keys == messages_door.access.env_keys == ("TINKER_API_KEY",)
    assert chat_door.access.base_url == "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    assert messages_door.access.base_url == "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1"
    assert chat_door.access.supports.models and not messages_door.access.supports.models  # the Messages door's /models is a 404
    assert chat_door.aliases == ("thinkingmachines",)


# ─── routing: a URI is one name ───────────────────────────────────────


@pytest.mark.parametrize("model,provider,wire_model", [
    (CHECKPOINT, "tinker", CHECKPOINT),
    (f"tinker:{CHECKPOINT}", "tinker", CHECKPOINT),
    (f"tinker-anthropic:{CHECKPOINT}", "tinker-anthropic", CHECKPOINT),
    (f"thinkingmachines:{CHECKPOINT}", "tinker", CHECKPOINT),
    ("tinker:Qwen/Qwen3.5-4B", "tinker", "Qwen/Qwen3.5-4B"),
    ("thinkingmachines:thinkingmachines/Inkling", "tinker", "thinkingmachines/Inkling"),
])
def test_routing(model, provider, wire_model):
    resolution = LMRouter(RouterConfig(env={})).resolve(model)
    assert (resolution.provider, resolution.model) == (provider, wire_model)


def test_a_uri_is_never_provider_model():
    # `openai://x` names no route: before the rule, it routed to openai with model "//x".
    with pytest.raises(UnknownModelError):
        LMRouter(RouterConfig(env={})).resolve("openai://x")


def test_registry_alias_in_litellm_spelling_and_not_declarable():
    assert openai_chat_model_string("thinkingmachines/Qwen/Qwen3-8B") == "tinker:Qwen/Qwen3-8B"
    declared = ProviderDefinition.chat(AccessPolicy(provider="thinkingmachines", env_keys=("X",), base_url="https://x"),
                                       compat=OpenAIChatCompat())
    with pytest.raises(NotConfiguredError, match="already names lm15's 'tinker' door"):
        RouterConfig(providers=(declared,))


# ─── reasoning, per base family ──────────────────────────────────────


@pytest.mark.parametrize("model,off_wire", [
    ("thinkingmachines/Inkling-Small", "none"),       # the documented off word
    ("Qwen/Qwen3.5-4B", False),                         # a switch
    ("moonshotai/Kimi-K2.6", False),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", False),
    ("Qwen/Qwen3.8-27B", False),                        # words, and false is off
    ("deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", False),
    (CHECKPOINT, False),                                # base unknown: the host default
])
def test_reasoning_off(model, off_wire):
    body, records = wire(chat(), ask(model, reasoning=Reasoning(effort="off")))
    assert body["reasoning_effort"] is off_wire if isinstance(off_wire, bool) else body["reasoning_effort"] == off_wire
    assert not records


def test_effort_word_on_a_switch_is_on_and_recorded():
    body, records = wire(chat(), ask("Qwen/Qwen3.5-4B", reasoning=Reasoning(effort="high")))
    assert body["reasoning_effort"] is True
    (record,) = records
    assert (record.field, record.action, record.asked, record.applied) == ("config.reasoning.effort", "substituted", "high", "on")


@pytest.mark.parametrize("model", ["thinkingmachines/Inkling", "openai/gpt-oss-20b", "Qwen/Qwen3.8-27B", CHECKPOINT])
def test_effort_words_go_verbatim(model):
    body, records = wire(chat(), ask(model, reasoning=Reasoning(effort="low")))
    assert body["reasoning_effort"] == "low" and not records


def test_a_checkpoint_follows_its_declared_base_model():
    lm = chat(base_models={CHECKPOINT: "thinkingmachines/Inkling-Small"})
    assert wire(lm, ask(CHECKPOINT, reasoning=Reasoning(effort="off")))[0]["reasoning_effort"] == "none"
    lm = chat(base_models={CHECKPOINT: "Qwen/Qwen3.5-4B"})
    body, records = wire(lm, ask(CHECKPOINT, reasoning=Reasoning(effort="medium")))
    assert body["reasoning_effort"] is True and records[0].applied == "on"


def test_router_base_models_reach_the_chat_door():
    config = RouterConfig(env={"TINKER_API_KEY": "k"}, base_models={CHECKPOINT: "thinkingmachines/Inkling"})
    plan_lm = LMRouter(config).lm(CHECKPOINT)
    assert plan_lm.base_models == {CHECKPOINT: "thinkingmachines/Inkling"}
    assert wire(plan_lm, ask(CHECKPOINT, reasoning=Reasoning(effort="off")))[0]["reasoning_effort"] == "none"
    with pytest.raises(TypeError):
        RouterConfig(base_models={CHECKPOINT: ""})


def test_ingest_reads_the_switch_back():
    lm = chat()
    assert lm.request_from_openai_chat({"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "hi"}],
                                        "reasoning_effort": False}).config.reasoning == Reasoning(effort="off")
    assert lm.request_from_openai_chat({"model": CHECKPOINT, "messages": [{"role": "user", "content": "hi"}],
                                        "reasoning_effort": False}).config.reasoning == Reasoning(effort="off")


# ─── response_format: applied by nobody ──────────────────────────────


@pytest.mark.parametrize("fmt", [{"type": "json_object"},
                                 {"type": "json_schema", "name": "x", "schema": {"type": "object"}}])
def test_response_format_is_dropped_and_recorded(fmt):
    body, records = wire(chat(), ask("thinkingmachines/Inkling", response_format=fmt))
    assert "response_format" not in body
    assert [(r.field, r.action) for r in records] == [("config.response_format", "dropped")]


# ─── the stop sequence comes back in the text (MAP-22) ───────────────


def _chat_reply(text: str) -> bytes:
    return json.dumps({"id": "x", "object": "chat.completion", "model": "thinkingmachines/Inkling-Small",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12,
                                 "prompt_tokens_details": {"cached_tokens": 0}}}).encode()


def test_stop_sequence_cut_complete():
    lm = chat(transport=FakeTransport([FakeResponse(status=200, body=_chat_reply("1 2 3 4 5 6 7"))]))
    response = lm.complete(ask("thinkingmachines/Inkling-Small", stop=(" 7",)))
    assert response.text == "1 2 3 4 5 6" and not response.adaptations
    sent = json.loads(lm.transport.requests[0].body)
    assert sent["stop"] == [" 7"]  # the server still stops there: no tokens past it are paid for


def test_stop_sequence_cut_stream():
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "1 2 3 4 5 6 7"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
    ]
    sse = b"".join(b"data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": "m", **f}).encode() + b"\n\n"
                   for f in frames) + b"data: [DONE]\n\n"
    lm = chat(transport=FakeTransport([FakeResponse(status=200, body=sse, chunks=[sse])]))
    text = "".join(e.delta.text for e in lm.stream(ask("thinkingmachines/Inkling-Small", stop=(" 7",)))
                   if e.type == "delta" and getattr(e.delta, "text", None))
    assert text == "1 2 3 4 5 6"


def test_stop_sequence_cut_messages_door():
    reply = {"id": "msg", "type": "message", "role": "assistant", "model": "thinkingmachines/Inkling-Small",
             "content": [{"type": "text", "text": "1 2 3 4 5 6 7"}], "stop_reason": "end_turn",
             "usage": {"input_tokens": 0, "output_tokens": 15, "cache_creation_input_tokens": 26, "cache_read_input_tokens": 0}}
    lm = AnthropicLM(api_key="k", compat="tinker", access=PROVIDERS["tinker-anthropic"].access,
                     transport=FakeTransport([FakeResponse(status=200, body=json.dumps(reply).encode())]))
    response = lm.complete(Request(model="thinkingmachines/Inkling-Small", messages=(Message.user("count"),),
                                   config=Config(stop=(" 7",), max_tokens=50)))
    assert response.text == "1 2 3 4 5 6"
    # MAP-19: the uncached prompt is reported as a cache write; input is the whole prompt.
    assert (response.usage.input_tokens, response.usage.cache_write_tokens, response.usage.cache_read_tokens) == (26, 26, 0)


def test_other_providers_keep_the_stop_sequence_rule():
    lm = OpenAIChatLM(api_key="k", compat="openai",
                      transport=FakeTransport([FakeResponse(status=200, body=_chat_reply("a END b"))]))
    assert lm.complete(ask("gpt-4.1-mini", stop=("END",))).text == "a END b"  # never cut where the server's own rule holds


# ─── errors: FastAPI's {"detail": ...} (MAP-21) and the MAP-15 forms ──


@pytest.mark.parametrize("status,body,cls,message", [
    (401, {"detail": "Unable to validate api-key"}, AuthError, "Unable to validate api-key"),
    (404, {"detail": f"Weights not found for model_path {CHECKPOINT}"}, UnsupportedModelError,
     f"Weights not found for model_path {CHECKPOINT}"),
    (400, {"detail": "Sampling is not supported for lm15/none. (organization: \"<redacted> - default org\" <redacted>)"},
     UnsupportedModelError, None),
    (400, {"detail": "model_path 'tinker://not-a-path' is not a valid tinker path. Expected a checkpoint path"},
     UnsupportedModelError, None),
    (422, {"detail": "model returned no tool call when tool_choice required one"}, InvalidRequestError,
     "model returned no tool call when tool_choice required one"),
    (422, {"detail": "Validation Error", "errors": [{"loc": "body -> messages -> 0 -> role", "msg": "Input should be 'system'",
                                                     "type": "literal_error"}]},
     InvalidRequestError, "Validation Error: body -> messages -> 0 -> role: Input should be 'system'"),
])
def test_chat_errors(status, body, cls, message):
    error = chat().normalize_error(status, json.dumps(body))
    assert type(error) is cls and error.provider_code is None
    if message:
        assert error.message.split("\n")[0] == message


def test_messages_door_errors():
    lm = AnthropicLM(api_key="k", compat="tinker", access=PROVIDERS["tinker-anthropic"].access)
    assert isinstance(lm.normalize_error(404, json.dumps({"detail": f"Weights not found for model_path {CHECKPOINT}"})),
                      UnsupportedModelError)
    err = lm.normalize_error(400, json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                                          "message": "Sampling is not supported for x. (org)"}}))
    assert isinstance(err, UnsupportedModelError)


def test_detail_message_shapes():
    assert detail_message({"detail": [{"loc": ["body", "messages", 0], "msg": "field required"}]}) == "body -> messages -> 0: field required"
    assert detail_message({"error": {"message": "x"}, "detail": "y"}) is None
    assert detail_message({"detail": ""}) is None


# ─── the catalog: checkpoints are fine-tunes of a named base (MAP-23) ──


def test_catalog_names_the_base_model():
    body = {"object": "list", "data": [
        {"id": CHECKPOINT, "object": "model", "created": 1794234918, "owned_by": "tml:organization_user:<redacted>",
         "base_model": "Qwen/Qwen3.5-4B"},
    ]}
    lm = chat(transport=FakeTransport([FakeResponse(status=200, body=json.dumps(body).encode())]))
    (info,) = lm.list_models()
    assert (info.id, info.provider, info.origin.type, info.origin.base_model) == (CHECKPOINT, "tinker", "fine-tune", "Qwen/Qwen3.5-4B")
    # The catalog feeds base_models, explicitly:
    config = RouterConfig(env={"TINKER_API_KEY": "k"}, base_models={info.id: info.origin.base_model})
    assert wire(LMRouter(config).lm(CHECKPOINT), ask(CHECKPOINT, reasoning=Reasoning(effort="high")))[0]["reasoning_effort"] is True


def test_async_door_mirrors_base_models_and_stop():
    from lm15.providers.async_base import AsyncOpenAIChatLM

    lm = AsyncOpenAIChatLM(api_key="k", compat="tinker", access=PROVIDERS["tinker"].access, base_models={CHECKPOINT: "thinkingmachines/Inkling"})
    assert json.loads(lm._inner.build_request(ask(CHECKPOINT, reasoning=Reasoning(effort="off")), stream=False).body)["reasoning_effort"] == "none"
    assert lm._inner._stop_returned_in_text(ask("x", stop=("a",)))
