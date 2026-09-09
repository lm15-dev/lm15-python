"""MAP-12: request_from_openai_chat — a Chat Completions request body → Request.

The contract pins the round trip over every recorded chat-dialect body and
the refusals (lm15-contract `--direction ingest`); this file pins what the
contract deliberately does not: the malformed-input class (ValueError /
TypeError, MAP-12 rule 6), the preset-conditioned rows under presets the
corpus has no door for, and — so the Python suite alone catches drift — the
same round trip read straight from the contract checkout.
"""

from __future__ import annotations

import json

import pytest

import lm15
from conformance.sources import contract_root
from lm15 import serde
from lm15.compat import OpenAIChatCompat
from lm15.errors import UnsupportedFeatureError
from lm15.providers import OpenAIChatLM
from lm15.providers.openai_chat import request_from_openai_chat
from lm15.types import (
    CacheConfig,
    Config,
    FunctionTool,
    Message,
    Reasoning,
    Request,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
)
from lm15.vet import adapter_for_provider

USER = [{"role": "user", "content": "Hi"}]


def body(**extra):
    return {"model": "gpt-5-mini", "messages": USER, **extra}


# ─── the public surface ──────────────────────────────────────────────

def test_exported_at_top_level_and_on_the_adapter() -> None:
    assert lm15.request_from_openai_chat is request_from_openai_chat
    req = OpenAIChatLM(api_key="k").request_from_openai_chat(body())
    assert req == request_from_openai_chat(body())
    assert req == Request(model="gpt-5-mini", messages=(Message.user("Hi"),))


def test_compat_argument_forms() -> None:
    b = body(reasoning={"effort": "low"})
    assert request_from_openai_chat(b, compat="openrouter").config.reasoning == Reasoning(effort="low")
    assert request_from_openai_chat(b, compat=OpenAIChatCompat.preset("openrouter")).config.reasoning == Reasoning(effort="low")
    with pytest.raises(UnsupportedFeatureError):
        request_from_openai_chat(b)  # OpenAI's dial is spelled reasoning_effort
    with pytest.raises(TypeError):
        request_from_openai_chat(b, compat=42)  # type: ignore[arg-type]


def test_model_overrides_apply_per_model() -> None:
    # A per-model override flips the spelling the decoder reads.
    compat = OpenAIChatCompat(user_field="user", model_overrides=(("deep", {"user_field": "user_id"}),))
    assert request_from_openai_chat({"model": "deep-1", "messages": USER, "user_id": "u"}, compat=compat).config.user_id == "u"
    with pytest.raises(UnsupportedFeatureError):
        request_from_openai_chat({"model": "other", "messages": USER, "user_id": "u"}, compat=compat)


# ─── the round trip, read straight from the contract (rule 7) ────────

def _chat_cases():
    root = contract_root()
    for path in sorted((root / "cases").glob("*/*.json")):
        case = json.loads(path.read_text())
        req = case.get("request") or {}
        if not str(req.get("url", "")).split("?", 1)[0].endswith("/chat/completions"):
            continue
        if "canonical_request" not in case:
            continue
        raises = (case.get("expect_lm15") or {}).get("raises")
        if raises and raises.get("op") == "build_request":
            continue
        yield case


@pytest.mark.parametrize("case", list(_chat_cases()), ids=lambda c: c["id"])
def test_every_recorded_chat_body_reads_back_to_its_canonical_request(case) -> None:
    lm = adapter_for_provider(case["provider"], "k", case.get("base_url"), settings=case.get("settings"))
    got = serde.request_to_dict(lm.request_from_openai_chat(case["request"]["body"]))
    ingest = case.get("ingest")
    want = ingest["canonical_request"] if isinstance(ingest, dict) else case["canonical_request"]
    assert got == want


def test_build_then_ingest_is_identity_on_a_rich_request() -> None:
    req = Request(
        model="gpt-5-mini",
        system="Be brief.",
        messages=(
            Message.user("Weather in Paris and Lyon?"),
            Message.assistant((ToolCallPart(id="c1", name="w", input={"city": "Paris"}), ToolCallPart(id="c2", name="w", input={"city": "Lyon"}))),
            Message.tool({"c1": "18C", "c2": "21C"}),
            Message.user("Thanks"),
        ),
        tools=(FunctionTool(name="w", description="Weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}}),),
        config=Config(max_tokens=100, temperature=0.5, top_p=0.9, stop=("END",), logprobs=2,
                      response_format={"type": "json_schema", "schema": {"type": "object"}, "name": "Out", "strict": True},
                      tool_choice=ToolChoice(mode="auto", parallel=False), reasoning=Reasoning(effort="low"),
                      service_tier="flex", user_id="u", store=False, extensions={"seed": 7}),
    )
    lm = OpenAIChatLM(api_key="k")
    wire = lm.build_request(req, stream=False)
    assert lm.request_from_openai_chat(json.loads(wire.body)) == req


# ─── the buckets (rule 2) ────────────────────────────────────────────

def test_extensions_bucket_is_verbatim_and_round_trips() -> None:
    req = request_from_openai_chat(body(seed=7, logit_bias={"1": -100}, presence_penalty=0.5, frequency_penalty=0, metadata={"k": "v"}, verbosity="low"))
    assert req.config.extensions == {"seed": 7, "logit_bias": {"1": -100}, "presence_penalty": 0.5, "frequency_penalty": 0, "metadata": {"k": "v"}, "verbosity": "low"}
    sent = json.loads(OpenAIChatLM(api_key="k").build_request(req, stream=False).body)
    assert sent["seed"] == 7 and sent["logit_bias"] == {"1": -100} and sent["frequency_penalty"] == 0


@pytest.mark.parametrize("extra", [
    {"n": 2}, {"functions": []}, {"function_call": "auto"}, {"audio": {"voice": "alloy", "format": "wav"}},
    {"modalities": ["text", "audio"]}, {"prediction": {"type": "content", "content": "x"}},
    {"web_search_options": {}}, {"top_k": 3},
    {"never_heard_of_it": 1},  # no verdict -> refused, never dropped
])
def test_refused_keys_name_the_key(extra) -> None:
    with pytest.raises(UnsupportedFeatureError) as exc:
        request_from_openai_chat(body(**extra))
    assert next(iter(extra)) in str(exc.value)


def test_call_mode_keys_are_the_only_drop() -> None:
    assert request_from_openai_chat(body(stream=True, stream_options={"include_usage": True})) == request_from_openai_chat(body())


def test_default_valued_keys_read_as_absent() -> None:
    plain = request_from_openai_chat(body())
    assert request_from_openai_chat(body(response_format={"type": "text"})) == plain
    assert request_from_openai_chat(body(logprobs=False)) == plain
    named = request_from_openai_chat(body(response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": {"type": "object"}}}))
    assert named.config.response_format == {"type": "json_schema", "schema": {"type": "object"}}
    tool = request_from_openai_chat(body(tools=[{"type": "function", "function": {"name": "t", "strict": False}}]))
    assert tool.tools == (FunctionTool(name="t"),)


# ─── rows and blocks (rules 3–5) ─────────────────────────────────────

def test_consecutive_tool_rows_coalesce_and_name_maps() -> None:
    req = request_from_openai_chat({"model": "m", "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "1", "name": "f"},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "2"}]},
        {"role": "user", "content": "and?"},
    ]})
    assert [m.role for m in req.messages] == ["user", "assistant", "tool", "user"]
    assert req.messages[2].parts == (
        ToolResultPart(id="a", content=(TextPart("1"),), name="f"),
        ToolResultPart(id="b", content=(TextPart("2"),)),
    )
    assert req.messages[1].parts[1] == ToolCallPart(id="b", name="f", input={"x": 1})


def test_error_prefix_is_not_reversed() -> None:
    req = request_from_openai_chat({"model": "m", "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "[error] boom"}]})
    part = req.messages[2].parts[0]
    assert part.is_error is False and part.content == (TextPart("[error] boom"),)


def test_reasoning_content_reads_on_every_preset_but_the_knob_does_not() -> None:
    rows = [{"role": "user", "content": "hi"}, {"role": "assistant", "reasoning_content": "hmm", "content": "yo"}, {"role": "user", "content": "k"}]
    req = request_from_openai_chat({"model": "m", "messages": rows})
    assert req.messages[1].parts == (ThinkingPart("hmm"), TextPart("yo"))
    with pytest.raises(UnsupportedFeatureError):
        request_from_openai_chat({"model": "m", "messages": rows, "thinking": {"type": "disabled"}})
    assert request_from_openai_chat({"model": "m", "messages": rows, "thinking": {"type": "disabled"}}, compat="deepseek").config.reasoning == Reasoning(effort="off")


def test_system_row_placement() -> None:
    req = request_from_openai_chat({"model": "m", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]},
        {"role": "user", "content": "hi"},
        {"role": "developer", "content": "now B"}]})
    assert req.system == (TextPart("A"), TextPart("B"))
    assert req.messages[1] == Message.developer("now B")


def test_breakpoints_map_to_cache_config() -> None:
    stable = request_from_openai_chat({"model": "gpt-5.6-sol", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "S", "prompt_cache_breakpoint": {"mode": "explicit"}}]},
        {"role": "user", "content": "hi"}], "prompt_cache_options": {"mode": "explicit"}})
    assert stable.config.cache == CacheConfig(prefix="stable")
    off = request_from_openai_chat({"model": "gpt-5.6-sol", "messages": USER, "prompt_cache_options": {"mode": "explicit"}})
    assert off.config.cache == CacheConfig(mode="off")
    with pytest.raises(UnsupportedFeatureError):
        request_from_openai_chat(body(prompt_cache_key="k"), compat="groq")  # no cache control on that preset


# ─── malformed input: ValueError / TypeError, not pinned by the contract (rule 6) ─

@pytest.mark.parametrize("bad, exc", [
    ({"model": "", "messages": USER}, ValueError),
    ({"model": "m"}, ValueError),
    ({"model": "m", "messages": "hi"}, TypeError),
    ({"model": "m", "messages": [{"role": "user", "content": 5}]}, TypeError),
    ({"model": "m", "messages": [{"role": "narrator", "content": "x"}]}, ValueError),
    ({"model": "m", "messages": [{"role": "tool", "content": "x"}]}, TypeError),  # no tool_call_id
    ({"model": "m", "messages": [{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "not json"}}]}]}, ValueError),
    ({"model": "m", "messages": [{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "[1]"}}]}]}, ValueError),
    ({"model": "m", "messages": USER, "max_tokens": 1, "max_completion_tokens": 2}, ValueError),
    ({"model": "m", "messages": USER, "user": "a", "safety_identifier": "b"}, ValueError),
    ({"model": "m", "messages": USER, "top_logprobs": 3}, ValueError),
    ({"model": "m", "messages": USER, "tool_choice": {"type": "function", "function": {"name": "ghost"}}}, ValueError),  # INV-031
    ({"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,notbase64"}}]}]}, ValueError),
    ({"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "a", "prompt_cache_breakpoint": {"mode": "explicit"}}, {"type": "text", "text": "b"}]}]}, ValueError),  # not last
    ([], TypeError),
])
def test_malformed_input_raises_native_errors(bad, exc) -> None:
    with pytest.raises(exc):
        request_from_openai_chat(bad)


def test_vet_op_refuses_a_non_chat_provider() -> None:
    from lm15.vet import op_ingest_openai_chat

    with pytest.raises(ValueError):
        op_ingest_openai_chat({"provider": "anthropic", "body": body()})
    out = op_ingest_openai_chat({"provider": "openai_chat", "body": body()})
    assert out == {"canonical_request": {"model": "gpt-5-mini", "messages": [{"role": "user", "parts": [{"type": "text", "text": "Hi"}]}]}}


# ─── message objects dumped back into history (MAP-12 addendum) ──────

SDK_MESSAGE = {"content": "Ok! How can I help?", "refusal": None, "role": "assistant", "annotations": [], "audio": None, "function_call": None, "tool_calls": None}
LITELLM_MESSAGE = {"content": "Ok!", "role": "assistant", "tool_calls": None, "function_call": None, "provider_specific_fields": {"refusal": None}, "annotations": []}


def test_dumped_message_objects_read_as_one_text_part() -> None:
    for m in (SDK_MESSAGE, LITELLM_MESSAGE):
        req = request_from_openai_chat({"model": "m", "messages": [{"role": "user", "content": "x"}, m, {"role": "user", "content": "y"}]})
        assert req.messages[1] == Message.assistant(TextPart(m["content"]))


def test_annotations_become_citations_and_object_model_fields_refuse_when_filled() -> None:
    from lm15.types import CitationPart

    text = "The sky is blue because of Rayleigh scattering."
    row = {"role": "assistant", "content": text, "annotations": [{"type": "url_citation", "url_citation": {"url": "https://w/rs", "title": "RS", "start_index": 27, "end_index": 46}}]}
    req = request_from_openai_chat({"model": "m", "messages": [{"role": "user", "content": "x"}, row]})
    assert req.messages[1].parts == (TextPart(text), CitationPart(url="https://w/rs", title="RS", text="Rayleigh scattering"))
    with pytest.raises(UnsupportedFeatureError):
        request_from_openai_chat({"model": "m", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "provider_specific_fields": {"refusal": "no"}}]})
    with pytest.raises(UnsupportedFeatureError):
        request_from_openai_chat({"model": "m", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "a", "annotations": [{"type": "file_citation", "file_citation": {}}]}]})
