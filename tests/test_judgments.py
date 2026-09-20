"""Judgments (changes/2026-09-17-judgments.md): DataPart, Config.probabilities,
the MAP-14 schema convention, the typesafe adapter, the cloud adapters'
answers, and the vLLM token-trie driver — all offline."""
from __future__ import annotations

import json

import pytest

from lm15 import (
    Config,
    DataPart,
    Message,
    Request,
    TypeSafeLM,
    choice,
    data,
    judgments,
    score,
    yes_no,
)
from lm15 import serde
from lm15.errors import AuthError, InvalidRequestError, UnsupportedFeatureError, UnsupportedModelError
from lm15.judgments import anthropic_schema, gemini_schema, judgments_in_schema, normalize_logprobs, replace_text_with_data
from lm15.providers import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import FunctionTool, TextPart

WINE = judgments(
    quality=score("How good?", {"bad": "Bad wine", "ok": "Fine wine", "great": "Great wine"}),
    style=choice("Which style?", {"fruit": "Fruit-forward", "oak": "Oak-driven", "other": None}),
    ageing=yes_no("Will it age?"),
)
NOTE = Message.user("Blackberry, oak, decades ahead.")


# ─── types & serde ───────────────────────────────────────────────────


def test_data_part_validation_and_inv_052() -> None:
    part = data({"q": 1}, probabilities={"q": {"1": 0.9, "0": 0.1}}, method="provider_classification")
    assert part.type == "data" and part.probabilities == {"q": {"1": 0.9, "0": 0.1}}
    Message.assistant(part)
    with pytest.raises(TypeError):
        Message.user(part)  # INV-052: input carries value alone
    with pytest.raises(ValueError):
        DataPart(value=1, probabilities={"q": {"a": 0.5}})  # method iff probabilities
    with pytest.raises(ValueError):
        DataPart(value=1, probabilities={"q": {"a": 1.5}}, method="provider_classification")
    with pytest.raises(ValueError):
        DataPart(value=1, probabilities={"q": {"a": 0.5}}, method="guessed")
    assert data(None).value is None  # null is a value


def test_data_part_serde_roundtrip_and_null_value() -> None:
    wire = {"type": "data", "value": {"q": 2}, "probabilities": {"q": {"2": 1.0}}, "method": "candidate_sequence_likelihood"}
    assert serde.part_to_dict(serde.part_from_dict(wire)) == wire
    assert serde.part_to_dict(serde.part_from_dict({"type": "data", "value": None})) == {"type": "data", "value": None}
    with pytest.raises(ValueError):
        serde.part_from_dict({"type": "data"})
    assert serde.config_to_dict(Config(probabilities="required")) == {"probabilities": "required"}
    with pytest.raises(ValueError):
        Config(probabilities="maybe")


# ─── the convention ──────────────────────────────────────────────────


def test_schema_convention_reads_the_three_shapes_and_nothing_else() -> None:
    found = judgments_in_schema(WINE["schema"])
    assert [(j.name, j.kind, j.keys) for j in found.values()] == [
        ("quality", "ordered", ("0", "1", "2")),
        ("style", "choice", ("fruit", "oak", "other")),
        ("ageing", "boolean", ("true", "false")),
    ]
    assert found["quality"].titles["0"] == "bad" and found["style"].descriptions["other"] is None
    plain = judgments_in_schema({"type": "object", "properties": {
        "name": {"type": "string"}, "n": {"enum": [1, 2]}, "lvl": {"enum": [0, 1]}, "tag": {"enum": ["a", "a"]}}})
    assert list(plain) == ["lvl"]  # free text, a non-0-based int enum and duplicate keys are not judgments


def test_wire_rewrites_are_exactly_the_receipted_ones() -> None:
    found = judgments_in_schema(WINE["schema"])
    a = anthropic_schema(WINE["schema"], found)
    assert "type" not in a["properties"]["quality"] and a["properties"]["quality"]["anyOf"][0]["type"] == "integer"
    assert a["properties"]["ageing"] == WINE["schema"]["properties"]["ageing"]
    g = gemini_schema(WINE["schema"], found)
    assert g["properties"]["style"]["enum"] == ["fruit", "oak", "other"] and "anyOf" not in g["properties"]["style"]
    assert g["properties"]["quality"]["enum"] == [0, 1, 2]
    assert g["properties"]["style"]["description"] == "Which style? Options: fruit = Fruit-forward; oak = Oak-driven; other"
    assert WINE["schema"]["properties"]["style"].get("enum") is None  # the canonical schema is untouched


def test_answer_text_becomes_a_data_part_only_when_it_is_a_json_object() -> None:
    found = judgments_in_schema(WINE["schema"])
    parts = replace_text_with_data((TextPart(text='{"quality": 2}'),), found)
    assert isinstance(parts[0], DataPart) and parts[0].value == {"quality": 2}
    assert isinstance(replace_text_with_data((TextPart(text='{"quality":'),), found)[0], TextPart)
    assert normalize_logprobs({"a": -1.0, "b": -1.0}) == {"a": 0.5, "b": 0.5}


# ─── typesafe ────────────────────────────────────────────────────────

JEV_BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "quality": {"type": "score", "score": 1.9, "confidence": 0.9, "legend": {"0": "Bad wine", "1": "Fine wine", "2": "Great wine"},
                    "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}},
        "style": {"type": "choice", "choice": "oak", "confidence": 1.0, "probabilities": {"fruit": 0.2, "oak": 0.8, "other": 0.0}},
        "ageing": {"type": "noul", "noul": 0.97},
    },
    "usage": {"input_tokens": 50, "output_tokens": 7},
}


def _jev(*bodies, status=200):
    return TypeSafeLM(api_key="k", transport=FakeTransport([
        FakeResponse(status=status, body=json.dumps(b).encode(), headers=[("x-typesafe-request-id", "req_1")]) for b in bodies]))


def test_typesafe_builds_questions_state_and_answers_a_data_part() -> None:
    lm = _jev(JEV_BODY)
    req = Request(model="jev-latest", messages=(NOTE,), config=Config(response_format=WINE, temperature=0.3))
    wire = json.loads(lm.build_request(req, stream=False).body)
    assert wire["state"] == "Blackberry, oak, decades ahead."
    assert wire["questions"]["quality"] == {"type": "score", "instructions": "How good?", "criteria": ["Bad wine", "Fine wine", "Great wine"]}
    assert wire["questions"]["style"]["criteria"] == {"fruit": "Fruit-forward", "oak": "Oak-driven", "other": None}
    assert wire["questions"]["ageing"] == {"type": "noul", "instructions": "Will it age?"}
    assert [(a.field, a.action) for a in lm.plan(req)] == [("config.temperature", "dropped")]
    r = lm.complete(req)
    assert r.data == {"quality": 2, "style": "oak", "ageing": True}
    assert r.probabilities["ageing"] == {"true": 0.97, "false": pytest.approx(0.03)}
    assert r.method == "provider_classification" and r.expected("quality") == pytest.approx(1.9)
    assert r.id == "req_1" and r.model == "jev-1.13.0" and r.usage.input_tokens == 50
    assert r.provider_data["typesafe"]["answers"]["quality"]["confidence"] == 0.9
    assert r.adaptations[0].field == "config.temperature"


def test_typesafe_state_shapes_and_refusals() -> None:
    lm = TypeSafeLM(api_key="k")
    one_data = Request(model="jev-latest", messages=(Message.user(data({"note": "x"})),), config=Config(response_format=judgments(ok=yes_no("Ok?"))))
    assert json.loads(lm.build_request(one_data, stream=False).body)["state"] == {"note": "x"}
    # 2026-09-19 D1: the state is the one user part, verbatim — an array too; a transcript is the caller's object, not the adapter's.
    fmt = Config(response_format={"type": "json_schema", "name": "t", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}})
    transcript = Request(model="jev-latest", messages=(Message.user(data({"context": "Triage.", "messages": [{"from": "customer", "text": "a"}, {"from": "agent", "text": "b"}]})),), config=fmt)
    wire = json.loads(lm.build_request(transcript, stream=False).body)
    assert wire["state"] == {"context": "Triage.", "messages": [{"from": "customer", "text": "a"}, {"from": "agent", "text": "b"}]}
    assert wire["questions"]["ok"]["instructions"] == "ok"
    assert [(a.field, a.action, a.applied) for a in lm.plan(transcript)] == [("config.response_format.schema.properties.ok.description", "defaulted", "ok")]
    assert json.loads(lm.build_request(Request(model="jev-latest", messages=(Message.user(data(["Hi", "My card was charged twice."])),), config=fmt), stream=False).body)["state"] == ["Hi", "My card was charged twice."]
    # 2026-09-19 D2: what Jev has no slot for is refused with the native place named, never merged.
    for req, feature, why in [
        (Request(model="jev-latest", system="Triage.", messages=(NOTE,), config=fmt), "system", "no system prompt"),
        (Request(model="jev-latest", messages=(Message.user("a"), Message.assistant("b"), Message.user("c")), config=fmt), "messages", "one state, got 3 messages"),
        (Request(model="jev-latest", messages=(Message.user(["a", "b"]),), config=fmt), "messages[0].parts", "got 2 parts"),
    ]:
        with pytest.raises(UnsupportedFeatureError) as exc:
            lm.build_request(req, stream=False)
        assert exc.value.feature == feature and why in str(exc.value)
    for req, feature in [
        (Request(model="jev-latest", messages=(NOTE,)), "config.response_format"),
        (Request(model="jev-latest", messages=(NOTE,), config=Config(response_format={"type": "json_schema", "name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}}}})), "config.response_format"),
        (Request(model="jev-latest", messages=(NOTE,), tools=(FunctionTool(name="f"),), config=Config(response_format=WINE)), "tools"),
    ]:
        with pytest.raises(UnsupportedFeatureError) as exc:
            lm.build_request(req, stream=False)
        assert exc.value.feature == feature
    with pytest.raises(UnsupportedFeatureError):
        lm.build_request(Request(model="jev-latest", messages=(NOTE,), config=Config(response_format=WINE)), stream=True)


def test_typesafe_errors() -> None:
    lm = TypeSafeLM(api_key="k")
    e = lm.normalize_error(401, json.dumps({"detail": {"error_type": "authentication_error", "message": "bad key"}}))
    assert isinstance(e, AuthError) and e.provider_code == "authentication_error"
    assert isinstance(lm.normalize_error(400, json.dumps({"detail": {"error_type": "api_usage_error", "message": "Unknown model: x"}})), UnsupportedModelError)
    e = lm.normalize_error(422, json.dumps({"detail": [{"type": "missing", "loc": ["body", "questions", "q", "choice", "criteria"], "msg": "Field required"}]}))
    assert isinstance(e, InvalidRequestError) and "questions.q.choice.criteria: Field required" in str(e)
    assert isinstance(lm._models_from_body(json.dumps({"models": [{"name": "jev-latest", "description": "d", "release_date": "2026"}]}))[0].id, str)


# ─── cloud wires: the pick, never a fabricated number ────────────────


def test_cloud_wires_answer_a_data_part_and_record_or_refuse_probabilities() -> None:
    req = Request(model="gpt-5-mini", messages=(NOTE,), config=Config(response_format=WINE, probabilities="if_available", max_tokens=50))
    chat = OpenAIChatLM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=json.dumps({
        "id": "c1", "model": "gpt-5-mini", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"quality": 2, "style": "oak", "ageing": true}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())]))
    r = chat.complete(req)
    assert r.data == {"quality": 2, "style": "oak", "ageing": True} and r.probabilities is None and r.method is None
    assert [(a.field, a.action, a.asked) for a in r.adaptations] == [("config.probabilities", "dropped", "if_available")]
    with pytest.raises(UnsupportedFeatureError) as exc:
        chat.plan(Request(model="gpt-5-mini", messages=(NOTE,), config=Config(response_format=WINE, probabilities="required")))
    assert exc.value.feature == "config.probabilities"
    # the rewrites reach the wire
    a = json.loads(AnthropicLM(api_key="k").build_request(Request(model="claude-haiku-4-5", messages=(NOTE,), config=Config(response_format=WINE, max_tokens=50)), stream=False).body)
    assert "type" not in a["output_config"]["format"]["schema"]["properties"]["quality"]
    g = json.loads(GeminiLM(api_key="k").build_request(Request(model="gemini-2.5-flash", messages=(NOTE,), config=Config(response_format=WINE)), stream=False).body)
    assert g["generationConfig"]["responseJsonSchema"]["properties"]["style"]["enum"] == ["fruit", "oak", "other"]
    o = json.loads(OpenAILM(api_key="k").build_request(Request(model="gpt-5-mini", messages=(NOTE,), config=Config(response_format=WINE)), stream=False).body)
    assert o["text"]["format"]["schema"] == WINE["schema"]  # verbatim


# ─── the token trie on a server that scores named tokens ─────────────

PREFIX = [1, 2, 3]  # the rendered prompt up to "Answer:"
PATHS = {"fruit": [10, 99], "oak": [11, 99], "other": [12, 99], "true": [20, 99], "false": [21, 99]}


def _tokenize_body(req_body: bytes) -> bytes:
    payload = json.loads(req_body)
    answer = payload["messages"][-1]["content"]
    key = answer.removeprefix("Answer:").strip()
    tokens = list(PREFIX) + (PATHS[key][:1] if key else [])
    if key and not payload["continue_final_message"]:
        tokens += PATHS[key][1:] + [7]  # terminator then a trailing newline
    return json.dumps({"tokens": tokens, "count": len(tokens), "max_model_len": 4096}).encode()


class TrieTransport(FakeTransport):
    """Answers /tokenize from PATHS and /completions with one scripted choice per prompt."""

    def __init__(self, logprobs: dict[int, float], *, drop_ids: bool = False) -> None:
        super().__init__([])
        self.logprobs, self.drop_ids, self.prompts = logprobs, drop_ids, None

    def stream(self, request):
        if request.url.endswith("/tokenize"):
            return FakeResponse(status=200, body=_tokenize_body(request.body))
        payload = json.loads(request.body)
        self.prompts = payload["prompt"]
        choices = []
        for i, prompt in enumerate(self.prompts):
            top = {} if self.drop_ids else {f"token_id:{t}": self.logprobs.get(t, -9.0) for t in payload["logprob_token_ids"]}
            choices.append({"index": i, "text": "x", "finish_reason": "length", "logprobs": {"top_logprobs": [top]}})
        return FakeResponse(status=200, body=json.dumps({"model": "m", "choices": choices, "usage": {"prompt_tokens": 40, "completion_tokens": len(choices)}}).encode())


def test_trie_driver_scores_every_key_path_in_one_batched_call() -> None:
    fmt = judgments(style=choice("Which style?", ["fruit", "oak", "other"]), ageing=yes_no("Will it age?"))
    transport = TrieTransport({10: -0.1, 11: -2.0, 12: -3.0, 20: -0.05, 21: -3.0, 99: -0.01})
    lm = OpenAIChatLM(api_key="k", base_url="http://vllm:8001/v1", compat="vllm", transport=transport)
    r = lm.complete(Request(model="m", messages=(NOTE,), config=Config(response_format=fmt, probabilities="required")))
    assert r.method == "candidate_sequence_likelihood" and r.data == {"style": "fruit", "ageing": True}
    assert r.probabilities["style"]["fruit"] == pytest.approx(0.830, abs=0.005)
    assert r.provider_data["judgments"] == {"nodes": 7, "tokenize_calls": 12, "method": "candidate_sequence_likelihood"}
    assert set(r.provider_data["coverage"]) == {"style", "ageing"}
    assert len(transport.prompts) == 7 and transport.prompts[0] == PREFIX  # root node = the prefill alone
    assert r.adaptations == ()  # MAP-14: requested measurement is not itself an adaptation.


def test_trie_driver_detects_a_server_that_drops_logprob_token_ids() -> None:
    fmt = judgments(ageing=yes_no("Will it age?"))
    dropped = TrieTransport({}, drop_ids=True)
    dropped.responses = [FakeResponse(status=200, body=json.dumps({"id": "c", "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"ageing": true}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())]
    original = dropped.stream

    def stream(request):
        if request.url.endswith("/chat/completions"):
            return dropped.responses.pop(0)
        return original(request)
    dropped.stream = stream
    lm = OpenAIChatLM(api_key="k", base_url="http://vllm:8000/v1", compat="vllm", transport=dropped)
    r = lm.complete(Request(model="m", messages=(NOTE,), config=Config(response_format=fmt, probabilities="if_available")))
    assert r.data == {"ageing": True} and r.probabilities is None
    assert ("config.probabilities", "dropped") in [(a.field, a.action) for a in r.adaptations]
    with pytest.raises(UnsupportedFeatureError):
        OpenAIChatLM(api_key="k", base_url="http://vllm:8000/v1", compat="vllm", transport=TrieTransport({}, drop_ids=True)).complete(
            Request(model="m", messages=(NOTE,), config=Config(response_format=fmt, probabilities="required")))


def test_router_rule_and_no_extra_calls_when_off() -> None:
    from lm15 import LMRouter
    assert LMRouter().resolve("jev-latest").provider == "typesafe"
    fmt = judgments(ageing=yes_no("Will it age?"))
    plain = OpenAIChatLM(api_key="k", base_url="http://vllm:8001/v1", compat="vllm", transport=FakeTransport([FakeResponse(status=200, body=json.dumps({
        "id": "c", "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"ageing": false}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())]))
    r = plain.complete(Request(model="m", messages=(NOTE,), config=Config(response_format=fmt)))
    assert r.data == {"ageing": False} and r.probabilities is None and r.adaptations == ()
