"""response_from_openai_chat — a Chat Completions response body → Response.

The reader is parse_response's own (the contract's response direction pins
what it maps and what it records as unmapped); this file pins what that
direction cannot: the module-level door on foreign dicts (litellm's
ModelResponse.model_dump(), captured live), the multi-choice rule, the
model fallback, the error envelope, and that parse_response and the door
agree byte for byte on every recorded chat body.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

import lm15
from conformance.sources import contract_root
from lm15 import serde
from lm15.errors import RateLimitError, UnsupportedFeatureError
from lm15.providers import HttpResponse, OpenAIChatLM
from lm15.providers.openai_chat import response_from_openai_chat
from lm15.types import Message, TextPart, ToolCallPart
from lm15.vet import adapter_for_provider

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "litellm_responses.json").read_text())


def test_exported_and_on_both_adapters() -> None:
    assert lm15.response_from_openai_chat is response_from_openai_chat
    body = FIXTURES["gemini_text"]
    a = OpenAIChatLM(api_key="k").response_from_openai_chat(body)
    b = lm15.AsyncOpenAIChatLM(api_key="k").response_from_openai_chat(body)
    assert a == b == response_from_openai_chat(body)


def test_litellm_gemini_dict_reads_as_a_response() -> None:
    r = response_from_openai_chat(FIXTURES["gemini_text"])
    assert r.text == "ok" and r.finish_reason == "stop" and r.model == "gemini-3.8-flash"
    assert r.id == "dUqhaqT7Hcmx_PUP6oqJiQI"
    assert r.message == Message.assistant(TextPart("ok"))
    # usage: litellm sums reasoning into completion_tokens; the reader keeps the wire's fields
    assert r.usage.output_tokens == 93 and r.usage.reasoning_tokens == 92 and r.usage.input_tokens == 8
    # nothing lost: litellm's extra keys ride in provider_data, and nothing was unmapped
    assert r.provider_data["vertex_ai_safety_results"] == [] and "_lm15_unmapped" not in r.provider_data


def test_multi_choice_is_refused_unless_named_then_read_per_choice() -> None:
    body = FIXTURES["openai_tool_calls_n2"]
    with pytest.raises(UnsupportedFeatureError) as exc:
        response_from_openai_chat(body)
    assert "2 choices" in str(exc.value)
    first = response_from_openai_chat(body, choice=0)
    second = response_from_openai_chat(body, choice=1)
    assert first.finish_reason == second.finish_reason == "tool_call"
    assert first.tool_calls[0] == ToolCallPart(id="call_oElL0LzkB8xwrI56Z4cpEiXb", name="get_weather", input={"city": "Paris"})
    assert second.tool_calls[0].id == "call_8AtaRZOFJSXZgQCPCjc7IuUO"
    assert first.usage == second.usage  # the wire reports usage once, for all choices
    with pytest.raises(ValueError):
        response_from_openai_chat(body, choice=2)


def test_parse_response_refuses_multi_choice_too() -> None:
    """A user who put n in config.extensions used to get choices[0] silently."""
    body = json.dumps(FIXTURES["openai_tool_calls_n2"]).encode()
    lm = OpenAIChatLM(api_key="k")
    req = lm15.Request(model="gpt-4o-mini", messages=(Message.user("hi"),))
    with pytest.raises(UnsupportedFeatureError):
        lm.parse_response(req, HttpResponse(status=200, reason="OK", headers={}, body=body))


def test_model_fallback_and_error_envelope() -> None:
    bare = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    with pytest.raises(ValueError):
        response_from_openai_chat(bare)
    assert response_from_openai_chat(bare, model="m").model == "m"
    with pytest.raises(RateLimitError) as exc:
        response_from_openai_chat({"error": {"code": "rate_limit_exceeded", "message": "slow down"}})
    assert exc.value.provider == "openai-chat" and exc.value.provider_code == "rate_limit_exceeded"
    with pytest.raises(TypeError):
        response_from_openai_chat([])  # type: ignore[arg-type]


def _recorded_chat_bodies():
    root = contract_root()
    for path in sorted((root / "cases").glob("*/*.json")):
        case = json.loads(path.read_text())
        req = case.get("request") or {}
        if not str(req.get("url", "")).split("?", 1)[0].endswith("/chat/completions"):
            continue
        if case.get("stream") or "pinned_body" not in case or "canonical_request" not in case:
            continue
        body = (root / "bodies" / case["id"] / case["pinned_body"]).read_bytes()
        try:
            data = json.loads(body)
        except ValueError:
            continue  # an SSE capture flagged by sniffing
        if not isinstance(data, dict) or "choices" not in data:
            continue
        yield case, data


@pytest.mark.parametrize("case, data", list(_recorded_chat_bodies()), ids=lambda x: x["id"] if isinstance(x, dict) and "id" in x else "")
def test_door_and_parse_response_agree_on_every_recorded_chat_body(case, data) -> None:
    lm = adapter_for_provider(case["provider"], "k", case.get("base_url"), settings=case.get("settings"))
    request = serde.request_from_dict(case["canonical_request"])
    raises = (case.get("expect_lm15") or {}).get("raises")
    if raises and raises.get("op") == "parse_response":
        # A pinned refusal (MAP-9 on the complete path): the door refuses the same way.
        with pytest.raises(Exception) as via_adapter_exc:
            lm.parse_response(request, HttpResponse(status=200, reason="OK", headers={}, body=json.dumps(data).encode()))
        with pytest.raises(type(via_adapter_exc.value)):
            lm.response_from_openai_chat(data, model=request.model)
        return
    via_adapter = lm.parse_response(request, HttpResponse(status=200, reason="OK", headers={}, body=json.dumps(data).encode()))
    via_door = lm.response_from_openai_chat(data, model=request.model)
    assert serde.response_to_dict(via_door, include_provider_data=True) == serde.response_to_dict(via_adapter, include_provider_data=True)
