"""Adapters obey INV-029 and MAP-3 at the wire boundary.

Two implementation facts pinned here, neither of which is an oracle change:

1. Primary usage counters (input_tokens, output_tokens) are ``None`` when
   the provider did not report them.  The adapters used to write ``0``,
   which INV-029 forbids: absent is "unknown", never zero.  Gemini is the
   one stated exception: its proto3-JSON wire omits zero-valued fields, so
   inside a present ``usageMetadata`` an absent primary is a reported ``0``
   (pinned by the reviewed golden gemini.max_output_tokens).

2. A bare ``[DONE]`` terminator on the Responses dialect carries no finish
   reason.  It used to say ``stop``, which the MAP-3 coalescer then let
   overwrite the ``tool_call`` that ``response.completed`` had already
   established.
"""

from __future__ import annotations

import json

from lm15.providers import HttpResponse
from lm15.providers.anthropic import AnthropicLM
from lm15.providers.gemini import GeminiLM
from lm15.providers.openai import OpenAILM
from lm15.providers.openai_chat import OpenAIChatLM
from lm15.result import coalesce_stream
from lm15.sse import SSEEvent
from lm15.types import Message, Request

_REQ = Request(model="m-test", messages=(Message.user("Hi"),))


def _http(body: dict) -> HttpResponse:
    return HttpResponse(status=200, reason="OK", headers=[("content-type", "application/json")], body=json.dumps(body).encode("utf-8"))


def _openai_chat_body(usage):
    body = {
        "id": "chatcmpl-1",
        "model": "m-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


# ─── Primary counters: absent stays None ─────────────────────────────


def test_openai_chat_missing_counters_are_none() -> None:
    lm = OpenAIChatLM(api_key="sk-test")
    resp = lm.parse_response(_REQ, _http(_openai_chat_body({"prompt_tokens": 7})))
    assert resp.usage.input_tokens == 7
    assert resp.usage.output_tokens is None
    assert resp.usage.total_tokens is None  # never 7 + 0


def test_openai_chat_no_usage_block_is_all_none() -> None:
    lm = OpenAIChatLM(api_key="sk-test")
    resp = lm.parse_response(_REQ, _http(_openai_chat_body(None)))
    assert resp.usage.input_tokens is None
    assert resp.usage.output_tokens is None


def test_openai_chat_reported_zero_stays_zero() -> None:
    lm = OpenAIChatLM(api_key="sk-test")
    resp = lm.parse_response(_REQ, _http(_openai_chat_body({"prompt_tokens": 7, "completion_tokens": 0})))
    assert resp.usage.output_tokens == 0
    assert resp.usage.total_tokens == 7


def test_openai_responses_missing_counters_are_none() -> None:
    lm = OpenAILM(api_key="sk-test")
    body = {
        "id": "resp_1",
        "model": "m-test",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"output_tokens": 3},
    }
    resp = lm.parse_response(_REQ, _http(body))
    assert resp.usage.input_tokens is None
    assert resp.usage.output_tokens == 3
    assert resp.usage.total_tokens is None


def test_anthropic_missing_usage_is_none_not_zero() -> None:
    lm = AnthropicLM(api_key="sk-test")
    body = {
        "id": "msg_1",
        "model": "m-test",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12},
    }
    resp = lm.parse_response(_REQ, _http(body))
    # MAP-19 rule 5: the wire's input_tokens excludes the cache; without
    # both cache counters canonical input is unknown, never 12.
    assert resp.usage.input_tokens is None
    assert resp.usage.output_tokens is None
    assert resp.usage.total_tokens is None


def test_anthropic_input_includes_both_cache_counters() -> None:
    lm = AnthropicLM(api_key="sk-test")
    body = {
        "id": "msg_1",
        "model": "m-test",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "cache_creation_input_tokens": 24, "cache_read_input_tokens": 3275,
                  "output_tokens": 6},
    }
    usage = lm.parse_response(_REQ, _http(body)).usage
    assert (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (3302, 3275, 24)
    assert usage.total_tokens == 3308


def test_anthropic_stream_message_delta_partial_usage_is_none() -> None:
    lm = AnthropicLM(api_key="sk-test")
    payload = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}
    events = list(lm.parse_stream_events(_REQ, SSEEvent(event="message_delta", data=json.dumps(payload))))
    ends = [e for e in events if e.type == "end"]
    assert len(ends) == 1
    assert ends[0].usage.output_tokens == 5
    assert ends[0].usage.input_tokens is None


# ─── Gemini: proto3 omission inside a present usageMetadata is 0 ────


def _gemini_body(usage_metadata):
    body = {
        "responseId": "r1",
        "modelVersion": "m-test",
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    }
    if usage_metadata is not None:
        body["usageMetadata"] = usage_metadata
    return body


def test_gemini_absent_primary_inside_usage_metadata_is_zero() -> None:
    lm = GeminiLM(api_key="k")
    body = _gemini_body({"promptTokenCount": 14, "totalTokenCount": 18, "thoughtsTokenCount": 4})
    resp = lm.parse_response(_REQ, _http(body))
    assert resp.usage.input_tokens == 14
    assert resp.usage.output_tokens == 4  # MAP-19 rule 6: answer (absent: 0) + thoughts
    assert resp.usage.total_tokens == 18
    assert resp.usage.reasoning_tokens == 4
    assert resp.usage.cache_read_tokens == 0  # proto3: absent inside usageMetadata is 0
    assert resp.usage.cache_write_tokens is None  # Gemini reports no cache-write count


def test_gemini_no_usage_metadata_is_all_none() -> None:
    lm = GeminiLM(api_key="k")
    resp = lm.parse_response(_REQ, _http(_gemini_body(None)))
    assert resp.usage.input_tokens is None
    assert resp.usage.output_tokens is None
    assert resp.usage.total_tokens is None


# ─── [DONE] on the Responses dialect carries no finish reason ─────────


def test_openai_responses_done_frame_has_no_finish_reason() -> None:
    lm = OpenAILM(api_key="sk-test")
    events = list(lm.parse_stream_events(_REQ, SSEEvent(event=None, data="[DONE]")))
    assert [e.type for e in events] == ["end"]
    assert events[0].finish_reason is None
    assert events[0].usage is None


def test_openai_responses_done_does_not_overwrite_tool_call() -> None:
    lm = OpenAILM(api_key="sk-test")
    completed = {
        "type": "response.completed",
        "response": {
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "output": [{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}],
        },
    }
    raw = [
        SSEEvent(event=None, data=json.dumps({"type": "response.created", "response": {"id": "r", "model": "m-test"}})),
        SSEEvent(event=None, data=json.dumps(completed)),
        SSEEvent(event=None, data="[DONE]"),
    ]
    events = list(coalesce_stream(e for r in raw for e in lm.parse_stream_events(_REQ, r)))
    ends = [e for e in events if e.type == "end"]
    assert len(ends) == 1
    assert ends[0].finish_reason == "tool_call"
    assert ends[0].usage.total_tokens == 3


def test_gemini_modality_breakdowns_fill_the_audio_slots() -> None:
    lm = GeminiLM(api_key="k")
    body = _gemini_body({
        "promptTokenCount": 153, "candidatesTokenCount": 59, "totalTokenCount": 212,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 130}, {"modality": "AUDIO", "tokenCount": 23}],
        "candidatesTokensDetails": [{"modality": "AUDIO", "tokenCount": 59}],
    })
    usage = lm.parse_response(_REQ, _http(body)).usage
    assert usage.input_audio_tokens == 23 and usage.output_audio_tokens == 59
    # No AUDIO entry: not reported, never 0.
    text_only = lm.parse_response(_REQ, _http(_gemini_body({"promptTokenCount": 1, "candidatesTokenCount": 1, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1}]}))).usage
    assert text_only.input_audio_tokens is None and text_only.output_audio_tokens is None
