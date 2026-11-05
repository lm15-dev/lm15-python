"""Rules ratified 2026-09-06 (lm15-contract lm15-contract/changes/2026-09-06-decisions.md).

D7  — a continuation's ``provider`` is the dialect, never the door.
D8  — no message-level id continuation; INV-051 stream/complete parity.
D9  — ``StreamEndEvent.provider_data`` is the usage frame, else the
      finish_reason frame; bare terminators contribute nothing.
D10 — thinking comes only from a typed wire field; tags stay literal text.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from lm15 import serde, vet
from lm15.providers import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM
from lm15.providers.base import HttpResponse
from lm15.result import coalesce_stream, materialize_response
from lm15.sse import parse_sse
from lm15.types import (
    ContinuationState,
    Message,
    Request,
    StreamDeltaEvent,
    StreamEndEvent,
    TextPart,
    ThinkingPart,
    Usage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT.parent / "lm15-contract"
requires_contract = pytest.mark.skipif(
    not CONTRACT_ROOT.exists(), reason="lm15-contract corpus not checked out"
)

_REQ = Request(model="m-test", messages=(Message.user("Hi"),))


def _http(body: dict | str, status: int = 200) -> HttpResponse:
    raw = body if isinstance(body, str) else json.dumps(body)
    return HttpResponse(status=status, reason="OK", headers=[("content-type", "application/json")], body=raw.encode())


def _stream(lm, request: Request, body: bytes):
    def raw_events():
        for raw in parse_sse(iter(body.splitlines(keepends=True))):
            yield from (e for e in lm.parse_stream_events(request, raw) if e is not None)

    return list(coalesce_stream(raw_events(), model=request.model))


def _sse(*frames: dict | str) -> bytes:
    out = []
    for frame in frames:
        if isinstance(frame, str):
            out.append(f"data: {frame}\n\n")
        else:
            out.append(f"data: {json.dumps(frame)}\n\n")
    return "".join(out).encode()


# ─── D7: the continuation namespace is the dialect ───────────────────

_REASONING_BODY = {
    "id": "resp_1",
    "model": "muse-spark",
    "status": "completed",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA"},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "hi", "annotations": []}]},
    ],
    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
}


@pytest.mark.parametrize("provider", ["meta", "azure", "moonshotai-responses"])
def test_reasoning_item_state_names_the_openai_dialect_on_every_door(provider: str) -> None:
    # Only the cloud door takes host settings; a door that reads none now
    # refuses a settings entry instead of dropping it (AUTH-10, 2026-09-30).
    lm = vet.adapter_for_provider(provider, "k", None, settings={"resource": "r"} if provider == "azure" else None)
    assert isinstance(lm, OpenAILM)
    response = lm.parse_response(_REQ, _http(_REASONING_BODY))
    thinking = response.message.parts[0]
    assert isinstance(thinking, ThinkingPart)
    assert thinking.continuation == (
        ContinuationState(provider="openai", kind="reasoning_item", data={"id": "rs_1", "encrypted_content": "gAAAA"}),
    )
    assert lm.provider == provider  # the door is still named on errors and models


@pytest.mark.parametrize("provider", ["meta-anthropic", "deepseek-anthropic", "moonshotai-anthropic"])
def test_anthropic_state_names_the_anthropic_dialect_on_every_door(provider: str) -> None:
    lm = vet.adapter_for_provider(provider, "k", None)
    assert isinstance(lm, AnthropicLM)
    body = {
        "id": "msg_1", "model": "m", "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "t", "signature": "sig"},
            {"type": "redacted_thinking", "data": "blob"},
            {"type": "text", "text": "hi"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    response = lm.parse_response(_REQ, _http(body))
    providers = {state.provider for part in response.message.parts for state in part.continuation}
    assert providers == {"anthropic"}


# ─── D8: no message-level id continuation ────────────────────────────

def test_no_dialect_mints_message_level_id_state() -> None:
    openai_resp = OpenAILM(api_key="k").parse_response(_REQ, _http(_REASONING_BODY))
    assert openai_resp.id == "resp_1" and openai_resp.message.continuation == ()

    anthropic_body = {"id": "msg_9", "model": "m", "stop_reason": "end_turn",
                      "content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
    anthropic_resp = AnthropicLM(api_key="k").parse_response(_REQ, _http(anthropic_body))
    assert anthropic_resp.id == "msg_9" and anthropic_resp.message.continuation == ()

    gemini_body = {"candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                   "responseId": "gem_1"}
    gemini_resp = GeminiLM(api_key="k").parse_response(_REQ, _http(gemini_body))
    assert gemini_resp.id == "gem_1" and gemini_resp.message.continuation == ()


def test_stream_starts_carry_the_id_without_a_continuation_delta() -> None:
    anthropic = AnthropicLM(api_key="k")
    events = _stream(anthropic, _REQ, _sse(
        {"type": "message_start", "message": {"id": "msg_1", "model": "m"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ))
    assert events[0].type == "start" and events[0].id == "msg_1"
    assert not any(e.type == "delta" and e.delta.type == "continuation" for e in events)

    gemini = GeminiLM(api_key="k")
    events = _stream(gemini, _REQ, _sse(
        {"candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}, "responseId": "gem_1"},
    ))
    assert not any(e.type == "delta" and e.delta.type == "continuation" for e in events)


# ─── INV-051: stream/complete parity over the contract stream bodies ──

def _load_stream_cases() -> list[dict]:
    sys.path.insert(0, str(CONTRACT_ROOT / "harness"))
    import check  # noqa: WPS433 — the harness's own loaders

    cases = []
    for case in check.load_wire_cases():
        if "pinned_body" not in case or "canonical_request" not in case:
            continue
        if check.expected_raise(case, "build_request") is not None:
            continue
        if check.expected_raise(case, "replay_stream") is not None:
            continue  # the pinned refusal (MAP-9) assembles no Response
        if not check.is_stream_case(case):
            continue
        cases.append(case)
    return cases


def _dialect(provider: str) -> str:
    from lm15.registry import lookup

    definition = lookup(provider)
    assert definition is not None, provider
    return definition.dialect


def _frames(body: bytes) -> list[dict]:
    out = []
    for raw in parse_sse(iter(body.splitlines(keepends=True))):
        if raw.data and raw.data != "[DONE]":
            payload = json.loads(raw.data)
            if isinstance(payload, dict):
                out.append(payload)
    return out


def _complete_body_from_stream(dialect: str, frames: list[dict]) -> dict:
    """Fold a stream's frames into the body the complete call returns.

    The fold is the wire's own: a Responses stream ends with the complete
    response object; an Anthropic stream is a message skeleton plus blocks;
    a chat stream is one message assembled from deltas; a Gemini stream is
    the last chunk with the text parts concatenated.
    """
    if dialect == "openai-responses":
        # A turn cut short ends with response.incomplete (MAP-20 change,
        # 2026-11-05): the same final copy of the whole response.
        completed = [f for f in frames if f.get("type") in ("response.completed", "response.incomplete")]
        assert completed, "no response.completed / response.incomplete frame"
        return completed[-1]["response"]

    if dialect == "anthropic":
        message = None
        blocks: dict[int, dict] = {}
        for frame in frames:
            et = frame.get("type")
            if et == "message_start":
                message = dict(frame["message"])
            elif et == "content_block_start":
                blocks[int(frame["index"])] = dict(frame["content_block"])
                block = blocks[int(frame["index"])]
                if block.get("type") == "tool_use":
                    block["_json"] = ""
            elif et == "content_block_delta":
                block = blocks[int(frame["index"])]
                delta = frame["delta"]
                if delta["type"] == "text_delta":
                    block["text"] = block.get("text", "") + delta["text"]
                elif delta["type"] == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta["thinking"]
                elif delta["type"] == "signature_delta":
                    block["signature"] = delta["signature"]
                elif delta["type"] == "input_json_delta":
                    block["_json"] += delta["partial_json"]
            elif et == "message_delta":
                assert message is not None
                message.update(frame.get("delta") or {})
                usage = dict(message.get("usage") or {})
                usage.update(frame.get("usage") or {})
                message["usage"] = usage
        assert message is not None
        content = []
        for index in sorted(blocks):
            block = blocks[index]
            if "_json" in block:
                raw = block.pop("_json")
                block["input"] = json.loads(raw) if raw else {}
            content.append(block)
        message["content"] = content
        return message

    if dialect == "openai-chat":
        head = dict(frames[0])
        message: dict = {"role": "assistant", "content": None}
        tool_calls: dict[int, dict] = {}
        finish = None
        usage = None
        for frame in frames:
            if isinstance(frame.get("usage"), dict):
                usage = frame["usage"]
            choices = frame.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning"):
                if delta.get(key):
                    message[key] = (message.get(key) or "") + delta[key]
            for call in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(int(call.get("index", 0)), {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        body = {k: v for k, v in head.items() if k not in ("choices", "usage")}
        body["object"] = "chat.completion"
        body["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
        if usage is not None:
            body["usage"] = usage
        return body

    if dialect == "gemini":
        last = dict(frames[-1])
        text_parts: list[dict] = []
        other: list[dict] = []
        for frame in frames:
            for cand in frame.get("candidates") or []:
                for part in (cand.get("content") or {}).get("parts") or []:
                    if "text" in part and not part.get("thought"):
                        if text_parts and "thoughtSignature" not in part and "thoughtSignature" not in text_parts[-1]:
                            text_parts[-1] = {**text_parts[-1], "text": text_parts[-1]["text"] + part["text"]}
                        else:
                            text_parts.append(dict(part))
                    else:
                        other.append(part)
        candidate = dict((last.get("candidates") or [{}])[0])
        candidate["content"] = {"role": "model", "parts": text_parts + other}
        last["candidates"] = [candidate]
        return last

    raise AssertionError(f"unknown dialect {dialect}")


# Fields the wire withholds on one path (MAP-9.6).  The chat dialect and
# Gemini SSE have no start frame (MAP-4): the synthesized start carries the
# request's model and no id, so the per-chunk id (and, on chat, the served
# model snapshot) is not lifted into the Response (MAP-9 rule 6).
_WITHHELD: dict[str, tuple[str, ...]] = {
    "openai-chat": ("id", "model"),
    "gemini": ("id",),
}

_KNOWN_GAPS: dict[str, str] = {}


def _parity_params() -> list:
    params = []
    for case in (_load_stream_cases() if CONTRACT_ROOT.exists() else []):
        marks = ()
        if case["id"] in _KNOWN_GAPS:
            marks = (pytest.mark.xfail(strict=True, reason=_KNOWN_GAPS[case["id"]]),)
        params.append(pytest.param(case, id=case["id"], marks=marks))
    return params


@requires_contract
@pytest.mark.parametrize("case", _parity_params())
def test_inv051_stream_materializes_to_the_complete_parse(case: dict) -> None:
    sys.path.insert(0, str(CONTRACT_ROOT / "harness"))
    import check

    dialect = _dialect(case["provider"])
    body = check.pinned_body(case)
    msg = {
        "provider": case["provider"],
        "canonical_request": case["canonical_request"],
        **check.case_base_url(case),
        **check.host_fields(case),
    }
    streamed = vet.op_replay_stream({**msg, "body_b64": base64.b64encode(body).decode()})
    complete_body = _complete_body_from_stream(dialect, _frames(body))
    complete = vet.op_parse_response({
        **msg, "status": 200,
        "body_b64": base64.b64encode(json.dumps(complete_body).encode()).decode(),
    })
    assert not complete.get("unmapped"), complete.get("unmapped")

    expected = dict(complete["canonical_response"])
    actual = dict(streamed["canonical_response"])
    for key in _WITHHELD.get(dialect, ()):
        expected.pop(key, None)
        actual.pop(key, None)
    assert actual == expected


# ─── D9: end provider_data ───────────────────────────────────────────

def test_chat_end_provider_data_is_the_usage_chunk() -> None:
    lm = OpenAIChatLM(api_key="k")
    finish_chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    usage_chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    events = _stream(lm, _REQ, _sse(
        {"id": "c1", "object": "chat.completion.chunk", "model": "m",
         "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        finish_chunk, usage_chunk, "[DONE]",
    ))
    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.finish_reason == "stop" and end.usage == Usage(input_tokens=1, output_tokens=2, total_tokens=3)
    assert end.provider_data == usage_chunk


def test_chat_end_provider_data_falls_back_to_the_finish_chunk() -> None:
    lm = OpenAIChatLM(api_key="k")
    finish_chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    events = _stream(lm, _REQ, _sse(
        {"id": "c1", "object": "chat.completion.chunk", "model": "m",
         "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        finish_chunk, "[DONE]",
    ))
    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.usage is None and end.provider_data == finish_chunk


def test_anthropic_end_provider_data_is_the_message_delta_frame() -> None:
    lm = AnthropicLM(api_key="k")
    message_delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                     "usage": {"input_tokens": 3, "output_tokens": 4}}
    events = _stream(lm, _REQ, _sse(
        {"type": "message_start", "message": {"id": "msg_1", "model": "m", "usage": {"input_tokens": 3, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        message_delta,
        {"type": "message_stop"},
    ))
    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.provider_data == message_delta


def test_coalescer_keeps_the_usage_frame_over_a_later_finish_frame() -> None:
    usage_frame = {"frame": "usage"}
    finish_frame = {"frame": "finish"}
    merged = list(coalesce_stream(iter((
        StreamEndEvent(usage=Usage(input_tokens=1, output_tokens=1), provider_data=usage_frame),
        StreamEndEvent(finish_reason="stop", provider_data=finish_frame),
        StreamEndEvent(),
    )), model="m"))
    end = merged[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.finish_reason == "stop" and end.provider_data == usage_frame

    merged = list(coalesce_stream(iter((
        StreamEndEvent(finish_reason="stop", provider_data=finish_frame),
        StreamEndEvent(usage=Usage(input_tokens=1, output_tokens=1), provider_data=usage_frame),
        StreamEndEvent(),
    )), model="m"))
    assert merged[-1].provider_data == usage_frame


@requires_contract
@pytest.mark.parametrize("case", _load_stream_cases() if CONTRACT_ROOT.exists() else [], ids=lambda c: c["id"])
def test_every_contract_stream_end_carries_a_frame_as_provider_data(case: dict) -> None:
    sys.path.insert(0, str(CONTRACT_ROOT / "harness"))
    import check

    body = check.pinned_body(case)
    reply = vet.op_replay_stream({
        "provider": case["provider"],
        "canonical_request": case["canonical_request"],
        "body_b64": base64.b64encode(body).decode(),
        **check.case_base_url(case),
        **check.host_fields(case),
    })
    end = reply["events"][-1]
    assert end["type"] == "end"
    assert isinstance(end.get("provider_data"), dict) and end["provider_data"]
    dialect = _dialect(case["provider"])
    frames = _frames(body)
    if dialect == "openai-responses":
        candidates = [f["response"] for f in frames if f.get("type") in ("response.completed", "response.incomplete")]
    else:
        candidates = frames
    assert end["provider_data"] in candidates, "end provider_data is not a verbatim wire frame"


# ─── D10: delimiters in provider text stay literal ───────────────────

_TAGGED = "<think>secret plan</think>\n<reasoning>more</reasoning>The answer is 4."


def test_chat_text_with_think_tags_stays_literal() -> None:
    lm = OpenAIChatLM(api_key="k")
    body = {"id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": _TAGGED}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    response = lm.parse_response(_REQ, _http(body))
    assert response.message.parts == (TextPart(text=_TAGGED),)

    chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "m",
             "choices": [{"index": 0, "delta": {"content": _TAGGED}, "finish_reason": "stop"}]}
    streamed = materialize_response(iter(_stream(lm, _REQ, _sse(chunk, "[DONE]"))), _REQ)
    assert streamed.message.parts == (TextPart(text=_TAGGED),)


def test_responses_and_anthropic_text_with_think_tags_stays_literal() -> None:
    openai_body = {"id": "resp_1", "model": "m", "status": "completed",
                   "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                               "content": [{"type": "output_text", "text": _TAGGED, "annotations": []}]}],
                   "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
    assert OpenAILM(api_key="k").parse_response(_REQ, _http(openai_body)).message.parts == (TextPart(text=_TAGGED),)

    anthropic_body = {"id": "msg_1", "model": "m", "stop_reason": "end_turn",
                      "content": [{"type": "text", "text": _TAGGED}], "usage": {"input_tokens": 1, "output_tokens": 1}}
    assert AnthropicLM(api_key="k").parse_response(_REQ, _http(anthropic_body)).message.parts == (TextPart(text=_TAGGED),)

    gemini_body = {"candidates": [{"content": {"parts": [{"text": _TAGGED}], "role": "model"}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}
    assert GeminiLM(api_key="k").parse_response(_REQ, _http(gemini_body)).message.parts == (TextPart(text=_TAGGED),)


def test_serialized_thinking_part_has_no_redacted_key() -> None:
    part = ThinkingPart(text="", continuation=(ContinuationState(provider="anthropic", kind="redacted_thinking", data={"data": "b"}),))
    d = serde.part_to_dict(part)
    assert "redacted" not in d and d["text"] == ""
    assert serde.part_from_dict(d) == part
    with pytest.raises(TypeError):
        ThinkingPart(text="x", redacted=True)  # type: ignore[call-arg]
