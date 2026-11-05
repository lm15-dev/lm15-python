"""MAP-20: a stream's final copy completes its fragments and never contradicts them.

The contract harness checks the SSE readers over every pinned stream and its
variants (lm15-contract PROTOCOL.md § Stream checks). These tests pin what
the harness cannot reach: the per-event mapping stays per event, the
realtime completion path (WebSocket frames, no SSE), and the refusal.
"""
from __future__ import annotations

import json

import pytest

from lm15 import FunctionTool, Message, Request
from lm15.errors import StreamAssemblyError
from lm15.providers.gemini import GeminiLM
from lm15.providers.openai import OpenAILM, _FinalCopies
from lm15.result import StreamAccumulator, coalesce_stream, materialize_response
from lm15.sse import SSEEvent
from lm15.types import ToolCallPart

REQUEST = Request(model="m", messages=(Message.user("weather in Paris and Oslo"),),
                  tools=(FunctionTool(name="get_weather", parameters={"type": "object"}),))


def _sse(payloads):
    return [SSEEvent(event=p.get("type"), data=json.dumps(p)) for p in payloads]


def _read(lm, payloads):
    reader = lm.stream_reader(REQUEST)
    events = [e for raw in _sse(payloads) for e in reader.feed(raw)]
    return materialize_response(coalesce_stream(iter(events), model="m"), REQUEST)


def _calls(response):
    return [(p.name, p.input) for p in response.message.parts if isinstance(p, ToolCallPart)]


# The shape the Codex backend sent on 2026-11-05 for two parallel calls:
# no arguments fragment, the arguments only in the two done frames.
CODEX_PARALLEL = [
    {"type": "response.created", "response": {"id": "r", "model": "m"}},
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "call_id": "a", "name": "get_weather", "arguments": ""}},
    {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"city": "Paris"}'},
    {"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "call_id": "a", "name": "get_weather", "arguments": '{"city": "Paris"}'}},
    {"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "call_id": "b", "name": "get_weather", "arguments": ""}},
    {"type": "response.function_call_arguments.done", "output_index": 1, "arguments": '{"city": "Oslo"}'},
    {"type": "response.output_item.done", "output_index": 1, "item": {"type": "function_call", "call_id": "b", "name": "get_weather", "arguments": '{"city": "Oslo"}'}},
    {"type": "response.completed", "response": {"id": "r", "status": "completed", "output": [{"type": "function_call"}, {"type": "function_call"}]}},
]


def test_arguments_only_in_the_final_copy_reach_the_response():
    assert _calls(_read(OpenAILM(api_key="k"), CODEX_PARALLEL)) == [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Oslo"})]


def test_fragments_then_final_copy_count_once():
    payloads = CODEX_PARALLEL[:2] + [
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city"'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ': "Paris"}'},
    ] + CODEX_PARALLEL[2:]
    assert _calls(_read(OpenAILM(api_key="k"), payloads))[0] == ("get_weather", {"city": "Paris"})


def test_a_fragment_the_final_copy_contradicts_is_refused():
    payloads = CODEX_PARALLEL[:2] + [
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city": "Lima"}'},
    ] + CODEX_PARALLEL[2:]
    with pytest.raises(StreamAssemblyError) as caught:
        _read(OpenAILM(api_key="k"), payloads)
    assert caught.value.part_index == 0 and caught.value.code == "stream_assembly"


def test_the_same_json_value_in_other_text_is_not_a_contradiction():
    payloads = CODEX_PARALLEL[:2] + [
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city":"Paris"}'},
    ] + CODEX_PARALLEL[2:]
    assert _calls(_read(OpenAILM(api_key="k"), payloads))[0] == ("get_weather", {"city": "Paris"})


def test_members_in_another_order_are_a_contradiction():
    copies = _FinalCopies()
    copies.saw(0, "arguments", '{"b": 1, "a": 2}')
    with pytest.raises(StreamAssemblyError):
        copies.remainder(0, "arguments", '{"a": 2, "b": 1}', json_value=True)


def test_summary_parts_join_with_a_newline_as_on_the_complete_path():
    payloads = [
        {"type": "response.created", "response": {"id": "r", "model": "m"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "**One**"},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 2, "delta": "**Three**"},
        {"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "id": "rs",
                                                                         "summary": [{"text": "**One**"}, {"text": ""}, {"text": "**Three**"}]}},
        {"type": "response.completed", "response": {"id": "r", "status": "completed", "output": []}},
    ]
    assert _read(OpenAILM(api_key="k"), payloads).message.parts[0].text == "**One**\n\n**Three**"


def test_an_incomplete_stream_ends_with_length():
    payloads = [
        {"type": "response.created", "response": {"id": "r", "model": "m"}},
        {"type": "response.incomplete", "response": {"id": "r", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                                                     "output": [], "usage": {"input_tokens": 3, "output_tokens": 40}}},
    ]
    response = _read(OpenAILM(api_key="k"), payloads)
    assert response.finish_reason == "length" and response.usage.output_tokens == 40


def test_the_per_event_mapping_stays_per_event():
    # parse_stream_events maps one event on its own; only the reader completes.
    lm = OpenAILM(api_key="k")
    events = [e for raw in _sse(CODEX_PARALLEL) for e in lm.parse_stream_events(REQUEST, raw)]
    assert all(e.delta.input == "" for e in events if e.type == "delta")


def test_gemini_calls_in_chunks_of_their_own_get_slots_of_their_own():
    chunks = [
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}], "role": "model"}}]},
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}}}], "role": "model"},
                         "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}},
    ]
    assert _calls(_read(GeminiLM(api_key="k"), chunks)) == [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Oslo"})]


def test_gemini_text_continues_across_chunks_and_a_new_kind_opens_a_slot():
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "plan", "thought": True}], "role": "model"}}]},
        {"candidates": [{"content": {"parts": [{"text": "Hel"}], "role": "model"}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}], "role": "model"}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}},
    ]
    parts = _read(GeminiLM(api_key="k"), chunks).message.parts
    assert [(p.type, p.text) for p in parts] == [("thinking", "plan"), ("text", "Hello")]


# The realtime completion path (a realtime model called with complete/stream):
# frames as the GA server sends them for one call (receipts/2026-09-01-live).
REALTIME_CALL = [
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "call_id": "c", "name": "get_weather", "arguments": ""}},
    {"type": "response.function_call_arguments.delta", "output_index": 0, "call_id": "c", "delta": '{"city": '},
    {"type": "response.function_call_arguments.delta", "output_index": 0, "call_id": "c", "delta": '"Montreal"}'},
    {"type": "response.function_call_arguments.done", "output_index": 0, "call_id": "c", "arguments": '{"city": "Montreal"}'},
    {"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "call_id": "c", "name": "get_weather", "arguments": '{"city": "Montreal"}'}},
]


def test_realtime_completion_counts_the_arguments_once():
    lm = OpenAILM(api_key="k")
    copies = _FinalCopies()
    acc = StreamAccumulator(REQUEST)
    for frame in REALTIME_CALL:
        for event in lm._decode_live_completion_stream_events(REQUEST, json.dumps(frame), copies):
            acc.push(event)
    assert [(p.name, p.input) for p in acc.response().message.parts if isinstance(p, ToolCallPart)] == [("get_weather", {"city": "Montreal"})]


def test_realtime_completion_never_names_a_call_tool():
    lm = OpenAILM(api_key="k")
    events = lm._decode_live_completion_stream_events(REQUEST, json.dumps(REALTIME_CALL[1]), _FinalCopies())
    assert events[0].delta.name is None
