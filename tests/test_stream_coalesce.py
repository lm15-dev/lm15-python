"""MAP-3 — a stream yields exactly one StreamEndEvent, as the final event.

Providers split terminal data across frames (finish_reason chunk, usage-only
chunk, [DONE] / message_delta + message_stop).  The coalescer owns the merge;
adapters stay stateless.  See docs/mapping-rules.md MAP-3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from lm15.providers.anthropic import AnthropicLM
from lm15.providers.openai_chat import OpenAIChatLM
from lm15.result import Result
from lm15.types import Message, Request


@dataclass
class _FakeStreamResponse:
    status: int
    body: bytes
    headers: list[tuple[str, str]] | None = None
    reason: str = "OK"
    http_version: str = "HTTP/1.1"

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = [("content-type", "text/event-stream")]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield self.body

    def iter_lines(self) -> Iterator[bytes]:
        yield from self.body.splitlines(keepends=True)

    def read(self) -> bytes:
        return self.body


class _FakeTransport:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


_REQ = Request(model="m-test", messages=(Message.user("Hi"),))


def _chat_chunk(**kwargs) -> str:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "m-test"}
    base.update(kwargs)
    return "data: " + json.dumps(base) + "\n\n"


# vLLM/SGLang/ollama shape: finish_reason chunk, then a usage-only chunk
# (stream_options.include_usage), then [DONE].
_VLLM_SSE = (
    _chat_chunk(choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}])
    + _chat_chunk(choices=[{"index": 0, "delta": {"content": "!"}, "finish_reason": None}])
    + _chat_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    + _chat_chunk(
        choices=[],
        usage={"prompt_tokens": 14, "completion_tokens": 22, "total_tokens": 36},
    )
    + "data: [DONE]\n\n"
).encode("utf-8")


def _anthropic_sse() -> bytes:
    frames = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-test", "content": [],
                "usage": {"input_tokens": 9, "output_tokens": 1},
            },
        }),
        ("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Hi there"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 9, "output_tokens": 12},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = []
    for event, payload in frames:
        out.append(f"event: {event}\ndata: {json.dumps(payload)}\n\n")
    return "".join(out).encode("utf-8")


def test_openai_chat_stream_single_end_event_with_usage() -> None:
    lm = OpenAIChatLM(api_key="sk-test", transport=_FakeTransport(
        [_FakeStreamResponse(status=200, body=_VLLM_SSE)]
    ))
    events = list(lm.stream(_REQ))

    ends = [e for e in events if e.type == "end"]
    assert len(ends) == 1, f"expected exactly one end event, got {len(ends)}"
    assert events[-1].type == "end", "the end event must be the final event"
    end = ends[0]
    assert end.finish_reason == "stop"
    assert end.usage is not None
    assert end.usage.input_tokens == 14
    assert end.usage.output_tokens == 22
    assert end.usage.total_tokens == 36


def test_openai_chat_stream_materialized_usage_nonzero() -> None:
    lm = OpenAIChatLM(api_key="sk-test", transport=_FakeTransport(
        [_FakeStreamResponse(status=200, body=_VLLM_SSE)]
    ))
    response = Result(events=lm.stream(_REQ), request=_REQ).response
    assert response.text == "Hello!"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 14
    assert response.usage.output_tokens == 22


def test_anthropic_stream_single_end_event_with_finish_and_usage() -> None:
    lm = AnthropicLM(api_key="sk-test", transport=_FakeTransport(
        [_FakeStreamResponse(status=200, body=_anthropic_sse())]
    ))
    events = list(lm.stream(_REQ))

    ends = [e for e in events if e.type == "end"]
    assert len(ends) == 1, f"expected exactly one end event, got {len(ends)}"
    assert events[-1].type == "end", "the end event must be the final event"
    end = ends[0]
    assert end.finish_reason == "stop"
    assert end.usage is not None
    assert end.usage.input_tokens == 9
    assert end.usage.output_tokens == 12


def test_coalescer_never_overwrites_non_none_with_none() -> None:
    from lm15.result import coalesce_stream
    from lm15.types import StreamEndEvent, Usage

    merged = list(coalesce_stream(iter([
        StreamEndEvent(finish_reason="stop", usage=None),
        StreamEndEvent(usage=Usage(input_tokens=3, output_tokens=4)),
        StreamEndEvent(),  # bare terminator must not erase anything
    ])))
    assert len(merged) == 1
    end = merged[0]
    assert end.type == "end"
    assert end.finish_reason == "stop"
    assert end.usage == Usage(input_tokens=3, output_tokens=4)


def test_coalescer_passes_through_non_end_events() -> None:
    from lm15.result import coalesce_stream
    from lm15.types import StreamDeltaEvent, StreamEndEvent, StreamStartEvent, TextDelta

    src = [
        StreamStartEvent(id="x", model="m"),
        StreamDeltaEvent(delta=TextDelta(text="a")),
        StreamEndEvent(finish_reason="stop"),
    ]
    out = list(coalesce_stream(iter(src)))
    assert [e.type for e in out] == ["start", "delta", "end"]
    assert out[0] is src[0]
    assert out[1] is src[1]
