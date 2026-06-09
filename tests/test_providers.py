from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from lm15.providers import AnthropicLM, GeminiLM, OpenAILM
from lm15.transports import StdlibTransport
from lm15.types import (
    BuiltinTool,
    CitationDelta,
    CitationPart,
    Config,
    ContinuationDelta,
    ContinuationState,
    FunctionTool,
    ImagePart,
    LiveClientImageEvent,
    LiveClientTurnEvent,
    Message,
    Reasoning,
    Request,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallPart,
)


@dataclass
class _FakeResponse:
    status: int
    body: bytes
    headers: list[tuple[str, str]] | None = None
    reason: str = "OK"
    http_version: str = "HTTP/1.1"
    chunks: list[bytes] | None = None

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = [("content-type", "application/json")]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield from (self.chunks if self.chunks is not None else [self.body])

    def read(self) -> bytes:
        return b"".join(iter(self))

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for key, value in self.headers or []:
            if key.lower() == lname:
                return value
        return None


class _FakeTransport:
    def __init__(self, responses: list[_FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def test_provider_lms_default_to_stdlib_transport() -> None:
    lms = [OpenAILM(api_key="test"), AnthropicLM(api_key="test"), GeminiLM(api_key="test")]
    try:
        assert all(isinstance(lm.transport, StdlibTransport) for lm in lms)
    finally:
        for lm in lms:
            lm.close()


def test_openai_builds_responses_request_with_new_types() -> None:
    lm = OpenAILM(api_key="sk-test", transport=_FakeTransport(), base_url="https://example.test/v1")
    request = Request(
        model="gpt-test",
        messages=(Message.developer("Be terse."), Message.user("Hi")),
        tools=(FunctionTool(name="lookup", parameters={"type": "object", "properties": {}}),),
        config=Config(max_tokens=12, extensions={"store": False}),
    )

    http = lm.build_request(request, stream=False)
    payload = json.loads(http.body)

    assert http.url == "https://example.test/v1/responses"
    assert payload["model"] == "gpt-test"
    assert payload["input"][0]["role"] == "developer"
    assert payload["input"][1]["content"][0] == {"type": "input_text", "text": "Hi"}
    assert payload["tools"][0]["name"] == "lookup"
    assert payload["max_output_tokens"] == 12
    assert payload["store"] is False


def test_openai_reasoning_summary_is_requested_when_configured() -> None:
    lm = OpenAILM(api_key="sk-test", transport=_FakeTransport())
    request = Request(
        model="gpt-test",
        messages=(Message.user("What is 143 times 27?"),),
        config=Config(reasoning=Reasoning(effort="high", summary="auto")),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}


def test_response_format_json_schema_maps_to_provider_payloads() -> None:
    recipe_schema = {
        "type": "json_schema",
        "name": "recipe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "ingredients"],
            "additionalProperties": False,
        },
    }
    request = Request(
        model="model-test",
        messages=(Message.user("Give me a cookie recipe."),),
        config=Config(response_format=recipe_schema),
    )

    openai = OpenAILM(api_key="sk-test", transport=_FakeTransport())
    openai_payload = json.loads(openai.build_request(request, stream=False).body)
    assert openai_payload["text"] == {"format": recipe_schema}
    assert "type" not in openai_payload

    anthropic = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    anthropic_payload = json.loads(anthropic.build_request(request, stream=False).body)
    assert anthropic_payload["output_config"] == {
        "format": {"type": "json_schema", "schema": recipe_schema["schema"]}
    }
    assert "response_format" not in anthropic_payload

    gemini = GeminiLM(api_key="sk-gem", transport=_FakeTransport())
    gemini_payload = json.loads(gemini.build_request(request, stream=False).body)
    assert gemini_payload["generationConfig"] == {
        "responseMimeType": "application/json",
        "responseJsonSchema": recipe_schema["schema"],
    }
    assert "responseSchema" not in gemini_payload["generationConfig"]
    assert "type" not in gemini_payload["generationConfig"]
    assert "schema" not in gemini_payload["generationConfig"]


def test_gemini_response_format_uses_response_schema_for_openapi_subset() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    request = Request(
        model="gemini-test",
        messages=(Message.user("Return JSON."),),
        config=Config(response_format={"type": "json_schema", "schema": schema}),
    )

    lm = GeminiLM(api_key="sk-gem", transport=_FakeTransport())
    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["generationConfig"] == {
        "responseMimeType": "application/json",
        "responseSchema": schema,
    }


def test_openai_builds_assistant_history_with_output_text_parts() -> None:
    lm = OpenAILM(api_key="sk-test", transport=_FakeTransport(), base_url="https://example.test/v1")
    request = Request(
        model="gpt-test",
        messages=(
            Message.user("What is 2 + 2? Reply with one word."),
            Message.assistant("four"),
            Message.user("Repeat your previous answer in uppercase."),
        ),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["input"][1] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "four"}],
    }


def test_openai_complete_parses_reasoning_summary() -> None:
    body = json.dumps(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Multiplied 143 by 20 and 7, then added."}
                    ],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "3861"}],
                },
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    ).encode()
    transport = _FakeTransport([_FakeResponse(200, body)])
    lm = OpenAILM(api_key="sk-test", transport=transport)

    response = lm.complete(Request(model="gpt-test", messages=(Message.user("Hi"),)))

    assert response.message.parts_of(ThinkingPart) == [
        ThinkingPart("Multiplied 143 by 20 and 7, then added.")
    ]
    assert response.message.parts_of(TextPart) == [TextPart("3861")]


def test_openai_complete_parses_tool_call_response() -> None:
    body = json.dumps(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"query":"weather"}',
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    ).encode()
    transport = _FakeTransport([_FakeResponse(200, body)])
    lm = OpenAILM(api_key="sk-test", transport=transport)

    response = lm.complete(Request(model="gpt-test", messages=(Message.user("Hi"),)))

    tool_call = response.message.first(ToolCallPart)
    assert tool_call is not None
    assert tool_call.id == "call_1"
    assert tool_call.name == "lookup"
    assert tool_call.input == {"query": "weather"}
    assert response.finish_reason == "tool_call"
    assert response.usage.total_tokens == 7


def test_openai_complete_parses_web_search_citations() -> None:
    answer = "Python 3.14.4 is the latest stable release. ([Python.org](https://www.python.org/downloads/))"
    citation_start = answer.index("([Python.org]")
    body = json.dumps(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "latest stable Python release"},
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": answer,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": citation_start,
                                    "end_index": len(answer),
                                    "title": "Download Python | Python.org",
                                    "url": "https://www.python.org/downloads/",
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    ).encode()
    transport = _FakeTransport([_FakeResponse(200, body)])
    lm = OpenAILM(api_key="sk-test", transport=transport)

    response = lm.complete(
        Request(
            model="gpt-test",
            messages=(Message.user("What is the latest Python release?"),),
            tools=(BuiltinTool("web_search"),),
        )
    )

    assert response.message.parts_of(TextPart) == [TextPart(answer)]
    citations = response.message.parts_of(CitationPart)
    assert citations == [
        CitationPart(
            url="https://www.python.org/downloads/",
            title="Download Python | Python.org",
            text=answer[citation_start:],
        )
    ]
    # OpenAI builtin tool calls are provider-executed and completed; they are
    # kept in provider_data, not surfaced as app-actionable function calls.
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


def test_openai_stream_parses_reasoning_summary_delta() -> None:
    lm = OpenAILM(api_key="sk-test", transport=_FakeTransport())
    request = Request(model="gpt-test", messages=(Message.user("Hi"),))
    raw = type("Raw", (), {})()
    raw.data = json.dumps(
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "delta": "Multiplied partial products.",
        }
    )

    parsed = next(iter(lm.parse_stream_events(request, raw)))

    assert isinstance(parsed, StreamDeltaEvent)
    assert parsed.delta == ThinkingDelta(
        text="Multiplied partial products.",
        part_index=0,
    )


def test_openai_stream_parses_output_text_annotations() -> None:
    lm = OpenAILM(api_key="sk-test", transport=_FakeTransport())
    request = Request(model="gpt-test", messages=(Message.user("Hi"),))
    raw = type("Raw", (), {})()
    raw.data = json.dumps(
        {
            "type": "response.output_text.annotation.added",
            "output_index": 0,
            "annotation": {
                "type": "url_citation",
                "title": "Example",
                "url": "https://example.com",
                "text": "Example cited text",
            },
        }
    )

    parsed = next(iter(lm.parse_stream_events(request, raw)))

    assert isinstance(parsed, StreamDeltaEvent)
    assert parsed.delta == CitationDelta(
        title="Example",
        url="https://example.com",
        text="Example cited text",
        part_index=0,
    )


def test_provider_stream_splits_arbitrary_sse_chunks() -> None:
    events = b"".join(
        [
            b'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-test"}}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"Hel"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"lo"}\n\n',
            b'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3},"output":[]}}\n\n',
        ]
    )
    chunks = [events[:17], events[17:41], events[41:83], events[83:]]
    transport = _FakeTransport([_FakeResponse(200, events, headers=[("content-type", "text/event-stream")], chunks=chunks)])
    lm = OpenAILM(api_key="sk-test", transport=transport)

    parsed = list(lm.stream(Request(model="gpt-test", messages=(Message.user("Hi"),))))

    assert isinstance(parsed[0], StreamStartEvent)
    assert parsed[0].id == "resp_1"
    deltas = [event.delta for event in parsed if isinstance(event, StreamDeltaEvent)]
    assert deltas == [TextDelta(text="Hel"), TextDelta(text="lo")]
    assert isinstance(parsed[-1], StreamEndEvent)
    assert parsed[-1].usage is not None
    assert parsed[-1].usage.total_tokens == 3


def test_anthropic_stream_redacted_thinking_emits_content_and_continuation() -> None:
    lm = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    request = Request(model="claude-test", messages=(Message.user("Hi"),))
    raw = type("Raw", (), {})()
    raw.data = json.dumps(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque"},
        }
    )

    parsed = list(lm.parse_stream_events(request, raw))

    assert len(parsed) == 2
    assert isinstance(parsed[0], StreamDeltaEvent)
    assert parsed[0].delta == ThinkingDelta(text="[redacted]", part_index=0)
    assert isinstance(parsed[1], StreamDeltaEvent)
    assert parsed[1].delta == ContinuationDelta(
        provider="anthropic",
        kind="redacted_thinking",
        data={"data": "opaque"},
        part_index=0,
    )


def test_anthropic_replays_unsigned_thinking_as_text() -> None:
    lm = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    request = Request(
        model="claude-test",
        messages=(Message.assistant((ThinkingPart("scratch"),)), Message.user("Continue")),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["messages"][0]["content"] == [{"type": "text", "text": "scratch"}]


def test_anthropic_replays_signed_thinking_as_native_thinking() -> None:
    lm = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    request = Request(
        model="claude-test",
        messages=(
            Message.assistant((
                ThinkingPart(
                    "scratch",
                    continuation=(
                        ContinuationState(
                            provider="anthropic",
                            kind="thinking_signature",
                            data={"signature": "sig"},
                        ),
                    ),
                ),
            )),
            Message.user("Continue"),
        ),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "scratch", "signature": "sig"}
    ]


def test_anthropic_payload_uses_developer_prefix_and_reasoning() -> None:
    lm = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    request = Request(
        model="claude-test",
        messages=(Message.developer("Follow policy."), Message.user("Hi")),
        config=Config(extensions={"metadata": {"user_id": "u1"}}),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["text"].startswith("[developer]")
    assert payload["metadata"] == {"user_id": "u1"}


def test_anthropic_reasoning_max_tokens_reserves_answer_tokens() -> None:
    lm = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    request = Request(
        model="claude-test",
        messages=(Message.user("What is 143 times 27? Think carefully."),),
        config=Config(reasoning=Reasoning(effort="high", thinking_budget=1024)),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert payload["max_tokens"] == 2048


def test_anthropic_reasoning_max_tokens_adds_explicit_visible_budget() -> None:
    lm = AnthropicLM(api_key="sk-ant", transport=_FakeTransport())
    request = Request(
        model="claude-test",
        messages=(Message.user("Summarize briefly, after thinking."),),
        config=Config(
            max_tokens=200,
            reasoning=Reasoning(effort="medium", thinking_budget=1024),
        ),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert payload["max_tokens"] == 1224


def test_gemini_live_image_event_uses_image_mime_as_realtime_frame() -> None:
    lm = GeminiLM(api_key="sk-gem", transport=_FakeTransport())
    event = LiveClientImageEvent(data="aGk=", media_type="image/jpeg")

    payload = lm._encode_live_client_event(event)

    assert payload == [{"realtimeInput": {"video": {"mimeType": "image/jpeg", "data": "aGk="}}}]


def test_gemini_live_turn_event_reuses_prompt_parts() -> None:
    lm = GeminiLM(api_key="sk-gem", transport=_FakeTransport())
    event = LiveClientTurnEvent(parts=(TextPart("look"), ImagePart(data="aGk=", media_type="image/png")))

    payload = lm._encode_live_client_event(event)

    assert payload == [{
        "clientContent": {
            "turns": [{
                "role": "user",
                "parts": [
                    {"text": "look"},
                    {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
                ],
            }],
            "turnComplete": True,
        }
    }]


def test_gemini_stream_event_can_emit_delta_and_end_from_one_sse() -> None:
    lm = GeminiLM(api_key="sk-gem", transport=_FakeTransport())
    request = Request(model="gemini-test", messages=(Message.user("Hi"),))
    raw = type("Raw", (), {})()
    raw.data = json.dumps(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        }
    )

    parsed = list(lm.parse_stream_events(request, raw))

    assert isinstance(parsed[0], StreamDeltaEvent)
    assert parsed[0].delta == TextDelta(text="Hello")
    assert isinstance(parsed[1], StreamEndEvent)
    assert parsed[1].usage is not None
    assert parsed[1].usage.total_tokens == 2
