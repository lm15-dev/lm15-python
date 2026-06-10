from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import pytest

from lm15.compat import OpenAIChatCompat
from lm15.errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    RateLimitError,
    UnsupportedModelError,
)
from lm15.providers import HttpResponse, OpenAIChatLM
from lm15.sse import parse_sse
from lm15.types import (
    Config,
    FunctionTool,
    ImagePart,
    Message,
    Reasoning,
    Request,
    StreamDeltaEvent,
    StreamEndEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
)


@dataclass
class _FakeResponse:
    status: int
    body: bytes
    headers: list[tuple[str, str]] | None = None
    reason: str = "OK"
    http_version: str = "HTTP/1.1"

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = [("content-type", "application/json")]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield self.body

    def read(self) -> bytes:
        return self.body


class _FakeTransport:
    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _lm(**kwargs) -> OpenAIChatLM:
    kwargs.setdefault("api_key", "sk-test")
    kwargs.setdefault("transport", _FakeTransport())
    return OpenAIChatLM(**kwargs)


def _payload(lm: OpenAIChatLM, request: Request, stream: bool = False) -> dict:
    return json.loads(lm.build_request(request, stream=stream).body)


def _http(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status=status,
        reason="OK",
        headers=[("content-type", "application/json")],
        body=json.dumps(body).encode("utf-8"),
    )


def _events(lm: OpenAIChatLM, request: Request, sse_text: str):
    events = []
    for raw in parse_sse(iter(sse_text.encode("utf-8").splitlines(keepends=True))):
        events.extend(lm.parse_stream_events(request, raw))
    return events


_REQ = Request(model="m-test", messages=(Message.user("Hi"),))


# ─── build_request ───────────────────────────────────────────────────


def test_build_request_url_headers_and_basic_shape() -> None:
    lm = _lm(base_url="https://example.test/v1")
    http = lm.build_request(_REQ, stream=False)
    assert http.url == "https://example.test/v1/chat/completions"
    assert http.method == "POST"
    headers = dict(http.headers)
    assert headers["Authorization"] == "Bearer sk-test"
    payload = json.loads(http.body)
    assert payload["model"] == "m-test"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert "stream" not in payload


def test_build_request_system_and_text_only_content_is_plain_string() -> None:
    request = Request(
        model="m-test",
        system="Be terse.",
        messages=(Message.user("Hi"),),
    )
    payload = _payload(_lm(), request)
    assert payload["messages"][0] == {"role": "system", "content": "Be terse."}
    assert payload["messages"][1] == {"role": "user", "content": "Hi"}


def test_build_request_multimodal_uses_content_array_with_data_uri() -> None:
    request = Request(
        model="m-test",
        messages=(
            Message(role="user", parts=(
                TextPart(text="What is this?"),
                ImagePart(media_type="image/png", data="QUJD"),
            )),
        ),
    )
    payload = _payload(_lm(), request)
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def test_build_request_tools_and_tool_choice_forms() -> None:
    tool = FunctionTool(
        name="lookup",
        description="Look things up",
        parameters={"type": "object", "properties": {}},
    )
    base = dict(model="m-test", messages=(Message.user("Hi"),), tools=(tool,))

    payload = _payload(_lm(), Request(**base))
    assert payload["tools"] == [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look things up",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    assert "tool_choice" not in payload

    for mode, expected in (("auto", "auto"), ("required", "required"), ("none", "none")):
        payload = _payload(_lm(), Request(**base, config=Config(tool_choice=ToolChoice(mode=mode))))
        assert payload["tool_choice"] == expected

    payload = _payload(_lm(), Request(**base, config=Config(tool_choice=ToolChoice(mode="required", allowed=("lookup",)))))
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}

    payload = _payload(_lm(), Request(**base, config=Config(tool_choice=ToolChoice(mode="auto", parallel=False))))
    assert payload["parallel_tool_calls"] is False


def test_build_request_tool_result_and_assistant_tool_call_echo() -> None:
    request = Request(
        model="m-test",
        messages=(
            Message.user("Weather in Hull?"),
            Message(role="assistant", parts=(
                ToolCallPart(id="call_1", name="weather", input={"city": "Hull"}),
            )),
            Message(role="tool", parts=(
                ToolResultPart(id="call_1", name="weather", content=(TextPart(text="snow"),)),
            )),
        ),
    )
    payload = _payload(_lm(), request)
    assistant = payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert assistant["tool_calls"] == [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Hull"}'},
    }]
    assert payload["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "snow",
    }


def test_build_request_config_fields_and_max_tokens_field_policy() -> None:
    config = Config(max_tokens=42, temperature=0.5, top_p=0.9, stop=("END",))
    request = Request(model="m-test", messages=(Message.user("Hi"),), config=config)

    payload = _payload(_lm(), request)  # default openai policy
    assert payload["max_completion_tokens"] == 42
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["END"]

    payload = _payload(_lm(compat="groq"), request)
    assert payload["max_tokens"] == 42
    assert "max_completion_tokens" not in payload


def test_build_request_response_format_forms() -> None:
    request = Request(
        model="m-test",
        messages=(Message.user("Hi"),),
        config=Config(response_format={"type": "json_object"}),
    )
    assert _payload(_lm(), request)["response_format"] == {"type": "json_object"}

    schema_format = {
        "type": "json_schema",
        "name": "recipe",
        "strict": True,
        "schema": {"type": "object", "properties": {}},
    }
    request = Request(
        model="m-test",
        messages=(Message.user("Hi"),),
        config=Config(response_format=schema_format),
    )
    assert _payload(_lm(), request)["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "recipe",
            "strict": True,
            "schema": {"type": "object", "properties": {}},
        },
    }


def test_build_request_reasoning_effort_policy() -> None:
    request = Request(
        model="m-test",
        messages=(Message.user("Hi"),),
        config=Config(reasoning=Reasoning(effort="high")),
    )
    assert _payload(_lm(), request)["reasoning_effort"] == "high"
    # ollama policy: server does not support reasoning_effort — omit.
    ollama_payload = _payload(_lm(compat="ollama"), request)
    assert "reasoning_effort" not in ollama_payload
    assert "reasoning" not in ollama_payload
    # openrouter policy: nested reasoning object.
    assert _payload(_lm(compat="openrouter"), request)["reasoning"] == {"effort": "high"}


def test_build_request_stream_adds_stream_options_usage() -> None:
    payload = _payload(_lm(), _REQ, stream=True)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_build_request_extensions_passthrough_wins_last() -> None:
    request = Request(
        model="m-test",
        messages=(Message.user("Hi"),),
        config=Config(extensions={"seed": 7, "temperature": 0.1}),
    )
    payload = _payload(_lm(), request)
    assert payload["seed"] == 7
    assert payload["temperature"] == 0.1


# ─── presets ─────────────────────────────────────────────────────────


def test_preset_resolution_sets_default_base_url() -> None:
    cases = {
        "ollama": "http://localhost:11434/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "vllm": "http://localhost:8000/v1",
        "sglang": "http://localhost:30000/v1",
    }
    for name, url in cases.items():
        assert _lm(compat=name).base_url == url


def test_preset_explicit_base_url_overrides_preset_default() -> None:
    lm = _lm(compat="ollama", base_url="http://gpu-box:11434/v1")
    assert lm.base_url == "http://gpu-box:11434/v1"
    assert lm.build_request(_REQ, stream=False).url == "http://gpu-box:11434/v1/chat/completions"


def test_compat_object_is_accepted_directly() -> None:
    lm = _lm(compat=OpenAIChatCompat(max_tokens_field="max_tokens"))
    request = Request(model="m-test", messages=(Message.user("Hi"),), config=Config(max_tokens=5))
    assert _payload(lm, request)["max_tokens"] == 5


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        _lm(compat="not-a-preset")


# ─── parse_response ──────────────────────────────────────────────────


def test_parse_response_text_and_usage_details() -> None:
    body = {
        "id": "chatcmpl-1",
        "model": "m-live",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 6},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    response = _lm().parse_response(_REQ, _http(body))
    assert response.id == "chatcmpl-1"
    assert response.model == "m-live"
    assert response.message.parts == (TextPart(text="Hello!"),)
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 14
    assert response.usage.cache_read_tokens == 6
    assert response.usage.reasoning_tokens == 2
    assert "_lm15_unmapped" not in response.provider_data


def test_parse_response_tool_calls_and_reasoning_content() -> None:
    body = {
        "id": "chatcmpl-2",
        "model": "m-live",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "thinking about it",
                "tool_calls": [{
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Hull"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    response = _lm().parse_response(_REQ, _http(body))
    assert response.message.parts == (
        ThinkingPart(text="thinking about it"),
        ToolCallPart(id="call_9", name="weather", input={"city": "Hull"}),
    )
    assert response.finish_reason == "tool_call"


def test_parse_response_finish_reasons_map() -> None:
    for raw, expected in (("stop", "stop"), ("length", "length"), ("content_filter", "content_filter")):
        body = {"choices": [{"message": {"content": "x"}, "finish_reason": raw}]}
        assert _lm().parse_response(_REQ, _http(body)).finish_reason == expected


def test_parse_response_empty_message_rule_map2() -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]}
    response = _lm().parse_response(_REQ, _http(body))
    assert response.message.parts == (TextPart(text=""),)
    assert response.finish_reason == "length"


def test_parse_response_records_unmapped_canary() -> None:
    body = {
        "choices": [{
            "message": {"content": [{"type": "weird_block", "x": 1}]},
            "finish_reason": "stop",
        }],
    }
    response = _lm().parse_response(_REQ, _http(body))
    unmapped = response.provider_data["_lm15_unmapped"]
    assert unmapped == [{"path": "choices[0].message.content[0]", "type": "weird_block"}]


# ─── parse_stream_events ─────────────────────────────────────────────


def _chunk(**kwargs) -> str:
    choices = kwargs.pop("choices", [])
    payload = {"id": "chatcmpl-s", "object": "chat.completion.chunk", "model": "m-live", "choices": choices, **kwargs}
    return "data: " + json.dumps(payload) + "\n\n"


def test_stream_event_sequence_text_reasoning_tools_usage_done() -> None:
    sse = (
        _chunk(choices=[{"index": 0, "delta": {"reasoning_content": "hmm"}}])
        + _chunk(choices=[{"index": 0, "delta": {"content": "Hel"}}])
        + _chunk(choices=[{"index": 0, "delta": {"content": "lo"}}])
        + _chunk(choices=[{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "weather", "arguments": '{"ci'},
        }]}}])
        + _chunk(choices=[{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": 'ty":"Hull"}'},
        }]}}])
        + _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        + _chunk(choices=[], usage={"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14})
        + "data: [DONE]\n\n"
    )
    events = _events(_lm(), _REQ, sse)
    assert [type(e).__name__ for e in events] == [
        "StreamDeltaEvent", "StreamDeltaEvent", "StreamDeltaEvent",
        "StreamDeltaEvent", "StreamDeltaEvent",
        "StreamEndEvent", "StreamEndEvent", "StreamEndEvent",
    ]
    assert events[0].delta == ThinkingDelta(text="hmm")
    assert events[1].delta == TextDelta(text="Hel")
    assert events[2].delta == TextDelta(text="lo")
    assert events[3].delta == ToolCallDelta(input='{"ci', part_index=0, id="call_1", name="weather")
    assert events[4].delta == ToolCallDelta(input='ty":"Hull"}', part_index=0)
    assert events[5].finish_reason == "tool_call"
    assert events[6].finish_reason is None
    assert events[6].usage.input_tokens == 5
    assert events[6].usage.output_tokens == 9
    assert events[7] == StreamEndEvent()


def test_stream_finish_reason_stop_and_bare_done() -> None:
    sse = (
        _chunk(choices=[{"index": 0, "delta": {"content": "ok"}}])
        + _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
        + "data: [DONE]\n\n"
    )
    events = _events(_lm(), _REQ, sse)
    assert isinstance(events[0], StreamDeltaEvent)
    assert events[1].finish_reason == "stop"
    assert events[2] == StreamEndEvent()


# ─── normalize_error ─────────────────────────────────────────────────


def _error_body(**err) -> str:
    return json.dumps({"error": {"message": "boom", **err}})


def test_normalize_error_mapping() -> None:
    lm = _lm()
    assert isinstance(lm.normalize_error(400, _error_body(code="context_length_exceeded")), ContextLengthError)
    assert isinstance(lm.normalize_error(404, _error_body(code="model_not_found")), UnsupportedModelError)
    assert isinstance(lm.normalize_error(429, _error_body(code="insufficient_quota")), BillingError)
    assert isinstance(lm.normalize_error(401, _error_body(code="invalid_api_key")), AuthError)
    assert isinstance(lm.normalize_error(429, _error_body(type="rate_limit_error")), RateLimitError)
    err = lm.normalize_error(429, _error_body(code="rate_limit_exceeded"))
    assert isinstance(err, RateLimitError)
    assert err.provider_code == "rate_limit_exceeded"


def test_normalize_error_non_json_body() -> None:
    err = _lm().normalize_error(503, "upstream exploded")
    assert "upstream exploded" in err.message


# ─── live smoke vs local ollama ──────────────────────────────────────


def _ollama_model() -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
    if not models:
        return None
    for name in models:
        if name.startswith("qwen"):
            return name
    return models[0]


_OLLAMA_MODEL = _ollama_model()


@pytest.mark.skipif(_OLLAMA_MODEL is None, reason="local ollama is not running")
class TestOllamaLiveSmoke:
    def _lm(self) -> OpenAIChatLM:
        return OpenAIChatLM(api_key="ollama", compat="ollama")

    def test_complete_returns_text(self) -> None:
        with self._lm() as lm:
            response = lm.complete(Request(
                model=_OLLAMA_MODEL,
                messages=(Message.user("Reply with exactly: pong"),),
                # qwen thinking models otherwise spend the whole budget thinking;
                # ollama honors reasoning_effort="none" as think:false.
                config=Config(max_tokens=200, extensions={"reasoning_effort": "none"}),
            ))
        assert isinstance(response.text, str) and response.text.strip()
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_stream_yields_deltas_and_end_with_usage(self) -> None:
        deltas: list[str] = []
        ends: list[StreamEndEvent] = []
        with self._lm() as lm:
            for event in lm.stream(Request(
                model=_OLLAMA_MODEL,
                messages=(Message.user("Count from 1 to 5."),),
                config=Config(max_tokens=200, extensions={"reasoning_effort": "none"}),
            )):
                if isinstance(event, StreamDeltaEvent) and isinstance(event.delta, TextDelta):
                    deltas.append(event.delta.text)
                elif isinstance(event, StreamEndEvent):
                    ends.append(event)
        assert "".join(deltas).strip()
        assert any(e.usage is not None and e.usage.output_tokens > 0 for e in ends)
