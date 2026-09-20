"""INV-013/052 input boundaries and MAP-14 native score criteria."""
import json

import pytest

from lm15 import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM, TypeSafeLM
from lm15.judgments import judgments
from lm15.types import (
    Config, DataPart, LiveClientToolResultEvent, LiveClientTurnEvent,
    Message, Request, data, tool_call, tool_result,
)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def test_unmeasured_data_is_tool_input_not_a_protocol_part():
    part = data({"count": 1})
    assert tool_result("call", part).content == (part,)
    assert LiveClientToolResultEvent(id="call", content=(part,)).content == (part,)
    assert LiveClientTurnEvent(parts=(part,)).parts == (part,)


@pytest.mark.parametrize("construct", [
    lambda p: tool_result("call", p),
    lambda p: LiveClientToolResultEvent(id="call", content=(p,)),
    lambda p: LiveClientTurnEvent(parts=(p,)),
])
def test_measurements_cannot_reenter_as_tool_or_live_input(construct):
    measured = DataPart(value={"q": True}, probabilities={"q": {"true": 0.7, "false": 0.3}}, method="provider_classification")
    with pytest.raises(TypeError, match="assistant"):
        construct(measured)


@pytest.mark.parametrize("adapter", [OpenAILM, OpenAIChatLM, AnthropicLM, GeminiLM])
def test_tool_data_reaches_text_wire_without_invented_wrapper(adapter):
    request = Request(model="test-model", messages=(
        Message.user("count"),
        Message.assistant(tool_call("call", "count", {})),
        Message.tool(tool_result("call", data({"count": 1}), name="count")),
    ))
    wire = adapter(api_key="synthetic-key").build_request(request, False)
    assert '{"count":1}' in tuple(strings(json.loads(wire.body)))


def test_typesafe_ordered_titles_are_not_sent_as_criteria():
    request = Request(model="jev-test", messages=(Message.user("state"),), config=Config(response_format=judgments(q={
        "type": "integer", "description": "Rate it", "anyOf": [
            {"const": 0, "title": "LOW"}, {"const": 1, "title": "HIGH"},
        ],
    })))
    wire = TypeSafeLM(api_key="synthetic-key").build_request(request, False)
    assert json.loads(wire.body)["questions"]["q"]["criteria"] == ["0", "1"]


def test_typesafe_extensions_remain_explicit_passthrough_but_multiple_answers_refuse():
    from lm15 import UnsupportedFeatureError
    lm = TypeSafeLM(api_key="synthetic-key")
    fmt = judgments(q={"type": "boolean"})
    request = Request(model="jev-test", messages=(Message.user("state"),), config=Config(response_format=fmt, extensions={"state": "explicit override"}))
    assert json.loads(lm.build_request(request, False).body)["state"] == "explicit override"
    multiple = Request(model="jev-test", messages=request.messages, config=Config(response_format=fmt, extensions={"n": 2}))
    with pytest.raises(UnsupportedFeatureError) as error:
        lm.build_request(multiple, False)
    assert error.value.feature == "config.extensions.n"


def test_streamed_judgment_materializes_as_data_like_complete():
    from lm15.result import StreamAccumulator
    from lm15.types import StreamDeltaEvent, StreamEndEvent, TextDelta
    request = Request(model="test-model", messages=(Message.user("state"),), config=Config(response_format=judgments(q={"type": "boolean"})))
    accumulator = StreamAccumulator(request)
    accumulator.push(StreamDeltaEvent(delta=TextDelta(text='{"q":')))
    assert accumulator.response().message.parts[0].type == "text"
    accumulator.push(StreamDeltaEvent(delta=TextDelta(text='true}')))
    accumulator.push(StreamEndEvent(finish_reason="stop"))
    part = accumulator.response().message.parts[0]
    assert isinstance(part, DataPart)
    assert part.value == {"q": True}
    assert part.probabilities is None

    ordinary = StreamAccumulator(Request(model="test-model", messages=(Message.user("state"),), config=Config(response_format={"type": "json_object"})))
    ordinary.push(StreamDeltaEvent(delta=TextDelta(text='{"q":true}')))
    assert ordinary.response().message.parts[0].type == "text"


@pytest.mark.parametrize("adapter", [OpenAILM, OpenAIChatLM, AnthropicLM, GeminiLM])
def test_planning_does_not_read_media_but_actual_build_does(adapter, monkeypatch):
    from pathlib import Path
    from lm15.types import image
    def forbidden_read(_path):
        raise AssertionError("media read")
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    request = Request(model="test-model", messages=(Message.user(image(path=Path("missing.png"))),))
    lm = adapter(api_key="synthetic-key")
    lm.plan(request)
    with pytest.raises(AssertionError, match="media read"):
        lm.build_request(request, False)
