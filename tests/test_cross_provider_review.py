"""Negative boundaries identified while reviewing saved cross-provider replies."""
import json
from pathlib import Path

import pytest

from lm15 import (
    Adaptation,
    GeminiLM,
    ImageGenerationRequest,
    Message,
    OpenAIChatLM,
    OpenAILM,
    ProviderError,
    Request,
    StreamAssemblyError,
    StreamEndEvent,
    TokenLogprob,
    TopLogprob,
    UnsupportedFeatureError,
    XaiLM,
)
from lm15.models import InferenceModelInfo, InferencePricing, ModelInfo, ModelOrigin, PricingTier
from lm15.providers.base import HttpResponse
from lm15.providers.openai import _FinalCopies
from lm15.result import StreamAccumulator
from lm15.serde import request_from_dict
from lm15.testing import FakeTransport


def frame(shape, name, omit=False):
    call = {"id": "call-preserved", "args": {"city": "Montreal"}}
    if not omit:
        call["name"] = name
    if shape == "toolCall":
        return json.dumps({"toolCall": {"functionCalls": [call]}})
    return json.dumps({"serverContent": {"modelTurn": {"parts": [{"functionCall": call}]}}})


@pytest.mark.parametrize("shape", ["toolCall", "modelTurn"])
@pytest.mark.parametrize("name,omit", [(None, True), (None, False), ("", False), (123, False)])
def test_live_complete_call_never_invents_a_name(shape, name, omit):
    lm = GeminiLM(api_key="fixture")
    with pytest.raises(ProviderError, match="no name"):
        lm._decode_live_server_event(frame(shape, name, omit))


@pytest.mark.parametrize("shape", ["toolCall", "modelTurn"])
@pytest.mark.parametrize("name,omit", [(None, True), (None, False), ("", False), (123, False)])
def test_live_completion_stream_salvages_but_never_names_a_call(shape, name, omit):
    lm = GeminiLM(api_key="fixture")
    events, _, _ = lm._decode_live_completion_stream_events(frame(shape, name, omit))
    acc = StreamAccumulator(Request(model="m", messages=(Message.user("hi"),)))
    for event in events:
        acc.push(event)
    with pytest.raises(StreamAssemblyError):
        acc.response()


def test_live_completion_name_failure_preserves_text_and_usage():
    payload = {"serverContent": {"modelTurn": {"parts": [
        {"text": "Already delivered."}, {"functionCall": {"id": "c", "args": {"city": "Montreal"}}}
    ]}, "turnComplete": True}, "usageMetadata": {"promptTokenCount": 10, "responseTokenCount": 5, "totalTokenCount": 15}}
    events, complete, usage = GeminiLM(api_key="fixture")._decode_live_completion_stream_events(json.dumps(payload))
    assert complete
    acc = StreamAccumulator(Request(model="m", messages=(Message.user("hi"),)))
    for event in events:
        acc.push(event)
    acc.push(StreamEndEvent(finish_reason="tool_call", usage=usage))
    with pytest.raises(StreamAssemblyError) as exc:
        acc.response()
    assert exc.value.partial.text == "Already delivered."
    assert exc.value.partial.usage.total_tokens == 15
    assert not exc.value.partial.tool_calls


@pytest.mark.parametrize("shape", ["toolCall", "modelTurn"])
@pytest.mark.parametrize("name", ["get_weather", "tool"])
def test_valid_live_tool_names_and_correlators_stay_verbatim(shape, name):
    event, = GeminiLM(api_key="fixture")._decode_live_server_event(frame(shape, name))
    assert (event.name, event.id, event.input) == (name, "call-preserved", {"city": "Montreal"})


@pytest.mark.parametrize("feature,field", [("reasoning_effort_switch", "config.reasoning.effort"), ("response_format_json_schema", "config.response_format")])
def test_tinker_reviewed_adaptations_are_visible_and_strictly_refused(feature, field):
    root = Path(__file__).resolve().parents[2] / "lm15-contract"
    case = json.loads((root / "cases/tinker" / f"{feature}.json").read_text(encoding="utf-8"))
    request = request_from_dict(case["canonical_request"])
    lm = OpenAIChatLM(api_key="fixture", compat="tinker")
    note, = lm.plan(request)
    assert note.field == field
    expected = case["expect_lm15"]["adaptations"][0]
    assert note.action == expected["action"]
    assert note.asked == expected["asked"]
    if "applied" in expected:
        assert note.applied == expected["applied"]
    # Python's low-level build_request hook is policy-free; complete owns the
    # adaptation scope. An empty fake transport forbids any attempted send.
    strict = OpenAIChatLM(api_key="fixture", compat="tinker", adaptations="refuse", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError) as exc:
        strict.complete(request)
    assert exc.value.feature == field


def openai_live_frame(name, omit=False):
    item = {"id": "item_1", "type": "function_call", "status": "completed", "call_id": "call-preserved", "arguments": '{"city":"Montreal"}'}
    if not omit:
        item["name"] = name
    return json.dumps({"type": "response.output_item.done", "output_index": 0, "item": item})


@pytest.mark.parametrize("name,omit", [(None, True), (None, False), ("", False), (123, False)])
def test_openai_live_complete_call_never_invents_a_name(name, omit):
    lm = OpenAILM(api_key="fixture")
    with pytest.raises(ProviderError, match="no name"):
        lm._decode_live_server_event(openai_live_frame(name, omit))


@pytest.mark.parametrize("name,omit", [(None, True), (None, False), ("", False), (123, False)])
def test_openai_live_completion_stream_preserves_missing_name_for_assembly_error(name, omit):
    lm = OpenAILM(api_key="fixture")
    req = Request(model="gpt-realtime-mini", messages=(Message.user("hi"),))
    events = lm._decode_live_completion_stream_events(req, openai_live_frame(name, omit), _FinalCopies())
    acc = StreamAccumulator(req)
    for event in events:
        acc.push(event)
    acc.push(StreamEndEvent(finish_reason="tool_call"))
    with pytest.raises(StreamAssemblyError):
        acc.response()


@pytest.mark.parametrize("name", ["get_weather", "tool"])
def test_openai_live_valid_tool_names_stay_verbatim(name):
    event, = OpenAILM(api_key="fixture")._decode_live_server_event(openai_live_frame(name))
    assert (event.name, event.id, event.input) == (name, "call-preserved", {"city": "Montreal"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_rejected_in_pricing_logprobs_and_adaptations(bad):
    with pytest.raises(ValueError):
        InferencePricing(input_per_million=bad)
    with pytest.raises(ValueError):
        PricingTier(above_input_tokens=1000, output_per_million=bad)
    with pytest.raises(ValueError):
        TopLogprob(token="hi", logprob=bad)
    with pytest.raises(ValueError):
        TokenLogprob(token="hi", logprob=bad)
    with pytest.raises(TypeError):
        Adaptation(field="config.temperature", action="clamped", reason="r", asked=bad)
    with pytest.raises(TypeError):
        Adaptation(field="config.temperature", action="clamped", reason="r", applied={"nested": bad})
    with pytest.raises(TypeError):
        InferencePricing(dimensions={"bad": bad})
    with pytest.raises(TypeError):
        InferenceModelInfo(extensions={"bad": bad})
    with pytest.raises(TypeError):
        ModelOrigin(provider_data={"bad": bad})
    with pytest.raises(TypeError):
        ModelInfo(id="m", provider="openai", api_family="openai_responses", extensions={"bad": bad})


def test_image_generation_records_unmapped_items():
    req = ImageGenerationRequest(model="gpt-image-1", prompt="cat")
    body = json.dumps({"output_format": "png", "data": [{"b64_json": "YWJj"}, {"revised_only": "x"}, "not-an-object"]}).encode("utf-8")
    resp = HttpResponse(status=200, reason="OK", headers=[("content-type", "application/json")], body=body)
    openai_out = OpenAILM(api_key="fixture")._image_generation_from_response(req, resp)
    assert openai_out.provider_data is not None
    assert openai_out.provider_data["_lm15_unmapped"] == [
        {"path": "data[1]", "type": "revised_only"},
        {"path": "data[2]", "type": "str"},
    ]
    xai_out = XaiLM(api_key="fixture")._image_generation_from_response(req, resp)
    assert xai_out.provider_data is not None
    assert xai_out.provider_data["_lm15_unmapped"] == [
        {"path": "data[1]", "type": "revised_only"},
        {"path": "data[2]", "type": "str"},
    ]
