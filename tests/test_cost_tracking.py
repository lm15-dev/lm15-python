"""Tests for cost tracking integration (configure, Result.cost, Model.total_cost)."""

from __future__ import annotations

import pytest

import lm15.cost as cost_mod
from lm15.cost import CostBreakdown, enable_cost_tracking, disable_cost_tracking, lookup_cost, get_cost_index
from lm15.model_catalog import ModelSpec
from lm15.model import Model, HistoryEntry
from lm15.result import Result
from lm15.types import Config, LMRequest, LMResponse, Message, Part, StreamEvent, PartDelta, Usage


def _spec(model_id: str, provider: str, input_cost: float, output_cost: float, **extra_cost) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider=provider,
        context_window=128000,
        max_output=4096,
        input_modalities=("text",),
        output_modalities=("text",),
        tool_call=False,
        structured_output=False,
        reasoning=False,
        raw={"cost": {"input": input_cost, "output": output_cost, **extra_cost}},
    )


def _install_fake_index(specs: list[ModelSpec]):
    """Inject a fake cost index without hitting the network."""
    cost_mod._cost_index = {s.id: s for s in specs}


def _clear_index():
    cost_mod._cost_index = None


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    _clear_index()


class TestLookupCost:
    def test_returns_none_when_disabled(self):
        _clear_index()
        usage = Usage(input_tokens=100, output_tokens=50)
        assert lookup_cost("gpt-4o", usage) is None

    def test_returns_none_for_unknown_model(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        usage = Usage(input_tokens=100, output_tokens=50)
        assert lookup_cost("unknown-model", usage) is None

    def test_returns_cost_for_known_model(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        usage = Usage(input_tokens=1000, output_tokens=500)
        result = lookup_cost("gpt-4o", usage)
        assert result is not None
        assert result.total == pytest.approx(1000 * 3.0 / 1e6 + 500 * 15.0 / 1e6)


class TestResultCost:
    def _make_result(self, model: str, usage: Usage) -> Result:
        request = LMRequest(model=model, messages=(Message.user("test"),))
        response = LMResponse(
            id="r1", model=model,
            message=Message(role="assistant", parts=(Part.text_part("hi"),)),
            finish_reason="stop", usage=usage,
        )

        def events_from_response(req):
            yield StreamEvent(type="start", model=model)
            yield StreamEvent(type="delta", delta=PartDelta(type="text", text="hi"))
            yield StreamEvent(type="end", finish_reason="stop", usage=usage)

        return Result(request=request, start_stream=events_from_response)

    def test_cost_none_when_not_enabled(self):
        _clear_index()
        r = self._make_result("gpt-4o", Usage(input_tokens=100, output_tokens=50))
        assert r.cost is None

    def test_cost_works_when_enabled(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        r = self._make_result("gpt-4o", Usage(input_tokens=1000, output_tokens=500))
        cost = r.cost
        assert cost is not None
        assert cost.total > 0
        assert cost.input == pytest.approx(1000 * 3.0 / 1e6)
        assert cost.output == pytest.approx(500 * 15.0 / 1e6)


class TestModelTotalCost:
    def _make_model(self) -> Model:
        from lm15.client import UniversalLM
        lm = UniversalLM()
        return Model(lm=lm, model="gpt-4o")

    def test_total_cost_none_when_not_enabled(self):
        _clear_index()
        m = self._make_model()
        m.history.append(HistoryEntry(
            request=LMRequest(model="gpt-4o", messages=(Message.user("hi"),)),
            response=LMResponse(
                id="r1", model="gpt-4o",
                message=Message(role="assistant", parts=(Part.text_part("hello"),)),
                finish_reason="stop",
                usage=Usage(input_tokens=100, output_tokens=50),
            ),
        ))
        assert m.total_cost is None

    def test_total_cost_sums_history(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        m = self._make_model()

        for i in range(3):
            m.history.append(HistoryEntry(
                request=LMRequest(model="gpt-4o", messages=(Message.user(f"msg {i}"),)),
                response=LMResponse(
                    id=f"r{i}", model="gpt-4o",
                    message=Message(role="assistant", parts=(Part.text_part("ok"),)),
                    finish_reason="stop",
                    usage=Usage(input_tokens=1000, output_tokens=500),
                ),
            ))

        cost = m.total_cost
        assert cost is not None
        per_call = 1000 * 3.0 / 1e6 + 500 * 15.0 / 1e6
        assert cost.total == pytest.approx(per_call * 3)

    def test_total_cost_empty_history(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        m = self._make_model()
        cost = m.total_cost
        assert cost is not None
        assert cost.total == 0.0

    def test_total_cost_resets_on_clear(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        m = self._make_model()
        m.history.append(HistoryEntry(
            request=LMRequest(model="gpt-4o", messages=(Message.user("hi"),)),
            response=LMResponse(
                id="r1", model="gpt-4o",
                message=Message(role="assistant", parts=(Part.text_part("hello"),)),
                finish_reason="stop",
                usage=Usage(input_tokens=1000, output_tokens=500),
            ),
        ))
        assert m.total_cost.total > 0
        m.history.clear()
        assert m.total_cost.total == 0.0


class TestEnableDisable:
    def test_disable(self):
        _install_fake_index([_spec("gpt-4o", "openai", 3.0, 15.0)])
        assert get_cost_index() is not None
        disable_cost_tracking()
        assert get_cost_index() is None
