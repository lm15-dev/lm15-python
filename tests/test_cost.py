"""Tests for lm15.cost — cost estimation from Usage + pricing data."""

from __future__ import annotations

import pytest

from lm15.cost import CostBreakdown, estimate_cost
from lm15.model_catalog import ModelSpec
from lm15.types import Usage


def _spec(provider: str, cost: dict) -> ModelSpec:
    return ModelSpec(
        id="test-model",
        provider=provider,
        context_window=128000,
        max_output=4096,
        input_modalities=("text",),
        output_modalities=("text",),
        tool_call=False,
        structured_output=False,
        reasoning=False,
        raw={"cost": cost},
    )


class TestEstimateCostOpenAI:
    """OpenAI: input_tokens is total (includes cached), reasoning is subset of output."""

    def test_basic(self):
        spec = _spec("openai", {"input": 3.0, "output": 15.0})
        usage = Usage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        result = estimate_cost(usage, spec)
        assert result.input == pytest.approx(1000 * 3.0 / 1e6)
        assert result.output == pytest.approx(500 * 15.0 / 1e6)
        assert result.total == pytest.approx(result.input + result.output)

    def test_with_cache(self):
        spec = _spec("openai", {"input": 3.0, "output": 15.0, "cache_read": 1.5})
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=300)
        result = estimate_cost(usage, spec)
        # input_tokens includes cached on OpenAI, so non-cached = 1000 - 300 = 700
        assert result.input == pytest.approx(700 * 3.0 / 1e6)
        assert result.cache_read == pytest.approx(300 * 1.5 / 1e6)

    def test_with_reasoning(self):
        spec = _spec("openai", {"input": 3.0, "output": 15.0, "reasoning": 15.0})
        usage = Usage(input_tokens=1000, output_tokens=500, reasoning_tokens=200)
        result = estimate_cost(usage, spec)
        # reasoning is subset of output on OpenAI, so text output = 500 - 200 = 300
        assert result.output == pytest.approx(300 * 15.0 / 1e6)
        assert result.reasoning == pytest.approx(200 * 15.0 / 1e6)

    def test_with_audio(self):
        spec = _spec("openai", {"input": 3.0, "output": 15.0, "input_audio": 100.0, "output_audio": 200.0})
        usage = Usage(input_tokens=1000, output_tokens=500, input_audio_tokens=100, output_audio_tokens=50)
        result = estimate_cost(usage, spec)
        # audio is subset of total
        assert result.input == pytest.approx(900 * 3.0 / 1e6)
        assert result.output == pytest.approx(450 * 15.0 / 1e6)
        assert result.input_audio == pytest.approx(100 * 100.0 / 1e6)
        assert result.output_audio == pytest.approx(50 * 200.0 / 1e6)

    def test_all_together(self):
        spec = _spec("openai", {
            "input": 3.0, "output": 15.0,
            "cache_read": 1.5, "reasoning": 15.0,
            "input_audio": 100.0, "output_audio": 200.0,
        })
        usage = Usage(
            input_tokens=1000, output_tokens=500, total_tokens=1500,
            cache_read_tokens=200, reasoning_tokens=100,
            input_audio_tokens=50, output_audio_tokens=30,
        )
        result = estimate_cost(usage, spec)
        # text input = 1000 - 200 (cache) - 50 (audio) = 750
        assert result.input == pytest.approx(750 * 3.0 / 1e6)
        # text output = 500 - 100 (reasoning) - 30 (audio) = 370
        assert result.output == pytest.approx(370 * 15.0 / 1e6)
        assert result.total == pytest.approx(
            result.input + result.output + result.cache_read
            + result.reasoning + result.input_audio + result.output_audio
        )


class TestEstimateCostAnthropic:
    """Anthropic: input_tokens is non-cached only, cache counts are additive."""

    def test_basic(self):
        spec = _spec("anthropic", {"input": 3.0, "output": 15.0})
        usage = Usage(input_tokens=1000, output_tokens=500)
        result = estimate_cost(usage, spec)
        assert result.input == pytest.approx(1000 * 3.0 / 1e6)
        assert result.output == pytest.approx(500 * 15.0 / 1e6)

    def test_with_cache(self):
        spec = _spec("anthropic", {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75})
        usage = Usage(input_tokens=700, output_tokens=500, cache_read_tokens=200, cache_write_tokens=100)
        result = estimate_cost(usage, spec)
        # Anthropic: input_tokens is already non-cached
        assert result.input == pytest.approx(700 * 3.0 / 1e6)
        assert result.cache_read == pytest.approx(200 * 0.3 / 1e6)
        assert result.cache_write == pytest.approx(100 * 3.75 / 1e6)


class TestEstimateCostGemini:
    """Gemini: input_tokens is total (includes cached), reasoning is separate."""

    def test_with_cache(self):
        spec = _spec("gemini", {"input": 0.075, "output": 0.3, "cache_read": 0.01875})
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=400)
        result = estimate_cost(usage, spec)
        # cached is subset of input
        assert result.input == pytest.approx(600 * 0.075 / 1e6)
        assert result.cache_read == pytest.approx(400 * 0.01875 / 1e6)

    def test_with_reasoning(self):
        spec = _spec("gemini", {"input": 0.075, "output": 0.3, "reasoning": 0.3})
        usage = Usage(input_tokens=1000, output_tokens=500, reasoning_tokens=200)
        result = estimate_cost(usage, spec)
        # Gemini: reasoning is separate from output (thoughtsTokenCount ≠ subset of candidatesTokenCount)
        assert result.output == pytest.approx(500 * 0.3 / 1e6)
        assert result.reasoning == pytest.approx(200 * 0.3 / 1e6)


class TestEstimateCostDict:
    """Using a plain cost dict instead of ModelSpec."""

    def test_requires_provider(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        with pytest.raises(ValueError, match="provider is required"):
            estimate_cost(usage, {"input": 3.0, "output": 15.0})

    def test_with_provider(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        result = estimate_cost(usage, {"input": 3.0, "output": 15.0}, provider="openai")
        assert result.input == pytest.approx(100 * 3.0 / 1e6)
        assert result.output == pytest.approx(50 * 15.0 / 1e6)


class TestCostBreakdownRepr:
    def test_basic_repr(self):
        cb = CostBreakdown(input=0.003, output=0.015, total=0.018)
        r = repr(cb)
        assert "$0.018000" in r
        assert "input=" in r
        assert "output=" in r

    def test_zero_fields_omitted(self):
        cb = CostBreakdown(input=0.001, total=0.001)
        r = repr(cb)
        assert "cache_read" not in r
        assert "reasoning" not in r


class TestEdgeCases:
    def test_zero_usage(self):
        spec = _spec("openai", {"input": 3.0, "output": 15.0})
        result = estimate_cost(Usage(), spec)
        assert result.total == 0.0

    def test_missing_cost_keys(self):
        spec = _spec("openai", {"input": 3.0})
        usage = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=10)
        result = estimate_cost(usage, spec)
        assert result.input == pytest.approx(90 * 3.0 / 1e6)
        assert result.output == 0.0  # no output rate
        assert result.cache_read == 0.0  # no cache_read rate

    def test_no_negative_tokens(self):
        """If cache_read > input_tokens somehow, don't go negative."""
        spec = _spec("openai", {"input": 3.0, "output": 15.0, "cache_read": 1.5})
        usage = Usage(input_tokens=100, cache_read_tokens=200)  # shouldn't happen, but be safe
        result = estimate_cost(usage, spec)
        assert result.input >= 0.0
