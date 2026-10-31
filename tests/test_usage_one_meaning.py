"""MAP-19 (one meaning per usage counter) and InferencePricing.estimate.

The contract's mapping/usage.json vectors pin the same rules across SDKs;
these tests pin the Python surface around them: the compat field, the
per-model override, the pricing types and the estimate's edges.
"""

from __future__ import annotations

import json

import pytest

from lm15 import serde
from lm15.compat import OpenAIChatCompat
from lm15.models import InferencePricing, PricingTier
from lm15.providers import OpenAIChatLM
from lm15.providers.common import settle_usage
from lm15.providers import HttpResponse
from lm15.types import Message, Request, TextPart, Usage

_REQ = Request(model="m", messages=(Message(role="user", parts=(TextPart(text="hi"),)),))


def _chat(usage: dict) -> HttpResponse:
    body = {"id": "c", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                 "finish_reason": "stop"}], "usage": usage}
    return HttpResponse(status=200, reason="OK", headers=[("content-type", "application/json")],
                        body=json.dumps(body).encode())


XAI_USAGE = {"prompt_tokens": 254, "completion_tokens": 9, "total_tokens": 374,
             "completion_tokens_details": {"reasoning_tokens": 111}}


def test_reasoning_beside_completion_is_added_under_separate() -> None:
    lm = OpenAIChatLM(api_key="k", compat=OpenAIChatCompat(reasoning_usage="separate"))
    usage = lm.parse_response(_REQ, _chat(XAI_USAGE)).usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens, usage.reasoning_tokens) == (254, 120, 374, 111)


def test_the_same_body_under_the_dialect_rule_withholds_output() -> None:
    # Without the preset saying so, reasoning above completion means the wire
    # is not counting the declared way: output is withheld, never 9.
    usage = OpenAIChatLM(api_key="k").parse_response(_REQ, _chat(XAI_USAGE)).usage
    assert usage.output_tokens is None and usage.input_tokens == 254 and usage.total_tokens == 374


def test_reasoning_usage_is_a_per_model_override() -> None:
    compat = OpenAIChatCompat(model_overrides=(("grok-", {"reasoning_usage": "separate"}),))
    lm = OpenAIChatLM(api_key="k", compat=compat)
    request = Request(model="grok-4.6", messages=_REQ.messages)
    assert lm.parse_response(request, _chat(XAI_USAGE)).usage.output_tokens == 120
    assert lm.parse_response(_REQ, _chat(XAI_USAGE)).usage.output_tokens is None


def test_xai_preset_is_separate() -> None:
    assert OpenAIChatCompat.preset("xai").reasoning_usage == "separate"
    assert OpenAIChatCompat.preset("deepseek").reasoning_usage is None  # the dialect's rule


def test_reasoning_usage_vocabulary_is_closed() -> None:
    with pytest.raises(ValueError):
        OpenAIChatCompat(reasoning_usage="beside")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("counters", "expected"),
    [
        ({"input_tokens": 10, "output_tokens": 5, "total_tokens": 20}, {"total_tokens": 20}),
        ({"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 12}, {"output_tokens": 5, "cache_read_tokens": 12}),
        ({"input_tokens": 10, "output_tokens": 2, "reasoning_tokens": 8, "total_tokens": 12},
         {"input_tokens": 10, "total_tokens": 12, "reasoning_tokens": 8}),
        ({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
         {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
    ],
)
def test_settle_withholds_what_the_numbers_do_not_determine(counters, expected) -> None:
    assert serde.usage_to_dict(settle_usage(**{"input_tokens": None, "output_tokens": None, **counters})) == expected


SOL = InferencePricing(input_per_million=2, output_per_million=10, cache_read_per_million=0.1,
                       cache_write_per_million=2.5,
                       tiers=[PricingTier(above_input_tokens=272_000, input_per_million=4, output_per_million=15,
                                          cache_read_per_million=0.2, cache_write_per_million=5)])


def test_estimate_charges_cached_tokens_once() -> None:
    usage = Usage(input_tokens=100_000, output_tokens=1_000, cache_read_tokens=90_000, cache_write_tokens=0)
    assert SOL.estimate(usage) == pytest.approx(0.039, abs=0)


def test_estimate_takes_a_usage() -> None:
    with pytest.raises(TypeError):
        SOL.estimate(input_tokens=1, output_tokens=1)  # type: ignore[call-arg]  # the 1.x keyword form is gone


def test_estimate_unknown_is_none_not_zero() -> None:
    assert SOL.estimate(Usage(input_tokens=10, output_tokens=1)) is None


@pytest.mark.parametrize(
    "tiers",
    [
        [PricingTier(above_input_tokens=2000), PricingTier(above_input_tokens=1000)],
        [PricingTier(above_input_tokens=1000), PricingTier(above_input_tokens=1000)],
    ],
)
def test_tiers_are_strictly_ascending(tiers) -> None:
    with pytest.raises(ValueError, match="ascending"):
        InferencePricing(tiers=tiers)


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5])
def test_tier_threshold_is_a_positive_integer(threshold) -> None:
    with pytest.raises(ValueError):
        PricingTier(above_input_tokens=threshold)


def test_pricing_round_trips_with_tiers() -> None:
    assert serde.inference_pricing_from_dict(serde.inference_pricing_to_dict(SOL)) == SOL
    assert "tiers" not in serde.inference_pricing_to_dict(InferencePricing(input_per_million=1))
