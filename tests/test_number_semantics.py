"""Number rule: every numeric field has a DECLARED JSON number type.

Float fields always serialize as JSON floats, int fields as JSON ints,
regardless of which Python literal the caller typed.  Constructors coerce
same-valued cross-type input (int 1 -> 1.0 for a float field, float 2.0 -> 2
for an int field), reject non-integral floats for int fields, and never
coerce bools.  Opaque payloads are untouched.  See docs/serde-rules.md,
"Number rule".
"""

import json

import pytest

from lm15.models import InferencePricing, TrainingPricing
from lm15.serde import config_to_dict, request_to_dict
from lm15.types import (
    AudioFormat,
    CacheConfig,
    Config,
    Message,
    Reasoning,
    Request,
    TextDelta,
    Usage,
)


def canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ─── One wire form: float fields ─────────────────────────────────────


def test_config_temperature_int_and_float_are_byte_identical():
    a = canonical(config_to_dict(Config(temperature=1)))
    b = canonical(config_to_dict(Config(temperature=1.0)))
    assert a == b
    assert '"temperature":1.0' in a


def test_config_top_p_int_coerces_to_float():
    c = Config(top_p=1)
    assert type(c.top_p) is float
    assert '"top_p":1.0' in canonical(config_to_dict(c))


def test_temperature_coercion_inside_request_to_dict():
    req = Request(
        model="m",
        messages=(Message.user("hi"),),
        config=Config(temperature=2),
    )
    assert '"temperature":2.0' in canonical(request_to_dict(req))


def test_pricing_rates_are_float_fields():
    p = InferencePricing(input_per_million=3, output_per_million=15)
    assert type(p.input_per_million) is float
    assert type(p.output_per_million) is float
    t = TrainingPricing(training_tokens_per_million=25, gpu_second=2)
    assert type(t.training_tokens_per_million) is float
    assert type(t.gpu_second) is float


# ─── One wire form: int fields ───────────────────────────────────────


def test_config_top_k_integral_float_coerces_to_int():
    c = Config(top_k=2.0)
    assert type(c.top_k) is int
    assert c.top_k == 2
    assert '"top_k":2' in canonical(config_to_dict(c))


def test_config_max_tokens_integral_float_coerces_to_int():
    c = Config(max_tokens=64.0)
    assert type(c.max_tokens) is int
    assert c.max_tokens == 64


def test_config_top_k_non_integral_float_rejected():
    with pytest.raises((TypeError, ValueError)):
        Config(top_k=2.5)


def test_config_max_tokens_non_integral_float_rejected():
    with pytest.raises((TypeError, ValueError)):
        Config(max_tokens=10.5)


def test_reasoning_budgets_coerce_and_reject():
    r = Reasoning(effort="medium", thinking_budget=1024.0, total_budget=2048.0)
    assert type(r.thinking_budget) is int
    assert type(r.total_budget) is int
    with pytest.raises((TypeError, ValueError)):
        Reasoning(effort="medium", thinking_budget=10.5)


def test_cache_prefix_until_index_coerces():
    c = CacheConfig(prefix_until_index=3.0)
    assert type(c.prefix_until_index) is int


def test_usage_token_counts_coerce_and_reject():
    u = Usage(input_tokens=7.0, output_tokens=2.0)
    assert type(u.input_tokens) is int
    assert type(u.output_tokens) is int
    assert u.total_tokens == 9 and type(u.total_tokens) is int
    with pytest.raises((TypeError, ValueError)):
        Usage(input_tokens=1.5)


def test_audio_format_coerces():
    af = AudioFormat(encoding="pcm16", sample_rate=24000.0, channels=1.0)
    assert type(af.sample_rate) is int
    assert type(af.channels) is int


def test_delta_part_index_coerces():
    d = TextDelta(text="x", part_index=1.0)
    assert type(d.part_index) is int


# ─── Bool never coerces ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": True},
        {"top_p": False},
        {"max_tokens": True},
        {"top_k": True},
    ],
)
def test_bool_rejected_for_numeric_config_fields(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Config(**kwargs)


def test_bool_rejected_for_usage_and_pricing():
    with pytest.raises((TypeError, ValueError)):
        Usage(input_tokens=True)
    with pytest.raises((TypeError, ValueError)):
        InferencePricing(input_per_million=True)


# ─── Opaque payloads untouched ───────────────────────────────────────


def test_extensions_numbers_untouched():
    c = Config(extensions={"x": 1, "y": 1.0})
    d = config_to_dict(c)
    assert type(d["extensions"]["x"]) is int
    assert type(d["extensions"]["y"]) is float
    assert canonical(d["extensions"]) == '{"x":1,"y":1.0}'


def test_response_format_numbers_untouched():
    c = Config(response_format={"type": "json_schema", "n": 1})
    assert type(config_to_dict(c)["response_format"]["n"]) is int
