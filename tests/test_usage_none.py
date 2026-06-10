"""Usage zeros-vs-absent: None means "provider reported nothing".

The three core counters (input_tokens, output_tokens, total_tokens) are
``int | None`` with default ``None``.  A provider-reported zero is a real
fact and stays ``0``; an absent count stays ``None`` and is omitted on the
wire per the omission rule (docs/serde-rules.md).
"""

from __future__ import annotations

import pytest

from lm15.serde import response_from_dict, response_to_dict, usage_from_dict, usage_to_dict
from lm15.types import Message, Response, TextPart, Usage


# ─── Defaults: nothing reported ──────────────────────────────────────

def test_default_usage_means_nothing_reported() -> None:
    u = Usage()
    assert u.input_tokens is None
    assert u.output_tokens is None
    assert u.total_tokens is None


def test_default_usage_serializes_empty() -> None:
    assert usage_to_dict(Usage()) == {}


def test_empty_usage_omitted_by_enclosing_response_serializer() -> None:
    r = Response(
        id="r1",
        model="m",
        message=Message(role="assistant", parts=[TextPart(text="hi")]),
        finish_reason="stop",
        usage=Usage(),
    )
    assert "usage" not in response_to_dict(r)


# ─── Reported zeros are preserved, distinct from None ────────────────

def test_reported_zeros_stay_zero() -> None:
    u = Usage(input_tokens=0, output_tokens=0)
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.total_tokens == 0  # auto-computed: both present

    d = usage_to_dict(u)
    assert d == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_none_vs_zero_distinction_survives_round_trip() -> None:
    zeros = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    nothing = Usage()

    rt_zeros = usage_from_dict(usage_to_dict(zeros))
    rt_nothing = usage_from_dict(usage_to_dict(nothing))

    assert rt_zeros == zeros
    assert rt_nothing == nothing
    assert rt_zeros != rt_nothing
    assert rt_nothing.input_tokens is None
    assert rt_zeros.input_tokens == 0


def test_partial_usage_round_trip() -> None:
    u = Usage(input_tokens=7)
    d = usage_to_dict(u)
    assert d == {"input_tokens": 7}
    rt = usage_from_dict(d)
    assert rt.input_tokens == 7
    assert rt.output_tokens is None
    assert rt.total_tokens is None


def test_response_round_trip_without_usage() -> None:
    r = Response(
        id="r1",
        model="m",
        message=Message(role="assistant", parts=[TextPart(text="hi")]),
        finish_reason="stop",
        usage=Usage(),
    )
    rt = response_from_dict(response_to_dict(r))
    assert rt.usage == Usage()
    assert rt.usage.total_tokens is None


# ─── Auto-total rules ────────────────────────────────────────────────

def test_total_auto_computes_when_both_present() -> None:
    assert Usage(input_tokens=3, output_tokens=4).total_tokens == 7


def test_total_stays_none_when_either_side_unknown() -> None:
    assert Usage(input_tokens=3).total_tokens is None
    assert Usage(output_tokens=4).total_tokens is None
    assert Usage().total_tokens is None


def test_explicit_total_wins() -> None:
    assert Usage(input_tokens=1, output_tokens=2, total_tokens=10).total_tokens == 10
    assert Usage(total_tokens=42).total_tokens == 42


def test_negative_counts_still_rejected() -> None:
    with pytest.raises(ValueError):
        Usage(input_tokens=-1)
    with pytest.raises(ValueError):
        Usage(total_tokens=-1)
