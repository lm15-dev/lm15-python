from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import repl
from lm15.errors import InvalidRequestError, RateLimitError
from lm15.repl import format_lm15_error


def test_format_model_not_found_suggests_close_models(monkeypatch):
    monkeypatch.setattr(
        repl,
        "_known_model_ids",
        lambda: ("gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "claude-sonnet-4-5"),
    )
    exc = InvalidRequestError("The requested model 'gpt-4.1-min' does not exist.")
    out = format_lm15_error(exc)
    assert "LM15 InvalidRequestError" in out
    assert "Did you mean:" in out
    assert "gpt-4.1-mini" in out


def test_format_rate_limit_has_hint():
    exc = RateLimitError("Too many requests")
    out = format_lm15_error(exc)
    assert "LM15 RateLimitError" in out
    assert "Wait a moment and retry" in out
