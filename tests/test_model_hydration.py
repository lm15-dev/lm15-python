"""Tests for the model-hydration contract (docs/model-hydration.md)."""

from __future__ import annotations

import warnings

import pytest

from lm15 import ModelRegistry, model_info_from_dict, model_info_to_dict
from lm15.models import InferencePricing, ModelInfo


FULL = {
    "id": "omega-4",
    "provider": "exampleai",
    "api_family": "openai-chat",
    "aliases": ["omega-4-latest"],
    "origin": {"type": "fine-tune", "id": "ft-123", "base_model": "omega-3"},
    "inference": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "context_window": 200000,
        "max_output_tokens": 8192,
        "supports_reasoning": True,
        "reasoning_efforts": ["low", "medium", "high"],
        "pricing": {
            "input_per_million": 3.0,
            "output_per_million": 15.0,
            "cache_read_per_million": 0.3,
            "cache_write_per_million": 3.75,
            "currency": "USD",
        },
    },
    "training": {
        "supports_lora": True,
        "trainable_modalities": ["text"],
        "pricing": {"training_tokens_per_million": 25.0, "currency": "USD"},
    },
    "extensions": {"vendor": {"empty": {}}},
}

MINIMAL = {"id": "tiny-1", "provider": "exampleai", "api_family": "openai-chat"}


# ─── Serde round-trips ───────────────────────────────────────────────

def test_full_round_trip_exact():
    assert model_info_to_dict(model_info_from_dict(FULL)) == FULL


def test_minimal_round_trip_exact():
    assert model_info_to_dict(model_info_from_dict(MINIMAL)) == MINIMAL


def test_default_origin_omitted():
    info = ModelInfo(id="m", provider="p", api_family="openai-chat")
    assert "origin" not in model_info_to_dict(info)


def test_opaque_extensions_verbatim():
    value = dict(MINIMAL, extensions={"defaults": {}, "note": ""})
    assert model_info_to_dict(model_info_from_dict(value)) == value


def test_round_trip_preserves_types():
    info = model_info_from_dict(FULL)
    assert isinstance(info.inference.pricing, InferencePricing)
    assert info.inference.pricing.cache_read_per_million == 0.3
    assert info.training.pricing.training_tokens_per_million == 25.0
    assert info.origin.base_model == "omega-3"
    assert info.aliases == ("omega-4-latest",)


# ─── from_dicts validation ───────────────────────────────────────────

def test_from_dicts_builds_registry():
    registry = ModelRegistry.from_dicts([FULL, MINIMAL])
    assert registry.get("exampleai", "omega-4").id == "omega-4"
    assert registry.get("exampleai", "omega-4-latest").id == "omega-4"  # alias
    assert len(registry.list()) == 2


def test_from_dicts_rejects_negative_price():
    bad = dict(MINIMAL, inference={"pricing": {"input_per_million": -1.0}})
    with pytest.raises(ValueError):
        ModelRegistry.from_dicts([bad])


def test_from_dicts_rejects_empty_modalities():
    bad = dict(MINIMAL, inference={"input_modalities": [""]})
    with pytest.raises(ValueError):
        ModelRegistry.from_dicts([bad])


def test_from_dicts_rejects_missing_id():
    with pytest.raises(KeyError):
        ModelRegistry.from_dicts([{"provider": "p", "api_family": "openai-chat"}])


# ─── discover() ──────────────────────────────────────────────────────

class _FakeEntryPoint:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def load(self):
        return self._fn


def _patch_entry_points(monkeypatch, eps):
    import lm15.models as models_module  # noqa: F401
    import importlib.metadata

    def fake_entry_points(*, group):
        assert group == "lm15.model_catalogs"
        return eps

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    # discover() does `from importlib.metadata import entry_points` at call
    # time, which re-reads the module attribute patched above.


def test_discover_empty_group(monkeypatch):
    _patch_entry_points(monkeypatch, [])
    registry = ModelRegistry.discover()
    assert registry.list() == ()


def test_discover_sorted_first_wins(monkeypatch):
    a_model = dict(MINIMAL, extensions={"from": "a"})
    b_model = dict(MINIMAL, extensions={"from": "b"})
    other = dict(MINIMAL, id="other-1")
    eps = [
        _FakeEntryPoint("zeta", lambda: [b_model, other]),
        _FakeEntryPoint("alpha", lambda: [a_model]),
    ]
    _patch_entry_points(monkeypatch, eps)
    registry = ModelRegistry.discover()
    # alpha sorts before zeta, so alpha's copy of tiny-1 wins.
    assert registry.get("exampleai", "tiny-1").extensions == {"from": "a"}
    assert registry.get("exampleai", "other-1") is not None
    assert len(registry.list()) == 2


def test_discover_failing_catalog_warns_and_continues(monkeypatch):
    def boom():
        raise RuntimeError("catalog exploded")

    eps = [
        _FakeEntryPoint("bad", boom),
        _FakeEntryPoint("good", lambda: [MINIMAL]),
    ]
    _patch_entry_points(monkeypatch, eps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry = ModelRegistry.discover()
    assert any("bad" in str(w.message) for w in caught)
    assert registry.get("exampleai", "tiny-1") is not None


def test_discover_invalid_dict_skips_catalog(monkeypatch):
    bad = dict(MINIMAL, inference={"pricing": {"input_per_million": -5}})
    eps = [_FakeEntryPoint("junk", lambda: [bad])]
    _patch_entry_points(monkeypatch, eps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry = ModelRegistry.discover()
    assert any("junk" in str(w.message) for w in caught)
    assert registry.list() == ()
