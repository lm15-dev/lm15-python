"""Serde coverage for the May features: CacheConfig and ContinuationState kinds.

Round-trips, the omission rule, and rejection of invalid vocabulary values
for the "cache_config" and "continuation_state" serde kinds, plus
Config.cache wiring inside config_to_dict/config_from_dict.
"""

from __future__ import annotations

import pytest

from lm15.serde import (
    cache_config_from_dict,
    cache_config_to_dict,
    config_from_dict,
    config_to_dict,
    continuation_from_dict,
    continuation_to_dict,
)
from lm15.types import CacheConfig, Config, ContinuationState


# ── CacheConfig ──────────────────────────────────────────────────────

def test_cache_config_full_round_trip() -> None:
    cfg = CacheConfig(mode="auto", retention="long", key="session-42", prefix_until_index=3)
    d = cache_config_to_dict(cfg)
    assert d == {"mode": "auto", "retention": "long", "key": "session-42", "prefix_until_index": 3}
    assert cache_config_from_dict(d) == cfg


def test_cache_config_minimal_omits_empty_optionals() -> None:
    d = cache_config_to_dict(CacheConfig())
    assert d == {"mode": "auto"}
    assert cache_config_from_dict(d) == CacheConfig()


def test_cache_config_prefix_zero_is_preserved() -> None:
    cfg = CacheConfig(prefix_until_index=0)
    d = cache_config_to_dict(cfg)
    assert d == {"mode": "auto", "prefix_until_index": 0}
    assert cache_config_from_dict(d) == cfg


def test_cache_config_mode_defaults_to_auto() -> None:
    assert cache_config_from_dict({}) == CacheConfig(mode="auto")


def test_cache_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unsupported cache mode"):
        cache_config_from_dict({"mode": "always"})


def test_cache_config_rejects_invalid_retention() -> None:
    with pytest.raises(ValueError, match="unsupported cache retention"):
        cache_config_from_dict({"mode": "auto", "retention": "forever"})


def test_cache_config_off_with_key_rejected_by_type() -> None:
    # Type-level invariant surfaces through deserialization too.
    with pytest.raises(ValueError):
        cache_config_from_dict({"mode": "off", "key": "k"})


# ── Config.cache wiring ──────────────────────────────────────────────

def test_config_with_cache_round_trip() -> None:
    cfg = Config(max_tokens=1024, cache=CacheConfig(retention="short", key="tenant-a"))
    d = config_to_dict(cfg)
    assert d["cache"] == {"mode": "auto", "retention": "short", "key": "tenant-a"}
    assert config_from_dict(d) == cfg


def test_config_without_cache_omits_key() -> None:
    d = config_to_dict(Config(max_tokens=8))
    assert "cache" not in d
    assert config_from_dict(d).cache is None


# ── ContinuationState ────────────────────────────────────────────────

def test_continuation_state_round_trip() -> None:
    state = ContinuationState(provider="openai", kind="response_id", data={"id": "resp_123"})
    d = continuation_to_dict(state)
    assert d == {"provider": "openai", "kind": "response_id", "data": {"id": "resp_123"}}
    assert continuation_from_dict(d) == state


def test_continuation_state_opaque_data_round_trips_verbatim() -> None:
    # Empties inside opaque payloads are user data, not noise (serde-rules.md).
    data = {"name": "", "nested": {"empty": {}}, "items": []}
    state = ContinuationState(provider="gemini", kind="cached_content", data=data)
    assert continuation_from_dict(continuation_to_dict(state)).data == data


def test_continuation_state_missing_data_defaults_empty() -> None:
    state = continuation_from_dict({"provider": "anthropic", "kind": "signature"})
    assert state.data == {}


# ── Kind registration ────────────────────────────────────────────────

def test_kinds_registered_in_vet() -> None:
    from lm15.vet import KIND_SERDE

    assert "cache_config" in KIND_SERDE
    assert "continuation_state" in KIND_SERDE
