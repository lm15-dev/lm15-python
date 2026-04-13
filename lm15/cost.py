"""Cost estimation from Usage + models.dev pricing data."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .model_catalog import ModelSpec
from .types import Usage


# ── Global cost index (populated by configure(track_costs=True)) ──────

_cost_index: dict[str, ModelSpec] | None = None
_cost_lock = threading.Lock()


def _hydrate_cost_index() -> dict[str, ModelSpec]:
    """Fetch models.dev and build {model_id: ModelSpec} index."""
    from .model_catalog import fetch_models_dev

    specs = fetch_models_dev()
    return {s.id: s for s in specs if s.raw.get("cost")}


def enable_cost_tracking() -> None:
    """Fetch pricing data and enable automatic cost tracking.

    Called by ``lm15.configure(track_costs=True)``.  After this,
    ``Result.cost`` and ``Model.total_cost`` work without extra setup.
    """
    global _cost_index
    with _cost_lock:
        _cost_index = _hydrate_cost_index()


def disable_cost_tracking() -> None:
    """Disable cost tracking and free the pricing index."""
    global _cost_index
    with _cost_lock:
        _cost_index = None


def get_cost_index() -> dict[str, ModelSpec] | None:
    """Return the global cost index, or ``None`` if not enabled."""
    return _cost_index


def lookup_cost(
    model: str,
    usage: Usage,
    provider: str | None = None,
) -> "CostBreakdown | None":
    """Estimate cost using the global index.  Returns ``None`` if
    cost tracking is not enabled or the model is not found."""
    index = _cost_index
    if index is None:
        return None
    spec = index.get(model)
    if spec is None:
        return None
    return estimate_cost(usage, spec)


@dataclass(slots=True, frozen=True)
class CostBreakdown:
    """Itemised cost breakdown in US dollars."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    reasoning: float = 0.0
    input_audio: float = 0.0
    output_audio: float = 0.0
    total: float = 0.0

    def __repr__(self) -> str:
        parts = [f"${self.total:.6f}"]
        detail = []
        if self.input:
            detail.append(f"input=${self.input:.6f}")
        if self.output:
            detail.append(f"output=${self.output:.6f}")
        if self.cache_read:
            detail.append(f"cache_read=${self.cache_read:.6f}")
        if self.cache_write:
            detail.append(f"cache_write=${self.cache_write:.6f}")
        if self.reasoning:
            detail.append(f"reasoning=${self.reasoning:.6f}")
        if self.input_audio:
            detail.append(f"input_audio=${self.input_audio:.6f}")
        if self.output_audio:
            detail.append(f"output_audio=${self.output_audio:.6f}")
        if detail:
            parts.append(f"({', '.join(detail)})")
        return " ".join(parts)


# Providers where input_tokens already excludes cached tokens
# (cache counts are additive to get total input).
_ADDITIVE_CACHE_PROVIDERS = frozenset({"anthropic"})

# Providers where reasoning_tokens is a *separate* count (not a
# subset of output_tokens).
_SEPARATE_REASONING_PROVIDERS = frozenset({"gemini", "google"})


def _per_token(rate_per_million: float | None) -> float:
    """Convert $/M-tokens to $/token."""
    if rate_per_million is None:
        return 0.0
    return rate_per_million / 1_000_000


def estimate_cost(
    usage: Usage,
    spec: ModelSpec | dict[str, Any],
    *,
    provider: str | None = None,
) -> CostBreakdown:
    """Estimate the cost of a single request from its ``Usage``.

    Parameters
    ----------
    usage:
        Token counts returned by the provider.
    spec:
        A ``ModelSpec`` (from ``fetch_models_dev``) or a raw cost dict
        like ``{"input": 3.0, "output": 15.0, "cache_read": 1.5, ...}``.
        All rates are in **$/million tokens**.
    provider:
        Provider name (``"openai"``, ``"anthropic"``, ``"gemini"``).
        Required when *spec* is a plain dict so the function can
        correctly interpret the token-counting semantics.
        Ignored when *spec* is a ``ModelSpec`` (uses ``spec.provider``).

    Returns
    -------
    CostBreakdown
        Itemised costs in US dollars.

    Notes
    -----
    Token-counting semantics vary by provider:

    * **OpenAI / Gemini** — ``input_tokens`` is the *total* (including
      cached tokens).  ``cache_read_tokens`` is a *subset*.
    * **Anthropic** — ``input_tokens`` is *non-cached only*.
      ``cache_read_input_tokens`` and ``cache_creation_input_tokens``
      are *additive* to get the true total.

    Similarly for reasoning/thinking tokens:

    * **OpenAI** — ``reasoning_tokens`` is a *subset* of ``output_tokens``.
    * **Gemini** — ``thoughtsTokenCount`` is *separate* from
      ``candidatesTokenCount``.
    * **Anthropic** — not reported.
    """
    if isinstance(spec, ModelSpec):
        cost = spec.raw.get("cost") or {}
        provider = spec.provider
    else:
        cost = spec
        if provider is None:
            raise ValueError(
                "provider is required when spec is a plain dict\n\n"
                "  Pass the provider name so token semantics are applied correctly:\n"
                "    estimate_cost(usage, cost_dict, provider='openai')\n"
            )

    r_input = _per_token(cost.get("input"))
    r_output = _per_token(cost.get("output"))
    r_cache_read = _per_token(cost.get("cache_read"))
    r_cache_write = _per_token(cost.get("cache_write"))
    r_reasoning = _per_token(cost.get("reasoning"))
    r_input_audio = _per_token(cost.get("input_audio"))
    r_output_audio = _per_token(cost.get("output_audio"))

    cache_read = usage.cache_read_tokens or 0
    cache_write = usage.cache_write_tokens or 0
    reasoning = usage.reasoning_tokens or 0
    input_audio = usage.input_audio_tokens or 0
    output_audio = usage.output_audio_tokens or 0

    # --- Input tokens ---
    if provider in _ADDITIVE_CACHE_PROVIDERS:
        # Anthropic: input_tokens is non-cached; cache counts are additive.
        text_input = usage.input_tokens - input_audio
    else:
        # OpenAI / Gemini: input_tokens is total; subtract cached subsets.
        text_input = usage.input_tokens - cache_read - cache_write - input_audio

    text_input = max(text_input, 0)

    # --- Output tokens ---
    if provider in _SEPARATE_REASONING_PROVIDERS:
        # Gemini: thoughtsTokenCount is separate from candidatesTokenCount.
        text_output = usage.output_tokens - output_audio
    else:
        # OpenAI: reasoning_tokens is a subset of output_tokens.
        text_output = usage.output_tokens - reasoning - output_audio

    text_output = max(text_output, 0)

    c_input = text_input * r_input
    c_output = text_output * r_output
    c_cache_read = cache_read * r_cache_read
    c_cache_write = cache_write * r_cache_write
    c_reasoning = reasoning * r_reasoning
    c_input_audio = input_audio * r_input_audio
    c_output_audio = output_audio * r_output_audio

    total = c_input + c_output + c_cache_read + c_cache_write + c_reasoning + c_input_audio + c_output_audio

    return CostBreakdown(
        input=c_input,
        output=c_output,
        cache_read=c_cache_read,
        cache_write=c_cache_write,
        reasoning=c_reasoning,
        input_audio=c_input_audio,
        output_audio=c_output_audio,
        total=total,
    )
