from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from .protocols import Capabilities


@dataclass(slots=True, frozen=True)
class ModelSpec:
    id: str
    provider: str
    context_window: int | None
    max_output: int | None
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    tool_call: bool
    structured_output: bool
    reasoning: bool
    raw: dict

    def to_capabilities(self) -> Capabilities:
        features = set()
        if self.tool_call:
            features.add("tools")
        if self.structured_output:
            features.add("json_output")
        if self.reasoning:
            features.add("reasoning")
        return Capabilities(
            input_modalities=frozenset(self.input_modalities),
            output_modalities=frozenset(self.output_modalities),
            features=frozenset(features),
        )


def fetch_models_dev(timeout: float = 20.0) -> list[ModelSpec]:
    req = urllib.request.Request(
        "https://models.dev/api.json",
        headers={"User-Agent": "lm15"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())

    out: list[ModelSpec] = []
    # models.dev may nest providers under a "providers" key or
    # expose them directly at the top level.
    providers = data.get("providers") or data
    for provider_id, provider_payload in providers.items():
        if not isinstance(provider_payload, dict) or "models" not in provider_payload:
            continue
        models = provider_payload.get("models", {})
        for model_id, m in models.items():
            limit = m.get("limit", {})
            modalities = m.get("modalities", {})
            out.append(
                ModelSpec(
                    id=model_id,
                    provider=provider_id,
                    context_window=limit.get("context"),
                    max_output=limit.get("output"),
                    input_modalities=tuple(modalities.get("input", [])),
                    output_modalities=tuple(modalities.get("output", [])),
                    tool_call=bool(m.get("tool_call", False)),
                    structured_output=bool(m.get("structured_output", False)),
                    reasoning=bool(m.get("reasoning", False)),
                    raw=m,
                )
            )
    return out


def build_provider_model_index(specs: list[ModelSpec]) -> dict[str, dict[str, ModelSpec]]:
    out: dict[str, dict[str, ModelSpec]] = {}
    for s in specs:
        out.setdefault(s.provider, {})[s.id] = s
    return out
