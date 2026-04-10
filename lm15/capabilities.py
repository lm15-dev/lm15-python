from __future__ import annotations

from dataclasses import dataclass

from .errors import UnsupportedModelError
from .model_catalog import ModelSpec
from .protocols import Capabilities


@dataclass(slots=True, frozen=True)
class ModelCapabilities:
    provider: str
    pattern: str
    caps: Capabilities


REGISTRY: tuple[ModelCapabilities, ...] = (
    ModelCapabilities(
        provider="anthropic",
        pattern="claude",
        caps=Capabilities(
            input_modalities=frozenset({"text", "image", "document"}),
            output_modalities=frozenset({"text"}),
            features=frozenset({"streaming", "tools", "reasoning"}),
        ),
    ),
    ModelCapabilities(
        provider="gemini",
        pattern="gemini",
        caps=Capabilities(
            input_modalities=frozenset({"text", "image", "audio", "video", "document"}),
            output_modalities=frozenset({"text"}),
            features=frozenset({"streaming", "tools", "json_output", "live"}),
        ),
    ),
    ModelCapabilities(
        provider="openai",
        pattern="gpt",
        caps=Capabilities(
            input_modalities=frozenset({"text", "image", "audio", "video", "document"}),
            output_modalities=frozenset({"text", "audio"}),
            features=frozenset({"streaming", "tools", "json_output", "reasoning", "live", "embeddings"}),
        ),
    ),
)


class CapabilityResolver:
    def __init__(self):
        self._model_index: dict[str, ModelSpec] = {}

    def hydrate(self, specs: list[ModelSpec]) -> None:
        self._model_index = {s.id: s for s in specs}

    def resolve_provider(self, model: str) -> str:
        if model in self._model_index:
            return self._model_index[model].provider
        lower = model.lower()
        for item in REGISTRY:
            if lower.startswith(item.pattern):
                return item.provider
        raise UnsupportedModelError(
            f"unable to resolve provider for model '{model}'\n\n"
            f"  To fix, do one of:\n"
            f"    1. Use provider= to specify the provider explicitly:\n"
            f"       lm15.call('{model}', ..., provider='openai')\n"
            f"    2. Check available models with lm15.models()\n"
            f"    3. Verify the model name is correct (common prefixes: gpt-, claude-, gemini-)\n"
        )

    def resolve_capabilities(self, model: str) -> Capabilities:
        spec = self._model_index.get(model)
        if spec:
            return spec.to_capabilities()
        lower = model.lower()
        for item in REGISTRY:
            if lower.startswith(item.pattern):
                return item.caps
        return REGISTRY[-1].caps

    def known_models(self) -> tuple[str, ...]:
        return tuple(self._model_index.keys())


_DEFAULT_RESOLVER = CapabilityResolver()


def hydrate_with_specs(specs: list[ModelSpec]) -> None:
    _DEFAULT_RESOLVER.hydrate(specs)


def resolve_provider(model: str) -> str:
    return _DEFAULT_RESOLVER.resolve_provider(model)


def resolve_capabilities(model: str) -> Capabilities:
    return _DEFAULT_RESOLVER.resolve_capabilities(model)


def known_models() -> tuple[str, ...]:
    return _DEFAULT_RESOLVER.known_models()
