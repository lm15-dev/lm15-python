from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class EndpointSupport:
    complete: bool = True
    stream: bool = True
    live: bool = False
    embeddings: bool = False
    files: bool = False
    batches: bool = False
    images: bool = False
    audio: bool = False
    responses_api: bool = False


@dataclass(slots=True, frozen=True)
class ProviderManifest:
    provider: str
    supports: EndpointSupport
    auth_modes: tuple[str, ...] = field(default_factory=tuple)
    enterprise_variants: tuple[str, ...] = field(default_factory=tuple)
    env_keys: tuple[str, ...] = field(default_factory=tuple)
    """Environment variable names this provider reads for API keys.

    First match wins. Example: ``("OPENAI_API_KEY",)``.
    Used by ``env=`` file loading and ``api_key=`` routing.
    """
