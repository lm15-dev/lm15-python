"""
lm15.router — Minimalist model-string router.

The router is a lookup table you can read, not a framework.  Three
resolution rungs, in fixed order, with no configuration of the order
itself:

  1. explicit prefix   ``"openai:gpt-4.1-mini"``       -> source="prefix"
  2. catalog           match against ``registry.list()`` -> source="catalog"
                       entries by id or alias; exact-id matches beat
                       alias matches; multiple providers (or multiple
                       same-provider entries) raise AmbiguousModelError
                       (only if you passed a registry; catalogs are
                       opt-in via ``ModelRegistry.discover()``)
  3. built-in rules    ``DEFAULT_RULES`` prefix match  -> source="rule"

Nothing else.  No plugins, no callbacks, no fallback chains.

Model-string grammar
--------------------
A model string is split on the FIRST ``:``.  If the head is a known
provider string (a key of :data:`ADAPTERS`), the remainder is the model
id sent on the wire.  Otherwise the whole string (colons and all) is
treated as a bare model id and falls through to the catalog and rule
rungs.  Consequence: a fine-tune id like ``ft:gpt-4.1:org`` needs the
explicit form ``openai:ft:gpt-4.1:org``.

Credentials
-----------
``resolve()`` is pure: it touches no network and reads no secret values
(it records WHICH env var would be used, never the value).  ``lm()``
reads the key — first from ``RouterConfig.api_keys`` (explicit,
repr-suppressed), then from the env mapping via the provider's existing
``ProviderManifest.env_keys`` (first hit wins).  OAuth providers
(``claude-code``, ``openai-codex``) declare no env keys and pass through
to their self-resolving constructors.

The direct LM classes remain first-class; the router is just the
recommended front door.  Needing a custom ``base_url``/transport/compat
(ollama, vllm, azure, openrouter) is the documented escape hatch:
``lm()`` returns the ordinary provider LM — keep it and configure it
yourself next time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Iterator, Mapping

from .errors import LM15Error, NotConfiguredError
from .models import ModelInfo, ModelRegistry
from .providers import (
    AnthropicLM,
    AsyncAnthropicLM,
    AsyncClaudeCodeLM,
    AsyncGeminiLM,
    AsyncOpenAIChatLM,
    AsyncOpenAICodexLM,
    AsyncOpenAILM,
    ClaudeCodeLM,
    GeminiLM,
    OpenAIChatLM,
    OpenAICodexLM,
    OpenAILM,
)
from .types import Request, Response, StreamEvent

__all__ = [
    "RouteRule",
    "DEFAULT_RULES",
    "ADAPTERS",
    "ASYNC_ADAPTERS",
    "Resolution",
    "RouterConfig",
    "RouterError",
    "UnknownModelError",
    "AmbiguousModelError",
    "MissingCredentialError",
    "LMRouter",
    "AsyncLMRouter",
]


# ---------------------------------------------------------------- rules ----


@dataclass(frozen=True, slots=True)
class RouteRule:
    """Maps a model-id prefix to a provider.  That's all a rule is.

    ``note`` is a short human rationale surfaced in docs and
    :meth:`Resolution.describe` output.
    """

    prefix: str
    provider: str
    note: str = ""


# The complete built-in knowledge of the router.  Inspectable, printable,
# overridable by passing rules=... in RouterConfig.  First match wins.
# This table is a convenience, not a registry of truth: new model
# families need a release (or a catalog, or the provider: prefix).
DEFAULT_RULES: tuple[RouteRule, ...] = (
    RouteRule("claude-", "anthropic", note="Anthropic Claude family"),
    RouteRule("gpt-", "openai", note="OpenAI GPT family (Responses API; use openai_chat: for Chat Completions)"),
    RouteRule("o1", "openai", note="OpenAI o1 reasoning family"),
    RouteRule("o3", "openai", note="OpenAI o3 reasoning family"),
    RouteRule("o4", "openai", note="OpenAI o4 reasoning family"),
    RouteRule("gemini-", "gemini", note="Google Gemini family"),
)


# provider string -> LM class.  Hardcoded, exported, inspectable.
# Values are the *sync* classes; AsyncLMRouter uses ASYNC_ADAPTERS.
ADAPTERS: Mapping[str, type] = {
    "openai": OpenAILM,
    "openai_chat": OpenAIChatLM,
    "anthropic": AnthropicLM,
    "gemini": GeminiLM,
    "claude-code": ClaudeCodeLM,
    "openai-codex": OpenAICodexLM,
}

ASYNC_ADAPTERS: Mapping[str, type] = {
    "openai": AsyncOpenAILM,
    "openai_chat": AsyncOpenAIChatLM,
    "anthropic": AsyncAnthropicLM,
    "gemini": AsyncGeminiLM,
    "claude-code": AsyncClaudeCodeLM,
    "openai-codex": AsyncOpenAICodexLM,
}

# Providers whose constructors self-resolve local OAuth credentials and
# therefore take no api_key from the router.
_OAUTH_PROVIDERS: frozenset[str] = frozenset({"claude-code", "openai-codex"})


# --------------------------------------------------------------- errors ----


class RouterError(LM15Error):
    """Base for all routing failures."""

    default_code = "router"


class UnknownModelError(RouterError):
    """No resolution rung matched the model string."""

    default_code = "unknown_model"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        rules_tried: tuple[RouteRule, ...] = (),
        catalog_searched: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.rules_tried = tuple(rules_tried)
        self.catalog_searched = catalog_searched
        super().__init__(message, **kwargs)


class AmbiguousModelError(RouterError):
    """Catalog matched the model id under more than one provider."""

    default_code = "ambiguous_model"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        providers: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        self.model = model
        self.providers = tuple(providers)
        self.candidates = self.providers  # alias: full candidate list
        super().__init__(message, **kwargs)


class MissingCredentialError(RouterError, NotConfiguredError):
    """Provider resolved but no API key was found.

    Subclasses the existing :class:`lm15.errors.NotConfiguredError` —
    semantically the credential case IS not-configured — so existing
    ``except NotConfiguredError`` handlers keep working.  Carries
    ``provider`` and ``env_keys`` straight from the ProviderManifest.
    """

    default_code = "not_configured"


# ----------------------------------------------------------- resolution ----


@dataclass(frozen=True, slots=True)
class Resolution:
    """The complete answer to "how did you route this string".

    ``resolve()`` returning this IS the explain() method — there is no
    separate one.
    """

    requested: str                  # verbatim input string
    model: str                      # id sent on the wire (prefix stripped)
    provider: str                   # canonical provider string
    adapter: str                    # LM class name, e.g. "AnthropicLM"
    source: str                     # "prefix" | "catalog" | "rule"
    rule: RouteRule | None = None   # the matching rule when source == "rule"
    env_key: str | None = None      # env var the key would be read from;
                                    # None for OAuth providers or when an
                                    # explicit api_keys entry overrides env
    model_info: ModelInfo | None = None  # catalog metadata when source == "catalog"

    def describe(self) -> str:
        """One-paragraph human-readable explanation of this resolution."""
        parts = [f"{self.requested!r} -> provider {self.provider!r} ({self.adapter})"]
        if self.source == "prefix":
            parts.append("via explicit provider prefix")
        elif self.source == "catalog":
            parts.append("via catalog match")
        elif self.source == "rule" and self.rule is not None:
            note = f" — {self.rule.note}" if self.rule.note else ""
            parts.append(f"via built-in rule prefix={self.rule.prefix!r}{note}")
        parts.append(f"wire model {self.model!r}")
        if self.env_key is not None:
            parts.append(f"key from ${self.env_key}")
        elif self.provider in _OAUTH_PROVIDERS:
            parts.append("local OAuth credential (no env key)")
        else:
            parts.append("key from explicit api_keys")
        return "; ".join(parts) + "."

    def __str__(self) -> str:
        return self.describe()


# ---------------------------------------------------------------- config ----


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Everything the router consults.  All explicit, nothing discovered
    behind your back.  Catalog use is opt-in: pass
    ``registry=ModelRegistry.discover()``.

    ``api_keys`` maps provider string -> key and beats env (repr-
    suppressed; lets hermetic tests pass ``env={}``).  ``env`` defaults
    to ``os.environ`` at lookup time.
    """

    registry: ModelRegistry | None = None
    rules: tuple[RouteRule, ...] = DEFAULT_RULES
    env: Mapping[str, str] | None = None
    api_keys: Mapping[str, str] | None = field(default=None, repr=False)


# ------------------------------------------------------------- internals ----


def _resolve(model: str, config: RouterConfig, adapters: Mapping[str, type]) -> Resolution:
    if not isinstance(model, str) or not model:
        raise UnknownModelError(
            "model must be a non-empty string", model=str(model),
            rules_tried=config.rules, catalog_searched=config.registry is not None,
        )

    requested = model

    # Rung 1: explicit provider prefix (split on FIRST colon).
    if ":" in model:
        head, rest = model.split(":", 1)
        if head in adapters and rest:
            return Resolution(
                requested=requested,
                model=rest,
                provider=head,
                adapter=adapters[head].__name__,
                source="prefix",
                env_key=_env_key_for(head, config, adapters),
            )

    # Rung 2: catalog (only if a registry was explicitly supplied).
    if config.registry is not None:
        matches = tuple(
            info
            for info in config.registry.list()
            if info.id == model or model in info.aliases
        )
        providers = tuple(dict.fromkeys(info.provider for info in matches))
        if len(providers) > 1:
            options = " or ".join(f'"{p}:{model}"' for p in providers)
            raise AmbiguousModelError(
                f"model {model!r} is offered by multiple providers: "
                f"{', '.join(providers)}. Fix: use the explicit form, e.g. "
                f"Request(model={providers[0] + ':' + model!r}) — options: {options}.",
                model=model,
                providers=providers,
            )
        if matches:
            # Exact-id matches beat alias matches; an alias must never
            # shadow an entry whose canonical id IS the requested string.
            exact = tuple(info for info in matches if info.id == model)
            narrowed = exact if exact else matches
            if len(narrowed) > 1:
                # Same provider (multi-provider was caught above), multiple
                # entries: never pick one by insertion order.
                ids = ", ".join(info.id for info in narrowed)
                raise AmbiguousModelError(
                    f"model {model!r} matches multiple catalog entries "
                    f"({ids}) under provider {narrowed[0].provider!r}. "
                    "Fix: request a canonical id directly.",
                    model=model,
                    providers=providers,
                )
            info = narrowed[0]
            if info.provider not in adapters:
                raise UnknownModelError(
                    f"model {model!r} resolved in the catalog to provider "
                    f"{info.provider!r}, but lm15 has no adapter for it. "
                    f"Known providers: {', '.join(sorted(adapters))}. "
                    "Construct a provider LM directly (e.g. OpenAIChatLM with "
                    "a custom base_url) for OpenAI-compatible servers.",
                    model=model,
                    rules_tried=config.rules,
                    catalog_searched=True,
                )
            return Resolution(
                requested=requested,
                model=info.id if model in info.aliases else model,
                provider=info.provider,
                adapter=adapters[info.provider].__name__,
                source="catalog",
                env_key=_env_key_for(info.provider, config, adapters),
                model_info=info,
            )

    # Rung 3: built-in prefix rules, first match wins.
    for rule in config.rules:
        if model.startswith(rule.prefix):
            if rule.provider not in adapters:
                raise UnknownModelError(
                    f"rule {rule!r} names provider {rule.provider!r}, which has "
                    f"no adapter. Known providers: {', '.join(sorted(adapters))}.",
                    model=model,
                    rules_tried=config.rules,
                    catalog_searched=config.registry is not None,
                )
            return Resolution(
                requested=requested,
                model=model,
                provider=rule.provider,
                adapter=adapters[rule.provider].__name__,
                source="rule",
                rule=rule,
                env_key=_env_key_for(rule.provider, config, adapters),
            )

    hints = [
        f"Use an explicit provider prefix, e.g. \"anthropic:{model}\" "
        f"(known providers: {', '.join(sorted(adapters))})."
    ]
    if config.registry is None:
        hints.append(
            "Or pass a model catalog: "
            "LMRouter(config=RouterConfig(registry=ModelRegistry.discover())) "
            "— install a catalog package such as 'aimo' first."
        )
    raise UnknownModelError(
        f"could not route model {model!r}: no provider prefix, "
        f"{'no catalog match' if config.registry is not None else 'no catalog supplied'}, "
        f"and none of the {len(config.rules)} built-in rules matched. "
        + " ".join(hints),
        model=model,
        rules_tried=config.rules,
        catalog_searched=config.registry is not None,
    )


def _env_key_for(provider: str, config: RouterConfig, adapters: Mapping[str, type]) -> str | None:
    """WHICH env var lm() would read for this provider (never the value).

    None when the provider is OAuth-based (declares no env keys) or when
    an explicit ``api_keys`` entry overrides env lookup entirely.
    """
    if config.api_keys is not None and provider in config.api_keys:
        return None
    env_keys = adapters[provider].manifest.env_keys
    if not env_keys:
        return None
    env = config.env if config.env is not None else os.environ
    for key in env_keys:
        if env.get(key):
            return key
    return env_keys[0]


def _build_lm(resolution: Resolution, config: RouterConfig, adapters: Mapping[str, type]):
    cls = adapters[resolution.provider]
    if resolution.provider in _OAUTH_PROVIDERS:
        return cls()  # self-resolving local OAuth constructor
    api_key = None
    if config.api_keys is not None:
        api_key = config.api_keys.get(resolution.provider)
    if api_key is None:
        env = config.env if config.env is not None else os.environ
        for key in cls.manifest.env_keys:
            value = env.get(key)
            if value:
                api_key = value
                break
    if not api_key:
        env_keys = cls.manifest.env_keys
        raise MissingCredentialError(
            f"no API key found for provider {resolution.provider!r}. "
            f"Set {' or '.join(env_keys)} in the environment, or pass "
            f"RouterConfig(api_keys={{{resolution.provider!r}: \"...\"}}).",
            provider=resolution.provider,
            env_keys=env_keys,
        )
    return cls(api_key=api_key)


def _routed_request(request: Request, resolution: Resolution) -> Request:
    if request.model == resolution.model:
        return request
    return replace(request, model=resolution.model)


# ---------------------------------------------------------------- router ----


class LMRouter:
    """Routes model strings to provider LMs.

    Four methods, no state you can't see: config is frozen; the only
    mutation is an LM cache keyed by provider (one LM per provider,
    built lazily, reused).
    """

    def __init__(self, config: RouterConfig = RouterConfig()) -> None:
        self.config = config
        self._lms: dict[str, object] = {}

    _adapters: Mapping[str, type] = ADAPTERS

    def resolve(self, model: str) -> Resolution:
        """Pure lookup; touches no network and reads no secret values.

        Raises UnknownModelError / AmbiguousModelError.  This IS the
        explain() method — there is no separate one.
        """
        return _resolve(model, self.config, self._adapters)

    def lm(self, model: str):
        """resolve(), then construct-or-reuse the provider LM.

        Raises MissingCredentialError when no key is found.  The
        returned LM is an ordinary OpenAILM/AnthropicLM/... — the escape
        hatch is built in: keep it, configure transports yourself.
        """
        resolution = self.resolve(model)
        lm = self._lms.get(resolution.provider)
        if lm is None:
            lm = _build_lm(resolution, self.config, self._adapters)
            self._lms[resolution.provider] = lm
        return lm

    def complete(self, request: Request) -> Response:
        resolution = self.resolve(request.model)
        return self.lm(request.model).complete(_routed_request(request, resolution))

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        resolution = self.resolve(request.model)
        return self.lm(request.model).stream(_routed_request(request, resolution))


class AsyncLMRouter:
    """Async mirror of :class:`LMRouter`.

    Same four methods; ``lm()`` returns Async* mirrors; ``complete`` is
    a coroutine and ``stream`` returns an async iterator.  ``resolve()``
    stays sync (it is pure).
    """

    def __init__(self, config: RouterConfig = RouterConfig()) -> None:
        self.config = config
        self._lms: dict[str, object] = {}

    _adapters: Mapping[str, type] = ASYNC_ADAPTERS

    def resolve(self, model: str) -> Resolution:
        return _resolve(model, self.config, self._adapters)

    def lm(self, model: str):
        resolution = self.resolve(model)
        lm = self._lms.get(resolution.provider)
        if lm is None:
            lm = _build_lm(resolution, self.config, self._adapters)
            self._lms[resolution.provider] = lm
        return lm

    async def complete(self, request: Request) -> Response:
        resolution = self.resolve(request.model)
        return await self.lm(request.model).complete(_routed_request(request, resolution))

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        resolution = self.resolve(request.model)
        return self.lm(request.model).stream(_routed_request(request, resolution))
