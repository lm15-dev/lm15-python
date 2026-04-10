from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import resolve_provider
from .client import UniversalLM
from .discovery import models as _models, providers_info as _providers_info
from .factory import build_default, providers as _providers
from .model import Model
from .types import LMRequest, LMResponse, Part


_defaults: dict[str, Any] = {}


def configure(
    *,
    env: str | None = None,
    api_key: str | dict[str, str] | None = None,
) -> None:
    """Set module-level defaults so you don't repeat them on every call.

    >>> lm15.configure(env=".env")
    >>> resp = lm15.call("gpt-4.1-mini", "Hello.")  # no env= needed

    Per-call ``env=`` or ``api_key=`` overrides the default for that call only.
    Call ``lm15.configure()`` with no arguments to clear defaults.
    """
    _defaults.clear()
    if env is not None:
        _defaults["env"] = env
    if api_key is not None:
        _defaults["api_key"] = api_key


def _resolve(key: str, explicit: Any) -> Any:
    """Return *explicit* if provided, otherwise the configured default."""
    if explicit is not None:
        return explicit
    return _defaults.get(key)


def model(
    model_name: str,
    *,
    system: str | None = None,
    tools=None,
    provider: str | None = None,
    retries: int = 0,
    cache: bool | dict = False,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> Model:
    lm = build_default(
        api_key=_resolve("api_key", api_key),
        provider_hint=provider,
        env=_resolve("env", env),
    )
    return Model(
        lm=lm,
        model=model_name,
        system=system,
        tools=list(tools or []),
        provider=provider,
        retries=retries,
        cache=cache,
        prompt_caching=prompt_caching,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def prepare(
    model_name: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages=None,
    system: str | None = None,
    tools=None,
    reasoning=None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop=None,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> LMRequest:
    """Build the LMRequest without sending it.

    Useful for inspecting exactly what would be sent — tool schemas,
    messages, config — before making the API call.

    >>> req = lm15.prepare("gpt-4.1-mini", "Hello.", tools=[get_weather])
    >>> print(req.tools)   # inspect the inferred tool schema
    >>> print(req.messages) # inspect the constructed messages
    >>> resp = lm15.send(req)  # send it when ready
    """
    m = model(model_name, provider=provider, prompt_caching=prompt_caching, system=system, api_key=api_key, env=env)
    return m.prepare(
        prompt,
        messages=messages,
        tools=tools,
        reasoning=reasoning,
        prefill=prefill,
        output=output,
        prompt_caching=prompt_caching,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
    )


def send(
    request: LMRequest,
    *,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> LMResponse:
    """Send a prepared LMRequest.

    Pair with ``prepare()`` for inspect-then-send workflows:

    >>> req = lm15.prepare("gpt-4.1-mini", "Hello.")
    >>> resp = lm15.send(req)
    """
    resolved_provider = provider or resolve_provider(request.model)
    lm = build_default(
        api_key=_resolve("api_key", api_key),
        provider_hint=resolved_provider,
        env=_resolve("env", env),
    )
    return lm.complete(request, provider=resolved_provider)


def call(
    model_name: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages=None,
    system: str | None = None,
    tools=None,
    reasoning=None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop=None,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
):
    m = model(model_name, provider=provider, prompt_caching=prompt_caching, system=system, api_key=api_key, env=env)
    return m(
        prompt,
        messages=messages,
        tools=tools,
        reasoning=reasoning,
        prefill=prefill,
        output=output,
        prompt_caching=prompt_caching,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        provider=provider,
    )


def stream(
    model_name: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages=None,
    system: str | None = None,
    tools=None,
    reasoning=None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop=None,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
):
    m = model(model_name, provider=provider, prompt_caching=prompt_caching, system=system, api_key=api_key, env=env)
    return m.stream(
        prompt,
        messages=messages,
        tools=tools,
        reasoning=reasoning,
        prefill=prefill,
        output=output,
        prompt_caching=prompt_caching,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        provider=provider,
    )


def providers() -> dict[str, tuple[str, ...]]:
    return _providers()


def providers_info(
    *,
    live: bool = True,
    refresh: bool = False,
    timeout: float = 5.0,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
):
    return _providers_info(
        live=live, refresh=refresh, timeout=timeout,
        api_key=_resolve("api_key", api_key),
        env=_resolve("env", env),
    )


def models(
    *,
    provider: str | None = None,
    live: bool = True,
    refresh: bool = False,
    timeout: float = 5.0,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
    supports: set[str] | None = None,
    input_modalities: set[str] | None = None,
    output_modalities: set[str] | None = None,
):
    return _models(
        provider=provider,
        live=live,
        refresh=refresh,
        timeout=timeout,
        api_key=_resolve("api_key", api_key),
        env=_resolve("env", env),
        supports=supports,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
    )





def upload(
    model_name: str,
    path: str | Path | bytes,
    *,
    media_type: str | None = None,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> Part:
    p = str(path) if isinstance(path, Path) else path
    resolved = provider or resolve_provider(model_name)
    m = model(
        model_name, provider=resolved,
        api_key=_resolve("api_key", api_key),
        env=_resolve("env", env),
    )
    return m.upload(p, media_type=media_type)
