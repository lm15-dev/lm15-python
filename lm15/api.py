from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import resolve_provider
from .client import UniversalLM
from .discovery import models as _models, providers_info as _providers_info
from .factory import build_default, providers as _providers
from .model import Model
from .result import AsyncResult, Result
from .types import LMRequest, Part, Tool


_defaults: dict[str, Any] = {}
_client_cache: dict[tuple, Any] = {}


def configure(
    *,
    env: str | None = None,
    api_key: str | dict[str, str] | None = None,
) -> None:
    """Set module-level defaults so you don't repeat them on every call."""
    _defaults.clear()
    _client_cache.clear()
    if env is not None:
        _defaults["env"] = env
    if api_key is not None:
        _defaults["api_key"] = api_key


def _resolve(key: str, explicit: Any) -> Any:
    if explicit is not None:
        return explicit
    return _defaults.get(key)


def _get_client(
    api_key: str | dict[str, str] | None = None,
    provider_hint: str | None = None,
    env: str | None = None,
) -> UniversalLM:
    if isinstance(api_key, dict):
        ak = tuple(sorted(api_key.items()))
    else:
        ak = api_key
    cache_key = (ak, provider_hint, env)

    client = _client_cache.get(cache_key)
    if client is not None:
        return client

    client = build_default(api_key=api_key, provider_hint=provider_hint, env=env)
    _client_cache[cache_key] = client
    return client


def model(
    model_name: str,
    *,
    system: str | None = None,
    tools=None,
    on_tool_call=None,
    provider: str | None = None,
    retries: int = 0,
    cache: bool | dict = False,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_tool_rounds: int = 8,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> Model:
    lm = _get_client(
        api_key=_resolve("api_key", api_key),
        provider_hint=provider,
        env=_resolve("env", env),
    )
    return Model(
        lm=lm,
        model=model_name,
        system=system,
        tools=list(tools or []),
        on_tool_call=on_tool_call,
        provider=provider,
        retries=retries,
        cache=cache,
        prompt_caching=prompt_caching,
        temperature=temperature,
        max_tokens=max_tokens,
        max_tool_rounds=max_tool_rounds,
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
    m = model(
        model_name,
        provider=provider,
        prompt_caching=prompt_caching,
        system=system,
        api_key=api_key,
        env=env,
    )
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
) -> Result:
    resolved_provider = provider or resolve_provider(request.model)
    lm = _get_client(
        api_key=_resolve("api_key", api_key),
        provider_hint=resolved_provider,
        env=_resolve("env", env),
    )

    callable_registry = {
        tool.name: tool.fn
        for tool in request.tools
        if isinstance(tool, Tool) and tool.fn is not None and callable(tool.fn)
    }

    def start_stream(req: LMRequest):
        return lm.stream(req, provider=resolved_provider)

    return Result(
        request=request,
        start_stream=start_stream,
        callable_registry=callable_registry,
    )


def call(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages=None,
    system: str | None = None,
    tools=None,
    on_tool_call=None,
    reasoning=None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop=None,
    max_tool_rounds: int = 8,
    retries: int = 0,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> Result:
    m = globals()["model"](
        model,
        provider=provider,
        prompt_caching=prompt_caching,
        system=system,
        tools=list(tools or []),
        on_tool_call=on_tool_call,
        retries=retries,
        max_tool_rounds=max_tool_rounds,
        api_key=api_key,
        env=env,
    )
    return m.call(
        prompt,
        messages=messages,
        tools=tools,
        on_tool_call=on_tool_call,
        reasoning=reasoning,
        prefill=prefill,
        output=output,
        prompt_caching=prompt_caching,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        max_tool_rounds=max_tool_rounds,
        provider=provider,
    )


def acall(*args, **kwargs) -> AsyncResult:
    return AsyncResult(call, *args, **kwargs)


def stream(*args, **kwargs) -> Result:
    return call(*args, **kwargs)


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
        live=live,
        refresh=refresh,
        timeout=timeout,
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
        model_name,
        provider=resolved,
        api_key=_resolve("api_key", api_key),
        env=_resolve("env", env),
    )
    return m.upload(p, media_type=media_type)
