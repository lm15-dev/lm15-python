from __future__ import annotations

from pathlib import Path

from .capabilities import resolve_provider
from .factory import build_default
from .model import Model
from .types import Part


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
) -> Model:
    lm = build_default()
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


def complete(
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
):
    m = model(model_name, provider=provider, prompt_caching=prompt_caching, system=system)
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
):
    m = model(model_name, provider=provider, prompt_caching=prompt_caching, system=system)
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


def upload(
    model_name: str,
    path: str | Path | bytes,
    *,
    media_type: str | None = None,
    provider: str | None = None,
) -> Part:
    p = str(path) if isinstance(path, Path) else path
    resolved = provider or resolve_provider(model_name)
    m = model(model_name, provider=resolved)
    return m.upload(p, media_type=media_type)
