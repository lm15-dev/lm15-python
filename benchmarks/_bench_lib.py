"""
Shared benchmark primitives: per-library HTTP streaming with fine-grained
timing.  Each benchmark function takes an opaque `TokenParser` callable
that examines the accumulated SSE buffer and returns the first visible
content token (or None).  This lets us reuse the exact same library code
across OpenAI, Gemini, Anthropic, etc. — only the request shape and the
parser differ.
"""

from __future__ import annotations

import json
import time
from typing import Callable


TokenParser = Callable[[bytes], "str | None"]


def pack(
    *,
    lib: str,
    scenario: str,
    model: str | None,
    t_start: float,
    t_import: float,
    t_client: float,
    t_headers: float,
    t_first_byte: float | None,
    t_first_tok: float | None,
    t_end: float,
    status: int,
    first_tok: str | None,
) -> dict:
    """Build the standard per-run result dict."""
    ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
    out = {
        "lib": lib,
        "scenario": scenario,
        "status": status,
        "first_token": first_tok,
        "import_ms": ms(t_start, t_import),
        "client_ms": ms(t_import, t_client),
        "request_ms": ms(t_client, t_headers),
        "first_byte_ms": ms(t_headers, t_first_byte) if t_first_byte else None,
        "first_token_ms": ms(t_headers, t_first_tok) if t_first_tok else None,
        "complete_ms": ms(t_headers, t_end),
        "total_ms": ms(t_start, t_end),
    }
    if model is not None:
        out["model"] = model
    return out


# ─── Per-library implementations ─────────────────────────────────────


def run_lm15_sync(
    *,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    parser: TokenParser,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    t0 = time.perf_counter()
    from lm15.transports import TransportRequest, StdlibTransport
    t1 = time.perf_counter()
    tr = StdlibTransport()
    t2 = time.perf_counter()
    req = TransportRequest(method="POST", url=url, headers=headers, body=body)
    with tr.stream(req) as resp:
        t3 = time.perf_counter()
        t_first_byte = t_first_tok = None
        first_tok = None
        buf = b""
        for chunk in resp:
            if t_first_byte is None:
                t_first_byte = time.perf_counter()
            buf += chunk
            if first_tok is None:
                tok = parser(buf)
                if tok:
                    first_tok = tok
                    t_first_tok = time.perf_counter()
        t4 = time.perf_counter()
        status = resp.status
    tr.close()
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=t2, t_headers=t3,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=t4, status=status, first_tok=first_tok)


def run_lm15_async(
    *,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    parser: TokenParser,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    t0 = time.perf_counter()
    from lm15.transports import TransportRequest, StdlibAsyncTransport
    import asyncio
    t1 = time.perf_counter()

    async def run() -> tuple:
        tr = StdlibAsyncTransport()
        ta = time.perf_counter()
        req = TransportRequest(method="POST", url=url, headers=headers, body=body)
        async with tr.stream(req) as resp:
            tb = time.perf_counter()
            t_first_byte = t_first_tok = None
            first_tok = None
            buf = b""
            async for chunk in resp:
                if t_first_byte is None:
                    t_first_byte = time.perf_counter()
                buf += chunk
                if first_tok is None:
                    tok = parser(buf)
                    if tok:
                        first_tok = tok
                        t_first_tok = time.perf_counter()
            tc = time.perf_counter()
            status = resp.status
        await tr.aclose()
        return ta, tb, t_first_byte, t_first_tok, tc, status, first_tok

    ta, tb, t_first_byte, t_first_tok, tc, status, first_tok = asyncio.run(run())
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=ta, t_headers=tb,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=tc, status=status, first_tok=first_tok)


def run_httpx_sync(
    *,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    parser: TokenParser,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    t0 = time.perf_counter()
    import httpx
    t1 = time.perf_counter()
    client = httpx.Client(headers=dict(headers), timeout=30)
    t2 = time.perf_counter()
    with client.stream("POST", url, content=body) as resp:
        t3 = time.perf_counter()
        t_first_byte = t_first_tok = None
        first_tok = None
        buf = b""
        for chunk in resp.iter_bytes():
            if t_first_byte is None:
                t_first_byte = time.perf_counter()
            buf += chunk
            if first_tok is None:
                tok = parser(buf)
                if tok:
                    first_tok = tok
                    t_first_tok = time.perf_counter()
        t4 = time.perf_counter()
        status = resp.status_code
    client.close()
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=t2, t_headers=t3,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=t4, status=status, first_tok=first_tok)


def run_httpx_async(
    *,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    parser: TokenParser,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    t0 = time.perf_counter()
    import httpx
    import asyncio
    t1 = time.perf_counter()

    async def run() -> tuple:
        client = httpx.AsyncClient(headers=dict(headers), timeout=30)
        ta = time.perf_counter()
        async with client.stream("POST", url, content=body) as resp:
            tb = time.perf_counter()
            t_first_byte = t_first_tok = None
            first_tok = None
            buf = b""
            async for chunk in resp.aiter_bytes():
                if t_first_byte is None:
                    t_first_byte = time.perf_counter()
                buf += chunk
                if first_tok is None:
                    tok = parser(buf)
                    if tok:
                        first_tok = tok
                        t_first_tok = time.perf_counter()
            tc = time.perf_counter()
            status = resp.status_code
        await client.aclose()
        return ta, tb, t_first_byte, t_first_tok, tc, status, first_tok

    ta, tb, t_first_byte, t_first_tok, tc, status, first_tok = asyncio.run(run())
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=ta, t_headers=tb,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=tc, status=status, first_tok=first_tok)


def run_requests(
    *,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    parser: TokenParser,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    t0 = time.perf_counter()
    import requests
    t1 = time.perf_counter()
    session = requests.Session()
    t2 = time.perf_counter()
    resp = session.post(url, headers=dict(headers), data=body, stream=True, timeout=30)
    t3 = time.perf_counter()
    t_first_byte = t_first_tok = None
    first_tok = None
    buf = b""
    for chunk in resp.iter_content(chunk_size=None):
        if t_first_byte is None:
            t_first_byte = time.perf_counter()
        buf += chunk
        if first_tok is None:
            tok = parser(buf)
            if tok:
                first_tok = tok
                t_first_tok = time.perf_counter()
    t4 = time.perf_counter()
    status = resp.status_code
    session.close()
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=t2, t_headers=t3,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=t4, status=status, first_tok=first_tok)


def run_aiohttp(
    *,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    parser: TokenParser,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    t0 = time.perf_counter()
    import aiohttp
    import asyncio
    t1 = time.perf_counter()

    async def run() -> tuple:
        session = aiohttp.ClientSession(headers=dict(headers))
        ta = time.perf_counter()
        async with session.post(
            url, data=body, timeout=aiohttp.ClientTimeout(30)
        ) as resp:
            tb = time.perf_counter()
            t_first_byte = t_first_tok = None
            first_tok = None
            buf = b""
            async for chunk in resp.content.iter_any():
                if t_first_byte is None:
                    t_first_byte = time.perf_counter()
                buf += chunk
                if first_tok is None:
                    tok = parser(buf)
                    if tok:
                        first_tok = tok
                        t_first_tok = time.perf_counter()
            tc = time.perf_counter()
            status = resp.status
        await session.close()
        return ta, tb, t_first_byte, t_first_tok, tc, status, first_tok

    ta, tb, t_first_byte, t_first_tok, tc, status, first_tok = asyncio.run(run())
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=ta, t_headers=tb,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=tc, status=status, first_tok=first_tok)


RUNNERS = {
    "lm15-sync": run_lm15_sync,
    "lm15-async": run_lm15_async,
    "httpx-sync": run_httpx_sync,
    "httpx-async": run_httpx_async,
    "requests": run_requests,
    "aiohttp": run_aiohttp,
}


# ─── SSE parsers ─────────────────────────────────────────────────────


def openai_chat_completions_parser(buf: bytes) -> str | None:
    """Extract the first non-empty `delta.content` string from an OpenAI
    /v1/chat/completions SSE stream."""
    for line in buf.split(b"\n"):
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]" or not data:
            continue
        try:
            obj = json.loads(data)
            content = obj["choices"][0]["delta"].get("content")
            if content:
                return content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
    return None


def gemini_parser(buf: bytes) -> str | None:
    """Extract the first non-empty `candidates[0].content.parts[*].text`
    from a Gemini streamGenerateContent?alt=sse stream.

    Gemini's first chunk may contain only a thoughtSignature (for thinking
    models) with an empty text; we skip those and wait for real content.
    """
    for line in buf.split(b"\n"):
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if not data:
            continue
        try:
            obj = json.loads(data)
            parts = obj["candidates"][0]["content"].get("parts", [])
            for part in parts:
                text = part.get("text")
                if text:
                    return text
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
    return None


# ─── SDK runners ─────────────────────────────────────────────────────
#
# Each provider SDK exposes its own streaming API.  The benchmark scripts
# pass in an SDKOps tuple describing how to:
#   1. import the SDK (returns the module, timed as `import_ms`)
#   2. construct a client from an api_key (timed as `client_ms`)
#   3. open a stream (timed as `request_ms` up to the first yielded event)
#   4. extract the first visible content token from a yielded event
#
# This lets us reuse the fine-grained timing across every SDK while each
# SDK keeps its native streaming shape.

import typing as _t


class SDKOps(_t.NamedTuple):
    """Describes how to drive one provider SDK for the TTFT benchmark."""
    name: str                          # e.g. "openai-sdk", "groq-sdk", "genai-sdk"
    do_import: _t.Callable[[], _t.Any]  # returns something opaque
    make_client: _t.Callable[[_t.Any, str], _t.Any]  # (imported, api_key) -> client
    open_stream: _t.Callable[[_t.Any], _t.Iterator[_t.Any]]  # (client) -> event iter
    event_to_token: _t.Callable[[_t.Any], "str | None"]  # (event) -> token or None


def run_sdk_sync(
    *,
    ops: SDKOps,
    api_key: str,
    lib: str,
    scenario: str,
    model: str | None,
) -> dict:
    """Generic SDK benchmark runner with the same column semantics as the
    HTTP-client runners above.

    Because SDKs hide the HTTP response object, we map our columns like so:
      * request_ms   = time from client construction until the stream
                       iterator yields its first event (= headers in the
                       HTTP-client benchmarks).
      * first_byte_ms= 0 (same as request_ms for SDKs; the SDK's first
                       event IS the first byte).
      * first_token_ms = time from first yielded event until we see an
                       event carrying non-empty content.
      * complete_ms  = time from first event until the iterator is drained.
    """
    t0 = time.perf_counter()
    sdk = ops.do_import()
    t1 = time.perf_counter()
    client = ops.make_client(sdk, api_key)
    t2 = time.perf_counter()
    stream = ops.open_stream(client)
    first_tok: str | None = None
    t3 = t_first_byte = t_first_tok = None
    status = 200  # SDKs raise on non-200, so if we're here we got 200
    try:
        for event in stream:
            if t3 is None:
                t3 = time.perf_counter()
                t_first_byte = t3
            if first_tok is None:
                tok = ops.event_to_token(event)
                if tok:
                    first_tok = tok
                    t_first_tok = time.perf_counter()
        t4 = time.perf_counter()
    except Exception as exc:  # pragma: no cover — surface as non-200 result
        t4 = time.perf_counter()
        status = getattr(exc, "status_code", 0) or -1
    if t3 is None:
        t3 = t4
    return pack(lib=lib, scenario=scenario, model=model, t_start=t0,
                t_import=t1, t_client=t2, t_headers=t3,
                t_first_byte=t_first_byte, t_first_tok=t_first_tok,
                t_end=t4, status=status, first_tok=first_tok)


# ─── Concrete SDK ops ───────────────────────────────────────────────


def _openai_sdk_ops(*, model: str, prompt: str, max_tokens: int = 20) -> SDKOps:
    def do_import() -> _t.Any:
        import openai
        return openai

    def make_client(sdk: _t.Any, api_key: str) -> _t.Any:
        return sdk.OpenAI(api_key=api_key)

    def open_stream(client: _t.Any) -> _t.Iterator[_t.Any]:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
        )

    def event_to_token(event: _t.Any) -> "str | None":
        try:
            content = event.choices[0].delta.content
            return content or None
        except (AttributeError, IndexError):
            return None

    return SDKOps(
        name="openai-sdk",
        do_import=do_import,
        make_client=make_client,
        open_stream=open_stream,
        event_to_token=event_to_token,
    )


def _groq_sdk_ops(*, model: str, prompt: str, max_tokens: int = 20) -> SDKOps:
    def do_import() -> _t.Any:
        import groq
        return groq

    def make_client(sdk: _t.Any, api_key: str) -> _t.Any:
        return sdk.Groq(api_key=api_key)

    def open_stream(client: _t.Any) -> _t.Iterator[_t.Any]:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_completion_tokens=max_tokens,
        )

    def event_to_token(event: _t.Any) -> "str | None":
        try:
            content = event.choices[0].delta.content
            return content or None
        except (AttributeError, IndexError):
            return None

    return SDKOps(
        name="groq-sdk",
        do_import=do_import,
        make_client=make_client,
        open_stream=open_stream,
        event_to_token=event_to_token,
    )


def _genai_sdk_ops(*, model: str, prompt: str) -> SDKOps:
    def do_import() -> _t.Any:
        from google import genai
        return genai

    def make_client(sdk: _t.Any, api_key: str) -> _t.Any:
        return sdk.Client(api_key=api_key)

    def open_stream(client: _t.Any) -> _t.Iterator[_t.Any]:
        return client.models.generate_content_stream(
            model=model,
            contents=prompt,
        )

    def event_to_token(event: _t.Any) -> "str | None":
        text = getattr(event, "text", None)
        return text or None

    return SDKOps(
        name="genai-sdk",
        do_import=do_import,
        make_client=make_client,
        open_stream=open_stream,
        event_to_token=event_to_token,
    )


def _litellm_ops(
    *,
    litellm_model: str,
    env_key: str,
    prompt: str,
    max_tokens: int,
) -> SDKOps:
    """LiteLLM is a provider-agnostic wrapper; it expects credentials via env
    vars (OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY) rather than a client
    constructor.  We honor that: `make_client` sets the env var, `open_stream`
    calls `litellm.completion(stream=True)`.
    """
    import os

    def do_import() -> _t.Any:
        import litellm
        return litellm

    def make_client(sdk: _t.Any, api_key: str) -> _t.Any:
        os.environ[env_key] = api_key
        return sdk  # LiteLLM has no client object; the module is the client

    def open_stream(client: _t.Any) -> _t.Iterator[_t.Any]:
        return client.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
        )

    def event_to_token(event: _t.Any) -> "str | None":
        try:
            content = event.choices[0].delta.content
            return content or None
        except (AttributeError, IndexError):
            return None

    return SDKOps(
        name="litellm",
        do_import=do_import,
        make_client=make_client,
        open_stream=open_stream,
        event_to_token=event_to_token,
    )


# Public accessors — each returns SDKOps configured with defaults matching
# the HTTP-client benchmarks (same prompt, same model, same max_tokens).


def openai_sdk_ops(
    *, model: str = "gpt-4.1-nano",
    prompt: str = "Say 'hello' and nothing else.",
    max_tokens: int = 20,
) -> SDKOps:
    return _openai_sdk_ops(model=model, prompt=prompt, max_tokens=max_tokens)


def groq_sdk_ops(
    *, model: str = "llama-3.1-8b-instant",
    prompt: str = "Say 'hello' and nothing else.",
    max_tokens: int = 20,
) -> SDKOps:
    return _groq_sdk_ops(model=model, prompt=prompt, max_tokens=max_tokens)


def genai_sdk_ops(
    *, model: str = "gemini-3.1-flash-lite-preview",
    prompt: str = "Say 'hello' and nothing else.",
) -> SDKOps:
    return _genai_sdk_ops(model=model, prompt=prompt)


def litellm_openai_ops(
    *, model: str = "gpt-4.1-nano",
    prompt: str = "Say 'hello' and nothing else.",
    max_tokens: int = 20,
) -> SDKOps:
    return _litellm_ops(
        litellm_model=f"openai/{model}", env_key="OPENAI_API_KEY",
        prompt=prompt, max_tokens=max_tokens,
    )


def litellm_groq_ops(
    *, model: str = "llama-3.1-8b-instant",
    prompt: str = "Say 'hello' and nothing else.",
    max_tokens: int = 20,
) -> SDKOps:
    return _litellm_ops(
        litellm_model=f"groq/{model}", env_key="GROQ_API_KEY",
        prompt=prompt, max_tokens=max_tokens,
    )


def litellm_gemini_ops(
    *, model: str = "gemini-flash-lite-latest",
    prompt: str = "Say 'hello' and nothing else.",
    max_tokens: int = 20,
) -> SDKOps:
    return _litellm_ops(
        litellm_model=f"gemini/{model}", env_key="GEMINI_API_KEY",
        prompt=prompt, max_tokens=max_tokens,
    )

