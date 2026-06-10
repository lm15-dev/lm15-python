"""Async mirror provider LMs.

Each Async* class is a perfect mirror of its sync sibling: same constructor
fields, same canonical Request in, same canonical Response / StreamEvents
out — ``await`` is the only user-visible difference.

Design (see docs/design-rationale.md, "Async"): composition, not
inheritance.  Subclassing the sync adapter and overriding sync methods with
async ones would be a typing violation (``complete`` would no longer be
substitutable).  Instead each async class owns the async transport and
delegates ALL pure mapping — build_request, parse_response,
parse_stream_events, normalize_error, payload/header helpers — to an inner
instance of the sync adapter class, constructed with a transport that raises
if it is ever used: the inner adapter must never touch the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar, Protocol

from ..errors import (
    ProviderError,
    TransportError as LM15TransportError,
    UnsupportedFeatureError,
)
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from ..sse import aparse_sse
from ..transports import (
    AsyncTransportResponse,
    StdlibAsyncTransport,
    TransportRequest,
    TransportError as NetworkTransportError,
)
from ..types import (
    AudioGenerationRequest,
    BatchRequest,
    CacheConfig,
    EmbeddingRequest,
    FileUploadRequest,
    ImageGenerationRequest,
    LiveConfig,
    Request,
    Response,
    StreamEvent,
)
from .anthropic import AnthropicLM
from .base import BaseProviderLM, HttpResponse, SyncTransport
from .claude_code import DEFAULT_CLAUDE_CODE_VERSION, ClaudeCodeLM
from .common import make_json_request
from .gemini import GeminiLM
from .openai import OpenAILM
from .openai_chat import OpenAIChatLM
from .openai_codex import DEFAULT_CODEX_BASE_URL, DEFAULT_CODEX_ORIGINATOR, OpenAICodexLM


class AsyncTransport(Protocol):
    """Minimal async transport surface used by async provider LMs.

    ``stream`` returns an async context manager producing an
    :class:`AsyncTransportResponse` (StdlibAsyncTransport's shape).
    """

    def stream(self, request: TransportRequest) -> Any: ...


def default_async_transport() -> AsyncTransport:
    """Create the default async transport for standalone async provider LMs."""
    return StdlibAsyncTransport()


def _mirror_default(cls, name: str):
    """Default value of a sync sibling's dataclass field (slots-safe)."""
    return cls.__dataclass_fields__[name].default


class _ForbiddenTransport:
    """Transport for the inner sync adapter: pure mapping only, no I/O."""

    def stream(self, request: TransportRequest) -> Any:
        raise RuntimeError(
            "inner sync adapter of an Async* provider LM must never touch the "
            "network; all I/O goes through the async transport"
        )


class AsyncBaseProviderLM:
    """Shared asynchronous provider LM implementation.

    Mirrors :class:`BaseProviderLM` faithfully: build (delegated, pure) ->
    await async transport -> parse (delegated, pure); status>=400 raises the
    delegated normalize_error; transport errors are wrapped in
    lm15.TransportError.  Streaming applies MAP-3 via
    :func:`lm15.result.acoalesce_stream`.
    """

    transport: AsyncTransport
    _inner: BaseProviderLM  # set by subclass __post_init__

    # Mirrored metadata (subclasses override like their sync siblings).
    provider: str = "unknown"
    capabilities: Capabilities = Capabilities()
    supports: ClassVar[EndpointSupport] = EndpointSupport()
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="unknown", supports=EndpointSupport()
    )

    async def complete(self, request: Request) -> Response:
        req = self._inner.build_request(request, stream=False)
        resp = await self._send(req)
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner.parse_response(request, resp)

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        # MAP-3 (docs/mapping-rules.md): adapters may emit one end event per
        # provider terminal frame; the coalescer merges them so the public
        # stream yields exactly one final StreamEndEvent.
        from ..result import acoalesce_stream

        return acoalesce_stream(self._stream_raw(request))

    async def _stream_raw(self, request: Request) -> AsyncIterator[StreamEvent]:
        req = self._inner.build_request(request, stream=True)
        try:
            async with self.transport.stream(req) as resp:
                if resp.status >= 400:
                    body = await resp.read()
                    raise self._inner.normalize_error(
                        resp.status, body.decode("utf-8", errors="replace")
                    )
                async for raw in aparse_sse(_aiter_lines(resp)):
                    for event in self._inner.parse_stream_events(request, raw):
                        if event is not None:
                            yield event
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    async def _send(self, request: TransportRequest) -> HttpResponse:
        try:
            async with self.transport.stream(request) as resp:
                body = await resp.read()
                return HttpResponse(
                    status=resp.status,
                    reason=resp.reason,
                    headers=resp.headers,
                    body=body,
                    http_version=resp.http_version,
                )
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    def normalize_error(self, status: int, body: str) -> ProviderError:
        return self._inner.normalize_error(status, body)

    async def aclose(self) -> None:
        aclose = getattr(self.transport, "aclose", None)
        if callable(aclose):
            await aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ── Non-chat endpoints: sync-only for now (honest surface) ──────

    def _async_unsupported(self, endpoint: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(
            f"{self.provider}: {endpoint}: use the sync adapter for this "
            "endpoint (async endpoints planned)",
            provider=self.provider,
        )

    def live(self, config: LiveConfig):
        raise self._async_unsupported("live")

    def embeddings(self, request: EmbeddingRequest):
        raise self._async_unsupported("embeddings")

    def file_upload(self, request: FileUploadRequest):
        raise self._async_unsupported("file upload")

    def batch_submit(self, request: BatchRequest):
        raise self._async_unsupported("batch submit")

    def image_generate(self, request: ImageGenerationRequest):
        raise self._async_unsupported("image generation")

    def audio_generate(self, request: AudioGenerationRequest):
        raise self._async_unsupported("audio generation")


async def _aiter_lines(resp: AsyncTransportResponse) -> AsyncIterator[bytes]:
    aiter_lines = getattr(resp, "aiter_lines", None)
    if aiter_lines is not None:
        async for line in aiter_lines():
            yield line
        return
    buf = bytearray()
    async for chunk in resp:
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            idx = buf.find(b"\n")
            if idx < 0:
                break
            yield bytes(buf[: idx + 1])
            del buf[: idx + 1]
    if buf:
        yield bytes(buf)


# ─── Mirror classes ──────────────────────────────────────────────────


@dataclass(slots=True)
class AsyncOpenAILM(AsyncBaseProviderLM):
    api_key: str
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://api.openai.com/v1"
    profile: Any | None = None

    provider: str = "openai"
    capabilities: Capabilities = _mirror_default(OpenAILM, "capabilities")
    supports: ClassVar[EndpointSupport] = OpenAILM.supports
    manifest: ClassVar[ProviderManifest] = OpenAILM.manifest

    _inner: OpenAILM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = OpenAILM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            profile=self.profile,
        )


@dataclass(slots=True)
class AsyncAnthropicLM(AsyncBaseProviderLM):
    api_key: str
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"

    provider: str = "anthropic"
    capabilities: Capabilities = _mirror_default(AnthropicLM, "capabilities")
    supports: ClassVar[EndpointSupport] = AnthropicLM.supports
    manifest: ClassVar[ProviderManifest] = AnthropicLM.manifest

    _inner: AnthropicLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = AnthropicLM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            api_version=self.api_version,
        )


@dataclass(slots=True)
class AsyncGeminiLM(AsyncBaseProviderLM):
    api_key: str
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    upload_base_url: str = "https://generativelanguage.googleapis.com/upload/v1beta"
    _cached_content_ids: dict[str, str] = field(default_factory=dict, repr=False)

    provider: str = "gemini"
    capabilities: Capabilities = _mirror_default(GeminiLM, "capabilities")
    supports: ClassVar[EndpointSupport] = GeminiLM.supports
    manifest: ClassVar[ProviderManifest] = GeminiLM.manifest

    _inner: GeminiLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = GeminiLM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            upload_base_url=self.upload_base_url,
            # Share the cache-id map so the inner adapter's pure
            # _apply_prompt_cache sees ids resolved by the async port below.
            _cached_content_ids=self._cached_content_ids,
        )

    async def resolve_prompt_cache(self, request: Request) -> str | None:
        """Async port of GeminiLM.resolve_prompt_cache — the one network call
        the sync mapping layer owns.  Mirrors the sync logic exactly, but the
        cachedContents POST goes through the async transport."""
        inner = self._inner
        cache_cfg: CacheConfig | None = request.config.cache
        if not (cache_cfg is None or cache_cfg.mode != "off"):
            return None
        payload = inner._payload(request, apply_cache=False)
        plan = inner._prompt_cache_plan(request, payload)
        if plan is None:
            return None
        cache_id = self._cached_content_ids.get(plan["key"])
        if cache_id is not None:
            return cache_id

        body: dict[str, Any] = {
            "model": inner._model_path(request.model),
            "contents": plan["prefix"],
        }
        if payload.get("systemInstruction"):
            body["systemInstruction"] = payload["systemInstruction"]
        if cache_cfg is not None and cache_cfg.retention == "long":
            body["ttl"] = "86400s"  # 24 hours

        resp = await self._send(make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/cachedContents",
            headers=inner._auth_headers({"Content-Type": "application/json"}),
            payload=body,
            read_timeout=60.0,
        ))
        if resp.status < 400:
            data = resp.json()
            name = data.get("name")
            if name:
                cache_id = str(name)
                self._cached_content_ids[plan["key"]] = cache_id
                return cache_id
        return None

    async def complete(self, request: Request) -> Response:
        await self.resolve_prompt_cache(request)
        return await AsyncBaseProviderLM.complete(self, request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        await self.resolve_prompt_cache(request)
        async for event in AsyncBaseProviderLM.stream(self, request):
            yield event


@dataclass(slots=True)
class AsyncOpenAIChatLM(AsyncBaseProviderLM):
    api_key: str
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = _mirror_default(OpenAIChatLM, "base_url")
    compat: Any | None = None

    provider: str = "openai_chat"
    capabilities: Capabilities = _mirror_default(OpenAIChatLM, "capabilities")
    supports: ClassVar[EndpointSupport] = OpenAIChatLM.supports
    manifest: ClassVar[ProviderManifest] = OpenAIChatLM.manifest

    _inner: OpenAIChatLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = OpenAIChatLM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            compat=self.compat,
        )
        # The sync sibling's __post_init__ resolves compat presets (and may
        # supply that server's default base_url); mirror the resolved values.
        self.base_url = self._inner.base_url
        self.compat = self._inner.compat


# ─── Subscription mirrors (Claude Code / Codex CLI OAuth) ────────────
#
# Same composition pattern: the inner sync adapter resolves the local OAuth
# credential at construction time (file read; a token refresh, if needed, is
# one blocking call) and owns all pure mapping; the resolved token fields are
# repr-suppressed so secrets never leak.


@dataclass(slots=True)
class AsyncClaudeCodeLM(AsyncBaseProviderLM):
    api_key: str | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = None
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    claude_code_version: str = DEFAULT_CLAUDE_CODE_VERSION

    # Not constructor params on the sync sibling either (it is not a dataclass).
    provider: str = field(default="claude-code", init=False)
    capabilities: Capabilities = field(default=ClaudeCodeLM.capabilities, init=False)
    supports: ClassVar[EndpointSupport] = ClaudeCodeLM.supports
    manifest: ClassVar[ProviderManifest] = ClaudeCodeLM.manifest

    _inner: ClaudeCodeLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = ClaudeCodeLM(
            api_key=self.api_key,
            credentials_path=self.credentials_path,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            api_version=self.api_version,
            claude_code_version=self.claude_code_version,
        )
        self.api_key = self._inner.api_key  # resolved OAuth token (repr-suppressed)

    def file_upload(self, request: FileUploadRequest):
        return self._inner.file_upload(request)  # raises UnsupportedFeatureError

    def batch_submit(self, request: BatchRequest):
        return self._inner.batch_submit(request)  # raises UnsupportedFeatureError

    def live(self, config: LiveConfig):
        return self._inner.live(config)  # raises UnsupportedFeatureError


@dataclass(slots=True)
class AsyncOpenAICodexLM(AsyncBaseProviderLM):
    api_key: str | None = field(default=None, repr=False)
    account_id: str | None = None
    auth_path: "str | os.PathLike[str] | None" = None
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = DEFAULT_CODEX_BASE_URL
    originator: str = DEFAULT_CODEX_ORIGINATOR

    # Not constructor params on the sync sibling either (it is not a dataclass).
    provider: str = field(default="openai-codex", init=False)
    capabilities: Capabilities = field(default=OpenAICodexLM.capabilities, init=False)
    supports: ClassVar[EndpointSupport] = OpenAICodexLM.supports
    manifest: ClassVar[ProviderManifest] = OpenAICodexLM.manifest

    _inner: OpenAICodexLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = OpenAICodexLM(
            api_key=self.api_key,
            account_id=self.account_id,
            auth_path=self.auth_path,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            originator=self.originator,
        )
        self.api_key = self._inner.api_key  # resolved OAuth token (repr-suppressed)
        self.account_id = self._inner.account_id

    async def complete(self, request: Request) -> Response:
        # Mirror of OpenAICodexLM.complete: the Codex subscription backend is
        # streaming-first; materialize the (coalesced) stream.
        from ..result import materialize_response

        events = [event async for event in self.stream(request)]
        return materialize_response(iter(events), request)

    def live(self, config: LiveConfig):
        return self._inner.live(config)  # raises UnsupportedFeatureError

    def embeddings(self, request: EmbeddingRequest):
        return self._inner.embeddings(request)  # raises UnsupportedFeatureError

    def file_upload(self, request: FileUploadRequest):
        return self._inner.file_upload(request)  # raises UnsupportedFeatureError

    def batch_submit(self, request: BatchRequest):
        return self._inner.batch_submit(request)  # raises UnsupportedFeatureError

    def image_generate(self, request: ImageGenerationRequest):
        return self._inner.image_generate(request)  # raises UnsupportedFeatureError

    def audio_generate(self, request: AudioGenerationRequest):
        return self._inner.audio_generate(request)  # raises UnsupportedFeatureError


__all__ = [
    "AsyncBaseProviderLM",
    "AsyncTransport",
    "AsyncOpenAILM",
    "AsyncAnthropicLM",
    "AsyncGeminiLM",
    "AsyncOpenAIChatLM",
    "AsyncClaudeCodeLM",
    "AsyncOpenAICodexLM",
    "default_async_transport",
]
