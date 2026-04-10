from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .capabilities import resolve_provider
from .errors import ProviderError, UnsupportedFeatureError
from .middleware import MiddlewarePipeline
from .protocols import LMAdapter, LiveSession
from .types import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    BatchRequest,
    BatchResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LMRequest,
    LMResponse,
    LiveConfig,
    StreamEvent,
)


@dataclass(slots=True)
class UniversalLM:
    adapters: dict[str, LMAdapter] = field(default_factory=dict)
    middleware: MiddlewarePipeline = field(default_factory=MiddlewarePipeline)

    def register(self, adapter: LMAdapter) -> None:
        self.adapters[adapter.provider] = adapter

    def _adapter(self, model: str, provider: str | None = None) -> LMAdapter:
        p = provider or resolve_provider(model)
        adapter = self.adapters.get(p)
        if not adapter:
            registered = list(self.adapters.keys()) or ["(none)"]
            raise ProviderError(
                f"no adapter registered for provider '{p}'\n\n"
                f"  Registered providers: {', '.join(registered)}\n"
                f"\n"
                f"  To fix, do one of:\n"
                f"    1. Set the API key: export {p.upper()}_API_KEY=...\n"
                f"    2. Pass it directly: lm15.call(..., api_key='...')\n"
                f"    3. Add it to your .env file and use lm15.configure(env='.env')\n"
            )
        return adapter

    def complete(self, request: LMRequest, provider: str | None = None) -> LMResponse:
        adapter = self._adapter(request.model, provider)
        if not adapter.supports.complete:
            raise UnsupportedFeatureError(f"{adapter.provider}: complete not supported")
        run = self.middleware.wrap_complete(adapter.complete)
        return run(request)

    def stream(self, request: LMRequest, provider: str | None = None) -> Iterator[StreamEvent]:
        adapter = self._adapter(request.model, provider)
        if not adapter.supports.stream:
            raise UnsupportedFeatureError(f"{adapter.provider}: stream not supported")
        run = self.middleware.wrap_stream(adapter.stream)
        yield from run(request)

    def live(self, config: LiveConfig, provider: str | None = None) -> LiveSession:
        adapter = self._adapter(config.model, provider)
        if not adapter.supports.live:
            raise UnsupportedFeatureError(f"{adapter.provider}: live not supported")
        return adapter.live(config)

    def embeddings(self, request: EmbeddingRequest, provider: str | None = None) -> EmbeddingResponse:
        adapter = self._adapter(request.model, provider)
        if not adapter.supports.embeddings:
            raise UnsupportedFeatureError(f"{adapter.provider}: embeddings not supported")
        return adapter.embeddings(request)

    def file_upload(self, request: FileUploadRequest, provider: str) -> FileUploadResponse:
        adapter = self._adapter(request.model or "", provider)
        if not adapter.supports.files:
            raise UnsupportedFeatureError(f"{adapter.provider}: files not supported")
        return adapter.file_upload(request)

    def batch_submit(self, request: BatchRequest, provider: str | None = None) -> BatchResponse:
        adapter = self._adapter(request.model, provider)
        if not adapter.supports.batches:
            raise UnsupportedFeatureError(f"{adapter.provider}: batches not supported")
        return adapter.batch_submit(request)

    def image_generate(self, request: ImageGenerationRequest, provider: str | None = None) -> ImageGenerationResponse:
        adapter = self._adapter(request.model, provider)
        if not adapter.supports.images:
            raise UnsupportedFeatureError(f"{adapter.provider}: images not supported")
        return adapter.image_generate(request)

    def audio_generate(self, request: AudioGenerationRequest, provider: str | None = None) -> AudioGenerationResponse:
        adapter = self._adapter(request.model, provider)
        if not adapter.supports.audio:
            raise UnsupportedFeatureError(f"{adapter.provider}: audio not supported")
        return adapter.audio_generate(request)
