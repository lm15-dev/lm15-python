from __future__ import annotations

import os
from typing import ClassVar, Iterator

from ..auth import (
    CODEX_CLI_AUTH_PATH,
    OPENAI_CODEX_LOGIN_HINT,
    extract_chatgpt_account_id,
    get_codex_cli_access_token,
)
from ..errors import NotConfiguredError, ProviderError, UnsupportedFeatureError, with_credential_hint
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities, LiveSession
from ..result import materialize_response
from ..types import (
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
    LiveConfig,
    Request,
    Response,
    StreamEvent,
)
from .base import BaseProviderLM, SyncTransport, default_transport
from .openai import OpenAILM

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "lm15"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."


class OpenAICodexLM(OpenAILM):
    """OpenAI Responses adapter authenticated with local Codex CLI OAuth."""

    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="openai-codex",
        supports=supports,
        auth_modes=("chatgpt-oauth", "bearer-oauth"),
        env_keys=(),
    )
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "document", "binary"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming", "tools", "json_output", "reasoning"}),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        account_id: str | None = None,
        auth_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
    ) -> None:
        # get_codex_cli_access_token raises typed, re-login-guided errors
        # (NotConfiguredError / AuthError) — let them propagate.
        credential = None if api_key else get_codex_cli_access_token(auth_path)
        token = api_key or (credential.access_token if credential is not None else None)
        if not token:  # defensive: loaders never return an empty token
            path = os.fspath(auth_path or CODEX_CLI_AUTH_PATH)
            raise NotConfiguredError(
                f"No Codex CLI OAuth token found at {path}.",
                provider="openai-codex",
                credential_hint=OPENAI_CODEX_LOGIN_HINT,
            )
        resolved_account_id = account_id or (credential.account_id if credential is not None else None)
        resolved_account_id = resolved_account_id or extract_chatgpt_account_id(token)
        if not resolved_account_id:
            raise NotConfiguredError(
                "No ChatGPT account id found in the Codex OAuth token.",
                provider="openai-codex",
                credential_hint=OPENAI_CODEX_LOGIN_HINT,
            )
        self.account_id = resolved_account_id
        self.originator = originator
        super().__init__(
            api_key=token,
            transport=transport or default_transport(),
            base_url=base_url,
            profile=None,
            provider="openai-codex",
            capabilities=self.capabilities,
        )

    @classmethod
    def from_codex_cli(
        cls,
        *,
        auth_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
    ) -> "OpenAICodexLM":
        return cls(auth_path=auth_path, transport=transport, base_url=base_url, originator=originator)

    def __repr__(self) -> str:  # never leak the OAuth token (dataclass repr would)
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"base_url={self.base_url!r}, account_id={self.account_id!r}, api_key=<redacted>)"
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        # Same canonical mapping as OpenAILM, but auth failures guide the
        # user to re-login (there is no env var for subscription auth).
        return with_credential_hint(super().normalize_error(status, body), OPENAI_CODEX_LOGIN_HINT)

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
            "chatgpt-account-id": self.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": self.originator,
        }

    def _payload(self, request: Request, stream: bool) -> dict[str, object]:
        payload = super()._payload(request, stream=True)
        payload.setdefault("instructions", DEFAULT_CODEX_INSTRUCTIONS)
        payload["store"] = False
        payload["stream"] = True
        payload.pop("max_output_tokens", None)
        payload.pop("max_completion_tokens", None)
        payload.pop("max_tokens", None)
        return payload

    def complete(self, request: Request) -> Response:
        # The Codex subscription backend is streaming-first.  Materialize the
        # stream so callers get the same synchronous complete() surface.
        return materialize_response(self.stream(request), request)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        # Bypass OpenAILM.stream's realtime/websocket dispatch and go straight
        # through BaseProviderLM.stream, which applies the MAP-3 coalescer:
        # the Codex backend sends `response.completed` (usage) and then
        # `[DONE]` — two adapter-level end frames that must merge into exactly
        # one final StreamEndEvent with usage intact.
        yield from BaseProviderLM.stream(self, request)

    def live(self, config: LiveConfig) -> LiveSession:
        raise UnsupportedFeatureError("openai-codex: live is not supported", provider=self.provider)

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedFeatureError("openai-codex: embeddings are not supported", provider=self.provider)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        raise UnsupportedFeatureError("openai-codex: file upload is not supported", provider=self.provider)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        raise UnsupportedFeatureError("openai-codex: batch submit is not supported", provider=self.provider)

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise UnsupportedFeatureError("openai-codex: image generation is not supported", provider=self.provider)

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        raise UnsupportedFeatureError("openai-codex: audio generation is not supported", provider=self.provider)
