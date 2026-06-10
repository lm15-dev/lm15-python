from __future__ import annotations

import os
from typing import ClassVar

from ..auth import (
    CLAUDE_CODE_CREDENTIALS_PATH,
    CLAUDE_CODE_LOGIN_HINT,
    get_claude_code_access_token,
)
from ..errors import NotConfiguredError, ProviderError, UnsupportedFeatureError, with_credential_hint
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities, LiveSession
from ..types import BatchRequest, BatchResponse, BuiltinTool, FileUploadRequest, FileUploadResponse, LiveConfig, Request
from .anthropic import AnthropicLM
from .base import SyncTransport, default_transport

DEFAULT_CLAUDE_CODE_VERSION = "2.1.170"
DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."


class ClaudeCodeLM(AnthropicLM):
    """Anthropic Messages adapter authenticated with local Claude Code OAuth."""

    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="claude-code",
        supports=supports,
        auth_modes=("claude-code-oauth", "bearer-oauth"),
        env_keys=(),
    )
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "document"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming", "tools", "reasoning"}),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2023-06-01",
        claude_code_version: str = DEFAULT_CLAUDE_CODE_VERSION,
    ) -> None:
        # get_claude_code_access_token raises typed, re-login-guided errors
        # (NotConfiguredError / AuthError) — let them propagate.
        token = api_key or get_claude_code_access_token(credentials_path)
        if not token:  # defensive: loaders never return an empty token
            path = os.fspath(credentials_path or CLAUDE_CODE_CREDENTIALS_PATH)
            raise NotConfiguredError(
                f"No Claude Code OAuth token found at {path}.",
                provider="claude-code",
                credential_hint=CLAUDE_CODE_LOGIN_HINT,
            )
        self.claude_code_version = claude_code_version
        super().__init__(
            api_key=token,
            transport=transport or default_transport(),
            base_url=base_url,
            api_version=api_version,
            provider="claude-code",
            capabilities=self.capabilities,
        )

    @classmethod
    def from_claude_code(
        cls,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        claude_code_version: str = DEFAULT_CLAUDE_CODE_VERSION,
    ) -> "ClaudeCodeLM":
        return cls(
            credentials_path=credentials_path,
            transport=transport,
            base_url=base_url,
            claude_code_version=claude_code_version,
        )

    def __repr__(self) -> str:  # never leak the OAuth token (dataclass repr would)
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"base_url={self.base_url!r}, api_key=<redacted>)"
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        # Same canonical mapping as AnthropicLM, but auth failures guide the
        # user to re-login (there is no env var for subscription auth).
        return with_credential_hint(super().normalize_error(status, body), CLAUDE_CODE_LOGIN_HINT)

    def _headers(self, request: Request | None = None) -> dict[str, str]:
        betas = ["claude-code-20250219", "oauth-2025-04-20"]
        if request is not None and any(
            isinstance(tool, BuiltinTool) and tool.name == "code_execution" for tool in request.tools
        ):
            betas.append("code-execution-2025-05-22")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
            "anthropic-beta": ",".join(betas),
            "x-app": "cli",
            "user-agent": f"claude-cli/{self.claude_code_version}",
        }

    def _payload(self, request: Request, stream: bool) -> dict[str, object]:
        payload = super()._payload(request, stream)
        default_system = {"type": "text", "text": DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT}
        existing = payload.get("system")
        if existing is None:
            payload["system"] = [default_system]
        elif isinstance(existing, list):
            payload["system"] = [default_system, *existing]
        else:
            payload["system"] = [default_system, {"type": "text", "text": str(existing)}]
        return payload

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        raise UnsupportedFeatureError("claude-code: file upload is not supported", provider=self.provider)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        raise UnsupportedFeatureError("claude-code: batch submit is not supported", provider=self.provider)

    def live(self, config: LiveConfig) -> LiveSession:
        raise UnsupportedFeatureError("claude-code: live is not supported", provider=self.provider)
