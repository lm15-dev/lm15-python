from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports.base import HttpRequest, HttpResponse, Transport
from ..types import (
    BatchRequest,
    BatchResponse,
    FileUploadRequest,
    FileUploadResponse,
    LMRequest,
    LMResponse,
    Message,
    Part,
    PartDelta,
    StreamEvent,
    ThinkingPart,
    Usage,
)
from ..errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
    canonical_error_code,
    map_http_error,
)
from .base import BaseProviderAdapter
from .common import ds_to_anthropic_source, parts_to_text

# Canonical builtin tool name → Anthropic tool format
_ANTHROPIC_BUILTIN_MAP: dict[str, str] = {
    "web_search": "web_search_20250305",
    "code_execution": "code_execution_20250522",
}


def _builtin_to_anthropic(tool: Any) -> dict[str, Any]:
    """Convert a BuiltinTool to Anthropic wire format."""
    wire_type = _ANTHROPIC_BUILTIN_MAP.get(tool.name)
    if wire_type:
        out: dict[str, Any] = {"type": wire_type, "name": tool.name}
        if tool.builtin_config:
            out.update(tool.builtin_config)
        return out
    # Unknown builtin — pass name as type
    out = {"type": tool.name, "name": tool.name}
    if tool.builtin_config:
        out.update(tool.builtin_config)
    return out


@dataclass(slots=True)
class AnthropicAdapter(BaseProviderAdapter):
    api_key: str
    transport: Transport
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"

    provider: str = "anthropic"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "document"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming", "tools", "reasoning", "files", "batch"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True, files=True, batches=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="anthropic", supports=supports, auth_modes=("x-api-key",), env_keys=("ANTHROPIC_API_KEY",))

    # https://platform.claude.com/docs/en/api/errors
    _error_type_map: ClassVar[dict[str, type[ProviderError]]] = {
        "authentication_error": AuthError,
        "permission_error": AuthError,
        "billing_error": BillingError,
        "rate_limit_error": RateLimitError,
        "request_too_large": InvalidRequestError,
        "not_found_error": InvalidRequestError,
        "invalid_request_error": InvalidRequestError,
        "api_error": ServerError,
        "overloaded_error": ServerError,
        "timeout_error": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        m = msg.lower()
        return (
            "prompt is too long" in m
            or "too many tokens" in m
            or "context window" in m
            or "context length" in m
            or ("token" in m and ("limit" in m or "exceed" in m))
        )

    def _stream_error(self, provider_code: str, message: str) -> dict[str, str]:
        cls = self._error_type_map.get(provider_code, ProviderError)
        if provider_code == "invalid_request_error" and self._is_context_length_message(message):
            cls = ContextLengthError
        return {
            "code": canonical_error_code(cls),
            "message": message,
            "provider_code": provider_code or "provider",
        }

    def normalize_error(self, status: int, body: str) -> ProviderError:
        """Extract message from Anthropic error shape.

        Shape: ``{"type": "error", "error": {"type": "...", "message": "..."}}``
        Source: https://docs.anthropic.com/en/api/errors

        ``error.type`` is the structured signal for most errors. The one
        exception is context overflow, which shares the generic
        ``invalid_request_error`` type with all other 400s — message
        matching is the only option there.
        """
        try:
            data = json.loads(body)
            err = data.get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_type = err.get("type", "") if isinstance(err, dict) else ""
            request_id = data.get("request_id", "") if isinstance(data, dict) else ""

            # Context overflow: no structured code, only message matching.
            if err_type == "invalid_request_error" and self._is_context_length_message(msg):
                if request_id and request_id not in msg:
                    msg = f"{msg} (request_id={request_id})"
                return ContextLengthError(msg)

            # Structured error.type → typed error class.
            cls = self._error_type_map.get(err_type)
            if cls:
                if request_id and request_id not in msg:
                    msg = f"{msg} (request_id={request_id})"
                return cls(msg)

            if err_type and err_type not in msg:
                msg = f"{msg} ({err_type})"
            if request_id and request_id not in msg:
                msg = f"{msg} (request_id={request_id})"
        except Exception:
            msg = body.strip()[:200] or f"HTTP {status}"
        return map_http_error(status, msg)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def _part(self, p: Part) -> dict[str, Any]:
        if p.type == "text":
            out = {"type": "text", "text": p.text or ""}
        elif p.type == "image" and p.source:
            out = {"type": "image", "source": ds_to_anthropic_source(p.source)}
        elif p.type == "document" and p.source:
            out = {"type": "document", "source": ds_to_anthropic_source(p.source)}
        elif p.type == "tool_call":
            out = {
                "type": "tool_use",
                "id": p.id,
                "name": p.name,
                "input": p.input or {},
            }
        elif p.type == "tool_result":
            content_text = parts_to_text(p.content) if p.content else ""
            out: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": p.id,
            }
            if content_text:
                out["content"] = content_text
            if p.is_error:
                out["is_error"] = True
        else:
            out = {"type": "text", "text": p.text or ""}

        cache_meta = (p.metadata or {}).get("cache") if p.metadata else None
        if cache_meta is True:
            out["cache_control"] = {"type": "ephemeral"}
        elif isinstance(cache_meta, dict):
            out["cache_control"] = cache_meta
        return out

    def _payload(self, request: LMRequest, stream: bool) -> dict:
        provider_cfg = request.config.provider or {}
        prompt_caching = bool(provider_cfg.get("prompt_caching"))

        messages = [{"role": "user" if m.role == "tool" else m.role, "content": [self._part(p) for p in m.parts]} for m in request.messages]
        if prompt_caching and len(messages) >= 2 and messages[-2].get("content"):
            prev_last = messages[-2]["content"][-1]
            prev_last.setdefault("cache_control", {"type": "ephemeral"})

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": request.config.max_tokens or 1024,
        }
        if request.system:
            if isinstance(request.system, str):
                if prompt_caching:
                    payload["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
                else:
                    payload["system"] = request.system
            else:
                payload["system"] = parts_to_text(tuple(request.system))
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.tools:
            tools_wire: list[dict] = []
            for t in request.tools:
                if t.type == "function":
                    tools_wire.append({"name": t.name, "description": t.description, "input_schema": t.parameters or {"type": "object", "properties": {}}})
                elif t.type == "builtin":
                    tools_wire.append(_builtin_to_anthropic(t))
            payload["tools"] = tools_wire
        if request.config.reasoning and request.config.reasoning.get("enabled"):
            payload["thinking"] = {"type": "enabled", "budget_tokens": request.config.reasoning.get("budget", 1024)}
        if provider_cfg:
            passthrough = {k: v for k, v in provider_cfg.items() if k != "prompt_caching"}
            payload.update(passthrough)
        return payload

    def build_request(self, request: LMRequest, stream: bool) -> HttpRequest:
        return HttpRequest(
            method="POST",
            url=f"{self.base_url}/messages",
            headers=self._headers(),
            json_body=self._payload(request, stream=stream),
            timeout=120.0 if stream else 60.0,
        )

    def parse_response(self, request: LMRequest, response: HttpResponse) -> LMResponse:
        data = response.json()
        parts: list[Part] = []
        for block in data.get("content", []):
            bt = block.get("type")
            if bt == "text":
                parts.append(Part.text_part(block.get("text", "")))
            elif bt == "tool_use":
                parts.append(Part.tool_call(id=block.get("id", ""), name=block.get("name", ""), input=block.get("input", {})))
            elif bt == "thinking":
                parts.append(ThinkingPart(text=block.get("thinking", ""), redacted=False))
            elif bt == "redacted_thinking":
                parts.append(ThinkingPart(text="[redacted]", redacted=True))

        finish = "tool_call" if any(p.type == "tool_call" for p in parts) else "stop"
        usage = Usage(
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            total_tokens=data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
            cache_read_tokens=data.get("usage", {}).get("cache_read_input_tokens"),
            cache_write_tokens=data.get("usage", {}).get("cache_creation_input_tokens"),
        )
        return LMResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            message=Message(role="assistant", parts=tuple(parts or [Part.text_part("")])),
            finish_reason=finish,
            usage=usage,
            provider=data,
        )

    def parse_stream_event(self, request: LMRequest, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None
        payload = json.loads(raw_event.data)
        et = payload.get("type")
        if et == "message_start":
            msg = payload.get("message", {})
            return StreamEvent(type="start", id=msg.get("id"), model=msg.get("model"))
        if et == "content_block_start":
            block = payload.get("content_block", {})
            if block.get("type") == "tool_use":
                return StreamEvent(
                    type="delta",
                    part_index=payload.get("index", 0),
                    delta={
                        "type": "tool_call",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": json.dumps(block.get("input", {})) if isinstance(block.get("input"), dict) else (block.get("input") or ""),
                    },
                )
            pt = block.get("type")
            return StreamEvent(type="part_start", part_index=payload.get("index", 0), part_type=pt)
        if et == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamEvent(type="delta", part_index=payload.get("index", 0), delta=PartDelta(type="text", text=delta.get("text", "")))
            if delta.get("type") == "input_json_delta":
                return StreamEvent(type="delta", part_index=payload.get("index", 0), delta=PartDelta(type="tool_call", input=delta.get("partial_json", "")))
            if delta.get("type") == "thinking_delta":
                return StreamEvent(type="delta", part_index=payload.get("index", 0), delta=PartDelta(type="thinking", text=delta.get("thinking", "")))
            return None
        if et == "content_block_stop":
            return StreamEvent(type="part_end", part_index=payload.get("index", 0))
        if et == "message_stop":
            return StreamEvent(type="end", finish_reason="stop")
        if et == "error":
            e = payload.get("error")
            if isinstance(e, dict):
                provider_code = str(e.get("type") or e.get("code") or payload.get("code") or "provider")
                message = str(e.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return StreamEvent(type="error", error=self._stream_error(provider_code, message))
        return None

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/files",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": request.media_type,
                "x-filename": request.filename,
            },
            body=request.bytes_data,
            timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        file_id = data.get("id") or (data.get("file") or {}).get("id") or ""
        return FileUploadResponse(id=file_id, provider=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        payload = {
            "requests": [
                {
                    "custom_id": f"req_{i}",
                    "params": self._payload(r, stream=False),
                }
                for i, r in enumerate(request.requests)
            ],
            **(request.provider or {}),
        }
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/messages/batches",
            headers=self._headers(),
            json_body=payload,
            timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        batch_id = data.get("id") or f"batch_{uuid.uuid4().hex[:12]}"
        status = data.get("processing_status") or data.get("status") or "submitted"
        return BatchResponse(id=batch_id, status=status, provider=data)
