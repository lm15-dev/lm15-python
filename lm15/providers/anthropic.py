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
    Usage,
)
from .base import BaseProviderAdapter
from .common import ds_to_anthropic_source, parts_to_text


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
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="anthropic", supports=supports, auth_modes=("x-api-key",))

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def _part(self, p: Part) -> dict[str, Any]:
        if p.type == "text":
            return {"type": "text", "text": p.text or ""}
        if p.type == "image" and p.source:
            return {"type": "image", "source": ds_to_anthropic_source(p.source)}
        if p.type == "document" and p.source:
            return {"type": "document", "source": ds_to_anthropic_source(p.source)}
        if p.type == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": p.id,
                "is_error": bool(p.is_error),
                "content": [{"type": "text", "text": parts_to_text(p.content)}],
            }
        return {"type": "text", "text": p.text or ""}

    def _payload(self, request: LMRequest, stream: bool) -> dict:
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": [self._part(p) for p in m.parts]} for m in request.messages],
            "stream": stream,
            "max_tokens": request.config.max_tokens or 1024,
        }
        if request.system:
            payload["system"] = request.system if isinstance(request.system, str) else parts_to_text(tuple(request.system))
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters or {"type": "object", "properties": {}}}
                for t in request.tools
                if t.type == "function"
            ]
        if request.config.reasoning and request.config.reasoning.get("enabled"):
            payload["thinking"] = {"type": "enabled", "budget_tokens": request.config.reasoning.get("budget", 1024)}
        if request.config.provider:
            payload.update(request.config.provider)
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
                parts.append(Part(type="thinking", text=block.get("thinking", ""), redacted=False))
            elif bt == "redacted_thinking":
                parts.append(Part(type="thinking", text="[redacted]", redacted=True))

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
            pt = "tool_call" if block.get("type") == "tool_use" else block.get("type")
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
            e = payload.get("error", {})
            return StreamEvent(type="error", error={"code": e.get("type", "provider"), "message": e.get("message", "")})
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
