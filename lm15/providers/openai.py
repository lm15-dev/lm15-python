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
    AudioGenerationRequest,
    AudioGenerationResponse,
    BatchRequest,
    BatchResponse,
    DataSource,
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LMRequest,
    LMResponse,
    Message,
    Part,
    PartDelta,
    StreamEvent,
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
from .common import message_to_openai_input


@dataclass(slots=True)
class OpenAIAdapter(BaseProviderAdapter):
    api_key: str
    transport: Transport
    base_url: str = "https://api.openai.com/v1"

    provider: str = "openai"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "audio", "video", "document"}),
        output_modalities=frozenset({"text", "audio", "image"}),
        features=frozenset({"streaming", "tools", "json_output", "reasoning", "live", "embeddings", "files", "batch", "images", "audio"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(
        complete=True,
        stream=True,
        embeddings=True,
        files=True,
        batches=True,
        images=True,
        audio=True,
        responses_api=True,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="openai",
        supports=supports,
        auth_modes=("bearer",),
        enterprise_variants=("azure-openai",),
        env_keys=("OPENAI_API_KEY",),
    )

    # Responses API Response.error codes:
    # https://developers.openai.com/api/reference/responses/create
    _response_error_code_map: ClassVar[dict[str, type[ProviderError]]] = {
        "server_error": ServerError,
        "rate_limit_exceeded": RateLimitError,
        "invalid_prompt": InvalidRequestError,
        "vector_store_timeout": TimeoutError,
        "invalid_image": InvalidRequestError,
        "invalid_image_format": InvalidRequestError,
        "invalid_base64_image": InvalidRequestError,
        "invalid_image_url": InvalidRequestError,
        "image_too_large": InvalidRequestError,
        "image_too_small": InvalidRequestError,
        "image_parse_error": InvalidRequestError,
        "image_content_policy_violation": InvalidRequestError,
        "invalid_image_mode": InvalidRequestError,
        "image_file_too_large": InvalidRequestError,
        "unsupported_image_media_type": InvalidRequestError,
        "empty_image_file": InvalidRequestError,
        "failed_to_download_image": InvalidRequestError,
        "image_file_not_found": InvalidRequestError,
    }

    _stream_error_code_map: ClassVar[dict[str, type[ProviderError]]] = {
        **_response_error_code_map,
        "context_length_exceeded": ContextLengthError,
        "invalid_api_key": AuthError,
        "insufficient_quota": BillingError,
        "authentication_error": AuthError,
        "rate_limit_error": RateLimitError,
    }

    def _response_error(self, code: str, message: str) -> ProviderError:
        cls = self._response_error_code_map.get(code, ServerError)
        msg = message
        if code and code not in msg:
            msg = f"{msg} ({code})"
        return cls(msg)

    def _stream_error(self, provider_code: str, message: str) -> dict[str, str]:
        cls = self._stream_error_code_map.get(provider_code, ProviderError)
        return {
            "code": canonical_error_code(cls),
            "message": message,
            "provider_code": provider_code or "provider",
        }

    def normalize_error(self, status: int, body: str) -> ProviderError:
        """Extract message from OpenAI error shape.

        Shape: ``{"error": {"message": "...", "type": "...", "code": "..."}}``
        Source: https://developers.openai.com/docs/guides/error-codes
        """
        try:
            data = json.loads(body)
            err = data.get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            code = err.get("code", "") if isinstance(err, dict) else ""
            err_type = err.get("type", "") if isinstance(err, dict) else ""

            # Structured code/type detection
            if code == "context_length_exceeded":
                return ContextLengthError(msg)
            # insufficient_quota is a billing issue, not a rate limit
            if code == "insufficient_quota" or err_type == "insufficient_quota":
                return BillingError(msg)
            if code == "invalid_api_key" or err_type == "authentication_error":
                return AuthError(msg)
            if code == "rate_limit_exceeded" or err_type == "rate_limit_error":
                return RateLimitError(msg)

            if code and code not in msg:
                msg = f"{msg} ({code})"
        except Exception:
            msg = body.strip()[:200] or f"HTTP {status}"
        return map_http_error(status, msg)

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
        }

    def _payload(self, request: LMRequest, stream: bool) -> dict:
        payload = {
            "model": request.model,
            "input": [message_to_openai_input(m) for m in request.messages],
            "stream": stream,
        }
        if request.system:
            payload["instructions"] = request.system if isinstance(request.system, str) else ""
        if request.config.max_tokens is not None:
            payload["max_output_tokens"] = request.config.max_tokens
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                }
                for t in request.tools
                if t.type == "function"
            ]
        if request.config.response_format:
            payload.update(request.config.response_format)
        if request.config.provider:
            passthrough = {k: v for k, v in request.config.provider.items() if k != "prompt_caching"}
            payload.update(passthrough)
        return payload

    def build_request(self, request: LMRequest, stream: bool) -> HttpRequest:
        return HttpRequest(
            method="POST",
            url=f"{self.base_url}/responses",
            headers=self._headers(),
            json_body=self._payload(request, stream=stream),
            timeout=120.0 if stream else 60.0,
        )

    def parse_response(self, request: LMRequest, response: HttpResponse) -> LMResponse:
        data = response.json()

        # The Responses API can return in-band errors on 200 responses
        # (e.g. background/async failures). Check Response.error field.
        resp_error = data.get("error")
        if resp_error and isinstance(resp_error, dict):
            code = str(resp_error.get("code", ""))
            msg = str(resp_error.get("message", str(resp_error)))
            raise self._response_error(code, msg)

        parts: list[Part] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    ctype = c.get("type")
                    if ctype in ("output_text", "text"):
                        parts.append(Part.text_part(c.get("text", "")))
                    elif ctype == "refusal":
                        parts.append(Part.refusal(c.get("refusal", "")))
                    elif ctype == "output_image":
                        b64 = c.get("b64_json") or c.get("image_base64") or ""
                        if b64:
                            parts.append(Part(type="image", source=DataSource(type="base64", media_type="image/png", data=b64)))
                    elif ctype == "output_audio":
                        b64 = c.get("audio", {}).get("data") or c.get("b64_json") or ""
                        if b64:
                            parts.append(Part(type="audio", source=DataSource(type="base64", media_type="audio/wav", data=b64)))
            elif item.get("type") == "function_call":
                args = item.get("arguments")
                parsed_args = json.loads(args) if isinstance(args, str) and args else {}
                parts.append(Part.tool_call(id=item.get("call_id", ""), name=item.get("name", ""), input=parsed_args))

        if not parts:
            parts = [Part.text_part(data.get("output_text", ""))]

        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            reasoning_tokens=usage_data.get("output_tokens_details", {}).get("reasoning_tokens"),
            cache_read_tokens=usage_data.get("input_tokens_details", {}).get("cached_tokens"),
        )

        finish = "tool_call" if any(p.type == "tool_call" for p in parts) else "stop"
        return LMResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason=finish,
            usage=usage,
            provider=data,
        )

    def parse_stream_event(self, request: LMRequest, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None
        if raw_event.data == "[DONE]":
            return StreamEvent(type="end", finish_reason="stop")
        payload = json.loads(raw_event.data)
        et = payload.get("type")
        if et == "response.created":
            return StreamEvent(type="start", id=payload.get("response", {}).get("id"), model=request.model)
        if et in {"response.output_text.delta", "response.refusal.delta"}:
            return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="text", text=payload.get("delta", "")))
        if et == "response.output_audio.delta":
            return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="audio", data=payload.get("delta", "")))
        if et == "response.output_item.added":
            item = payload.get("item", {})
            if item.get("type") == "function_call":
                return StreamEvent(
                    type="delta",
                    part_index=int(payload.get("output_index", 0)),
                    delta={
                        "type": "tool_call",
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "input": item.get("arguments") or "",
                    },
                )
            return None
        if et == "response.function_call_arguments.delta":
            return StreamEvent(
                type="delta",
                part_index=int(payload.get("output_index", 0)),
                delta={
                    "type": "tool_call",
                    "id": payload.get("call_id"),
                    "name": payload.get("name"),
                    "input": payload.get("delta", ""),
                },
            )
        if et == "response.completed":
            response = payload.get("response", {})
            u = response.get("usage", {})
            usage = Usage(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0), total_tokens=u.get("total_tokens", 0))
            finish_reason = "tool_call" if any(item.get("type") == "function_call" for item in response.get("output", [])) else "stop"
            return StreamEvent(type="end", finish_reason=finish_reason, usage=usage)
        if et in {"response.error", "error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return StreamEvent(type="error", error=self._stream_error(provider_code, message))
        return None

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/embeddings",
            headers=self._headers(),
            json_body={"model": request.model, "input": list(request.inputs), **(request.provider or {})},
            timeout=60.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        vectors = tuple(tuple(float(v) for v in item.get("embedding", [])) for item in data.get("data", []))
        u = data.get("usage", {})
        usage = Usage(input_tokens=u.get("prompt_tokens", 0), output_tokens=0, total_tokens=u.get("total_tokens", 0))
        return EmbeddingResponse(model=data.get("model", request.model), vectors=vectors, usage=usage, provider=data)

    def _multipart_file_body(self, *, purpose: str, filename: str, media_type: str, data: bytes) -> tuple[str, bytes]:
        boundary = f"lm15-{uuid.uuid4().hex}"
        lines: list[bytes] = []

        def add(s: str) -> None:
            lines.append(s.encode("utf-8"))

        add(f"--{boundary}\r\n")
        add('Content-Disposition: form-data; name="purpose"\r\n\r\n')
        add(f"{purpose}\r\n")

        add(f"--{boundary}\r\n")
        add(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n')
        add(f"Content-Type: {media_type}\r\n\r\n")
        lines.append(data)
        add("\r\n")

        add(f"--{boundary}--\r\n")
        return boundary, b"".join(lines)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        purpose = (request.provider or {}).get("purpose", "assistants")
        boundary, body = self._multipart_file_body(
            purpose=purpose,
            filename=request.filename,
            media_type=request.media_type,
            data=request.bytes_data,
        )
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/files",
            headers=self._headers(content_type=f"multipart/form-data; boundary={boundary}"),
            body=body,
            timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        return FileUploadResponse(id=data.get("id", ""), provider=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        # Native /batches path requires input_file_id. If not provided, fallback to local fan-out.
        provider = request.provider or {}
        input_file_id = provider.get("input_file_id")
        if input_file_id:
            payload = {
                "input_file_id": input_file_id,
                "endpoint": provider.get("endpoint", "/v1/responses"),
                "completion_window": provider.get("completion_window", "24h"),
            }
            req = HttpRequest(method="POST", url=f"{self.base_url}/batches", headers=self._headers(), json_body=payload, timeout=120.0)
            resp = self.transport.request(req)
            if resp.status >= 400:
                raise self.normalize_error(resp.status, resp.text())
            data = resp.json()
            return BatchResponse(id=data.get("id", ""), status=data.get("status", "submitted"), provider=data)

        results: list[dict[str, Any]] = []
        for r in request.requests:
            out = self.complete(r)
            results.append(
                {
                    "id": out.id,
                    "finish_reason": out.finish_reason,
                    "usage": {
                        "input_tokens": out.usage.input_tokens,
                        "output_tokens": out.usage.output_tokens,
                        "total_tokens": out.usage.total_tokens,
                    },
                }
            )
        return BatchResponse(id=f"batch_{uuid.uuid4().hex[:12]}", status="completed", provider={"results": results})

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "size": request.size,
            **(request.provider or {}),
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        req = HttpRequest(method="POST", url=f"{self.base_url}/images/generations", headers=self._headers(), json_body=payload, timeout=120.0)
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        images = []
        for d in data.get("data", []):
            if d.get("b64_json"):
                images.append(DataSource(type="base64", media_type="image/png", data=d["b64_json"]))
            elif d.get("url"):
                images.append(DataSource(type="url", url=d["url"], media_type="image/png"))
        return ImageGenerationResponse(images=tuple(images), provider=data)

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        payload = {
            "model": request.model,
            "input": request.prompt,
            "voice": request.voice or "alloy",
            "format": request.format or "wav",
            **(request.provider or {}),
        }
        req = HttpRequest(method="POST", url=f"{self.base_url}/audio/speech", headers=self._headers(), json_body=payload, timeout=120.0)
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())

        ctype = (resp.headers.get("content-type") or "audio/wav").split(";")[0].strip()
        b64 = resp.body.decode("utf-8", errors="ignore")
        # if endpoint returns binary, wrap as latin1-preserving base64 via provider passthrough is unavailable;
        # prefer explicit base64 in provider-specific mode when needed.
        try:
            payload_json = json.loads(resp.body)
            if isinstance(payload_json, dict):
                if payload_json.get("audio"):
                    b64 = payload_json.get("audio")
                elif payload_json.get("b64_json"):
                    b64 = payload_json.get("b64_json")
        except Exception:
            import base64

            b64 = base64.b64encode(resp.body).decode("ascii")

        return AudioGenerationResponse(audio=DataSource(type="base64", media_type=ctype, data=b64), provider={"content_type": ctype})
