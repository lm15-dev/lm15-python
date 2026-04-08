from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
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
    Config,
    LMRequest,
    LMResponse,
    Message,
    Part,
    PartDelta,
    StreamEvent,
    Usage,
)
from .base import BaseProviderAdapter


@dataclass(slots=True)
class GeminiAdapter(BaseProviderAdapter):
    api_key: str
    transport: Transport
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    upload_base_url: str = "https://generativelanguage.googleapis.com/upload/v1beta"
    _cached_content_ids: dict[str, str] = field(default_factory=dict, repr=False)

    provider: str = "gemini"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "audio", "video", "document"}),
        output_modalities=frozenset({"text", "image", "audio"}),
        features=frozenset({"streaming", "tools", "json_output", "live", "embeddings", "files", "batch", "images", "audio"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(
        complete=True,
        stream=True,
        embeddings=True,
        files=True,
        batches=True,
        images=True,
        audio=True,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="gemini", supports=supports, auth_modes=("query-api-key", "bearer"))

    def _model_path(self, model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    def _part(self, p: Part) -> dict[str, Any]:
        if p.type == "text":
            return {"text": p.text or ""}
        if p.type in {"image", "audio", "video", "document"} and p.source:
            mime = p.source.media_type or "application/octet-stream"
            if p.source.type == "url":
                return {"fileData": {"mimeType": mime, "fileUri": p.source.url}}
            if p.source.type == "base64":
                return {"inlineData": {"mimeType": mime, "data": p.source.data}}
            if p.source.type == "file":
                return {"fileData": {"mimeType": mime, "fileUri": p.source.file_id}}
        if p.type == "tool_result":
            return {
                "functionResponse": {
                    "name": p.name or "tool",
                    "response": {"result": {"text": "".join(x.text or "" for x in p.content if x.type == "text")}},
                }
            }
        return {"text": p.text or ""}

    def _payload(self, request: LMRequest) -> dict[str, Any]:
        provider_cfg = request.config.provider or {}
        prompt_caching = bool(provider_cfg.get("prompt_caching"))

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [self._part(p) for p in m.parts],
                }
                for m in request.messages
            ]
        }
        if request.system:
            text = request.system if isinstance(request.system, str) else "\n".join(p.text or "" for p in request.system)
            payload["systemInstruction"] = {"parts": [{"text": text}]}

        cfg: dict[str, Any] = {}
        if request.config.temperature is not None:
            cfg["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            cfg["maxOutputTokens"] = request.config.max_tokens
        if request.config.stop:
            cfg["stopSequences"] = list(request.config.stop)
        if request.config.response_format:
            cfg.update(request.config.response_format)
        if cfg:
            payload["generationConfig"] = cfg

        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters or {"type": "OBJECT", "properties": {}},
                        }
                        for t in request.tools
                        if t.type == "function"
                    ]
                }
            ]

        if prompt_caching:
            self._apply_prompt_cache(request, payload)

        if provider_cfg:
            passthrough = {k: v for k, v in provider_cfg.items() if k != "prompt_caching"}
            payload.update(passthrough)
        return payload

    def _apply_prompt_cache(self, request: LMRequest, payload: dict[str, Any]) -> None:
        contents = payload.get("contents") or []
        if len(contents) < 2:
            return

        prefix = contents[:-1]
        key_payload = {
            "model": self._model_path(request.model),
            "systemInstruction": payload.get("systemInstruction"),
            "contents": prefix,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_id = self._cached_content_ids.get(key)

        if cache_id is None:
            body: dict[str, Any] = {
                "model": self._model_path(request.model),
                "contents": prefix,
            }
            if payload.get("systemInstruction"):
                body["systemInstruction"] = payload["systemInstruction"]
            req = HttpRequest(
                method="POST",
                url=f"{self.base_url}/cachedContents",
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json_body=body,
                timeout=60.0,
            )
            resp = self.transport.request(req)
            if resp.status < 400:
                data = resp.json()
                cache_id = data.get("name")
                if cache_id:
                    self._cached_content_ids[key] = cache_id

        if cache_id:
            payload["cachedContent"] = cache_id
            payload["contents"] = contents[-1:]
            payload.pop("systemInstruction", None)

    def build_request(self, request: LMRequest, stream: bool) -> HttpRequest:
        endpoint = "streamGenerateContent" if stream else "generateContent"
        params = {"key": self.api_key}
        if stream:
            params["alt"] = "sse"
        return HttpRequest(
            method="POST",
            url=f"{self.base_url}/{self._model_path(request.model)}:{endpoint}",
            headers={"Content-Type": "application/json"},
            params=params,
            json_body=self._payload(request),
            timeout=120.0 if stream else 60.0,
        )

    def _parse_candidate_parts(self, parts_payload: list[dict[str, Any]]) -> list[Part]:
        parts: list[Part] = []
        for p in parts_payload:
            if "text" in p:
                parts.append(Part.text_part(p["text"]))
            elif "functionCall" in p:
                fc = p["functionCall"]
                parts.append(Part.tool_call(id=fc.get("id", "fc_0"), name=fc.get("name", ""), input=fc.get("args", {})))
            elif "inlineData" in p:
                inline = p["inlineData"]
                mime = inline.get("mimeType", "application/octet-stream")
                data = inline.get("data", "")
                if mime.startswith("image/"):
                    parts.append(Part(type="image", source=DataSource(type="base64", media_type=mime, data=data)))
                elif mime.startswith("audio/"):
                    parts.append(Part(type="audio", source=DataSource(type="base64", media_type=mime, data=data)))
                else:
                    parts.append(Part(type="document", source=DataSource(type="base64", media_type=mime, data=data)))
            elif "fileData" in p:
                fd = p["fileData"]
                uri = fd.get("fileUri", "")
                mime = fd.get("mimeType", "application/octet-stream")
                if mime.startswith("image/"):
                    parts.append(Part(type="image", source=DataSource(type="url", url=uri, media_type=mime)))
                elif mime.startswith("audio/"):
                    parts.append(Part(type="audio", source=DataSource(type="url", url=uri, media_type=mime)))
                else:
                    parts.append(Part(type="document", source=DataSource(type="url", url=uri, media_type=mime)))

        return parts

    def parse_response(self, request: LMRequest, response: HttpResponse) -> LMResponse:
        data = response.json()
        candidate = (data.get("candidates") or [{}])[0]
        content = candidate.get("content", {})
        parts = self._parse_candidate_parts(content.get("parts", []))

        usage = Usage(
            input_tokens=data.get("usageMetadata", {}).get("promptTokenCount", 0),
            output_tokens=data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
            total_tokens=data.get("usageMetadata", {}).get("totalTokenCount", 0),
        )
        if not parts:
            parts = [Part.text_part("")]
        return LMResponse(
            id=data.get("responseId", ""),
            model=request.model,
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason="tool_call" if any(p.type == "tool_call" for p in parts) else "stop",
            usage=usage,
            provider=data,
        )

    def parse_stream_event(self, request: LMRequest, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None
        payload = json.loads(raw_event.data)
        if "error" in payload:
            e = payload["error"]
            return StreamEvent(type="error", error={"code": str(e.get("code", "provider")), "message": e.get("message", "")})

        cands = payload.get("candidates") or []
        if not cands:
            return None
        part = (cands[0].get("content", {}).get("parts") or [{}])[0]
        if "text" in part:
            return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="text", text=part["text"]))
        if "functionCall" in part:
            fc = part["functionCall"]
            return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="tool_call", input=json.dumps(fc.get("args", {}))))
        if "inlineData" in part:
            inline = part["inlineData"]
            mime = inline.get("mimeType", "application/octet-stream")
            if mime.startswith("audio/"):
                return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="audio", data=inline.get("data", "")))
        return None

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_path = self._model_path(request.model)
        if len(request.inputs) <= 1:
            payload = {
                "model": model_path,
                "content": {"parts": [{"text": request.inputs[0] if request.inputs else ""}]},
                **(request.provider or {}),
            }
            req = HttpRequest(
                method="POST",
                url=f"{self.base_url}/{model_path}:embedContent",
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json_body=payload,
                timeout=60.0,
            )
            resp = self.transport.request(req)
            if resp.status >= 400:
                raise self.normalize_error(resp.status, resp.text())
            data = resp.json()
            values = tuple(float(v) for v in (data.get("embedding", {}) or {}).get("values", []))
            return EmbeddingResponse(model=request.model, vectors=(values,), provider=data)

        payload = {
            "requests": [{"model": model_path, "content": {"parts": [{"text": x}]}} for x in request.inputs],
            **(request.provider or {}),
        }
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/{model_path}:batchEmbedContents",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json_body=payload,
            timeout=60.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        vectors = tuple(tuple(float(v) for v in (e.get("values") or [])) for e in data.get("embeddings", []))
        return EmbeddingResponse(model=request.model, vectors=vectors, provider=data)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        req = HttpRequest(
            method="POST",
            url=f"{self.upload_base_url}/files",
            headers={
                "X-Goog-Upload-Protocol": "raw",
                "X-Goog-Upload-File-Name": request.filename,
                "Content-Type": request.media_type,
            },
            params={"key": self.api_key, **(request.provider or {})},
            body=request.bytes_data,
            timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        file_name = (data.get("file") or {}).get("name") or data.get("name") or ""
        return FileUploadResponse(id=file_name, provider=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        results: list[dict[str, Any]] = []
        for r in request.requests:
            resp = self.complete(r)
            results.append({
                "id": resp.id,
                "finish_reason": resp.finish_reason,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            })
        return BatchResponse(id=f"batch_{uuid.uuid4().hex[:12]}", status="completed", provider={"results": results})

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        provider_cfg = {"generationConfig": {"responseModalities": ["IMAGE"]}, **(request.provider or {})}
        lm_req = LMRequest(model=request.model, messages=(Message.user(request.prompt),), config=Config(provider=provider_cfg))
        resp = self.complete(lm_req)
        images = tuple(p.source for p in resp.message.parts if p.type == "image" and p.source is not None)
        return ImageGenerationResponse(images=images, provider=resp.provider)

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        provider_cfg = {
            "generationConfig": {
                "responseModalities": ["AUDIO"],
            },
            **(request.provider or {}),
        }
        if request.voice:
            provider_cfg.setdefault("speechConfig", {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}})

        lm_req = LMRequest(
            model=request.model,
            messages=(Message.user(request.prompt),),
            config=Config(provider=provider_cfg),
        )
        resp = self.complete(lm_req)
        audio_parts = [p for p in resp.message.parts if p.type == "audio" and p.source is not None]
        if not audio_parts:
            raise ValueError("provider did not return audio data")
        return AudioGenerationResponse(audio=audio_parts[0].source, provider=resp.provider)
