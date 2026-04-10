from __future__ import annotations

import hashlib
import json
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

from ..features import EndpointSupport, ProviderManifest
from ..live import WebSocketLiveSession, require_websocket_sync_connect
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
    LiveClientEvent,
    LiveConfig,
    LiveServerEvent,
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
        live=True,
        embeddings=True,
        files=True,
        batches=True,
        images=True,
        audio=True,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="gemini", supports=supports, auth_modes=("query-api-key", "bearer"), env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"))

    _error_status_map: ClassVar[dict[str, type[ProviderError]]] = {
        "INVALID_ARGUMENT": InvalidRequestError,
        "FAILED_PRECONDITION": BillingError,
        "PERMISSION_DENIED": AuthError,
        "NOT_FOUND": InvalidRequestError,
        "RESOURCE_EXHAUSTED": RateLimitError,
        "INTERNAL": ServerError,
        "UNAVAILABLE": ServerError,
        "DEADLINE_EXCEEDED": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        msg_lower = msg.lower()
        return (
            ("token" in msg_lower and ("limit" in msg_lower or "exceed" in msg_lower))
            or "too long" in msg_lower
            or "context is too long" in msg_lower
            or "context length" in msg_lower
        )

    def _stream_error(self, provider_code: str, message: str) -> dict[str, str]:
        cls = self._error_status_map.get(provider_code, ProviderError)
        if self._is_context_length_message(message):
            cls = ContextLengthError
        return {
            "code": canonical_error_code(cls),
            "message": message,
            "provider_code": provider_code or "provider",
        }

    @staticmethod
    def _is_candidate_finish_error(finish_reason: str) -> bool:
        return finish_reason in {
            "SAFETY",
            "RECITATION",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE",
            "MALFORMED_RESPONSE",
        }

    def _inband_error(self, data: dict[str, Any]) -> ProviderError | None:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = str(prompt_feedback.get("blockReason") or "")
            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                return InvalidRequestError(f"Prompt blocked: {block_reason}")

        candidate = (data.get("candidates") or [{}])[0]
        finish_reason = str(candidate.get("finishReason") or "")
        if self._is_candidate_finish_error(finish_reason):
            finish_message = str(candidate.get("finishMessage") or "")
            msg = finish_message or f"Candidate blocked: {finish_reason}"
            return InvalidRequestError(msg)

        return None

    def normalize_error(self, status: int, body: str) -> ProviderError:
        """Extract message from Gemini error shape: ``{"error": {"message": "...", "status": "...", "code": ...}}``.

        Source: https://ai.google.dev/gemini-api/docs/troubleshooting
        """
        try:
            data = json.loads(body)
            err = data.get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_status = err.get("status", "") if isinstance(err, dict) else ""
            # Context overflow: Gemini conflates this with RESOURCE_EXHAUSTED
            # (429) and INTERNAL (500). No structured code exists — message
            # matching is the only option.
            if self._is_context_length_message(msg):
                return ContextLengthError(msg)

            # Structured error.status → typed error class.
            # Exhaustive per Gemini docs as of 2025-06.
            cls = self._error_status_map.get(err_status)
            if cls:
                return cls(msg)

            if err_status and err_status not in msg:
                msg = f"{msg} ({err_status})"
        except Exception:
            msg = body.strip()[:200] or f"HTTP {status}"
        return map_http_error(status, msg)

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

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"x-goog-api-key": self.api_key}
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _auth_params(extra: dict[str, str] | None = None) -> dict[str, str]:
        return dict(extra or {})

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
                headers=self._auth_headers({"Content-Type": "application/json"}),
                params=self._auth_params(),
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
        params = self._auth_params({"alt": "sse"} if stream else None)
        return HttpRequest(
            method="POST",
            url=f"{self.base_url}/{self._model_path(request.model)}:{endpoint}",
            headers=self._auth_headers({"Content-Type": "application/json"}),
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

        inband_err = self._inband_error(data)
        if inband_err is not None:
            raise inband_err

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
            provider_code = str(e.get("status") or e.get("code") or "provider") if isinstance(e, dict) else "provider"
            message = str(e.get("message", "")) if isinstance(e, dict) else ""
            return StreamEvent(type="error", error=self._stream_error(provider_code, message))

        inband_err = self._inband_error(payload)
        if inband_err is not None:
            return StreamEvent(
                type="error",
                error={
                    "code": canonical_error_code(inband_err),
                    "provider_code": "inband_finish_reason",
                    "message": str(inband_err),
                },
            )

        cands = payload.get("candidates") or []
        if not cands:
            return None
        part = (cands[0].get("content", {}).get("parts") or [{}])[0]
        if "text" in part:
            return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="text", text=part["text"]))
        if "functionCall" in part:
            fc = part["functionCall"]
            return StreamEvent(
                type="delta",
                part_index=0,
                delta={
                    "type": "tool_call",
                    "id": fc.get("id", "fc_0"),
                    "name": fc.get("name", ""),
                    "input": json.dumps(fc.get("args", {})),
                },
            )
        if "inlineData" in part:
            inline = part["inlineData"]
            mime = inline.get("mimeType", "application/octet-stream")
            if mime.startswith("audio/"):
                return StreamEvent(type="delta", part_index=0, delta=PartDelta(type="audio", data=inline.get("data", "")))
        return None

    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        if self._should_use_live_completion(request):
            yield from self._stream_via_live_completion(request)
            return
        yield from super(GeminiAdapter, self).stream(request)

    def _should_use_live_completion(self, request: LMRequest) -> bool:
        provider_cfg = request.config.provider or {}
        transport_mode = str(provider_cfg.get("transport") or "").lower()
        if transport_mode in {"live", "websocket", "ws"}:
            return True
        model_name = request.model.lower()
        return "-live" in model_name or model_name.endswith("live")

    def _stream_via_live_completion(self, request: LMRequest) -> Iterator[StreamEvent]:
        ws = self._live_connect(self._live_url())
        saw_tool_call = False

        try:
            ws.send(json.dumps(self._live_setup_payload_from_request(request)))
            ws.send(json.dumps(self._live_client_content_payload_from_request(request)))

            yield StreamEvent(type="start", model=request.model)

            while True:
                raw = ws.recv()
                events, turn_complete, usage = self._decode_live_completion_stream_events(raw)
                for evt in events:
                    if evt.type == "delta":
                        d = evt.delta
                        if isinstance(d, dict) and d.get("type") == "tool_call":
                            saw_tool_call = True
                        elif isinstance(d, PartDelta) and d.type == "tool_call":
                            saw_tool_call = True
                        yield evt
                    elif evt.type == "error":
                        yield evt
                        return

                if turn_complete:
                    yield StreamEvent(type="end", finish_reason="tool_call" if saw_tool_call else "stop", usage=usage)
                    return
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _live_setup_payload_from_request(self, request: LMRequest) -> dict[str, Any]:
        provider_cfg = dict(request.config.provider or {})
        provider_cfg.pop("transport", None)
        provider_cfg.pop("prompt_caching", None)
        provider_cfg.pop("output", None)

        cfg = LiveConfig(
            model=request.model,
            system=request.system,
            tools=request.tools,
            provider=provider_cfg or None,
        )
        payload = self._live_setup_payload(cfg)

        output = (request.config.provider or {}).get("output")
        if output == "audio":
            setup = payload.setdefault("setup", {})
            generation = setup.setdefault("generationConfig", {})
            generation["responseModalities"] = ["AUDIO"]

        return payload

    def _live_client_content_payload_from_request(self, request: LMRequest) -> dict[str, Any]:
        turns = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [self._part(p) for p in m.parts],
            }
            for m in request.messages
        ]
        return {"clientContent": {"turns": turns, "turnComplete": True}}

    def _decode_live_completion_stream_events(self, raw: str | bytes) -> tuple[list[StreamEvent], bool, Usage]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return [], False, Usage()

        if not isinstance(payload, dict):
            return [], False, Usage()

        if "error" in payload:
            e = payload["error"]
            provider_code = str(e.get("status") or e.get("code") or "provider") if isinstance(e, dict) else "provider"
            message = str(e.get("message", "")) if isinstance(e, dict) else ""
            return [StreamEvent(type="error", error=self._stream_error(provider_code, message))], False, Usage()

        events: list[StreamEvent] = []
        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events, False, Usage()

        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for idx, part in enumerate(model_turn.get("parts", []) or []):
                if "text" in part:
                    events.append(StreamEvent(type="delta", part_index=idx, delta=PartDelta(type="text", text=str(part.get("text") or ""))))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(
                        StreamEvent(
                            type="delta",
                            part_index=idx,
                            delta={
                                "type": "tool_call",
                                "id": fc.get("id", "fc_0"),
                                "name": fc.get("name", "tool"),
                                "input": json.dumps(fc.get("args", {})),
                            },
                        )
                    )
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    if mime.startswith("audio/"):
                        events.append(StreamEvent(type="delta", part_index=idx, delta=PartDelta(type="audio", data=str(inline.get("data") or ""))))

        turn_complete = bool(server.get("turnComplete"))
        usage_payload = payload.get("usageMetadata")
        if not isinstance(usage_payload, dict):
            usage_payload = server.get("usageMetadata") if isinstance(server.get("usageMetadata"), dict) else {}
        usage_payload = usage_payload if isinstance(usage_payload, dict) else {}
        usage = Usage(
            input_tokens=int(usage_payload.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage_payload.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage_payload.get("totalTokenCount", 0) or 0),
        )
        return events, turn_complete, usage

    def live(self, config: LiveConfig):
        ws = self._live_connect(self._live_url())
        ws.send(json.dumps(self._live_setup_payload(config)))

        callable_registry = {
            t.name: t.fn
            for t in config.tools
            if t.type == "function" and callable(t.fn)
        }

        return WebSocketLiveSession(
            ws=ws,
            encode_event=self._encode_live_client_event,
            decode_event=self._decode_live_server_event,
            callable_registry=callable_registry,
        )

    def _live_connect(self, url: str):
        connect = require_websocket_sync_connect()
        return connect(url)

    def _live_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        query = urllib.parse.urlencode({"key": self.api_key})
        return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _live_setup_payload(self, config: LiveConfig) -> dict[str, Any]:
        setup: dict[str, Any] = {
            "model": self._model_path(config.model),
        }

        if config.system:
            if isinstance(config.system, str):
                text = config.system
            else:
                text = "\n".join(p.text or "" for p in config.system if p.type in {"text", "thinking", "refusal"})
            setup["systemInstruction"] = {"parts": [{"text": text}]}

        function_tools = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            }
            for t in config.tools
            if t.type == "function"
        ]
        if function_tools:
            setup["tools"] = [{"functionDeclarations": function_tools}]

        generation_config: dict[str, Any] = {}
        if config.output_format is not None:
            generation_config["responseModalities"] = ["AUDIO"]
        if generation_config:
            setup["generationConfig"] = generation_config

        if config.voice:
            setup.setdefault("speechConfig", {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}})

        if config.provider:
            setup.update(config.provider)

        return {"setup": setup}

    def _encode_live_client_event(self, event: LiveClientEvent) -> list[dict[str, Any]]:
        if event.type == "audio":
            return [{"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm", "data": event.data}]}}]

        if event.type == "video":
            return [{"realtimeInput": {"mediaChunks": [{"mimeType": "video/mp4", "data": event.data}]}}]

        if event.type == "interrupt":
            return [{"clientContent": {"turnComplete": True}}]

        if event.type == "end_audio":
            return [{"realtimeInput": {"audioStreamEnd": True}}]

        if event.type == "text":
            parts: list[dict[str, Any]] = [{"text": event.text or ""}]
            parts.extend(self._part(p) for p in event.content)
            return [{
                "clientContent": {
                    "turns": [{"role": "user", "parts": parts}],
                    "turnComplete": True,
                }
            }]

        if event.type == "tool_result":
            part = self._part(Part.tool_result(event.id, list(event.content), name=None))
            return [{
                "clientContent": {
                    "turns": [{"role": "user", "parts": [part]}],
                    "turnComplete": True,
                }
            }]

        return []

    def _decode_live_server_event(self, raw: str | bytes) -> list[LiveServerEvent]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []

        if "error" in payload:
            err = payload.get("error")
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            return [LiveServerEvent(type="error", error=self._stream_error(provider_code, message))]

        events: list[LiveServerEvent] = []
        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events

        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for part in model_turn.get("parts", []) or []:
                if "text" in part:
                    events.append(LiveServerEvent(type="text", text=str(part.get("text") or "")))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    if mime.startswith("audio/"):
                        events.append(LiveServerEvent(type="audio", data=str(inline.get("data") or "")))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    call_id = str(fc.get("id") or "fc_0")
                    name = str(fc.get("name") or "tool")
                    args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                    events.append(LiveServerEvent(type="tool_call", id=call_id, name=name, input=args))

        if server.get("interrupted"):
            events.append(LiveServerEvent(type="interrupted"))

        if server.get("turnComplete"):
            usage_payload = payload.get("usageMetadata")
            if not isinstance(usage_payload, dict):
                usage_payload = server.get("usageMetadata")
            usage_payload = usage_payload if isinstance(usage_payload, dict) else {}
            usage = Usage(
                input_tokens=int(usage_payload.get("promptTokenCount", 0) or 0),
                output_tokens=int(usage_payload.get("candidatesTokenCount", 0) or 0),
                total_tokens=int(usage_payload.get("totalTokenCount", 0) or 0),
            )
            events.append(LiveServerEvent(type="turn_end", usage=usage))

        return events

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
                headers=self._auth_headers({"Content-Type": "application/json"}),
                params=self._auth_params(),
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
            headers=self._auth_headers({"Content-Type": "application/json"}),
            params=self._auth_params(),
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
            headers=self._auth_headers(
                {
                    "X-Goog-Upload-Protocol": "raw",
                    "X-Goog-Upload-File-Name": request.filename,
                    "Content-Type": request.media_type,
                }
            ),
            params=self._auth_params(request.provider),
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
