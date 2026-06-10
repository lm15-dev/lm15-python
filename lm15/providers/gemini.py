from __future__ import annotations

import base64
import hashlib
import json
import struct
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

from ..errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
    UnsupportedModelError,
    canonical_error_code,
    map_http_error,
)
from ..features import EndpointSupport, ProviderManifest
from ..live import WebSocketLiveSession, require_websocket_sync_connect
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports import TransportRequest
from ..types import (
    AudioDelta,
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioPart,
    BatchRequest,
    ContinuationDelta,
    ContinuationState,
    BatchResponse,
    BinaryPart,
    BuiltinTool,
    CacheConfig,
    CitationPart,
    Config,
    DocumentPart,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    FileUploadRequest,
    FileUploadResponse,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveClientAudioEvent,
    LiveClientEndAudioEvent,
    LiveClientEvent,
    LiveClientImageEvent,
    LiveClientInterruptEvent,
    LiveClientTextEvent,
    LiveClientToolResultEvent,
    LiveClientTurnEvent,
    LiveConfig,
    LiveServerAudioEvent,
    LiveServerErrorEvent,
    LiveServerInterruptedEvent,
    LiveServerTextEvent,
    LiveServerToolCallDeltaEvent,
    LiveServerToolCallEvent,
    LiveServerTurnEndEvent,
    Message,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolResultPart,
    continuation_data,
    Usage,
    VideoPart,
)
from .base import BaseProviderLM, HttpResponse, SyncTransport, default_transport
from .common import build_url, make_json_request, parse_json_object, parts_to_text

# Canonical builtin tool name → Gemini tool key
_GEMINI_BUILTIN_MAP: dict[str, str] = {
    "web_search": "googleSearch",
    "code_execution": "codeExecution",
}

GEMINI_PROVIDER_EXECUTED_PART_KEYS = {
    "executableCode",
    "codeExecutionResult",
}


def _attach_unmapped(provider_data: dict[str, Any], unmapped: list[dict[str, str]]) -> dict[str, Any]:
    if not unmapped:
        return provider_data
    out = dict(provider_data)
    out["_lm15_unmapped"] = unmapped
    return out


def _record_unmapped(unmapped: list[dict[str, str]], path: str, typ: Any) -> None:
    unmapped.append({"path": path, "type": str(typ or "<missing>")})


def _builtin_to_gemini(tool: BuiltinTool) -> dict[str, Any]:
    return {_GEMINI_BUILTIN_MAP.get(tool.name, tool.name): tool.config or {}}


def _gemini_schema_field(schema: dict[str, Any]) -> str:
    """Pick Gemini's schema field for an lm15 JSON schema.

    ``responseSchema`` is Gemini's OpenAPI-ish schema type and rejects JSON
    Schema keywords such as ``additionalProperties``.  ``responseJsonSchema``
    accepts those keywords, so use it when the schema needs full JSON Schema.
    """
    return "responseJsonSchema" if _contains_key(schema, "additionalProperties") else "responseSchema"


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def _response_format_to_gemini_config(format_config: dict[str, Any]) -> dict[str, Any]:
    """Map canonical lm15 response_format to Gemini generationConfig."""
    generation_config = format_config.get("generationConfig")
    if isinstance(generation_config, dict):
        return dict(generation_config)

    out: dict[str, Any] = {}
    mime_type = format_config.get("responseMimeType") or format_config.get("response_mime_type")
    schema = format_config.get("responseSchema") or format_config.get("response_schema")
    json_schema = format_config.get("responseJsonSchema") or format_config.get("response_json_schema")

    if mime_type is not None:
        out["responseMimeType"] = str(mime_type)
    if isinstance(schema, dict):
        out["responseSchema"] = schema
    if isinstance(json_schema, dict):
        out["responseJsonSchema"] = json_schema

    fmt_type = format_config.get("type")
    if fmt_type == "json_object":
        out.setdefault("responseMimeType", "application/json")
        return out

    if fmt_type == "json_schema":
        schema = format_config.get("schema")
        if isinstance(schema, dict):
            out[_gemini_schema_field(schema)] = schema
            out.pop("responseSchema" if _gemini_schema_field(schema) == "responseJsonSchema" else "responseJsonSchema", None)
        out.setdefault("responseMimeType", "application/json")
        return out

    schema = format_config.get("schema") if isinstance(format_config.get("schema"), dict) else None
    if schema is not None:
        out[_gemini_schema_field(schema)] = schema
        out.pop("responseSchema" if _gemini_schema_field(schema) == "responseJsonSchema" else "responseJsonSchema", None)
        out.setdefault("responseMimeType", "application/json")
        return out

    if "type" in format_config or "properties" in format_config or "items" in format_config:
        out[_gemini_schema_field(format_config)] = dict(format_config)
        out.setdefault("responseMimeType", "application/json")

    return out or dict(format_config)


def _finish_reason(reason: str | None, *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    r = str(reason or "").upper()
    if r == "MAX_TOKENS":
        return "length"
    if r in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        return "content_filter"
    return "stop"


def _batch_status(status: str) -> str:
    status = status.lower()
    if status in {"completed", "failed", "cancelled"}:
        return status
    if status in {"running", "processing", "in_progress"}:
        return "running"
    if status in {"queued", "validating"}:
        return "queued"
    return "submitted"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gemini_segment_text(segment: dict[str, Any], full_text: str) -> str | None:
    text = segment.get("text")
    if isinstance(text, str) and text:
        return text
    start = _int_or_none(segment.get("startIndex"))
    end = _int_or_none(segment.get("endIndex"))
    if start is not None and end is not None and 0 <= start < end <= len(full_text):
        return full_text[start:end]
    return None


def _gemini_grounding_chunk(chunks: list[Any], index: Any) -> dict[str, Any]:
    idx = _int_or_none(index)
    if idx is None or idx < 0 or idx >= len(chunks):
        return {}
    chunk = chunks[idx]
    return chunk if isinstance(chunk, dict) else {}


def _gemini_citations(candidate: dict[str, Any], full_text: str) -> list[CitationPart]:
    grounding = candidate.get("groundingMetadata")
    if not isinstance(grounding, dict):
        return []

    chunks = grounding.get("groundingChunks") or []
    if not isinstance(chunks, list):
        chunks = []
    supports = grounding.get("groundingSupports") or []
    if not isinstance(supports, list):
        return []

    citations: list[CitationPart] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment") if isinstance(support.get("segment"), dict) else {}
        cited_text = _gemini_segment_text(segment, full_text)
        indices = support.get("groundingChunkIndices") or []
        if not isinstance(indices, list):
            continue
        for index in indices:
            chunk = _gemini_grounding_chunk(chunks, index)
            source = chunk.get("web") or chunk.get("retrievedContext") or chunk.get("googleSearch") or {}
            if not isinstance(source, dict):
                source = {}
            url = source.get("uri") or source.get("url")
            title = source.get("title") or source.get("name")
            url_s = str(url) if url else None
            title_s = str(title) if title else None
            key = (url_s, title_s, cited_text)
            if key in seen or (url_s is None and title_s is None and cited_text is None):
                continue
            seen.add(key)
            citations.append(CitationPart(url=url_s, title=title_s, text=cited_text))
    return citations


@dataclass(slots=True)
class GeminiLM(BaseProviderLM):
    api_key: str
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    upload_base_url: str = "https://generativelanguage.googleapis.com/upload/v1beta"
    _cached_content_ids: dict[str, str] = field(default_factory=dict, repr=False)

    provider: str = "gemini"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "audio", "video", "document", "binary"}),
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
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="gemini",
        supports=supports,
        auth_modes=("query-api-key", "x-goog-api-key"),
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    _error_status_map: ClassVar[dict[str, type[ProviderError]]] = {
        "INVALID_ARGUMENT": InvalidRequestError,
        "FAILED_PRECONDITION": BillingError,
        "PERMISSION_DENIED": AuthError,
        "UNAUTHENTICATED": AuthError,
        "NOT_FOUND": InvalidRequestError,
        "RESOURCE_EXHAUSTED": RateLimitError,
        "INTERNAL": ServerError,
        "UNAVAILABLE": ServerError,
        "DEADLINE_EXCEEDED": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        lowered = msg.lower()
        return (
            ("token" in lowered and ("limit" in lowered or "exceed" in lowered))
            or "too long" in lowered
            or "context is too long" in lowered
            or "context length" in lowered
        )

    @staticmethod
    def _is_model_error(message: str) -> bool:
        lowered = message.lower()
        return "model" in lowered and any(
            marker in lowered
            for marker in (
                "not found",
                "does not exist",
                "not exist",
                "not supported",
                "unsupported",
                "not available",
                "unknown",
            )
        )

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

    def _error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._error_status_map.get(provider_code, ProviderError)
        if self._is_context_length_message(message):
            cls = ContextLengthError
        elif provider_code == "NOT_FOUND" and self._is_model_error(message):
            cls = UnsupportedModelError
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message or provider_code or "provider error",
            provider_code=provider_code or "provider",
        )

    def _inband_error(self, data: dict[str, Any]) -> ProviderError | None:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = str(prompt_feedback.get("blockReason") or "")
            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                return self._provider_error(
                    InvalidRequestError,
                    f"Prompt blocked: {block_reason}",
                    provider_code="promptFeedback",
                )

        candidate = (data.get("candidates") or [{}])[0]
        if isinstance(candidate, dict):
            finish_reason = str(candidate.get("finishReason") or "")
            if self._is_candidate_finish_error(finish_reason):
                finish_message = str(candidate.get("finishMessage") or "")
                return self._provider_error(
                    InvalidRequestError,
                    finish_message or f"Candidate blocked: {finish_reason}",
                    provider_code=finish_reason or "finishReason",
                )
        return None

    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_status = str(err.get("status") or "") if isinstance(err, dict) else ""
            if self._is_context_length_message(msg):
                return self._provider_error(
                    ContextLengthError,
                    msg,
                    status=status,
                    provider_code=err_status or None,
                )
            if err_status == "NOT_FOUND" and self._is_model_error(msg):
                return self._provider_error(
                    UnsupportedModelError,
                    msg,
                    status=status,
                    provider_code=err_status,
                )
            cls = self._error_status_map.get(err_status)
            if cls:
                return self._provider_error(
                    cls,
                    msg,
                    status=status,
                    provider_code=err_status or None,
                )
            if err_status and err_status not in msg:
                msg = f"{msg} ({err_status})"
        except Exception:
            msg = body.strip()[:500] or f"HTTP {status}"
            err_status = ""
        return map_http_error(
            status,
            msg,
            provider=self.provider,
            env_keys=self.manifest.env_keys,
            provider_code=err_status or None,
        )

    # ─── Request serialization ──────────────────────────────────────

    def _model_path(self, model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"x-goog-api-key": self.api_key}
        if extra:
            headers.update(extra)
        return headers

    def _auth_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return dict(extra or {})

    def _part(self, part) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"text": part.text}
        if isinstance(part, (ImagePart, AudioPart, VideoPart, DocumentPart, BinaryPart)):
            mime = part.media_type or "application/octet-stream"
            if part.url is not None:
                return {"fileData": {"mimeType": mime, "fileUri": part.url}}
            if part.file_id is not None:
                return {"fileData": {"mimeType": mime, "fileUri": part.file_id}}
            if part.data is not None:
                return {"inlineData": {"mimeType": mime, "data": part.data}}
            if part.path is not None:
                return {"inlineData": {"mimeType": mime, "data": base64.b64encode(part.path.read_bytes()).decode("ascii")}}
        if isinstance(part, ToolCallPart):
            out: dict[str, Any] = {"functionCall": {"name": part.name, "args": part.input}}
            if part.id:
                out["functionCall"]["id"] = part.id
            thought = continuation_data(part, "gemini", "thought_signature")
            if thought and thought.get("value"):
                out["thoughtSignature"] = thought["value"]
            return out
        if isinstance(part, ToolResultPart):
            result_text = parts_to_text(part.content)
            fr: dict[str, Any] = {"name": part.name or "tool", "response": {"result": result_text}}
            if part.id:
                fr["id"] = part.id
            return {"functionResponse": fr}
        if isinstance(part, ThinkingPart):
            out: dict[str, Any] = {"text": part.text}
            thought = continuation_data(part, "gemini", "thought_signature")
            if thought and thought.get("value"):
                out["thought"] = True
                out["thoughtSignature"] = thought["value"]
            return out
        return {"text": getattr(part, "text", "") or ""}

    def _message(self, msg: Message) -> dict[str, Any]:
        role = "model" if msg.role == "assistant" else "user"
        if msg.role == "developer":
            return {"role": "user", "parts": [{"text": f"[developer]\n{parts_to_text(msg.parts)}"}]}
        return {"role": role, "parts": [self._part(part) for part in msg.parts]}

    def _tool_config_payload(self, request: Request) -> dict[str, Any] | None:
        tc = request.config.tool_choice
        if tc is None:
            return None
        mode = {"none": "NONE", "required": "ANY", "auto": "AUTO"}[tc.mode]
        cfg: dict[str, Any] = {"mode": mode}
        if tc.allowed:
            cfg["allowedFunctionNames"] = list(tc.allowed)
        return {"functionCallingConfig": cfg}

    def _payload(self, request: Request, *, apply_cache: bool = True) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        cache_cfg = request.config.cache
        use_cache = (cache_cfg is None or cache_cfg.mode != "off") and apply_cache

        payload: dict[str, Any] = {"contents": [self._message(m) for m in request.messages]}
        if request.system:
            text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            payload["systemInstruction"] = {"parts": [{"text": text}]}

        generation_config: dict[str, Any] = {}
        if request.config.temperature is not None:
            generation_config["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.config.max_tokens
        if request.config.top_p is not None:
            generation_config["topP"] = request.config.top_p
        if request.config.top_k is not None:
            generation_config["topK"] = request.config.top_k
        if request.config.stop:
            generation_config["stopSequences"] = list(request.config.stop)
        if request.config.response_format:
            generation_config.update(_response_format_to_gemini_config(request.config.response_format))
        if request.config.reasoning is not None:
            if request.config.reasoning.is_off:
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            else:
                thinking: dict[str, Any] = {"includeThoughts": True}
                if request.config.reasoning.thinking_budget is not None:
                    thinking["thinkingBudget"] = request.config.reasoning.thinking_budget
                generation_config["thinkingConfig"] = thinking
        if generation_config:
            payload["generationConfig"] = generation_config

        if request.tools:
            function_declarations = [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in request.tools
                if isinstance(t, FunctionTool)
            ]
            tools_wire: list[dict[str, Any]] = []
            if function_declarations:
                tools_wire.append({"functionDeclarations": function_declarations})
            for tool in request.tools:
                if isinstance(tool, BuiltinTool):
                    tools_wire.append(_builtin_to_gemini(tool))
            payload["tools"] = tools_wire

        tool_config = self._tool_config_payload(request)
        if tool_config is not None:
            payload["toolConfig"] = tool_config

        output = extensions.get("output")
        if output == "image":
            payload.setdefault("generationConfig", {})["responseModalities"] = ["IMAGE"]
        elif output == "audio":
            payload.setdefault("generationConfig", {})["responseModalities"] = ["AUDIO"]

        if use_cache:
            self._apply_prompt_cache(request, payload)

        if extensions:
            passthrough = {k: v for k, v in extensions.items() if k not in {"prompt_caching", "output"}}
            payload.update(passthrough)
        return payload

    def _prompt_cache_plan(self, request: Request, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Compute the cache prefix split and lookup key. Pure — no network."""
        cache_cfg: CacheConfig | None = request.config.cache
        contents = payload.get("contents") or []
        if len(contents) < 2:
            return None

        # Determine prefix
        if cache_cfg is not None and cache_cfg.prefix_until_index is not None:
            prefix_end = min(cache_cfg.prefix_until_index + 1, len(contents))
            prefix = contents[:prefix_end]
            remaining = contents[prefix_end:]
        else:
            prefix = contents[:-1]
            remaining = contents[-1:]

        if not prefix:
            return None

        # Build cache key
        key_parts = {
            "model": self._model_path(request.model),
            "systemInstruction": payload.get("systemInstruction"),
            "contents": prefix,
        }
        if cache_cfg is not None and cache_cfg.key:
            key_parts["user_key"] = cache_cfg.key

        key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode("utf-8")).hexdigest()
        return {"key": key, "prefix": prefix, "remaining": remaining}

    def resolve_prompt_cache(self, request: Request) -> str | None:
        """Create or look up the cachedContents entry for this request's prefix.

        This is the only place the cachedContents endpoint is called. complete()
        and stream() invoke it before sending; build_request never touches the
        network.
        """
        cache_cfg: CacheConfig | None = request.config.cache
        if not (cache_cfg is None or cache_cfg.mode != "off"):
            return None
        payload = self._payload(request, apply_cache=False)
        plan = self._prompt_cache_plan(request, payload)
        if plan is None:
            return None
        cache_id = self._cached_content_ids.get(plan["key"])
        if cache_id is not None:
            return cache_id

        body: dict[str, Any] = {
            "model": self._model_path(request.model),
            "contents": plan["prefix"],
        }
        if payload.get("systemInstruction"):
            body["systemInstruction"] = payload["systemInstruction"]

        # Set TTL if retention="long"
        if cache_cfg is not None and cache_cfg.retention == "long":
            body["ttl"] = "86400s"  # 24 hours

        resp = self._send(make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/cachedContents",
            headers=self._auth_headers({"Content-Type": "application/json"}),
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

    def _apply_prompt_cache(self, request: Request, payload: dict[str, Any]) -> None:
        """Rewrite the payload to use an already-resolved cache id. Pure — no network."""
        plan = self._prompt_cache_plan(request, payload)
        if plan is None:
            return
        cache_id = self._cached_content_ids.get(plan["key"])
        if cache_id:
            payload["cachedContent"] = cache_id
            payload["contents"] = plan["remaining"]
            payload.pop("systemInstruction", None)

    def complete(self, request: Request) -> Response:
        self.resolve_prompt_cache(request)
        # Explicit base call: conformance loads this module under a second
        # sys.path entry, which breaks zero-arg super()'s class-cell check.
        return BaseProviderLM.complete(self, request)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        self.resolve_prompt_cache(request)
        yield from BaseProviderLM.stream(self, request)

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        endpoint = "streamGenerateContent" if stream else "generateContent"
        params = self._auth_params({"alt": "sse"} if stream else None)
        return make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/{self._model_path(request.model)}:{endpoint}",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            params=params,
            payload=self._payload(request),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    def _parse_candidate_parts(
        self,
        parts_payload: list[dict[str, Any]],
        *,
        unmapped: list[dict[str, str]] | None = None,
        path_prefix: str = "parts",
    ) -> list[Any]:
        parts: list[Any] = []
        for part_index, part in enumerate(parts_payload):
            if not isinstance(part, dict):
                if unmapped is not None:
                    _record_unmapped(unmapped, f"{path_prefix}[{part_index}]", type(part).__name__)
                continue
            if "thought" in part and part.get("thought") and part.get("text"):
                continuation: tuple[ContinuationState, ...] = ()
                thought_signature = part.get("thoughtSignature")
                if thought_signature is not None:
                    continuation = (
                        ContinuationState(
                            provider="gemini",
                            kind="thought_signature",
                            data={"value": str(thought_signature)},
                        ),
                    )
                parts.append(ThinkingPart(text=str(part.get("text") or ""), continuation=continuation))
            elif "text" in part:
                parts.append(TextPart(text=str(part.get("text") or "")))
            elif "functionCall" in part and isinstance(part["functionCall"], dict):
                fc = part["functionCall"]
                continuation: tuple[ContinuationState, ...] = ()
                thought_signature = part.get("thoughtSignature") or fc.get("thoughtSignature")
                if thought_signature is not None:
                    continuation = (
                        ContinuationState(
                            provider="gemini",
                            kind="thought_signature",
                            data={"value": str(thought_signature)},
                        ),
                    )
                parts.append(ToolCallPart(
                    id=str(fc.get("id") or f"fc_{len(parts)}"),
                    name=str(fc.get("name") or "tool"),
                    input=fc.get("args") if isinstance(fc.get("args"), dict) else {},
                    continuation=continuation,
                ))
            elif "inlineData" in part and isinstance(part["inlineData"], dict):
                inline = part["inlineData"]
                mime = str(inline.get("mimeType") or "application/octet-stream")
                data = str(inline.get("data") or "")
                if not data:
                    continue
                if mime.startswith("image/"):
                    parts.append(ImagePart(media_type=mime, data=data))
                elif mime.startswith("audio/"):
                    parts.append(AudioPart(media_type=mime, data=data))
                else:
                    parts.append(DocumentPart(media_type=mime, data=data))
            elif "fileData" in part and isinstance(part["fileData"], dict):
                fd = part["fileData"]
                uri = str(fd.get("fileUri") or "")
                mime = str(fd.get("mimeType") or "application/octet-stream")
                if not uri:
                    continue
                if mime.startswith("image/"):
                    parts.append(ImagePart(media_type=mime, url=uri))
                elif mime.startswith("audio/"):
                    parts.append(AudioPart(media_type=mime, url=uri))
                else:
                    parts.append(DocumentPart(media_type=mime, url=uri))
            elif any(key in part for key in GEMINI_PROVIDER_EXECUTED_PART_KEYS):
                continue
            elif unmapped is not None:
                _record_unmapped(unmapped, f"{path_prefix}[{part_index}]", "+".join(sorted(part)) or "<empty>")
        return parts

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()
        inband = self._inband_error(data)
        if inband is not None:
            raise inband
        candidate = (data.get("candidates") or [{}])[0]
        candidate = candidate if isinstance(candidate, dict) else {}
        content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
        unmapped: list[dict[str, str]] = []
        parts = self._parse_candidate_parts(
            content.get("parts", []) or [],
            unmapped=unmapped,
            path_prefix="candidates[0].content.parts",
        )
        full_text = "".join(part.text for part in parts if isinstance(part, TextPart))
        parts.extend(_gemini_citations(candidate, full_text))
        if not parts:
            parts = [TextPart(text="")]
        usage_payload = data.get("usageMetadata") or {}
        usage = Usage(
            input_tokens=int(usage_payload.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage_payload.get("candidatesTokenCount", usage_payload.get("responseTokenCount", 0)) or 0),
            total_tokens=usage_payload.get("totalTokenCount"),
            cache_read_tokens=usage_payload.get("cachedContentTokenCount"),
            reasoning_tokens=usage_payload.get("thoughtsTokenCount"),
        )
        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        message_continuation: tuple[ContinuationState, ...] = ()
        if data.get("responseId"):
            message_continuation = (
                ContinuationState(
                    provider="gemini",
                    kind="response_id",
                    data={"id": str(data.get("responseId"))},
                ),
            )
        return Response(
            id=str(data.get("responseId")) if data.get("responseId") else None,
            model=request.model,
            message=Message(role="assistant", parts=tuple(parts), continuation=message_continuation),
            finish_reason=_finish_reason(candidate.get("finishReason"), has_tool_call=has_tool),
            usage=usage,
            provider_data=_attach_unmapped(data, unmapped),
        )

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        if not raw_event.data:
            return
        payload = json.loads(raw_event.data)
        if not isinstance(payload, dict):
            return
        if "error" in payload:
            err = payload["error"]
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            yield StreamErrorEvent(error=self._error_detail(provider_code, message))
            return

        inband = self._inband_error(payload)
        if inband is not None:
            yield StreamErrorEvent(error=ErrorDetail(code=canonical_error_code(inband), provider_code="inband_finish_reason", message=str(inband)))
            return

        candidates = payload.get("candidates") or []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else None
        yielded_delta = False
        saw_tool = False
        finish = None
        if candidate is not None:
            content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
            for idx, part in enumerate(content.get("parts", []) or []):
                if not isinstance(part, dict):
                    continue
                if part.get("thought") and "text" in part:
                    yielded_delta = True
                    yield StreamDeltaEvent(delta=ThinkingDelta(text=str(part.get("text") or ""), part_index=idx))
                    if part.get("thoughtSignature") is not None:
                        yield StreamDeltaEvent(
                            delta=ContinuationDelta(
                                provider="gemini",
                                kind="thought_signature",
                                data={"value": str(part.get("thoughtSignature"))},
                                part_index=idx,
                            )
                        )
                elif "text" in part:
                    yielded_delta = True
                    yield StreamDeltaEvent(delta=TextDelta(text=str(part.get("text") or ""), part_index=idx))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    saw_tool = True
                    yielded_delta = True
                    yield StreamDeltaEvent(delta=ToolCallDelta(
                        input=json.dumps(fc.get("args", {}), separators=(",", ":")),
                        part_index=idx,
                        id=str(fc.get("id") or "") or None,
                        name=str(fc.get("name") or "") or None,
                    ))
                    thought_signature = part.get("thoughtSignature") or fc.get("thoughtSignature")
                    if thought_signature is not None:
                        yield StreamDeltaEvent(
                            delta=ContinuationDelta(
                                provider="gemini",
                                kind="thought_signature",
                                data={"value": str(thought_signature)},
                                part_index=idx,
                            )
                        )
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "application/octet-stream")
                    data = str(inline.get("data") or "")
                    if mime.startswith("audio/"):
                        yielded_delta = True
                        yield StreamDeltaEvent(delta=AudioDelta(data=data, part_index=idx, media_type=mime))
                    elif mime.startswith("image/"):
                        yielded_delta = True
                        yield StreamDeltaEvent(delta=ImageDelta(data=data, part_index=idx, media_type=mime))
            finish = candidate.get("finishReason")

        if payload.get("responseId") is not None:
            yield StreamDeltaEvent(
                delta=ContinuationDelta(
                    provider="gemini",
                    kind="response_id",
                    data={"id": str(payload.get("responseId"))},
                    part_index=None,
                )
            )
        if finish:
            yield StreamEndEvent(finish_reason=_finish_reason(finish, has_tool_call=saw_tool), usage=self._usage_from_payload(payload), provider_data=payload)
        elif not yielded_delta and "usageMetadata" in payload:
            yield StreamEndEvent(finish_reason="stop", usage=self._usage_from_payload(payload), provider_data=payload)

    def _usage_from_payload(self, payload: dict[str, Any]) -> Usage:
        usage_payload = payload.get("usageMetadata") or {}
        return Usage(
            input_tokens=int(usage_payload.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage_payload.get("candidatesTokenCount", usage_payload.get("responseTokenCount", 0)) or 0),
            total_tokens=usage_payload.get("totalTokenCount"),
            cache_read_tokens=usage_payload.get("cachedContentTokenCount"),
            reasoning_tokens=usage_payload.get("thoughtsTokenCount"),
        )

    # ─── Streaming via Gemini Live for live models ──────────────────

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        if self._should_use_live_completion(request):
            yield from self._stream_via_live_completion(request)
            return
        yield from BaseProviderLM.stream(self, request)

    def _should_use_live_completion(self, request: Request) -> bool:
        extensions = request.config.extensions or {}
        transport_mode = str(extensions.get("transport") or "").lower()
        if transport_mode in {"live", "websocket", "ws"}:
            return True
        model_name = request.model.lower()
        return "-live" in model_name or model_name.endswith("live")

    @staticmethod
    def _is_audio_native_live_model(model: str) -> bool:
        lowered = model.lower()
        return "live-preview" in lowered or "native-audio" in lowered

    @staticmethod
    def _wav_to_pcm(data: bytes) -> tuple[bytes, int]:
        if len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            sample_rate = struct.unpack_from("<I", data, 24)[0]
            pos = 12
            while pos + 8 <= len(data):
                chunk_id = data[pos : pos + 4]
                chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
                if chunk_id == b"data":
                    return data[pos + 8 : pos + 8 + chunk_size], sample_rate
                pos += 8 + chunk_size
            return data[44:], sample_rate
        return data, 16000

    def _stream_via_live_completion(self, request: Request) -> Iterator[StreamEvent]:
        ws = self._live_connect(self._live_url())
        saw_tool_call = False
        audio_native = self._is_audio_native_live_model(request.model)
        acc_usage = Usage()
        try:
            setup_payload = self._live_setup_payload_from_request(request)
            setup_inner = setup_payload.setdefault("setup", {})
            if not audio_native:
                setup_inner.setdefault("generationConfig", {}).setdefault("responseModalities", ["TEXT"])
            ws.send(json.dumps(setup_payload))
            self._wait_for_setup_complete(ws)
            for msg in self._live_client_content_payload_from_request(request):
                ws.send(json.dumps(msg))

            yield StreamStartEvent(model=request.model)
            while True:
                raw = ws.recv()
                events, turn_complete, usage = self._decode_live_completion_stream_events(raw)
                acc_usage = Usage(
                    input_tokens=max(acc_usage.input_tokens, usage.input_tokens),
                    output_tokens=max(acc_usage.output_tokens, usage.output_tokens),
                    total_tokens=max(acc_usage.total_tokens or 0, usage.total_tokens or 0),
                )
                for event in events:
                    if event.type == "delta" and isinstance(event.delta, ToolCallDelta):
                        saw_tool_call = True
                    if event.type == "error":
                        yield event
                        return
                    yield event
                if turn_complete:
                    yield StreamEndEvent(finish_reason="tool_call" if saw_tool_call else "stop", usage=acc_usage)
                    return
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _live_setup_payload_from_request(self, request: Request) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        extensions.pop("transport", None)
        extensions.pop("prompt_caching", None)
        extensions.pop("output", None)
        config = LiveConfig(model=request.model, system=request.system, tools=request.tools, extensions=extensions or None)
        payload = self._live_setup_payload(config)
        output = (request.config.extensions or {}).get("output")
        audio_native = self._is_audio_native_live_model(request.model)
        if output == "audio" or audio_native:
            setup = payload.setdefault("setup", {})
            setup.setdefault("generationConfig", {})["responseModalities"] = ["AUDIO"]
            if output != "audio":
                setup["outputAudioTranscription"] = {}
            has_media = any(isinstance(p, (AudioPart, VideoPart)) for m in request.messages for p in m.parts)
            if has_media:
                setup.setdefault("realtimeInputConfig", {}).setdefault("automaticActivityDetection", {})["disabled"] = True
        elif output == "image":
            payload.setdefault("setup", {}).setdefault("generationConfig", {})["responseModalities"] = ["IMAGE"]
        return payload

    def _live_client_content_payload_from_request(self, request: Request) -> list[dict[str, Any]]:
        if self._is_audio_native_live_model(request.model):
            return self._build_realtime_input_payloads(request)
        if len(request.messages) == 1 and request.messages[0].role == "user" and all(isinstance(p, TextPart) for p in request.messages[0].parts):
            return [{"realtimeInput": {"text": parts_to_text(request.messages[0].parts)}}]
        return [{"clientContent": {"turns": [self._message(m) for m in request.messages], "turnComplete": True}}]

    def _build_realtime_input_payloads(self, request: Request) -> list[dict[str, Any]]:
        text_payloads: list[dict[str, Any]] = []
        media_payloads: list[dict[str, Any]] = []
        content_parts: list[dict[str, Any]] = []
        sent_audio_or_video = False
        for msg in request.messages:
            for part in msg.parts:
                if isinstance(part, TextPart) and part.text:
                    text_payloads.append({"realtimeInput": {"text": part.text}})
                elif isinstance(part, AudioPart):
                    if part.data is not None or part.path is not None:
                        mime = part.media_type or "audio/pcm"
                        raw = part.bytes
                        if "wav" in mime or "wave" in mime:
                            pcm, rate = self._wav_to_pcm(raw)
                            data = base64.b64encode(pcm).decode("ascii")
                            media_payloads.append({"realtimeInput": {"audio": {"mimeType": f"audio/pcm;rate={rate}", "data": data}}})
                        else:
                            data = part.data or base64.b64encode(raw).decode("ascii")
                            media_payloads.append({"realtimeInput": {"audio": {"mimeType": mime, "data": data}}})
                        sent_audio_or_video = True
                elif isinstance(part, VideoPart):
                    if part.data is not None or part.path is not None:
                        data = part.data or base64.b64encode(part.path.read_bytes()).decode("ascii")
                        media_payloads.append({"realtimeInput": {"video": {"mimeType": part.media_type or "video/mp4", "data": data}}})
                        sent_audio_or_video = True
                elif isinstance(part, (ImagePart, DocumentPart, BinaryPart)):
                    content_parts.append(self._part(part))
        payloads: list[dict[str, Any]] = []
        if content_parts:
            payloads.append({"clientContent": {"turns": [{"role": "user", "parts": content_parts}], "turnComplete": False}})
        payloads.extend(text_payloads + media_payloads)
        if sent_audio_or_video:
            payloads.insert(0, {"realtimeInput": {"activityStart": {}}})
            payloads.append({"realtimeInput": {"activityEnd": {}}})
        if not payloads:
            payloads.append({"realtimeInput": {"text": ""}})
        return payloads

    def _decode_live_completion_stream_events(self, raw: str | bytes) -> tuple[list[StreamEvent], bool, Usage]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return [], False, Usage()
        if not isinstance(payload, dict):
            return [], False, Usage()
        if "error" in payload:
            err = payload["error"]
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            return [StreamErrorEvent(error=self._error_detail(provider_code, message))], False, Usage()
        events: list[StreamEvent] = []
        tool_call = payload.get("toolCall")
        if isinstance(tool_call, dict):
            for idx, fc in enumerate(tool_call.get("functionCalls") or []):
                if isinstance(fc, dict):
                    events.append(StreamDeltaEvent(delta=ToolCallDelta(input=json.dumps(fc.get("args", {})), part_index=idx, id=str(fc.get("id") or f"fc_{idx}"), name=str(fc.get("name") or "tool"))))
        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events, False, self._live_usage(payload, None)
        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for idx, part in enumerate(model_turn.get("parts", []) or []):
                if "text" in part:
                    events.append(StreamDeltaEvent(delta=TextDelta(text=str(part.get("text") or ""), part_index=idx)))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(StreamDeltaEvent(delta=ToolCallDelta(input=json.dumps(fc.get("args", {})), part_index=idx, id=str(fc.get("id") or "fc_0"), name=str(fc.get("name") or "tool"))))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    data = str(inline.get("data") or "")
                    if mime.startswith("audio/"):
                        events.append(StreamDeltaEvent(delta=AudioDelta(data=data, part_index=idx, media_type=mime)))
                    elif mime.startswith("image/"):
                        events.append(StreamDeltaEvent(delta=ImageDelta(data=data, part_index=idx, media_type=mime)))
        out_tx = server.get("outputTranscription")
        if isinstance(out_tx, dict) and out_tx.get("text"):
            events.append(StreamDeltaEvent(delta=TextDelta(text=str(out_tx["text"]))))
        return events, bool(server.get("turnComplete")), self._live_usage(payload, server)

    # ─── Live sessions ──────────────────────────────────────────────

    def live(self, config: LiveConfig):
        ws = self._live_connect(self._live_url())
        payload = self._live_setup_payload(config)
        if self._is_audio_native_live_model(config.model):
            payload.setdefault("setup", {})["outputAudioTranscription"] = {}
        ws.send(json.dumps(payload))
        self._wait_for_setup_complete(ws)
        audio_native = self._is_audio_native_live_model(config.model)

        def encode_event(event: LiveClientEvent) -> list[dict[str, Any]]:
            if audio_native and isinstance(event, LiveClientTextEvent):
                return [{"realtimeInput": {"text": event.text}}]
            return self._encode_live_client_event(event)

        return WebSocketLiveSession(ws=ws, encode_event=encode_event, decode_event=self._decode_live_server_event)

    def _live_connect(self, url: str):
        connect = require_websocket_sync_connect()
        return connect(url)

    def _wait_for_setup_complete(self, ws: Any) -> None:
        while True:
            raw = ws.recv()
            try:
                payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if isinstance(payload, dict) and "setupComplete" in payload:
                return
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
                provider_code = str(err.get("status") or "live_setup") if isinstance(err, dict) else "live_setup"
                raise self._provider_error(
                    InvalidRequestError,
                    f"Live setup failed: {msg}",
                    provider_code=provider_code,
                )

    def _live_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        query = urllib.parse.urlencode({"key": self.api_key})
        return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _live_setup_payload(self, config: LiveConfig) -> dict[str, Any]:
        setup: dict[str, Any] = {"model": self._model_path(config.model)}
        if config.system:
            setup["systemInstruction"] = {"parts": [{"text": config.system if isinstance(config.system, str) else parts_to_text(config.system)}]}
        function_tools = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in config.tools
            if isinstance(t, FunctionTool)
        ]
        if function_tools:
            setup["tools"] = [{"functionDeclarations": function_tools}]
        generation_config: dict[str, Any] = {}
        if config.output_format is not None:
            generation_config["responseModalities"] = ["AUDIO"]
        elif self._is_audio_native_live_model(config.model):
            generation_config["responseModalities"] = ["AUDIO"]
        if config.voice:
            generation_config.setdefault("speechConfig", {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}})
        if generation_config:
            setup["generationConfig"] = generation_config
        if config.extensions:
            setup.update(config.extensions)
        return {"setup": setup}

    def _live_usage(self, payload: dict[str, Any], server: dict[str, Any] | None) -> Usage:
        usage_payload = payload.get("usageMetadata")
        if not isinstance(usage_payload, dict) and isinstance(server, dict):
            usage_payload = server.get("usageMetadata")
        usage_payload = usage_payload if isinstance(usage_payload, dict) else {}
        return Usage(
            input_tokens=int(usage_payload.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage_payload.get("responseTokenCount", usage_payload.get("candidatesTokenCount", 0)) or 0),
            total_tokens=usage_payload.get("totalTokenCount"),
            cache_read_tokens=usage_payload.get("cachedContentTokenCount"),
            reasoning_tokens=usage_payload.get("thoughtsTokenCount"),
        )

    def _encode_live_client_event(self, event: LiveClientEvent) -> list[dict[str, Any]]:
        if isinstance(event, LiveClientTurnEvent):
            return [{"clientContent": {"turns": [{"role": "user", "parts": [self._part(part) for part in event.parts]}], "turnComplete": event.turn_complete}}]
        if isinstance(event, LiveClientAudioEvent):
            return [{"realtimeInput": {"audio": {"mimeType": event.media_type, "data": event.data}}}]
        if isinstance(event, LiveClientImageEvent):
            return [{"realtimeInput": {"video": {"mimeType": event.media_type, "data": event.data}}}]
        if isinstance(event, LiveClientInterruptEvent):
            return [{"clientContent": {"turnComplete": True}}]
        if isinstance(event, LiveClientEndAudioEvent):
            return [{"realtimeInput": {"audioStreamEnd": True}}]
        if isinstance(event, LiveClientTextEvent):
            return [{"clientContent": {"turns": [{"role": "user", "parts": [{"text": event.text}]}], "turnComplete": True}}]
        if isinstance(event, LiveClientToolResultEvent):
            response_parts = [{"text": parts_to_text(event.content)}]
            return [{"toolResponse": {"functionResponses": [{"id": event.id, "response": {"output": response_parts}}]}}]
        return []

    def _decode_live_server_event(self, raw: str | bytes):
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
            return [LiveServerErrorEvent(error=self._error_detail(provider_code, message))]
        events: list[Any] = []
        tool_call = payload.get("toolCall")
        if isinstance(tool_call, dict):
            for fc in tool_call.get("functionCalls") or []:
                if isinstance(fc, dict):
                    events.append(LiveServerToolCallEvent(id=str(fc.get("id") or "fc_0"), name=str(fc.get("name") or "tool"), input=fc.get("args") if isinstance(fc.get("args"), dict) else {}))
        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events
        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for part in model_turn.get("parts", []) or []:
                if "text" in part:
                    events.append(LiveServerTextEvent(text=str(part.get("text") or "")))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    if mime.startswith("audio/"):
                        events.append(LiveServerAudioEvent(data=str(inline.get("data") or ""), media_type=mime or None))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(LiveServerToolCallEvent(id=str(fc.get("id") or "fc_0"), name=str(fc.get("name") or "tool"), input=fc.get("args") if isinstance(fc.get("args"), dict) else {}))
        out_tx = server.get("outputTranscription")
        if isinstance(out_tx, dict) and out_tx.get("text"):
            events.append(LiveServerTextEvent(text=str(out_tx["text"])))
        if server.get("interrupted"):
            events.append(LiveServerInterruptedEvent())
        if server.get("turnComplete"):
            events.append(LiveServerTurnEndEvent(usage=self._live_usage(payload, server)))
        return events

    # ─── Other endpoints ────────────────────────────────────────────

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_path = self._model_path(request.model)
        if len(request.inputs) <= 1:
            payload = {"model": model_path, "content": {"parts": [{"text": request.inputs[0] if request.inputs else ""}]}, **(request.extensions or {})}
            resp = self._send(make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/{model_path}:embedContent", headers=self._auth_headers({"Content-Type": "application/json"}), payload=payload, read_timeout=60.0))
            if resp.status >= 400:
                raise self.normalize_error(resp.status, resp.text())
            data = resp.json()
            values = tuple(float(v) for v in (data.get("embedding", {}) or {}).get("values", []))
            return EmbeddingResponse(model=request.model, vectors=(values,), provider_data=data)

        payload = {"requests": [{"model": model_path, "content": {"parts": [{"text": value}]}} for value in request.inputs], **(request.extensions or {})}
        resp = self._send(make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/{model_path}:batchEmbedContents", headers=self._auth_headers({"Content-Type": "application/json"}), payload=payload, read_timeout=60.0))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        vectors = tuple(tuple(float(v) for v in (item.get("values") or [])) for item in data.get("embeddings", []))
        return EmbeddingResponse(model=request.model, vectors=vectors, provider_data=data)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        url = build_url(f"{self.upload_base_url.rstrip('/')}/files", request.extensions)
        req = TransportRequest(
            method="POST",
            url=url,
            headers=list(self._auth_headers({
                "X-Goog-Upload-Protocol": "raw",
                "X-Goog-Upload-File-Name": request.filename,
                "Content-Type": request.media_type,
            }).items()),
            body=request.bytes,
            read_timeout=120.0,
        )
        resp = self._send(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        file_name = (data.get("file") or {}).get("name") or data.get("name") or ""
        return FileUploadResponse(id=str(file_name), provider_data=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        # Gemini's public batch surface changes across API versions; provide a
        # deterministic local fan-out fallback until a stable file-backed path is
        # selected through extensions.
        results: list[dict[str, Any]] = []
        for nested in request.requests:
            resp = self.complete(nested)
            results.append({"id": resp.id, "finish_reason": resp.finish_reason, "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "total_tokens": resp.usage.total_tokens}})
        return BatchResponse(id=f"batch_{uuid.uuid4().hex[:12]}", status="completed", provider_data={"results": results})

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        extensions = {"generationConfig": {"responseModalities": ["IMAGE"]}, **(request.extensions or {})}
        lm_req = Request(model=request.model, messages=(Message.user(request.prompt),), config=Config(extensions=extensions))
        resp = self.complete(lm_req)
        images = tuple(part for part in resp.message.parts if isinstance(part, ImagePart))
        return ImageGenerationResponse(images=images, id=resp.id, model=resp.model, usage=resp.usage, provider_data=resp.provider_data)

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if request.voice:
            generation_config["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}}
        extensions = {"generationConfig": generation_config, **(request.extensions or {})}
        lm_req = Request(model=request.model, messages=(Message.user(request.prompt),), config=Config(extensions=extensions))
        resp = self.complete(lm_req)
        audio = resp.message.first(AudioPart)
        if audio is None:
            raise ValueError("provider did not return audio data")
        return AudioGenerationResponse(audio=audio, id=resp.id, model=resp.model, usage=resp.usage, provider_data=resp.provider_data)
