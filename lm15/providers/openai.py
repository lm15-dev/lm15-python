from __future__ import annotations

import base64
import json
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
from ..profiles import ProviderProfile, ResolvedOpenAIResponsesCompat, resolve_openai_responses_compat
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports import Request as TransportRequest
from ..types import (
    AudioDelta,
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioPart,
    BatchRequest,
    BatchResponse,
    BuiltinTool,
    CitationDelta,
    ContinuationState,
    CitationPart,
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
    RefusalPart,
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
    Usage,
)
from .base import BaseProviderLM, HttpResponse, SyncTransport, default_transport
from .common import (
    make_json_request,
    parse_json_object,
    part_to_openai_input,
    parts_to_text,
)

# Canonical builtin tool name → OpenAI Responses API tool type
_OPENAI_BUILTIN_MAP: dict[str, str] = {
    "web_search": "web_search_preview",
    "code_execution": "code_interpreter",
    "file_search": "file_search",
    "computer_use": "computer_use_preview",
}

OPENAI_PROVIDER_EXECUTED_ITEMS = {
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "computer_call",
    "computer_use_call",
}


def _attach_unmapped(provider_data: dict[str, Any], unmapped: list[dict[str, str]]) -> dict[str, Any]:
    if not unmapped:
        return provider_data
    out = dict(provider_data)
    out["_lm15_unmapped"] = unmapped
    return out


def _record_unmapped(unmapped: list[dict[str, str]], path: str, typ: Any) -> None:
    unmapped.append({"path": path, "type": str(typ or "<missing>")})


def _builtin_to_openai(tool: BuiltinTool) -> dict[str, Any]:
    out: dict[str, Any] = {"type": _OPENAI_BUILTIN_MAP.get(tool.name, tool.name)}
    if tool.config:
        out.update(tool.config)
    return out


def _response_format_to_openai_text(format_config: dict[str, Any]) -> dict[str, Any]:
    """Map canonical lm15 response_format to OpenAI Responses text config."""
    text_config = format_config.get("text")
    if isinstance(text_config, dict):
        return text_config

    text_format = format_config.get("format")
    if isinstance(text_format, dict):
        return dict(format_config)

    if format_config.get("type") == "json_schema":
        text_format = dict(format_config)
        text_format.setdefault("name", "response")
        return {"format": text_format}

    if format_config.get("type") == "json_object":
        return {"format": dict(format_config)}

    schema = format_config.get("schema") if isinstance(format_config.get("schema"), dict) else format_config
    return {
        "format": {
            "type": "json_schema",
            "name": str(format_config.get("name") or "response"),
            "schema": schema,
        }
    }


def _finish_from_status(data: dict[str, Any], *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    status = str(data.get("status") or "").lower()
    incomplete = data.get("incomplete_details") or {}
    reason = str(incomplete.get("reason") or "").lower() if isinstance(incomplete, dict) else ""
    if status == "incomplete" and "token" in reason:
        return "length"
    if "content_filter" in reason or "safety" in reason:
        return "content_filter"
    return "stop"


def _batch_status(status: str) -> str:
    status = status.lower()
    if status in {"completed", "failed", "cancelled"}:
        return status
    if status in {"cancelling", "canceling"}:
        return "cancelled"
    if status in {"in_progress", "finalizing", "running"}:
        return "running"
    if status in {"validating", "queued"}:
        return "queued"
    return "submitted"


def _str_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _annotation_text(annotation: dict[str, Any], source_text: str | None) -> str | None:
    for key in ("text", "snippet", "cited_text", "quote"):
        text = _str_or_none(annotation.get(key))
        if text is not None:
            return text

    start = _int_or_none(annotation.get("start_index"))
    end = _int_or_none(annotation.get("end_index"))
    if source_text is not None and start is not None and end is not None:
        if 0 <= start < end <= len(source_text):
            return source_text[start:end]
    return None


def _citation_from_openai_annotation(annotation: dict[str, Any], source_text: str | None) -> CitationPart | None:
    url = _str_or_none(annotation.get("url") or annotation.get("uri"))
    title = _str_or_none(
        annotation.get("title")
        or annotation.get("filename")
        or annotation.get("file_id")
    )
    text = _annotation_text(annotation, source_text)
    if url is None and title is None and text is None:
        return None
    return CitationPart(url=url, title=title, text=text)


def _citation_delta_from_openai_annotation(
    annotation: dict[str, Any],
    *,
    part_index: int,
    source_text: str | None = None,
) -> CitationDelta | None:
    citation = _citation_from_openai_annotation(annotation, source_text)
    if citation is None:
        return None
    return CitationDelta(
        text=citation.text,
        url=citation.url,
        title=citation.title,
        part_index=part_index,
    )


@dataclass(slots=True)
class OpenAILM(BaseProviderLM):
    api_key: str
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = "https://api.openai.com/v1"
    profile: ProviderProfile | None = None

    provider: str = "openai"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "audio", "video", "document", "binary"}),
        output_modalities=frozenset({"text", "audio", "image"}),
        features=frozenset({"streaming", "tools", "json_output", "reasoning", "live", "embeddings", "files", "batch", "images", "audio"}),
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
        responses_api=True,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="openai",
        supports=supports,
        auth_modes=("bearer",),
        enterprise_variants=("azure-openai",),
        env_keys=("OPENAI_API_KEY",),
    )

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
        "model_not_found": UnsupportedModelError,
        "model_not_available": UnsupportedModelError,
        "unsupported_model": UnsupportedModelError,
    }

    _model_error_codes: ClassVar[frozenset[str]] = frozenset(
        {"model_not_found", "model_not_available", "unsupported_model"}
    )

    _stream_error_code_map: ClassVar[dict[str, type[ProviderError]]] = {
        **_response_error_code_map,
        "context_length_exceeded": ContextLengthError,
        "invalid_api_key": AuthError,
        "insufficient_quota": BillingError,
        "authentication_error": AuthError,
        "rate_limit_error": RateLimitError,
    }

    @classmethod
    def from_profile(
        cls,
        *,
        api_key: str,
        profile: ProviderProfile,
        transport: SyncTransport | None = None,
    ) -> "OpenAILM":
        endpoint = profile.endpoint("inference")
        base_url = endpoint.base_url if endpoint and endpoint.base_url else "https://api.openai.com/v1"
        return cls(
            api_key=api_key,
            transport=transport or default_transport(),
            base_url=base_url,
            profile=profile,
        )

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
        }

    @staticmethod
    def _is_model_error(message: str, *codes: str) -> bool:
        lowered = " ".join(value for value in (message, *codes) if value).lower()
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

    def _response_error(self, code: str, message: str) -> ProviderError:
        cls = self._response_error_code_map.get(code, ServerError)
        msg = message or code or "provider error"
        return self._provider_error(cls, msg, provider_code=code or None)

    def _error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._stream_error_code_map.get(provider_code, ProviderError)
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message or provider_code or "provider error",
            provider_code=provider_code or "provider",
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        """Extract message from OpenAI error shape."""
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            code = str(err.get("code") or "") if isinstance(err, dict) else ""
            err_type = str(err.get("type") or "") if isinstance(err, dict) else ""

            provider_code = code or err_type or None

            if code == "context_length_exceeded":
                return self._provider_error(
                    ContextLengthError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code in self._model_error_codes or (
                status == 404 and self._is_model_error(msg, code, err_type)
            ):
                return self._provider_error(
                    UnsupportedModelError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code == "insufficient_quota" or err_type == "insufficient_quota":
                return self._provider_error(
                    BillingError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code == "invalid_api_key" or err_type == "authentication_error":
                return self._provider_error(
                    AuthError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code == "rate_limit_exceeded" or err_type == "rate_limit_error":
                return self._provider_error(
                    RateLimitError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code and code not in msg:
                msg = f"{msg} ({code})"
        except Exception:
            msg = body.strip()[:500] or f"HTTP {status}"
            provider_code = None
        return map_http_error(
            status,
            msg,
            provider=self.provider,
            env_keys=self.manifest.env_keys,
            provider_code=provider_code,
        )

    # ─── Request serialization ──────────────────────────────────────

    def _compat(self, request: Request) -> ResolvedOpenAIResponsesCompat:
        return resolve_openai_responses_compat(
            base_url=self.base_url,
            model=request.model,
            profile=self.profile,
            request_extensions=request.config.extensions,
        )

    def _build_input(
        self,
        messages: tuple[Message, ...],
        compat: ResolvedOpenAIResponsesCompat,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                for part in msg.parts:
                    if isinstance(part, ToolResultPart):
                        output = parts_to_text(part.content)
                        if not output:
                            output = json.dumps([{"type": p.type} for p in part.content])
                        item = {
                            "type": "function_call_output",
                            "call_id": part.id,
                            "output": output,
                        }
                        if compat.tool_result_name == "include" and part.name:
                            item["name"] = part.name
                        items.append(item)
                continue

            if msg.role == "assistant":
                content_parts = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        content_parts.append({"type": "output_text", "text": part.text})
                    elif isinstance(part, RefusalPart):
                        content_parts.append({"type": "refusal", "refusal": part.text})
            else:
                content_parts = [
                    part_to_openai_input(part)
                    for part in msg.parts
                    if not isinstance(part, (ToolCallPart, ToolResultPart))
                ]
            if content_parts:
                role = msg.role
                if role == "developer":
                    role = compat.developer_role
                items.append({"role": role, "content": content_parts})

            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": part.id,
                            "name": part.name,
                            "arguments": json.dumps(part.input, separators=(",", ":")),
                        }
                    )
        return items

    def _tool_choice_payload(self, request: Request) -> Any:
        tc = request.config.tool_choice
        if tc is None:
            return None
        if tc.mode == "none":
            return "none"
        if tc.allowed:
            if len(tc.allowed) == 1:
                return {"type": "function", "name": tc.allowed[0]}
            # OpenAI has no portable multi-tool allowlist in Responses.  Keep
            # normal auto/required behavior and pass the allowlist through a
            # provider extension if the caller needs a preview feature.
        if tc.mode == "required":
            return "required"
        return "auto"

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        compat = self._compat(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "input": self._build_input(request.messages, compat),
            "stream": stream,
        }
        if request.system:
            payload["instructions"] = request.system if isinstance(request.system, str) else parts_to_text(request.system)
        if request.config.max_tokens is not None:
            payload[compat.max_output_tokens_field] = request.config.max_tokens
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.config.top_p is not None:
            payload["top_p"] = request.config.top_p
        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, FunctionTool):
                    tool_payload = {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    if compat.strict_tools == "include":
                        tool_payload["strict"] = False
                    tools_wire.append(tool_payload)
                elif isinstance(tool, BuiltinTool):
                    tools_wire.append(_builtin_to_openai(tool))
            payload["tools"] = tools_wire
        tool_choice = self._tool_choice_payload(request)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if request.config.tool_choice and request.config.tool_choice.parallel is not None:
            payload["parallel_tool_calls"] = request.config.tool_choice.parallel
        if request.config.response_format:
            payload["text"] = _response_format_to_openai_text(request.config.response_format)
        if request.config.reasoning:
            reasoning = request.config.reasoning
            if not reasoning.is_off:
                effort = {"adaptive": "medium", "xhigh": "high"}.get(reasoning.effort, reasoning.effort)
                if compat.reasoning_format == "responses_reasoning":
                    reasoning_payload: dict[str, Any] = {"effort": effort}
                    if reasoning.summary is not None:
                        reasoning_payload["summary"] = reasoning.summary
                    payload["reasoning"] = reasoning_payload
                elif compat.reasoning_format == "reasoning_effort":
                    payload["reasoning_effort"] = effort
                elif compat.reasoning_format == "openrouter":
                    payload["reasoning"] = {"effort": effort}
                elif compat.reasoning_format == "deepseek":
                    payload["thinking"] = {"type": "enabled"}
                    payload["reasoning_effort"] = effort
                elif compat.reasoning_format in {"qwen", "zai"}:
                    payload["enable_thinking"] = True
                elif compat.reasoning_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                    }
            else:
                if compat.reasoning_format == "deepseek":
                    payload["thinking"] = {"type": "disabled"}
                elif compat.reasoning_format in {"qwen", "zai"}:
                    payload["enable_thinking"] = False
                elif compat.reasoning_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Cache / Prompt Caching support
        cache_cfg = request.config.cache
        if cache_cfg is not None and cache_cfg.mode != "off" and compat.cache_control == "openai":
            if cache_cfg.key:
                payload["prompt_cache_key"] = cache_cfg.key
            if cache_cfg.retention == "long":
                payload["prompt_cache_retention"] = "24h"

        if compat.routing is not None:
            payload["provider"] = compat.routing

        if request.config.extensions:
            reserved = {
                "prompt_caching",
                "cache",
                "compat",
                "openai_compat",
                "openai_responses_compat",
            }
            passthrough = {k: v for k, v in request.config.extensions.items() if k not in reserved}
            payload.update(passthrough)
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        return make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/responses",
            headers=self._headers(),
            payload=self._payload(request, stream=stream),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()

        resp_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(resp_error, dict):
            raise self._response_error(
                str(resp_error.get("code") or ""),
                str(resp_error.get("message") or resp_error),
            )

        parts: list[Any] = []
        unmapped: list[dict[str, str]] = []
        for item_index, item in enumerate(data.get("output", []) or []):
            if not isinstance(item, dict):
                _record_unmapped(unmapped, f"output[{item_index}]", type(item).__name__)
                continue
            item_type = item.get("type")
            if item_type == "message":
                for content_index, content in enumerate(item.get("content", []) or []):
                    if not isinstance(content, dict):
                        _record_unmapped(unmapped, f"output[{item_index}].content[{content_index}]", type(content).__name__)
                        continue
                    ctype = content.get("type")
                    if ctype in ("output_text", "text"):
                        text = str(content.get("text") or "")
                        parts.append(TextPart(text=text))
                        for annotation in content.get("annotations", []) or []:
                            if not isinstance(annotation, dict):
                                continue
                            citation = _citation_from_openai_annotation(annotation, text)
                            if citation is not None:
                                parts.append(citation)
                    elif ctype == "refusal":
                        text = str(content.get("refusal") or content.get("text") or "")
                        parts.append(RefusalPart(text=text) if text else TextPart(text=""))
                    elif ctype == "output_image":
                        b64 = content.get("b64_json") or content.get("image_base64") or ""
                        if b64:
                            parts.append(ImagePart(media_type="image/png", data=str(b64)))
                    elif ctype == "output_audio":
                        audio_payload = content.get("audio") if isinstance(content.get("audio"), dict) else {}
                        b64 = audio_payload.get("data") or content.get("b64_json") or ""
                        if b64:
                            parts.append(AudioPart(media_type="audio/wav", data=str(b64)))
                    else:
                        _record_unmapped(unmapped, f"output[{item_index}].content[{content_index}]", ctype)
            elif item_type == "function_call":
                parts.append(
                    ToolCallPart(
                        id=str(item.get("call_id") or item.get("id") or f"call_{len(parts)}"),
                        name=str(item.get("name") or "tool"),
                        input=parse_json_object(item.get("arguments")),
                    )
                )
            elif item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    text = "\n".join(str(x.get("text") if isinstance(x, dict) else x) for x in summary)
                else:
                    text = str(summary or item.get("text") or "")
                if text:
                    parts.append(ThinkingPart(text=text))
            elif item_type in OPENAI_PROVIDER_EXECUTED_ITEMS:
                continue
            else:
                _record_unmapped(unmapped, f"output[{item_index}]", item_type)

        if not parts:
            parts = [TextPart(text=str(data.get("output_text") or ""))]

        usage_data = data.get("usage", {}) or {}
        input_details = usage_data.get("input_tokens_details") or {}
        output_details = usage_data.get("output_tokens_details") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("input_tokens", 0) or 0),
            output_tokens=int(usage_data.get("output_tokens", 0) or 0),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
            cache_read_tokens=input_details.get("cached_tokens"),
            input_audio_tokens=input_details.get("audio_tokens"),
            output_audio_tokens=output_details.get("audio_tokens"),
        )

        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        message_continuation: tuple[ContinuationState, ...] = ()
        if data.get("id"):
            message_continuation = (
                ContinuationState(
                    provider="openai",
                    kind="response_id",
                    data={"id": str(data.get("id"))},
                ),
            )
        return Response(
            id=str(data.get("id")) if data.get("id") else None,
            model=str(data.get("model") or request.model),
            message=Message(role="assistant", parts=tuple(parts), continuation=message_continuation),
            finish_reason=_finish_from_status(data, has_tool_call=has_tool),
            usage=usage,
            provider_data=_attach_unmapped(data, unmapped),
        )

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        event = self._parse_single_stream_event(request, raw_event)
        if event is not None:
            yield event

    def _parse_single_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None
        if raw_event.data == "[DONE]":
            return StreamEndEvent(finish_reason="stop")
        payload = json.loads(raw_event.data)
        et = str(payload.get("type") or "")

        if et == "response.created":
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            return StreamStartEvent(
                id=str(response.get("id")) if response.get("id") else None,
                model=str(response.get("model") or request.model),
            )

        if et in {"response.output_text.delta", "response.refusal.delta"}:
            return StreamDeltaEvent(
                delta=TextDelta(
                    text=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                )
            )

        if et in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            return StreamDeltaEvent(
                delta=ThinkingDelta(
                    text=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                )
            )

        if et == "response.output_text.annotation.added":
            annotation = payload.get("annotation")
            if isinstance(annotation, dict):
                delta = _citation_delta_from_openai_annotation(
                    annotation,
                    part_index=int(payload.get("output_index", 0) or 0),
                )
                if delta is not None:
                    return StreamDeltaEvent(delta=delta)
            return None

        if et == "response.output_audio.delta":
            return StreamDeltaEvent(
                delta=AudioDelta(
                    data=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    media_type="audio/wav",
                )
            )

        if et in {"response.output_image.delta", "response.image.delta"}:
            return StreamDeltaEvent(
                delta=ImageDelta(
                    data=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    media_type="image/png",
                )
            )

        if et == "response.output_item.added":
            item = payload.get("item", {}) if isinstance(payload.get("item"), dict) else {}
            if item.get("type") == "function_call":
                return StreamDeltaEvent(
                    delta=ToolCallDelta(
                        input=str(item.get("arguments") or ""),
                        part_index=int(payload.get("output_index", 0) or 0),
                        id=str(item.get("call_id") or item.get("id") or "") or None,
                        name=str(item.get("name") or "") or None,
                    )
                )
            return None

        if et == "response.function_call_arguments.delta":
            return StreamDeltaEvent(
                delta=ToolCallDelta(
                    input=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    id=str(payload.get("call_id") or payload.get("id") or "") or None,
                    name=str(payload.get("name") or "") or None,
                )
            )

        if et == "response.completed":
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
            input_details = usage_data.get("input_tokens_details") or {}
            output_details = usage_data.get("output_tokens_details") or {}
            usage = Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=output_details.get("reasoning_tokens"),
                cache_read_tokens=input_details.get("cached_tokens"),
                input_audio_tokens=input_details.get("audio_tokens"),
                output_audio_tokens=output_details.get("audio_tokens"),
            )
            output = response.get("output", []) if isinstance(response, dict) else []
            has_tool = any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)
            return StreamEndEvent(
                finish_reason="tool_call" if has_tool else "stop",
                usage=usage,
                provider_data=response if isinstance(response, dict) else None,
            )

        if et in {"response.error", "error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return StreamErrorEvent(error=self._error_detail(provider_code, message))

        return None

    # ─── Streaming over OpenAI Realtime for live models ──────────────

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
        return "realtime" in model_name or "-live" in model_name

    def _stream_via_live_completion(self, request: Request) -> Iterator[StreamEvent]:
        ws = self._live_connect(self._live_url(request.model), self._live_headers())
        saw_tool_call = False
        usage = Usage()
        try:
            ws.send(json.dumps(self._live_session_update_from_request(request)))
            for frame in self._live_message_frames_for_request(request):
                ws.send(json.dumps(frame))

            yield StreamStartEvent(model=request.model)
            while True:
                raw = ws.recv()
                for event in self._decode_live_completion_stream_events(request, raw):
                    if event.type == "delta":
                        if isinstance(event.delta, ToolCallDelta):
                            saw_tool_call = True
                        yield event
                    elif event.type == "error":
                        yield event
                        return
                    elif event.type == "end":
                        if event.usage is not None:
                            usage = event.usage
                        yield StreamEndEvent(
                            finish_reason="tool_call" if saw_tool_call else (event.finish_reason or "stop"),
                            usage=usage,
                        )
                        return
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _live_session_update_from_request(self, request: Request) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        extensions.pop("transport", None)
        extensions.pop("prompt_caching", None)
        extensions.pop("output", None)
        config = LiveConfig(
            model=request.model,
            system=request.system,
            tools=request.tools,
            extensions=extensions or None,
        )
        return self._live_session_update_payload(config)

    def _live_message_frames_for_request(self, request: Request) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                for part in message.parts:
                    if not isinstance(part, ToolResultPart):
                        continue
                    output = parts_to_text(part.content) or json.dumps([{"type": p.type} for p in part.content])
                    frames.append({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": part.id, "output": output}})
                continue

            content = [part_to_openai_input(p) for p in message.parts if not isinstance(p, (ToolCallPart, ToolResultPart))]
            if content:
                frames.append({"type": "conversation.item.create", "item": {"type": "message", "role": message.role, "content": content}})
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    frames.append({"type": "conversation.item.create", "item": {"type": "function_call", "call_id": part.id, "name": part.name, "arguments": json.dumps(part.input)}})

        response_create: dict[str, Any] = {"type": "response.create"}
        if (request.config.extensions or {}).get("output") == "audio":
            response_create["response"] = {"modalities": ["text", "audio"]}
        frames.append(response_create)
        return frames

    def _decode_live_completion_stream_events(self, request: Request, raw: str | bytes) -> list[StreamEvent]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        et = str(payload.get("type") or "")

        if et in {"response.output_text.delta", "response.text.delta", "response.audio_transcript.delta"}:
            delta = str(payload.get("delta") or payload.get("text") or "")
            return [StreamDeltaEvent(delta=TextDelta(text=delta))] if delta else []
        if et == "response.output_audio.delta":
            delta = str(payload.get("delta") or "")
            return [StreamDeltaEvent(delta=AudioDelta(data=delta, media_type="audio/wav"))] if delta else []
        if et in {"response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.output_item.done"}:
            if et in {"response.output_item.added", "response.output_item.done"}:
                item = payload.get("item", {}) if isinstance(payload.get("item"), dict) else {}
                if item.get("type") != "function_call":
                    return []
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "tool")
                arguments = item.get("arguments") or ""
            else:
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                name = str(payload.get("name") or "tool")
                arguments = payload.get("delta") if et.endswith("delta") else payload.get("arguments")
            return [StreamDeltaEvent(delta=ToolCallDelta(input=arguments if isinstance(arguments, str) else json.dumps(arguments or {}), id=call_id or None, name=name))]
        if et in {"response.done", "response.completed"}:
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
            u_in = usage_data.get("input_tokens_details") or {}
            u_out = usage_data.get("output_tokens_details") or {}
            usage = Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=u_out.get("reasoning_tokens"),
                cache_read_tokens=u_in.get("cached_tokens"),
                input_audio_tokens=u_in.get("audio_tokens"),
                output_audio_tokens=u_out.get("audio_tokens"),
            )
            return [StreamEndEvent(finish_reason="stop", usage=usage, provider_data=response if isinstance(response, dict) else None)]
        if et in {"error", "response.error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return [StreamErrorEvent(error=self._error_detail(provider_code, message))]
        return []

    # ─── Live sessions ──────────────────────────────────────────────

    def live(self, config: LiveConfig):
        ws = self._live_connect(self._live_url(config.model), self._live_headers())
        ws.send(json.dumps(self._live_session_update_payload(config)))

        return WebSocketLiveSession(
            ws=ws,
            encode_event=self._encode_live_client_event,
            decode_event=self._decode_live_server_event,
        )

    def _live_connect(self, url: str, headers: dict[str, str]):
        connect = require_websocket_sync_connect()
        return connect(url, additional_headers=headers)

    def _live_url(self, model: str) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        path = f"{base_path}/realtime" if base_path else "/realtime"
        query = urllib.parse.urlencode({"model": model})
        return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _live_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}

    def _live_session_update_payload(self, config: LiveConfig) -> dict[str, Any]:
        session: dict[str, Any] = {}
        if config.system:
            session["instructions"] = config.system if isinstance(config.system, str) else parts_to_text(config.system)
        if config.voice:
            session["voice"] = config.voice
        if config.output_format is not None:
            session["modalities"] = ["text", "audio"]
            session["output_audio_format"] = config.output_format.encoding
        else:
            session["modalities"] = ["text"]
        if config.input_format is not None:
            session["input_audio_format"] = config.input_format.encoding
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
                if isinstance(t, FunctionTool)
            ]
        if config.extensions:
            session.update(config.extensions)
        return {"type": "session.update", "session": session}

    def _encode_live_client_event(self, event: LiveClientEvent) -> list[dict[str, Any]]:
        if isinstance(event, LiveClientAudioEvent):
            return [{"type": "input_audio_buffer.append", "audio": event.data}]
        if isinstance(event, LiveClientEndAudioEvent):
            return [{"type": "input_audio_buffer.commit"}, {"type": "response.create"}]
        if isinstance(event, LiveClientInterruptEvent):
            return [{"type": "response.cancel"}]
        if isinstance(event, LiveClientTextEvent):
            return [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": event.text}]}},
                {"type": "response.create"},
            ]
        if isinstance(event, LiveClientTurnEvent):
            return [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [part_to_openai_input(part) for part in event.parts]}},
                {"type": "response.create"},
            ] if event.turn_complete else [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [part_to_openai_input(part) for part in event.parts]}},
            ]
        if isinstance(event, LiveClientImageEvent):
            return [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": f"data:{event.media_type};base64,{event.data}"}]}},
                {"type": "response.create"},
            ]
        if isinstance(event, LiveClientToolResultEvent):
            output = parts_to_text(event.content) or json.dumps([{"type": p.type} for p in event.content])
            return [
                {"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": event.id, "output": output}},
                {"type": "response.create"},
            ]
        return []

    def _decode_live_server_event(self, raw: str | bytes):
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        et = str(payload.get("type") or "")
        events: list[Any] = []
        if et in {"response.output_text.delta", "response.text.delta", "response.audio_transcript.delta"}:
            delta = str(payload.get("delta") or payload.get("text") or "")
            if delta:
                events.append(LiveServerTextEvent(text=delta))
        elif et == "response.output_audio.delta":
            delta = str(payload.get("delta") or "")
            if delta:
                events.append(LiveServerAudioEvent(data=delta))
        elif et == "response.function_call_arguments.delta":
            delta = str(payload.get("delta") or "")
            if delta:
                events.append(LiveServerToolCallDeltaEvent(input_delta=delta, id=str(payload.get("call_id") or payload.get("id") or "") or None, name=str(payload.get("name") or "") or None))
        elif et in {"response.function_call_arguments.done", "response.output_item.done"}:
            if et == "response.output_item.done":
                item = payload.get("item", {}) if isinstance(payload.get("item"), dict) else {}
                if item.get("type") != "function_call":
                    item = {}
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "tool")
                arguments = item.get("arguments")
            else:
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                name = str(payload.get("name") or "tool")
                arguments = payload.get("arguments")
            if call_id:
                events.append(LiveServerToolCallEvent(id=call_id, name=name, input=parse_json_object(arguments)))
        elif et in {"response.done", "response.completed"}:
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
            u_in = usage_data.get("input_tokens_details") or {}
            u_out = usage_data.get("output_tokens_details") or {}
            events.append(LiveServerTurnEndEvent(usage=Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=u_out.get("reasoning_tokens"),
                cache_read_tokens=u_in.get("cached_tokens"),
                input_audio_tokens=u_in.get("audio_tokens"),
                output_audio_tokens=u_out.get("audio_tokens"),
            )))
        elif et in {"response.cancelled", "response.canceled"}:
            events.append(LiveServerInterruptedEvent())
        elif et in {"error", "response.error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            events.append(LiveServerErrorEvent(error=self._error_detail(provider_code, message)))
        return events

    # ─── Other endpoints ────────────────────────────────────────────

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        resp = self._send(make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/embeddings",
            headers=self._headers(),
            payload={"model": request.model, "input": list(request.inputs), **(request.extensions or {})},
            read_timeout=60.0,
        ))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        vectors = tuple(tuple(float(v) for v in item.get("embedding", [])) for item in data.get("data", []))
        u = data.get("usage", {}) or {}
        usage = Usage(input_tokens=int(u.get("prompt_tokens", 0) or 0), output_tokens=0, total_tokens=u.get("total_tokens"))
        return EmbeddingResponse(model=str(data.get("model") or request.model), vectors=vectors, usage=usage, provider_data=data)

    def _multipart_file_body(self, *, purpose: str, filename: str, media_type: str, data: bytes) -> tuple[str, bytes]:
        boundary = f"lm15-{uuid.uuid4().hex}"
        safe_filename = filename.replace('"', "%22")
        lines: list[bytes] = []
        def add(s: str) -> None:
            lines.append(s.encode("utf-8"))
        add(f"--{boundary}\r\n")
        add('Content-Disposition: form-data; name="purpose"\r\n\r\n')
        add(f"{purpose}\r\n")
        add(f"--{boundary}\r\n")
        add(f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n')
        add(f"Content-Type: {media_type}\r\n\r\n")
        lines.append(data)
        add("\r\n")
        add(f"--{boundary}--\r\n")
        return boundary, b"".join(lines)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        purpose = str((request.extensions or {}).get("purpose", "assistants"))
        boundary, body = self._multipart_file_body(purpose=purpose, filename=request.filename, media_type=request.media_type, data=request.bytes)
        resp = self._send(TransportRequest(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/files",
            headers=list(self._headers(content_type=f"multipart/form-data; boundary={boundary}").items()),
            body=body,
            read_timeout=120.0,
        ))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        return FileUploadResponse(id=str(data.get("id") or ""), provider_data=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        extensions = request.extensions or {}
        input_file_id = extensions.get("input_file_id")
        if input_file_id:
            payload = {
                "input_file_id": input_file_id,
                "endpoint": extensions.get("endpoint", "/v1/responses"),
                "completion_window": extensions.get("completion_window", "24h"),
            }
            resp = self._send(make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/batches", headers=self._headers(), payload=payload, read_timeout=120.0))
            if resp.status >= 400:
                raise self.normalize_error(resp.status, resp.text())
            data = resp.json()
            status = _batch_status(str(data.get("status") or "submitted"))
            return BatchResponse(id=str(data.get("id") or ""), status=status, provider_data=data)

        results: list[dict[str, Any]] = []
        for nested in request.requests:
            out = self.complete(nested)
            results.append({"id": out.id, "finish_reason": out.finish_reason, "usage": {"input_tokens": out.usage.input_tokens, "output_tokens": out.usage.output_tokens, "total_tokens": out.usage.total_tokens}})
        return BatchResponse(id=f"batch_{uuid.uuid4().hex[:12]}", status="completed", provider_data={"results": results})

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        payload = {"model": request.model, "prompt": request.prompt, "size": request.size, **(request.extensions or {})}
        payload = {k: v for k, v in payload.items() if v is not None}
        resp = self._send(make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/images/generations", headers=self._headers(), payload=payload, read_timeout=120.0))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        images: list[ImagePart] = []
        for item in data.get("data", []) or []:
            if item.get("b64_json"):
                images.append(ImagePart(media_type="image/png", data=str(item["b64_json"])))
            elif item.get("url"):
                images.append(ImagePart(media_type="image/png", url=str(item["url"])))
        return ImageGenerationResponse(images=tuple(images), id=data.get("id"), model=data.get("model"), provider_data=data)

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        payload = {"model": request.model, "input": request.prompt, "voice": request.voice or "alloy", "format": request.format or "wav", **(request.extensions or {})}
        resp = self._send(make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/audio/speech", headers=self._headers(), payload=payload, read_timeout=120.0))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        ctype = (resp.header("content-type") or "audio/wav").split(";", 1)[0].strip()
        try:
            payload_json = json.loads(resp.body)
            if isinstance(payload_json, dict) and payload_json.get("audio"):
                b64 = str(payload_json["audio"])
            elif isinstance(payload_json, dict) and payload_json.get("b64_json"):
                b64 = str(payload_json["b64_json"])
            else:
                b64 = base64.b64encode(resp.body).decode("ascii")
        except Exception:
            b64 = base64.b64encode(resp.body).decode("ascii")
        return AudioGenerationResponse(audio=AudioPart(media_type=ctype, data=b64), model=request.model, provider_data={"content_type": ctype})
