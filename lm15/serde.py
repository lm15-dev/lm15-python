"""
lm15.serde — Serialization and deserialization for the lm15 type system.

One function pair per type: xxx_to_dict / xxx_from_dict.
All Part field names are consistent: "input" on the wire matches .input
in memory.  No name translation (no "arguments" vs "input" split).
"""

from __future__ import annotations

from typing import Any

from .models import (
    InferenceModelInfo,
    InferencePricing,
    ModelInfo,
    ModelOrigin,
    TrainingModelInfo,
    TrainingPricing,
)
from .types import (
    AudioDelta,
    AudioFormat,
    AudioPart,
    BinaryPart,
    BuiltinTool,
    CacheConfig,
    CitationDelta,
    CitationPart,
    Config,
    ContinuationDelta,
    ContinuationState,
    Delta,
    DocumentPart,
    ErrorDetail,
    FunctionTool,
    ImageDelta,
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
    LiveServerEvent,
    LiveServerInterruptedEvent,
    LiveServerTextEvent,
    LiveServerToolCallDeltaEvent,
    LiveServerToolCallEvent,
    LiveServerTurnEndEvent,
    Message,
    Part,
    PART_TYPES,
    Reasoning,
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
    RefusalPart,
    Tool,
    ToolCallDelta,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
    VideoPart,
)


# ─── Helpers ─────────────────────────────────────────────────────────

def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == () or value == [] or value == {}


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields — at the top level of this object ONLY.

    The omission rule (docs/serde-rules.md): each typed serializer omits its
    own empty optional fields; nested values are either already-serialized
    typed objects or opaque user/provider payloads, and both are embedded
    verbatim. Recursing here would give the same value different wire forms
    depending on the serialization entry point.
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, tuple):
            value = list(value)
        if _is_empty(value):
            continue
        out[key] = value
    return out



# ─── Continuation state ──────────────────────────────────────────────

def continuation_to_dict(state: ContinuationState) -> dict[str, Any]:
    return {
        "provider": state.provider,
        "kind": state.kind,
        "data": state.data,
    }


def continuation_from_dict(d: dict[str, Any]) -> ContinuationState:
    return ContinuationState(
        provider=d["provider"],
        kind=d["kind"],
        data=d.get("data", {}),
    )


def _continuation_to_json(values: tuple[ContinuationState, ...]) -> list[dict[str, Any]] | None:
    if not values:
        return None
    return [continuation_to_dict(state) for state in values]


def _continuation_from_json(value: Any) -> tuple[ContinuationState, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("continuation must be a list")
    states: list[ContinuationState] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("continuation entries must be objects")
        states.append(continuation_from_dict(item))
    return tuple(states)


# ─── Parts ───────────────────────────────────────────────────────────

def part_to_dict(part: Part) -> dict[str, Any]:
    """Serialize a Part to the canonical lm15 JSON format."""
    d: dict[str, Any] = {"type": part.type}

    if isinstance(part, TextPart):
        d["text"] = part.text

    elif isinstance(part, ThinkingPart):
        d["text"] = part.text
        if part.redacted:
            d["redacted"] = part.redacted

    elif isinstance(part, RefusalPart):
        d["text"] = part.text

    elif isinstance(part, CitationPart):
        if part.text is not None:
            d["text"] = part.text
        if part.url is not None:
            d["url"] = part.url
        if part.title is not None:
            d["title"] = part.title

    elif isinstance(part, (ImagePart, AudioPart, VideoPart, DocumentPart, BinaryPart)):
        d["media_type"] = part.media_type
        if part.data is not None:
            d["data"] = part.data
        if part.url is not None:
            d["url"] = part.url
        if part.file_id is not None:
            d["file_id"] = part.file_id
        if part.path is not None:
            d["path"] = str(part.path)
        if hasattr(part, "detail") and part.detail is not None:
            d["detail"] = part.detail

    elif isinstance(part, ToolCallPart):
        d["id"] = part.id
        d["name"] = part.name
        d["input"] = part.input  # consistent: "input" everywhere

    elif isinstance(part, ToolResultPart):
        d["id"] = part.id
        if part.name is not None:
            d["name"] = part.name
        d["content"] = [part_to_dict(p) for p in part.content] if part.content else []
        if part.is_error:
            d["is_error"] = part.is_error

    continuation = _continuation_to_json(part.continuation)
    if continuation is not None:
        d["continuation"] = continuation

    return d


def part_from_dict(d: dict[str, Any]) -> Part:
    """Deserialize a Part from the canonical lm15 JSON format."""
    t = d["type"]

    continuation = _continuation_from_json(d.get("continuation"))

    if t == "text":
        return TextPart(text=d.get("text", ""), continuation=continuation)

    if t == "thinking":
        return ThinkingPart(text=d.get("text", ""), redacted=d.get("redacted", False), continuation=continuation)

    if t == "refusal":
        return RefusalPart(text=d.get("text", ""), continuation=continuation)

    if t == "citation":
        return CitationPart(text=d.get("text"), url=d.get("url"), title=d.get("title"), continuation=continuation)

    if t in ("image", "audio", "video", "document", "binary"):
        cls = PART_TYPES[t]
        kwargs: dict[str, Any] = {
            "media_type": d.get("media_type", ""),
            "data": d.get("data"),
            "url": d.get("url"),
            "file_id": d.get("file_id"),
            "path": d.get("path"),
            "continuation": continuation,
        }
        if t == "image":
            kwargs["detail"] = d.get("detail")
        return cls(**kwargs)

    if t == "tool_call":
        return ToolCallPart(
            id=d["id"],
            name=d["name"],
            input=d.get("input", {}),
            continuation=continuation,
        )

    if t == "tool_result":
        raw_content = d.get("content", [])
        if isinstance(raw_content, str):
            content = (TextPart(text=raw_content),) if raw_content else ()
        elif isinstance(raw_content, list):
            content = tuple(
                part_from_dict(c) if isinstance(c, dict) else TextPart(text=str(c))
                for c in raw_content
            )
        else:
            content = ()
        return ToolResultPart(
            id=d["id"],
            content=content,
            name=d.get("name"),
            is_error=d.get("is_error", False),
            continuation=continuation,
        )

    raise ValueError(f"unsupported part type: {t}")


# ─── Messages ────────────────────────────────────────────────────────

def message_to_dict(msg: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role, "parts": [part_to_dict(p) for p in msg.parts]}
    continuation = _continuation_to_json(msg.continuation)
    if continuation is not None:
        out["continuation"] = continuation
    return out


def message_from_dict(d: dict[str, Any]) -> Message:
    role = d["role"]
    parts_raw = d.get("parts", [])
    parts = tuple(
        part_from_dict(p) if isinstance(p, dict) else TextPart(text=str(p))
        for p in parts_raw
    )
    if not parts:
        raise ValueError(f"message for role '{role}' has no parts")
    return Message(role=role, parts=parts, continuation=_continuation_from_json(d.get("continuation")))


def messages_to_json(messages: list[Message] | tuple[Message, ...]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]


def messages_from_json(data: list[dict[str, Any]]) -> list[Message]:
    return [message_from_dict(d) for d in data]


# ─── Tools ───────────────────────────────────────────────────────────

def tool_to_dict(t: Tool) -> dict[str, Any]:
    if isinstance(t, FunctionTool):
        return _clean_mapping({
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        })
    if isinstance(t, BuiltinTool):
        return _clean_mapping({
            "type": "builtin",
            "name": t.name,
            "config": t.config,
        })
    raise TypeError(f"unsupported tool type: {type(t)}")


def tool_from_dict(d: dict[str, Any]) -> Tool:
    if d.get("type") == "builtin":
        return BuiltinTool(name=d["name"], config=d.get("config"))
    return FunctionTool(
        name=d["name"],
        description=d.get("description"),
        parameters=d.get("parameters", {"type": "object", "properties": {}}),
    )


# ─── Config ──────────────────────────────────────────────────────────

def tool_choice_to_dict(tc: ToolChoice) -> dict[str, Any]:
    return _clean_mapping({
        "mode": tc.mode,
        "allowed": list(tc.allowed),
        "parallel": tc.parallel,
    })


def tool_choice_from_dict(d: dict[str, Any]) -> ToolChoice:
    return ToolChoice(
        mode=d.get("mode", "auto"),
        allowed=tuple(d.get("allowed", [])),
        parallel=d.get("parallel"),
    )


def reasoning_to_dict(r: Reasoning) -> dict[str, Any]:
    return _clean_mapping({
        "effort": r.effort,
        "thinking_budget": r.thinking_budget,
        "total_budget": r.total_budget,
        "summary": r.summary,
    })


def reasoning_from_dict(d: dict[str, Any]) -> Reasoning:
    default_effort = "off" if d.get("enabled") is False else "medium"
    effort = d.get("effort", default_effort)
    # Legacy payloads could combine enabled=false with a budget.  In the
    # current type system, off reasoning has no budgets, so discard them.
    if effort == "off":
        return Reasoning(effort="off")
    return Reasoning(
        effort=effort,
        thinking_budget=d.get("thinking_budget", d.get("budget")),
        total_budget=d.get("total_budget"),
        summary=d.get("summary"),
    )


_CACHE_MODES = ("auto", "off")
_CACHE_RETENTIONS = ("short", "long")


def cache_config_to_dict(c: CacheConfig) -> dict[str, Any]:
    return _clean_mapping({
        "mode": c.mode,
        "retention": c.retention,
        "key": c.key,
        "prefix_until_index": c.prefix_until_index,
    })


def cache_config_from_dict(d: dict[str, Any]) -> CacheConfig:
    mode = d.get("mode", "auto")
    if mode not in _CACHE_MODES:
        raise ValueError(f"unsupported cache mode: {mode}")
    retention = d.get("retention")
    if retention is not None and retention not in _CACHE_RETENTIONS:
        raise ValueError(f"unsupported cache retention: {retention}")
    return CacheConfig(
        mode=mode,
        retention=retention,
        key=d.get("key"),
        prefix_until_index=d.get("prefix_until_index"),
    )


def config_to_dict(c: Config) -> dict[str, Any]:
    return _clean_mapping({
        "max_tokens": c.max_tokens,
        "temperature": c.temperature,
        "top_p": c.top_p,
        "top_k": c.top_k,
        "stop": list(c.stop),
        "response_format": c.response_format,
        "tool_choice": tool_choice_to_dict(c.tool_choice) if c.tool_choice else None,
        "reasoning": reasoning_to_dict(c.reasoning) if c.reasoning else None,
        "cache": cache_config_to_dict(c.cache) if c.cache else None,
        "extensions": c.extensions,
    })


def config_from_dict(d: dict[str, Any]) -> Config:
    return Config(
        max_tokens=d.get("max_tokens"),
        temperature=d.get("temperature"),
        top_p=d.get("top_p"),
        top_k=d.get("top_k"),
        stop=tuple(d.get("stop", [])),
        response_format=d.get("response_format"),
        tool_choice=tool_choice_from_dict(d["tool_choice"]) if isinstance(d.get("tool_choice"), dict) else None,
        reasoning=reasoning_from_dict(d["reasoning"]) if isinstance(d.get("reasoning"), dict) else None,
        cache=cache_config_from_dict(d["cache"]) if isinstance(d.get("cache"), dict) else None,
        extensions=d.get("extensions"),
    )


# ─── ErrorDetail ─────────────────────────────────────────────────────

def error_detail_to_dict(e: ErrorDetail) -> dict[str, Any]:
    return _clean_mapping({
        "code": e.code,
        "message": e.message,
        "provider_code": e.provider_code,
    })


def error_detail_from_dict(d: dict[str, Any]) -> ErrorDetail:
    return ErrorDetail(
        code=d["code"],
        message=d.get("message", ""),
        provider_code=d.get("provider_code"),
    )


# ─── Delta ───────────────────────────────────────────────────────────

def delta_to_dict(d: Delta) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": d.type,
        "part_index": d.part_index,
    }

    if isinstance(d, (TextDelta, ThinkingDelta)):
        out["text"] = d.text
    elif isinstance(d, AudioDelta):
        out["data"] = d.data
        out["url"] = d.url
        out["file_id"] = d.file_id
        out["media_type"] = d.media_type
    elif isinstance(d, ImageDelta):
        out["data"] = d.data
        out["url"] = d.url
        out["file_id"] = d.file_id
        out["media_type"] = d.media_type
    elif isinstance(d, ToolCallDelta):
        out["input"] = d.input
        out["id"] = d.id
        out["name"] = d.name
    elif isinstance(d, CitationDelta):
        out["text"] = d.text
        out["url"] = d.url
        out["title"] = d.title
    elif isinstance(d, ContinuationDelta):
        out["provider"] = d.provider
        out["kind"] = d.kind
        out["data"] = d.data
        out["part_index"] = d.part_index
    else:
        raise TypeError(f"unsupported delta type: {type(d)}")

    return {key: value for key, value in out.items() if value is not None}


def delta_from_dict(d: dict[str, Any]) -> Delta:
    t = d["type"]
    part_index = d.get("part_index", 0)

    if t == "text":
        return TextDelta(text=d.get("text", ""), part_index=part_index)
    if t == "thinking":
        return ThinkingDelta(text=d.get("text", ""), part_index=part_index)
    if t == "audio":
        return AudioDelta(
            data=d.get("data"),
            url=d.get("url"),
            file_id=d.get("file_id"),
            part_index=part_index,
            media_type=d.get("media_type"),
        )
    if t == "image":
        return ImageDelta(
            data=d.get("data"),
            url=d.get("url"),
            file_id=d.get("file_id"),
            part_index=part_index,
            media_type=d.get("media_type"),
        )
    if t == "tool_call":
        return ToolCallDelta(
            input=d.get("input", ""),
            part_index=part_index,
            id=d.get("id"),
            name=d.get("name"),
        )
    if t == "citation":
        return CitationDelta(
            text=d.get("text"),
            url=d.get("url"),
            title=d.get("title"),
            part_index=part_index,
        )
    if t == "continuation":
        return ContinuationDelta(
            provider=d["provider"],
            kind=d["kind"],
            data=d.get("data", {}),
            part_index=d.get("part_index"),
        )

    raise ValueError(f"unsupported delta type: {t}")


# ─── Usage ───────────────────────────────────────────────────────────

def usage_to_dict(u: Usage) -> dict[str, Any]:
    return _clean_mapping({
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "cache_read_tokens": u.cache_read_tokens,
        "cache_write_tokens": u.cache_write_tokens,
        "reasoning_tokens": u.reasoning_tokens,
        "input_audio_tokens": u.input_audio_tokens,
        "output_audio_tokens": u.output_audio_tokens,
    })


def usage_from_dict(d: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=d.get("input_tokens"),
        output_tokens=d.get("output_tokens"),
        total_tokens=d.get("total_tokens"),
        cache_read_tokens=d.get("cache_read_tokens"),
        cache_write_tokens=d.get("cache_write_tokens"),
        reasoning_tokens=d.get("reasoning_tokens"),
        input_audio_tokens=d.get("input_audio_tokens"),
        output_audio_tokens=d.get("output_audio_tokens"),
    )


# ─── StreamEvent ─────────────────────────────────────────────────────

def stream_event_to_dict(e: StreamEvent) -> dict[str, Any]:
    if isinstance(e, StreamStartEvent):
        return _clean_mapping({"type": e.type, "id": e.id, "model": e.model})
    if isinstance(e, StreamDeltaEvent):
        return {"type": e.type, "delta": delta_to_dict(e.delta)}
    if isinstance(e, StreamEndEvent):
        return _clean_mapping({
            "type": e.type,
            "finish_reason": e.finish_reason,
            "usage": usage_to_dict(e.usage) if e.usage else None,
            "provider_data": e.provider_data,
        })
    if isinstance(e, StreamErrorEvent):
        return {"type": e.type, "error": error_detail_to_dict(e.error)}
    raise TypeError(f"unsupported stream event type: {type(e)}")


def stream_event_from_dict(d: dict[str, Any]) -> StreamEvent:
    t = d["type"]
    if t == "start":
        return StreamStartEvent(id=d.get("id"), model=d.get("model"))
    if t == "delta":
        return StreamDeltaEvent(delta=delta_from_dict(d["delta"]))
    if t == "end":
        return StreamEndEvent(
            finish_reason=d.get("finish_reason"),
            usage=usage_from_dict(d["usage"]) if isinstance(d.get("usage"), dict) else None,
            provider_data=d.get("provider_data"),
        )
    if t == "error":
        return StreamErrorEvent(error=error_detail_from_dict(d["error"]))
    raise ValueError(f"unsupported stream event type: {t}")


# ─── Request / Response ─────────────────────────────────────────────

def request_to_dict(r: Request) -> dict[str, Any]:
    system: Any
    if isinstance(r.system, tuple):
        system = [part_to_dict(p) for p in r.system]
    else:
        system = r.system
    return _clean_mapping({
        "model": r.model,
        "messages": [message_to_dict(m) for m in r.messages],
        "system": system,
        "tools": [tool_to_dict(t) for t in r.tools],
        "config": config_to_dict(r.config),
    })


def request_from_dict(d: dict[str, Any]) -> Request:
    raw_system = d.get("system")
    system: str | tuple[Part, ...] | None
    if isinstance(raw_system, list):
        system = tuple(part_from_dict(x) for x in raw_system)
    else:
        system = raw_system
    return Request(
        model=d["model"],
        messages=tuple(message_from_dict(m) for m in d["messages"]),
        system=system,
        tools=tuple(tool_from_dict(t) for t in d.get("tools", [])),
        config=config_from_dict(d.get("config", {})),
    )


def response_to_dict(r: Response, *, include_provider_data: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": r.id,
        "model": r.model,
        "message": message_to_dict(r.message),
        "finish_reason": r.finish_reason,
        "usage": usage_to_dict(r.usage),
    }
    if include_provider_data and r.provider_data is not None:
        out["provider_data"] = r.provider_data
    return _clean_mapping(out)


def response_from_dict(d: dict[str, Any]) -> Response:
    return Response(
        id=d.get("id"),
        model=d["model"],
        message=message_from_dict(d["message"]),
        finish_reason=d["finish_reason"],
        usage=usage_from_dict(d.get("usage", {})),
        provider_data=d.get("provider_data"),
    )


# ─── ModelInfo ───────────────────────────────────────────────────────

def _inference_pricing_to_dict(p: InferencePricing) -> dict[str, Any]:
    return _clean_mapping({
        "input_per_million": p.input_per_million,
        "output_per_million": p.output_per_million,
        "cache_read_per_million": p.cache_read_per_million,
        "cache_write_per_million": p.cache_write_per_million,
        "currency": p.currency,
        "dimensions": p.dimensions,
    })


def _inference_pricing_from_dict(d: dict[str, Any]) -> InferencePricing:
    return InferencePricing(
        input_per_million=d.get("input_per_million"),
        output_per_million=d.get("output_per_million"),
        cache_read_per_million=d.get("cache_read_per_million"),
        cache_write_per_million=d.get("cache_write_per_million"),
        currency=d.get("currency", "USD"),
        dimensions=d.get("dimensions"),
    )


def _training_pricing_to_dict(p: TrainingPricing) -> dict[str, Any]:
    return _clean_mapping({
        "training_tokens_per_million": p.training_tokens_per_million,
        "gpu_second": p.gpu_second,
        "currency": p.currency,
        "dimensions": p.dimensions,
    })


def _training_pricing_from_dict(d: dict[str, Any]) -> TrainingPricing:
    return TrainingPricing(
        training_tokens_per_million=d.get("training_tokens_per_million"),
        gpu_second=d.get("gpu_second"),
        currency=d.get("currency", "USD"),
        dimensions=d.get("dimensions"),
    )


def _inference_model_info_to_dict(i: InferenceModelInfo) -> dict[str, Any]:
    return _clean_mapping({
        "input_modalities": list(i.input_modalities),
        "output_modalities": list(i.output_modalities),
        "context_window": i.context_window,
        "max_output_tokens": i.max_output_tokens,
        "supports_reasoning": i.supports_reasoning or None,
        "reasoning_efforts": list(i.reasoning_efforts),
        "pricing": _inference_pricing_to_dict(i.pricing) if i.pricing else None,
        "extensions": i.extensions,
    })


def _inference_model_info_from_dict(d: dict[str, Any]) -> InferenceModelInfo:
    return InferenceModelInfo(
        input_modalities=tuple(d.get("input_modalities", ["text"])),
        output_modalities=tuple(d.get("output_modalities", ["text"])),
        context_window=d.get("context_window"),
        max_output_tokens=d.get("max_output_tokens"),
        supports_reasoning=d.get("supports_reasoning", False),
        reasoning_efforts=tuple(d.get("reasoning_efforts", [])),
        pricing=_inference_pricing_from_dict(d["pricing"]) if isinstance(d.get("pricing"), dict) else None,
        extensions=d.get("extensions"),
    )


def _training_model_info_to_dict(t: TrainingModelInfo) -> dict[str, Any]:
    return _clean_mapping({
        "supports_lora": t.supports_lora or None,
        "supports_full_finetune": t.supports_full_finetune or None,
        "trainable_modalities": list(t.trainable_modalities),
        "pricing": _training_pricing_to_dict(t.pricing) if t.pricing else None,
        "extensions": t.extensions,
    })


def _training_model_info_from_dict(d: dict[str, Any]) -> TrainingModelInfo:
    return TrainingModelInfo(
        supports_lora=d.get("supports_lora", False),
        supports_full_finetune=d.get("supports_full_finetune", False),
        trainable_modalities=tuple(d.get("trainable_modalities", [])),
        pricing=_training_pricing_from_dict(d["pricing"]) if isinstance(d.get("pricing"), dict) else None,
        extensions=d.get("extensions"),
    )


def _model_origin_to_dict(o: ModelOrigin) -> dict[str, Any]:
    return _clean_mapping({
        "type": o.type,
        "id": o.id,
        "base_model": o.base_model,
        "provider_data": o.provider_data,
    })


def _model_origin_from_dict(d: dict[str, Any]) -> ModelOrigin:
    return ModelOrigin(
        type=d.get("type", "provider"),
        id=d.get("id"),
        base_model=d.get("base_model"),
        provider_data=d.get("provider_data"),
    )


def model_info_to_dict(m: ModelInfo) -> dict[str, Any]:
    origin = _model_origin_to_dict(m.origin)
    if origin == {"type": "provider"}:  # the default origin carries no information
        origin = {}
    return _clean_mapping({
        "id": m.id,
        "provider": m.provider,
        "api_family": m.api_family,
        "aliases": list(m.aliases),
        "origin": origin,
        "inference": _inference_model_info_to_dict(m.inference) if m.inference else None,
        "training": _training_model_info_to_dict(m.training) if m.training else None,
        "extensions": m.extensions,
    })


def model_info_from_dict(d: dict[str, Any]) -> ModelInfo:
    return ModelInfo(
        id=d["id"],
        provider=d["provider"],
        api_family=d["api_family"],
        aliases=tuple(d.get("aliases", [])),
        origin=_model_origin_from_dict(d["origin"]) if isinstance(d.get("origin"), dict) else ModelOrigin(),
        inference=_inference_model_info_from_dict(d["inference"]) if isinstance(d.get("inference"), dict) else None,
        training=_training_model_info_from_dict(d["training"]) if isinstance(d.get("training"), dict) else None,
        extensions=d.get("extensions"),
    )


# ─── AudioFormat ─────────────────────────────────────────────────────

def audio_format_to_dict(af: AudioFormat) -> dict[str, Any]:
    return _clean_mapping({
        "encoding": af.encoding,
        "sample_rate": af.sample_rate,
        "channels": af.channels,
    })


def audio_format_from_dict(d: dict[str, Any]) -> AudioFormat:
    return AudioFormat(
        encoding=d["encoding"],
        sample_rate=d["sample_rate"],
        channels=d.get("channels", 1),
    )


# ─── LiveConfig ──────────────────────────────────────────────────────

def live_config_to_dict(lc: LiveConfig) -> dict[str, Any]:
    system: Any
    if isinstance(lc.system, tuple):
        system = [part_to_dict(p) for p in lc.system]
    else:
        system = lc.system
    return _clean_mapping({
        "model": lc.model,
        "system": system,
        "tools": [tool_to_dict(t) for t in lc.tools],
        "voice": lc.voice,
        "input_format": audio_format_to_dict(lc.input_format) if lc.input_format else None,
        "output_format": audio_format_to_dict(lc.output_format) if lc.output_format else None,
        "extensions": lc.extensions,
    })


def live_config_from_dict(d: dict[str, Any]) -> LiveConfig:
    raw_system = d.get("system")
    system: str | tuple[Part, ...] | None
    if isinstance(raw_system, list):
        system = tuple(part_from_dict(x) for x in raw_system)
    else:
        system = raw_system
    return LiveConfig(
        model=d["model"],
        system=system,
        tools=tuple(tool_from_dict(t) for t in d.get("tools", [])),
        voice=d.get("voice"),
        input_format=audio_format_from_dict(d["input_format"]) if isinstance(d.get("input_format"), dict) else None,
        output_format=audio_format_from_dict(d["output_format"]) if isinstance(d.get("output_format"), dict) else None,
        extensions=d.get("extensions"),
    )


# ─── LiveClientEvent / LiveServerEvent ───────────────────────────────

def live_client_event_to_dict(e: LiveClientEvent) -> dict[str, Any]:
    if isinstance(e, LiveClientTurnEvent):
        return {"type": e.type, "parts": [part_to_dict(p) for p in e.parts], "turn_complete": e.turn_complete}
    if isinstance(e, (LiveClientAudioEvent, LiveClientImageEvent)):
        return {"type": e.type, "data": e.data, "media_type": e.media_type}
    if isinstance(e, LiveClientTextEvent):
        return {"type": e.type, "text": e.text}
    if isinstance(e, LiveClientToolResultEvent):
        return {"type": e.type, "id": e.id, "content": [part_to_dict(p) for p in e.content]}
    if isinstance(e, (LiveClientInterruptEvent, LiveClientEndAudioEvent)):
        return {"type": e.type}
    raise TypeError(f"unsupported live client event type: {type(e)}")


def live_client_event_from_dict(d: dict[str, Any]) -> LiveClientEvent:
    t = d["type"]
    if t == "turn":
        return LiveClientTurnEvent(
            parts=tuple(part_from_dict(p) for p in d.get("parts", [])),
            turn_complete=d.get("turn_complete", True),
        )
    if t == "audio":
        return LiveClientAudioEvent(data=d["data"], media_type=d.get("media_type", "audio/pcm;rate=16000"))
    if t == "image":
        return LiveClientImageEvent(data=d["data"], media_type=d.get("media_type", "image/jpeg"))
    if t == "text":
        return LiveClientTextEvent(text=d.get("text", ""))
    if t == "tool_result":
        return LiveClientToolResultEvent(
            id=d["id"],
            content=tuple(part_from_dict(p) for p in d.get("content", [])),
        )
    if t == "interrupt":
        return LiveClientInterruptEvent()
    if t == "end_audio":
        return LiveClientEndAudioEvent()
    raise ValueError(f"unsupported live client event type: {t}")


def live_server_event_to_dict(e: LiveServerEvent) -> dict[str, Any]:
    if isinstance(e, LiveServerAudioEvent):
        return _clean_mapping({"type": e.type, "data": e.data, "media_type": e.media_type})
    if isinstance(e, LiveServerTextEvent):
        return {"type": e.type, "text": e.text}
    if isinstance(e, LiveServerToolCallEvent):
        return {"type": e.type, "id": e.id, "name": e.name, "input": e.input}
    if isinstance(e, LiveServerToolCallDeltaEvent):
        return _clean_mapping({
            "type": e.type,
            "id": e.id,
            "name": e.name,
            "input_delta": e.input_delta,
        })
    if isinstance(e, LiveServerInterruptedEvent):
        return {"type": e.type}
    if isinstance(e, LiveServerTurnEndEvent):
        return _clean_mapping({"type": e.type, "usage": usage_to_dict(e.usage)})
    if isinstance(e, LiveServerErrorEvent):
        return {"type": e.type, "error": error_detail_to_dict(e.error)}
    raise TypeError(f"unsupported live server event type: {type(e)}")


def live_server_event_from_dict(d: dict[str, Any]) -> LiveServerEvent:
    t = d["type"]
    if t == "audio":
        return LiveServerAudioEvent(data=d["data"], media_type=d.get("media_type"))
    if t == "text":
        return LiveServerTextEvent(text=d.get("text", ""))
    if t == "tool_call":
        return LiveServerToolCallEvent(id=d["id"], name=d["name"], input=d.get("input", {}))
    if t == "tool_call_delta":
        return LiveServerToolCallDeltaEvent(
            input_delta=d.get("input_delta", ""), id=d.get("id"), name=d.get("name")
        )
    if t == "interrupted":
        return LiveServerInterruptedEvent()
    if t == "turn_end":
        return LiveServerTurnEndEvent(usage=usage_from_dict(d.get("usage", {})))
    if t == "error":
        return LiveServerErrorEvent(error=error_detail_from_dict(d["error"]))
    raise ValueError(f"unsupported live server event type: {t}")
