from __future__ import annotations

from typing import Any

from .types import (
    Config,
    DataSource,
    LMRequest,
    LMResponse,
    Message,
    Part,
    PartDelta,
    StreamEvent,
    Tool,
    ToolConfig,
    Usage,
)


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == () or value == [] or value == {}


def _clean_sequence(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if isinstance(value, dict):
            value = _clean_mapping(value)
        elif isinstance(value, list):
            value = _clean_sequence(value)
        if _is_empty(value):
            continue
        out.append(value)
    return out


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            value = _clean_mapping(value)
        elif isinstance(value, list):
            value = _clean_sequence(value)
        elif isinstance(value, tuple):
            value = _clean_sequence(list(value))
        if _is_empty(value):
            continue
        out[key] = value
    return out


def data_source_to_dict(value: DataSource) -> dict[str, Any]:
    return _clean_mapping(
        {
            "type": value.type,
            "media_type": value.media_type,
            "data": value.data,
            "url": value.url,
            "file_id": value.file_id,
            "detail": value.detail,
        }
    )


def part_to_dict(value: Part) -> dict[str, Any]:
    return _clean_mapping(
        {
            "type": value.type,
            "text": value.text,
            "source": data_source_to_dict(value.source) if value.source is not None else None,
            "id": value.id,
            "name": value.name,
            "input": value.input,
            "content": [part_to_dict(x) for x in value.content],
            "is_error": value.is_error,
            "redacted": value.redacted,
            "summary": value.summary,
            "url": value.url,
            "title": value.title,
            "metadata": value.metadata,
        }
    )


def message_to_dict(value: Message) -> dict[str, Any]:
    return _clean_mapping(
        {
            "role": value.role,
            "parts": [part_to_dict(x) for x in value.parts],
            "name": value.name,
        }
    )


def tool_to_dict(value: Tool) -> dict[str, Any]:
    return _clean_mapping(
        {
            "name": value.name,
            "type": value.type,
            "description": value.description,
            "parameters": value.parameters,
            "builtin_config": value.builtin_config,
        }
    )


def tool_config_to_dict(value: ToolConfig) -> dict[str, Any]:
    return _clean_mapping(
        {
            "mode": value.mode,
            "allowed": list(value.allowed),
            "parallel": value.parallel,
        }
    )


def config_to_dict(value: Config) -> dict[str, Any]:
    return _clean_mapping(
        {
            "max_tokens": value.max_tokens,
            "temperature": value.temperature,
            "top_p": value.top_p,
            "top_k": value.top_k,
            "stop": list(value.stop),
            "response_format": value.response_format,
            "tool_config": tool_config_to_dict(value.tool_config) if value.tool_config is not None else None,
            "reasoning": value.reasoning,
            "provider": value.provider,
        }
    )


def request_to_dict(value: LMRequest) -> dict[str, Any]:
    system: str | list[dict[str, Any]] | None
    if isinstance(value.system, tuple):
        system = [part_to_dict(x) for x in value.system]
    else:
        system = value.system

    cfg = config_to_dict(value.config)
    return _clean_mapping(
        {
            "model": value.model,
            "messages": [message_to_dict(x) for x in value.messages],
            "system": system,
            "tools": [tool_to_dict(x) for x in value.tools],
            "config": cfg,
        }
    )


def usage_to_dict(value: Usage) -> dict[str, Any]:
    return _clean_mapping(
        {
            "input_tokens": value.input_tokens,
            "output_tokens": value.output_tokens,
            "total_tokens": value.total_tokens,
            "cache_read_tokens": value.cache_read_tokens,
            "cache_write_tokens": value.cache_write_tokens,
            "reasoning_tokens": value.reasoning_tokens,
        }
    )


def response_to_dict(value: LMResponse, *, include_provider: bool = False) -> dict[str, Any]:
    out = {
        "id": value.id,
        "model": value.model,
        "message": message_to_dict(value.message),
        "finish_reason": value.finish_reason,
        "usage": usage_to_dict(value.usage),
    }
    if include_provider:
        out["provider"] = value.provider
    return _clean_mapping(out)


def part_delta_to_dict(value: PartDelta) -> dict[str, Any]:
    return _clean_mapping(
        {
            "type": value.type,
            "text": value.text,
            "data": value.data,
            "input": value.input,
        }
    )


def stream_event_to_dict(value: StreamEvent) -> dict[str, Any]:
    if isinstance(value.delta, PartDelta):
        delta: dict[str, Any] | None = part_delta_to_dict(value.delta)
    elif isinstance(value.delta, dict):
        delta = _clean_mapping(value.delta)
    else:
        delta = None

    return _clean_mapping(
        {
            "type": value.type,
            "id": value.id,
            "model": value.model,
            "part_index": value.part_index,
            "delta": delta,
            "part_type": value.part_type,
            "finish_reason": value.finish_reason,
            "usage": usage_to_dict(value.usage) if value.usage is not None else None,
            "error": _clean_mapping(value.error) if value.error is not None else None,
        }
    )


def data_source_from_dict(value: dict[str, Any]) -> DataSource:
    return DataSource(**value)


def part_from_dict(value: dict[str, Any]) -> Part:
    return Part.from_dict(value)


def message_from_dict(value: dict[str, Any]) -> Message:
    return Message(
        role=value["role"],
        parts=tuple(part_from_dict(x) for x in value["parts"]),
        name=value.get("name"),
    )


def tool_from_dict(value: dict[str, Any]) -> Tool:
    return Tool(
        name=value["name"],
        type=value.get("type", "function"),
        description=value.get("description"),
        parameters=value.get("parameters"),
        builtin_config=value.get("builtin_config"),
    )


def tool_config_from_dict(value: dict[str, Any]) -> ToolConfig:
    return ToolConfig(
        mode=value.get("mode", "auto"),
        allowed=tuple(value.get("allowed", [])),
        parallel=value.get("parallel"),
    )


def config_from_dict(value: dict[str, Any]) -> Config:
    return Config(
        max_tokens=value.get("max_tokens"),
        temperature=value.get("temperature"),
        top_p=value.get("top_p"),
        top_k=value.get("top_k"),
        stop=tuple(value.get("stop", [])),
        response_format=value.get("response_format"),
        tool_config=tool_config_from_dict(value["tool_config"]) if isinstance(value.get("tool_config"), dict) else None,
        reasoning=value.get("reasoning"),
        provider=value.get("provider"),
    )


def request_from_dict(value: dict[str, Any]) -> LMRequest:
    raw_system = value.get("system")
    system: str | tuple[Part, ...] | None
    if isinstance(raw_system, list):
        system = tuple(part_from_dict(x) for x in raw_system)
    else:
        system = raw_system

    return LMRequest(
        model=value["model"],
        messages=tuple(message_from_dict(x) for x in value["messages"]),
        system=system,
        tools=tuple(tool_from_dict(x) for x in value.get("tools", [])),
        config=config_from_dict(value.get("config", {})),
    )


def usage_from_dict(value: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=value.get("input_tokens", 0),
        output_tokens=value.get("output_tokens", 0),
        total_tokens=value.get("total_tokens", 0),
        cache_read_tokens=value.get("cache_read_tokens"),
        cache_write_tokens=value.get("cache_write_tokens"),
        reasoning_tokens=value.get("reasoning_tokens"),
    )


def response_from_dict(value: dict[str, Any]) -> LMResponse:
    return LMResponse(
        id=value.get("id", ""),
        model=value["model"],
        message=message_from_dict(value["message"]),
        finish_reason=value["finish_reason"],
        usage=usage_from_dict(value.get("usage", {})),
        provider=value.get("provider"),
    )


def part_delta_from_dict(value: dict[str, Any]) -> PartDelta:
    return PartDelta(
        type=value["type"],
        text=value.get("text"),
        data=value.get("data"),
        input=value.get("input"),
    )


def stream_event_from_dict(value: dict[str, Any]) -> StreamEvent:
    raw_delta = value.get("delta")
    delta: PartDelta | dict[str, Any] | None
    if isinstance(raw_delta, dict) and raw_delta.get("type") in {"text", "tool_call", "thinking", "audio"}:
        delta = part_delta_from_dict(raw_delta)
    else:
        delta = raw_delta

    return StreamEvent(
        type=value["type"],
        id=value.get("id"),
        model=value.get("model"),
        part_index=value.get("part_index"),
        delta=delta,
        part_type=value.get("part_type"),
        finish_reason=value.get("finish_reason"),
        usage=usage_from_dict(value["usage"]) if isinstance(value.get("usage"), dict) else None,
        error=value.get("error"),
    )


__all__ = [
    "config_from_dict",
    "config_to_dict",
    "data_source_from_dict",
    "data_source_to_dict",
    "message_from_dict",
    "message_to_dict",
    "part_delta_from_dict",
    "part_delta_to_dict",
    "part_from_dict",
    "part_to_dict",
    "request_from_dict",
    "request_to_dict",
    "response_from_dict",
    "response_to_dict",
    "stream_event_from_dict",
    "stream_event_to_dict",
    "tool_config_from_dict",
    "tool_config_to_dict",
    "tool_from_dict",
    "tool_to_dict",
    "usage_from_dict",
    "usage_to_dict",
]
