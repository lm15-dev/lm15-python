from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias


JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonArray: TypeAlias = list[JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]

Role = Literal["user", "assistant", "tool"]
PartType = Literal[
    "text",
    "image",
    "audio",
    "video",
    "document",
    "tool_call",
    "tool_result",
    "thinking",
    "refusal",
    "citation",
]
ToolType = Literal["function", "builtin"]
ReasoningEffort = Literal["low", "medium", "high"]
FinishReason = Literal["stop", "length", "tool_call", "content_filter", "error"]
DataSourceType = Literal["base64", "url", "file"]
ResponseFormatType = Literal["text", "json", "json_schema"]
StreamEventType = Literal["start", "delta", "part_start", "part_end", "end", "error"]
PartDeltaType = Literal["text", "tool_call", "thinking", "audio"]
ErrorCode = Literal["auth", "billing", "rate_limit", "invalid_request", "context_length", "timeout", "server", "provider"]


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


def _validate_json_value(value: Any, *, field_name: str) -> None:
    if value is None:
        return
    if not _is_json_value(value):
        raise TypeError(f"{field_name} must be JSON-serializable")


def _validate_json_object(value: Any, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not _is_json_value(value):
        raise TypeError(f"{field_name} must be a JSON object")


@dataclass(slots=True, frozen=True)
class DataSource:
    type: DataSourceType
    media_type: str | None = None
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    detail: Literal["low", "high", "auto"] | None = None

    def __post_init__(self) -> None:
        if self.type == "base64":
            if not self.data:
                raise ValueError("DataSource(type='base64') requires data")
            if not self.media_type:
                raise ValueError("DataSource(type='base64') requires media_type")
        elif self.type == "url":
            if not self.url:
                raise ValueError("DataSource(type='url') requires url")
        elif self.type == "file":
            if not self.file_id:
                raise ValueError("DataSource(type='file') requires file_id")
        else:
            raise ValueError(f"unsupported data source type: {self.type}")

    @property
    def bytes(self) -> bytes:
        if self.type != "base64" or not self.data:
            raise ValueError(
                f"DataSource(type='{self.type}') has no inline bytes — "
                f"only base64 sources can be decoded. "
                f"{'Use the url to fetch the data.' if self.type == 'url' else ''}"
                f"{'Use the file_id to retrieve the file.' if self.type == 'file' else ''}"
            )
        import base64

        return base64.b64decode(self.data)


class Part:
    type: ClassVar[PartType]
    _missing_defaults: ClassVar[dict[str, Any]] = {
        "text": None,
        "source": None,
        "id": None,
        "name": None,
        "input": None,
        "content": (),
        "is_error": None,
        "redacted": None,
        "summary": None,
        "url": None,
        "title": None,
        "metadata": None,
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._missing_defaults:
            return self._missing_defaults[name]
        raise AttributeError(name)

    @staticmethod
    def text_part(text: str) -> "TextPart":
        return TextPart(text=text)

    @staticmethod
    def thinking(text: str, *, redacted: bool | None = None, summary: str | None = None, metadata: JsonObject | None = None) -> "ThinkingPart":
        return ThinkingPart(text=text, redacted=redacted, summary=summary, metadata=metadata)

    @staticmethod
    def refusal(text: str) -> "RefusalPart":
        return RefusalPart(text=text)

    @staticmethod
    def citation(text: str | None = None, url: str | None = None, title: str | None = None) -> "CitationPart":
        return CitationPart(text=text, url=url, title=title)

    @staticmethod
    def _cache_metadata(cache: bool | JsonObject | None = None) -> JsonObject | None:
        if cache is None:
            return None
        if cache is True:
            return {"cache": True}
        if isinstance(cache, dict):
            return {"cache": cache}
        return {"cache": bool(cache)}

    @staticmethod
    def _media_part(
        kind: Literal["image", "audio", "video", "document"],
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | JsonObject | None = None,
    ) -> "Part":
        provided = sum(1 for x in (url, data, file_id) if x is not None)
        if provided != 1:
            raise ValueError(f"Part.{kind} requires exactly one of url, data, file_id")

        if url is not None:
            source = DataSource(type="url", url=url, media_type=media_type, detail=detail)
        elif file_id is not None:
            source = DataSource(type="file", file_id=file_id, media_type=media_type, detail=detail)
        else:
            if isinstance(data, bytes):
                import base64

                payload = base64.b64encode(data).decode("ascii")
            else:
                payload = data or ""
            source = DataSource(type="base64", data=payload, media_type=media_type or "application/octet-stream", detail=detail)

        metadata = Part._cache_metadata(cache)
        if kind == "image":
            return ImagePart(source=source, metadata=metadata)
        if kind == "audio":
            return AudioPart(source=source, metadata=metadata)
        if kind == "video":
            return VideoPart(source=source, metadata=metadata)
        return DocumentPart(source=source, metadata=metadata)

    @staticmethod
    def image(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | JsonObject | None = None,
    ) -> "ImagePart":
        return Part._media_part("image", url=url, data=data, file_id=file_id, media_type=media_type or "image/png", detail=detail, cache=cache)  # type: ignore[return-value]

    @staticmethod
    def audio(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | JsonObject | None = None,
    ) -> "AudioPart":
        return Part._media_part("audio", url=url, data=data, file_id=file_id, media_type=media_type or "audio/wav", detail=detail, cache=cache)  # type: ignore[return-value]

    @staticmethod
    def video(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | JsonObject | None = None,
    ) -> "VideoPart":
        return Part._media_part("video", url=url, data=data, file_id=file_id, media_type=media_type or "video/mp4", detail=detail, cache=cache)  # type: ignore[return-value]

    @staticmethod
    def document(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | JsonObject | None = None,
    ) -> "DocumentPart":
        return Part._media_part("document", url=url, data=data, file_id=file_id, media_type=media_type or "application/pdf", detail=detail, cache=cache)  # type: ignore[return-value]

    @staticmethod
    def tool_call(id: str, name: str, input: JsonObject) -> "ToolCallPart":
        return ToolCallPart(id=id, name=name, input=input)

    @staticmethod
    def tool_result(id: str, content: list["Part"], is_error: bool | None = None, name: str | None = None) -> "ToolResultPart":
        return ToolResultPart(id=id, name=name, content=tuple(content), is_error=is_error)

    @property
    def bytes(self) -> bytes:
        if self.type in ("text", "thinking", "refusal", "citation", "tool_call", "tool_result"):
            raise TypeError(
                f"Part(type='{self.type}') is not a media part — "
                f".bytes is only available on image, audio, video, and document parts"
            )
        source = self.source
        if source is None:
            raise ValueError(f"Part(type='{self.type}') has no source")
        return source.bytes

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Part":
        part_type = value["type"]
        source_value = value.get("source")
        source = DataSource(**source_value) if isinstance(source_value, dict) else source_value
        content = tuple(cls.from_dict(x) if isinstance(x, dict) else x for x in value.get("content", []))
        payload = dict(value)
        payload.pop("type", None)
        if source is not None:
            payload["source"] = source
        if content:
            payload["content"] = content
        part_cls = _PART_TYPE_MAP.get(part_type)
        if part_cls is None:
            raise ValueError(f"unsupported part type: {part_type}")
        return part_cls(**payload)


@dataclass(slots=True, frozen=True)
class TextPart(Part):
    text: str | None = None
    metadata: JsonObject | None = None
    type: ClassVar[Literal["text"]] = "text"

    def __post_init__(self) -> None:
        if self.text is None:
            raise ValueError("TextPart requires text")
        _validate_json_object(self.metadata, field_name="metadata")


@dataclass(slots=True, frozen=True)
class ThinkingPart(Part):
    text: str | None = None
    redacted: bool | None = None
    summary: str | None = None
    metadata: JsonObject | None = None
    type: ClassVar[Literal["thinking"]] = "thinking"

    def __post_init__(self) -> None:
        if self.text is None:
            raise ValueError("ThinkingPart requires text")
        _validate_json_object(self.metadata, field_name="metadata")


@dataclass(slots=True, frozen=True)
class RefusalPart(Part):
    text: str | None = None
    type: ClassVar[Literal["refusal"]] = "refusal"

    def __post_init__(self) -> None:
        if self.text is None:
            raise ValueError("RefusalPart requires text")


@dataclass(slots=True, frozen=True)
class CitationPart(Part):
    text: str | None = None
    url: str | None = None
    title: str | None = None
    type: ClassVar[Literal["citation"]] = "citation"


@dataclass(slots=True, frozen=True)
class _MediaPart(Part):
    source: DataSource | None = None
    metadata: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError(f"{type(self).__name__} requires source")
        _validate_json_object(self.metadata, field_name="metadata")


@dataclass(slots=True, frozen=True)
class ImagePart(_MediaPart):
    type: ClassVar[Literal["image"]] = "image"


@dataclass(slots=True, frozen=True)
class AudioPart(_MediaPart):
    type: ClassVar[Literal["audio"]] = "audio"


@dataclass(slots=True, frozen=True)
class VideoPart(_MediaPart):
    type: ClassVar[Literal["video"]] = "video"


@dataclass(slots=True, frozen=True)
class DocumentPart(_MediaPart):
    type: ClassVar[Literal["document"]] = "document"


@dataclass(slots=True, frozen=True)
class ToolCallPart(Part):
    id: str | None = None
    name: str | None = None
    input: JsonObject | None = None
    type: ClassVar[Literal["tool_call"]] = "tool_call"

    def __post_init__(self) -> None:
        if not self.id or not self.name or self.input is None:
            raise ValueError("ToolCallPart requires id, name, input")
        _validate_json_object(self.input, field_name="input")


@dataclass(slots=True, frozen=True)
class ToolResultPart(Part):
    id: str | None = None
    name: str | None = None
    content: tuple[Part, ...] = field(default_factory=tuple)
    is_error: bool | None = None
    type: ClassVar[Literal["tool_result"]] = "tool_result"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolResultPart requires id")


_PART_TYPE_MAP: dict[PartType, type[Part]] = {
    "text": TextPart,
    "image": ImagePart,
    "audio": AudioPart,
    "video": VideoPart,
    "document": DocumentPart,
    "tool_call": ToolCallPart,
    "tool_result": ToolResultPart,
    "thinking": ThinkingPart,
    "refusal": RefusalPart,
    "citation": CitationPart,
}


class Tool:
    type: ClassVar[ToolType]
    _missing_defaults: ClassVar[dict[str, Any]] = {
        "description": None,
        "parameters": None,
        "builtin_config": None,
        "fn": None,
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._missing_defaults:
            return self._missing_defaults[name]
        raise AttributeError(name)

    @staticmethod
    def from_fn(fn: Any) -> "FunctionTool":
        import inspect

        sig = inspect.signature(fn)
        hints = inspect.get_annotations(fn, eval_str=True)
        properties: JsonObject = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                continue
            ann = hints.get(name, str)
            origin = getattr(ann, "__origin__", None)
            if origin in (list, tuple, set):
                json_type: JsonObject = {"type": "array"}
            elif origin is dict:
                json_type = {"type": "object"}
            elif ann in (int,):
                json_type = {"type": "integer"}
            elif ann in (float,):
                json_type = {"type": "number"}
            elif ann in (bool,):
                json_type = {"type": "boolean"}
            else:
                json_type = {"type": "string"}
            properties[name] = json_type
            if param.default is inspect.Parameter.empty:
                required.append(name)

        schema: JsonObject = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        return FunctionTool(
            name=fn.__name__,
            description=(inspect.getdoc(fn) or "").strip() or None,
            parameters=schema,
            # fn intentionally omitted: inferred schema, manual execution
        )


@dataclass(slots=True, frozen=True)
class FunctionTool(Tool):
    name: str
    description: str | None = None
    parameters: JsonObject | None = None
    fn: Any = None
    type: ClassVar[Literal["function"]] = "function"

    def __post_init__(self) -> None:
        if self.parameters is None:
            object.__setattr__(self, "parameters", {"type": "object", "properties": {}})
        _validate_json_object(self.parameters, field_name="parameters")


@dataclass(slots=True, frozen=True)
class BuiltinTool(Tool):
    name: str
    description: str | None = None
    builtin_config: JsonObject | None = None
    type: ClassVar[Literal["builtin"]] = "builtin"

    def __post_init__(self) -> None:
        _validate_json_object(self.builtin_config, field_name="builtin_config")


_TOOL_TYPE_MAP: dict[ToolType, type[Tool]] = {
    "function": FunctionTool,
    "builtin": BuiltinTool,
}


@dataclass(slots=True, frozen=True)
class ToolCallInfo:
    id: str
    name: str
    input: JsonObject


@dataclass(slots=True, frozen=True)
class ToolConfig:
    mode: Literal["auto", "required", "none"] = "auto"
    allowed: tuple[str, ...] = ()
    parallel: bool | None = None


@dataclass(slots=True, frozen=True)
class ReasoningConfig:
    enabled: bool
    budget: int | None = None
    effort: ReasoningEffort | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and self.budget <= 0:
            raise ValueError("budget must be > 0")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> JsonObject:
        out: JsonObject = {"enabled": self.enabled}
        if self.budget is not None:
            out["budget"] = self.budget
        if self.effort is not None:
            out["effort"] = self.effort
        return out


@dataclass(slots=True, frozen=True)
class Config:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    response_format: JsonObject | None = None
    tool_config: ToolConfig | None = None
    reasoning: ReasoningConfig | JsonObject | None = None
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")
        _validate_json_object(self.response_format, field_name="response_format")
        _validate_json_object(self.provider, field_name="provider")
        if isinstance(self.reasoning, dict):
            object.__setattr__(self, "reasoning", ReasoningConfig(**self.reasoning))
        elif self.reasoning is not None and not isinstance(self.reasoning, ReasoningConfig):
            raise TypeError("reasoning must be a ReasoningConfig or JSON object")


@dataclass(slots=True, frozen=True)
class LMRequest:
    model: str
    messages: tuple[Message, ...]
    system: str | tuple[Part, ...] | None = None
    tools: tuple[Tool, ...] = ()
    config: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if isinstance(self.system, tuple) and not self.system:
            raise ValueError("system parts cannot be empty")


@dataclass(slots=True, frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class Message:
    role: Role
    parts: tuple[Part, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {self.role}")
        if not self.parts:
            raise ValueError("Message.parts cannot be empty")

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", parts=(Part.text_part(text),))

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(role="assistant", parts=(Part.text_part(text),))

    @staticmethod
    def tool_results(results: dict[str, str | Part | list[Part]]) -> "Message":
        parts: list[Part] = []
        for call_id, value in results.items():
            if isinstance(value, Part):
                content = [value]
            elif isinstance(value, list) and all(isinstance(x, Part) for x in value):
                content = value
            else:
                content = [Part.text_part(str(value))]
            parts.append(Part.tool_result(call_id, content))
        return Message(role="tool", parts=tuple(parts))


@dataclass(slots=True, frozen=True)
class LMResponse:
    id: str
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")

    @property
    def text(self) -> str | None:
        texts = [p.text for p in self.message.parts if p.type == "text" and p.text is not None]
        return "\n".join(texts) if texts else None

    @property
    def image(self) -> ImagePart | None:
        for p in self.message.parts:
            if isinstance(p, ImagePart):
                return p
        return None

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.message.parts if isinstance(p, ImagePart)]

    @property
    def audio(self) -> AudioPart | None:
        for p in self.message.parts:
            if isinstance(p, AudioPart):
                return p
        return None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.message.parts if isinstance(p, ToolCallPart)]

    @property
    def thinking(self) -> str | None:
        texts = [p.text for p in self.message.parts if isinstance(p, ThinkingPart) and p.text is not None]
        return "\n".join(texts) if texts else None

    @property
    def citations(self) -> list[CitationPart]:
        return [p for p in self.message.parts if isinstance(p, CitationPart)]

    @property
    def json(self) -> Any:
        import json as _json

        text = self.text
        if text is None:
            raise ValueError(
                "Cannot parse response as JSON: response contains no text. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        try:
            return _json.loads(text)
        except _json.JSONDecodeError as e:
            preview = text[:200] + ("..." if len(text) > 200 else "")
            raise ValueError(
                f"Cannot parse response as JSON: {e}\n"
                f"Raw text: {preview}"
            ) from e

    @property
    def image_bytes(self) -> bytes:
        img = self.image
        if img is None:
            raise ValueError(
                "Response contains no image part. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        return img.bytes

    @property
    def audio_bytes(self) -> bytes:
        aud = self.audio
        if aud is None:
            raise ValueError(
                "Response contains no audio part. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        return aud.bytes


@dataclass(slots=True, frozen=True)
class ErrorInfo:
    code: ErrorCode
    message: str
    provider_code: str | None = None

    def __getitem__(self, key: str) -> Any:
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        data = {"code": self.code, "message": self.message}
        if self.provider_code is not None:
            data["provider_code"] = self.provider_code
        return data.items()

    def to_dict(self) -> dict[str, str]:
        out = {"code": self.code, "message": self.message}
        if self.provider_code is not None:
            out["provider_code"] = self.provider_code
        return out


@dataclass(slots=True, frozen=True)
class PartDelta:
    type: PartDeltaType
    text: str | None = None
    data: str | None = None
    input: str | None = None

    def __post_init__(self) -> None:
        if self.type == "text" and self.text is None:
            raise ValueError("PartDelta(type='text') requires text")
        if self.type == "thinking" and self.text is None:
            raise ValueError("PartDelta(type='thinking') requires text")
        if self.type == "audio" and self.data is None:
            raise ValueError("PartDelta(type='audio') requires data")
        if self.type == "tool_call" and self.input is None:
            raise ValueError("PartDelta(type='tool_call') requires input")


@dataclass(slots=True, frozen=True)
class StreamEvent:
    type: StreamEventType
    id: str | None = None
    model: str | None = None
    part_index: int | None = None
    delta: PartDelta | JsonObject | None = None
    part_type: str | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    error: ErrorInfo | dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.type == "delta" and self.delta is None:
            raise ValueError("StreamEvent(type='delta') requires delta")
        if self.type == "error" and self.error is None:
            raise ValueError("StreamEvent(type='error') requires error")
        if isinstance(self.delta, dict):
            _validate_json_object(self.delta, field_name="delta")
        if isinstance(self.error, dict):
            object.__setattr__(self, "error", ErrorInfo(**self.error))

    @property
    def delta_text(self) -> str | None:
        if self.delta is None:
            return None
        if isinstance(self.delta, PartDelta):
            return self.delta.text if self.delta.type == "text" else None
        return self.delta.get("text") if self.delta.get("type") == "text" else None


@dataclass(slots=True, frozen=True)
class EmbeddingRequest:
    model: str
    inputs: tuple[str, ...]
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class EmbeddingResponse:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage = field(default_factory=Usage)
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class FileUploadRequest:
    model: str | None = None
    filename: str = "file.bin"
    bytes_data: bytes = b""
    media_type: str = "application/octet-stream"
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class FileUploadResponse:
    id: str
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class BatchRequest:
    model: str
    requests: tuple[LMRequest, ...]
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class BatchResponse:
    id: str
    status: str
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class ImageGenerationRequest:
    model: str
    prompt: str
    size: str | None = None
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class ImageGenerationResponse:
    images: tuple[DataSource, ...]
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class AudioGenerationRequest:
    model: str
    prompt: str
    voice: str | None = None
    format: str | None = None
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class AudioGenerationResponse:
    audio: DataSource
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class AudioFormat:
    encoding: Literal["pcm16", "opus", "mp3", "aac"]
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")


@dataclass(slots=True, frozen=True)
class LiveConfig:
    model: str
    system: str | tuple[Part, ...] | None = None
    tools: tuple[Tool, ...] = ()
    voice: str | None = None
    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    provider: JsonObject | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if isinstance(self.system, tuple) and not self.system:
            raise ValueError("system parts cannot be empty")
        _validate_json_object(self.provider, field_name="provider")


@dataclass(slots=True, frozen=True)
class LiveClientEvent:
    type: Literal["audio", "video", "text", "tool_result", "interrupt", "end_audio"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    content: tuple[Part, ...] = ()

    def __post_init__(self) -> None:
        if self.type in {"audio", "video"} and self.data is None:
            raise ValueError(f"LiveClientEvent(type='{self.type}') requires data")
        if self.type == "text" and self.text is None:
            raise ValueError("LiveClientEvent(type='text') requires text")
        if self.type == "tool_result":
            if not self.id:
                raise ValueError("LiveClientEvent(type='tool_result') requires id")
            if not self.content:
                raise ValueError("LiveClientEvent(type='tool_result') requires content")


@dataclass(slots=True, frozen=True)
class LiveServerEvent:
    type: Literal["audio", "text", "tool_call", "interrupted", "turn_end", "error"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: JsonObject | None = None
    usage: Usage | None = None
    error: ErrorInfo | dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.type == "audio" and self.data is None:
            raise ValueError("LiveServerEvent(type='audio') requires data")
        if self.type == "text" and self.text is None:
            raise ValueError("LiveServerEvent(type='text') requires text")
        if self.type == "tool_call":
            if not self.id or not self.name or self.input is None:
                raise ValueError("LiveServerEvent(type='tool_call') requires id, name, input")
            _validate_json_object(self.input, field_name="input")
        if self.type == "turn_end" and self.usage is None:
            raise ValueError("LiveServerEvent(type='turn_end') requires usage")
        if self.type == "error" and self.error is None:
            raise ValueError("LiveServerEvent(type='error') requires error")
        if isinstance(self.error, dict):
            object.__setattr__(self, "error", ErrorInfo(**self.error))
