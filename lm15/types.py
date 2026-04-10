from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
FinishReason = Literal["stop", "length", "tool_call", "content_filter", "error"]
DataSourceType = Literal["base64", "url", "file"]
ResponseFormatType = Literal["text", "json", "json_schema"]
StreamEventType = Literal["start", "delta", "part_start", "part_end", "end", "error"]
PartDeltaType = Literal["text", "tool_call", "thinking", "audio"]


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
        """Decode base64 data and return raw bytes.

        Raises ValueError if the source is not base64-encoded.
        """
        if self.type != "base64" or not self.data:
            raise ValueError(
                f"DataSource(type='{self.type}') has no inline bytes — "
                f"only base64 sources can be decoded. "
                f"{'Use the url to fetch the data.' if self.type == 'url' else ''}"
                f"{'Use the file_id to retrieve the file.' if self.type == 'file' else ''}"
            )
        import base64
        return base64.b64decode(self.data)


@dataclass(slots=True, frozen=True)
class Part:
    type: PartType
    text: str | None = None
    source: DataSource | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: tuple["Part", ...] = field(default_factory=tuple)
    is_error: bool | None = None
    redacted: bool | None = None
    summary: str | None = None
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type in {"text", "thinking", "refusal"} and self.text is None:
            raise ValueError(f"Part(type='{self.type}') requires text")
        if self.type in {"image", "audio", "video", "document"} and self.source is None:
            raise ValueError(f"Part(type='{self.type}') requires source")
        if self.type == "tool_call":
            if not self.id or not self.name or self.input is None:
                raise ValueError("Part(type='tool_call') requires id, name, input")
        if self.type == "tool_result":
            if not self.id:
                raise ValueError("Part(type='tool_result') requires id")

    @staticmethod
    def text_part(text: str) -> "Part":
        return Part(type="text", text=text)

    @staticmethod
    def thinking(text: str, *, redacted: bool | None = None, summary: str | None = None) -> "Part":
        return Part(type="thinking", text=text, redacted=redacted, summary=summary)

    @staticmethod
    def _cache_metadata(cache: bool | dict[str, Any] | None = None) -> dict[str, Any] | None:
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
        cache: bool | dict[str, Any] | None = None,
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
        return Part(type=kind, source=source, metadata=Part._cache_metadata(cache))

    @staticmethod
    def image(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | dict[str, Any] | None = None,
    ) -> "Part":
        return Part._media_part("image", url=url, data=data, file_id=file_id, media_type=media_type or "image/png", detail=detail, cache=cache)

    @staticmethod
    def audio(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | dict[str, Any] | None = None,
    ) -> "Part":
        return Part._media_part("audio", url=url, data=data, file_id=file_id, media_type=media_type or "audio/wav", detail=detail, cache=cache)

    @staticmethod
    def video(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | dict[str, Any] | None = None,
    ) -> "Part":
        return Part._media_part("video", url=url, data=data, file_id=file_id, media_type=media_type or "video/mp4", detail=detail, cache=cache)

    @staticmethod
    def document(
        *,
        url: str | None = None,
        data: bytes | str | None = None,
        file_id: str | None = None,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] | None = None,
        cache: bool | dict[str, Any] | None = None,
    ) -> "Part":
        return Part._media_part("document", url=url, data=data, file_id=file_id, media_type=media_type or "application/pdf", detail=detail, cache=cache)

    @staticmethod
    def tool_call(id: str, name: str, input: dict[str, Any]) -> "Part":
        return Part(type="tool_call", id=id, name=name, input=input)

    @staticmethod
    def tool_result(id: str, content: list["Part"], is_error: bool | None = None, name: str | None = None) -> "Part":
        return Part(type="tool_result", id=id, name=name, content=tuple(content), is_error=is_error)

    @property
    def bytes(self) -> bytes:
        """Decode media data and return raw bytes.

        Works on image, audio, video, and document parts with base64 sources.
        Raises TypeError for non-media parts. Raises ValueError if the source
        is not base64-encoded.
        """
        if self.type in ("text", "thinking", "refusal", "citation", "tool_call", "tool_result"):
            raise TypeError(
                f"Part(type='{self.type}') is not a media part — "
                f".bytes is only available on image, audio, video, and document parts"
            )
        if self.source is None:
            raise ValueError(f"Part(type='{self.type}') has no source")
        return self.source.bytes

    @staticmethod
    def refusal(text: str) -> "Part":
        return Part(type="refusal", text=text)

    @staticmethod
    def citation(text: str | None = None, url: str | None = None, title: str | None = None) -> "Part":
        return Part(type="citation", text=text, url=url, title=title)
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Part":
        source_value = value.get("source")
        source = DataSource(**source_value) if isinstance(source_value, dict) else source_value
        content = tuple(cls.from_dict(x) if isinstance(x, dict) else x for x in value.get("content", []))
        return cls(
            type=value["type"],
            text=value.get("text"),
            source=source,
            id=value.get("id"),
            name=value.get("name"),
            input=value.get("input"),
            content=content,
            is_error=value.get("is_error"),
            redacted=value.get("redacted"),
            summary=value.get("summary"),
            url=value.get("url"),
            title=value.get("title"),
            metadata=value.get("metadata"),
        )


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


@dataclass(slots=True, frozen=True)
class Tool:
    name: str
    type: Literal["function", "builtin"] = "function"
    description: str | None = None
    parameters: dict[str, Any] | None = None
    builtin_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type == "function" and self.parameters is None:
            object.__setattr__(self, "parameters", {"type": "object", "properties": {}})


@dataclass(slots=True, frozen=True)
class ToolConfig:
    mode: Literal["auto", "required", "none"] = "auto"
    allowed: tuple[str, ...] = ()
    parallel: bool | None = None


@dataclass(slots=True, frozen=True)
class Config:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None
    tool_config: ToolConfig | None = None
    reasoning: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")


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


@dataclass(slots=True, frozen=True)
class LMResponse:
    id: str
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    provider: dict[str, Any] | None = None

    @property
    def text(self) -> str | None:
        """Return concatenated text from all text parts, or None if no text parts."""
        texts = [p.text for p in self.message.parts if p.type == "text" and p.text is not None]
        return "\n".join(texts) if texts else None

    @property
    def image(self) -> Part | None:
        for p in self.message.parts:
            if p.type == "image":
                return p
        return None

    @property
    def images(self) -> list[Part]:
        return [p for p in self.message.parts if p.type == "image"]

    @property
    def audio(self) -> Part | None:
        for p in self.message.parts:
            if p.type == "audio":
                return p
        return None

    @property
    def tool_calls(self) -> list[Part]:
        return [p for p in self.message.parts if p.type == "tool_call"]

    @property
    def thinking(self) -> str | None:
        texts = [p.text for p in self.message.parts if p.type == "thinking" and p.text is not None]
        return "\n".join(texts) if texts else None

    @property
    def citations(self) -> list[Part]:
        return [p for p in self.message.parts if p.type == "citation"]

    @property
    def json(self) -> Any:
        """Parse the text response as JSON and return the result.

        Raises ValueError if there is no text or the text is not valid JSON.
        The error message includes the raw text for debugging.
        """
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
        """Return decoded bytes of the first image part.

        Raises ValueError if there is no image part or if the image is not
        base64-encoded (e.g. URL or file reference).
        """
        img = self.image
        if img is None:
            raise ValueError(
                "Response contains no image part. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        return img.bytes

    @property
    def audio_bytes(self) -> bytes:
        """Return decoded bytes of the first audio part.

        Raises ValueError if there is no audio part or if the audio is not
        base64-encoded.
        """
        aud = self.audio
        if aud is None:
            raise ValueError(
                "Response contains no audio part. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        return aud.bytes


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
    delta: PartDelta | dict[str, Any] | None = None
    part_type: str | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    error: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.type == "delta" and self.delta is None:
            raise ValueError("StreamEvent(type='delta') requires delta")
        if self.type == "error" and self.error is None:
            raise ValueError("StreamEvent(type='error') requires error")

    @property
    def delta_text(self) -> str | None:
        """Extract text from a delta event, or None."""
        if self.delta is None:
            return None
        if isinstance(self.delta, PartDelta):
            return self.delta.text if self.delta.type == "text" else None
        return self.delta.get("text") if self.delta.get("type") == "text" else None


@dataclass(slots=True, frozen=True)
class EmbeddingRequest:
    model: str
    inputs: tuple[str, ...]
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class EmbeddingResponse:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage = field(default_factory=Usage)
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class FileUploadRequest:
    model: str | None = None
    filename: str = "file.bin"
    bytes_data: bytes = b""
    media_type: str = "application/octet-stream"
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class FileUploadResponse:
    id: str
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class BatchRequest:
    model: str
    requests: tuple[LMRequest, ...]
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class BatchResponse:
    id: str
    status: str
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ImageGenerationRequest:
    model: str
    prompt: str
    size: str | None = None
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ImageGenerationResponse:
    images: tuple[DataSource, ...]
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class AudioGenerationRequest:
    model: str
    prompt: str
    voice: str | None = None
    format: str | None = None
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class AudioGenerationResponse:
    audio: DataSource
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class AudioFormat:
    encoding: Literal["pcm16", "opus", "mp3", "aac"]
    sample_rate: int
    channels: int = 1


@dataclass(slots=True, frozen=True)
class LiveConfig:
    model: str
    system: str | tuple[Part, ...] | None = None
    tools: tuple[Tool, ...] = ()
    voice: str | None = None
    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    provider: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class LiveClientEvent:
    type: Literal["audio", "video", "text", "tool_result", "interrupt", "end_audio"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    content: tuple[Part, ...] = ()


@dataclass(slots=True, frozen=True)
class LiveServerEvent:
    type: Literal["audio", "text", "tool_call", "interrupted", "turn_end", "error"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    usage: Usage | None = None
    error: dict[str, str] | None = None
