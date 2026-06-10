"""
lm15.result — Stream materialization.

Result is the lazy stream-backed response wrapper.  It unifies
streaming and non-streaming behind one interface:

    # Stream text
    for text in result:
        print(text)

    # Block for full response
    print(result.text)

    # Stream typed chunks
    for chunk in result.events():
        ...

_RoundState accumulates stream deltas into a complete Response.
Result executes nothing: tool calls are surfaced as data, and any
execute-tools-until-done loop belongs to the layer above lm15.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Iterator

from .errors import LM15Error, error_class_for_code
from .types import (
    AudioDelta,
    AudioPart,
    CitationDelta,
    CitationPart,
    ContinuationDelta,
    ContinuationState,
    DocumentPart,
    ErrorDetail,
    ImageDelta,
    ImagePart,
    JsonObject,
    Message,
    Part,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    Usage,
    VideoPart,
)

_FINISH_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A user-facing chunk from the stream."""
    type: str
    text: str | None = None
    name: str | None = None
    input: JsonObject | None = None
    image: ImagePart | None = None
    audio: AudioPart | None = None
    response: Response | None = None


@dataclass(slots=True)
class _RoundState:
    """Accumulates stream deltas into a complete Response."""
    request: Request
    started_id: str | None = None
    started_model: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    text_parts: dict[int, list[str]] = field(default_factory=dict)
    thinking_parts: dict[int, list[str]] = field(default_factory=dict)
    audio_chunks: dict[int, list[str]] = field(default_factory=dict)
    audio_media_types: dict[int, str | None] = field(default_factory=dict)
    image_parts: dict[int, ImagePart] = field(default_factory=dict)
    citation_parts: dict[int, list[CitationPart]] = field(default_factory=dict)
    tool_call_raw: dict[int, str] = field(default_factory=dict)
    tool_call_meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    message_continuation: list[ContinuationState] = field(default_factory=list)
    part_continuation: dict[int, list[ContinuationState]] = field(default_factory=dict)
    provider_data: dict[str, Any] | None = None

    def apply(self, event: StreamEvent) -> list[StreamChunk]:
        """Process a StreamEvent and return user-facing chunks."""
        chunks: list[StreamChunk] = []

        if event.type == "start":
            self.started_id = event.id or self.started_id
            self.started_model = event.model or self.started_model
            return chunks

        if event.type == "end":
            self.finish_reason = event.finish_reason or self.finish_reason
            self.usage = event.usage or self.usage
            if event.provider_data is not None:
                self.provider_data = event.provider_data
            return chunks

        if event.type != "delta" or event.delta is None:
            return chunks

        delta = event.delta

        if delta.type == "text":
            t = delta.text or ""
            self.text_parts.setdefault(delta.part_index, []).append(t)
            chunks.append(StreamChunk(type="text", text=t))

        elif delta.type == "thinking":
            t = delta.text or ""
            self.thinking_parts.setdefault(delta.part_index, []).append(t)
            chunks.append(StreamChunk(type="thinking", text=t))

        elif delta.type == "audio":
            data = delta.data or ""
            self.audio_chunks.setdefault(delta.part_index, []).append(data)
            self.audio_media_types.setdefault(delta.part_index, delta.media_type)
            from .types import audio as make_audio
            chunks.append(StreamChunk(type="audio", audio=make_audio(data=data)))

        elif delta.type == "tool_call":
            idx = delta.part_index
            meta = self.tool_call_meta.setdefault(idx, {})
            if delta.id is not None:
                meta["id"] = str(delta.id)
            if delta.name is not None:
                meta["name"] = str(delta.name)
            aggregate = self.tool_call_raw.get(idx, "") + delta.input
            self.tool_call_raw[idx] = aggregate
            meta["input"] = _parse_json_best_effort(aggregate)

        elif delta.type == "image":
            mt = delta.media_type or "image/png"
            if delta.data is not None:
                part = ImagePart(media_type=mt, data=str(delta.data))
            elif delta.url is not None:
                part = ImagePart(media_type=mt, url=str(delta.url))
            elif delta.file_id is not None:
                part = ImagePart(media_type=mt, file_id=str(delta.file_id))
            else:
                return chunks
            self.image_parts[delta.part_index] = part
            chunks.append(StreamChunk(type="image", image=part))

        elif delta.type == "citation":
            self.citation_parts.setdefault(delta.part_index, []).append(CitationPart(
                text=delta.text, url=delta.url, title=delta.title,
            ))

        elif delta.type == "continuation":
            state = delta.to_state()
            if delta.part_index is None:
                self.message_continuation.append(state)
            else:
                self.part_continuation.setdefault(delta.part_index, []).append(state)

        return chunks

    def materialize(self) -> Response:
        """Build a complete Response from accumulated state."""
        parts: list[Part] = []

        tool_names = [t.name for t in self.request.tools if t.type == "function"]
        part_indexes = sorted(
            set(self.thinking_parts)
            | set(self.text_parts)
            | set(self.image_parts)
            | set(self.audio_chunks)
            | set(self.citation_parts)
            | set(self.tool_call_meta)
            | set(self.part_continuation)
        )

        for pos, idx in enumerate(part_indexes):
            continuation = tuple(self.part_continuation.get(idx, ()))
            if idx in self.thinking_parts:
                parts.append(ThinkingPart(text="".join(self.thinking_parts[idx]), continuation=continuation))
            if idx in self.text_parts:
                parts.append(TextPart(text="".join(self.text_parts[idx]), continuation=continuation))
            if idx in self.image_parts:
                parts.append(replace(self.image_parts[idx], continuation=continuation))
            if idx in self.audio_chunks:
                raw_data = _concat_b64_chunks(self.audio_chunks[idx])
                media_type = self.audio_media_types.get(idx)
                from .types import audio as make_audio
                if media_type in (None, "audio/pcm", "audio/pcm16"):
                    parts.append(make_audio(data=_pcm_to_wav(raw_data), media_type="audio/wav", continuation=continuation))
                else:
                    parts.append(make_audio(data=raw_data, media_type=media_type, continuation=continuation))
            if idx in self.citation_parts:
                parts.extend(replace(part, continuation=continuation) for part in self.citation_parts[idx])
            if idx in self.tool_call_meta:
                meta = self.tool_call_meta[idx]
                payload = meta.get("input")
                if not isinstance(payload, dict):
                    payload = _parse_json_best_effort(self.tool_call_raw.get(idx, ""))
                tc_name = meta.get("name")
                if not tc_name:
                    if len(tool_names) == 1:
                        tc_name = tool_names[0]
                    elif pos < len(tool_names):
                        tc_name = tool_names[pos]
                    else:
                        tc_name = "tool"
                tc_id = str(meta.get("id") or f"tool_call_{idx}")
                parts.append(ToolCallPart(id=tc_id, name=str(tc_name), input=payload, continuation=continuation))
            elif (
                idx not in self.thinking_parts
                and idx not in self.text_parts
                and idx not in self.image_parts
                and idx not in self.audio_chunks
                and idx not in self.citation_parts
            ):
                parts.append(TextPart(text="", continuation=continuation))

        if not parts:
            parts = [TextPart(text="")]

        finish = self.finish_reason
        has_tool_calls = any(isinstance(p, ToolCallPart) for p in parts)
        if finish is None:
            finish = "tool_call" if has_tool_calls else "stop"
        elif finish == "stop" and has_tool_calls:
            finish = "tool_call"

        return Response(
            id=self.started_id,
            model=self.started_model or self.request.model,
            message=Message(role="assistant", parts=tuple(parts), continuation=tuple(self.message_continuation)),
            finish_reason=finish,
            usage=self.usage or Usage(),
            provider_data=self.provider_data,
        )


class Result:
    """Lazy stream-backed response: a pure stream materializer.

    Consumes stream events and exposes chunks, text and the complete
    Response.  Tool calls are surfaced as data only; Result never
    executes tools.
    """

    def __init__(
        self,
        *,
        events: Iterator[StreamEvent] | None = None,
        request: Request,
        start_stream: Callable[[Request], Iterator[StreamEvent]] | None = None,
        on_finished: Callable[[Request, Response], None] | None = None,
    ) -> None:
        if events is None and start_stream is None:
            raise ValueError("Result requires events or start_stream")
        self._initial_events = events
        self._request = request
        self._start_stream = start_stream
        self._on_finished = on_finished
        self._response: Response | None = None
        self._final_request: Request = request
        self._done = False
        self._failure: Exception | None = None
        self._callback_called = False
        self._chunk_iter = self._chunks()

    def __iter__(self) -> Iterator[str]:
        for chunk in self.events():
            if chunk.type == "text" and chunk.text is not None:
                yield chunk.text

    def events(self) -> Iterator[StreamChunk]:
        while True:
            try:
                yield next(self._chunk_iter)
            except StopIteration:
                return

    def _require_part(self, cls: type[Any], label: str) -> Any:
        part = self.response.message.first(cls)
        if part is None:
            raise ValueError(
                f"Response contains no {label}. "
                f"Parts: {[p.type for p in self.response.message.parts]}"
            )
        return part

    @property
    def text(self) -> str | None:
        return self.response.text

    @property
    def thinking(self) -> str | None:
        texts = [p.text for p in self.response.message.parts_of(ThinkingPart)]
        return "\n".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.response.tool_calls

    @property
    def image(self) -> ImagePart | None:
        return self.response.message.first(ImagePart)

    @property
    def images(self) -> list[ImagePart]:
        return self.response.message.parts_of(ImagePart)

    @property
    def audio(self) -> AudioPart | None:
        return self.response.message.first(AudioPart)

    @property
    def video(self) -> VideoPart | None:
        return self.response.message.first(VideoPart)

    @property
    def videos(self) -> list[VideoPart]:
        return self.response.message.parts_of(VideoPart)

    @property
    def document(self) -> DocumentPart | None:
        return self.response.message.first(DocumentPart)

    @property
    def documents(self) -> list[DocumentPart]:
        return self.response.message.parts_of(DocumentPart)

    @property
    def citations(self) -> list[CitationPart]:
        return self.response.message.parts_of(CitationPart)

    @property
    def usage(self) -> Usage:
        return self.response.usage

    @property
    def finish_reason(self) -> str:
        return self.response.finish_reason

    @property
    def model(self) -> str:
        return self.response.model

    @property
    def json(self) -> Any:
        return self.response.json

    @property
    def image_bytes(self) -> bytes:
        return self._require_part(ImagePart, "image").bytes

    @property
    def audio_bytes(self) -> bytes:
        return self._require_part(AudioPart, "audio").bytes

    @property
    def video_bytes(self) -> bytes:
        return self._require_part(VideoPart, "video").bytes

    @property
    def document_bytes(self) -> bytes:
        return self._require_part(DocumentPart, "document").bytes

    @property
    def response(self) -> Response:
        return self._consume()

    def _consume(self) -> Response:
        if self._done and self._failure is not None:
            raise self._failure
        if self._response is not None and self._done:
            return self._response
        for _ in self.events():
            pass
        assert self._response is not None
        return self._response

    def _chunks(self) -> Iterator[StreamChunk]:
        request = self._request
        state = _RoundState(request=request)

        try:
            try:
                events = self._open_stream(request)
                for event in events:
                    if event.type == "error":
                        raise _exception_from_error(event)
                    for chunk in state.apply(event):
                        yield chunk
                    if event.type == "end":
                        break
                response = state.materialize()
            except Exception as exc:
                self._capture_partial(request, state)
                self._failure = exc
                raise

            self._final_request = request
            self._response = response

            for tc in response.tool_calls:
                yield StreamChunk(type="tool_call", name=tc.name, input=tc.input)

            self._finalize(request, response)
            yield StreamChunk(type="finished", response=response)
        finally:
            self._done = True

    def _open_stream(self, request: Request) -> Iterator[StreamEvent]:
        if self._initial_events is not None:
            events = self._initial_events
            self._initial_events = None
            return events
        assert self._start_stream is not None  # enforced in __init__
        return self._start_stream(request)

    def _capture_partial(self, request: Request, state: _RoundState | None) -> None:
        self._final_request = request
        if state is not None:
            try:
                self._response = state.materialize()
            except Exception:
                pass

    def _finalize(self, request: Request, response: Response) -> None:
        self._final_request = request
        self._response = response
        self._failure = None
        if not self._callback_called and self._on_finished is not None:
            self._callback_called = True
            self._on_finished(request, response)


class AsyncResult:
    """Async wrapper over a thread-backed Result."""

    def __init__(self, sync_fn: Callable[..., Result], *args: Any, **kwargs: Any) -> None:
        self._sync_fn = sync_fn
        self._args = args
        self._kwargs = kwargs
        self._result: Result | None = None
        self._await_started = False
        self._stream_started = False

    def __await__(self):
        self._await_started = True

        async def _consume() -> Result:
            def _run() -> Result:
                if self._result is None:
                    self._result = self._sync_fn(*self._args, **self._kwargs)
                self._result._consume()
                return self._result
            return await asyncio.to_thread(_run)

        return _consume().__await__()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._aiter(lambda result: iter(result))

    def events(self) -> AsyncIterator[StreamChunk]:
        return self._aiter(lambda result: result.events())

    def _aiter(self, iterator_factory: Callable[[Result], Iterator[Any]]) -> AsyncIterator[Any]:
        if self._await_started and self._result is None:
            raise RuntimeError("AsyncResult is already being awaited")
        self._stream_started = True

        async def _gen() -> AsyncIterator[Any]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Any] = asyncio.Queue()

            def _push(item: Any) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, item)

            def _produce() -> None:
                try:
                    if self._result is None:
                        self._result = self._sync_fn(*self._args, **self._kwargs)
                    for item in iterator_factory(self._result):
                        _push(item)
                    _push(_FINISH_SENTINEL)
                except Exception as exc:
                    _push(exc)
                    _push(_FINISH_SENTINEL)

            loop.run_in_executor(None, _produce)

            while True:
                item = await queue.get()
                if item is _FINISH_SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        return _gen()


# ─── Conversion utilities ────────────────────────────────────────────

def response_to_events(response: Response) -> Iterator[StreamEvent]:
    """Convert a complete Response to stream events.

    The conversion is intentionally lossless for the Delta vocabulary.  If a
    response contains a valid Part that has no Delta representation, this
    function raises instead of silently dropping content.
    """
    yield StreamStartEvent(id=response.id, model=response.model)
    for idx, part in enumerate(response.message.parts):
        if isinstance(part, TextPart):
            yield StreamDeltaEvent(delta=TextDelta(text=part.text, part_index=idx))
        elif isinstance(part, ThinkingPart):
            yield StreamDeltaEvent(delta=ThinkingDelta(text=part.text, part_index=idx))
        elif isinstance(part, ToolCallPart):
            yield StreamDeltaEvent(
                delta=ToolCallDelta(
                    input=json.dumps(part.input),
                    part_index=idx,
                    id=part.id,
                    name=part.name,
                )
            )
        elif isinstance(part, ImagePart):
            yield StreamDeltaEvent(
                delta=ImageDelta(
                    part_index=idx,
                    data=part.data,
                    url=part.url,
                    file_id=part.file_id,
                    media_type=part.media_type,
                )
            )
        elif isinstance(part, AudioPart):
            if part.data is None:
                _raise_non_streamable_part(
                    part,
                    reason="AudioDelta only supports inline data",
                )
            yield StreamDeltaEvent(
                delta=AudioDelta(
                    data=part.data,
                    part_index=idx,
                    media_type=part.media_type,
                )
            )
        elif isinstance(part, CitationPart):
            yield StreamDeltaEvent(
                delta=CitationDelta(
                    text=part.text,
                    url=part.url,
                    title=part.title,
                    part_index=idx,
                )
            )
        else:
            _raise_non_streamable_part(part)
        for state in part.continuation:
            yield StreamDeltaEvent(
                delta=ContinuationDelta(
                    provider=state.provider,
                    kind=state.kind,
                    data=state.data,
                    part_index=idx,
                )
            )
    for state in response.message.continuation:
        yield StreamDeltaEvent(
            delta=ContinuationDelta(
                provider=state.provider,
                kind=state.kind,
                data=state.data,
                part_index=None,
            )
        )
    yield StreamEndEvent(
        finish_reason=response.finish_reason,
        usage=response.usage,
        provider_data=response.provider_data,
    )


def coalesce_stream(events: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
    """Enforce MAP-3: a stream yields exactly one StreamEndEvent, as the final event.

    Adapters are stateless and may emit one end event per provider terminal
    frame (finish_reason chunk, usage-only chunk, ``[DONE]``,
    ``message_delta`` + ``message_stop``).  This wrapper passes start, delta
    and error events through unchanged, absorbs every end event's fields —
    a later non-None field replaces the accumulated value, a None field never
    erases one — and emits the single merged end event once the underlying
    iterator is exhausted.  If no end event was seen (e.g. the stream errored
    or was truncated), no end event is fabricated.

    See docs/mapping-rules.md MAP-3.
    """
    saw_end = False
    finish_reason = None
    usage: Usage | None = None
    provider_data = None
    for event in events:
        if event.type == "end":
            saw_end = True
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
            if event.usage is not None:
                usage = event.usage
            if event.provider_data is not None:
                provider_data = event.provider_data
            continue
        yield event
    if saw_end:
        yield StreamEndEvent(
            finish_reason=finish_reason,
            usage=usage,
            provider_data=provider_data,
        )


async def acoalesce_stream(events: "AsyncIterator[StreamEvent]") -> "AsyncIterator[StreamEvent]":
    """Async mirror of :func:`coalesce_stream` — same MAP-3 merge semantics.

    Passes start/delta/error events through unchanged, absorbs every end
    event's fields (later non-None replaces, None never erases), and emits
    exactly one merged final StreamEndEvent once the source is exhausted.
    No end event is fabricated if none was seen.
    """
    saw_end = False
    finish_reason = None
    usage: Usage | None = None
    provider_data = None
    async for event in events:
        if event.type == "end":
            saw_end = True
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
            if event.usage is not None:
                usage = event.usage
            if event.provider_data is not None:
                provider_data = event.provider_data
            continue
        yield event
    if saw_end:
        yield StreamEndEvent(
            finish_reason=finish_reason,
            usage=usage,
            provider_data=provider_data,
        )


def materialize_response(events: Iterator[StreamEvent], request: Request) -> Response:
    """Consume stream events and build a complete Response."""
    return Result(events=events, request=request).response


def _raise_non_streamable_part(part: Part, *, reason: str | None = None) -> None:
    detail = reason or f"no {part.type!r} Delta variant exists"
    raise TypeError(f"Cannot convert {type(part).__name__} to StreamEvent: {detail}")


# ─── Internal helpers ────────────────────────────────────────────────

def _exception_from_error(event: StreamEvent) -> Exception:
    err = event.error or ErrorDetail(code="provider", message="stream error")
    code = err.code
    message = err.message
    exc_cls = error_class_for_code(code)
    if issubclass(exc_cls, LM15Error):
        return exc_cls(message, provider_code=err.provider_code)
    return exc_cls(message)


def _concat_b64_chunks(chunks: list[str]) -> bytes:
    """Decode each base64 chunk and concatenate raw bytes."""
    import base64
    raw = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        try:
            raw.extend(base64.b64decode(chunk))
        except Exception:
            padded = chunk + "=" * (-len(chunk) % 4)
            try:
                raw.extend(base64.b64decode(padded))
            except Exception:
                pass
    return bytes(raw)


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    import struct
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1,
        channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm


def _parse_json_best_effort(raw: str | None) -> JsonObject:
    if not raw:
        return {}
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"partial_json": raw}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")
