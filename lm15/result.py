from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator

from .errors import RateLimitError, ServerError, TimeoutError, TransportError, error_class_for_canonical_code
from .types import AudioPart, CitationPart, DataSource, ErrorInfo, ImagePart, LMRequest, LMResponse, Message, Part, PartDelta, StreamEvent, ToolCallInfo, ToolCallPart, Usage


_RETRYABLE_ERRORS = (RateLimitError, TimeoutError, ServerError, TransportError)
_FINISH_SENTINEL = object()


@dataclass(slots=True, frozen=True)
class StreamChunk:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    image: ImagePart | None = None
    audio: AudioPart | None = None
    response: LMResponse | None = None


@dataclass(slots=True)
class _ExecutedTool:
    name: str | None
    part: Part
    preview: str | None = None


@dataclass(slots=True)
class _RoundState:
    request: LMRequest
    started_id: str | None = None
    started_model: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    text_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    audio_chunks: list[str] = field(default_factory=list)
    audio_parts: list[AudioPart] = field(default_factory=list)
    image_parts: list[ImagePart] = field(default_factory=list)
    citation_parts: list[CitationPart] = field(default_factory=list)
    tool_call_raw: dict[int, str] = field(default_factory=dict)
    tool_call_meta: dict[int, dict[str, Any]] = field(default_factory=dict)

    def apply(self, event: StreamEvent) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        if event.type == "start":
            self.started_id = event.id or self.started_id
            self.started_model = event.model or self.started_model
            return chunks

        if event.type == "end":
            self.finish_reason = event.finish_reason or self.finish_reason
            self.usage = event.usage or self.usage
            return chunks

        if event.type != "delta" or event.delta is None:
            return chunks

        delta = event.delta
        if isinstance(delta, PartDelta):
            if delta.type == "text":
                text = delta.text or ""
                self.text_parts.append(text)
                chunks.append(StreamChunk(type="text", text=text))
            elif delta.type == "thinking":
                text = delta.text or ""
                self.thinking_parts.append(text)
                chunks.append(StreamChunk(type="thinking", text=text))
            elif delta.type == "audio":
                data = delta.data or ""
                self.audio_chunks.append(data)
                chunks.append(StreamChunk(type="audio", audio=Part.audio(data=data)))
            elif delta.type == "tool_call":
                self._push_tool_call(event.part_index or 0, delta.input or "")
            return chunks

        dtype = delta.get("type")
        if dtype == "text":
            text = str(delta.get("text", ""))
            self.text_parts.append(text)
            chunks.append(StreamChunk(type="text", text=text))
        elif dtype == "thinking":
            text = str(delta.get("text", ""))
            self.thinking_parts.append(text)
            chunks.append(StreamChunk(type="thinking", text=text))
        elif dtype == "audio":
            part = _audio_part_from_delta(delta)
            if part.source and part.source.type == "base64" and part.source.data:
                self.audio_chunks.append(part.source.data)
            else:
                self.audio_parts.append(part)
            chunks.append(StreamChunk(type="audio", audio=part))
        elif dtype == "image":
            part = _image_part_from_delta(delta)
            self.image_parts.append(part)
            chunks.append(StreamChunk(type="image", image=part))
        elif dtype == "citation":
            self.citation_parts.append(
                CitationPart(
                    text=_coerce_optional_str(delta.get("text")),
                    url=_coerce_optional_str(delta.get("url")),
                    title=_coerce_optional_str(delta.get("title")),
                )
            )
        elif dtype == "tool_call":
            idx = event.part_index or 0
            meta = self.tool_call_meta.setdefault(idx, {})
            if delta.get("id") is not None:
                meta["id"] = str(delta.get("id"))
            if delta.get("name") is not None:
                meta["name"] = str(delta.get("name"))
            raw_input = delta.get("input", "")
            self._push_tool_call(idx, raw_input)
        return chunks

    def _push_tool_call(self, index: int, raw_input: Any) -> None:
        if isinstance(raw_input, dict):
            self.tool_call_meta.setdefault(index, {})["input"] = raw_input
            return
        chunk = "" if raw_input is None else str(raw_input)
        aggregate = self.tool_call_raw.get(index, "") + chunk
        self.tool_call_raw[index] = aggregate
        self.tool_call_meta.setdefault(index, {})["input"] = _parse_json_best_effort(aggregate)

    def materialize(self) -> LMResponse:
        parts: list[Part] = []
        if self.thinking_parts:
            parts.append(Part.thinking("".join(self.thinking_parts)))
        if self.text_parts:
            parts.append(Part.text_part("".join(self.text_parts)))
        parts.extend(self.image_parts)
        if self.audio_chunks and not self.audio_parts:
            pcm_data = _concat_b64_chunks_to_bytes(self.audio_chunks)
            wav_data = _pcm_to_wav(pcm_data)
            parts.append(Part.audio(data=wav_data, media_type="audio/wav"))
        parts.extend(self.audio_parts)
        parts.extend(self.citation_parts)

        tool_names = [t.name for t in self.request.tools if t.type == "function"]
        for pos, idx in enumerate(sorted(self.tool_call_meta)):
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
            parts.append(Part.tool_call(tc_id, str(tc_name), payload))

        if not parts:
            parts = [Part.text_part("")]

        finish = self.finish_reason
        if finish is None:
            finish = "tool_call" if any(p.type == "tool_call" for p in parts) else "stop"
        elif finish == "stop" and any(p.type == "tool_call" for p in parts):
            finish = "tool_call"

        return LMResponse(
            id=self.started_id or "",
            model=self.started_model or self.request.model,
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason=finish,
            usage=self.usage or Usage(),
        )


class Result:
    """Lazy stream-backed response."""

    def __init__(
        self,
        *,
        events: Iterator[StreamEvent] | None = None,
        request: LMRequest,
        start_stream: Callable[[LMRequest], Iterator[StreamEvent]] | None = None,
        on_finished: Callable[[LMRequest, LMResponse], None] | None = None,
        callable_registry: dict[str, Callable[..., Any]] | None = None,
        on_tool_call: Callable[[ToolCallInfo], Any] | None = None,
        max_tool_rounds: int = 8,
        retries: int = 0,
    ) -> None:
        if events is None and start_stream is None:
            raise ValueError("Result requires events or start_stream")
        self._initial_events = events
        self._request = request
        self._start_stream = start_stream
        self._on_finished = on_finished
        self._callable_registry = callable_registry or {}
        self._on_tool_call = on_tool_call
        self._max_tool_rounds = max_tool_rounds
        self._retries = max(int(retries), 0)
        self._response: LMResponse | None = None
        self._final_request: LMRequest = request
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

    @property
    def text(self) -> str | None:
        return self.response.text

    @property
    def thinking(self) -> str | None:
        return self.response.thinking

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.response.tool_calls

    @property
    def image(self) -> ImagePart | None:
        return self.response.image

    @property
    def images(self) -> list[ImagePart]:
        return self.response.images

    @property
    def audio(self) -> AudioPart | None:
        return self.response.audio

    @property
    def citations(self) -> list[CitationPart]:
        return self.response.citations

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
        return self.response.image_bytes

    @property
    def audio_bytes(self) -> bytes:
        return self.response.audio_bytes

    @property
    def cost(self) -> "CostBreakdown | None":
        """Estimated cost of this call, or ``None`` if cost tracking is not enabled.

        Enable with ``lm15.configure(track_costs=True)``.
        """
        from .cost import lookup_cost

        resp = self.response
        return lookup_cost(resp.model, resp.usage)

    @property
    def response(self) -> LMResponse:
        return self._consume()

    def _consume(self) -> LMResponse:
        if self._done and self._failure is not None:
            raise self._failure
        if self._response is not None and self._done:
            return self._response
        for _ in self.events():
            pass
        assert self._response is not None
        return self._response

    def _chunks(self) -> Iterator[StreamChunk]:
        current_request = self._request
        use_initial_events = True
        rounds = 0

        try:
            while True:
                state: _RoundState | None = None
                emitted_visible = False
                round_response: LMResponse | None = None
                attempt = 0

                while True:
                    state = _RoundState(request=current_request)
                    emitted_visible = False
                    round_response = None
                    try:
                        events = self._open_stream(current_request, use_initial_events=use_initial_events)
                        use_initial_events = False
                        for event in events:
                            if event.type == "error":
                                raise _exception_from_stream_error(event)
                            for chunk in state.apply(event):
                                if chunk.type in {"text", "thinking", "audio", "image"}:
                                    emitted_visible = True
                                yield chunk
                            if event.type == "end":
                                break
                        round_response = state.materialize()
                        break
                    except _RETRYABLE_ERRORS as exc:
                        if emitted_visible or attempt >= self._retries:
                            self._capture_partial(current_request, state)
                            self._failure = exc
                            raise
                        time.sleep(0.2 * (2**attempt))
                        attempt += 1
                        continue
                    except Exception as exc:
                        self._capture_partial(current_request, state)
                        self._failure = exc
                        raise

                assert round_response is not None
                self._final_request = current_request
                self._response = round_response

                pending_tool_calls = round_response.tool_calls
                for tc in pending_tool_calls:
                    yield StreamChunk(type="tool_call", name=tc.name, input=tc.input)

                if (
                    round_response.finish_reason == "tool_call"
                    and pending_tool_calls
                    and rounds < self._max_tool_rounds
                    and self._start_stream is not None
                ):
                    executed: list[_ExecutedTool] = []
                    unanswered = False
                    for tc in pending_tool_calls:
                        outcome = self._execute_tool_call(tc)
                        if outcome is None:
                            unanswered = True
                            break
                        executed.append(outcome)

                    if executed and not unanswered:
                        for outcome in executed:
                            yield StreamChunk(type="tool_result", text=outcome.preview, name=outcome.name)
                        tool_message = Message(role="tool", parts=tuple(item.part for item in executed))
                        current_request = LMRequest(
                            model=current_request.model,
                            messages=current_request.messages + (round_response.message, tool_message),
                            system=current_request.system,
                            tools=current_request.tools,
                            config=current_request.config,
                        )
                        rounds += 1
                        continue

                self._finalize(self._final_request, round_response)
                yield StreamChunk(type="finished", response=round_response)
                return
        finally:
            self._done = True

    def _open_stream(self, request: LMRequest, *, use_initial_events: bool) -> Iterator[StreamEvent]:
        if use_initial_events and self._initial_events is not None:
            events = self._initial_events
            self._initial_events = None
            return events
        if self._start_stream is None:
            raise ValueError("cannot start follow-up stream: no stream factory available")
        return self._start_stream(request)

    def _capture_partial(self, request: LMRequest, state: _RoundState | None) -> None:
        self._final_request = request
        if state is not None:
            try:
                self._response = state.materialize()
            except Exception:
                pass

    def _finalize(self, request: LMRequest, response: LMResponse) -> None:
        self._final_request = request
        self._response = response
        self._failure = None
        if not self._callback_called and self._on_finished is not None:
            self._callback_called = True
            self._on_finished(request, response)

    def _execute_tool_call(self, tool_call: ToolCallPart) -> _ExecutedTool | None:
        info = ToolCallInfo(
            id=tool_call.id or "",
            name=tool_call.name or "tool",
            input=tool_call.input or {},
        )

        if self._on_tool_call is not None:
            override = self._on_tool_call(info)
            if override is not None:
                content = _normalize_tool_output(override)
                return _ExecutedTool(
                    name=info.name,
                    part=Part.tool_result(info.id, content, name=info.name),
                    preview=_preview_parts(content),
                )

        fn = self._callable_registry.get(info.name)
        if fn is None:
            return None

        output = _invoke_tool(fn, info.input)
        content = _normalize_tool_output(output)
        return _ExecutedTool(
            name=info.name,
            part=Part.tool_result(info.id, content, name=info.name),
            preview=_preview_parts(content),
        )


class AsyncResult:
    """Async wrapper over a thread-backed Result."""

    def __init__(self, sync_fn: Callable[..., Result], *args, **kwargs) -> None:
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


def response_to_events(response: LMResponse, request: LMRequest | None = None) -> Iterator[StreamEvent]:
    yield StreamEvent(type="start", id=response.id, model=response.model)
    for idx, part in enumerate(response.message.parts):
        if part.type in {"text", "refusal"} and part.text is not None:
            yield StreamEvent(type="delta", part_index=idx, delta=PartDelta(type="text", text=part.text))
        elif part.type == "thinking" and part.text is not None:
            yield StreamEvent(type="delta", part_index=idx, delta=PartDelta(type="thinking", text=part.text))
        elif part.type == "tool_call":
            payload = json.dumps(part.input or {})
            yield StreamEvent(
                type="delta",
                part_index=idx,
                delta={"type": "tool_call", "id": part.id, "name": part.name, "input": payload},
            )
        elif part.type == "image" and part.source is not None:
            yield StreamEvent(
                type="delta",
                part_index=idx,
                delta={"type": "image", "source": _source_to_dict(part.source)},
            )
        elif part.type == "audio" and part.source is not None:
            if part.source.type == "base64" and part.source.data:
                yield StreamEvent(type="delta", part_index=idx, delta=PartDelta(type="audio", data=part.source.data))
            else:
                yield StreamEvent(
                    type="delta",
                    part_index=idx,
                    delta={"type": "audio", "source": _source_to_dict(part.source)},
                )
        elif part.type == "citation":
            yield StreamEvent(
                type="delta",
                part_index=idx,
                delta={"type": "citation", "text": part.text, "url": part.url, "title": part.title},
            )
    yield StreamEvent(type="end", finish_reason=response.finish_reason, usage=response.usage)


def materialize_response(events: Iterator[StreamEvent], request: LMRequest) -> LMResponse:
    return Result(events=events, request=request).response


def _coerce_optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _exception_from_stream_error(event: StreamEvent) -> Exception:
    err = event.error or ErrorInfo(code="provider", message="stream error")
    code = str(err.get("code") or "provider")
    message = str(err.get("message") or "stream error")
    provider_code = str(err.get("provider_code") or "")
    if provider_code and provider_code not in message:
        message = f"{message} (provider_code={provider_code})"
    exc_cls = error_class_for_canonical_code(code)
    return exc_cls(message)


def _concat_b64_chunks_to_bytes(chunks: list[str]) -> bytes:
    """Decode each base64 chunk and concatenate raw bytes.

    Individual SSE chunks are independently base64-encoded with their own
    padding.  Concatenating the b64 strings directly produces invalid base64
    (the ``=`` padding from early chunks corrupts the decode).  This function
    decodes each chunk to raw bytes and concatenates the result.
    """
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
    """Wrap raw PCM bytes in a WAV header so the result is playable.

    Gemini Live returns raw PCM audio (``audio/L16;rate=24000``).  Without a
    WAV header, media players can't open the file.
    """
    import struct
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = len(pcm)
    # RIFF header + fmt chunk + data chunk
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,     # file size - 8
        b"WAVE",
        b"fmt ",
        16,                 # fmt chunk size
        1,                  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
    return header + pcm


def _parse_json_best_effort(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"partial_json": raw}


def _normalize_tool_output(value: Any) -> list[Part]:
    if isinstance(value, Part):
        return [value]
    if isinstance(value, list) and all(isinstance(x, Part) for x in value):
        return list(value)
    return [Part.text_part(str(value))]


def _preview_parts(parts: list[Part]) -> str | None:
    text = "\n".join(p.text or "" for p in parts if p.type in {"text", "thinking", "refusal"} and p.text)
    return text or None


def _invoke_tool(fn: Callable[..., Any], payload: dict[str, Any]) -> Any:
    try:
        return fn(**payload)
    except TypeError:
        return fn(payload)


def _source_to_dict(source: DataSource) -> dict[str, Any]:
    return {
        "type": source.type,
        "media_type": source.media_type,
        "data": source.data,
        "url": source.url,
        "file_id": source.file_id,
        "detail": source.detail,
    }


def _image_part_from_delta(delta: dict[str, Any]) -> ImagePart:
    source_payload = delta.get("source")
    if isinstance(source_payload, dict):
        return ImagePart(source=DataSource(**source_payload))
    if delta.get("data") is not None:
        return Part.image(data=str(delta.get("data")), media_type=_coerce_optional_str(delta.get("media_type")) or "image/png")
    if delta.get("url") is not None:
        return Part.image(url=str(delta.get("url")), media_type=_coerce_optional_str(delta.get("media_type")) or "image/png")
    if delta.get("file_id") is not None:
        return Part.image(file_id=str(delta.get("file_id")), media_type=_coerce_optional_str(delta.get("media_type")) or "image/png")
    raise ValueError("image delta requires source, data, url, or file_id")


def _audio_part_from_delta(delta: dict[str, Any]) -> AudioPart:
    source_payload = delta.get("source")
    if isinstance(source_payload, dict):
        return AudioPart(source=DataSource(**source_payload))
    if delta.get("data") is not None:
        return Part.audio(data=str(delta.get("data")), media_type=_coerce_optional_str(delta.get("media_type")) or "audio/wav")
    if delta.get("url") is not None:
        return Part.audio(url=str(delta.get("url")), media_type=_coerce_optional_str(delta.get("media_type")) or "audio/wav")
    if delta.get("file_id") is not None:
        return Part.audio(file_id=str(delta.get("file_id")), media_type=_coerce_optional_str(delta.get("media_type")) or "audio/wav")
    raise ValueError("audio delta requires source, data, url, or file_id")
