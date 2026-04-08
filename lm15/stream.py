from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterator

from .types import LMRequest, LMResponse, Message, Part, PartDelta, StreamEvent, Usage


@dataclass(slots=True, frozen=True)
class StreamChunk:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    image: Part | None = None
    audio: Part | None = None
    response: LMResponse | None = None


class Stream:
    def __init__(
        self,
        *,
        events: Iterator[StreamEvent],
        request: LMRequest,
        on_finished: Callable[[LMRequest, LMResponse], None] | None = None,
    ) -> None:
        self._events = events
        self._request = request
        self._on_finished = on_finished
        self._done = False
        self._response: LMResponse | None = None
        self._started_id: str | None = None
        self._started_model: str | None = None
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_input: dict[int, str] = {}

    def __iter__(self):
        return self

    def __next__(self) -> StreamChunk:
        if self._done:
            raise StopIteration

        for event in self._events:
            if event.type == "start":
                self._started_id = event.id
                self._started_model = event.model
                continue

            if event.type == "delta" and event.delta is not None:
                delta = event.delta
                if isinstance(delta, dict):
                    dtype = delta.get("type")
                    if dtype == "text":
                        text = delta.get("text", "")
                        self._text_parts.append(text)
                        return StreamChunk(type="text", text=text)
                    if dtype == "thinking":
                        text = delta.get("text", "")
                        self._thinking_parts.append(text)
                        return StreamChunk(type="thinking", text=text)
                    if dtype == "tool_call":
                        idx = event.part_index or 0
                        chunk = delta.get("input", "")
                        agg = self._tool_input.get(idx, "") + chunk
                        self._tool_input[idx] = agg
                        parsed = self._parse_json_best_effort(agg)
                        return StreamChunk(type="tool_call", input=parsed)
                    if dtype == "audio":
                        return StreamChunk(type="audio", audio=Part.audio(data=delta.get("data", "")))
                    continue

                if isinstance(delta, PartDelta):
                    if delta.type == "text":
                        text = delta.text or ""
                        self._text_parts.append(text)
                        return StreamChunk(type="text", text=text)
                    if delta.type == "thinking":
                        text = delta.text or ""
                        self._thinking_parts.append(text)
                        return StreamChunk(type="thinking", text=text)
                    if delta.type == "tool_call":
                        idx = event.part_index or 0
                        chunk = delta.input or ""
                        agg = self._tool_input.get(idx, "") + chunk
                        self._tool_input[idx] = agg
                        parsed = self._parse_json_best_effort(agg)
                        return StreamChunk(type="tool_call", input=parsed)
                    if delta.type == "audio":
                        return StreamChunk(type="audio", audio=Part.audio(data=delta.data or ""))
                    continue

            if event.type == "error":
                self._done = True
                raise RuntimeError((event.error or {}).get("message", "stream error"))

            if event.type == "end":
                self._done = True
                response = self._materialize_response(event)
                return StreamChunk(type="finished", response=response)

        self._done = True
        response = self._materialize_response(None)
        return StreamChunk(type="finished", response=response)

    @property
    def text(self):
        for chunk in self:
            if chunk.type == "text" and chunk.text is not None:
                yield chunk.text

    @property
    def response(self) -> LMResponse:
        if self._response is None:
            for _ in self:
                pass
        assert self._response is not None
        return self._response

    def _materialize_response(self, end_event: StreamEvent | None) -> LMResponse:
        if self._response is not None:
            return self._response

        parts: list[Part] = []
        if self._thinking_parts:
            parts.append(Part.thinking("".join(self._thinking_parts)))
        if self._text_parts:
            parts.append(Part.text_part("".join(self._text_parts)))
        if not parts:
            parts = [Part.text_part("")]

        usage = end_event.usage if end_event and end_event.usage is not None else Usage()
        finish = end_event.finish_reason if end_event and end_event.finish_reason is not None else "stop"

        response = LMResponse(
            id=self._started_id or "",
            model=self._started_model or self._request.model,
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason=finish,
            usage=usage,
        )
        self._response = response
        if self._on_finished is not None:
            self._on_finished(self._request, response)
        return response

    @staticmethod
    def _parse_json_best_effort(raw: str) -> dict:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {"value": value}
        except Exception:
            return {"partial_json": raw}
