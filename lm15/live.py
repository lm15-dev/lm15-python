"""
lm15.live — WebSocket live session wrapper.

Provider-agnostic session around a WebSocket connection for realtime
(live) interactions with foundation models.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections import deque
from collections.abc import Sequence
from typing import Any, Callable, Deque

from .types import (
    LiveClientAudioEvent,
    LiveClientEndAudioEvent,
    LiveClientEvent,
    LiveClientImageEvent,
    LiveClientInterruptEvent,
    LiveClientTextEvent,
    LiveClientToolResultEvent,
    LiveClientTurnEvent,
    LiveServerEvent,
    PART_CLASSES,
    Part,
    PartInput,
    TextPart,
    _normalize_parts,
)

EncodeEventFn = Callable[[LiveClientEvent], list[dict[str, Any]]]
DecodeEventFn = Callable[[str | bytes], list[LiveServerEvent]]


def require_websocket_sync_connect():
    """Return `websockets.sync.client.connect` or raise a helpful ImportError."""
    try:
        from websockets.sync.client import connect  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Live sessions require the optional 'websockets' dependency.\n\n"
            "  Install it with:\n"
            "    pip install lm15[live]\n"
        ) from exc
    return connect


class WebSocketLiveSession:
    """Provider-agnostic session wrapper around a WebSocket connection."""

    def __init__(
        self,
        *,
        ws: Any,
        encode_event: EncodeEventFn,
        decode_event: DecodeEventFn,
    ) -> None:
        self._ws = ws
        self._encode_event = encode_event
        self._decode_event = decode_event
        self._pending: Deque[LiveServerEvent] = deque()
        self._send_lock = threading.Lock()
        self._closed = False

    def send(
        self,
        event: LiveClientEvent | None = None,
        *,
        audio: bytes | str | None = None,
        audio_media_type: str = "audio/pcm;rate=16000",
        image: bytes | str | None = None,
        image_media_type: str = "image/jpeg",
        text: str | None = None,
        turn: PartInput | None = None,
        tool_result: dict[str, Any] | None = None,
        interrupt: bool = False,
        end_audio: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("live session is closed")

        if event is not None:
            has_payload = any(x is not None for x in (audio, image, text, turn, tool_result))
            if has_payload or interrupt or end_audio:
                raise ValueError("pass either `event` or keyword payload, not both")
            events = [event]
        else:
            events = self._events_from_kwargs(
                audio=audio,
                audio_media_type=audio_media_type,
                image=image,
                image_media_type=image_media_type,
                text=text,
                turn=turn,
                tool_result=tool_result,
                interrupt=interrupt,
                end_audio=end_audio,
            )

        with self._send_lock:
            for evt in events:
                payloads = self._encode_event(evt)
                for payload in payloads:
                    self._ws.send(json.dumps(payload))

    def send_turn(self, content: PartInput, *, turn_complete: bool = True) -> None:
        self.send(LiveClientTurnEvent(parts=_normalize_parts(content), turn_complete=turn_complete))

    def send_audio(self, data: bytes | str, *, media_type: str = "audio/pcm;rate=16000") -> None:
        self.send(LiveClientAudioEvent(data=_to_base64_str(data), media_type=media_type))

    def send_image(self, data: bytes | str, *, media_type: str = "image/jpeg") -> None:
        self.send(LiveClientImageEvent(data=_to_base64_str(data), media_type=media_type))

    def send_text(self, text: str) -> None:
        self.send(LiveClientTextEvent(text=text))

    def send_tool_result(self, results: dict[str, Any]) -> None:
        self.send(tool_result=results)

    def interrupt(self) -> None:
        self.send(interrupt=True)

    def end_audio(self) -> None:
        self.send(end_audio=True)

    def recv(self) -> LiveServerEvent:
        if self._closed:
            raise RuntimeError("live session is closed")

        while True:
            if self._pending:
                return self._pending.popleft()

            raw = self._ws.recv()
            decoded = self._decode_event(raw)
            if not decoded:
                continue

            for event in decoded:
                self._pending.append(event)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            return

    def __iter__(self):
        return self

    def __next__(self) -> LiveServerEvent:
        if self._closed:
            raise StopIteration
        try:
            return self.recv()
        except RuntimeError:
            raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _events_from_kwargs(
        self,
        *,
        audio: bytes | str | None,
        audio_media_type: str,
        image: bytes | str | None,
        image_media_type: str,
        text: str | None,
        turn: PartInput | None,
        tool_result: dict[str, Any] | None,
        interrupt: bool,
        end_audio: bool,
    ) -> list[LiveClientEvent]:
        events: list[LiveClientEvent] = []

        if audio is not None:
            events.append(LiveClientAudioEvent(data=_to_base64_str(audio), media_type=audio_media_type))
        if image is not None:
            events.append(LiveClientImageEvent(data=_to_base64_str(image), media_type=image_media_type))
        if turn is not None:
            events.append(LiveClientTurnEvent(parts=_normalize_parts(turn)))
        if text is not None:
            events.append(LiveClientTextEvent(text=text))

        if tool_result:
            for call_id, value in tool_result.items():
                content = tuple(_tool_result_parts(value))
                events.append(LiveClientToolResultEvent(id=call_id, content=content))

        if interrupt:
            events.append(LiveClientInterruptEvent())
        if end_audio:
            events.append(LiveClientEndAudioEvent())

        if not events:
            raise ValueError("nothing to send")
        return events

class AsyncLiveSession:
    """Async wrapper over a sync live session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def send(self, event: LiveClientEvent | None = None, **kwargs: Any) -> None:
        await asyncio.to_thread(self._session.send, event, **kwargs)

    async def send_turn(self, content: PartInput, *, turn_complete: bool = True) -> None:
        await asyncio.to_thread(self._session.send_turn, content, turn_complete=turn_complete)

    async def send_audio(self, data: bytes | str, *, media_type: str = "audio/pcm;rate=16000") -> None:
        await asyncio.to_thread(self._session.send_audio, data, media_type=media_type)

    async def send_image(self, data: bytes | str, *, media_type: str = "image/jpeg") -> None:
        await asyncio.to_thread(self._session.send_image, data, media_type=media_type)

    async def send_text(self, text: str) -> None:
        await asyncio.to_thread(self._session.send_text, text)

    async def send_tool_result(self, results: dict[str, Any]) -> None:
        await asyncio.to_thread(self._session.send_tool_result, results)

    async def interrupt(self) -> None:
        await asyncio.to_thread(self._session.interrupt)

    async def end_audio(self) -> None:
        await asyncio.to_thread(self._session.end_audio)

    async def recv(self) -> LiveServerEvent:
        return await asyncio.to_thread(self._session.recv)

    async def close(self) -> None:
        await asyncio.to_thread(self._session.close)

    def __aiter__(self):
        async def _gen():
            while True:
                try:
                    yield await self.recv()
                except RuntimeError:
                    break
        return _gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()


def _tool_result_parts(value: Any) -> list[Part]:
    if isinstance(value, str):
        return [TextPart(text=value)]
    if isinstance(value, PART_CLASSES):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
        if all(isinstance(part, PART_CLASSES) for part in parts):
            return parts
    return [TextPart(text=str(value))]


def _to_base64_str(data: bytes | str) -> str:
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    return data
