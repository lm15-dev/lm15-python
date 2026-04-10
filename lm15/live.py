from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections import deque
from typing import Any, Callable, Deque

from .types import LiveClientEvent, LiveServerEvent, Part, ToolCallInfo


EncodeEventFn = Callable[[LiveClientEvent], list[dict[str, Any]]]
DecodeEventFn = Callable[[str | bytes], list[LiveServerEvent]]


def require_websocket_sync_connect():
    """Return `websockets.sync.client.connect` or raise a helpful ImportError."""
    try:
        from websockets.sync.client import connect  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch tests
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
        callable_registry: dict[str, Callable[..., Any]] | None = None,
        on_tool_call: Callable[[ToolCallInfo], Any] | None = None,
    ) -> None:
        self._ws = ws
        self._encode_event = encode_event
        self._decode_event = decode_event
        self._callable_registry = callable_registry or {}
        self._on_tool_call = on_tool_call
        self._pending: Deque[LiveServerEvent] = deque()
        self._send_lock = threading.Lock()
        self._closed = False

    def set_on_tool_call(self, callback: Callable[[ToolCallInfo], Any] | None) -> None:
        self._on_tool_call = callback

    def send(
        self,
        event: LiveClientEvent | None = None,
        *,
        audio: bytes | str | None = None,
        video: bytes | str | None = None,
        text: str | None = None,
        tool_result: dict[str, Any] | None = None,
        interrupt: bool = False,
        end_audio: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("live session is closed")

        if event is not None:
            if any(x is not None for x in (audio, video, text, tool_result)) or interrupt or end_audio:
                raise ValueError("pass either `event` or keyword payload, not both")
            events = [event]
        else:
            events = self._events_from_kwargs(
                audio=audio,
                video=video,
                text=text,
                tool_result=tool_result,
                interrupt=interrupt,
                end_audio=end_audio,
            )

        with self._send_lock:
            for evt in events:
                payloads = self._encode_event(evt)
                for payload in payloads:
                    self._ws.send(json.dumps(payload))

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
                self._maybe_auto_execute_tool(event)
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
        video: bytes | str | None,
        text: str | None,
        tool_result: dict[str, Any] | None,
        interrupt: bool,
        end_audio: bool,
    ) -> list[LiveClientEvent]:
        events: list[LiveClientEvent] = []

        if audio is not None:
            payload = _to_base64_str(audio)
            events.append(LiveClientEvent(type="audio", data=payload))
        if video is not None:
            payload = _to_base64_str(video)
            events.append(LiveClientEvent(type="video", data=payload))
        if text is not None:
            events.append(LiveClientEvent(type="text", text=text))

        if tool_result:
            for call_id, value in tool_result.items():
                content = tuple(_tool_result_parts(value))
                events.append(LiveClientEvent(type="tool_result", id=call_id, content=content))

        if interrupt:
            events.append(LiveClientEvent(type="interrupt"))
        if end_audio:
            events.append(LiveClientEvent(type="end_audio"))

        if not events:
            raise ValueError("nothing to send")
        return events

    def _maybe_auto_execute_tool(self, event: LiveServerEvent) -> None:
        if event.type != "tool_call" or not event.id:
            return

        info = ToolCallInfo(
            id=event.id,
            name=event.name or "tool",
            input=event.input or {},
        )

        result: Any | None = None
        if self._on_tool_call is not None:
            override = self._on_tool_call(info)
            if override is not None:
                result = override

        if result is None:
            fn = self._callable_registry.get(info.name)
            if fn is not None:
                result = _invoke_tool(fn, info.input)

        if result is None:
            return

        self.send(tool_result={info.id: result})


class AsyncLiveSession:
    """Async wrapper over a sync live session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def send(self, event: LiveClientEvent | None = None, **kwargs) -> None:
        await asyncio.to_thread(self._session.send, event, **kwargs)

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
    if isinstance(value, Part):
        return [value]
    if isinstance(value, list) and all(isinstance(x, Part) for x in value):
        return list(value)
    return [Part.text_part(str(value))]


def _invoke_tool(fn: Callable[..., Any], payload: dict[str, Any]) -> Any:
    try:
        return fn(**payload)
    except TypeError:
        return fn(payload)


def _to_base64_str(data: bytes | str) -> str:
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    return data
