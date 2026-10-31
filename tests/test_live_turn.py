"""Live ergonomics: turn-scoped iteration, Turn materialization, native async.

Ratified 2026-09-01: (1) turn()/Turn in, with the half-duplex caveat
documented; (2) no combined send-and-drain verb; (3) native async
session, NOT a thread wrapper (a blocked sync recv in a worker thread
cannot be cancelled from the event loop); (4) live event types stay
separate-but-parallel to stream deltas.
"""
from __future__ import annotations

import base64
import json

import pytest

from lm15.live import AsyncWebSocketLiveSession, Turn, WebSocketLiveSession
from lm15.types import (
    ErrorDetail,
    LiveServerAudioEvent,
    LiveServerErrorEvent,
    LiveServerInterruptedEvent,
    LiveServerTextEvent,
    LiveServerToolCallEvent,
    LiveServerTurnEndEvent,
    LiveServerUsageEvent,
    Usage,
)


# ─── Scripted sockets ────────────────────────────────────────────────

class ScriptedWS:
    """Yields pre-decoded frames; the decode fn is identity-ish."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self):
        if not self.frames:
            raise RuntimeError("script exhausted")
        return self.frames.pop(0)

    def close(self):
        pass


class AsyncScriptedWS(ScriptedWS):
    async def send(self, data):  # type: ignore[override]
        self.sent.append(json.loads(data))

    async def recv(self):  # type: ignore[override]
        if not self.frames:
            raise RuntimeError("script exhausted")
        return self.frames.pop(0)

    async def close(self):  # type: ignore[override]
        pass


def decode_scripted(raw):
    """Frames arrive as already-typed events wrapped in a list marker."""
    return raw if isinstance(raw, list) else [raw]


def sync_session(frames) -> WebSocketLiveSession:
    return WebSocketLiveSession(ws=ScriptedWS(frames), encode_event=lambda e: [{"t": e.type}], decode_event=decode_scripted)


def async_session(frames) -> AsyncWebSocketLiveSession:
    return AsyncWebSocketLiveSession(ws=AsyncScriptedWS(frames), encode_event=lambda e: [{"t": e.type}], decode_event=decode_scripted)


HELLO_TURN = [
    LiveServerAudioEvent(data=base64.b64encode(b"PCM1").decode(), media_type="audio/pcm;rate=24000"),
    LiveServerTextEvent(text="live "),
    LiveServerAudioEvent(data=base64.b64encode(b"PCM2").decode()),
    LiveServerTextEvent(text="hello"),
    LiveServerTurnEndEvent(usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7)),
]


# ─── turn(): the self-ending iterator ────────────────────────────────

def test_turn_ends_itself_at_turn_end() -> None:
    session = sync_session(HELLO_TURN + [LiveServerTextEvent(text="NEXT TURN")])
    events = list(session.turn())
    # terminal event included, exactly like stream(); nothing beyond it
    assert [e.type for e in events] == ["audio", "text", "audio", "text", "turn_end"]


def test_second_turn_continues_where_first_stopped() -> None:
    second = [LiveServerTextEvent(text="again"), LiveServerTurnEndEvent(usage=Usage())]
    session = sync_session(HELLO_TURN + second)
    assert list(session.turn())[-1].type == "turn_end"
    assert [e.type for e in session.turn()] == ["text", "turn_end"]


def test_turn_ends_at_interrupted_and_error() -> None:
    session = sync_session([LiveServerTextEvent(text="1,"), LiveServerInterruptedEvent()])
    assert [e.type for e in session.turn()] == ["text", "interrupted"]
    session = sync_session([LiveServerErrorEvent(error=ErrorDetail(code="provider", message="boom"))])
    assert [e.type for e in session.turn()] == ["error"]


def test_tool_call_does_not_end_iteration() -> None:
    # You hold the session mid-iteration, so you can answer and keep going.
    frames = [
        LiveServerToolCallEvent(id="c1", name="f", input={"x": 1}),
        LiveServerTextEvent(text="answer"),
        LiveServerTurnEndEvent(usage=Usage()),
    ]
    session = sync_session(frames)
    assert [e.type for e in session.turn()] == ["tool_call", "text", "turn_end"]


# ─── Turn materialization ────────────────────────────────────────────

def test_result_materializes_text_audio_usage() -> None:
    session = sync_session(list(HELLO_TURN))
    turn = session.turn().result()
    assert isinstance(turn, Turn) and turn.ok
    assert turn.ended_by == "turn_end"
    assert turn.text == "live hello"
    assert turn.audio == b"PCM1PCM2"  # decoded and concatenated
    assert turn.audio_media_type == "audio/pcm;rate=24000"
    assert turn.usage is not None and turn.usage.total_tokens == 7
    assert len(turn.events) == 5


def test_result_returns_at_tool_call_instead_of_deadlocking() -> None:
    # result() cannot answer a tool call for you; the model is waiting.
    # Mirrors the non-live contract: finish_reason="tool_call" hands
    # control back to the caller.
    frames = [LiveServerTextEvent(text="calling "),
              LiveServerToolCallEvent(id="c1", name="get_weather", input={"city": "Montreal"})]
    session = sync_session(frames)  # no terminal frame: answering is required
    turn = session.turn().result()
    assert turn.ended_by == "tool_call" and not turn.ok
    assert turn.tool_calls[0].name == "get_weather"
    assert turn.tool_calls[0].input == {"city": "Montreal"}
    assert turn.usage is None


def test_result_interrupted_and_error() -> None:
    session = sync_session([LiveServerTextEvent(text="1,"), LiveServerInterruptedEvent()])
    turn = session.turn().result()
    assert turn.ended_by == "interrupted" and turn.text == "1," and not turn.ok

    session = sync_session([LiveServerErrorEvent(error=ErrorDetail(code="provider", message="boom"))])
    turn = session.turn().result()
    assert turn.ended_by == "error" and turn.error is not None and turn.error.message == "boom"


def test_turn_holds_tool_calls_as_data_only() -> None:
    # The no-loop ruling: sessions and turns never own tool execution.
    assert not hasattr(Turn, "run_tools")
    frames = [LiveServerToolCallEvent(id="c1", name="f", input={})]
    turn = sync_session(frames).turn().result()
    assert turn.tool_calls[0].id == "c1"  # data, not a callable


# ─── Native async session ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_turn_iteration_and_result() -> None:
    session = async_session(list(HELLO_TURN))
    events = [e async for e in session.turn()]
    assert [e.type for e in events] == ["audio", "text", "audio", "text", "turn_end"]

    session = async_session(list(HELLO_TURN))
    turn = await session.turn().result()
    assert turn.text == "live hello" and turn.audio == b"PCM1PCM2" and turn.ok


@pytest.mark.asyncio
async def test_async_send_verbs_encode_frames() -> None:
    session = async_session([])
    await session.send_text("hi")
    await session.interrupt()
    assert session._ws.sent == [{"t": "text"}, {"t": "interrupt"}]


@pytest.mark.asyncio
async def test_async_close_stops_iteration() -> None:
    session = async_session(list(HELLO_TURN))
    async with session:
        pass  # __aexit__ closes
    assert [e async for e in session] == []
    with pytest.raises(RuntimeError):
        await session.recv()


@pytest.mark.asyncio
async def test_async_recv_is_cancellable() -> None:
    # The reason the thread wrapper was rejected: a real awaitable
    # cancels; a blocked thread does not.
    import asyncio

    class HangingWS(AsyncScriptedWS):
        async def recv(self):
            await asyncio.sleep(3600)

    session = AsyncWebSocketLiveSession(ws=HangingWS([]), encode_event=lambda e: [], decode_event=decode_scripted)
    task = asyncio.ensure_future(session.recv())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ─── usage events: a turn's bill is everything it saw ────────────────

def test_tool_call_response_usage_lands_in_the_continuation_turn() -> None:
    # Wire order on OpenAI Realtime: output_item.done (tool_call) then
    # response.done (usage). result() returns at the tool_call; the usage
    # event is the first thing the continuation turn sees, and the
    # continuation's bill includes it (the semantic turn stayed open).
    frames = [
        LiveServerTextEvent(text="calling "),
        LiveServerToolCallEvent(id="c1", name="get_weather", input={"city": "Montreal"}),
        LiveServerUsageEvent(usage=Usage(input_tokens=54, output_tokens=21, total_tokens=75)),
        LiveServerTextEvent(text="It is sunny."),
        LiveServerTurnEndEvent(usage=Usage(input_tokens=90, output_tokens=10, total_tokens=100)),
    ]
    session = sync_session(frames)
    first = session.turn().result()
    assert first.ended_by == "tool_call" and first.usage is None
    second = session.turn().result()
    assert second.ended_by == "turn_end" and second.text == "It is sunny."
    assert second.usage == Usage(input_tokens=144, output_tokens=31, total_tokens=175)


def test_interrupted_turn_keeps_its_usage() -> None:
    frames = [
        LiveServerTextEvent(text="I was say"),
        LiveServerUsageEvent(usage=Usage(input_tokens=127, output_tokens=16, total_tokens=143)),
        LiveServerInterruptedEvent(),
    ]
    turn = sync_session(frames).turn().result()
    assert turn.ended_by == "interrupted" and turn.usage is not None and turn.usage.total_tokens == 143


def test_usage_sum_keeps_unknown_unknown() -> None:
    # INV-029 arithmetic: a counter absent on either side is unknown in the sum.
    frames = [
        LiveServerUsageEvent(usage=Usage(input_tokens=1, output_tokens=6, reasoning_tokens=5)),
        LiveServerTurnEndEvent(usage=Usage(input_tokens=2, output_tokens=2)),
    ]
    turn = sync_session(frames).turn().result()
    assert turn.usage.input_tokens == 3 and turn.usage.reasoning_tokens is None
