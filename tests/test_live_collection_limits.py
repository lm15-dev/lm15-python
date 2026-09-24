"""Bound collection, preserve partial/rejected data, never fabricate a turn end."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lm15 import CollectionLimitError
from lm15.errors import LM15Error, RETRYABLE_ERRORS, canonical_error_code, error_class_for_code
from lm15.live import (
    AsyncTurnView, TurnView, DEFAULT_TURN_MAX_BYTES, DEFAULT_TURN_MAX_EVENTS,
)
from lm15.serde import live_server_event_from_dict
from lm15.types import LiveServerTextEvent, LiveServerTurnEndEvent, Usage
from tests.test_live_turn import async_session, sync_session


class Peer:
    def __init__(self, events):
        self.events = iter(events)
        self.reads = 0

    def recv(self):
        self.reads += 1
        return next(self.events)


class AsyncPeer(Peer):
    async def recv(self):
        return super().recv()


async def read(view, asynchronous):
    if asynchronous:
        return await view.__anext__()
    try:
        return next(view)
    except StopIteration:
        raise StopAsyncIteration from None


async def result(view, asynchronous):
    return await view.result() if asynchronous else view.result()


CORPUS = Path(__file__).resolve().parents[2] / "lm15-contract" / "consumer" / "live-collection-limits.json"


def shared_cases():
    if not CORPUS.exists():
        return [pytest.param(None, marks=pytest.mark.skip(reason="sibling contract collection vectors unavailable"))]
    return [pytest.param(c, id=c["id"]) for c in json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("case", shared_cases())
def test_shared_collection_vectors(case, asynchronous):
    events = [live_server_event_from_dict(e) for e in case["events"]]
    peer = (AsyncPeer if asynchronous else Peer)(events)
    view = (AsyncTurnView if asynchronous else TurnView)(peer, **case["limits"])
    expected = case["expect"]

    async def exercise():
        yielded = []
        failure = None
        try:
            while True:
                yielded.append(await read(view, asynchronous))
        except (StopIteration, StopAsyncIteration):
            pass
        except CollectionLimitError as exc:
            failure = exc
        assert yielded == events[:expected["accepted"]]
        assert peer.reads == expected["reads"]
        assert view.retained_bytes == expected["retained_bytes"]
        assert view.retained_events == expected["accepted"]
        if expected["limit"] is None:
            assert failure is None and (await result(view, asynchronous)).ok
        else:
            assert failure is not None
            assert failure.limit == expected["limit"]
            assert failure.maximum == case["limits"][failure.limit]
            assert failure.partial_events == tuple(yielded)
            rejected = expected["rejected_index"]
            assert failure.rejected_event is (events[rejected] if rejected is not None else None)
            assert failure.partial.ended_by == "incomplete" and not failure.partial.ok
            assert view.snapshot().ended_by == "incomplete"
            # The failed view is sealed, even after close; no further reads.
            view.close()
            for operation in (read, result):
                with pytest.raises(CollectionLimitError) as again:
                    await operation(view, asynchronous)
                assert again.value is failure
            assert peer.reads == expected["reads"]

    asyncio.run(exercise())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["iterate", "result"])
def test_endless_peer_hits_the_default_event_cap(asynchronous, mode):
    def endless():
        while True:
            yield LiveServerTextEvent(text="")

    peer = (AsyncPeer if asynchronous else Peer)(endless())
    view = (AsyncTurnView if asynchronous else TurnView)(peer)

    async def exercise():
        with pytest.raises(CollectionLimitError) as info:
            if mode == "result":
                await result(view, asynchronous)
            else:
                while True:
                    await read(view, asynchronous)
        assert info.value.limit == "max_events"
        assert info.value.maximum == 10_000
        assert peer.reads == 10_000
        assert view.retained_events == 10_000
        assert info.value.rejected_event is None
        assert not info.value.partial.ok

    asyncio.run(exercise())


def test_default_limits_match_shared_contract():
    assert DEFAULT_TURN_MAX_BYTES == 16 * 1024 * 1024
    assert DEFAULT_TURN_MAX_EVENTS == 10_000
    if CORPUS.exists():
        assert json.loads(CORPUS.read_text(encoding="utf-8"))["defaults"] == {
            "max_bytes": DEFAULT_TURN_MAX_BYTES, "max_events": DEFAULT_TURN_MAX_EVENTS,
        }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("key", ["max_bytes", "max_events"])
@pytest.mark.parametrize("bad", [0, -1, True, False, None, 1.5, "100", float("inf")])
def test_invalid_limits_are_rejected_before_reading(asynchronous, key, bad):
    session = (async_session if asynchronous else sync_session)([])
    with pytest.raises(ValueError, match=key):
        session.turn(**{key: bad})


@pytest.mark.parametrize("asynchronous", [False, True])
def test_session_accepts_custom_limits_and_remains_open(asynchronous):
    first = LiveServerTextEvent(text="abc")
    rejected = LiveServerTextEvent(text="this event will not fit")
    end = LiveServerTurnEndEvent(usage=Usage())
    # Several decoded events can share a frame. Rejecting one must not
    # discard the others already queued by the session.
    session = (async_session if asynchronous else sync_session)([[first, rejected, end]])
    view = session.turn(max_bytes=28, max_events=10)

    async def exercise():
        assert await read(view, asynchronous) is first
        with pytest.raises(CollectionLimitError) as info:
            await read(view, asynchronous)
        assert info.value.rejected_event is rejected
        assert not session._closed and session._ws.sent == []
        # Recovery: process rejected_event first; raw reads then continue
        # with the remaining events, with no silent loss or auto-drain.
        following = await session.recv() if asynchronous else session.recv()
        assert following is end
        assert info.value.partial.text == "abc" and not info.value.partial.ok

    asyncio.run(exercise())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_limit_error_does_not_eagerly_materialize_or_expose_payload(asynchronous, monkeypatch):
    import lm15.live as live

    def forbidden(*args):
        raise AssertionError("do not allocate a combined response while reporting a limit")

    monkeypatch.setattr(live, "_materialize_turn", forbidden)
    event = LiveServerTextEvent(text="private content")
    peer = (AsyncPeer if asynchronous else Peer)([event])
    view = (AsyncTurnView if asynchronous else TurnView)(peer, max_bytes=1)

    async def exercise():
        with pytest.raises(CollectionLimitError) as info:
            await read(view, asynchronous)
        assert "private content" not in str(info.value)
        assert info.value.partial_events == ()
        assert info.value.rejected_event is event

    asyncio.run(exercise())


def test_collection_limit_is_local_and_not_retryable():
    error = CollectionLimitError("budget reached")
    assert isinstance(error, LM15Error)
    assert not isinstance(error, RETRYABLE_ERRORS)
    assert canonical_error_code(error) == "collection_limit"
    assert error_class_for_code("collection_limit") is CollectionLimitError
    assert error.status is None and error.provider is None
    assert error.partial.ended_by == "incomplete"
