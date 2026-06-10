"""Async mirror adapters: same Request in, same canonical Response/StreamEvents out.

The async classes are composition-based mirrors of their sync siblings:
all pure mapping is delegated to an inner sync adapter whose transport must
never be touched.  These tests assert the mirror guarantee literally.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from lm15 import Message, Request, StreamEndEvent
from lm15.providers import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM
from lm15.providers.async_base import (
    AsyncAnthropicLM,
    AsyncBaseProviderLM,
    AsyncGeminiLM,
    AsyncOpenAIChatLM,
    AsyncOpenAILM,
)
from lm15.providers.base import HttpResponse
from lm15.result import acoalesce_stream
from lm15.errors import UnsupportedFeatureError
from lm15.transports import AsyncTransportResponse, TransportRequest

CONTRACT_BODIES = Path(__file__).resolve().parents[2] / "lm15-contract" / "bodies"

PAIRS = [
    ("openai", OpenAILM, AsyncOpenAILM),
    ("anthropic", AnthropicLM, AsyncAnthropicLM),
    ("gemini", GeminiLM, AsyncGeminiLM),
    ("openai_chat", OpenAIChatLM, AsyncOpenAIChatLM),
]

_REQ = Request(model="m-test", messages=(Message.user("Hi"),))


def _latest_body(fixture_id: str) -> bytes:
    body_dir = CONTRACT_BODIES / fixture_id
    paths = sorted(body_dir.glob("*.txt"))
    if not paths:
        pytest.skip(f"no recorded body for {fixture_id}")
    return paths[-1].read_bytes()


class FakeAsyncTransport:
    """Replays one recorded body; never opens a socket."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[TransportRequest] = []

    def stream(self, request: TransportRequest) -> AsyncTransportResponse:
        self.requests.append(request)

        async def chunks():
            yield self.body

        async def release(body_consumed: bool) -> None:
            return None

        return AsyncTransportResponse(
            status=self.status,
            reason="OK",
            headers=[("content-type", "application/json")],
            http_version="HTTP/1.1",
            chunks=chunks(),
            release=release,
        )


def _make_pair(sync_cls, async_cls, body: bytes):
    sync_lm = sync_cls(api_key="test")
    async_lm = async_cls(api_key="test", transport=FakeAsyncTransport(body))
    return sync_lm, async_lm


# ─── (a) mirror parity: constructor fields ───────────────────────────

@pytest.mark.parametrize(("provider", "sync_cls", "async_cls"), PAIRS)
def test_constructor_field_parity(provider, sync_cls, async_cls):
    sync_fields = [f.name for f in dataclasses.fields(sync_cls) if f.init]
    async_fields = [f.name for f in dataclasses.fields(async_cls) if f.init]
    assert async_fields == sync_fields, (
        f"{async_cls.__name__} drifted from {sync_cls.__name__}: "
        f"{async_fields} != {sync_fields}"
    )


# ─── (b) complete()/stream(): async result == sync parse of same bytes ───

@pytest.mark.parametrize(("provider", "sync_cls", "async_cls"), PAIRS)
def test_complete_mirrors_sync_parse(provider, sync_cls, async_cls):
    body = _latest_body(f"{provider}.basic_text")
    sync_lm, async_lm = _make_pair(sync_cls, async_cls, body)

    expected = sync_lm.parse_response(
        _REQ,
        HttpResponse(status=200, reason="OK",
                     headers=[("content-type", "application/json")], body=body),
    )
    actual = asyncio.run(async_lm.complete(_REQ))
    assert actual == expected


@pytest.mark.parametrize(("provider", "sync_cls", "async_cls"), PAIRS)
def test_stream_mirrors_sync_parse(provider, sync_cls, async_cls):
    from lm15.sse import parse_sse
    from lm15.result import coalesce_stream

    body = _latest_body(f"{provider}.streaming")
    sync_lm, async_lm = _make_pair(sync_cls, async_cls, body)

    def sync_events():
        for raw in parse_sse(iter(body.splitlines(keepends=True))):
            yield from (e for e in sync_lm.parse_stream_events(_REQ, raw) if e is not None)

    expected = list(coalesce_stream(sync_events()))

    async def collect():
        return [event async for event in async_lm.stream(_REQ)]

    actual = asyncio.run(collect())
    assert actual == expected
    ends = [e for e in actual if isinstance(e, StreamEndEvent)]
    assert len(ends) == 1 and actual[-1] is ends[0]


# ─── (c) MAP-3: post-finish usage chunk, exactly one end event ───────

def _chat_chunk(**kwargs) -> str:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "m-test"}
    base.update(kwargs)
    return "data: " + json.dumps(base) + "\n\n"


def test_async_stream_coalesces_post_finish_usage():
    sse = (
        _chat_chunk(choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}])
        + _chat_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
        + _chat_chunk(choices=[], usage={"prompt_tokens": 14, "completion_tokens": 22, "total_tokens": 36})
        + "data: [DONE]\n\n"
    ).encode("utf-8")
    lm = AsyncOpenAIChatLM(api_key="test", transport=FakeAsyncTransport(sse))

    async def collect():
        return [event async for event in lm.stream(_REQ)]

    events = asyncio.run(collect())
    ends = [e for e in events if isinstance(e, StreamEndEvent)]
    assert len(ends) == 1
    assert events[-1] is ends[0]
    assert ends[0].finish_reason == "stop"
    assert ends[0].usage is not None
    assert ends[0].usage.input_tokens == 14
    assert ends[0].usage.output_tokens == 22


def test_acoalesce_stream_exported_top_level():
    import lm15

    assert lm15.acoalesce_stream is acoalesce_stream
    for name in ("AsyncOpenAILM", "AsyncAnthropicLM", "AsyncGeminiLM", "AsyncOpenAIChatLM"):
        assert name in lm15.__all__
        assert getattr(lm15, name) is not None


# ─── (d) purity guard: inner sync adapter never touches the network ──

@pytest.mark.parametrize(("provider", "sync_cls", "async_cls"), PAIRS)
def test_inner_sync_transport_raises_if_touched(provider, sync_cls, async_cls):
    lm = async_cls(api_key="test", transport=FakeAsyncTransport(b"{}"))
    inner = lm._inner
    assert isinstance(inner, sync_cls)
    with pytest.raises(RuntimeError, match="must never touch the network"):
        inner.transport.stream(TransportRequest(method="GET", url="http://example.invalid/"))


def test_async_unsupported_endpoints_are_honest():
    lm = AsyncOpenAILM(api_key="test", transport=FakeAsyncTransport(b"{}"))
    from lm15.types import EmbeddingRequest

    with pytest.raises(UnsupportedFeatureError, match="use the sync adapter"):
        lm.embeddings(EmbeddingRequest(model="m", inputs=("x",)))


# ─── live smoke vs local ollama ──────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"


def _ollama_model() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as resp:
            tags = json.loads(resp.read())
        models = [m["name"] for m in tags.get("models", [])]
        return models[0] if models else None
    except Exception:
        return None


_OLLAMA_MODEL = _ollama_model()
ollama = pytest.mark.skipif(_OLLAMA_MODEL is None, reason="local ollama not reachable")


def _ollama_lm() -> AsyncOpenAIChatLM:
    from lm15.transports import StdlibAsyncTransport

    # Generous read timeout: local models can be slow under load.
    return AsyncOpenAIChatLM(
        api_key="ollama", compat="ollama",
        transport=StdlibAsyncTransport(read_timeout=300.0),
    )


def _ollama_request(prompt: str) -> Request:
    from lm15 import Config

    # reasoning_effort=none keeps thinking-tuned local models from burning the
    # whole token budget on reasoning (passthrough extension; the ollama
    # compat policy omits the canonical reasoning knob).
    return Request(
        model=_OLLAMA_MODEL,
        messages=(Message.user(prompt),),
        config=Config(max_tokens=80, extensions={"reasoning_effort": "none"}),
    )


@ollama
def test_live_async_complete_ollama():
    lm = _ollama_lm()
    req = _ollama_request("Say hi in one word.")

    async def main():
        try:
            return await lm.complete(req)
        finally:
            await lm.aclose()

    response = asyncio.run(main())
    assert response.text and response.text.strip()


@ollama
def test_live_async_stream_ollama():
    lm = _ollama_lm()
    req = _ollama_request("Count to three.")

    async def main():
        try:
            return [event async for event in lm.stream(req)]
        finally:
            await lm.aclose()

    events = asyncio.run(main())
    deltas = [e for e in events if e.type == "delta"]
    ends = [e for e in events if isinstance(e, StreamEndEvent)]
    assert deltas, "expected at least one delta event"
    assert len(ends) == 1 and events[-1] is ends[0]
    assert ends[0].usage is not None


@ollama
def test_live_async_concurrency_no_threads():
    lm = _ollama_lm()
    req = _ollama_request("Reply with the word ok.")
    before = threading.active_count()

    async def main():
        try:
            return await asyncio.gather(*(lm.complete(req) for _ in range(5)))
        finally:
            await lm.aclose()

    responses = asyncio.run(main())
    assert len(responses) == 5
    assert all(r.text and r.text.strip() for r in responses)
    assert threading.active_count() == before
