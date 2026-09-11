"""MAP-7 rules 7/12: ask Groq for typed reasoning; never parse text tags.

Synthetic wire replies exercise the existing policy, not new provider facts.
The Groq behavior is recorded in the contract's 2026-09-02 reasoning decision
and its groq.ingest_groq_reasoning_format_alone case.
"""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import re
import socket

import pytest

import lm15
from lm15 import (
    AsyncLMRouter, AsyncResponseStream, Config, LMRouter, Message, Request,
    ResponseStream, RouterConfig, ThinkingPart,
)
from lm15.testing import FakeResponse, FakeTransport
from tests.test_async_adapters import FakeAsyncTransport

MODEL = "qwen/qwen3.6-27b"
THINKING = "Compute 17 * (20 + 3)."
ANSWER = "391"


def reply(answer=ANSWER):
    return json.dumps({
        "model": MODEL,
        "choices": [{"message": {"role": "assistant", "reasoning": THINKING,
                                 "content": answer}, "finish_reason": "stop"}],
    }).encode()


def stream_reply(answer=ANSWER):
    frames = [
        {"choices": [{"delta": {"reasoning": THINKING}}]},
        {"choices": [{"delta": {"content": answer}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    return b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames) + b"data: [DONE]\n\n"


def assert_wire(request):
    assert request.url == "https://api.groq.com/openai/v1/chat/completions"
    body = json.loads(request.body)
    assert body["model"] == MODEL
    assert body["reasoning_format"] == "parsed"
    assert "reasoning_effort" not in body  # no invented Qwen effort level
    assert "include_reasoning" not in body  # mutually exclusive Groq controls


def test_groq_recipe_runs_as_documented(monkeypatch, capsys):
    transport = FakeTransport([FakeResponse(200, reply()), FakeResponse(200, stream_reply())])
    router = LMRouter(RouterConfig(env={}, api_keys={"groq": "test"}, transport=transport))
    monkeypatch.setattr(lm15, "LMRouter", lambda: router)

    def no_network(*args, **kwargs):
        pytest.fail("The recipe test must not make network calls")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    doc = (Path(__file__).parents[1] / "docs/cookbooks/10-audio-video-reasoning.md").read_text()
    section = doc.split("## Groq/Qwen: keep thinking out of the answer\n", 1)[1].split("## How it works", 1)[0]
    blocks = re.findall(r"^```python\n(.*?)^```", section, re.M | re.S)
    assert len(blocks) == 1
    scope = {}
    exec(compile(blocks[0], "Groq reasoning recipe", "exec"), scope)
    assert capsys.readouterr().out == "391\n391"
    assert scope["reasoning"][0].text == THINKING
    assert scope["completed"].message.first(ThinkingPart).text == THINKING
    assert len(transport.requests) == 2
    for request in transport.requests:
        assert_wire(request)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_literal_think_tags_in_answer_are_not_removed(asynchronous, streaming):
    # Even with parsed reasoning, a model may quote literal markup in its
    # answer. Only the typed reasoning field becomes a ThinkingPart.
    answer = "Example markup: <think>this is literal text</think>."
    body = stream_reply(answer) if streaming else reply(answer)
    transport = FakeAsyncTransport(body) if asynchronous else FakeTransport([FakeResponse(200, body)])
    config = RouterConfig(env={}, api_keys={"groq": "test"}, transport=transport)
    req = Request(model=f"groq:{MODEL}", messages=(Message.user("Quote some markup"),),
                  config=Config(extensions={"reasoning_format": "parsed"}))

    async def run():
        router = AsyncLMRouter(config)
        if not streaming:
            return await router.complete(req)
        stream = AsyncResponseStream(router.stream(req), req)
        chunks = [chunk async for chunk in stream]
        assert "".join(chunks) == answer
        return await stream.response()

    if asynchronous:
        response = asyncio.run(run())
    else:
        router = LMRouter(config)
        if streaming:
            stream = ResponseStream(router.stream(req), req)
            assert "".join(stream) == answer
            response = stream.response
        else:
            response = router.complete(req)
    assert response.text == answer
    assert response.message.first(ThinkingPart).text == THINKING
    assert_wire(transport.requests[0])


def test_groq_default_remains_provider_default():
    router = LMRouter(RouterConfig(env={}, api_keys={"groq": "test"}))
    req = Request(model=MODEL, messages=(Message.user("hi"),))
    lm = router.lm(f"groq:{MODEL}")
    body = json.loads(lm.build_request(req, stream=False).body)
    assert "reasoning_format" not in body
    assert "reasoning_effort" not in body
    parsed = replace(req, config=Config(extensions={"reasoning_format": "parsed"}))
    assert_wire(lm.build_request(parsed, stream=False))
