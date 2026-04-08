from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import LMRequest, Message, Part

FIX = Path(__file__).parent / "fixtures"


class FakeTransport:
    def __init__(self, response_payload: dict, stream_lines: list[bytes] | None = None):
        self.response_payload = response_payload
        self.stream_lines = stream_lines or []
        self.last_request: HttpRequest | None = None

    def request(self, req: HttpRequest) -> HttpResponse:
        self.last_request = req
        return HttpResponse(status=200, headers={"content-type": "application/json"}, body=json.dumps(self.response_payload).encode())

    def stream(self, req: HttpRequest):
        self.last_request = req
        for line in self.stream_lines:
            yield line


class ProviderConformanceTests(unittest.TestCase):
    def _req(self, model: str) -> LMRequest:
        return LMRequest(model=model, messages=(Message.user("hi"),))

    def test_openai_roundtrip(self):
        payload = json.loads((FIX / "openai_response.json").read_text())
        t = FakeTransport(payload)
        a = OpenAIAdapter(api_key="k", transport=t)
        r = a.complete(self._req("gpt-4.1-mini"))
        self.assertEqual(r.message.parts[0].text, "ok")
        self.assertEqual(r.usage.total_tokens, 4)
        self.assertIn("/responses", t.last_request.url)

    def test_anthropic_roundtrip(self):
        payload = json.loads((FIX / "anthropic_response.json").read_text())
        t = FakeTransport(payload)
        a = AnthropicAdapter(api_key="k", transport=t)
        r = a.complete(self._req("claude-sonnet-4-5"))
        self.assertEqual(r.message.parts[0].text, "ok")
        self.assertEqual(r.finish_reason, "stop")
        self.assertIn("/messages", t.last_request.url)

    def test_gemini_roundtrip(self):
        payload = json.loads((FIX / "gemini_response.json").read_text())
        t = FakeTransport(payload)
        a = GeminiAdapter(api_key="k", transport=t)
        r = a.complete(self._req("gemini-2.0-flash-lite"))
        self.assertEqual(r.message.parts[0].text, "ok")
        self.assertEqual(r.usage.total_tokens, 3)
        self.assertIn(":generateContent", t.last_request.url)

    def test_openai_stream_replay(self):
        lines = [
            b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n",
            b"\n",
            b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"o\"}\n",
            b"\n",
            b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"k\"}\n",
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        t = FakeTransport({}, lines)
        a = OpenAIAdapter(api_key="k", transport=t)
        events = list(a.stream(self._req("gpt-4.1-mini")))
        deltas = [e for e in events if e.type == "delta"]
        self.assertEqual("".join(d.delta.text for d in deltas), "ok")

    def test_anthropic_stream_replay(self):
        lines = [
            b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude\"}}\n",
            b"\n",
            b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n",
            b"\n",
            b"data: {\"type\":\"message_stop\"}\n",
            b"\n",
        ]
        t = FakeTransport({}, lines)
        a = AnthropicAdapter(api_key="k", transport=t)
        events = list(a.stream(self._req("claude-sonnet-4-5")))
        self.assertTrue(any(e.type == "delta" for e in events))
        self.assertTrue(any(e.type == "end" for e in events))


if __name__ == "__main__":
    unittest.main()
