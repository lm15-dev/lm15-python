from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm15.api as api
from lm15.client import UniversalLM
from lm15.model import Model
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.types import Config, LMRequest, Message, Usage


class _DummyTransport:
    def request(self, req):  # pragma: no cover
        raise AssertionError("not used")

    def stream(self, req):  # pragma: no cover
        yield from ()


class _FakeWS:
    def __init__(self, recv_payloads: list[dict]):
        self._recv = [json.dumps(x) for x in recv_payloads]
        self.sent: list[str] = []
        self.closed = False

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self) -> str:
        if not self._recv:
            raise RuntimeError("no more websocket frames")
        return self._recv.pop(0)

    def close(self) -> None:
        self.closed = True


class _OpenAICompletionAdapter(OpenAIAdapter):
    def __init__(self, sessions: list[_FakeWS]):
        super().__init__(api_key="k", transport=_DummyTransport())
        self._sessions = sessions

    def _live_connect(self, url: str, headers: dict[str, str]):
        return self._sessions.pop(0)


class _GeminiCompletionAdapter(GeminiAdapter):
    def __init__(self, sessions: list[_FakeWS]):
        super().__init__(api_key="k", transport=_DummyTransport())
        self._sessions = sessions

    def _live_connect(self, url: str):
        return self._sessions.pop(0)


class LiveCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_build_default = api.build_default
        api._client_cache.clear()

    def tearDown(self) -> None:
        api.build_default = self._old_build_default
        api._client_cache.clear()
        api.configure()

    def test_openai_live_model_streams_via_websocket_completion(self):
        ws = _FakeWS(
            [
                {"type": "response.output_text.delta", "delta": "ok"},
                {"type": "response.done", "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}},
            ]
        )
        adapter = _OpenAICompletionAdapter([ws])

        req = LMRequest(model="gpt-4o-realtime-preview", messages=(Message.user("hi"),), config=Config())
        events = list(adapter.stream(req))

        self.assertEqual(events[0].type, "start")
        self.assertEqual(events[1].type, "delta")
        self.assertEqual(events[1].delta.type, "text")
        self.assertEqual(events[1].delta.text, "ok")
        self.assertEqual(events[-1].type, "end")
        self.assertEqual(events[-1].finish_reason, "stop")

    def test_gemini_live_model_streams_via_websocket_completion(self):
        ws = _FakeWS(
            [
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "ok"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                }
            ]
        )
        adapter = _GeminiCompletionAdapter([ws])

        req = LMRequest(model="gemini-2.0-flash-live", messages=(Message.user("hi"),), config=Config())
        events = list(adapter.stream(req))

        self.assertEqual(events[0].type, "start")
        self.assertEqual(events[1].type, "delta")
        self.assertEqual(events[1].delta.type, "text")
        self.assertEqual(events[1].delta.text, "ok")
        self.assertEqual(events[-1].type, "end")
        self.assertEqual(events[-1].finish_reason, "stop")

    def test_openai_live_model_supports_tool_loop_in_model_call(self):
        ws_round_1 = _FakeWS(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Montreal"}',
                    },
                },
                {"type": "response.done", "response": {"usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}},
            ]
        )
        ws_round_2 = _FakeWS(
            [
                {"type": "response.output_text.delta", "delta": "Tool says: 22C in Montreal"},
                {"type": "response.done", "response": {"usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}}},
            ]
        )
        lm = UniversalLM()
        lm.register(_OpenAICompletionAdapter([ws_round_1, ws_round_2]))

        def get_weather(city: str) -> str:
            return f"22C in {city}"

        agent = Model(lm=lm, model="gpt-4o-realtime-preview", provider="openai")
        resp = agent.call("weather", tools=[get_weather])
        self.assertIn("22C", resp.text or "")

    def test_module_call_uses_live_completion_transport(self):
        ws = _FakeWS(
            [
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "ok"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                }
            ]
        )
        lm = UniversalLM()
        lm.register(_GeminiCompletionAdapter([ws]))
        api.build_default = lambda **_kw: lm

        resp = api.call("gemini-2.0-flash-live", "hi", provider="gemini")
        self.assertEqual(resp.text, "ok")

    def test_gemini_live_model_supports_tool_loop_in_model_call(self):
        ws_round_1 = _FakeWS(
            [
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call_1",
                                        "name": "get_weather",
                                        "args": {"city": "Montreal"},
                                    }
                                }
                            ]
                        },
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
                }
            ]
        )
        ws_round_2 = _FakeWS(
            [
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "Tool says: 22C in Montreal"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2, "totalTokenCount": 4},
                }
            ]
        )
        lm = UniversalLM()
        lm.register(_GeminiCompletionAdapter([ws_round_1, ws_round_2]))

        def get_weather(city: str) -> str:
            return f"22C in {city}"

        agent = Model(lm=lm, model="gemini-2.0-flash-live", provider="gemini")
        resp = agent.call("weather", tools=[get_weather])
        self.assertIn("22C", resp.text or "")


if __name__ == "__main__":
    unittest.main()
