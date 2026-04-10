from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm15.api as api
from lm15.client import UniversalLM
from lm15.live import AsyncLiveSession
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.types import LiveConfig, Tool


class _DummyTransport:
    def request(self, req):  # pragma: no cover - not used by these tests
        raise AssertionError("not expected")

    def stream(self, req):  # pragma: no cover - not used by these tests
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


class _OpenAITestAdapter(OpenAIAdapter):
    def __init__(self, *, ws: _FakeWS):
        super().__init__(api_key="k", transport=_DummyTransport())
        self._ws = ws
        self.live_url: str | None = None
        self.live_headers: dict[str, str] | None = None

    def _live_connect(self, url: str, headers: dict[str, str]):
        self.live_url = url
        self.live_headers = headers
        return self._ws


class _GeminiTestAdapter(GeminiAdapter):
    def __init__(self, *, ws: _FakeWS):
        super().__init__(api_key="k", transport=_DummyTransport())
        self._ws = ws
        self.live_url: str | None = None

    def _live_connect(self, url: str):
        self.live_url = url
        return self._ws


class LiveRuntimeTests(unittest.TestCase):
    def test_openai_live_send_and_receive(self):
        ws = _FakeWS(
            [
                {"type": "response.output_text.delta", "delta": "hi"},
                {"type": "response.done", "response": {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}},
            ]
        )
        adapter = _OpenAITestAdapter(ws=ws)

        session = adapter.live(LiveConfig(model="gpt-4o-realtime-preview"))
        self.assertTrue(adapter.live_url and adapter.live_url.startswith("wss://"))
        self.assertTrue(adapter.live_headers and "Authorization" in adapter.live_headers)

        # Session bootstrap
        self.assertTrue(any(json.loads(m).get("type") == "session.update" for m in ws.sent))

        session.send(text="hello")
        sent_types = [json.loads(m).get("type") for m in ws.sent]
        self.assertIn("conversation.item.create", sent_types)
        self.assertIn("response.create", sent_types)

        e1 = session.recv()
        self.assertEqual(e1.type, "text")
        self.assertEqual(e1.text, "hi")

        e2 = session.recv()
        self.assertEqual(e2.type, "turn_end")
        self.assertEqual(e2.usage.total_tokens, 3)

        session.close()
        self.assertTrue(ws.closed)

    def test_openai_live_auto_tool_execution(self):
        ws = _FakeWS(
            [
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Montreal"}',
                }
            ]
        )
        adapter = _OpenAITestAdapter(ws=ws)

        def get_weather(city: str) -> str:
            return f"22C in {city}"

        tool = Tool(
            name="get_weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            fn=get_weather,
        )

        session = adapter.live(LiveConfig(model="gpt-4o-realtime-preview", tools=(tool,)))
        event = session.recv()
        self.assertEqual(event.type, "tool_call")
        self.assertEqual(event.id, "call_1")

        sent = [json.loads(x) for x in ws.sent]
        outputs = [x for x in sent if x.get("type") == "conversation.item.create" and (x.get("item") or {}).get("type") == "function_call_output"]
        self.assertEqual(len(outputs), 1)
        self.assertIn("22C", outputs[0]["item"]["output"])

    def test_gemini_live_send_and_receive(self):
        ws = _FakeWS(
            [
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "ok"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }
            ]
        )
        adapter = _GeminiTestAdapter(ws=ws)

        session = adapter.live(LiveConfig(model="gemini-2.0-flash-live"))
        self.assertTrue(adapter.live_url and adapter.live_url.startswith("wss://"))

        # setup frame is sent immediately
        self.assertTrue(any("setup" in json.loads(m) for m in ws.sent))

        session.send(text="hello")
        self.assertTrue(any("clientContent" in json.loads(m) for m in ws.sent))

        e1 = session.recv()
        self.assertEqual(e1.type, "text")
        self.assertEqual(e1.text, "ok")

        e2 = session.recv()
        self.assertEqual(e2.type, "turn_end")
        self.assertEqual(e2.usage.total_tokens, 2)


class LiveAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_build_default = api.build_default
        api._client_cache.clear()

    def tearDown(self) -> None:
        api.build_default = self._old_build_default
        api._client_cache.clear()
        api.configure()

    def test_module_live(self):
        ws = _FakeWS([{"type": "response.output_text.delta", "delta": "hi"}])
        lm = UniversalLM()
        lm.register(_OpenAITestAdapter(ws=ws))
        api.build_default = lambda **_kw: lm

        session = api.live("gpt-4o-realtime-preview", provider="openai")
        evt = session.recv()
        self.assertEqual(evt.type, "text")


class AsyncLiveRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_live_session_wrapper(self):
        ws = _FakeWS([
            {"type": "response.output_text.delta", "delta": "hi"},
        ])
        adapter = _OpenAITestAdapter(ws=ws)
        sync_session = adapter.live(LiveConfig(model="gpt-4o-realtime-preview"))
        session = AsyncLiveSession(sync_session)

        await session.send(text="hello")
        evt = await session.recv()
        self.assertEqual(evt.type, "text")
        self.assertEqual(evt.text, "hi")

        await session.close()
        self.assertTrue(ws.closed)

    async def test_module_alive(self):
        old_build_default = api.build_default
        try:
            api._client_cache.clear()
            ws = _FakeWS([{"type": "response.output_text.delta", "delta": "ok"}])
            lm = UniversalLM()
            lm.register(_OpenAITestAdapter(ws=ws))
            api.build_default = lambda **_kw: lm

            session = await api.alive("gpt-4o-realtime-preview", provider="openai")
            evt = await session.recv()
            self.assertEqual(evt.type, "text")
        finally:
            api.build_default = old_build_default
            api._client_cache.clear()
            api.configure()


if __name__ == "__main__":
    unittest.main()
