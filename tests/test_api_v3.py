from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm15.api as api
from lm15.api import acall, call, configure, model
from lm15.client import UniversalLM
from lm15.conversation import Conversation
from lm15.features import EndpointSupport, ProviderManifest
from lm15.protocols import Capabilities
from lm15.result import Result, response_to_events
from lm15.types import LMRequest, LMResponse, Message, Part, Tool, Usage


class _V3Adapter:
    provider = "openai"
    capabilities = Capabilities()
    supports = EndpointSupport(complete=True, stream=True)
    manifest = ProviderManifest(provider="openai", supports=supports)

    def _response(self, request: LMRequest) -> LMResponse:
        last = request.messages[-1]
        text = " ".join(p.text or "" for p in last.parts if p.type == "text")

        if request.tools and "weather" in text.lower() and not any(m.role == "tool" for m in request.messages):
            return LMResponse(
                id="r1",
                model=request.model,
                message=Message(role="assistant", parts=(Part.tool_call("call_1", "get_weather", {"city": "Montreal"}),)),
                finish_reason="tool_call",
                usage=Usage(),
            )

        tool_msgs = [m for m in request.messages if m.role == "tool"]
        if tool_msgs:
            tool_text = "\n".join(
                item.text or ""
                for part in tool_msgs[-1].parts
                for item in part.content
                if item.type == "text"
            )
            return LMResponse(
                id="r2",
                model=request.model,
                message=Message.assistant(f"Tool says: {tool_text}"),
                finish_reason="stop",
                usage=Usage(total_tokens=7),
            )

        return LMResponse(
            id="r0",
            model=request.model,
            message=Message.assistant(f"Echo: {text}"),
            finish_reason="stop",
            usage=Usage(total_tokens=3),
        )

    def complete(self, request: LMRequest) -> LMResponse:
        return self._response(request)

    def stream(self, request: LMRequest):
        yield from response_to_events(self._response(request), request)


class _APIV3Base(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = UniversalLM()
        self.lm.register(_V3Adapter())
        self._old_build_default = api.build_default
        api.build_default = lambda **_kw: self.lm
        api._client_cache.clear()

    def tearDown(self) -> None:
        api.build_default = self._old_build_default
        api._client_cache.clear()
        configure()


class APIV3Tests(_APIV3Base):
    def test_call_returns_result_and_iterates_text(self):
        resp = call("gpt-4.1-mini", "hello")
        self.assertIsInstance(resp, Result)
        self.assertEqual("".join(resp), "Echo: hello")
        self.assertEqual(resp.text, "Echo: hello")

    def test_on_tool_call_can_override_execution(self):
        weather = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        )

        def override(call_info):
            return f"manual result for {call_info.input['city']}"

        resp = call("gpt-4.1-mini", "weather please", tools=[weather], on_tool_call=override)
        self.assertEqual(resp.text, "Tool says: manual result for Montreal")

    def test_conversation_helper_builds_messages(self):
        conv = Conversation(system="You are helpful.")
        conv.user("Hello")
        first = call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
        conv.assistant(first.response)
        conv.prefill("{")
        conv.tool_results({"call_1": "ok"})

        self.assertEqual(conv.system, "You are helpful.")
        self.assertEqual(conv.messages[0].role, "user")
        self.assertEqual(conv.messages[1].role, "assistant")
        self.assertEqual(conv.messages[2].parts[0].text, "{")
        self.assertEqual(conv.messages[3].role, "tool")

    def test_model_copy_preserves_or_resets_history(self):
        agent = model("gpt-4.1-mini")
        self.assertEqual(agent.call("hello").text, "Echo: hello")

        fork = agent.copy()
        fresh = agent.copy(history=False)

        self.assertEqual(len(agent.history), 1)
        self.assertEqual(len(fork.history), 1)
        self.assertEqual(len(fresh.history), 0)

        self.assertEqual(fork.call("again").text, "Echo: again")
        self.assertEqual(len(agent.history), 1)
        self.assertEqual(len(fork.history), 2)


class AsyncAPIV3Tests(_APIV3Base, unittest.IsolatedAsyncioTestCase):
    async def test_await_acall_returns_completed_result(self):
        resp = await acall("gpt-4.1-mini", "hello")
        self.assertIsInstance(resp, Result)
        self.assertEqual(resp.text, "Echo: hello")

    async def test_async_for_streams_text(self):
        chunks: list[str] = []
        async for text in acall("gpt-4.1-mini", "hello"):
            chunks.append(text)
        self.assertEqual("".join(chunks), "Echo: hello")
