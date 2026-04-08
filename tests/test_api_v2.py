from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.api import complete, model, upload
from lm15.client import UniversalLM
from lm15.features import EndpointSupport, ProviderManifest
from lm15.protocols import Capabilities
from lm15.types import FileUploadRequest, FileUploadResponse, LMRequest, LMResponse, Message, Part, StreamEvent, Tool, Usage


class FakeAdapter:
    provider = "openai"
    capabilities = Capabilities()
    supports = EndpointSupport(complete=True, stream=True, files=True)
    manifest = ProviderManifest(provider="openai", supports=supports)

    def complete(self, request: LMRequest) -> LMResponse:
        last = request.messages[-1]
        text = " ".join(p.text or "" for p in last.parts if p.type == "text")

        if request.tools and "weather" in text.lower() and not any(m.role == "tool" for m in request.messages):
            msg = Message(role="assistant", parts=(Part.tool_call("call_1", "get_weather", {"city": "Montreal"}),))
            return LMResponse(id="r1", model=request.model, message=msg, finish_reason="tool_call", usage=Usage())

        tool_msgs = [m for m in request.messages if m.role == "tool"]
        if tool_msgs:
            tool_text = "\n".join(
                item.text or ""
                for part in tool_msgs[-1].parts
                for item in part.content
                if item.type == "text"
            )
            msg = Message.assistant(f"Tool says: {tool_text}")
            return LMResponse(id="r2", model=request.model, message=msg, finish_reason="stop", usage=Usage(cache_read_tokens=10))

        return LMResponse(id="r0", model=request.model, message=Message.assistant(f"Echo: {text}"), finish_reason="stop", usage=Usage())

    def stream(self, request: LMRequest):
        yield StreamEvent(type="start", id="s1", model=request.model)
        yield StreamEvent(type="delta", part_index=0, delta={"type": "text", "text": "ok"})
        yield StreamEvent(type="end", finish_reason="stop", usage=Usage(total_tokens=3))

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        return FileUploadResponse(id="file_123")


class APIV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = UniversalLM()
        self.lm.register(FakeAdapter())

        import lm15.api as api

        self._old_build_default = api.build_default
        api.build_default = lambda: self.lm

    def tearDown(self) -> None:
        import lm15.api as api

        api.build_default = self._old_build_default

    def test_complete_simple(self):
        resp = complete("gpt-4.1-mini", "hello")
        self.assertEqual(resp.text, "Echo: hello")

    def test_model_history_and_stream_response(self):
        gpt = model("gpt-4.1-mini")
        stream_obj = gpt.stream("hi")
        self.assertEqual("".join(stream_obj.text), "ok")
        self.assertEqual(stream_obj.response.usage.total_tokens, 3)
        self.assertEqual(len(gpt.history), 1)

    def test_callable_tool_auto_execute(self):
        def get_weather(city: str) -> str:
            return f"22C in {city}"

        gpt = model("gpt-4.1-mini")
        resp = gpt("what is the weather", tools=[get_weather])
        self.assertIn("22C", resp.text or "")

    def test_submit_tools_manual(self):
        gpt = model("gpt-4.1-mini")
        resp = gpt("weather", tools=[Tool(name="get_weather")])
        self.assertEqual(resp.finish_reason, "tool_call")
        out = gpt.submit_tools({"call_1": "22C"})
        self.assertIn("22C", out.text or "")

    def test_upload_returns_part(self):
        p = upload("gpt-4.1-mini", b"abc", media_type="application/pdf")
        self.assertEqual(p.type, "document")
        self.assertEqual(p.source.type, "file")

    def test_part_constructors(self):
        img = Part.image(url="https://x/y.png")
        self.assertEqual(img.type, "image")
        self.assertEqual(img.source.type, "url")

        doc = Part.document(data="YmFzZTY0", media_type="application/pdf", cache=True)
        self.assertEqual(doc.metadata, {"cache": True})


if __name__ == "__main__":
    unittest.main()
