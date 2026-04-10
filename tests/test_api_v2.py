from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.api import call, configure, model, prepare, send, upload
from lm15.client import UniversalLM
from lm15.errors import RateLimitError
from lm15.features import EndpointSupport, ProviderManifest
from lm15.model import Model
from lm15.protocols import Capabilities
from lm15.result import response_to_events
from lm15.types import DataSource, FileUploadRequest, FileUploadResponse, LMRequest, LMResponse, Message, Part, PartDelta, StreamEvent, Tool, Usage


class FakeAdapter:
    provider = "openai"
    capabilities = Capabilities()
    supports = EndpointSupport(complete=True, stream=True, files=True)
    manifest = ProviderManifest(provider="openai", supports=supports)

    def _response(self, request: LMRequest) -> LMResponse:
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

    def complete(self, request: LMRequest) -> LMResponse:
        return self._response(request)

    def stream(self, request: LMRequest):
        yield from response_to_events(self._response(request), request)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        return FileUploadResponse(id="file_123")


class ErrorStreamAdapter(FakeAdapter):
    provider = "anthropic"
    manifest = ProviderManifest(provider="anthropic", supports=FakeAdapter.supports)

    def stream(self, request: LMRequest):
        yield StreamEvent(type="start", id="s1", model=request.model)
        yield StreamEvent(
            type="error",
            error={"code": "rate_limit", "provider_code": "rate_limit_error", "message": "Too many requests"},
        )


class APIV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = UniversalLM()
        self.lm.register(FakeAdapter())

        import lm15.api as api

        self._old_build_default = api.build_default
        api.build_default = lambda **_kw: self.lm

    def tearDown(self) -> None:
        import lm15.api as api

        api.build_default = self._old_build_default
        configure()  # clear any defaults set during tests

    def test_call_simple(self):
        resp = call("gpt-4.1-mini", "hello")
        self.assertEqual(resp.text, "Echo: hello")

    def test_model_history_and_stream_response(self):
        gpt = model("gpt-4.1-mini")
        stream_obj = gpt.stream("hi")
        self.assertEqual("".join(stream_obj.text), "Echo: hi")
        self.assertEqual(stream_obj.response.finish_reason, "stop")
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

    def test_stream_error_raises_typed_provider_error(self):
        lm = UniversalLM()
        lm.register(ErrorStreamAdapter())
        m = Model(lm=lm, model="claude-sonnet-4-5", provider="anthropic")

        stream_obj = m.stream("hi")
        with self.assertRaises(RateLimitError) as ctx:
            list(stream_obj.text)
        self.assertIn("provider_code=rate_limit_error", str(ctx.exception))

    def test_stream_tool_call_materializes_in_response(self):
        class ToolStreamAdapter(FakeAdapter):
            def stream(self, request: LMRequest):
                yield StreamEvent(type="start", id="s1", model=request.model)
                yield StreamEvent(type="delta", part_index=0, delta=PartDelta(type="tool_call", input='{"city"'))
                yield StreamEvent(type="delta", part_index=0, delta=PartDelta(type="tool_call", input=':"Montreal"}'))
                yield StreamEvent(type="end", finish_reason="tool_call", usage=Usage(total_tokens=4))

        lm = UniversalLM()
        lm.register(ToolStreamAdapter())
        m = Model(lm=lm, model="gpt-4.1-mini", provider="openai")

        stream_obj = m.stream("weather")
        list(stream_obj)
        resp = stream_obj.response
        self.assertEqual(resp.finish_reason, "tool_call")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].input, {"city": "Montreal"})

    def test_submit_tools_preserves_conversation_context(self):
        class StrictToolAdapter:
            provider = "openai"
            capabilities = Capabilities()
            supports = EndpointSupport(complete=True)
            manifest = ProviderManifest(provider="openai", supports=supports)

            def complete(self, request: LMRequest) -> LMResponse:
                last = request.messages[-1]
                if last.role == "user":
                    return LMResponse(
                        id="r1",
                        model=request.model,
                        message=Message(role="assistant", parts=(Part.tool_call("call_1", "get_weather", {"city": "Montreal"}),)),
                        finish_reason="tool_call",
                        usage=Usage(),
                    )
                prev = request.messages[-2] if len(request.messages) >= 2 else None
                has_prev_tool_call = bool(prev and prev.role == "assistant" and any(p.type == "tool_call" for p in prev.parts))
                text = "ok" if has_prev_tool_call else "missing_prev_tool_call"
                return LMResponse(id="r2", model=request.model, message=Message.assistant(text), finish_reason="stop", usage=Usage())

        lm = UniversalLM()
        lm.register(StrictToolAdapter())
        m = Model(lm=lm, model="gpt-4.1-mini", provider="openai")

        first = m("weather", tools=[Tool(name="get_weather")])
        self.assertEqual(first.finish_reason, "tool_call")
        out = m.submit_tools({"call_1": "22C"})
        self.assertEqual(out.text, "ok")

    def test_configure_sets_defaults(self):
        """configure() sets module-level defaults used by call/model/etc."""
        import lm15.api as api

        # Track what build_default receives
        captured = {}
        original = api.build_default
        def spy(**kw):
            captured.update(kw)
            return self.lm
        api.build_default = spy

        # Without configure: env and api_key are None
        call("gpt-4.1-mini", "hello")
        self.assertIsNone(captured.get("env"))
        self.assertIsNone(captured.get("api_key"))

        # With configure: defaults flow through
        configure(env=".env", api_key="sk-test")
        call("gpt-4.1-mini", "hello")
        self.assertEqual(captured["env"], ".env")
        self.assertEqual(captured["api_key"], "sk-test")

        # Per-call override wins
        call("gpt-4.1-mini", "hello", env="other.env")
        self.assertEqual(captured["env"], "other.env")
        self.assertEqual(captured["api_key"], "sk-test")  # still from configure

        # Clear defaults
        configure()
        call("gpt-4.1-mini", "hello")
        self.assertIsNone(captured.get("env"))
        self.assertIsNone(captured.get("api_key"))

        api.build_default = original

    def test_model_retries_transient_errors(self):
        class FlakyAdapter:
            provider = "openai"
            capabilities = Capabilities()
            supports = EndpointSupport(complete=True)
            manifest = ProviderManifest(provider="openai", supports=supports)

            def __init__(self) -> None:
                self.calls = 0

            def complete(self, request: LMRequest) -> LMResponse:
                self.calls += 1
                if self.calls < 3:
                    raise RateLimitError("retry me")
                return LMResponse(id="ok", model=request.model, message=Message.assistant("ok"), finish_reason="stop", usage=Usage())

        lm = UniversalLM()
        adapter = FlakyAdapter()
        lm.register(adapter)

        m = Model(lm=lm, model="gpt-4.1-mini", provider="openai", retries=2)
        resp = m("hello")
        self.assertEqual(resp.text, "ok")
        self.assertEqual(adapter.calls, 3)


    def test_prepare_returns_request_without_sending(self):
        """prepare() builds the LMRequest without making an API call."""
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"22C in {city}"

        req = prepare("gpt-4.1-mini", "What is the weather?",
                      system="Be concise.", tools=[get_weather], temperature=0)

        # Returns an LMRequest, not an LMResponse
        self.assertIsInstance(req, LMRequest)

        # Model and system are set
        self.assertEqual(req.model, "gpt-4.1-mini")
        self.assertEqual(req.system, "Be concise.")

        # Messages are constructed
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, "user")
        self.assertEqual(req.messages[0].parts[0].text, "What is the weather?")

        # Tools are inferred from the callable
        self.assertEqual(len(req.tools), 1)
        self.assertEqual(req.tools[0].name, "get_weather")
        self.assertEqual(req.tools[0].description, "Get the current weather for a city.")
        self.assertIn("city", req.tools[0].parameters["properties"])

        # Config is set
        self.assertEqual(req.config.temperature, 0)

    def test_prepare_then_send(self):
        """prepare() + send() produces the same result as call()."""
        req = prepare("gpt-4.1-mini", "hello")
        resp = send(req)
        self.assertEqual(resp.text, "Echo: hello")
        self.assertEqual(resp.finish_reason, "stop")

    def test_model_prepare_method(self):
        """Model.prepare() builds a request using bound config."""
        gpt = model("gpt-4.1-mini", system="You are terse.", temperature=0.5)
        req = gpt.prepare("Hello.")

        self.assertIsInstance(req, LMRequest)
        self.assertEqual(req.system, "You are terse.")
        self.assertEqual(req.config.temperature, 0.5)
        self.assertEqual(req.messages[0].parts[0].text, "Hello.")


class TestResponseConvenience(unittest.TestCase):
    """Tests for response convenience properties: .json, .image_bytes, .audio_bytes, Part.bytes."""

    def test_json_parses_text(self):
        msg = Message(role="assistant", parts=(Part.text_part('{"name": "Alice", "age": 30}'),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        data = resp.json
        self.assertEqual(data, {"name": "Alice", "age": 30})

    def test_json_raises_on_invalid(self):
        msg = Message(role="assistant", parts=(Part.text_part("not json at all"),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        with self.assertRaises(ValueError) as ctx:
            resp.json
        self.assertIn("Cannot parse response as JSON", str(ctx.exception))
        self.assertIn("not json at all", str(ctx.exception))

    def test_json_raises_on_no_text(self):
        msg = Message(role="assistant", parts=(Part.tool_call("c1", "fn", {}),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="tool_call", usage=Usage())
        with self.assertRaises(ValueError) as ctx:
            resp.json
        self.assertIn("no text", str(ctx.exception))

    def test_json_with_prefill(self):
        """JSON that starts with { from a prefilled response."""
        msg = Message(role="assistant", parts=(Part.text_part('{"label": "POSITIVE"}'),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        self.assertEqual(resp.json["label"], "POSITIVE")

    def test_json_returns_list(self):
        msg = Message(role="assistant", parts=(Part.text_part('[1, 2, 3]'),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        self.assertEqual(resp.json, [1, 2, 3])

    def test_datasource_bytes(self):
        import base64
        raw = b"hello world"
        encoded = base64.b64encode(raw).decode("ascii")
        ds = DataSource(type="base64", data=encoded, media_type="application/octet-stream")
        self.assertEqual(ds.bytes, raw)

    def test_datasource_bytes_raises_for_url(self):
        ds = DataSource(type="url", url="https://example.com/img.png")
        with self.assertRaises(ValueError) as ctx:
            ds.bytes
        self.assertIn("url", str(ctx.exception).lower())

    def test_part_bytes_for_image(self):
        import base64
        raw = b"\x89PNG\r\n"
        encoded = base64.b64encode(raw).decode("ascii")
        part = Part.image(data=raw, media_type="image/png")
        self.assertEqual(part.bytes, raw)

    def test_part_bytes_raises_for_text(self):
        part = Part.text_part("hello")
        with self.assertRaises(TypeError) as ctx:
            part.bytes
        self.assertIn("not a media part", str(ctx.exception))

    def test_image_bytes_on_response(self):
        import base64
        raw = b"\x89PNG image data"
        encoded = base64.b64encode(raw).decode("ascii")
        img_part = Part(type="image", source=DataSource(type="base64", data=encoded, media_type="image/png"))
        msg = Message(role="assistant", parts=(Part.text_part("Here's the image."), img_part))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        self.assertEqual(resp.image_bytes, raw)

    def test_image_bytes_raises_when_no_image(self):
        msg = Message(role="assistant", parts=(Part.text_part("Just text."),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        with self.assertRaises(ValueError) as ctx:
            resp.image_bytes
        self.assertIn("no image", str(ctx.exception).lower())

    def test_audio_bytes_on_response(self):
        import base64
        raw = b"RIFF audio data"
        encoded = base64.b64encode(raw).decode("ascii")
        aud_part = Part(type="audio", source=DataSource(type="base64", data=encoded, media_type="audio/wav"))
        msg = Message(role="assistant", parts=(aud_part,))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        self.assertEqual(resp.audio_bytes, raw)

    def test_audio_bytes_raises_when_no_audio(self):
        msg = Message(role="assistant", parts=(Part.text_part("No audio here."),))
        resp = LMResponse(id="r1", model="m", message=msg, finish_reason="stop", usage=Usage())
        with self.assertRaises(ValueError) as ctx:
            resp.audio_bytes
        self.assertIn("no audio", str(ctx.exception).lower())


class TestInstructionalErrors(unittest.TestCase):
    """Verify that error messages include actionable guidance."""

    def test_unsupported_model_error_has_instructions(self):
        from lm15.errors import UnsupportedModelError
        from lm15.capabilities import resolve_provider
        with self.assertRaises(UnsupportedModelError) as ctx:
            resolve_provider("nonexistent-model-xyz")
        msg = str(ctx.exception)
        self.assertIn("provider=", msg)
        self.assertIn("lm15.models()", msg)
        self.assertIn("gpt-", msg)

    def test_no_adapter_error_has_instructions(self):
        from lm15.errors import ProviderError
        client = UniversalLM()
        req = LMRequest(model="gpt-4.1-mini", messages=(Message.user("Hi"),))
        with self.assertRaises(ProviderError) as ctx:
            client.complete(req)
        msg = str(ctx.exception)
        self.assertIn("To fix", msg)
        self.assertIn("API key", msg)  # present in guidance
        self.assertIn(".env", msg)

    def test_auth_error_has_instructions(self):
        from lm15.errors import AuthError
        exc = AuthError("Invalid API key")
        msg = str(exc)
        self.assertIn("To fix", msg)
        self.assertIn("api_key=", msg)
        self.assertIn("configure", msg)

    def test_rate_limit_error_has_instructions(self):
        exc = RateLimitError("Too many requests")
        msg = str(exc)
        self.assertIn("To fix", msg)
        self.assertIn("retries=", msg)

    def test_context_length_error_has_instructions(self):
        from lm15.errors import ContextLengthError
        exc = ContextLengthError("Input too long")
        msg = str(exc)
        self.assertIn("To fix", msg)
        self.assertIn("history.clear()", msg)
        self.assertIn("max_tokens", msg)

    def test_submit_tools_no_pending_has_instructions(self):
        adapter = FakeAdapter()
        client = UniversalLM()
        client.register(adapter)
        m = Model(lm=client, model="gpt-4.1-mini")
        with self.assertRaises(ValueError) as ctx:
            m.submit_tools({"id": "result"})
        msg = str(ctx.exception)
        self.assertIn("finish_reason", msg)
        self.assertIn("tool_call", msg)

    def test_prompt_and_messages_exclusive_has_instructions(self):
        adapter = FakeAdapter()
        client = UniversalLM()
        client.register(adapter)
        m = Model(lm=client, model="gpt-4.1-mini")
        with self.assertRaises(ValueError) as ctx:
            m("hello", messages=[Message.user("hi")])
        msg = str(ctx.exception)
        self.assertIn("mutually exclusive", msg)
        self.assertIn("prompt=", msg.lower() or msg)
        self.assertIn("messages=", msg.lower() or msg)


class TestCallRetries(unittest.TestCase):
    """Tests that call() and stream() accept retries=."""

    def setUp(self) -> None:
        import lm15.api as api
        self._old_build_default = api.build_default
        api.build_default = lambda **kw: self._make_lm()
        api._client_cache.clear()

    def tearDown(self) -> None:
        import lm15.api as api
        api.build_default = self._old_build_default
        api._client_cache.clear()
        configure()

    def _make_lm(self):
        lm = UniversalLM()
        lm.register(FakeAdapter())
        return lm

    def test_call_accepts_retries(self):
        """call() passes retries through to the internal model."""
        resp = call("gpt-4.1-mini", "hello", retries=3)
        self.assertEqual(resp.text, "Echo: hello")

    def test_call_with_tools_and_retries(self):
        """call() with tools and retries works."""
        def get_weather(city: str) -> str:
            return f"22C in {city}"

        resp = call("gpt-4.1-mini", "what is the weather", tools=[get_weather], retries=2)
        self.assertIn("22C", resp.text or "")


class TestClientCaching(unittest.TestCase):
    """Tests that build_default is cached for repeated calls."""

    def setUp(self) -> None:
        import lm15.api as api
        self.build_count = 0
        self._old_build_default = api.build_default

        def counting_build(**kw):
            self.build_count += 1
            lm = UniversalLM()
            lm.register(FakeAdapter())
            return lm

        api.build_default = counting_build
        api._client_cache.clear()

    def tearDown(self) -> None:
        import lm15.api as api
        api.build_default = self._old_build_default
        api._client_cache.clear()
        configure()

    def test_repeated_calls_reuse_client(self):
        """Multiple call() invocations with same args reuse the client."""
        call("gpt-4.1-mini", "first")
        call("gpt-4.1-mini", "second")
        call("gpt-4.1-mini", "third")
        self.assertEqual(self.build_count, 1)  # built once, reused

    def test_configure_clears_cache(self):
        """configure() clears the client cache."""
        call("gpt-4.1-mini", "first")
        self.assertEqual(self.build_count, 1)
        configure()  # clears cache
        call("gpt-4.1-mini", "second")
        self.assertEqual(self.build_count, 2)  # rebuilt


class TestToolControlSeparation(unittest.TestCase):
    """Tests that tool format (inferred vs explicit) is independent of control (auto vs manual)."""

    def setUp(self) -> None:
        self.lm = UniversalLM()
        self.lm.register(FakeAdapter())

    def test_bare_callable_auto_executes(self):
        """Bare callable: inferred schema + auto-execute (existing behavior)."""
        def get_weather(city: str) -> str:
            return f"22C in {city}"

        gpt = Model(lm=self.lm, model="gpt-4.1-mini")
        resp = gpt("what is the weather", tools=[get_weather])
        self.assertIn("22C", resp.text or "")

    def test_tool_from_fn_manual(self):
        """Tool.from_fn(): inferred schema + manual (no auto-execute)."""
        def get_weather(city: str) -> str:
            return f"22C in {city}"

        tool = Tool.from_fn(get_weather)
        self.assertEqual(tool.name, "get_weather")
        self.assertIn("city", tool.parameters["properties"])
        self.assertIsNone(tool.fn)  # no fn = manual

        gpt = Model(lm=self.lm, model="gpt-4.1-mini")
        resp = gpt("what is the weather", tools=[tool])
        # Manual: model requested tool_call but it wasn't auto-executed
        self.assertEqual(resp.finish_reason, "tool_call")
        self.assertTrue(len(resp.tool_calls) > 0)

    def test_tool_with_fn_auto_executes(self):
        """Tool(fn=callable): explicit schema + auto-execute."""
        def weather_impl(city: str) -> str:
            return f"22C in {city}"

        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            fn=weather_impl,
        )
        gpt = Model(lm=self.lm, model="gpt-4.1-mini")
        resp = gpt("what is the weather", tools=[tool])
        # Auto-executed because fn is set
        self.assertIn("22C", resp.text or "")

    def test_tool_without_fn_manual(self):
        """Tool(no fn): explicit schema + manual (existing behavior)."""
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        )
        gpt = Model(lm=self.lm, model="gpt-4.1-mini")
        resp = gpt("what is the weather", tools=[tool])
        self.assertEqual(resp.finish_reason, "tool_call")

    def test_from_fn_inspect_schema(self):
        """Tool.from_fn() allows inspecting the inferred schema."""
        def search(query: str, max_results: int = 5) -> str:
            """Search for documents matching the query."""
            return "results"

        tool = Tool.from_fn(search)
        self.assertEqual(tool.name, "search")
        self.assertEqual(tool.description, "Search for documents matching the query.")
        self.assertIn("query", tool.parameters["properties"])
        self.assertIn("max_results", tool.parameters["properties"])
        self.assertEqual(tool.parameters["properties"]["query"], {"type": "string"})
        self.assertEqual(tool.parameters["properties"]["max_results"], {"type": "integer"})
        self.assertEqual(tool.parameters["required"], ["query"])

    def test_callable_to_tool_public_export(self):
        """callable_to_tool is publicly importable."""
        from lm15 import callable_to_tool

        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        tool = callable_to_tool(greet)
        self.assertEqual(tool.name, "greet")
        self.assertEqual(tool.description, "Greet someone.")


if __name__ == "__main__":
    unittest.main()
