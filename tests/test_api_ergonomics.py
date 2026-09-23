"""API-review additive set (2026-07-13): tool-result ergonomics, the
tools coercion gap, and errors that state their own cure."""

from __future__ import annotations

import pytest

import lm15
from lm15 import Message, Request
from lm15.errors import InvalidRequestError
from lm15.router import LMRouter, RouterConfig, UnknownModelError
from lm15.types import TextPart, ToolResultPart


# ─── A1: tool-result spellings ───────────────────────────────────────


class TestToolResultErgonomics:
    def test_tool_result_is_top_level(self) -> None:
        assert lm15.tool_result is not None
        assert "tool_result" in lm15.__all__
        part = lm15.tool_result("call_1", "42")
        assert isinstance(part, ToolResultPart)

    def test_message_tool_positional_spelling(self) -> None:
        msg = Message.tool("call_1", "42")
        assert msg.role == "tool"
        assert msg.parts == (ToolResultPart(id="call_1", content=(TextPart("42"),)),)

    def test_message_tool_positional_is_error(self) -> None:
        msg = Message.tool("call_1", "ValueError: boom", is_error=True)
        (part,) = msg.parts
        assert part.is_error is True

    def test_message_tool_dict_spelling_unchanged(self) -> None:
        msg = Message.tool({"call_1": "42"})
        assert msg.parts == (ToolResultPart(id="call_1", content=(TextPart("42"),)),)

    def test_message_tool_bad_call_names_the_shapes(self) -> None:
        # The unguessable-shape finding: wrong spellings must teach.
        with pytest.raises(TypeError) as exc_info:
            Message.tool("call_1")  # id without output
        text = str(exc_info.value)
        assert "Message.tool" in text
        assert "{call_id: output}" in text or "call_id" in text

    def test_message_tool_dict_rejects_is_error(self) -> None:
        # The dict form cannot say WHICH result errored; require the
        # positional or part forms for errors.
        with pytest.raises(TypeError, match="is_error"):
            Message.tool({"call_1": "boom"}, is_error=True)


# ─── A2: Request.tools bare coercion ─────────────────────────────────


class TestToolsCoercion:
    def test_bare_function_tool_is_coerced(self) -> None:
        weather = lm15.FunctionTool(name="get_weather", description="Weather.")
        req = Request(model="m", messages=Message.user("hi"), tools=weather)
        assert req.tools == (weather,)


# ─── A3: errors state their cure ─────────────────────────────────────


class TestErrorCures:
    def test_messages_type_error_names_the_fix(self) -> None:
        with pytest.raises(TypeError, match=r"Message\.user"):
            Request(model="m", messages=["hello"])

    def test_provider_error_str_carries_provider_and_status(self) -> None:
        err = InvalidRequestError(
            "model: claude-instant-99", provider="anthropic", status=404
        )
        text = str(err)
        assert "anthropic" in text
        assert "404" in text
        assert err.message == "model: claude-instant-99"  # field untouched

    def test_unknown_model_prefix_example_is_neutral(self) -> None:
        # The old message suggested "anthropic:<model>" verbatim — a
        # copy-pasteable pairing that could 404 (e.g. a llama model).
        with pytest.raises(UnknownModelError) as exc_info:
            LMRouter(RouterConfig(env={})).resolve("mystery-model-9000")
        text = str(exc_info.value)
        assert "anthropic:mystery-model-9000" not in text
        assert "provider:mystery-model-9000" in text or "PROVIDER" in text
