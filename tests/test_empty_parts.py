"""MAP-24 — a part with nothing for a wire writes nothing
(lm15-contract changes/2026-11-11-empty-parts.md, issue #3).

The contract's vectors (mapping/empty-parts.json) and the history replay
pin the bytes for the four dialects.  This file holds what they do not:
nothing is recorded, the caller's request is not touched, the state comes
back on its own dialect, and the doors beside `build_request` (Gemini's
cached prefix and live turns) follow the same rule.
"""
from __future__ import annotations

import json

import pytest

from lm15 import (
    AnthropicLM,
    CacheConfig,
    Config,
    GeminiLM,
    Message,
    OpenAIChatLM,
    OpenAILM,
    Request,
)
from lm15.providers import AsyncAnthropicLM, AsyncGeminiLM, AsyncOpenAIChatLM, AsyncOpenAILM
from lm15.providers.common import is_empty_text_block, parts_to_text
from lm15.types import (
    CitationPart,
    ContinuationState,
    ImagePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)

# What lm15 reads from a Responses reasoning item with no summary (MAP-7
# rules 9 and 11): hidden thinking for the OpenAI dialect.
HIDDEN = ThinkingPart(text="", continuation=(ContinuationState(
    provider="openai", kind="reasoning_item", data={"id": "rs_1", "encrypted_content": "gAAAA"}),))
CALL = ToolCallPart(id="call_1", name="delete_program", input={})
RESULT = Message(role="tool", parts=(ToolResultPart(id="call_1", name="delete_program", content=(TextPart(text="ok"),)),))


def report(model: str) -> Request:
    """The history of issue #3: two steps answered by an OpenAI Responses
    model with hidden reasoning, then a new user message."""
    return Request(model=model, messages=(
        Message.user("Delete the test program now."),
        Message(role="assistant", parts=(HIDDEN, CALL)),
        RESULT,
        Message(role="assistant", parts=(HIDDEN, TextPart(text="Deleted."))),
        Message.user("hello"),
    ))


ADAPTERS = [
    (AnthropicLM, "claude-opus-5-5", "messages"),
    (GeminiLM, "gemini-2.5-flash", "contents"),
    (OpenAIChatLM, "gpt-5.4", "messages"),
]


def body(lm, request: Request) -> dict:
    return json.loads(lm.build_request(request, stream=False).body)


def test_the_report_builds_a_body_claude_accepts() -> None:
    wire = body(AnthropicLM(api_key="k"), report("claude-opus-5-5"))["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user", "assistant", "user"]
    assert wire[1]["content"] == [{"type": "tool_use", "id": "call_1", "name": "delete_program", "input": {}}]
    assert wire[3]["content"] == [{"type": "text", "text": "Deleted."}]
    assert '"text": ""' not in json.dumps(wire)


@pytest.mark.parametrize("adapter, model, key", ADAPTERS)
def test_nothing_is_recorded_under_any_policy(adapter, model, key) -> None:
    # MAP-24 rule 5: this is translation, not an adaptation.
    request = report(model)
    for policy in ("note", "silent", "refuse"):
        lm = adapter(api_key="k", adaptations=policy)
        assert [a.field for a in lm.plan(request) if a.field.startswith("messages")] == []
        lm.build_request(request, stream=False)  # "refuse" raises on any recorded deviation


@pytest.mark.parametrize("adapter, model, key", ADAPTERS)
def test_the_callers_history_is_not_touched(adapter, model, key) -> None:
    request = report(model)
    before = request.messages
    adapter(api_key="k").build_request(request, stream=False)
    assert request.messages is before
    assert request.messages[1].parts[0] is HIDDEN and request.messages[3].parts[0] is HIDDEN


def test_the_state_goes_back_natively_on_its_own_dialect() -> None:
    # The same history, sent to Claude and then to the Responses dialect
    # again: the reasoning items are still there, `summary` present and empty.
    request = report("gpt-5.4")
    body(AnthropicLM(api_key="k"), report("claude-opus-5-5"))
    items = body(OpenAILM(api_key="k"), request)["input"]
    reasoning = [i for i in items if i.get("type") == "reasoning"]
    assert reasoning == [{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA", "summary": []}] * 2


def test_an_empty_thinking_part_with_claudes_signature_is_still_a_thinking_block() -> None:
    # Claude's thinking with display omitted (MAP-7 rule 7): empty text AND state this dialect consumes.
    omitted = ThinkingPart(text="", continuation=(ContinuationState(
        provider="anthropic", kind="thinking_signature", data={"signature": "sig"}),))
    request = Request(model="claude-opus-5-5", messages=(
        Message.user("hi"), Message(role="assistant", parts=(omitted, CALL)), RESULT))
    assert body(AnthropicLM(api_key="k"), request)["messages"][1]["content"][0] == {
        "type": "thinking", "thinking": "", "signature": "sig"}


def test_a_signature_with_no_value_is_not_state() -> None:
    # A server that signs nothing sends `signature: ""`; that is no state to replay.
    unsigned = ThinkingPart(text="", continuation=(ContinuationState(
        provider="anthropic", kind="thinking_signature", data={"signature": ""}),))
    request = Request(model="claude-opus-5-5", messages=(
        Message.user("hi"), Message(role="assistant", parts=(unsigned, TextPart(text="Hello."))), Message.user("again")))
    assert body(AnthropicLM(api_key="k"), request)["messages"][1]["content"] == [{"type": "text", "text": "Hello."}]


@pytest.mark.parametrize("asynchronous, sync", [
    (AsyncAnthropicLM, AnthropicLM), (AsyncGeminiLM, GeminiLM), (AsyncOpenAILM, OpenAILM), (AsyncOpenAIChatLM, OpenAIChatLM)])
def test_the_async_adapters_build_with_the_same_code(asynchronous, sync) -> None:
    # The async adapters delegate every build to the sync adapter they wrap.
    assert type(asynchronous(api_key="k")._inner) is sync


def test_geminis_cached_prefix_and_live_turns_leave_the_message_out() -> None:
    lm = GeminiLM(api_key="k")
    prefix = Request(model="gemini-2.5-flash", system=(TextPart(text=""),), messages=(
        Message.user("Here is the manual."), Message(role="assistant", parts=(HIDDEN,)), Message.user("And the appendix.")))
    created = json.loads(lm._cache_create_request(prefix, None, None).body)
    assert [c["role"] for c in created["contents"]] == ["user", "user"]
    assert "systemInstruction" not in created
    turns = lm._live_client_content_payload_from_request(Request(model="gemini-2.5-flash", messages=prefix.messages))
    assert [c["role"] for c in turns[0]["clientContent"]["turns"]] == ["user", "user"]


def test_a_cache_mark_asked_on_a_message_left_out_goes_to_the_one_written_before_it() -> None:
    # Not pinned by the contract (stated in the change entry): the reference's choice on the Messages wire.
    request = Request(model="claude-opus-5-5", config=Config(cache=CacheConfig(prefix_until_index=1)), messages=(
        Message.user("Here is the manual."), Message(role="assistant", parts=(HIDDEN,)), Message.user("And the appendix.")))
    wire = body(AnthropicLM(api_key="k"), request)["messages"]
    assert "cache_control" in wire[0]["content"][-1] and "cache_control" not in wire[1]["content"][-1]


def test_a_tool_result_keeps_its_empty_output_and_drops_an_empty_block_beside_an_image() -> None:
    image = ImagePart(url="https://example.com/receipt.png", media_type="image/png")
    for content, expected in (
        ((TextPart(text=""),), ""),
        ((TextPart(text=""), image), [{"type": "image", "source": {"type": "url", "url": "https://example.com/receipt.png"}}]),
    ):
        request = Request(model="claude-opus-5-5", messages=(
            Message.user("hi"), Message(role="assistant", parts=(CALL,)),
            Message(role="tool", parts=(ToolResultPart(id="call_1", name="delete_program", content=content),))))
        block = body(AnthropicLM(api_key="k"), request)["messages"][2]["content"][0]
        assert block["type"] == "tool_result" and block["content"] == expected


def test_text_joined_for_a_text_only_slot_has_no_line_for_an_empty_part() -> None:
    assert parts_to_text((TextPart(text=""), TextPart(text="a"), TextPart(text=""), TextPart(text="b"))) == "a\nb"
    assert parts_to_text((TextPart(text=""),)) == ""
    assert parts_to_text((CitationPart(url="https://example.com"),)) == "https://example.com"


def test_whitespace_is_content_and_is_written() -> None:
    # Not this rule: text that is only whitespace is what a model wrote.
    request = Request(model="claude-opus-5-5", messages=(
        Message.user("hi"), Message(role="assistant", parts=(TextPart(text="\n"),)), Message.user("again")))
    assert body(AnthropicLM(api_key="k"), request)["messages"][1]["content"] == [{"type": "text", "text": "\n"}]


def test_the_block_test_reads_the_three_typed_spellings_only() -> None:
    assert is_empty_text_block({"type": "text", "text": ""})
    assert is_empty_text_block({"type": "input_text", "text": ""})
    assert is_empty_text_block({"type": "output_text", "text": ""})
    assert not is_empty_text_block({"type": "text", "text": " "})
    assert not is_empty_text_block({"type": "thinking", "thinking": "", "signature": "s"})
    assert not is_empty_text_block({"type": "tool_use", "id": "c", "name": "f", "input": {"text": ""}})
