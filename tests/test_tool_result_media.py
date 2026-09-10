"""MAP-10: every part in a tool result reaches the wire natively or raises.

The matrix the corpus lacked on 2026-09-07 (297 canonical requests, 17 with a
tool result, none with media in one). One request shape, four dialects,
every preset value; the expected wires are the ones the live matrix proved
(lm15-contract/research/tool-result-content/20-results.md).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from lm15 import (
    AnthropicLM, DocumentPart, GeminiLM, ImagePart, Message, OpenAIChatLM, OpenAILM, Request, TextPart,
    ToolCallPart, ToolResultPart, tool_result,
)
from lm15.compat import AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat
from lm15.errors import UnsupportedFeatureError
from lm15.registry import PROVIDERS
from lm15.router import LMRouter, RouterConfig

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
PDF = base64.b64encode(b"%PDF-1.4 fake").decode()
IMAGE = ImagePart(media_type="image/png", data=PNG)
DOC = DocumentPart(media_type="application/pdf", data=PDF)
CALL = ToolCallPart(id="call_1", name="fetch_panel", input={"label": "A"})


def request(*results: ToolResultPart, model: str = "m") -> Request:
    return Request(model=model, messages=(Message.user("go"), Message.assistant((CALL,)), Message.tool(results)))


def body(lm, req: Request) -> dict:
    return json.loads(lm.build_request(req, stream=False).body)


# ─── OpenAI Responses ───────────────────────────────────────────────

def test_responses_text_only_stays_a_string():
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", "sunny")))
    assert out["input"][-1] == {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}


def test_responses_image_only_and_mixed_become_the_documented_array():
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", IMAGE)))
    assert out["input"][-1]["output"] == [{"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"}]
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", (TextPart(text="panel"), IMAGE, DOC))))
    assert [b["type"] for b in out["input"][-1]["output"]] == ["input_text", "input_image", "input_file"]
    assert out["input"][-1]["output"][2]["filename"] == "file.pdf"


def test_responses_image_detail_rides_along():
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", ImagePart(media_type="image/png", data=PNG, detail="high"))))
    assert out["input"][-1]["output"][0]["detail"] == "high"


def test_responses_path_is_read_not_sent_as_empty_text(tmp_path: Path):
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", ImagePart(media_type="image/png", path=path))))
    assert out["input"][-1]["output"][0]["image_url"] == f"data:image/png;base64,{PNG}"
    # the same rule in a user message (the 2026-09-07 review found an empty input_text here)
    req = Request(model="m", messages=(Message.user((ImagePart(media_type="image/png", path=path),)),))
    assert body(OpenAILM(api_key="k"), req)["input"][0]["content"][0]["type"] == "input_image"


def test_responses_is_error_rides_as_a_text_prefix():
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", "boom", is_error=True)))
    assert out["input"][-1]["output"] == "[error] boom"
    out = body(OpenAILM(api_key="k"), request(tool_result("call_1", IMAGE, is_error=True)))
    assert out["input"][-1]["output"][0] == {"type": "input_text", "text": "[error]"}


def test_responses_two_results_keep_their_call_ids_and_order():
    call_b = ToolCallPart(id="call_2", name="fetch_panel", input={"label": "B"})
    req = Request(model="m", messages=(Message.user("go"), Message.assistant((CALL, call_b)),
                                        Message.tool((tool_result("call_2", (TextPart(text="B"), IMAGE)), tool_result("call_1", IMAGE)))))
    out = body(OpenAILM(api_key="k"), req)
    assert [(i["call_id"], [b["type"] for b in i["output"]]) for i in out["input"][-2:]] == [
        ("call_2", ["input_text", "input_image"]), ("call_1", ["input_image"])]


@pytest.mark.parametrize("policy,part,ok", [
    ("native", IMAGE, True), ("native", DOC, True),
    ("images", IMAGE, True), ("images", DOC, False),
    ("reject", IMAGE, False), ("reject", DOC, False),
])
def test_responses_policy_matrix(policy, part, ok):
    lm = OpenAILM(api_key="k", compat=OpenAIResponsesCompat(tool_result_media=policy))
    if ok:
        assert isinstance(body(lm, request(tool_result("call_1", part)))["input"][-1]["output"], list)
    else:
        with pytest.raises(UnsupportedFeatureError, match=f"{part.type} part in tool_result 'call_1'.*tool_result_media"):
            lm.build_request(request(tool_result("call_1", part)), stream=False)


def test_responses_stop_and_top_k_raise_instead_of_vanishing():
    from lm15 import Config
    for cfg in (Config(stop=("END",)), Config(top_k=5)):
        with pytest.raises(UnsupportedFeatureError, match="no field on the Responses wire"):
            OpenAILM(api_key="k").build_request(Request(model="m", messages=(Message.user("x"),), config=cfg), stream=False)


# ─── OpenAI Chat Completions ────────────────────────────────────────

def test_chat_default_preset_rejects_media_and_names_the_door():
    with pytest.raises(UnsupportedFeatureError) as err:
        OpenAIChatLM(api_key="k").build_request(request(tool_result("call_1", IMAGE)), stream=False)
    assert "text-only tool results" in str(err.value) and "Responses" in str(err.value)


def test_chat_native_preset_sends_the_content_array_and_text_stays_a_string():
    lm = OpenAIChatLM(api_key="k", compat=OpenAIChatCompat(tool_result_media="images"))
    out = body(lm, request(tool_result("call_1", "sunny")))
    assert out["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    out = body(lm, request(tool_result("call_1", (TextPart(text="panel"), IMAGE))))
    assert out["messages"][-1]["content"] == [{"type": "text", "text": "panel"},
                                              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]
    with pytest.raises(UnsupportedFeatureError, match="carries images but not document"):
        lm.build_request(request(tool_result("call_1", DOC)), stream=False)


def test_chat_no_placeholder_ever():
    lm = OpenAIChatLM(api_key="k", compat=OpenAIChatCompat(tool_result_media="native"))
    out = body(lm, request(tool_result("call_1", IMAGE)))
    assert '[{"type": "image"}]' not in json.dumps(out)


def test_chat_is_error_prefix():
    lm = OpenAIChatLM(api_key="k", compat=OpenAIChatCompat(tool_result_media="images"))
    assert body(lm, request(tool_result("call_1", "boom", is_error=True)))["messages"][-1]["content"] == "[error] boom"


def test_chat_user_media_without_a_slot_raises_and_top_k_raises():
    from lm15 import AudioPart, Config
    req = Request(model="m", messages=(Message.user((AudioPart(media_type="audio/wav", data=PNG),)),))
    with pytest.raises(UnsupportedFeatureError, match="audio part in a user message has no slot"):
        OpenAIChatLM(api_key="k").build_request(req, stream=False)
    with pytest.raises(UnsupportedFeatureError, match="top_k has no field"):
        OpenAIChatLM(api_key="k").build_request(Request(model="m", messages=(Message.user("x"),), config=Config(top_k=3)), stream=False)


@pytest.mark.parametrize("provider,expected", [
    ("xai", "images"), ("moonshotai", "images"), ("zai", "images"),
    ("groq", "reject"), ("deepseek", "reject"), ("meta-chat", "reject"), ("openai-chat", "reject"),
    ("bedrock-chat", "reject"), ("ollama", "reject"), ("openrouter", "reject"),
    ("openai", "native"), ("meta", "native"), ("moonshotai-responses", "images"),
    ("anthropic", "native"), ("deepseek-anthropic", "reject"), ("moonshotai-anthropic", "images"), ("meta-anthropic", "native"),
])
def test_registry_presets_carry_the_measured_verdicts(provider, expected):
    """Through the router, the way a user reaches the binding (adapter-owned
    entries such as XaiLM bind their preset in the constructor)."""
    model = {"deepseek-anthropic": "deepseek-v4-flash", "moonshotai-anthropic": "kimi-k3"}.get(provider, "m")
    # The test must not depend on a developer's ~/.aws/config for its region.
    settings = {provider: {"region": "us-east-1"}} if provider == "bedrock-chat" else None
    lm = LMRouter(RouterConfig(api_keys={provider: "k"}, env={}, settings=settings)).lm(f"{provider}:{model}")
    req = request(tool_result("call_1", IMAGE), model=model)
    if expected == "reject":
        with pytest.raises(UnsupportedFeatureError, match="tool_result_media='reject'"):
            lm.build_request(req, stream=False)
    else:
        wire = body(lm, req)
        item = (wire.get("input") or wire.get("messages") or wire.get("contents"))[-1]
        assert isinstance(item.get("output") or item.get("content"), list)


def test_router_bound_preset_is_what_build_request_uses():
    lm = LMRouter(RouterConfig(api_keys={"xai": "k", "groq": "k"}, env={})).lm("xai:grok-4.20")
    assert isinstance(body(lm, request(tool_result("call_1", IMAGE)))["messages"][-1]["content"], list)
    with pytest.raises(UnsupportedFeatureError, match="groq"):
        LMRouter(RouterConfig(api_keys={"xai": "k", "groq": "k"}, env={})).lm("groq:qwen/qwen3.8-27b").build_request(
            request(tool_result("call_1", IMAGE)), stream=False)


# ─── Anthropic Messages ─────────────────────────────────────────────

def test_anthropic_blocks_error_flag_and_policy():
    out = body(AnthropicLM(api_key="k"), request(tool_result("call_1", (TextPart(text="panel"), IMAGE, DOC), is_error=True)))
    block = out["messages"][-1]["content"][0]
    assert block["tool_use_id"] == "call_1" and block["is_error"] is True
    assert [b["type"] for b in block["content"]] == ["text", "image", "document"]
    with pytest.raises(UnsupportedFeatureError, match="text-only tool results"):
        AnthropicLM(api_key="k", compat=AnthropicCompat(tool_result_media="reject")).build_request(
            request(tool_result("call_1", IMAGE)), stream=False)


# ─── Gemini ─────────────────────────────────────────────────────────

def test_gemini_nests_media_under_the_function_response():
    out = body(GeminiLM(api_key="k"), request(tool_result("call_1", (TextPart(text="panel"), IMAGE, DOC))))
    fr = out["contents"][-1]["parts"][0]["functionResponse"]
    assert fr["name"] == "fetch_panel" and fr["id"] == "call_1" and fr["response"] == {"result": "panel"}
    assert [list(p)[0] for p in fr["parts"]] == ["inlineData", "inlineData"]
    assert fr["parts"][1]["inlineData"]["mimeType"] == "application/pdf"


def test_gemini_error_key_and_name_resolution():
    fr = body(GeminiLM(api_key="k"), request(tool_result("call_1", "boom", is_error=True)))["contents"][-1]["parts"][0]["functionResponse"]
    assert fr["response"] == {"error": "boom"}
    # no call in the transcript and no name: raise, never "tool"
    req = Request(model="m", messages=(Message.user("go"), Message.tool((tool_result("orphan", "x"),))))
    with pytest.raises(UnsupportedFeatureError, match="needs a function name"):
        GeminiLM(api_key="k").build_request(req, stream=False)
    req = Request(model="m", messages=(Message.user("go"), Message.tool((tool_result("orphan", "x", name="named"),))))
    assert body(GeminiLM(api_key="k"), req)["contents"][-1]["parts"][0]["functionResponse"]["name"] == "named"


def test_gemini_audio_in_a_tool_result_raises():
    from lm15 import AudioPart
    with pytest.raises(UnsupportedFeatureError, match="images .* and documents"):
        GeminiLM(api_key="k").build_request(request(tool_result("call_1", AudioPart(media_type="audio/wav", data=PNG))), stream=False)


# ─── the rule behind all of it ───────────────────────────────────────

def test_parts_to_text_never_renders_media():
    from lm15.providers.common import parts_to_text
    assert parts_to_text((TextPart(text="a"), TextPart(text="b"))) == "a\nb"
    with pytest.raises(UnsupportedFeatureError, match="image part cannot reach"):
        parts_to_text((TextPart(text="a"), IMAGE), provider="p")
