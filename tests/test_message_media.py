"""MAP-10 for message parts: a media part reaches the wire natively or raises
(lm15-contract changes/2026-09-24-message-media.md). Until 2026-09-24 the
cells below were lost silently: an empty text block, a dropped message,
``content: null``."""
from __future__ import annotations

import json

import pytest

from lm15 import AnthropicLM, Message, OpenAIChatLM, OpenAILM, Request, TextPart
from lm15.errors import UnsupportedFeatureError
from lm15.types import AudioPart, BinaryPart, DocumentPart, ImagePart, VideoPart

AUDIO = AudioPart(media_type="audio/wav", data="QUJD")
MEDIA = {
    "image": ImagePart(media_type="image/png", data="QUJD"),
    "audio": AUDIO,
    "video": VideoPart(media_type="video/mp4", url="https://example.com/a.mp4"),
    "document": DocumentPart(media_type="application/pdf", data="QUJD"),
    "binary": BinaryPart(media_type="image/svg+xml", data="QUJD"),
}


def _request(model: str, role: str, part) -> Request:
    if role == "user":
        return Request(model=model, messages=[Message(role="user", parts=[TextPart("q"), part])])
    return Request(model=model, messages=[
        Message.user("q"), Message(role=role, parts=[TextPart("a"), part]), Message.user("more")])


def _body(lm, request: Request) -> str:
    return json.dumps(json.loads(lm.build_request(request, False).body))


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("kind", ["audio", "video", "binary"])
def test_anthropic_has_no_audio_video_or_binary_block(role, kind):
    lm = AnthropicLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError) as info:
        lm.build_request(_request("claude-haiku-4-5", role, MEDIA[kind]), False)
    index = 0 if role == "user" else 1
    assert info.value.feature == f"messages[{index}].parts[1]"


@pytest.mark.parametrize("lm,model", [(OpenAILM(api_key="k"), "gpt-5.6-sol"), (OpenAIChatLM(api_key="k"), "gpt-4.1")])
@pytest.mark.parametrize("kind", sorted(MEDIA))
def test_openai_assistant_content_is_text_only(lm, model, kind):
    with pytest.raises(UnsupportedFeatureError) as info:
        lm.build_request(_request(model, "assistant", MEDIA[kind]), False)
    assert info.value.feature == "messages[1].parts[1]"


@pytest.mark.parametrize("kind", sorted(MEDIA))
def test_responses_user_media_is_sent_natively(kind):
    body = _body(OpenAILM(api_key="k"), _request("gpt-5.6-sol", "user", MEDIA[kind]))
    assert "QUJD" in body or "example.com/a.mp4" in body, body


@pytest.mark.parametrize("kind", ["image", "document"])
def test_anthropic_user_images_and_documents_are_sent(kind):
    assert "QUJD" in _body(AnthropicLM(api_key="k"), _request("claude-haiku-4-5", "user", MEDIA[kind]))
