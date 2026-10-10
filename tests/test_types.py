import base64

import pytest

from lm15.serde import (
    config_from_dict,
    config_to_dict,
    delta_from_dict,
    delta_to_dict,
    message_from_dict,
    message_to_dict,
    part_from_dict,
    part_to_dict,
    tool_to_dict,
)
from lm15.types import (
    AudioDelta,
    AudioPart,
    BatchRequest,
    BinaryPart,
    CitationPart,
    Config,
    ContinuationDelta,
    ContinuationState,
    DocumentPart,
    FileUploadRequest,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImagePart,
    LiveClientAudioEvent,
    LiveClientImageEvent,
    LiveClientToolResultEvent,
    LiveClientTurnEvent,
    LiveServerInterruptedEvent,
    LiveServerToolCallDeltaEvent,
    Message,
    Reasoning,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
    VideoPart,
    continuation_data,
    audio,
    binary,
    document,
    image,
    tool_call,
    tool_result,
    video,
)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def test_media_parts_share_validation_and_byte_access() -> None:
    assert ImagePart(data=_b64(b"image")).bytes == b"image"
    assert AudioPart(data=_b64(b"audio")).bytes == b"audio"
    assert DocumentPart(data=_b64(b"doc")).bytes == b"doc"
    assert (
        ImagePart(data=f"data:image/png;base64,{_b64(b'image')}").bytes
        == b"image"
    )

    with pytest.raises(ValueError, match="requires exactly one"):
        ImagePart(data=_b64(b"image"), url="https://example.com/image.png")


def test_media_reprs_summarize_large_payloads() -> None:
    payload = _b64(b"x" * 128)
    raw = b"y" * 128

    for value in (
        ImagePart(data=payload),
        AudioDelta(data=payload),
        ImageDelta(data=payload),
    ):
        rendered = repr(value)
        assert payload not in rendered
        assert "<base64:" in rendered

    rendered_upload = repr(FileUploadRequest(filename="f.bin", bytes_data=raw))
    assert repr(raw) not in rendered_upload
    assert "<bytes: 128 bytes>" in rendered_upload


def test_media_factories_accept_paths(tmp_path) -> None:
    png = tmp_path / "cat.png"
    wav = tmp_path / "sound.wav"
    mp4 = tmp_path / "clip.mp4"
    pdf = tmp_path / "doc.pdf"
    png.write_bytes(b"image")
    wav.write_bytes(b"audio")
    mp4.write_bytes(b"video")
    pdf.write_bytes(b"doc")

    img = image(path=png)
    assert img.media_type == "image/png"
    assert img.data is None
    assert img.path == png
    assert img.bytes == b"image"
    png.write_bytes(b"updated")
    assert img.bytes == b"updated"
    assert audio(path=wav).bytes == b"audio"
    assert video(path=mp4).bytes == b"video"
    assert document(path=pdf).bytes == b"doc"

    with pytest.raises(ValueError, match="exactly one"):
        image(path=png, data=b"image")


def test_binary_part_covers_arbitrary_tool_result_bytes() -> None:
    blob = binary(data=b"zip", media_type="application/zip")
    result = tool_result("call_1", blob)

    assert isinstance(blob, BinaryPart)
    assert blob.bytes == b"zip"
    assert result.content == (blob,)


def test_message_filters_power_response_helpers() -> None:
    image = ImagePart(data=_b64(b"image"))
    video = VideoPart(data=_b64(b"video"))
    document = DocumentPart(data=_b64(b"doc"))
    citation = CitationPart(url="https://example.com")
    message = Message.assistant(
        [
            TextPart("hello"),
            ThinkingPart("hidden"),
            image,
            video,
            document,
            citation,
        ]
    )
    response = Response(
        id="r1",
        model="m",
        message=message,
        finish_reason="stop",
        usage=Usage(),
    )

    assert message.parts_of(TextPart) == [TextPart("hello")]
    assert message.first(ImagePart) is image
    assert message.parts_of(VideoPart) == [video]
    assert message.parts_of(DocumentPart) == [document]
    assert message.parts_of(CitationPart) == [citation]
    assert response.text is None
    assert response.tool_calls == []
    assert not hasattr(response, "image")


def test_response_text_allows_metadata_parts() -> None:
    citation = CitationPart(url="https://example.com", title="Example")
    thinking = ThinkingPart("summary")
    message = Message.assistant([thinking, TextPart("hello"), citation])
    response = Response(
        id="r1",
        model="m",
        message=message,
        finish_reason="stop",
        usage=Usage(),
    )

    rendered = repr(response)

    assert message.text is None
    assert response.text == "hello"
    assert response.citations == [citation]
    assert response.message.parts_of(ThinkingPart) == [thinking]
    assert "text='hello'" in rendered
    assert "CitationPart" in rendered


def test_response_json_requires_exact_json() -> None:
    response = Response(
        id="r1",
        model="m",
        message=Message.assistant('Here you go:\n```json\n{"a": 1}\n```\nDone.'),
        finish_reason="stop",
        usage=Usage(),
    )

    assert response.json is None


def test_response_json_returns_plain_json() -> None:
    response = Response(
        id="r1",
        model="m",
        message=Message.assistant('{"items": [1]}'),
        finish_reason="stop",
        usage=Usage(),
    )

    parsed = response.json
    parsed["items"].append(2)
    assert parsed == {"items": [1, 2]}
    assert response.json == {"items": [1]}


def test_continuation_state_validates_and_roundtrips() -> None:
    state = ContinuationState(provider="anthropic", kind="thinking_signature", data={"signature": "abc"})
    part = ThinkingPart("reasoning", continuation=(state,))
    msg = Message.assistant([part])

    assert continuation_data(part, "anthropic", "thinking_signature") == {"signature": "abc"}
    assert continuation_data(msg, "anthropic", "thinking_signature") is None
    assert part_from_dict(part_to_dict(part)) == part
    assert message_from_dict(message_to_dict(msg)) == msg

    with pytest.raises(TypeError, match="ContinuationState objects"):
        TextPart("hello", continuation=("bad",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="continuation entries"):
        part_from_dict({"type": "text", "text": "x", "continuation": ["bad"]})


def test_delta_variants_are_proper_unions() -> None:
    delta = TextDelta("hello")

    assert delta.type == "text"
    assert delta.text == "hello"
    assert not hasattr(delta, "data")


def test_delta_text_payloads_must_be_strings() -> None:
    with pytest.raises(TypeError, match="TextDelta.text"):
        TextDelta(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="input"):
        ToolCallDelta(input={})  # type: ignore[arg-type]


def test_image_delta_accepts_partial_chunks_and_metadata() -> None:
    assert ImageDelta().data is None
    assert ImageDelta(data="abc").data == "abc"
    assert ImageDelta(data="").data == ""
    assert ImageDelta(media_type="image/png").media_type == "image/png"

    with pytest.raises(TypeError, match="media_type"):
        ImageDelta(
            url="https://example.com/image.png",
            media_type=123,  # type: ignore[arg-type]
        )

    assert (
        ImageDelta(url="https://example.com/image.png").url
        == "https://example.com/image.png"
    )


def test_delta_serde_roundtrips_variant_types() -> None:
    tool_delta = ToolCallDelta(input='{"x": 1}', id="call_1", name="lookup")
    image_delta = ImageDelta(url="https://example.com/image.png", media_type="image/png")
    continuation_delta = ContinuationDelta(provider="gemini", kind="thought_signature", data={"value": "sig"}, part_index=1)

    assert delta_from_dict(delta_to_dict(tool_delta)) == tool_delta
    assert delta_from_dict(delta_to_dict(image_delta)) == image_delta
    assert delta_from_dict(delta_to_dict(continuation_delta)) == continuation_delta
    assert delta_from_dict(delta_to_dict(TextDelta(""))) == TextDelta("")


def test_function_tool_is_explicit_serializable_spec() -> None:
    tool = FunctionTool(
        name="lookup",
        description="Look up a query.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    assert tool.name == "lookup"
    assert tool.description == "Look up a query."
    assert "fn" not in tool_to_dict(tool)


def test_endpoint_request_bases_validate_model_and_prompt() -> None:
    with pytest.raises(ValueError, match="model is required"):
        ImageGenerationRequest(model="", prompt="draw a duck")

    with pytest.raises(ValueError, match="prompt is required"):
        ImageGenerationRequest(model="m", prompt="")


def test_reasoning_serde_uses_current_fields_and_reads_legacy_budget() -> None:
    config = Config(
        reasoning=Reasoning(
            effort="high",
            thinking_budget=12,
            summary="auto",
        )
    )

    assert config_to_dict(config)["reasoning"] == {
        "effort": "high",
        "thinking_budget": 12,
        "summary": "auto",
    }
    # Pre-MAP-7 spellings read leniently (INV-043): total_budget is dropped,
    # "adaptive" (which meant "absent") reads as medium.
    old = config_from_dict({"reasoning": {"effort": "adaptive", "thinking_budget": 12, "total_budget": 100}})
    assert old.reasoning == Reasoning(effort="medium", thinking_budget=12)

    # Legacy payloads with enabled=False + budget collapse to effort="off";
    # the budget is dropped because reasoning is disabled.
    legacy = config_from_dict({"reasoning": {"enabled": False, "budget": 7}})
    assert legacy.reasoning == Reasoning(effort="off")

    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="off", thinking_budget=7)
    with pytest.raises(ValueError, match="reasoning summary"):
        Reasoning(effort="high", summary="verbose")  # type: ignore[arg-type]


def test_json_fields_are_validated() -> None:
    with pytest.raises(TypeError, match="input"):
        tool_call("call_1", "lookup", {"bad": object()})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="response_format"):
        Config(response_format={"enum": ("a", "b")})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="extensions"):
        Config(extensions={"bad": object()})  # type: ignore[dict-item]


def test_sequence_inputs_are_normalized_to_tuples() -> None:
    msg = Message.user((TextPart("a"), TextPart("b")))
    mixed_msg = Message.user(("caption", ImagePart(url="https://example.com/cat.png")))
    choice = ToolChoice(allowed=["lookup"])  # type: ignore[arg-type]
    config = Config(stop="END")  # type: ignore[arg-type]

    assert msg.parts == (TextPart("a"), TextPart("b"))
    assert mixed_msg.parts == (
        TextPart("caption"),
        ImagePart(url="https://example.com/cat.png"),
    )
    assert choice.allowed == ("lookup",)
    assert config.stop == ("END",)


def test_tool_choice_allowed_accepts_tool_names() -> None:
    lookup = FunctionTool(name="lookup")

    assert ToolChoice(allowed="lookup").allowed == ("lookup",)
    assert ToolChoice.from_tools([lookup]).allowed == ("lookup",)
    with pytest.raises(ValueError, match="tool names"):
        ToolChoice(allowed=[lookup])  # type: ignore[list-item]


def test_message_tool_accepts_single_tool_result_part() -> None:
    result = ToolResultPart(id="call_1", content=(TextPart("ok"),))

    assert Message.tool(result).parts == (result,)


def test_batch_request_infers_model_and_allows_mixed_nested_models() -> None:
    first = Request(model="m1", messages=(Message.user("one"),))
    second = Request(model="m2", messages=(Message.user("two"),))

    batch = BatchRequest(requests=(first, second))

    assert batch.model == "m1"
    assert [request.model for request in batch.requests] == ["m1", "m2"]


def test_stream_events_are_variant_dataclasses() -> None:
    with pytest.raises(TypeError):
        StreamDeltaEvent()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        StreamErrorEvent()  # type: ignore[call-arg]

    # Providers do not always expose ids or usage at stream boundaries;
    # the materializer can fill sane defaults from the request.
    assert StreamStartEvent().type == "start"
    assert StreamEndEvent().type == "end"


def test_usage_preserves_provider_reported_total() -> None:
    usage = Usage(input_tokens=1, output_tokens=2, total_tokens=10)

    assert usage.total_tokens == 10


def test_numeric_budgets_and_usage_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="thinking_budget"):
        Reasoning(effort="low", thinking_budget=0)
    with pytest.raises(TypeError):  # effort is required (decision E, MAP-7)
        Reasoning()  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="input_tokens"):
        Usage(input_tokens=-1)


def test_json_objects_are_validated_not_wrapped() -> None:
    payload = {"a": {"b": [1]}}
    call = ToolCallPart(id="c", name="f", input=payload)

    assert call.input is payload
    call.input["mutated"] = True
    assert call.input["mutated"] is True


def test_tool_result_rejects_thinking_and_protocol_parts() -> None:
    """ToolResultPart rejects parts outside ToolResultContentPart at runtime."""
    with pytest.raises(TypeError, match="thinking parts"):
        ToolResultPart(id="c", content=(ThinkingPart("internal"),))
    with pytest.raises(TypeError, match="refusals"):
        ToolResultPart(id="c", content=(RefusalPart("no"),))
    with pytest.raises(TypeError, match="is_error"):
        ToolResultPart(
            id="c",
            content=(TextPart("ok"),),
            is_error=None,  # type: ignore[arg-type]
        )


def test_live_media_events_require_matching_media_type_prefixes() -> None:
    LiveClientAudioEvent(data=_b64(b"audio"), media_type="audio/pcm;rate=16000")
    LiveClientImageEvent(data=_b64(b"image"), media_type="image/jpeg")
    with pytest.raises(ValueError, match="audio/"):
        LiveClientAudioEvent(data=_b64(b"audio"), media_type="image/jpeg")
    with pytest.raises(ValueError, match="image/"):
        LiveClientImageEvent(data=_b64(b"image"), media_type="video/mp4")


def test_live_turn_event_reuses_prompt_part_validation() -> None:
    LiveClientTurnEvent(parts=(TextPart("hello"), ImagePart(data=_b64(b"image"))))
    with pytest.raises(TypeError, match="protocol parts"):
        LiveClientTurnEvent(parts=(ToolCallPart(id="c", name="tool", input={}),))


def test_live_client_tool_result_rejects_model_and_protocol_parts() -> None:
    with pytest.raises(TypeError, match="model or protocol parts"):
        LiveClientToolResultEvent(
            id="c",
            content=(ThinkingPart("internal"),),
        )
    with pytest.raises(TypeError, match="model or protocol parts"):
        LiveClientToolResultEvent(id="c", content=(RefusalPart("no"),))


def test_user_messages_reject_citations() -> None:
    """User and developer messages cannot carry model-emitted citation parts."""
    with pytest.raises(TypeError, match="protocol parts"):
        Message.user(CitationPart(url="https://example.com"))
    with pytest.raises(TypeError, match="protocol parts"):
        Message.developer(CitationPart(url="https://example.com"))


def test_text_parts_reject_non_string_text() -> None:
    with pytest.raises(TypeError, match="TextPart.text"):
        TextPart(text=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ThinkingPart.text"):
        ThinkingPart(text=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="RefusalPart.text"):
        RefusalPart(text="")


def test_numeric_validators_reject_bool() -> None:
    """`bool` subclasses `int`; numeric fields should still reject it."""
    with pytest.raises(TypeError, match="max_tokens"):
        Config(max_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="part_index"):
        TextDelta(text="x", part_index=True)  # type: ignore[arg-type]


def test_audio_delta_accepts_partial_chunks_and_metadata() -> None:
    assert AudioDelta(data="not base64!@#").data == "not base64!@#"
    assert AudioDelta(data="").data == ""
    assert AudioDelta(media_type="audio/wav").media_type == "audio/wav"
    assert AudioDelta(url="https://example.com/a.wav").url == "https://example.com/a.wav"
    valid = AudioDelta(data=base64.b64encode(b"hi").decode("ascii"))
    assert valid.data == base64.b64encode(b"hi").decode("ascii")


def test_reasoning_off_rejects_budgets() -> None:
    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="off", thinking_budget=10)
    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="off", summary="auto")


def test_generated_media_optional_strings_reject_empty() -> None:
    with pytest.raises(ValueError, match="size"):
        ImageGenerationRequest(model="m", prompt="draw", size="")


def test_file_upload_request_requires_payload() -> None:
    with pytest.raises(TypeError):
        FileUploadRequest()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="bytes_data"):
        FileUploadRequest(filename="f", bytes_data=b"")


def test_file_upload_request_accepts_lazy_path(tmp_path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")

    request = FileUploadRequest(filename="payload.bin", path=payload)

    assert request.bytes_data is None
    assert request.bytes == b"payload"


def test_live_and_stream_tool_call_deltas_are_symmetric_on_empty_input() -> None:
    """Both code paths must accept empty fragments equally."""
    ToolCallDelta(input="")  # accepted
    LiveServerToolCallDeltaEvent(input_delta="")  # accepted


def test_event_variants_do_not_have_other_variants_fields() -> None:
    start = StreamStartEvent()
    interrupted = LiveServerInterruptedEvent()

    assert not hasattr(start, "delta")
    assert not hasattr(interrupted, "text")


def test_a_budget_alone_fills_effort_from_the_grading_table() -> None:
    # MAP-7 rule 3 read the other way (amended 2026-10-10).
    assert [Reasoning(thinking_budget=b).effort for b in (512, 1024, 2047, 2048, 8192, 16384, 24576, 32768, 10**6)] == [
        "minimal", "minimal", "minimal", "low", "medium", "high", "xhigh", "max", "max"]
    assert Reasoning(effort="high", thinking_budget=1024).effort == "high"  # a given effort is kept
    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="none")


def test_response_text_and_json_read_a_data_part_answer() -> None:
    from lm15.types import DataPart, Message, Response, TextPart, Usage

    # types.md §Response convenience (amended 2026-10-10): MAP-14 answers a
    # schema with a judgment property as a DataPart.
    response = Response(id=None, model="m", message=Message.assistant((DataPart(value={"ok": True, "n": 1.50}),)),
                        finish_reason="stop", usage=Usage())
    assert response.text == '{"ok":true,"n":1.5}'
    assert response.json == {"ok": True, "n": 1.5} == response.data
    text_too = Response(id=None, model="m", message=Message.assistant((TextPart("hi"), DataPart(value={"ok": True}))),
                        finish_reason="stop", usage=Usage())
    assert text_too.text is None  # mixed text and data: no guess
