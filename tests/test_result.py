from __future__ import annotations

import base64

import pytest

from lm15.result import (
    Result,
    response_to_events,
)
from lm15.types import (
    AudioPart,
    ContinuationDelta,
    ContinuationState,
    DocumentPart,
    ImageDelta,
    ImagePart,
    Message,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    TextPart,
    Usage,
    VideoPart,
)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _response_with(part) -> Response:
    return Response(
        id="r1",
        model="m",
        message=Message.assistant(part),
        finish_reason="stop",
        usage=Usage(),
    )


def test_response_to_events_preserves_image_file_ids() -> None:
    image = ImagePart(file_id="file_123")
    events = list(response_to_events(_response_with(image)))

    assert events[0].type == "start"
    assert isinstance(events[1].delta, ImageDelta)
    assert events[1].delta.file_id == "file_123"
    assert events[-1].type == "end"


def test_materialize_preserves_continuation_only_part_index() -> None:
    response = Result(
        events=iter((
            StreamDeltaEvent(
                delta=ContinuationDelta(
                    provider="anthropic",
                    kind="redacted_thinking",
                    data={"data": "opaque"},
                    part_index=0,
                )
            ),
            StreamEndEvent(finish_reason="stop"),
        )),
        request=Request(model="m", messages=(Message.user("hi"),)),
    ).response

    assert response.message.parts == (
        TextPart(
            "",
            continuation=(
                ContinuationState(provider="anthropic", kind="redacted_thinking", data={"data": "opaque"}),
            ),
        ),
    )


def test_response_to_events_and_materialize_preserve_continuation_state() -> None:
    response = Response(
        id="r1",
        model="m",
        message=Message(
            role="assistant",
            parts=(
                TextPart(
                    "hello",
                    continuation=(
                        ContinuationState(provider="openai", kind="response_item_id", data={"id": "item_1"}),
                    ),
                ),
            ),
            continuation=(
                ContinuationState(provider="openai", kind="response_id", data={"id": "resp_1"}),
            ),
        ),
        finish_reason="stop",
        usage=Usage(),
    )

    events = list(response_to_events(response))
    assert any(
        event.type == "delta" and isinstance(event.delta, ContinuationDelta) and event.delta.part_index is None
        for event in events
    )
    rebuilt = Result(
        events=iter(events),
        request=Request(model="m", messages=(Message.user("hi"),)),
    ).response
    assert rebuilt == response


@pytest.mark.parametrize(
    "part",
    [
        # AudioPart by reference is non-streamable: AudioDelta requires inline data.
        AudioPart(url="https://example.com/audio.wav"),
        VideoPart(data=_b64(b"video")),
        DocumentPart(data=_b64(b"pdf")),
        RefusalPart("no"),
    ],
)
def test_response_to_events_raises_for_parts_without_delta_variants(part) -> None:
    with pytest.raises(TypeError, match="Cannot convert"):
        list(response_to_events(_response_with(part)))


def test_result_delegates_video_and_document_helpers() -> None:
    video = VideoPart(data=_b64(b"video"))
    document = DocumentPart(data=_b64(b"doc"))
    response = Response(
        id="r1",
        model="m",
        message=Message.assistant([video, document]),
        finish_reason="stop",
        usage=Usage(),
    )
    result = Result(events=iter(()), request=Request(model="m", messages=(Message.user("hi"),)))
    result._response = response
    result._done = True

    assert result.video is video
    assert result.videos == [video]
    assert result.video_bytes == b"video"
    assert result.document is document
    assert result.documents == [document]
    assert result.document_bytes == b"doc"


def test_result_rejects_tool_loop_parameters() -> None:
    """Result is a pure stream materializer: the automatic tool-execution
    loop was removed (positioning decision, 2026-06-11)."""
    req = Request(model="m", messages=(Message.user("hi"),))
    for kwarg in ("callable_registry", "on_tool_call", "max_tool_rounds", "retries"):
        with pytest.raises(TypeError):
            Result(events=iter(()), request=req, **{kwarg: None})


def test_result_module_has_no_tool_execution_helpers() -> None:
    import lm15.result as result_mod

    for name in ("_invoke_tool", "_normalize_tool_output", "_preview_parts",
                 "_ExecutedTool"):
        assert not hasattr(result_mod, name)
