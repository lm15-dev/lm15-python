from __future__ import annotations

from typing import Any

from ..types import DataSource, Message, Part


def parts_to_text(parts: tuple[Part, ...]) -> str:
    return "\n".join(p.text for p in parts if p.type == "text" and p.text)


def part_to_openai_input(part: Part) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "input_text", "text": part.text or ""}

    if part.type == "image" and part.source:
        if part.source.type == "url":
            payload = {"type": "input_image", "image_url": part.source.url}
            if part.source.detail:
                payload["detail"] = part.source.detail
            return payload
        if part.source.type == "base64":
            return {
                "type": "input_image",
                "image_url": f"data:{part.source.media_type};base64,{part.source.data}",
            }
        if part.source.type == "file":
            return {"type": "input_image", "file_id": part.source.file_id}

    if part.type == "audio" and part.source:
        if part.source.type == "base64":
            media = (part.source.media_type or "audio/wav").split("/")[-1]
            return {
                "type": "input_audio",
                "audio": part.source.data,
                "format": media,
            }
        if part.source.type == "url":
            return {"type": "input_audio", "audio_url": part.source.url}
        if part.source.type == "file":
            return {"type": "input_audio", "file_id": part.source.file_id}

    if part.type == "document" and part.source:
        if part.source.type == "url":
            return {"type": "input_file", "file_url": part.source.url}
        if part.source.type == "base64":
            return {
                "type": "input_file",
                "file_data": f"data:{part.source.media_type};base64,{part.source.data}",
            }
        if part.source.type == "file":
            return {"type": "input_file", "file_id": part.source.file_id}

    if part.type == "video" and part.source:
        if part.source.type == "url":
            return {"type": "input_video", "video_url": part.source.url}
        if part.source.type == "base64":
            return {
                "type": "input_video",
                "video_data": f"data:{part.source.media_type};base64,{part.source.data}",
            }
        if part.source.type == "file":
            return {"type": "input_video", "file_id": part.source.file_id}

    if part.type == "tool_result":
        return {
            "type": "input_text",
            "text": parts_to_text(part.content),
        }

    return {"type": "input_text", "text": part.text or ""}


def message_to_openai_input(msg: Message) -> dict[str, Any]:
    return {"role": msg.role, "content": [part_to_openai_input(p) for p in msg.parts]}


def ds_to_anthropic_source(ds: DataSource) -> dict[str, Any]:
    if ds.type == "url":
        return {"type": "url", "url": ds.url}
    if ds.type == "file":
        return {"type": "file", "file_id": ds.file_id}
    return {"type": "base64", "media_type": ds.media_type, "data": ds.data}
