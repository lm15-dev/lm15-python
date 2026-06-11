#!/usr/bin/env python3
"""Dump the provider HTTP request for one canonical lm15 logical case.

Input is a single JSON object, either as an argument or on stdin. Output is a
normalized JSON object with method/url/headers/params/body. The request is built
but never sent.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm15.providers.anthropic import AnthropicLM
from lm15.providers.gemini import GeminiLM
from lm15.providers.openai import OpenAILM
from lm15.serde import messages_from_json, reasoning_from_dict, tool_choice_from_dict
from lm15.types import BuiltinTool, Config, FunctionTool, Message, Request

JsonObject = dict[str, Any]


def dump_request(case: JsonObject) -> JsonObject:
    """Return the normalized provider HTTP request for a logical test case."""
    case = deepcopy(case)
    provider = provider_for_case(case)
    lm = adapter_for_provider(provider)
    request = request_for_case(case)
    transport_req = lm.build_request(request, stream=bool(case.get("stream", False)))
    return normalize_transport_request(transport_req)


def provider_for_case(case: JsonObject) -> str:
    case_id = str(case.get("id") or "")
    if "." not in case_id:
        raise ValueError("case id must start with '<provider>.'")
    provider = case_id.split(".", 1)[0]
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ValueError(f"unknown provider: {provider}")
    return provider


def adapter_for_provider(provider: str):
    if provider == "openai":
        return OpenAILM(api_key="test-key")
    if provider == "anthropic":
        return AnthropicLM(api_key="test-key")
    if provider == "gemini":
        return GeminiLM(api_key="test-key")
    raise ValueError(f"unknown provider: {provider}")


def request_for_case(case: JsonObject) -> Request:
    config_kwargs: JsonObject = {}

    for source_key, config_key in (
        ("temperature", "temperature"),
        ("max_tokens", "max_tokens"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop", "stop"),
        ("response_format", "response_format"),
    ):
        if source_key in case:
            config_kwargs[config_key] = case[source_key]

    if case.get("reasoning"):
        config_kwargs["reasoning"] = reasoning_from_dict(case["reasoning"])

    tools = tools_for_case(case)
    apply_provider_passthrough(case.get("provider"), config_kwargs)

    messages: tuple[Message, ...]
    if case.get("messages"):
        messages = tuple(messages_from_json(normalize_legacy_messages(case["messages"])))
    elif "prompt" in case:
        messages = (Message.user(str(case["prompt"])),)
    else:
        raise ValueError("case must contain either 'prompt' or 'messages'")

    return Request(
        model=str(case["model"]),
        messages=messages,
        system=case.get("system"),
        tools=tuple(tools),
        config=Config(**config_kwargs),
    )


def tools_for_case(case: JsonObject) -> list[FunctionTool | BuiltinTool]:
    tools: list[FunctionTool | BuiltinTool] = []
    for tool in case.get("tools") or []:
        tools.append(
            FunctionTool(
                name=tool["name"],
                description=tool.get("description"),
                parameters=tool.get("parameters", {"type": "object", "properties": {}}),
            )
        )
    for tool in case.get("builtin_tools") or []:
        tools.append(BuiltinTool(name=tool["name"], config=tool.get("builtin_config")))
    return tools


def apply_provider_passthrough(provider_data: Any, config_kwargs: JsonObject) -> None:
    if not isinstance(provider_data, dict):
        return

    passthrough = dict(provider_data)

    if "tool_choice" in passthrough:
        raw = passthrough.pop("tool_choice")
        config_kwargs["tool_choice"] = tool_choice_from_dict(normalize_tool_choice(raw))

    if "response_format" in passthrough:
        config_kwargs["response_format"] = passthrough.pop("response_format")

    if passthrough:
        extensions = dict(config_kwargs.get("extensions") or {})
        extensions.update(passthrough)
        config_kwargs["extensions"] = extensions


def normalize_tool_choice(raw: Any) -> JsonObject:
    if isinstance(raw, str):
        if raw == "any":
            return {"mode": "required"}
        if raw == "tool":
            return {"mode": "required"}
        return {"mode": raw}

    if isinstance(raw, dict):
        mode = str(raw.get("type") or raw.get("mode") or "auto")
        out: JsonObject = {"mode": mode}
        if mode in {"tool", "function", "any"}:
            out["mode"] = "required"
        if raw.get("name"):
            out["allowed"] = [raw["name"]]
        if "disable_parallel_tool_use" in raw:
            out["parallel"] = not bool(raw["disable_parallel_tool_use"])
        if "parallel" in raw:
            out["parallel"] = raw["parallel"]
        return out

    return {"mode": "auto"}


def normalize_legacy_messages(messages: list[JsonObject]) -> list[JsonObject]:
    """Accept legacy cross-sdk message JSON and return lm15 serde JSON."""
    normalized = deepcopy(messages)
    for msg in normalized:
        for part in msg.get("parts", []):
            source = part.pop("source", None)
            if isinstance(source, dict):
                # Older fixtures used source.type=url/base64/file. lm15-python serde
                # uses direct media fields: url/data/file_id/media_type/detail.
                source.pop("type", None)
                part.update(source)
            if "arguments" in part and "input" not in part:
                part["input"] = part.pop("arguments")
    return normalized


def normalize_transport_request(transport_req: Any) -> JsonObject:
    parsed = urllib.parse.urlparse(transport_req.url)
    params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    headers: JsonObject = {}
    for key, value in transport_req.headers:
        lower = key.lower()
        if lower in {"authorization", "x-api-key", "x-goog-api-key"}:
            headers[lower] = "REDACTED"
        else:
            headers[lower] = value

    out: JsonObject = {
        "method": transport_req.method,
        "url": url,
        "headers": headers,
        "params": params,
    }

    body = transport_req.body
    if body:
        try:
            out["body"] = json.loads(body.decode("utf-8"))
        except Exception:
            out["body"] = body.decode("utf-8", errors="replace")
    return out


def _load_case(argv: list[str]) -> JsonObject:
    if len(argv) > 1:
        arg = argv[1]
        path = Path(arg)
        if path.exists():
            return json.loads(path.read_text())
        return json.loads(arg)
    return json.load(sys.stdin)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    try:
        case = _load_case(argv)
        print(json.dumps(dump_request(case), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"dump_request.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
