"""
lm15.vet — the lm15 vet shim (reference implementation).

Speaks the newline-delimited JSON protocol defined in
``lm15-contract/harness/PROTOCOL.md``: one JSON request per stdin line, one
JSON reply per stdout line, same order, one output per input. The shim only
transforms — the harness performs all comparison and runs the shim inside a
no-network sandbox, so this module must never open a connection.

Run as: ``python -m lm15.vet`` (cwd: lm15-python2).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import sys
import urllib.parse
from typing import Any, Callable, Literal, get_args, get_origin

from . import types as lm15_types
from . import serde
from .errors import canonical_error_code
from .providers import AnthropicLM, GeminiLM, HttpResponse, OpenAIChatLM, OpenAILM
from .providers.base import BaseProviderLM
from .result import materialize_response
from .sse import parse_sse
from .types import Request, Response, StreamEvent

JsonObject = dict[str, Any]

LANGUAGE = "python"

try:
    from importlib.metadata import version as _dist_version

    IMPL_VERSION = _dist_version("lm15")
except Exception:  # pragma: no cover - metadata is absent in odd installs
    IMPL_VERSION = "0.0.0"

# Parse-only ops (parse_response, replay_stream, normalize_error) construct an
# adapter but never build auth headers; the key value is irrelevant and must
# never come from the environment.
_PARSE_ONLY_KEY = "vet-parse-only"


# ─── Adapters ────────────────────────────────────────────────────────

def adapter_for_provider(provider: str, api_key: str, base_url: str | None = None) -> BaseProviderLM:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if provider == "openai":
        return OpenAILM(**kwargs)
    if provider == "openai_chat":
        return OpenAIChatLM(**kwargs)
    if provider == "anthropic":
        return AnthropicLM(**kwargs)
    if provider == "gemini":
        return GeminiLM(**kwargs)
    raise ValueError(f"unknown provider: {provider}")


# ─── Serde kind table ────────────────────────────────────────────────

JsonToObj = Callable[[JsonObject], Any]
ObjToJson = Callable[[Any], JsonObject]

KIND_SERDE: dict[str, tuple[JsonToObj, ObjToJson]] = {
    "part": (serde.part_from_dict, serde.part_to_dict),
    "message": (serde.message_from_dict, serde.message_to_dict),
    "tool": (serde.tool_from_dict, serde.tool_to_dict),
    "tool_choice": (serde.tool_choice_from_dict, serde.tool_choice_to_dict),
    "reasoning": (serde.reasoning_from_dict, serde.reasoning_to_dict),
    "config": (serde.config_from_dict, serde.config_to_dict),
    "cache_config": (serde.cache_config_from_dict, serde.cache_config_to_dict),
    "continuation_state": (serde.continuation_from_dict, serde.continuation_to_dict),
    "error_detail": (serde.error_detail_from_dict, serde.error_detail_to_dict),
    "delta": (serde.delta_from_dict, serde.delta_to_dict),
    "usage": (serde.usage_from_dict, serde.usage_to_dict),
    "stream_event": (serde.stream_event_from_dict, serde.stream_event_to_dict),
    "request": (serde.request_from_dict, serde.request_to_dict),
    "response": (serde.response_from_dict, serde.response_to_dict),
    "model_info": (serde.model_info_from_dict, serde.model_info_to_dict),
    "audio_format": (serde.audio_format_from_dict, serde.audio_format_to_dict),
    "live_config": (serde.live_config_from_dict, serde.live_config_to_dict),
    "live_client_event": (serde.live_client_event_from_dict, serde.live_client_event_to_dict),
    "live_server_event": (serde.live_server_event_from_dict, serde.live_server_event_to_dict),
}


def _serde_for_kind(kind: str) -> tuple[JsonToObj, ObjToJson]:
    if kind not in KIND_SERDE:
        raise ValueError(f"unknown kind: {kind}")
    return KIND_SERDE[kind]


# ─── Normalization helpers ───────────────────────────────────────────

def normalize_transport_request(transport_req: Any) -> JsonObject:
    """Normalize a TransportRequest into the protocol's build_request shape.

    Mirrors conformance/cross_sdk/dump_request.py, except nothing is ever
    redacted — the harness asserts exact auth formatting against the api_key
    it injected.
    """
    parsed = urllib.parse.urlparse(transport_req.url)
    params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    headers: JsonObject = {key.lower(): value for key, value in transport_req.headers}

    out: JsonObject = {
        "method": transport_req.method,
        "url": url,
        "params": params,
        "headers": headers,
        "body": None,
    }

    body = transport_req.body
    if body:
        content_type = headers.get("content-type", "")
        if "json" in content_type.lower():
            try:
                out["body"] = json.loads(body.decode("utf-8"))
                return out
            except Exception:
                pass
        out["body_b64"] = base64.b64encode(body).decode("ascii")
    return out


def _http_response(status: int, body: bytes) -> HttpResponse:
    return HttpResponse(
        status=status,
        reason="OK",
        headers=[("content-type", "application/json")],
        body=body,
    )


def _response_result(response: Response) -> JsonObject:
    """Serialize a Response per the protocol, surfacing the unmapped canary."""
    result: JsonObject = {"canonical_response": serde.response_to_dict(response)}
    provider_data = response.provider_data if isinstance(response.provider_data, dict) else {}
    unmapped = provider_data.get("_lm15_unmapped")
    if unmapped is not None:
        result["unmapped"] = list(unmapped)
    return result


def _parse_stream_body(lm: BaseProviderLM, request: Request, body: bytes) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for raw in parse_sse(iter(body.splitlines(keepends=True))):
        for event in lm.parse_stream_events(request, raw):
            if event is not None:
                events.append(event)
    return events


# ─── Ops ─────────────────────────────────────────────────────────────

def op_capabilities(msg: JsonObject) -> JsonObject:
    return {
        "language": LANGUAGE,
        "ops": sorted(HANDLERS),
        "impl_version": IMPL_VERSION,
    }


def _base_url(msg: JsonObject) -> str | None:
    base_url = msg.get("base_url")
    return str(base_url) if base_url is not None else None


def op_build_request(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), str(msg["api_key"]), _base_url(msg))
    request = serde.request_from_dict(msg["canonical_request"])
    transport_req = lm.build_request(request, stream=bool(msg.get("stream", False)))
    return normalize_transport_request(transport_req)


def op_parse_response(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg))
    request = serde.request_from_dict(msg["canonical_request"])
    body = base64.b64decode(msg["body_b64"])
    response = lm.parse_response(request, _http_response(int(msg["status"]), body))
    return _response_result(response)


def op_replay_stream(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg))
    request = serde.request_from_dict(msg["canonical_request"])
    body = base64.b64decode(msg["body_b64"])
    events = _parse_stream_body(lm, request, body)
    response = materialize_response(iter(events), request)
    result: JsonObject = {"events": [serde.stream_event_to_dict(e) for e in events]}
    result.update(_response_result(response))
    return result


def op_normalize_error(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg))
    err = lm.normalize_error(int(msg["status"]), str(msg["body_text"]))
    return {
        "class": type(err).__name__,
        "code": err.code or canonical_error_code(err),
        "provider_code": err.provider_code,
        "message": err.message,
    }


def op_serde_roundtrip(msg: JsonObject) -> JsonObject:
    from_dict, to_dict = _serde_for_kind(str(msg["kind"]))
    return {"value": to_dict(from_dict(msg["value"]))}


def op_validate(msg: JsonObject) -> JsonObject:
    from_dict, to_dict = _serde_for_kind(str(msg["kind"]))
    obj = from_dict(msg["value"])
    return {"ok": True, "normalized": to_dict(obj)}


def op_surface_dump(msg: JsonObject) -> JsonObject:
    return {
        "types": _reflect_types(),
        "enums": _reflect_enums(),
    }


def _reflect_types() -> JsonObject:
    """Every public dataclass in lm15.types, by reflection."""
    out: JsonObject = {}
    for name in sorted(vars(lm15_types)):
        obj = getattr(lm15_types, name)
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if obj.__module__ != lm15_types.__name__ or not dataclasses.is_dataclass(obj):
            continue
        out[obj.__name__] = {"fields": [f.name for f in dataclasses.fields(obj)]}
    return out


def _reflect_enums() -> JsonObject:
    """Every string vocabulary in lm15.types, by reflection.

    Harvests Literal type aliases (Role, FinishReason, …) and module-level
    string collections (FINISH_REASONS, PART_TYPES, …) — never a
    hand-maintained list. Unordered collections are sorted; Literal aliases
    keep declaration order.
    """
    out: JsonObject = {}
    for name in sorted(vars(lm15_types)):
        if name.startswith("_"):
            continue
        vocabulary = _string_vocabulary(getattr(lm15_types, name))
        if vocabulary:
            out[name] = vocabulary
    return out


def _string_vocabulary(obj: Any) -> list[str] | None:
    if get_origin(obj) is Literal:
        args = get_args(obj)
        if args and all(isinstance(a, str) for a in args):
            return list(args)
        return None
    if isinstance(obj, (frozenset, set)):
        if obj and all(isinstance(v, str) for v in obj):
            return sorted(obj)
        return None
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) for k in obj) and all(isinstance(v, type) for v in obj.values()):
            return sorted(obj)
        return None
    if isinstance(obj, (tuple, list)):
        if obj and all(isinstance(v, str) for v in obj):
            return sorted(obj)
        return None
    return None


# ─── Framing ─────────────────────────────────────────────────────────

HANDLERS: dict[str, Callable[[JsonObject], JsonObject]] = {
    "capabilities": op_capabilities,
    "build_request": op_build_request,
    "parse_response": op_parse_response,
    "replay_stream": op_replay_stream,
    "normalize_error": op_normalize_error,
    "serde_roundtrip": op_serde_roundtrip,
    "validate": op_validate,
    "surface_dump": op_surface_dump,
}


def _error_reply(req_id: Any, exc: BaseException) -> JsonObject:
    # lm15-typed errors report the canonical class name; unexpected
    # exceptions report the native exception name. Both are __name__.
    return {
        "id": req_id,
        "ok": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def handle_line(line: str) -> JsonObject:
    try:
        msg = json.loads(line)
    except Exception as exc:
        return _error_reply(None, exc)

    if not isinstance(msg, dict):
        return _error_reply(None, ValueError("request must be a JSON object"))

    req_id = msg.get("id")
    try:
        op = msg.get("op")
        handler = HANDLERS.get(str(op))
        if handler is None:
            raise ValueError(f"unknown op: {op}")
        result = handler(msg)
    except Exception as exc:
        return _error_reply(req_id, exc)
    return {"id": req_id, "ok": True, "result": result}


def main(argv: list[str] | None = None) -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle_line(line)
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
