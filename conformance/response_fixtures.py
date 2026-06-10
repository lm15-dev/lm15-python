from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

CONFORMANCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CONFORMANCE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lm15.providers import AnthropicLM, GeminiLM, HttpResponse, OpenAILM
from lm15.providers.base import BaseProviderLM
from lm15.sse import parse_sse
from lm15.stream import coalesce_stream, materialize_response
from lm15.types import (
    AudioPart,
    BuiltinTool,
    CitationPart,
    DocumentPart,
    FunctionTool,
    ImagePart,
    Message,
    RefusalPart,
    Request,
    Response,
    StreamErrorEvent,
    StreamEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

PROVIDER_REQUESTS_ROOT = CONFORMANCE_ROOT / "provider_requests"
CASES_ROOT = PROVIDER_REQUESTS_ROOT / "cases"
BODIES_ROOT = PROVIDER_REQUESTS_ROOT / "results" / "bodies"


def load_body(path: str | Path) -> bytes:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.read_bytes()


def http_response(body: bytes) -> HttpResponse:
    return HttpResponse(
        status=200,
        reason="OK",
        headers=[("content-type", "application/json")],
        body=body,
    )


def load_case(provider: str, feature: str) -> dict[str, Any]:
    return json.loads((CASES_ROOT / provider / f"{feature}.json").read_text())


def iter_cases_with_expect_lm15() -> Iterator[tuple[str, str, dict[str, Any]]]:
    for path in sorted(CASES_ROOT.glob("*/*.json")):
        case = json.loads(path.read_text())
        if "expect_lm15" not in case:
            continue
        yield str(case["provider"]), str(case["feature"]), case


def latest_body_path(provider: str, feature: str) -> Path | None:
    body_dir = BODIES_ROOT / f"{provider}.{feature}"
    if not body_dir.exists():
        return None
    bodies = sorted(body_dir.glob("*.txt"))
    return bodies[-1] if bodies else None


def latest_body(provider: str, feature: str) -> bytes:
    path = latest_body_path(provider, feature)
    if path is None:
        raise FileNotFoundError(f"No response bodies for {provider}.{feature}")
    return path.read_bytes()


def body_path_for_expect(provider: str, feature: str, expect: dict[str, Any], *, stream: bool) -> Path | None:
    """Return the newest saved body that satisfies an lm15 expectation.

    Provider live outputs can be structurally nondeterministic (for example,
    web-search answers sometimes include citations and sometimes do not).  The
    expectation layer proves parser coverage against saved real shapes, so use
    the newest fixture that actually contains the expected shape.
    """
    body_dir = BODIES_ROOT / f"{provider}.{feature}"
    if not body_dir.exists():
        return None
    case = load_case(provider, feature)
    request = request_from_case(case)
    for path in sorted(body_dir.glob("*.txt"), reverse=True):
        body = path.read_bytes()
        try:
            if stream:
                events = parse_stream(provider, request, body)
                assert_no_stream_errors(events)
                response = materialize_response(iter(events), request)
            else:
                response = parse_complete(provider, request, body)
            assert_expect_lm15(response, expect)
        except Exception:
            continue
        return path
    return latest_body_path(provider, feature)


def response_kind(body: bytes) -> str:
    stripped = body.lstrip()
    if stripped.startswith((b"event:", b"data:")):
        return "stream"
    if stripped.startswith((b"{", b"[")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return "unknown"
        if isinstance(payload, dict) and "error" in payload:
            return "error"
        return "json"
    return "unknown"


def is_stream_case(case: dict[str, Any]) -> bool:
    request = case.get("request", {}) if isinstance(case.get("request"), dict) else {}
    body = request.get("body", {}) if isinstance(request.get("body"), dict) else {}
    url = str(request.get("url") or "")
    return bool(body.get("stream") is True or "streamGenerateContent" in url)


def provider_lm(provider: str) -> BaseProviderLM:
    if provider == "openai":
        return OpenAILM(api_key="test")
    if provider == "anthropic":
        return AnthropicLM(api_key="test")
    if provider == "gemini":
        return GeminiLM(api_key="test")
    raise ValueError(f"unknown provider: {provider}")


def request_from_case(case: dict[str, Any]) -> Request:
    provider = str(case["provider"])
    request = case.get("request", {}) if isinstance(case.get("request"), dict) else {}
    body = request.get("body", {}) if isinstance(request.get("body"), dict) else {}
    model = _model_from_case(provider, request, body)
    tools = tuple(_tools_from_case(provider, body))
    return Request(model=model, messages=(Message.user("fixture"),), tools=tools)


def parse_complete(provider: str, request: Request, body: bytes) -> Response:
    lm = provider_lm(provider)
    return lm.parse_response(request, http_response(body))


def parse_stream(provider: str, request: Request, body: bytes) -> list[StreamEvent]:
    lm = provider_lm(provider)

    def raw_events() -> Iterator[StreamEvent]:
        lines = iter(body.splitlines(keepends=True))
        for raw in parse_sse(lines):
            parse_many = getattr(lm, "parse_stream_events", None)
            if parse_many is not None:
                yield from (event for event in parse_many(request, raw) if event is not None)
            else:
                event = lm.parse_stream_event(request, raw)
                if event is not None:
                    yield event

    # MAP-3: the canonical event trace is the POST-coalesce trace.
    return list(coalesce_stream(raw_events()))


def parse_complete_fixture(provider: str, feature: str, *, body_path: str | Path | None = None) -> Response:
    case = load_case(provider, feature)
    body = load_body(body_path) if body_path is not None else latest_body(provider, feature)
    return parse_complete(provider, request_from_case(case), body)


def parse_fixture_path(path: str | Path) -> Response:
    p = Path(path)
    if not p.is_absolute():
        p = BODIES_ROOT / p
    fixture_id = p.parent.name
    provider, feature = fixture_id.split(".", 1)
    return parse_complete_fixture(provider, feature, body_path=p)


def parse_openai_fixture(path: str | Path) -> Response:
    return parse_fixture_path(path)


def parse_stream_fixture(provider: str, feature: str, *, body_path: str | Path | None = None) -> list[StreamEvent]:
    case = load_case(provider, feature)
    body = load_body(body_path) if body_path is not None else latest_body(provider, feature)
    return parse_stream(provider, request_from_case(case), body)


def materialize_stream_fixture(provider: str, feature: str, *, body_path: str | Path | None = None) -> Response:
    case = load_case(provider, feature)
    body = load_body(body_path) if body_path is not None else latest_body(provider, feature)
    request = request_from_case(case)
    events = parse_stream(provider, request, body)
    assert_no_stream_errors(events)
    return materialize_response(iter(events), request)


def assert_no_stream_errors(events: Iterable[StreamEvent]) -> None:
    errors = [event for event in events if isinstance(event, StreamErrorEvent)]
    assert not errors, errors


def assert_expect_lm15(response: Response, expect: dict[str, Any]) -> None:
    provider_data = response.provider_data if isinstance(response.provider_data, dict) else {}
    assert not provider_data.get("_lm15_unmapped"), provider_data.get("_lm15_unmapped")

    parts_expect = expect.get("parts", {}) if isinstance(expect.get("parts"), dict) else {}
    counts = _part_counts(response)
    for part_type, rule in parts_expect.items():
        if not isinstance(rule, dict):
            continue
        actual = counts.get(part_type, 0)
        if "exact" in rule:
            assert actual == int(rule["exact"]), f"{part_type}: expected exact {rule['exact']}, got {actual}"
        if "min" in rule:
            assert actual >= int(rule["min"]), f"{part_type}: expected >= {rule['min']}, got {actual}"
        if "max" in rule:
            assert actual <= int(rule["max"]), f"{part_type}: expected <= {rule['max']}, got {actual}"

    if "finish_reason" in expect:
        assert response.finish_reason == expect["finish_reason"]

    usage_expect = expect.get("usage", {}) if isinstance(expect.get("usage"), dict) else {}
    if usage_expect.get("required"):
        total = response.usage.total_tokens or 0
        assert total > 0 or response.usage.input_tokens > 0 or response.usage.output_tokens > 0


def _part_counts(response: Response) -> dict[str, int]:
    counts = {
        "text": len(response.message.parts_of(TextPart)),
        "tool_call": len(response.message.parts_of(ToolCallPart)),
        "thinking": len(response.message.parts_of(ThinkingPart)),
        "citation": len(response.message.parts_of(CitationPart)),
        "refusal": len(response.message.parts_of(RefusalPart)),
        "image": len(response.message.parts_of(ImagePart)),
        "audio": len(response.message.parts_of(AudioPart)),
        "document": len(response.message.parts_of(DocumentPart)),
    }
    return counts


def _model_from_case(provider: str, request: dict[str, Any], body: dict[str, Any]) -> str:
    if isinstance(body.get("model"), str) and body["model"]:
        return str(body["model"])
    url = str(request.get("url") or "")
    if provider == "gemini":
        match = re.search(r"/models/([^/:]+)", url)
        if match:
            return match.group(1)
    return "fixture-model"


def _tools_from_case(provider: str, body: dict[str, Any]) -> Iterator[FunctionTool | BuiltinTool]:
    seen: set[str] = set()
    for tool in body.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        parsed_tools = list(_parse_tool(provider, tool))
        for parsed in parsed_tools:
            if parsed.name in seen:
                continue
            seen.add(parsed.name)
            yield parsed


def _parse_tool(provider: str, tool: dict[str, Any]) -> Iterator[FunctionTool | BuiltinTool]:
    if provider == "openai":
        t = str(tool.get("type") or "")
        if t == "function":
            yield FunctionTool(
                name=str(tool.get("name") or "tool"),
                description=tool.get("description") if isinstance(tool.get("description"), str) else None,
                parameters=tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {"type": "object", "properties": {}},
            )
        elif t == "web_search_preview":
            yield BuiltinTool("web_search")
        elif t == "code_interpreter":
            yield BuiltinTool("code_execution")
        elif t == "file_search":
            yield BuiltinTool("file_search")
        elif t == "computer_use_preview":
            yield BuiltinTool("computer_use")
        return

    if provider == "anthropic":
        t = str(tool.get("type") or "")
        if t.startswith("web_search"):
            yield BuiltinTool("web_search")
        elif t.startswith("code_execution"):
            yield BuiltinTool("code_execution")
        elif "input_schema" in tool or "name" in tool:
            yield FunctionTool(
                name=str(tool.get("name") or "tool"),
                description=tool.get("description") if isinstance(tool.get("description"), str) else None,
                parameters=tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
            )
        return

    if provider == "gemini":
        for declaration in tool.get("functionDeclarations", []) or []:
            if not isinstance(declaration, dict):
                continue
            yield FunctionTool(
                name=str(declaration.get("name") or "tool"),
                description=declaration.get("description") if isinstance(declaration.get("description"), str) else None,
                parameters=declaration.get("parameters") if isinstance(declaration.get("parameters"), dict) else {"type": "object", "properties": {}},
            )
        if "googleSearch" in tool:
            yield BuiltinTool("web_search")
        if "codeExecution" in tool:
            yield BuiltinTool("code_execution")
