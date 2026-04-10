from __future__ import annotations

import json
from pathlib import Path

from lm15.errors import ProviderError, canonical_error_code, map_http_error
from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.serde import request_from_dict, stream_event_to_dict
from lm15.sse import SSEEvent
from lm15.transports.base import HttpResponse

from ._helpers import FakeTransport, ProbeResult, load_portability_fixture


_ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def _adapter(provider: str):
    cls = _ADAPTERS.get(provider)
    if cls is None:
        raise KeyError(provider)
    return cls(api_key="k", transport=FakeTransport())


def _matches_error(err: ProviderError, expected: dict) -> bool:
    if type(err).__name__ != expected.get("class"):
        return False
    if canonical_error_code(err) != expected.get("canonical_code"):
        return False
    for needle in expected.get("message_contains", []):
        if needle not in str(err):
            return False
    return True


def run(test: dict, root: Path) -> ProbeResult:
    bundle = load_portability_fixture(root, "errors.json")

    for case in bundle.get("http_status_cases", []):
        err = map_http_error(case["status"], case["message"])
        if not _matches_error(err, case["expected"]):
            return ProbeResult(status="fail", details=f"http fixture mismatch: {case['id']}")

    for case in bundle.get("normalize_error_cases", []):
        err = _adapter(case["provider"]).normalize_error(case["status"], json.dumps(case["body"]))
        if not _matches_error(err, case["expected"]):
            return ProbeResult(status="fail", details=f"normalize_error fixture mismatch: {case['id']}")

    for case in bundle.get("stream_error_cases", []):
        adapter = _adapter(case["provider"])
        req = request_from_dict(case["request"])
        evt = adapter.parse_stream_event(req, SSEEvent(event=None, data=case["raw_event"]["data"]))
        if evt is None or stream_event_to_dict(evt) != case["expected_event"]:
            return ProbeResult(status="fail", details=f"stream_error fixture mismatch: {case['id']}")

    for case in bundle.get("inband_response_error_cases", []):
        adapter = _adapter(case["provider"])
        req = request_from_dict(case["request"])
        resp = HttpResponse(status=200, headers={}, body=json.dumps(case["response"]).encode("utf-8"))
        try:
            adapter.parse_response(req, resp)
        except ProviderError as err:
            if not _matches_error(err, case["expected"]):
                return ProbeResult(status="fail", details=f"inband response fixture mismatch: {case['id']}")
        else:
            return ProbeResult(status="fail", details=f"expected ProviderError for fixture: {case['id']}")

    return ProbeResult(status="pass", details="error mapping matches frozen fixtures")
