from __future__ import annotations

from pathlib import Path

from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.serde import request_from_dict, request_to_dict, response_to_dict

from ._helpers import FakeTransport, ProbeResult, load_portability_fixture


_ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def run(test: dict, root: Path) -> ProbeResult:
    provider = test["provider"]
    cases = load_portability_fixture(root, "tool_call.json").get("cases", [])
    case = next((c for c in cases if c.get("provider") == provider), None)
    if case is None:
        return ProbeResult(status="skip", details=f"no frozen fixture for provider: {provider}")

    adapter_cls = _ADAPTERS.get(provider)
    if adapter_cls is None:
        return ProbeResult(status="skip", details=f"unsupported provider: {provider}")

    req = request_from_dict(case["request"])
    if request_to_dict(req) != case["request"]:
        return ProbeResult(status="fail", details=f"request roundtrip mismatch for {case['id']}")

    adapter = adapter_cls(api_key="k", transport=FakeTransport(payload=case["provider_response"]))
    resp = adapter.complete(req)
    actual = response_to_dict(resp)
    expected = case["expected_response"]
    if actual == expected:
        return ProbeResult(status="pass", details=f"matched frozen fixture: {case['id']}")
    return ProbeResult(status="fail", details=f"fixture mismatch for {case['id']}")
