from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.errors import ProviderError, canonical_error_code, map_http_error
from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.serde import request_from_dict, request_to_dict, response_to_dict, stream_event_to_dict
from lm15.sse import SSEEvent
from lm15.transports.base import HttpRequest, HttpResponse

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "spec" / "fixtures" / "v1"


class FakeTransport:
    def __init__(self, payload: dict | None = None, stream_lines: list[bytes] | None = None):
        self.payload = payload or {}
        self.stream_lines = stream_lines or []
        self.last_request: HttpRequest | None = None

    def request(self, req: HttpRequest) -> HttpResponse:
        self.last_request = req
        return HttpResponse(status=200, headers={"content-type": "application/json"}, body=json.dumps(self.payload).encode("utf-8"))

    def stream(self, req: HttpRequest):
        self.last_request = req
        for line in self.stream_lines:
            yield line


ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def load_bundle(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def make_adapter(provider: str, *, payload: dict | None = None, stream_lines: list[str] | None = None):
    cls = ADAPTERS[provider]
    lines = [line.encode("utf-8") for line in (stream_lines or [])]
    return cls(api_key="k", transport=FakeTransport(payload=payload, stream_lines=lines))


class FrozenFixtureTests(unittest.TestCase):
    def assertErrorMatches(self, err: ProviderError, expected: dict) -> None:
        self.assertEqual(type(err).__name__, expected["class"])
        self.assertEqual(canonical_error_code(err), expected["canonical_code"])
        for needle in expected.get("message_contains", []):
            self.assertIn(needle, str(err))

    def test_complete_fixtures_match_adapters(self):
        bundle = load_bundle("complete.json")
        for case in bundle["cases"]:
            with self.subTest(case=case["id"]):
                req = request_from_dict(case["request"])
                self.assertEqual(request_to_dict(req), case["request"])
                adapter = make_adapter(case["provider"], payload=case["provider_response"])
                resp = adapter.complete(req)
                self.assertEqual(response_to_dict(resp), case["expected_response"])

    def test_tool_call_fixtures_match_adapters(self):
        bundle = load_bundle("tool_call.json")
        for case in bundle["cases"]:
            with self.subTest(case=case["id"]):
                req = request_from_dict(case["request"])
                self.assertEqual(request_to_dict(req), case["request"])
                adapter = make_adapter(case["provider"], payload=case["provider_response"])
                resp = adapter.complete(req)
                self.assertEqual(response_to_dict(resp), case["expected_response"])

    def test_stream_fixtures_match_adapters(self):
        bundle = load_bundle("stream.json")
        for case in bundle["cases"]:
            with self.subTest(case=case["id"]):
                req = request_from_dict(case["request"])
                self.assertEqual(request_to_dict(req), case["request"])
                adapter = make_adapter(case["provider"], stream_lines=case["raw_sse_lines"])
                events = [stream_event_to_dict(e) for e in adapter.stream(req)]
                self.assertEqual(events, case["expected_events"])

    def test_http_error_fixtures_match_runtime(self):
        bundle = load_bundle("errors.json")
        for case in bundle["http_status_cases"]:
            with self.subTest(case=case["id"]):
                err = map_http_error(case["status"], case["message"])
                self.assertErrorMatches(err, case["expected"])

    def test_provider_error_fixtures_match_runtime(self):
        bundle = load_bundle("errors.json")

        for case in bundle["normalize_error_cases"]:
            with self.subTest(case=case["id"]):
                adapter = make_adapter(case["provider"])
                err = adapter.normalize_error(case["status"], json.dumps(case["body"]))
                self.assertErrorMatches(err, case["expected"])

        for case in bundle["stream_error_cases"]:
            with self.subTest(case=case["id"]):
                adapter = make_adapter(case["provider"])
                req = request_from_dict(case["request"])
                evt = adapter.parse_stream_event(req, SSEEvent(event=None, data=case["raw_event"]["data"]))
                self.assertIsNotNone(evt)
                assert evt is not None
                self.assertEqual(stream_event_to_dict(evt), case["expected_event"])

        for case in bundle["inband_response_error_cases"]:
            with self.subTest(case=case["id"]):
                adapter = make_adapter(case["provider"])
                req = request_from_dict(case["request"])
                resp = HttpResponse(status=200, headers={}, body=json.dumps(case["response"]).encode("utf-8"))
                with self.assertRaises(ProviderError) as ctx:
                    adapter.parse_response(req, resp)
                self.assertErrorMatches(ctx.exception, case["expected"])


if __name__ == "__main__":
    unittest.main()
