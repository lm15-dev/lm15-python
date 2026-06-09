"""Regression test (S0.1): Anthropic streaming must consume message_delta.

Anthropic sends the authoritative final usage and stop_reason in the
``message_delta`` event. Dropping it loses streamed usage entirely and
reports finish_reason "stop" even when the response was truncated by
max_tokens (which must map to "length", as the complete path does).
"""

from __future__ import annotations

import unittest

from lm15.providers.anthropic import AnthropicLM
from lm15.result import materialize_response
from lm15.sse import parse_sse
from lm15.types import Message, Request, TextPart

# Shaped after the recorded fixture
# conformance/provider_requests/results/bodies/anthropic.streaming/2026-04-13T13-25-39Z.txt,
# with stop_reason switched to max_tokens truncation.
SSE_BODY = b"""event: message_start
data: {"type":"message_start","message":{"model":"claude-sonnet-4-5-20250929","id":"msg_01Test","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":9,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":4}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"input_tokens":9,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":42}}

event: message_stop
data: {"type":"message_stop"}

"""


def _parse_events(lm, request, body: bytes):
    events = []
    for raw in parse_sse(iter(body.splitlines(keepends=True))):
        parse_many = getattr(lm, "parse_stream_events", None)
        if parse_many is not None:
            events.extend(e for e in parse_many(request, raw) if e is not None)
        else:
            event = lm.parse_stream_event(request, raw)
            if event is not None:
                events.append(event)
    return events


class TestAnthropicMessageDelta(unittest.TestCase):
    def _response(self):
        lm = AnthropicLM(api_key="test-key")
        request = Request(
            model="claude-sonnet-4-5",
            messages=(Message(role="user", parts=(TextPart(text="hi"),)),),
        )
        events = _parse_events(lm, request, SSE_BODY)
        return materialize_response(iter(events), request)

    def test_streamed_usage_comes_from_message_delta(self):
        response = self._response()
        self.assertEqual(response.usage.output_tokens, 42)
        self.assertEqual(response.usage.input_tokens, 9)

    def test_max_tokens_truncation_maps_to_length(self):
        response = self._response()
        self.assertEqual(response.finish_reason, "length")


if __name__ == "__main__":
    unittest.main()
