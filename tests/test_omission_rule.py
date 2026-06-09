"""Regression test (S0.4): one omission rule, every entry point.

Canonical JSON omits a typed object's own empty optional fields, but the
contents of opaque user payloads (tool ``input``, JSON-Schema ``parameters``,
config ``extensions``) are never mutated — and serializing a value embedded
in a larger object must produce byte-identical JSON to serializing it
directly.
"""

from __future__ import annotations

import json
import unittest

from lm15.serde import config_to_dict, part_to_dict, request_to_dict, tool_to_dict
from lm15.types import Config, FunctionTool, Message, Request, TextPart, ToolCallPart


def dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class TestOmissionRule(unittest.TestCase):
    def test_function_tool_schema_is_never_mutated(self):
        tool = FunctionTool(name="f", parameters={"type": "object", "properties": {}})
        request = Request(
            model="m",
            messages=(Message(role="user", parts=(TextPart(text="hi"),)),),
            tools=(tool,),
        )
        direct = tool_to_dict(tool)
        embedded = request_to_dict(request)["tools"][0]
        self.assertEqual(direct["parameters"], {"type": "object", "properties": {}})
        self.assertEqual(dumps(direct), dumps(embedded))

    def test_tool_call_input_is_never_mutated(self):
        part = ToolCallPart(id="call_1", name="f", input={"q": ""})
        request = Request(
            model="m",
            messages=(
                Message(role="user", parts=(TextPart(text="hi"),)),
                Message(role="assistant", parts=(part,)),
            ),
        )
        direct = part_to_dict(part)
        embedded = request_to_dict(request)["messages"][1]["parts"][0]
        self.assertEqual(direct["input"], {"q": ""})
        self.assertEqual(dumps(direct), dumps(embedded))

    def test_config_extensions_are_never_mutated(self):
        config = Config(extensions={"a": {}})
        request = Request(
            model="m",
            messages=(Message(role="user", parts=(TextPart(text="hi"),)),),
            config=config,
        )
        direct = config_to_dict(config)
        embedded = request_to_dict(request)["config"]
        self.assertEqual(direct["extensions"], {"a": {}})
        self.assertEqual(dumps(direct), dumps(embedded))


if __name__ == "__main__":
    unittest.main()
