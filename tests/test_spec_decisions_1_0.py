"""1.0 spec decisions, red-first (INV-033 and INV-042 resolutions).

INV-033: FunctionTool.parameters is required-with-shape — always emitted,
an explicit {} round-trips verbatim; absent on input restores the default
schema.

INV-042: config_from_dict raises TypeError on malformed (non-dict) nested
tool_choice/reasoning/cache instead of silently dropping them.
"""

import unittest

from lm15.serde import config_from_dict, tool_from_dict, tool_to_dict
from lm15.types import FunctionTool

DEFAULT_SCHEMA = {"type": "object", "properties": {}}


class TestParametersAlwaysEmitted(unittest.TestCase):
    def test_explicit_empty_object_is_emitted(self):
        d = tool_to_dict(FunctionTool(name="noop", parameters={}))
        self.assertIn("parameters", d)
        self.assertEqual(d["parameters"], {})

    def test_explicit_empty_object_round_trips_verbatim(self):
        wire = {"type": "function", "name": "noop", "parameters": {}}
        self.assertEqual(tool_to_dict(tool_from_dict(wire)), wire)

    def test_default_schema_is_emitted(self):
        d = tool_to_dict(FunctionTool(name="noop"))
        self.assertEqual(d["parameters"], DEFAULT_SCHEMA)

    def test_absent_on_input_restores_default_schema(self):
        t = tool_from_dict({"type": "function", "name": "noop"})
        self.assertEqual(t.parameters, DEFAULT_SCHEMA)

    def test_nonempty_schema_round_trips(self):
        wire = {
            "type": "function",
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        self.assertEqual(tool_to_dict(tool_from_dict(wire)), wire)


class TestConfigNestsReject(unittest.TestCase):
    def test_non_dict_tool_choice_raises(self):
        for bad in ("auto", 1, True, ["auto"]):
            with self.assertRaises(TypeError):
                config_from_dict({"tool_choice": bad})

    def test_non_dict_reasoning_raises(self):
        for bad in ("high", 0, False, []):
            with self.assertRaises(TypeError):
                config_from_dict({"reasoning": bad})

    def test_non_dict_cache_raises(self):
        for bad in ("off", 2.5, [{}]):
            with self.assertRaises(TypeError):
                config_from_dict({"cache": bad})

    def test_null_nests_still_mean_absent(self):
        c = config_from_dict({"tool_choice": None, "reasoning": None, "cache": None})
        self.assertIsNone(c.tool_choice)
        self.assertIsNone(c.reasoning)
        self.assertIsNone(c.cache)

    def test_valid_nests_still_parse(self):
        c = config_from_dict({"reasoning": {"effort": "low"}, "cache": {"mode": "off"}})
        self.assertEqual(c.reasoning.effort, "low")
        self.assertEqual(c.cache.mode, "off")


if __name__ == "__main__":
    unittest.main()
