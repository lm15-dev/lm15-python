from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import get_args, get_type_hints

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.types import (
    AudioFormat,
    DataSourceType,
    FinishReason,
    LiveClientEvent,
    LiveServerEvent,
    PartDeltaType,
    PartType,
    Role,
    StreamEventType,
)

ROOT = Path(__file__).resolve().parents[1]
PORTABILITY_SPEC_VERSION = "v2"
CONTRACT = ROOT / "spec" / "contract" / f"{PORTABILITY_SPEC_VERSION}.json"
FIXTURES = ROOT / "spec" / "fixtures" / PORTABILITY_SPEC_VERSION


class PortabilitySpecTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text())

    def test_contract_version_is_frozen(self):
        self.assertEqual(self.contract["contract_version"], PORTABILITY_SPEC_VERSION)
        self.assertEqual(self.contract["versioning"]["policy"], "additive-only")

    def test_contract_discriminators_match_runtime(self):
        audio_hints = get_type_hints(AudioFormat)
        client_hints = get_type_hints(LiveClientEvent)
        server_hints = get_type_hints(LiveServerEvent)
        expected = {
            "role": list(get_args(Role)),
            "part_type": list(get_args(PartType)),
            "data_source_type": list(get_args(DataSourceType)),
            "finish_reason": list(get_args(FinishReason)),
            "stream_event_type": list(get_args(StreamEventType)),
            "part_delta_type": list(get_args(PartDeltaType)),
            "audio_encoding": list(get_args(audio_hints["encoding"])),
            "live_client_event_type": list(get_args(client_hints["type"])),
            "live_server_event_type": list(get_args(server_hints["type"])),
            "tool_type": ["function", "builtin"],
            "reasoning_effort": ["low", "medium", "high"],
        }
        self.assertEqual(self.contract["discriminators"], expected)

    def test_root_types_exist_in_defs(self):
        defs = self.contract["$defs"]
        for name in self.contract["root_types"]:
            with self.subTest(name=name):
                self.assertIn(name, defs)

    def test_v2_additions_exist(self):
        defs = self.contract["$defs"]
        self.assertIn("JsonValue", defs)
        self.assertIn("ReasoningConfig", defs)
        self.assertIn("FunctionTool", defs)
        self.assertIn("BuiltinTool", defs)
        self.assertIn("ErrorInfo", defs)
        self.assertEqual(defs["Tool"].get("oneOf"), [
            {"$ref": "#/$defs/FunctionTool"},
            {"$ref": "#/$defs/BuiltinTool"},
        ])
        self.assertEqual(defs["JsonObject"].get("additionalProperties"), {"$ref": "#/$defs/JsonValue"})
        self.assertEqual(defs["Config"]["properties"]["reasoning"], {"$ref": "#/$defs/ReasoningConfig"})
        self.assertEqual(defs["ErrorEvent"]["properties"]["error"], {"$ref": "#/$defs/ErrorInfo"})

    def test_fixture_bundles_target_same_contract_version(self):
        for path in sorted(FIXTURES.glob("*.json")):
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                self.assertEqual(payload["contract_version"], PORTABILITY_SPEC_VERSION)
                self.assertTrue(payload["fixture_version"])


if __name__ == "__main__":
    unittest.main()
