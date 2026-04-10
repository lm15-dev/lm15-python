from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.types import DataSourceType, FinishReason, PartDeltaType, PartType, Role, StreamEventType

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "spec" / "contract" / "v1.json"
FIXTURES = ROOT / "spec" / "fixtures" / "v1"


class PortabilitySpecTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text())

    def test_contract_version_is_frozen(self):
        self.assertEqual(self.contract["contract_version"], "v1")
        self.assertEqual(self.contract["versioning"]["policy"], "additive-only")

    def test_contract_discriminators_match_runtime(self):
        expected = {
            "role": list(get_args(Role)),
            "part_type": list(get_args(PartType)),
            "data_source_type": list(get_args(DataSourceType)),
            "finish_reason": list(get_args(FinishReason)),
            "stream_event_type": list(get_args(StreamEventType)),
            "part_delta_type": list(get_args(PartDeltaType)),
        }
        self.assertEqual(self.contract["discriminators"], expected)

    def test_root_types_exist_in_defs(self):
        defs = self.contract["$defs"]
        for name in self.contract["root_types"]:
            with self.subTest(name=name):
                self.assertIn(name, defs)

    def test_fixture_bundles_target_same_contract_version(self):
        for path in sorted(FIXTURES.glob("*.json")):
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                self.assertEqual(payload["contract_version"], "v1")
                self.assertTrue(payload["fixture_version"])


if __name__ == "__main__":
    unittest.main()
