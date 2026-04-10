from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.types import AudioFormat, DataSource, LMRequest, LiveClientEvent, LiveConfig, LiveServerEvent, Message, Part, Usage


class ContractTests(unittest.TestCase):
    def test_datasource_validation(self):
        with self.assertRaises(ValueError):
            DataSource(type="url")
        with self.assertRaises(ValueError):
            DataSource(type="base64", media_type="image/png")

    def test_part_factory(self):
        p = Part.from_dict({"type": "tool_call", "id": "c1", "name": "search", "input": {"q": "x"}})
        self.assertEqual(p.type, "tool_call")
        self.assertEqual(p.input["q"], "x")

    def test_request_requires_messages(self):
        with self.assertRaises(ValueError):
            LMRequest(model="gpt-4.1-mini", messages=())

    def test_message_requires_parts(self):
        with self.assertRaises(ValueError):
            Message(role="user", parts=())

    def test_live_config_validation(self):
        with self.assertRaises(ValueError):
            LiveConfig(model="", system=None)
        with self.assertRaises(ValueError):
            LiveConfig(model="gemini-live", system=())

    def test_audio_format_validation(self):
        with self.assertRaises(ValueError):
            AudioFormat(encoding="pcm16", sample_rate=0)
        with self.assertRaises(ValueError):
            AudioFormat(encoding="pcm16", sample_rate=16000, channels=0)

    def test_live_client_event_validation(self):
        with self.assertRaises(ValueError):
            LiveClientEvent(type="audio")
        with self.assertRaises(ValueError):
            LiveClientEvent(type="text")
        with self.assertRaises(ValueError):
            LiveClientEvent(type="tool_result", content=(Part.text_part("ok"),))
        with self.assertRaises(ValueError):
            LiveClientEvent(type="tool_result", id="call_1")

    def test_live_server_event_validation(self):
        with self.assertRaises(ValueError):
            LiveServerEvent(type="audio")
        with self.assertRaises(ValueError):
            LiveServerEvent(type="text")
        with self.assertRaises(ValueError):
            LiveServerEvent(type="tool_call", id="call_1", name="search")
        with self.assertRaises(ValueError):
            LiveServerEvent(type="turn_end")
        with self.assertRaises(ValueError):
            LiveServerEvent(type="error")
        ok = LiveServerEvent(type="turn_end", usage=Usage(total_tokens=1))
        self.assertEqual(ok.type, "turn_end")


if __name__ == "__main__":
    unittest.main()
