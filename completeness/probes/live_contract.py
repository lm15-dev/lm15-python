from __future__ import annotations

from pathlib import Path

from lm15.serde import (
    live_client_event_from_dict,
    live_client_event_to_dict,
    live_config_from_dict,
    live_config_to_dict,
    live_server_event_from_dict,
    live_server_event_to_dict,
)

from ._helpers import ProbeResult, load_portability_fixture


def run(test: dict, root: Path) -> ProbeResult:
    bundle = load_portability_fixture(root, "live.json")

    for case in bundle.get("config_cases", []):
        if live_config_to_dict(live_config_from_dict(case["config"])) != case["config"]:
            return ProbeResult(status="fail", details=f"live config fixture mismatch: {case['id']}")

    for case in bundle.get("client_event_cases", []):
        if live_client_event_to_dict(live_client_event_from_dict(case["event"])) != case["event"]:
            return ProbeResult(status="fail", details=f"live client event fixture mismatch: {case['id']}")

    for case in bundle.get("server_event_cases", []):
        if live_server_event_to_dict(live_server_event_from_dict(case["event"])) != case["event"]:
            return ProbeResult(status="fail", details=f"live server event fixture mismatch: {case['id']}")

    for case in bundle.get("session_cases", []):
        if live_config_to_dict(live_config_from_dict(case["config"])) != case["config"]:
            return ProbeResult(status="fail", details=f"live session config fixture mismatch: {case['id']}")
        client_events = [live_client_event_to_dict(live_client_event_from_dict(e)) for e in case.get("client_events", [])]
        server_events = [live_server_event_to_dict(live_server_event_from_dict(e)) for e in case.get("server_events", [])]
        if client_events != case.get("client_events", []):
            return ProbeResult(status="fail", details=f"live session client fixture mismatch: {case['id']}")
        if server_events != case.get("server_events", []):
            return ProbeResult(status="fail", details=f"live session server fixture mismatch: {case['id']}")

    return ProbeResult(status="pass", details="live contract fixtures roundtrip through canonical serde")
