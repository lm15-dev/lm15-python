"""Offline smoke test for the vet shim (`python -m lm15.vet`).

Exercises every PROTOCOL.md op once, end to end through the JSONL framing,
against real corpus data from lm15-contract. The shim never touches the
network — every body comes from saved fixtures.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT.parent / "lm15-contract"

pytestmark = pytest.mark.skipif(
    not CONTRACT_ROOT.exists(), reason="lm15-contract corpus not checked out"
)

ALL_OPS = [
    "build_request",
    "capabilities",
    "normalize_error",
    "parse_response",
    "replay_stream",
    "serde_roundtrip",
    "surface_dump",
    "validate",
]


def load_case(provider: str, feature: str) -> dict:
    return json.loads((CONTRACT_ROOT / "cases" / provider / f"{feature}.json").read_text())


def pinned_body_b64(case: dict) -> str:
    body = (CONTRACT_ROOT / "bodies" / case["id"] / case["pinned_body"]).read_bytes()
    return base64.b64encode(body).decode("ascii")


def run_shim(requests: list[dict]) -> dict[str, dict]:
    """Drive the shim over stdin/stdout and return replies keyed by id."""
    stdin = "".join(json.dumps(req) + "\n" for req in requests)
    proc = subprocess.run(
        [sys.executable, "-m", "lm15.vet"],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert len(lines) == len(requests), f"one reply per request expected: {proc.stdout!r}"
    replies = [json.loads(line) for line in lines]
    assert [r["id"] for r in replies] == [r["id"] for r in requests], "replies must keep order"
    return {reply["id"]: reply for reply in replies}


def ok_result(reply: dict) -> dict:
    assert reply["ok"] is True, reply
    return reply["result"]


@pytest.fixture(scope="module")
def replies() -> dict[str, dict]:
    basic = load_case("openai", "basic_text")
    streaming = load_case("openai", "streaming")
    error_cases = json.loads((CONTRACT_ROOT / "errors" / "cases" / "openai.json").read_text())["cases"]
    auth_case = next(c for c in error_cases if c["id"] == "openai.auth_invalid_key")
    serde_cases = json.loads((CONTRACT_ROOT / "serde" / "canonical.json").read_text())["cases"]

    requests = [
        {"op": "capabilities", "id": "capabilities"},
        {
            "op": "build_request",
            "id": "build_request",
            "provider": "openai",
            "canonical_request": basic["canonical_request"],
            "stream": False,
            "api_key": "test-key-123",
        },
        {
            "op": "parse_response",
            "id": "parse_response",
            "provider": "openai",
            "canonical_request": basic["canonical_request"],
            "status": 200,
            "body_b64": pinned_body_b64(basic),
        },
        {
            "op": "replay_stream",
            "id": "replay_stream",
            "provider": "openai",
            "canonical_request": streaming["canonical_request"],
            "body_b64": pinned_body_b64(streaming),
        },
        {
            "op": "normalize_error",
            "id": "normalize_error",
            "provider": "openai",
            "status": auth_case["status"],
            "body_text": json.dumps(auth_case["body"]),
        },
        {"op": "validate", "id": "validate_accept", "kind": "part", "value": {"type": "text", "text": "hi"}},
        {"op": "validate", "id": "validate_reject", "kind": "part", "value": {"type": "bogus"}},
        {"op": "surface_dump", "id": "surface_dump"},
    ]
    for case in serde_cases:
        requests.append(
            {"op": "serde_roundtrip", "id": f"serde:{case['id']}", "kind": case["kind"], "value": case["value"]}
        )

    out = run_shim(requests)
    out["_serde_cases"] = serde_cases  # type: ignore[assignment]
    out["_basic_case"] = basic  # type: ignore[assignment]
    out["_auth_case"] = auth_case  # type: ignore[assignment]
    return out


def test_capabilities(replies: dict[str, dict]) -> None:
    result = ok_result(replies["capabilities"])
    assert result["language"] == "python"
    assert sorted(result["ops"]) == ALL_OPS
    assert isinstance(result["impl_version"], str) and result["impl_version"]


def test_build_request(replies: dict[str, dict]) -> None:
    result = ok_result(replies["build_request"])
    expected = replies["_basic_case"]["request"]
    assert result["method"] == "POST"
    assert result["url"] == expected["url"]
    assert "?" not in result["url"]
    assert result["params"] == {}
    # Auth header verbatim — exact formatting against the injected api_key.
    assert result["headers"]["authorization"] == "Bearer test-key-123"
    assert result["headers"]["content-type"] == "application/json"
    assert result["body"] == expected["body"]


def test_parse_response(replies: dict[str, dict]) -> None:
    result = ok_result(replies["parse_response"])
    assert "unmapped" not in result
    response = result["canonical_response"]
    assert "provider_data" not in response
    assert response["finish_reason"] == "stop"
    assert response["message"]["role"] == "assistant"
    texts = [p for p in response["message"]["parts"] if p["type"] == "text"]
    assert texts and texts[0]["text"]
    assert response["usage"]["input_tokens"] > 0


def test_replay_stream(replies: dict[str, dict]) -> None:
    result = ok_result(replies["replay_stream"])
    assert "unmapped" not in result
    events = result["events"]
    assert events, "expected a non-empty canonical event trace"
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "end"
    assert any(e["type"] == "delta" and e["delta"]["type"] == "text" for e in events)
    response = result["canonical_response"]
    assert response["finish_reason"] == "stop"
    assert any(p["type"] == "text" and p["text"] for p in response["message"]["parts"])


def test_normalize_error(replies: dict[str, dict]) -> None:
    result = ok_result(replies["normalize_error"])
    expected = replies["_auth_case"]["expected"]
    assert result["class"] == expected["class"]
    assert result["code"] == expected["code"]
    assert result["provider_code"] == expected["provider_code"]
    assert expected and result["message"]


def test_serde_roundtrip_canonical_corpus(replies: dict[str, dict]) -> None:
    serde_cases = replies["_serde_cases"]
    assert serde_cases, "canonical serde corpus is empty"
    for case in serde_cases:
        result = ok_result(replies[f"serde:{case['id']}"])
        assert result["value"] == case["value"], case["id"]


def test_validate(replies: dict[str, dict]) -> None:
    accepted = ok_result(replies["validate_accept"])
    assert accepted["ok"] is True
    assert accepted["normalized"] == {"type": "text", "text": "hi"}

    rejected = replies["validate_reject"]
    assert rejected["ok"] is False
    assert rejected["error"]["type"]
    assert rejected["error"]["message"]


def test_surface_dump(replies: dict[str, dict]) -> None:
    result = ok_result(replies["surface_dump"])
    types = result["types"]
    enums = result["enums"]
    assert len(types) > 30
    assert types["Request"]["fields"] == ["model", "messages", "system", "tools", "config"]
    assert "text" in types["TextPart"]["fields"]
    assert "stop" in enums["FINISH_REASONS"]
    assert "tool_call" in enums["PART_TYPES"]
