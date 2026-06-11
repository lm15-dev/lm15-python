#!/usr/bin/env python3
"""Compare lm15-python provider requests against live-tested curl fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_sdk.dump_request import dump_request

CASES_FILE = ROOT / "cross_sdk" / "test_cases.json"
FIXTURE_ROOT = ROOT / "provider_requests" / "cases"
REPORT_DIR = ROOT / "reports"

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    provider: str
    status: str
    reason: str | None = None
    expected: JsonObject | None = None
    actual: JsonObject | None = None


def load_logical_cases() -> list[JsonObject]:
    data = json.loads(CASES_FILE.read_text())
    return list(data.get("cases", []))


def load_fixture(case_id: str) -> JsonObject | None:
    provider, feature = case_id.split(".", 1)
    path = FIXTURE_ROOT / provider / f"{feature}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def expected_request(fixture: JsonObject) -> JsonObject:
    req = fixture["request"]
    url, params = split_url(str(req["url"]))
    if isinstance(req.get("params"), dict):
        params.update(req["params"])
    headers = normalize_headers(req.get("headers", {}))
    body = req.get("body")
    return clean_request({
        "method": req.get("method", "POST"),
        "url": url,
        "params": params,
        "headers": headers,
        "body": body,
    })


def normalize_actual(actual: JsonObject) -> JsonObject:
    return clean_request({
        "method": actual.get("method"),
        "url": actual.get("url"),
        "params": actual.get("params") or {},
        "headers": normalize_headers(actual.get("headers", {})),
        "body": actual.get("body"),
    })


def split_url(url: str) -> tuple[str, JsonObject]:
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    base = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return base, params


def normalize_headers(headers: JsonObject) -> JsonObject:
    out: JsonObject = {}
    for key, value in headers.items():
        lower = str(key).lower()
        if lower in {"authorization", "x-api-key", "x-goog-api-key"}:
            out[lower] = "REDACTED"
        else:
            out[lower] = value
    return out


def clean_request(req: JsonObject) -> JsonObject:
    """Normalize away transport-noise while preserving semantic wire shape."""
    out = {k: v for k, v in req.items() if v not in (None, {}, [])}

    headers = out.get("headers")
    if isinstance(headers, dict):
        out["headers"] = {
            k: v
            for k, v in sorted(headers.items())
            if k not in {"user-agent", "accept", "accept-encoding", "content-length", "host"}
        }

    body = out.get("body")
    if isinstance(body, dict):
        out["body"] = drop_empty(body)
    return out


def drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {k: drop_empty(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, {}, [])}
    if isinstance(value, list):
        return [drop_empty(v) for v in value]
    return value


def compare_case(case: JsonObject) -> CaseResult:
    case_id = str(case["id"])
    provider = case_id.split(".", 1)[0]
    fixture = load_fixture(case_id)
    if fixture is None:
        return CaseResult(case_id, provider, "missing_fixture", "no provider fixture found")

    try:
        expected = expected_request(fixture)
        actual = normalize_actual(dump_request(case))
    except Exception as exc:
        return CaseResult(case_id, provider, "error", str(exc))

    if expected == actual:
        return CaseResult(case_id, provider, "pass", expected=expected, actual=actual)

    return CaseResult(
        case_id,
        provider,
        "fail",
        first_difference(expected, actual),
        expected=expected,
        actual=actual,
    )


def first_difference(expected: Any, actual: Any, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: missing={missing} extra={extra}"
        for key in sorted(expected):
            diff = first_difference(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: len {len(expected)} != {len(actual)}"
        for idx, (left, right) in enumerate(zip(expected, actual)):
            diff = first_difference(left, right, f"{path}[{idx}]")
            if diff:
                return diff
        return ""
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return ""


def result_to_dict(result: CaseResult) -> JsonObject:
    return {
        "id": result.case_id,
        "provider": result.provider,
        "status": result.status,
        "reason": result.reason,
        "expected": result.expected,
        "actual": result.actual,
    }


def write_markdown(results: list[CaseResult], path: Path) -> None:
    counts: JsonObject = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    lines = [
        "# Request fixture conformance",
        "",
        "Generated by `conformance/check_request_fixtures.py`.",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in ("pass", "fail", "error", "missing_fixture"):
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    lines.extend([
        "",
        "## Cases",
        "",
        "| Case | Provider | Status | First difference/error |",
        "|---|---|---|---|",
    ])
    for result in results:
        reason = (result.reason or "").replace("|", "\\|")
        lines.append(f"| `{result.case_id}` | {result.provider} | {result.status} | {reason} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run only one case id")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on failures/gaps")
    parser.add_argument("--json", type=Path, help="write machine-readable report JSON")
    parser.add_argument("--markdown", type=Path, help="write markdown report")
    args = parser.parse_args(argv)

    cases = load_logical_cases()
    if args.case:
        cases = [case for case in cases if case.get("id") == args.case]
        if not cases:
            raise SystemExit(f"unknown case: {args.case}")

    results = [compare_case(case) for case in cases]
    counts: JsonObject = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    REPORT_DIR.mkdir(exist_ok=True)
    json_path = args.json or REPORT_DIR / "request-fixtures.json"
    md_path = args.markdown or REPORT_DIR / "request-fixtures.md"

    report = {
        "summary": counts,
        "total": len(results),
        "results": [result_to_dict(result) for result in results],
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(results, md_path)

    print(f"request fixture conformance: {counts} / total={len(results)}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")

    if args.strict and any(result.status != "pass" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
