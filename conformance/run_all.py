#!/usr/bin/env python3
"""Run every conformance check and aggregate the results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_doc_drift  # noqa: E402
import check_endpoint_fixtures  # noqa: E402
import check_error_fixtures  # noqa: E402
import check_request_fixtures  # noqa: E402
import check_response_fixtures  # noqa: E402
import check_serde_fixtures  # noqa: E402

REPORT_DIR = ROOT / "reports"
JsonObject = dict[str, Any]


CHECKS: list[tuple[str, Callable[[list[str] | None], int]]] = [
    ("request_fixtures", check_request_fixtures.main),
    ("response_fixtures", check_response_fixtures.main),
    ("error_fixtures", check_error_fixtures.main),
    ("endpoint_fixtures", check_endpoint_fixtures.main),
    ("serde_fixtures", check_serde_fixtures.main),
    ("doc_drift", check_doc_drift.main),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any check fails")
    parser.add_argument("--quiet", action="store_true", help="suppress per-check stdout")
    args = parser.parse_args(argv)

    REPORT_DIR.mkdir(exist_ok=True)
    summary: JsonObject = {}
    failed = 0
    started = time.time()

    for name, fn in CHECKS:
        check_args: list[str] = []
        if args.strict:
            check_args.append("--strict")
        if args.quiet:
            check_args.extend(["--json", str(REPORT_DIR / f"{name}.json"), "--markdown", str(REPORT_DIR / f"{name}.md")])
        print(f"\n=== {name} ===")
        rc = fn(check_args)
        summary[name] = {"exit_code": rc}
        if rc != 0:
            failed += 1

    elapsed = round(time.time() - started, 2)
    print()
    print("=== conformance summary ===")
    for name, info in summary.items():
        flag = "OK" if info["exit_code"] == 0 else "FAIL"
        print(f"  {flag:>4} {name}")
    print(f"elapsed: {elapsed}s")

    (REPORT_DIR / "summary.json").write_text(
        json.dumps({"summary": summary, "elapsed_seconds": elapsed}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return 1 if failed and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
