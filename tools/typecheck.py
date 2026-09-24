"""The type-checking gate: what users see is clean; the rest may only improve.

Two checks, both with the mypy pinned in requirements/typecheck.txt:

1. **User programs (strict, zero errors).** ``typecheck/examples/`` holds real
   programs written against lm15's public API: the Python programs the
   documentation site shows, as a reader copies them. They are checked with
   ``--strict`` (except the rules about annotating the example's own
   functions), importing lm15 as an installed, typed package. Any error fails:
   it is a type a user's checker would reject in code the docs told them to
   write.

2. **lm15's own code (ratchet).** mypy's findings inside ``lm15/`` are compared
   with ``typecheck/baseline.txt``: the known findings, one per line, without
   line numbers (``path|code|message``), with a count. A finding not in the
   baseline fails. So does a baseline entry that no longer occurs: the fix is
   real, so record it (``--update``) and the number only goes down.

    python tools/typecheck.py              # both checks
    python tools/typecheck.py --update     # rewrite the baseline to today's findings
    python tools/typecheck.py --refresh-examples ../website
                                           # re-copy the docs programs from a website checkout

--update is for recording fixes. Adding a finding to the baseline to make CI
pass is a decision to ship a known type error; say so in the commit.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "typecheck" / "examples"
BASELINE = ROOT / "typecheck" / "baseline.txt"
USER_FLAGS = ["--strict", "--allow-untyped-defs", "--allow-incomplete-defs", "--allow-untyped-calls",
              "--follow-imports=silent", "--python-version", "3.10", "--no-incremental"]
PACKAGE_FLAGS = ["--python-version", "3.10", "--no-incremental", "--hide-error-context", "--no-pretty",
                 "--show-error-codes", "--no-error-summary"]
LINE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+): error: (?P<message>.*?)  \[(?P<code>[a-z-]+)\]$")


def mypy(args: list[str]) -> list[str]:
    done = subprocess.run([sys.executable, "-m", "mypy", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if done.returncode not in (0, 1):
        raise SystemExit(f"mypy failed to run:\n{done.stdout}{done.stderr}")
    return [line for line in done.stdout.splitlines() if ": error:" in line]


def check_examples() -> int:
    files = sorted(str(p.relative_to(ROOT)) for p in EXAMPLES.glob("*.py"))
    if not files:
        raise SystemExit(f"no example programs in {EXAMPLES}")
    errors = mypy([*USER_FLAGS, *files])
    for line in errors:
        print(line)
    print(f"user programs: {len(files)} checked, {len(errors)} error(s)")
    return 1 if errors else 0


def package_findings() -> collections.Counter[str]:
    found: collections.Counter[str] = collections.Counter()
    for line in mypy([*PACKAGE_FLAGS, "lm15"]):
        m = LINE.match(line)
        if m is None:
            raise SystemExit(f"unparsed mypy line: {line}")
        found[f"{m['path']}|{m['code']}|{m['message']}"] += 1
    return found


def read_baseline() -> collections.Counter[str]:
    known: collections.Counter[str] = collections.Counter()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            count, key = line.split(" ", 1)
            known[key] = int(count)
    return known


def write_baseline(found: collections.Counter[str]) -> None:
    header = ("# Known mypy findings inside lm15/ (tools/typecheck.py). Count, then path|code|message.\n"
              "# The gate fails on anything new, and on any entry that no longer occurs.\n"
              f"# Total: {sum(found.values())}\n")
    BASELINE.write_text(header + "".join(f"{n} {k}\n" for k, n in sorted(found.items())), encoding="utf-8")


def check_package(update: bool) -> int:
    found = package_findings()
    if update:
        write_baseline(found)
        print(f"baseline rewritten: {sum(found.values())} known finding(s)")
        return 0
    known = read_baseline()
    new = found - known
    fixed = known - found
    for key, n in sorted(new.items()):
        print(f"NEW ({n}x): {key}")
    for key, n in sorted(fixed.items()):
        print(f"FIXED ({n}x), remove from the baseline: {key}")
    print(f"lm15 package: {sum(found.values())} finding(s), baseline {sum(known.values())}, "
          f"{sum(new.values())} new, {sum(fixed.values())} fixed")
    if new:
        print("A new type error inside lm15/. Fix it; the baseline is not a place to add errors.")
    if fixed and not new:
        print("Fewer errors than recorded: run `python tools/typecheck.py --update` and commit the baseline.")
    return 1 if new or fixed else 0


def refresh_examples(website: Path) -> None:
    """The docs' Python programs, as the site generates and shows them."""
    script = (
        "import { tourPrograms } from './src/data/tour-examples.ts';"
        "console.log(JSON.stringify(tourPrograms('python', 'anthropic', 'claude-haiku-4-5')));"
    )
    out = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", script],
                         cwd=website, capture_output=True, text=True, check=True, encoding="utf-8").stdout
    for old in EXAMPLES.glob("*.py"):
        old.unlink()
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for program in json.loads(out):
        name = re.sub(r"[^a-z0-9]+", "_", program["name"].lower())
        (EXAMPLES / f"docs_{name}.py").write_text(program["source"].rstrip("\n") + "\n", encoding="utf-8")
    # Pages whose examples are steps of one session: joined in page order.
    sessions = {
        "judgments_page": ("judgments", ["answers", "ask", "expected", "chat", "required", "table"]),
        "authentication_page": ("authentication", ["request", "doctor", "explicit", "rotating", "subscription",
                                                    "managed", "configure", "status", "methods", "logout"]),
        "authentication_connect": ("authentication", ["connect"]),
        "authentication_login": ("authentication", ["login"]),
    }
    for target, (folder, steps) in sessions.items():
        parts = [(website / "src/data" / folder / f"{s}.py").read_text(encoding="utf-8").rstrip("\n") for s in steps]
        (EXAMPLES / f"{target}.py").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"examples refreshed from {website}: {len(list(EXAMPLES.glob('*.py')))} programs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true", help="rewrite typecheck/baseline.txt to today's findings")
    parser.add_argument("--refresh-examples", type=Path, metavar="WEBSITE", help="re-copy the docs programs")
    args = parser.parse_args()
    if args.refresh_examples:
        refresh_examples(args.refresh_examples.resolve())
    status = check_examples()
    status |= check_package(args.update)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
