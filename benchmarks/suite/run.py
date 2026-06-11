#!/usr/bin/env python
"""lm15 canonical benchmark suite — single entrypoint.

    .venv/bin/python benchmarks/suite/run.py [--quick] [--keep-venvs]

Runs, in order:
  1. scratch venv builds (lm15 wheel + competitors) under /tmp/lm15-bench-venvs/
  2. install footprint (size on disk, transitive dep count)
  3. cold-process import time + import RSS delta
  4. lm15 hot-path microbenchmarks (build/parse/serde/stream)
  5. TTFR tax vs raw urllib against local ollama (skips if down)

Emits benchmarks/RESULTS.json and regenerates benchmarks/BENCHMARKS.md and
the README "Footprint" section from it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# hot-path benchmarks import lm15 from the repo checkout
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import bench_concurrency
import bench_footprint
import bench_hotpath
import bench_import
import bench_live
import bench_steady_state
import bench_ttfr
import report
import venvs

BENCH_DIR = Path(__file__).resolve().parents[1]


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    return {
        "cpu": _cpu_model(),
        "cpu_count": __import__("os").cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "uv": uv.stdout.strip().removeprefix("uv "),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer iterations")
    ap.add_argument("--keep-venvs", action="store_true",
                    help="reuse existing scratch venvs (skip reinstall)")
    args = ap.parse_args()

    import_n = 5 if args.quick else 10
    rss_n = 3 if args.quick else 5
    hot_n, hot_w = (300, 30) if args.quick else (2000, 200)
    ttfr_n = 3 if args.quick else 9
    steady_n = 4 if args.quick else 10
    conc_n = 12 if args.quick else 50

    res: dict = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "environment": environment(),
        "config": {"import_n": import_n, "rss_n": rss_n,
                   "hotpath_n": hot_n, "hotpath_warmup": hot_w,
                   "ttfr_n": ttfr_n, "steady_state_n": steady_n,
                   "concurrency_calls": conc_n, "quick": args.quick},
        "packages": {},
    }

    print("== building scratch venvs ==", flush=True)
    baseline = venvs.make_baseline()
    if not baseline.ok:
        sys.exit(f"baseline venv failed: {baseline.error}")
    built: dict[str, venvs.Venv] = {}
    for name in venvs.PACKAGES:
        v = venvs.make_venv(name, fresh=not args.keep_venvs)
        built[name] = v
        print(f"  {name}: {'ok' if v.ok else 'INSTALL FAILED'}", flush=True)

    for name, v in built.items():
        entry: dict = {"ok": v.ok, "install_target": v.install_target,
                       "import_stmt": v.import_stmt}
        if not v.ok:
            entry["error"] = v.error
            res["packages"][name] = entry
            continue
        print(f"== {name}: footprint / import / rss ==", flush=True)
        try:
            # import first: its warmup byte-compiles the package, so the
            # size measurement is identical for fresh and reused venvs
            entry["import"] = bench_import.measure_import(v, n=import_n)
            entry["rss"] = bench_import.measure_rss(v, n=rss_n)
            entry["footprint"] = bench_footprint.measure_footprint(v, baseline)
        except Exception as exc:  # degrade gracefully, keep going
            entry["ok"] = False
            entry["error"] = f"measurement failed: {exc!r}"
        res["packages"][name] = entry

    print("== hot path (lm15) ==", flush=True)
    res["hotpath"] = bench_hotpath.run(n=hot_n, warmup=hot_w)

    print("== ttfr vs raw urllib (local ollama) ==", flush=True)
    res["ttfr"] = bench_ttfr.run(n=ttfr_n)
    if res["ttfr"].get("skipped"):
        print(f"  skipped: {res['ttfr']['reason']}", flush=True)

    print("== steady state: pooled lm15 vs fresh-connection urllib ==", flush=True)
    res["steady_state"] = bench_steady_state.run(n=steady_n)
    for target, st in res["steady_state"].items():
        if st.get("skipped"):
            print(f"  {target}: skipped: {st['reason']}", flush=True)
        else:
            print(f"  {target}: lm15 {st['lm15_median_ms']:.1f} ms vs urllib "
                  f"{st['urllib_median_ms']:.1f} ms (delta {st['delta_ms']:+.1f} ms)",
                  flush=True)

    print("== concurrency: async gather vs sequential sync (local ollama) ==",
          flush=True)
    res["concurrency"] = bench_concurrency.run(n_calls=conc_n)
    if res["concurrency"].get("skipped"):
        print(f"  skipped: {res['concurrency']['reason']}", flush=True)
    else:
        c = res["concurrency"]
        print(f"  {c['n_calls']} calls: async {c['async_gather_wall_s']:.2f}s vs "
              f"sync {c['sync_sequential_wall_s']:.2f}s ({c['speedup_x']}x)",
              flush=True)

    print("== live session round trip (Gemini Live) ==", flush=True)
    res["live"] = bench_live.run()
    if res["live"].get("skipped"):
        print(f"  skipped: {res['live']['reason']}", flush=True)

    out = BENCH_DIR / "RESULTS.json"
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)

    report.main()
    print(f"wrote {BENCH_DIR / 'BENCHMARKS.md'} and README Footprint section",
          flush=True)


if __name__ == "__main__":
    main()
