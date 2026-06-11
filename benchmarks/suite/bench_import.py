"""Cold-process import time + import RSS delta, per scratch venv.

Import time: wall time of ``python -c "<import>"`` in a fresh process,
best of N (best-of is the right statistic for a cold-start floor: it
removes scheduler noise while every run still pays the full import cost
because the process is new each time).  A ``python -c pass`` baseline is
recorded for context.

Memory: VmRSS (from /proc/self/status) after the import, minus the median
VmRSS of a bare interpreter in the same venv.
"""

from __future__ import annotations

import statistics
import subprocess
import time

from venvs import Venv

_RSS_SNIPPET = (
    "{imp}\n"
    "rss = 0\n"
    "for line in open('/proc/self/status'):\n"
    "    if line.startswith('VmRSS:'):\n"
    "        rss = int(line.split()[1])\n"
    "print(rss)\n"
)


def _wall(python: str, code: str) -> float:
    t0 = time.perf_counter()
    subprocess.run([python, "-c", code], check=True, capture_output=True)
    return time.perf_counter() - t0


def measure_import(v: Venv, n: int = 10) -> dict:
    py = str(v.python)
    # one warmup to populate OS page cache / .pyc files
    _wall(py, v.import_stmt)
    baseline = min(_wall(py, "pass") for _ in range(n))
    times = [_wall(py, v.import_stmt) for _ in range(n)]
    return {
        "n": n,
        "best_s": min(times),
        "median_s": statistics.median(times),
        "interpreter_baseline_s": baseline,
    }


def _rss_kib(python: str, imp: str) -> int:
    out = subprocess.run(
        [python, "-c", _RSS_SNIPPET.format(imp=imp)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def measure_rss(v: Venv, n: int = 5) -> dict:
    py = str(v.python)
    base = statistics.median(_rss_kib(py, "pass") for _ in range(n))
    with_import = statistics.median(_rss_kib(py, v.import_stmt) for _ in range(n))
    return {
        "n": n,
        "baseline_kib": base,
        "with_import_kib": with_import,
        "delta_kib": with_import - base,
    }
