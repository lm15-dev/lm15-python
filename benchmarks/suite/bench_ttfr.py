"""TTFR tax: time-to-first-byte against a LOCAL ollama server, lm15's
StdlibTransport vs raw urllib.request, identical request bytes.

The claim under test: lm15's transport layer adds ~0 over raw stdlib
HTTP.  Both clients POST the same streaming /api/generate request to
http://localhost:11434 and we time headers-in -> first body byte, plus
the full request wall time.  Median of N, runs interleaved so server
warm-up affects both sides equally.  Skips gracefully if ollama is down.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"


def _pick_model() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2) as r:
            models = json.loads(r.read()).get("models", [])
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    if not models:
        return None
    return min(models, key=lambda m: m.get("size", 0))["name"]


def _body(model: str) -> bytes:
    return json.dumps({
        "model": model,
        "prompt": "Say hi.",
        "stream": True,
        "think": False,  # reasoning effort: none
        "options": {"num_predict": 8, "temperature": 0.0},
    }).encode()


def _run_urllib(body: bytes) -> tuple[float, float]:
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        first = None
        while True:
            # read1: return as soon as any bytes arrive (read() would block
            # for the full 8192 and turn TTFB into total time)
            chunk = resp.read1(8192)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter()
    t1 = time.perf_counter()
    return (first - t0), (t1 - t0)


def _run_lm15(body: bytes) -> tuple[float, float]:
    from lm15.transports import StdlibTransport, TransportRequest
    tr = StdlibTransport()
    req = TransportRequest(method="POST", url=f"{OLLAMA}/api/generate",
                           headers=[("Content-Type", "application/json")],
                           body=body)
    t0 = time.perf_counter()
    with tr.stream(req) as resp:
        first = None
        for chunk in resp:
            if first is None:
                first = time.perf_counter()
    t1 = time.perf_counter()
    tr.close()
    return (first - t0), (t1 - t0)


def run(n: int = 9) -> dict:
    model = _pick_model()
    if model is None:
        return {"skipped": True, "reason": "ollama not reachable at " + OLLAMA}
    body = _body(model)
    # warmup: load the model once
    _run_urllib(body)
    _run_lm15(body)
    lm15_ttfb, lm15_total, raw_ttfb, raw_total = [], [], [], []
    for _ in range(n):  # interleaved
        f, t = _run_urllib(body)
        raw_ttfb.append(f); raw_total.append(t)
        f, t = _run_lm15(body)
        lm15_ttfb.append(f); lm15_total.append(t)
    med = statistics.median
    spread = lambda xs: round((max(xs) - min(xs)) * 1000, 2)  # noqa: E731
    return {
        "lm15_ttfb_spread_ms": spread(lm15_ttfb),
        "urllib_ttfb_spread_ms": spread(raw_ttfb),
        "skipped": False,
        "model": model,
        "n": n,
        "lm15_ttfb_ms": round(med(lm15_ttfb) * 1000, 2),
        "urllib_ttfb_ms": round(med(raw_ttfb) * 1000, 2),
        "ttfb_tax_ms": round((med(lm15_ttfb) - med(raw_ttfb)) * 1000, 2),
        "lm15_total_ms": round(med(lm15_total) * 1000, 2),
        "urllib_total_ms": round(med(raw_total) * 1000, 2),
    }
