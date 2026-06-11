"""Steady-state per-call wall time: lm15 (pooled keep-alive transport)
vs raw urllib (new TLS/TCP connection every call).

The claim under test: at steady state lm15 is FASTER than naive raw HTTP
against a remote TLS endpoint, because its connection pool amortizes the
handshake across calls while urllib.request pays it on every call.  Both
sides POST byte-identical /chat/completions requests (non-streaming,
max_tokens=4); 1 warmup pair, then N interleaved pairs; medians reported.

Targets: Groq (remote TLS; GROQ_API_KEY from the environment or
/home/maxime/Projects/lm15-dev/.env) and local ollama (plain HTTP; the
handshake saving is expected to be small there).  Each target skips
gracefully when unreachable.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_FILE = Path("/home/maxime/Projects/lm15-dev/.env")
GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA = "http://localhost:11434"
# Groq's edge 403s urllib's default python-urllib User-Agent.
UA = "curl/8.5.0"


def _env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    except OSError:
        pass
    return None


def _pick_ollama_model() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2) as r:
            models = json.loads(r.read()).get("models", [])
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    if not models:
        return None
    return min(models, key=lambda m: m.get("size", 0))["name"]


def _chat_body(model: str) -> bytes:
    return json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }).encode()


def _run_urllib(url: str, body: bytes, headers: dict[str, str]) -> float:
    """One full call on a FRESH connection (urllib opens a new socket)."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()
    return time.perf_counter() - t0


def _measure_target(*, base_url: str, model: str, api_key: str | None,
                    compat: str, n: int) -> dict:
    from lm15 import Config, Message, Request
    from lm15.providers.openai_chat import OpenAIChatLM

    lm = OpenAIChatLM(api_key=api_key or "unused", compat=compat)
    request = Request(model=model, messages=(Message.user("Say hi."),),
                      config=Config(max_tokens=4, temperature=0.0))
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = _chat_body(model)

    def one_lm15() -> float:
        t0 = time.perf_counter()
        lm.complete(request)
        return time.perf_counter() - t0

    # warmup pair: establishes lm15's pooled connection, primes both paths
    _run_urllib(url, body, headers)
    one_lm15()

    lm15_s, raw_s = [], []
    for _ in range(n):  # interleaved so server-side drift hits both equally
        raw_s.append(_run_urllib(url, body, headers))
        lm15_s.append(one_lm15())
    lm.transport.close()
    med = statistics.median
    return {
        "skipped": False,
        "model": model,
        "n": n,
        "lm15_median_ms": round(med(lm15_s) * 1000, 2),
        "urllib_median_ms": round(med(raw_s) * 1000, 2),
        "delta_ms": round((med(lm15_s) - med(raw_s)) * 1000, 2),
        "lm15_spread_ms": round((max(lm15_s) - min(lm15_s)) * 1000, 2),
        "urllib_spread_ms": round((max(raw_s) - min(raw_s)) * 1000, 2),
    }


def run(n: int = 10) -> dict:
    out: dict = {}

    groq_key = _env_key("GROQ_API_KEY")
    if groq_key is None:
        out["groq"] = {"skipped": True, "reason": "GROQ_API_KEY not available"}
    else:
        try:
            out["groq"] = _measure_target(
                base_url="https://api.groq.com/openai/v1", model=GROQ_MODEL,
                api_key=groq_key, compat="groq", n=n)
        except Exception as exc:
            out["groq"] = {"skipped": True, "reason": f"groq call failed: {exc!r}"}

    model = _pick_ollama_model()
    if model is None:
        out["ollama"] = {"skipped": True,
                         "reason": "ollama not reachable at " + OLLAMA}
    else:
        try:
            out["ollama"] = _measure_target(
                base_url=f"{OLLAMA}/v1", model=model, api_key=None,
                compat="ollama", n=n)
        except Exception as exc:
            out["ollama"] = {"skipped": True,
                             "reason": f"ollama call failed: {exc!r}"}
    return out
