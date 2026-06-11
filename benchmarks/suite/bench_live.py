"""Best-effort live-session round trip (Gemini Live over WebSocket).

Opens a live session via ``GeminiLM.live`` (the optional ``websockets``
extra), sends one tiny text turn, and times:

- connect+setup: WebSocket open through the server's ``setupComplete``
- first event: text turn sent -> first server event received
- turn total: text turn sent -> ``turn_complete``

Needs GEMINI_API_KEY (environment or /home/maxime/Projects/lm15-dev/.env).
Every failure mode is recorded as skipped-with-reason — a live endpoint is
inherently best-effort.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from bench_steady_state import _env_key

BASE = "https://generativelanguage.googleapis.com/v1beta"


def _live_models(api_key: str) -> list[str]:
    """Models supporting bidiGenerateContent, '-live-' ones first."""
    names: list[str] = []
    token = None
    while True:
        qs = {"key": api_key, "pageSize": "100"}
        if token:
            qs["pageToken"] = token
        with urllib.request.urlopen(
                f"{BASE}/models?{urllib.parse.urlencode(qs)}", timeout=15) as r:
            page = json.loads(r.read())
        for m in page.get("models", []):
            if "bidiGenerateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].removeprefix("models/"))
        token = page.get("nextPageToken")
        if not token:
            break
    return sorted(names, key=lambda n: ("-live-" not in n, n))


def _one_session(api_key: str, model: str) -> dict:
    from lm15.providers.gemini import GeminiLM
    from lm15.types import (
        LiveConfig,
        LiveServerErrorEvent,
        LiveServerTurnEndEvent,
    )

    lm = GeminiLM(api_key=api_key)
    t0 = time.perf_counter()
    session = lm.live(LiveConfig(model=model))
    setup_s = time.perf_counter() - t0
    try:
        t1 = time.perf_counter()
        session.send_turn("Reply with the single word: hi")
        first_s = None
        deadline = time.time() + 30
        while time.time() < deadline:
            event = session.recv()
            if isinstance(event, LiveServerErrorEvent):
                raise RuntimeError(f"server error: {event.error}")
            if first_s is None:
                first_s = time.perf_counter() - t1
            if isinstance(event, LiveServerTurnEndEvent):
                break
        if first_s is None:
            raise TimeoutError("no server event within 30s")
        return {
            "skipped": False,
            "model": model,
            "connect_setup_ms": round(setup_s * 1000, 1),
            "first_event_ms": round(first_s * 1000, 1),
            "turn_total_ms": round((time.perf_counter() - t1) * 1000, 1),
        }
    finally:
        session.close()


def run() -> dict:
    try:
        import websockets  # noqa: F401
    except ImportError:
        return {"skipped": True,
                "reason": "websockets extra not installed in this venv"}
    api_key = _env_key("GEMINI_API_KEY")
    if api_key is None:
        return {"skipped": True, "reason": "GEMINI_API_KEY not available"}
    try:
        models = _live_models(api_key)
    except Exception as exc:
        return {"skipped": True, "reason": f"model discovery failed: {exc!r}"}
    if not models:
        return {"skipped": True,
                "reason": "no models support bidiGenerateContent for this key"}
    reasons = []
    for model in models[:3]:
        try:
            return _one_session(api_key, model)
        except Exception as exc:
            reasons.append(f"{model}: {exc!r}")
    return {"skipped": True, "reason": "; ".join(reasons)}
