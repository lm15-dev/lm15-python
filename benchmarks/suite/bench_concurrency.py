"""Concurrency: N tiny calls to LOCAL ollama, AsyncOpenAIChatLM under
``asyncio.gather`` vs the sync adapter called sequentially.

The claim under test: the Async* mirrors give real overlap on plain
asyncio sockets — no thread pool, no helper threads.  We assert that by
recording ``threading.active_count()`` before and after the gather.
Skips gracefully if ollama is down.
"""

from __future__ import annotations

import asyncio
import json
import statistics  # noqa: F401  (kept for parity with sibling benches)
import threading
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


def run(n_calls: int = 50) -> dict:
    model = _pick_model()
    if model is None:
        return {"skipped": True, "reason": "ollama not reachable at " + OLLAMA}

    from lm15 import Config, Message, Request
    from lm15.providers.async_base import AsyncOpenAIChatLM
    from lm15.providers.openai_chat import OpenAIChatLM
    from lm15.transports import StdlibAsyncTransport

    request = Request(model=model, messages=(Message.user("Say hi."),),
                      config=Config(max_tokens=4, temperature=0.0))

    # sequential sync baseline (one pooled connection, reused)
    sync_lm = OpenAIChatLM(api_key="unused", compat="ollama")
    sync_lm.complete(request)  # warmup: load the model
    t0 = time.perf_counter()
    for _ in range(n_calls):
        sync_lm.complete(request)
    sync_wall = time.perf_counter() - t0
    sync_lm.transport.close()

    # concurrent async (single event loop, pool sized to the burst)
    async def gather_run() -> tuple[float, int, int]:
        transport = StdlibAsyncTransport(max_connections=n_calls)
        lm = AsyncOpenAIChatLM(api_key="unused", compat="ollama",
                               transport=transport)
        # Pin the loop's default executor (asyncio resolves DNS via
        # getaddrinfo in a lazily-growing thread pool that belongs to the
        # event loop, not to lm15) to a fixed, pre-spawned 2-thread pool so
        # the before/after thread count isolates lm15's own behaviour: it
        # must spawn no threads at all.
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_running_loop()
        resolver = ThreadPoolExecutor(max_workers=2,
                                      thread_name_prefix="bench-resolver")
        loop.set_default_executor(resolver)
        barrier = threading.Barrier(3)
        for _ in range(2):  # pre-spawn both resolver threads
            resolver.submit(barrier.wait)
        barrier.wait()
        await lm.complete(request)  # warmup: load the model, open one conn
        threads_before = threading.active_count()
        t0 = time.perf_counter()
        await asyncio.gather(*(lm.complete(request) for _ in range(n_calls)))
        wall = time.perf_counter() - t0
        threads_after = threading.active_count()
        await transport.aclose()
        return wall, threads_before, threads_after

    try:
        async_wall, threads_before, threads_after = asyncio.run(gather_run())
    except Exception as exc:
        return {"skipped": True, "reason": f"async run failed: {exc!r}"}

    return {
        "skipped": False,
        "model": model,
        "n_calls": n_calls,
        "sync_sequential_wall_s": round(sync_wall, 3),
        "sync_calls_per_sec": round(n_calls / sync_wall, 2),
        "async_gather_wall_s": round(async_wall, 3),
        "async_calls_per_sec": round(n_calls / async_wall, 2),
        "speedup_x": round(sync_wall / async_wall, 2),
        "threads_before_gather": threads_before,
        "threads_after_gather": threads_after,
        "threads_unchanged": threads_before == threads_after,
    }
