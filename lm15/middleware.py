from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .types import LMRequest, LMResponse, StreamEvent

CompleteFn = Callable[[LMRequest], LMResponse]
StreamFn = Callable[[LMRequest], Iterator[StreamEvent]]
CompleteMiddleware = Callable[[LMRequest, CompleteFn], LMResponse]
StreamMiddleware = Callable[[LMRequest, StreamFn], Iterator[StreamEvent]]


@dataclass(slots=True)
class MiddlewarePipeline:
    complete_mw: list[CompleteMiddleware] = field(default_factory=list)
    stream_mw: list[StreamMiddleware] = field(default_factory=list)

    def add(self, middleware: CompleteMiddleware) -> None:
        """Add a completion middleware to the pipeline."""
        self.complete_mw.append(middleware)

    def wrap_complete(self, fn: CompleteFn) -> CompleteFn:
        wrapped = fn
        for mw in reversed(self.complete_mw):
            prev = wrapped
            wrapped = lambda req, mw=mw, prev=prev: mw(req, prev)
        return wrapped

    def wrap_stream(self, fn: StreamFn) -> StreamFn:
        wrapped = fn
        for mw in reversed(self.stream_mw):
            prev = wrapped
            wrapped = lambda req, mw=mw, prev=prev: mw(req, prev)
        return wrapped


def with_history(history: list[dict[str, Any]]) -> CompleteMiddleware:
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        started = time.time()
        resp = nxt(req)
        history.append(
            {
                "ts": started,
                "model": req.model,
                "messages": len(req.messages),
                "finish_reason": resp.finish_reason,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            }
        )
        return resp

    return middleware


def with_retries(max_retries: int = 2, sleep_base: float = 0.2) -> CompleteMiddleware:
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        last: Exception | None = None
        for i in range(max_retries + 1):
            try:
                return nxt(req)
            except Exception as e:  # pragma: no cover - generic retry wrapper
                last = e
                if i == max_retries:
                    raise
                time.sleep(sleep_base * (2**i))
        raise RuntimeError("unreachable") from last

    return middleware


def with_cache(cache: dict[str, LMResponse]) -> CompleteMiddleware:
    def key(req: LMRequest) -> str:
        return str((req.model, req.system, req.messages, req.tools, req.config))

    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        k = key(req)
        if k in cache:
            return cache[k]
        resp = nxt(req)
        cache[k] = resp
        return resp

    return middleware
