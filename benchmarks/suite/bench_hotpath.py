"""lm15-only hot-path microbenchmarks (run in the repo venv).

- build_request: canonical Request -> provider wire bytes, per provider.
- parse_response: real pinned response body (lm15-contract/bodies/) ->
  canonical Response, per provider.
- serde round-trip: request_to_dict -> json -> request_from_dict of a
  mid-size Request (multi-turn + tools).
- stream mapping throughput: recorded SSE transcript through
  parse_sse -> parse_stream_events -> coalesce_stream, events/sec.

Methodology: WARMUP untimed iterations, then N timed iterations; report
the median ns/op (median over per-iteration wall times).
"""

from __future__ import annotations

import glob
import json
import statistics
import time
from pathlib import Path

from lm15 import (
    AnthropicLM, GeminiLM, HttpResponse, Message, OpenAIChatLM, OpenAILM,
    Request, TextPart, ToolCallPart, ToolResultPart, FunctionTool,
)
from lm15.serde import request_from_dict, request_to_dict
from lm15.sse import parse_sse
from lm15.stream import coalesce_stream

BODIES = Path(__file__).resolve().parents[3] / "lm15-contract" / "bodies"

PROVIDERS = {
    "anthropic": AnthropicLM,
    "openai": OpenAILM,
    "gemini": GeminiLM,
    "openai_chat": OpenAIChatLM,
}


def _latest_body(case: str) -> bytes:
    files = sorted(glob.glob(str(BODIES / case / "*.txt")))
    return Path(files[-1]).read_bytes()


def _small_request() -> Request:
    return Request(
        model="bench-model",
        messages=(Message(role="user", parts=(TextPart(text="Say hi."),)),),
    )


def _midsize_request() -> Request:
    tools = (
        FunctionTool(
            name="get_weather",
            description="Get the weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"},
                               "unit": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        ),
    )
    msgs = [Message(role="user", parts=(TextPart(text="What is the weather in Gatineau and Ottawa?"),))]
    for i in range(4):
        msgs.append(Message(role="assistant", parts=(
            ToolCallPart(id=f"call_{i}", name="get_weather",
                         input={"city": f"city-{i}", "unit": "c"}),
        )))
        msgs.append(Message(role="tool", parts=(
            ToolResultPart(id=f"call_{i}", name="get_weather",
                           content=(TextPart(text=f"{{\"temp_c\": {10 + i}, \"sky\": \"overcast\"}}"),)),
        )))
    msgs.append(Message(role="assistant", parts=(TextPart(text="It is mild and overcast in both cities today."),)))
    return Request(model="bench-model", messages=tuple(msgs),
                   system="You are a meticulous weather assistant.", tools=tools)


def _time_op(fn, n: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    med = statistics.median(samples)
    return {"n": n, "warmup": warmup, "median_ns": med,
            "ops_per_sec": round(1e9 / med, 1) if med else None}


def run(n: int = 2000, warmup: int = 200) -> dict:
    out: dict = {"build_request": {}, "parse_response": {}}
    req = _small_request()

    for name, cls in PROVIDERS.items():
        lm = cls(api_key="bench-key")
        out["build_request"][name] = _time_op(lambda lm=lm: lm.build_request(req, False), n, warmup)
        body = _latest_body(f"{name}.basic_text")
        resp = HttpResponse(status=200, reason="OK", headers=[], body=body)
        out["parse_response"][name] = _time_op(lambda lm=lm, r=resp: lm.parse_response(req, r), n, warmup)

    mid = _midsize_request()
    def roundtrip() -> None:
        d = request_to_dict(mid)
        s = json.dumps(d)
        request_from_dict(json.loads(s))
    rt = _time_op(roundtrip, n, warmup)
    rt["wire_bytes"] = len(json.dumps(request_to_dict(mid)))
    out["serde_roundtrip_midsize_request"] = rt

    # stream mapping throughput over a recorded anthropic SSE transcript
    raw = _latest_body("anthropic.streaming")
    lines = raw.splitlines(keepends=True)
    lm = AnthropicLM(api_key="bench-key")
    n_events = sum(1 for e in parse_sse(iter(lines))
                   for _ in lm.parse_stream_events(req, e))

    def pipeline() -> int:
        c = 0
        for _ in coalesce_stream(
            ev for sse in parse_sse(iter(lines))
            for ev in lm.parse_stream_events(req, sse)
        ):
            c += 1
        return c

    stream_n, stream_warm = max(200, n // 10), max(20, warmup // 10)
    t = _time_op(pipeline, stream_n, stream_warm)
    out["stream_pipeline"] = {
        "transcript": "anthropic.streaming",
        "sse_bytes": len(raw),
        "stream_events_per_pass": n_events,
        "n": stream_n,
        "warmup": stream_warm,
        "median_ns_per_pass": t["median_ns"],
        "events_per_sec": round(n_events * 1e9 / t["median_ns"], 1),
        "mib_per_sec": round(len(raw) * 1e9 / t["median_ns"] / (1024 * 1024), 2),
    }
    return out
