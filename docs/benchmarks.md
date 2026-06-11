# Benchmarks

All numbers are produced by one canonical suite —
[`benchmarks/suite/run.py`](https://github.com/lm15-dev/lm15-python/blob/main/benchmarks/suite/run.py)
— which builds the wheel, installs every competitor into fresh scratch
venvs, measures, and **regenerates the results files itself**. Nothing on
this page or in the README footprint table is hand-edited; the full
machine-generated report with methodology is
[benchmarks/BENCHMARKS.md](https://github.com/lm15-dev/lm15-python/blob/main/benchmarks/BENCHMARKS.md).

Latest run: Python 3.13.3, Linux, Intel i7-1065G7 — 2026-06-11.

## Footprint

| package | install size | transitive deps | cold import | import RSS |
|---|---:|---:|---:|---:|
| **lm15** | **0.5 MiB** | **0** | **152 ms** | **16.6 MiB** |
| openai | 18.0 MiB | 15 | 468 ms | 35.3 MiB |
| anthropic | 17.1 MiB | 15 | 589 ms | 41.2 MiB |
| google-genai | 37.2 MiB | 24 | 934 ms | 60.8 MiB |
| litellm | 133.0 MiB | 54 | 2298 ms | 161.0 MiB |
| langchain-openai | 63.3 MiB | 35 | 930 ms | 61.0 MiB |

## The abstraction costs nothing on the wire

- **Time-to-first-byte tax vs raw `urllib`: ≈ 0 ms** (measured −2.8 ms,
  i.e. indistinguishable) against a local server.
- **Steady-state, pooled connections**: 99.4 ms/call through lm15 vs
  177.6 ms/call for fresh-connection raw `urllib` against a real hosted
  endpoint — connection pooling you don't have to write.
- **Hot path**: build a request in ~12 µs, parse a response in ~37 µs,
  push ~110,000 stream events/second through the pipeline.
- **Live sessions** (Gemini Live, WebSocket): ~273 ms connect+setup,
  ~452 ms to first event, ~1.2 s full audio turn.

## Reproduce it

```bash
git clone https://github.com/lm15-dev/lm15-python && cd lm15-python
python3 benchmarks/suite/run.py          # full run
python3 benchmarks/suite/run.py --quick  # faster iteration
```

The suite writes `benchmarks/RESULTS.json` (the single authoritative
blob), regenerates `benchmarks/BENCHMARKS.md`, and re-injects the
footprint table into the README between generated-content markers.
