# Cookbook 14 — Cost Tracking

Track per-call and cumulative costs across providers with automatic pricing from [models.dev](https://models.dev).

## Enable cost tracking

```python
import lm15

lm15.configure(track_costs=True)  # fetches pricing once (~300ms)
```

This fetches the models.dev pricing catalog and enables `.cost` on every result.

## Per-call cost

```python
import lm15

lm15.configure(track_costs=True)

result = lm15.call("gpt-4o", "Explain TCP in one paragraph")
print(result.cost)
# $0.000255 (input=$0.000045, output=$0.000210)

print(result.cost.total)        # 0.000255
print(result.cost.input)        # 0.000045
print(result.cost.output)       # 0.000210
print(result.cost.cache_read)   # 0.0
```

`.cost` returns a `CostBreakdown` with itemized fields, or `None` if tracking is not enabled.

## Cumulative cost on a model

```python
import lm15

lm15.configure(track_costs=True)

m = lm15.model("claude-sonnet-4")
m("What is TCP?")
m("What about UDP?")
m("Compare them")

print(m.total_cost)
# $0.001830 (input=$0.000330, output=$0.001500)
```

`model.total_cost` sums all calls in `model.history`. It resets when you call `model.history.clear()`.

## Cost with prompt caching

```python
import lm15

lm15.configure(track_costs=True)

m = lm15.model("claude-sonnet-4", system="<long system prompt>", prompt_caching=True)
r1 = m("First question")    # cache write
r2 = m("Second question")   # cache hit

print(r1.cost)  # higher — includes cache_write cost
print(r2.cost)  # lower — cache_read is cheaper than input
print(m.total_cost)
```

## Cost with reasoning models

```python
import lm15

lm15.configure(track_costs=True)

result = lm15.call("o4-mini", "Solve this step by step: ...", reasoning=True)
print(result.cost)
# reasoning tokens are priced separately from output tokens
print(result.cost.reasoning)  # cost of thinking tokens
print(result.cost.output)     # cost of visible output only
```

## Cost with audio

```python
import lm15

lm15.configure(track_costs=True)

result = lm15.call("gpt-4o-audio-preview", [
    "Transcribe this:",
    lm15.Part.audio(data=open("meeting.wav", "rb").read(), media_type="audio/wav"),
])
print(result.cost)
print(result.cost.input_audio)   # audio input tokens priced separately
print(result.cost.output_audio)  # audio output tokens priced separately
```

## Agent loop with cost budget

```python
import lm15

lm15.configure(track_costs=True)

def search(query: str) -> str:
    return f"Results for: {query}"

agent = lm15.model("gpt-4o", tools=[search])
MAX_BUDGET = 0.05  # $0.05

resp = agent("Research quantum computing advances in 2025")
while resp.finish_reason == "tool_call":
    if agent.total_cost and agent.total_cost.total > MAX_BUDGET:
        print(f"Budget exceeded: {agent.total_cost}")
        break
    resp = agent("continue")

print(f"Final cost: {agent.total_cost}")
```

## Manual cost estimation

For advanced use without `configure(track_costs=True)`:

```python
from lm15 import estimate_cost, fetch_models_dev

# Build index once
specs = fetch_models_dev()
cost_index = {s.id: s for s in specs}

# Estimate from any Usage + ModelSpec
result = lm15.call("gpt-4o", "hello")
cost = estimate_cost(result.usage, cost_index[result.model])
print(cost.total)
```

You can also pass a raw cost dict:

```python
from lm15 import estimate_cost

cost = estimate_cost(
    result.usage,
    {"input": 2.5, "output": 10.0, "cache_read": 1.25},
    provider="openai",
)
```

## CostBreakdown fields

| Field | Description |
|---|---|
| `total` | Sum of all cost components |
| `input` | Non-cached, non-audio input tokens |
| `output` | Non-reasoning, non-audio output tokens |
| `cache_read` | Cached input tokens (cheaper rate) |
| `cache_write` | Cache creation tokens (Anthropic) |
| `reasoning` | Thinking/reasoning tokens |
| `input_audio` | Audio input tokens (OpenAI) |
| `output_audio` | Audio output tokens (OpenAI) |

All values are in US dollars.

## Provider token semantics

The cost calculation handles provider differences automatically:

| | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| `input_tokens` | total (includes cached) | non-cached only | total (includes cached) |
| `output_tokens` | total (includes reasoning) | total | total |
| `cache_read` | subset of input | additive | subset of input |
| `reasoning` | subset of output | n/a | separate count |

You don't need to worry about these — `estimate_cost` and `.cost` handle them correctly for each provider.
