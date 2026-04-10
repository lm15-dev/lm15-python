# Cookbook 06 — Reasoning / Extended Thinking

## Enable reasoning

```python
import lm15

resp = lm15.call("claude-sonnet-4-5", "Prove that √2 is irrational.", reasoning=True)

print("Thinking:")
print(resp.thinking)

print("\nAnswer:")
print(resp.text)
```

## Fine-grained control

```python
# Anthropic: budget in tokens
resp = lm15.call(
    "claude-sonnet-4-5",
    "Hard math problem.",
    reasoning={"effort": "high", "budget": 10000},
)

# OpenAI: effort level
resp = lm15.call(
    "gpt-4.1-mini",
    "Hard math problem.",
    reasoning={"effort": "high"},
)
```

## Streaming with reasoning

```python
import lm15

for event in lm15.call("claude-sonnet-4-5", "Explain P vs NP.", reasoning=True).events():
    match event.type:
        case "thinking":
            print(f"💭 {event.text}", end="")
        case "text":
            print(event.text, end="")
        case "finished":
            print(f"\n\nReasoning tokens: {event.response.usage.reasoning_tokens}")
```

## Reasoning on a model object

```python
import lm15

claude = lm15.model("claude-sonnet-4-5")

resp = claude.call("Solve this step by step: what is 17! / 15! ?", reasoning=True)
print(resp.thinking)
print(resp.text)
```

## Async reasoning

```python
import lm15

resp = await lm15.acall("claude-sonnet-4-5", "Explain the halting problem.", reasoning=True)
print(resp.text)
```
