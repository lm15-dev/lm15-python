# Cookbook 06 — Reasoning / Extended Thinking

## Enable reasoning

```python
import lm15

resp = lm15.complete("claude-sonnet-4-5", "Prove that √2 is irrational.", reasoning=True)

# Chain of thought
print("Thinking:")
print(resp.thinking)

# Final answer
print("\nAnswer:")
print(resp.text)
```

## Fine-grained control

```python
# Anthropic: budget in tokens
resp = lm15.complete("claude-sonnet-4-5", "Hard math problem.",
    reasoning={"effort": "high", "budget": 10000})

# OpenAI: effort level
resp = lm15.complete("gpt-4.1-mini", "Hard math problem.",
    reasoning={"effort": "high"})
```

## Streaming with reasoning

```python
import lm15

stream = lm15.stream("claude-sonnet-4-5", "Explain P vs NP.", reasoning=True)

for event in stream:
    match event.type:
        case "thinking":  print(f"💭 {event.text}", end="")
        case "text":      print(event.text, end="")
        case "finished":
            print(f"\n\nReasoning tokens: {event.response.usage.reasoning_tokens}")
```

## Reasoning on a model object

```python
import lm15

claude = lm15.model("claude-sonnet-4-5")

resp = claude("Solve this step by step: what is 17! / 15! ?", reasoning=True)
print(resp.thinking)
print(resp.text)
```
