# Cookbook 02 — Streaming

## Text only (common case)

```python
import lm15

for text in lm15.stream("gpt-4.1-mini", "Write a haiku.").text:
    print(text, end="")
print()
```

## Full events

```python
stream = lm15.stream("gpt-4.1-mini", "Write a haiku.")

for event in stream:
    match event.type:
        case "text":      print(event.text, end="")
        case "thinking":  print(f"💭 {event.text}", end="")
        case "finished":  print(f"\n📊 {event.response.usage}")
```

## Materialized response after streaming

```python
stream = lm15.stream("gpt-4.1-mini", "Explain TCP.")

for text in stream.text:
    print(text, end="")

# Full response available after stream is consumed
resp = stream.response
print(f"\nTokens: {resp.usage.total_tokens}")
print(f"Finish: {resp.finish_reason}")
```

## Streaming on a model object (records to history)

```python
gpt = lm15.model("gpt-4.1-mini")

for text in gpt.stream("Write a haiku.").text:
    print(text, end="")

# Stream is now in history — next call has context
resp = gpt("Now write another one about the sea.")
print(resp.text)
```
