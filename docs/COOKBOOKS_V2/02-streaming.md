# Cookbook 02 — Streaming

In v3, there is one high-level entry point: `call()`.

- **Blocking:** access `.text`, `.usage`, `.response`, etc.
- **Streaming text:** iterate the `Result`
- **Streaming events:** call `.events()`

---

## Text only

```python
import lm15

for text in lm15.call("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")
print()
```

## Full events

```python
import lm15

for event in lm15.call("gpt-4.1-mini", "Write a haiku.").events():
    match event.type:
        case "text":      print(event.text, end="")
        case "thinking":  print(f"💭 {event.text}", end="")
        case "finished":  print(f"\n📊 {event.response.usage}")
```

## Get the final response after streaming

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "Explain TCP.")

for text in resp:
    print(text, end="")

print(f"\nTokens: {resp.usage.total_tokens}")
print(f"Finish: {resp.finish_reason}")
```

The full response is cached as the stream is consumed, so properties still work after iteration.

## Streaming on a model object

```python
import lm15

gpt = lm15.model("gpt-4.1-mini")

for text in gpt.call("Write a haiku."):
    print(text, end="")

resp = gpt.call("Now write another one about the sea.")
print(resp.text)
```

## Tool + streaming visibility

```python
import lm15

def lookup(topic: str) -> str:
    """Look up a topic."""
    return "42 is the answer."

for event in lm15.call(
    "claude-sonnet-4-5",
    "Research quantum computing.",
    tools=[lookup],
    reasoning=True,
).events():
    match event.type:
        case "thinking":    print(f"💭 {event.text}", end="")
        case "tool_call":   print(f"\n🔧 {event.name}({event.input})")
        case "tool_result": print(f"📎 {event.text}")
        case "text":        print(event.text, end="")
        case "finished":    print(f"\n📊 {event.response.usage}")
```

## Async streaming

```python
import lm15

async for text in lm15.acall("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")
```

`lm15.stream()` still exists as a compatibility alias, but `lm15.call()` is the preferred API.
