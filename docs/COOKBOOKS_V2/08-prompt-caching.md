# Cookbook 08 — Prompt Caching

Provider-side prompt caching reduces cost and latency by caching repeated prefixes.

## Automatic caching on a model (recommended)

```python
import lm15

claude = lm15.model("claude-sonnet-4-5",
    system="<very long system prompt — thousands of tokens>",
    prompt_caching=True,
)

resp = claude("Question 1")   # cache write
print(f"Cache write: {resp.usage.cache_write_tokens} tokens")

resp = claude("Question 2")   # cache hit
print(f"Cache read: {resp.usage.cache_read_tokens} tokens")

resp = claude("Question 3")   # cache hit
print(f"Cache read: {resp.usage.cache_read_tokens} tokens")
```

## Agent loop with caching

Each turn caches all prior context. Only the new message is uncached.

```python
import lm15

def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()

def write_file(path: str, content: str) -> str:
    """Write a file."""
    open(path, "w").write(content)
    return f"Wrote {len(content)} bytes"

agent = lm15.model("claude-sonnet-4-5",
    system="You are a coding assistant.",
    tools=[read_file, write_file],
    prompt_caching=True,
)

resp = agent("Add input validation to models.py")
while resp.finish_reason == "tool_call":
    results = execute_tools(resp.tool_calls)
    resp = agent.submit_tools(results)
    print(f"Cache hit: {resp.usage.cache_read_tokens} tokens")

print(resp.text)
```

How the cache advances per turn:

```
Turn 1: [system ✎ | user₁]
Turn 2: [system ✓ | user₁ ✎ | asst₁ | tool₁ | user₂]
Turn 3: [system ✓ | user₁ ✓ | asst₁ ✓ | tool₁ ✓ | asst₂ ✎ | tool₂ | user₃]
```

`✓` = cache hit, `✎` = cache breakpoint

## Per-part cache hints

```python
import lm15
from lm15 import Part

contract = Part.document(
    data=open("contract.pdf", "rb").read(),
    media_type="application/pdf",
    cache=True,
)

resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 1.", contract])
resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 2.", contract])  # cache hit
resp = lm15.complete("claude-sonnet-4-5", ["Find liability clauses.", contract])  # cache hit
```

## Provider-specific cache control

```python
from lm15 import Part

# Anthropic: custom TTL
doc = Part.document(url="...", cache={"type": "ephemeral", "ttl": 300})

# Simple boolean (adapter uses sensible default)
doc = Part.document(url="...", cache=True)
```

## Provider behavior

| Provider | `prompt_caching=True` behavior | Per-part `cache=True` |
|---|---|---|
| **Anthropic** | `cache_control` on system + advancing breakpoint | `cache_control: {"type": "ephemeral"}` |
| **Gemini** | Creates/reuses `CachedContent` for prefix | Included in cached content |
| **OpenAI** | No-op (automatic prefix caching) | No-op |
