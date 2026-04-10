# Cookbook 11 — `call()`, `acall()`, and `Result` Reference

<details>
<summary><strong>API keys</strong> — you need at least one to run these examples.</summary>

Three ways to provide credentials (see [Cookbook 00](00-api-key.md) for full details):

1. **`.env` file (recommended)** — put your keys in a `.env` file and pass `env=".env"`.
2. **Environment variables** — `export OPENAI_API_KEY="sk-..."`.
3. **Inline** — pass `api_key="sk-..."` directly.

All examples below use `env=".env"`.
</details>

---

## High-level entry points

```python
lm15.call(...)   -> Result
lm15.acall(...)  -> AsyncResult
lm15.model(...)  -> Model
```

Delivery mode is a **consumption choice**, not a different request function:

- **Blocking:** `resp.text`, `resp.usage`, `resp.response`
- **Streaming text:** `for text in resp:`
- **Streaming events:** `for event in resp.events():`

`lm15.stream()` still exists as a compatibility alias, but `lm15.call()` is the preferred entry point.

---

## `lm15.call()` signature

```python
lm15.call(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list[Callable | Tool | str] | None = None,
    on_tool_call: Callable[[ToolCallInfo], str | None] | None = None,
    reasoning: bool | dict | None = None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    max_tool_rounds: int = 8,
    retries: int = 0,
    provider: str | None = None,
    api_key: str | dict | None = None,
    env: str | None = None,
) -> Result
```

---

## `prompt` and `messages`

### Plain string

```python
resp = lm15.call("gpt-4.1-mini", "What is DNA?", env=".env")
```

### Mixed list of text and parts

```python
from lm15 import Part

resp = lm15.call(
    "gemini-2.5-flash",
    ["What's in this image?", Part.image(url="https://example.com/photo.jpg")],
    env=".env",
)
```

### Full `messages=` control

```python
from lm15 import Message

resp = lm15.call(
    "gpt-4.1-mini",
    messages=[
        Message.user("My name is Max."),
        Message.assistant("Nice to meet you, Max!"),
        Message.user("What's my name?"),
    ],
    env=".env",
)
print(resp.text)
```

`prompt` and `messages` are mutually exclusive.

---

## `system`

```python
resp = lm15.call(
    "gpt-4.1-mini",
    "Explain TCP.",
    system="You are a networking expert. Be concise.",
    env=".env",
)
```

---

## `tools`

### Auto-execute with Python functions

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"22°C in {city}"

resp = lm15.call("gpt-4.1-mini", "Weather in Paris?", tools=[get_weather], env=".env")
print(resp.text)
```

### Manual tools with `Tool`

```python
from lm15 import Tool

weather = Tool(
    name="get_weather",
    description="Get weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)

agent = lm15.model("gpt-4.1-mini", env=".env")
resp = agent.call("Weather in Paris?", tools=[weather])
print(resp.tool_calls)
```

### Intercept tool calls with `on_tool_call`

```python
def approve(call):
    print(call.name, call.input)
    return None  # continue with normal execution

resp = lm15.call(
    "gpt-4.1-mini",
    "Weather in Paris?",
    tools=[get_weather],
    on_tool_call=approve,
    env=".env",
)
```

If `on_tool_call()` returns a string, lm15 uses that as the tool result instead of calling the Python function.

---

## `reasoning`

```python
resp = lm15.call("claude-sonnet-4-5", "Prove √2 is irrational.", reasoning=True, env=".env")
print(resp.thinking)
print(resp.text)
```

Fine-grained:

```python
resp = lm15.call(
    "claude-sonnet-4-5",
    "Prove √2 is irrational.",
    reasoning={"budget": 10000},
    env=".env",
)
```

---

## `prefill`

```python
resp = lm15.call(
    "claude-sonnet-4-5",
    "Output JSON for a person.",
    prefill="{",
    env=".env",
)
print(resp.text)
```

---

## `output`

```python
resp = lm15.call("gpt-4.1-mini", "Draw a sunset.", output="image", env=".env")
print(resp.image)

resp = lm15.call("gpt-4o-mini-tts", "Say hello in French.", output="audio", env=".env")
print(resp.audio)
```

---

## Sampling and limits

```python
resp = lm15.call(
    "gpt-4.1-mini",
    "Write a haiku.",
    temperature=0.9,
    max_tokens=50,
    top_p=0.95,
    stop=["\n\n"],
    env=".env",
)
```

---

## `provider`

lm15 usually infers the provider from the model name. Override it when using custom or fine-tuned names:

```python
resp = lm15.call("ft:my-model", "Hello.", provider="openai", env=".env")
```

---

## `prompt_caching`

```python
long_doc = open("contract.txt").read()

resp1 = lm15.call(
    "claude-sonnet-4-5",
    f"Summarize section 1:\n\n{long_doc}",
    prompt_caching=True,
    env=".env",
)
resp2 = lm15.call(
    "claude-sonnet-4-5",
    f"Summarize section 2:\n\n{long_doc}",
    prompt_caching=True,
    env=".env",
)
print(resp2.usage.cache_read_tokens)
```

---

## Consuming a `Result`

### Blocking

```python
resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
print(resp.text)
print(resp.model)
print(resp.finish_reason)
print(resp.usage.total_tokens)
```

### Streaming text

```python
for text in lm15.call("gpt-4.1-mini", "Write a haiku.", env=".env"):
    print(text, end="")
```

### Streaming events

```python
for event in lm15.call("gpt-4.1-mini", "Explain TCP.", env=".env").events():
    match event.type:
        case "text":
            print(event.text, end="")
        case "thinking":
            print(f"💭 {event.text}", end="")
        case "tool_call":
            print(event.name, event.input)
        case "tool_result":
            print(event.text)
        case "finished":
            print(event.response.usage)
```

### Access the underlying `LMResponse`

```python
resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
full = resp.response
print(full.provider)
```

---

## `Result` convenience properties

| Property | Type |
|---|---|
| `resp.text` | `str | None` |
| `resp.thinking` | `str | None` |
| `resp.tool_calls` | `list[Part]` |
| `resp.image` | `Part | None` |
| `resp.images` | `list[Part]` |
| `resp.audio` | `Part | None` |
| `resp.citations` | `list[Part]` |
| `resp.usage` | `Usage` |
| `resp.finish_reason` | `str` |
| `resp.model` | `str` |
| `resp.json` | parsed JSON |
| `resp.image_bytes` | `bytes` |
| `resp.audio_bytes` | `bytes` |
| `resp.response` | `LMResponse` |

---

## `lm15.acall()`

### Await the whole result

```python
resp = await lm15.acall("gpt-4.1-mini", "Hello.", env=".env")
print(resp.text)
```

### Async streaming

```python
async for text in lm15.acall("gpt-4.1-mini", "Write a haiku.", env=".env"):
    print(text, end="")
```

### Async events

```python
async for event in lm15.acall("claude-sonnet-4-5", "Explain TCP.", reasoning=True, env=".env").events():
    if event.type == "text":
        print(event.text, end="")
```

---

## `lm15.model()`

```python
agent = lm15.model(
    "gpt-4.1-mini",
    system="You are terse.",
    tools=[get_weather],
    retries=2,
    prompt_caching=True,
    env=".env",
)

resp = agent.call("Weather in Montreal?")
print(resp.text)
```

Useful methods:

```python
agent.call(...)
agent.acall(...)
agent.submit_tools(...)
agent.prepare(...)
agent.upload(...)
agent.copy(...)
```

---

## `send()` and `prepare()`

```python
req = lm15.prepare("gpt-4.1-mini", "Hello.", env=".env")
resp = lm15.send(req, env=".env")
print(resp.text)
```

`send()` now returns a `Result`, just like `call()`.
