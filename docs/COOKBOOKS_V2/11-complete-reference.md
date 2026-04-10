# Cookbook 11 — `call()` and `stream()` Reference

<details>
<summary><strong>API keys</strong> — you need at least one to run these examples.</summary>

Three ways to provide credentials (see [Cookbook 00](00-api-key.md) for full details):

1. **`.env` file (recommended)** — put your keys in a `.env` file and pass `env=".env"` to the function.
2. **Environment variables** — `export OPENAI_API_KEY="sk-..."` in your terminal. lm15 picks them up automatically.
3. **Inline** — pass `api_key="sk-..."` directly. Fine for quick experiments, not for code you share.

All examples below use `env=".env"`.
</details>

`lm15.call()` and `lm15.stream()` are the stateless entry points. They share the same parameters — the only difference is how the response arrives.

---

## Full signature

```python
lm15.call(
    model: str,                              # required — e.g. "gpt-4.1-mini", "claude-sonnet-4-5"
    prompt: str | list[str | Part] = None,   # a string or mixed list of text + parts
    *,
    messages: list[Message] = None,          # raw message list (mutually exclusive with prompt)
    system: str = None,                      # system prompt
    tools: list[Callable | Tool | str] = None,  # tools the model can call
    reasoning: bool | dict = None,           # enable chain-of-thought
    prefill: str = None,                     # seed the assistant response
    output: str = None,                      # "image" or "audio"
    prompt_caching: bool = False,            # provider-side prefix caching
    temperature: float = None,               # 0.0 = deterministic, 1.0+ = creative
    max_tokens: int = None,                  # max output tokens
    top_p: float = None,                     # nucleus sampling
    stop: list[str] = None,                  # stop sequences
    provider: str = None,                    # force a specific provider
    api_key: str | dict = None,              # pass key(s) directly
    env: str = None,                         # path to .env / .bashrc / etc.
) -> LMResponse
```

`lm15.stream()` has the identical signature but returns a `Stream` instead of `LMResponse`.

---

## `prompt` — what you send

### Plain string

```python
resp = lm15.call("gpt-4.1-mini", "What is DNA?", env=".env")
```

### Mixed list of text and parts

```python
from lm15 import Part

resp = lm15.call("gemini-2.5-flash", [
    "What's in this image?",
    Part.image(url="https://example.com/photo.jpg"),
], env=".env")
```

Strings in the list become text parts. Everything is wrapped into one user message.

### `messages` — full control

For multi-turn or when you need explicit role assignment. **Mutually exclusive with `prompt`.**

```python
from lm15 import Message

resp = lm15.call("gpt-4.1-mini", messages=[
    Message.user("My name is Max."),
    Message.assistant("Nice to meet you, Max!"),
    Message.user("What's my name?"),
], env=".env")
print(resp.text)  # Knows it's Max — the history is in the messages
```

---

## `system` — instructions for the model

```python
resp = lm15.call("gpt-4.1-mini", "Explain TCP.",
    system="You are a networking expert. Be concise.",
    env=".env",
)
```

---

## `tools` — let the model call functions

Tools work at every level. `call()` auto-executes callable tools and loops until the model stops calling them.

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"22°C in {city}"

resp = lm15.call("gpt-4.1-mini", "Weather in Paris?",
    tools=[get_weather], env=".env",
)
print(resp.text)  # "It's 22°C in Paris."
```

See [Cookbook 03 (Tools Auto)](03-tools-auto.md) and [Cookbook 04 (Tools Manual)](04-tools-manual.md).

---

## `reasoning` — chain-of-thought

```python
resp = lm15.call("claude-sonnet-4-5", "Prove √2 is irrational.",
    reasoning=True, env=".env",
)
print(resp.thinking)  # The model's reasoning steps
print(resp.text)       # The final answer
```

Fine-grained:

```python
resp = lm15.call("claude-sonnet-4-5", "Prove √2 is irrational.",
    reasoning={"budget": 10000}, env=".env",
)
```

See [Cookbook 06 (Reasoning)](06-reasoning.md).

---

## `prefill` — seed the assistant response

Force the model to start its response with a specific string. Useful for steering output format:

```python
resp = lm15.call("claude-sonnet-4-5", "List 3 colors.",
    prefill="1.", env=".env",
)
print(resp.text)  # Starts with "1. ..."
```

---

## `output` — request non-text output

```python
# Image generation
resp = lm15.call("gpt-4.1-mini", "Draw a sunset.", output="image", env=".env")
resp.image  # Part with the generated image

# Audio generation
resp = lm15.call("gpt-4.1-mini", "Say hello in French.", output="audio", env=".env")
resp.audio  # Part with the generated audio
```

See [Cookbook 05 (Multimodal)](05-multimodal.md).

---

## `temperature`, `max_tokens`, `top_p`, `stop`

```python
resp = lm15.call("gpt-4.1-mini", "Write a haiku.",
    temperature=0.9,     # more creative
    max_tokens=50,       # short response
    top_p=0.95,          # nucleus sampling
    stop=["\n\n"],       # stop at double newline
    env=".env",
)
```

---

## `provider` — force a specific provider

lm15 infers the provider from the model name (`gpt-*` → openai, `claude-*` → anthropic, `gemini-*` → gemini). Override when using custom or fine-tuned models:

```python
resp = lm15.call("ft:gpt-4.1-mini:my-org:my-finetune",
    "Hello.", provider="openai", env=".env",
)
```

---

## `prompt_caching` — reduce cost on repeated prefixes

```python
long_doc = open("contract.txt").read()

resp1 = lm15.call("claude-sonnet-4-5",
    f"Summarize section 1:\n\n{long_doc}",
    prompt_caching=True, env=".env",
)
resp2 = lm15.call("claude-sonnet-4-5",
    f"Summarize section 2:\n\n{long_doc}",
    prompt_caching=True, env=".env",
)
print(resp2.usage.cache_read_tokens)  # Prefix was cached
```

See [Cookbook 08 (Prompt Caching)](08-prompt-caching.md).

---

## Streaming

`lm15.stream()` takes the same parameters. The difference is how you consume the result.

### Text only (most common)

```python
for text in lm15.stream("gpt-4.1-mini", "Write a haiku.", env=".env").text:
    print(text, end="")
```

### Full event access

```python
stream = lm15.stream("gpt-4.1-mini", "Write a haiku.", env=".env")

for event in stream:
    match event.type:
        case "text":      print(event.text, end="")
        case "thinking":  print(f"💭 {event.text}", end="")
        case "finished":  print(f"\n📊 {event.response.usage}")
```

### Get the full response after streaming

```python
stream = lm15.stream("gpt-4.1-mini", "Explain TCP.", env=".env")

for text in stream.text:
    print(text, end="")

resp = stream.response  # Full LMResponse available after stream is consumed
print(f"\nTokens: {resp.usage.total_tokens}")
```

See [Cookbook 02 (Streaming)](02-streaming.md).

---

## The response object

Both `call()` and `stream()` (after consumption) give you an `LMResponse`:

| Property | Type | What it is |
|----------|------|-----------|
| `resp.text` | `str \| None` | All text parts joined |
| `resp.thinking` | `str \| None` | Chain-of-thought text |
| `resp.image` | `Part \| None` | First image part |
| `resp.images` | `list[Part]` | All image parts |
| `resp.audio` | `Part \| None` | First audio part |
| `resp.tool_calls` | `list[Part]` | All tool call parts |
| `resp.citations` | `list[Part]` | All citation parts |
| `resp.model` | `str` | Model that responded |
| `resp.finish_reason` | `str` | `"stop"`, `"tool_call"`, `"length"`, etc. |
| `resp.usage` | `Usage` | Token counts |
| `resp.provider` | `dict` | Raw provider response (escape hatch) |

### Usage breakdown

```python
resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
u = resp.usage

print(u.input_tokens)        # Tokens sent
print(u.output_tokens)       # Tokens received
print(u.total_tokens)        # Sum
print(u.reasoning_tokens)    # Tokens spent on chain-of-thought (if reasoning=True)
print(u.cache_read_tokens)   # Tokens served from cache
print(u.cache_write_tokens)  # Tokens written to cache
```
