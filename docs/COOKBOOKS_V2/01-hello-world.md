# Cookbook 01 — Hello World

<details>
<summary><strong>Note on API keys</strong> — you need at least one to run these examples.</summary>

Three ways to provide credentials (see [Cookbook 00](00-api-key.md) for full details):

1. **`.env` file (recommended)** — put your keys in a `.env` file and pass `env=".env"`.
2. **Environment variables** — `export OPENAI_API_KEY="sk-..."` and friends.
3. **Inline** — pass `api_key="sk-..."` directly.

All examples below use `env=".env"`.
</details>

---

lm15 has three layers, from simplest to most manual:

1. **`lm15.call()`** — stateless high-level entry point, returns a `Result`
2. **`lm15.model()`** — reusable stateful object with history, also returns `Result`
3. **`UniversalLM` + adapters** — low-level request/adapter plumbing

Most users only need #1 or #2.

---

## `lm15.call()` — one question, one result

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "say ok", env=".env")
print(resp.text)
```

`call()` is stateless. Every call starts fresh unless you pass explicit `messages=`.

```python
resp1 = lm15.call("gpt-4.1-mini", "My name is Max.", env=".env")
resp2 = lm15.call("gpt-4.1-mini", "What's my name?", env=".env")
print(resp2.text)  # no memory across separate calls
```

A `Result` can be consumed two ways:

```python
# Blocking
resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
print(resp.text)

# Streaming
for text in lm15.call("gpt-4.1-mini", "Write a haiku.", env=".env"):
    print(text, end="")
```

Use `call()` when you have one self-contained request.

> **Deep dive:** [Cookbook 11 — `call()`, `acall()`, and `Result` reference](11-complete-reference.md)

### Adding a system prompt

```python
resp = lm15.call(
    "claude-sonnet-4-5",
    "Summarize DNA.",
    system="You are terse.",
    env=".env",
)
print(resp.text)
```

### Config from a dict

```python
config = {
    "model": "gpt-4.1-mini",
    "system": "You are terse.",
    "temperature": 0.7,
    "max_tokens": 100,
    "env": ".env",
}

resp = lm15.call(prompt="Summarize RNA.", **config)
print(resp.text)
```

---

## `lm15.model()` — reusable object with memory

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", env=".env")

resp = gpt.call("Hello!")
print(resp.text)
```

A model object holds config once and remembers prior turns.

```python
gpt = lm15.model("gpt-4.1-mini", env=".env", system="You remember everything.")

gpt.call("My name is Max.")
gpt.call("I like chess.")

resp = gpt.call("What do you know about me?")
print(resp.text)
```

Use `model()` when you need:
- **Conversation** — multi-turn back-and-forth
- **Reuse** — same config across many calls
- **Bound tools** — attach tools once and reuse them

> **Tools** and **streaming** work here too. `agent.call()` returns the same `Result` type as `lm15.call()`, so you can either read `.text` or iterate it.

### Config lives on the object

```python
claude = lm15.model(
    "claude-sonnet-4-5",
    env=".env",
    system="You are a helpful coding assistant.",
    temperature=0.7,
    max_tokens=500,
)

resp = claude.call("Explain Python generators.")
print(resp.text)
```

### Deriving new models with `.copy()`

```python
gpt = lm15.model("gpt-4.1-mini", env=".env", system="You are helpful.")

terse_gpt = gpt.copy(system="You are extremely terse.")
claude = gpt.copy(model="claude-sonnet-4-5")

print(gpt.system)        # original unchanged
print(terse_gpt.system)  # new model
```

Per-call overrides still work without copying:

```python
resp = claude.call("Write a poem.")
resp = claude.call("Write a poem.", temperature=0.2)
```

### Clearing history

```python
gpt = lm15.model("gpt-4.1-mini", env=".env")

gpt.call("My name is Max.")
gpt.call("I like chess.")
print(len(gpt.history))

gpt.history.clear()
print(len(gpt.history))
```

> **Deep dive:** [Cookbook 09 — Model Objects and Configuration](09-model-config.md) and [Cookbook 07 — Multi-Turn Conversation](07-conversation.md)

---

## `UniversalLM` + adapters — full manual control

```python
from lm15 import UniversalLM, LMRequest, Message, Config
from lm15.providers.openai import OpenAIAdapter
from lm15.transports.urllib_transport import UrlLibTransport

client = UniversalLM()
client.register(OpenAIAdapter(api_key="sk-proj-...", transport=UrlLibTransport()))

request = LMRequest(
    model="gpt-4.1-mini",
    messages=(Message.user("Hello."),),
    config=Config(temperature=0.5, max_tokens=100),
)

resp = client.complete(request)
print(resp.text)
```

Use this layer when you need:
- custom transports
- manual adapter registration
- plugin development
- tests and mocks

---

## When to use which

| | `lm15.call()` | `lm15.model()` | `UniversalLM` |
|---|---|---|---|
| **State** | Stateless | Stateful | You manage it |
| **Return type** | `Result` | `Result` | `LMResponse` / stream events |
| **Best for** | Scripts, one-offs | Conversations, agents | Plugins, tests, low-level control |
| **Streaming** | `for text in lm15.call(...)` | `for text in agent.call(...)` | `client.stream(request)` |

**Rule of thumb:** start with `call()`. Move to `model()` when you need memory or reusable config.

---

## Inspecting the result

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")

print(resp.text)
print(resp.model)
print(resp.finish_reason)
print(resp.usage.input_tokens)
print(resp.usage.output_tokens)
print(resp.usage.total_tokens)

# Full underlying LMResponse if you want it
print(resp.response.provider)
```

---

## Switching models

Just change the string:

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
resp = lm15.call("claude-sonnet-4-5", "Hello.", env=".env")
resp = lm15.call("gemini-2.5-flash", "Hello.", env=".env")
```

If lm15 can't infer the provider from the name:

```python
resp = lm15.call("my-fine-tune", "Hello.", provider="openai", env=".env")
agent = lm15.model("my-fine-tune", provider="openai", env=".env")
```
