# Cookbook 01 — Hello World

<details>
<summary><strong>Note on API keys</strong> — you need at least one to run these examples.</summary>

Three ways to provide credentials (see [Cookbook 00](00-api-key.md) for full details):

1. **`.env` file (recommended)** — put your keys in a `.env` file and pass `env=".env"` to the function.
2. **Environment variables** — `export OPENAI_API_KEY="sk-..."` in your terminal. lm15 picks them up automatically.
3. **Inline** — pass `api_key="sk-..."` directly. Fine for quick experiments, not for code you share.

All examples below use `env=".env"`.

---
</details>

---

lm15 has three ways to talk to a model, from simplest to most control:

1. **`lm15.call()`** — one function call, no state
2. **`lm15.model()`** — reusable object with conversation memory
3. **`UniversalLM` + adapters** — full manual wiring

Most people only need #1 or #2. This cookbook shows all three so you know which to reach for.

---

## `lm15.call()` — one question, one answer

```python
import lm15

resp = lm15.call("gpt-5.4-nano", "say ok", env=".env")
print(resp.text)
```
```output | ✓ 650ms | 5 vars
Ok.
```

`call()` is a **standalone function call**. You pass the model name and a prompt, you get a response. No setup, no state. Every call is independent.

```python
# Each call knows nothing about the previous one
resp1 = lm15.call("gpt-5.4-nano", "My name is Max.", env=".env")
resp2 = lm15.call("gpt-5.4-nano", "What's my name?", env=".env")
print(resp2.text)  # It doesn't know — each call starts fresh
```

output | ✓ 3.0s | 5 vars
I don’t know your name from this chat. If you tell me what you’d like me to call you, I’ll use it.
    

```python
resp1.usage
```
```output | ✓ 1ms | 5 vars
Usage(input_tokens=11, output_tokens=20, total_tokens=31, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0)
```

Use `call()` when you have a **single, self-contained question** and don't need conversation history.

> **Deep dive:** [Cookbook 11 — `call()` and `stream()` Reference](11-complete-reference.md) covers every parameter, multimodal prompts, tool passing, reasoning, prefill, and output modalities.

### Adding a system prompt

```python
resp = lm15.call("claude-sonnet-4-5", "Summarize DNA.", system="You are terse.", env=".env")
print(resp.text)
```

### Config from a dict

A good way to manage repetitive config is to put it in a dict and unpack it:
```python
config = {
    "model": "gpt-4.1-mini",
    "system": "You are terse.",
    "temperature": 0.7,
    "max_tokens": 100,
    'env': ".env",
}

resp = lm15.call(prompt="Summarize RNA.", **config)
print(resp.text)
```

This way, next call you can just do that:

```python
resp = lm15.call(prompt="Summarize proteins.", **config)
print(resp.text)
```

---

## `lm15.model()` — a reusable, stateful object

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", env=".env")

resp = gpt("Hello!")
print(resp.text)
```

`model()` creates a **model object**. You configure it once (model name, system prompt, tools, temperature, credentials) and then call it like a function. It **remembers conversation history** — each call sees all previous turns.

```python
gpt = lm15.model("gpt-4.1-mini", env=".env", system="You remember everything.")

gpt("My name is Max.")
gpt("I like chess.")

resp = gpt("What do you know about me?")
print(resp.text)  # Knows name and hobby — it has the full conversation
```

Use `model()` when you need:
- **Conversation** — multi-turn back-and-forth
- **Reuse** — same config across many calls without repeating yourself

> **Tools** and **streaming** work at every level. `lm15.call()` and `lm15.stream()` both accept `tools=`. The difference: `model()` lets you bind tools once and reuse them across turns without repeating yourself. See [Cookbook 02 (Streaming)](02-streaming.md) and [Cookbook 03 (Tools)](03-tools-auto.md).

### Configuration lives on the objectI 

```python
claude = lm15.model("claude-sonnet-4-5",
    env=".env",
    system="You are a helpful coding assistant.",
    temperature=0.7,
    max_tokens=500,
)

# No need to repeat config on each call
resp = claude("Explain Python generators.")
print(resp.text)
```

### Changing settings after creation

The model name, system prompt, tools, and provider are **set once** when you create the object. You can't change them later by editing a property. Instead, use the `with_*` methods — they return a **new** model object with the change applied, leaving the original untouched:

```python
gpt = lm15.model("gpt-4.1-mini", env=".env", system="You are helpful.")

# Create a new model with a different system prompt
terse_gpt = gpt.with_system("You are extremely terse.")

print(gpt.system)        # "You are helpful."       — original unchanged
print(terse_gpt.system)  # "You are extremely terse." — new copy
```

Available methods: `with_model()`, `with_system()`, `with_tools()`, `with_provider()`.

Settings like `temperature` and `max_tokens` **can** be overridden per call without creating a new object:

```python
# Uses the default temperature=0.7 from above
resp = claude("Write a poem.")

# Override just for this one call
resp = claude("Write a poem.", temperature=0.2)
```

### Clearing history

```python
gpt = lm15.model("gpt-4.1-mini", env=".env")

gpt("My name is Max.")
gpt("I like chess.")
print(len(gpt.history))  # 2

gpt.history.clear()
print(len(gpt.history))  # 0 — fresh start
```

> **Deep dive:** [Cookbook 09 — Model Objects and Configuration](09-model-config.md) covers derived models, local caching, config-driven setup, and batch patterns. [Cookbook 07 — Multi-Turn Conversation](07-conversation.md) goes deeper on history and multi-turn.

---

## `UniversalLM` + adapters — full manual control

```python
from lm15 import UniversalLM, LMRequest, Message, Config
from lm15.providers.openai import OpenAIAdapter
from lm15.transports.urllib_transport import UrlLibTransport

# Build the client yourself — you choose the transport, adapters, and keys
client = UniversalLM()
client.register(OpenAIAdapter(api_key="sk-proj-...", transport=UrlLibTransport()))

# Build the request yourself — you control every field
request = LMRequest(
    model="gpt-4.1-mini",
    messages=(Message.user("Hello."),),
    config=Config(temperature=0.5, max_tokens=100),
)

resp = client.complete(request)
print(resp.text)
```

At this level nothing is automatic — no model-name routing, no env var lookup, no tool auto-execution. You wire everything explicitly.

Use this when you need:
- **Custom transports** — proxy, mTLS, custom HTTP client
- **Multiple keys for the same provider** — e.g. two OpenAI orgs side by side
- **Plugin development** — writing your own adapter
- **Testing** — injecting a mock adapter

> **Deep dive:** [Adapter Guide](../ADAPTER_GUIDE.md) for writing custom adapters. [Plugin Guide](../ADD_PROVIDER_GUIDE.md) for packaging and distributing them. [Architecture](../ARCHITECTURE.md) for the full layering.

---

## When to use which

| | `lm15.call()` | `lm15.model()` | `UniversalLM` |
|---|---|---|---|
| **State** | Stateless | Stateful (conversation) | You manage it |
| **Config** | Pass every time | Configure once | Fully manual |
| **Credentials** | `env=` / `api_key=` / env vars | Same | You pass the key to the adapter |
| **Best for** | One-off questions, scripts | Conversations, agents | Plugins, custom transports, tests |
| **Streaming** | `lm15.stream(...)` | `gpt.stream(...)` | `client.stream(request)` |

**Rule of thumb:** start with `call()`. Move to `model()` when you need conversation or reuse. Reach for `UniversalLM` only when the high-level API doesn't give you enough control.

---

## Inspecting the response

Both return the same `LMResponse` object:

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")

print(resp.text)                    # The text content
print(resp.model)                   # Model that actually responded
print(resp.finish_reason)           # "stop", "tool_call", etc.
print(resp.usage.input_tokens)      # Tokens in
print(resp.usage.output_tokens)     # Tokens out
print(resp.usage.total_tokens)      # Total
```

---

## Switching models

Just change the string:

```python
import lm15

# OpenAI
resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")

# Anthropic
resp = lm15.call("claude-sonnet-4-5", "Hello.", env=".env")

# Google
resp = lm15.call("gemini-2.5-flash", "Hello.", env=".env")
```

Same code, same response type, same fields. The model name is the only thing that changes.

If you're unsure which models are currently available, inspect live discovery first:

```python
import lm15

print(lm15.providers_info(env=".env"))
for m in lm15.models(provider="openai", env=".env")[:5]:
    print(m.id)
```

> **Deep dive:** [Cookbook 12 — Model Discovery and Provider Status](12-model-discovery.md)

### Explicit provider

If you use a custom or fine-tuned model whose name lm15 can't auto-detect:

```python
resp = lm15.call("my-fine-tune", "Hello.", provider="openai", env=".env")
```

This applies to both `call()` and `model()`:

```python
gpt = lm15.model("my-fine-tune", provider="openai", env=".env")
```
