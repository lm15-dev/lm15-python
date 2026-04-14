# Cookbook 15 — Provider Passthrough

lm15 abstracts the common surface — messages, tools, temperature, reasoning — but every provider has features that only they offer. The `provider` passthrough lets you use them without dropping down to raw HTTP.

## How it works

Any key you put in `provider={}` gets merged into the request body, alongside the fields lm15 already sets. The SDKs don't interpret these — they go straight to the wire.

## OpenAI: logprobs

```python
import lm15

resp = lm15.call(
    "gpt-4.1-mini",
    "Say cat",
    max_tokens=16,
    provider={
        "top_logprobs": 2,
        "include": ["message.output_text.logprobs"],
    },
)
print(resp.text)
# Access raw response for logprobs data
print(resp.raw)
```

## OpenAI: store and metadata

```python
resp = lm15.call(
    "gpt-4.1-mini",
    "Hello",
    provider={
        "store": True,
        "metadata": {"session": "abc-123", "user": "demo"},
    },
)
```

## OpenAI: truncation

```python
resp = lm15.call(
    "gpt-4.1-mini",
    "Summarize this long thread.",
    messages=long_conversation,
    provider={"truncation": "auto"},
)
```

## OpenAI: service tier

```python
resp = lm15.call(
    "gpt-4.1-mini",
    "Hello",
    provider={"service_tier": "default"},
)
```

## Anthropic: stop sequences

```python
resp = lm15.call(
    "claude-sonnet-4-5",
    "Count upward from 1 with commas.",
    max_tokens=32,
    provider={"stop_sequences": ["5"]},
)
print(resp.text)  # "1,2,3,4,"
```

## Anthropic: metadata

```python
resp = lm15.call(
    "claude-sonnet-4-5",
    "Hello",
    max_tokens=64,
    provider={"metadata": {"user_id": "user-abc-123"}},
)
```

## Anthropic: output_config (structured JSON)

```python
resp = lm15.call(
    "claude-sonnet-4-5",
    "Return a JSON object with animal set to cat.",
    max_tokens=128,
    provider={
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"animal": {"type": "string"}},
                    "required": ["animal"],
                },
            }
        }
    },
)
```

## Gemini: safety settings

```python
resp = lm15.call(
    "gemini-2.5-flash",
    "Tell me about Montreal.",
    provider={
        "safetySettings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
    },
)
```

## Gemini: response format (JSON mode)

```python
resp = lm15.call(
    "gemini-2.5-flash",
    "Return a JSON object with keys title and summary about DNA.",
    provider={
        "generationConfig": {
            "responseMimeType": "application/json",
        }
    },
)
```

## Gemini: response schema (structured output)

```python
resp = lm15.call(
    "gemini-2.5-flash",
    "List a few cookie recipes.",
    provider={
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "recipeName": {"type": "string"},
                        "ingredients": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["recipeName", "ingredients"],
                },
            },
        }
    },
)
```

## Mixing passthrough with native fields

Native fields (`temperature`, `max_tokens`, `tools`, etc.) and `provider` work together. lm15 sets the native fields, then merges the passthrough on top:

```python
resp = lm15.call(
    "gpt-4.1-mini",
    "What is the weather?",
    temperature=0.5,                   # native — mapped by lm15
    tools=[get_weather],               # native — mapped by lm15
    provider={"tool_choice": "auto"},  # passthrough — merged as-is
)
```

## Using passthrough with the low-level API

```python
from lm15 import LMRequest, Config, Message, FunctionTool, build_default

lm = build_default()

request = LMRequest(
    model="gpt-4.1-mini",
    messages=(Message.user("Hello"),),
    config=Config(
        temperature=0.5,
        provider={
            "top_logprobs": 3,
            "include": ["message.output_text.logprobs"],
            "service_tier": "default",
        },
    ),
)

resp = lm.complete(request)
```

## Using passthrough on model objects

```python
import lm15

gpt = lm15.model("gpt-4.1-mini")

# provider dict goes through the call's provider kwarg
# (currently requires the low-level path — see above)
```

## What goes native vs passthrough?

| If lm15 has a parameter for it | Use the native parameter |
|---|---|
| `temperature`, `max_tokens`, `top_p` | `lm15.call(..., temperature=0.7)` |
| `system` | `lm15.call(..., system="You are terse.")` |
| `tools` (function + builtin) | `lm15.call(..., tools=[fn, "web_search"])` |
| `reasoning` | `lm15.call(..., reasoning=True)` |
| `stop` | `lm15.call(..., stop=["END"])` |

| If lm15 doesn't abstract it | Use `provider={}` |
|---|---|
| `top_logprobs`, `include` | `provider={"top_logprobs": 2, ...}` |
| `service_tier`, `metadata`, `store` | `provider={"service_tier": "default"}` |
| `safetySettings` (Gemini) | `provider={"safetySettings": [...]}` |
| `output_config` (Anthropic) | `provider={"output_config": {...}}` |

## Provider passthrough is provider-specific

The same `provider={}` dict produces different results depending on which provider handles the model. OpenAI fields won't work on Anthropic and vice versa. If you need portability, use lm15's native parameters.

```python
# ✅ Portable — works on any provider
lm15.call("gpt-4.1-mini", "Hello", temperature=0.5)
lm15.call("claude-sonnet-4-5", "Hello", temperature=0.5)

# ⚠️  Provider-specific — only works on OpenAI
lm15.call("gpt-4.1-mini", "Hello", provider={"service_tier": "default"})
```
