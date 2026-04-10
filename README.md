# lm15

[![PyPI version](https://img.shields.io/pypi/v/lm15.svg)](https://pypi.org/project/lm15/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/lm15.svg)](https://pypi.org/project/lm15/)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One interface for OpenAI, Anthropic, and Gemini. Zero dependencies.

| | lm15 | google-genai | litellm |
|---|---:|---:|---:|
| **install** | 72ms | 137ms | 184ms |
| **import** | 95ms | 2,656ms | 4,534ms |
| **total (install → response)** | **1,090ms** | **3,992ms** | **5,840ms** |
| dependencies | 0 | 25 | 55 |
| disk footprint | 408K | 41M | 155M |

<sub>Median of 10 cold-start runs. Fresh venv, single completion against `gemini-3.1-flash-lite-preview`. [Benchmark source.](benchmarks/cold_start.sh)</sub>

```python
import lm15

resp = lm15.call("claude-sonnet-4-5", "Hello.")
print(resp.text)
```

Switch models by changing the string. Same types, same streaming, same tool calling. That's it.

> Yes, [we know](https://xkcd.com/927/).

## Install

```bash
pip install lm15
```

Set at least one provider key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...         # or GOOGLE_API_KEY
```

Or use a `.env` file and configure once:

```python
import lm15
lm15.configure(env=".env")

# No env= needed on any subsequent call
resp = lm15.call("gpt-4.1-mini", "Hello.")
```

Discover what is available:

```python
import lm15

print(lm15.providers_info())
for m in lm15.models(provider="openai")[:5]:
    print(m.id)
```

## Usage

### Streaming

```python
for text in lm15.call("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")
```

Full event access:

```python
for event in lm15.call("gpt-4.1-mini", "Write a haiku.").events():
    match event.type:
        case "text":     print(event.text, end="")
        case "thinking": print(f"💭 {event.text}", end="")
        case "finished": print(f"\n📊 {event.response.usage}")
```

### Tools (auto-execute)

Pass Python functions — schema is inferred, execution is automatic:

```python
def get_weather(city: str) -> str:
    """Get weather by city."""
    return f"22°C in {city}"

resp = lm15.call("gpt-4.1-mini", "Weather in Montreal?", tools=[get_weather])
print(resp.text)  # "It's 22°C in Montreal."
```

### Tools (manual)

```python
from lm15 import Tool

weather = Tool(name="get_weather", description="Get weather", parameters={...})
gpt = lm15.model("gpt-4.1-mini")

resp = gpt.call("Weather in Montreal?", tools=[weather])
results = {tc.id: "22°C, sunny" for tc in resp.tool_calls}
resp = gpt.submit_tools(results)
print(resp.text)
```

### Inspect before sending

```python
req = lm15.prepare("gpt-4.1-mini", "Weather?", tools=[get_weather])
print(req.tools[0].name)        # "get_weather"
print(req.tools[0].parameters)  # inferred JSON Schema
print(req.messages)             # constructed messages

resp = lm15.send(req)           # send when ready
```

### Images, audio, video, documents

```python
from lm15 import Part

# Image from URL
resp = lm15.call("gemini-2.5-flash", ["Describe this.", Part.image(url="https://example.com/cat.jpg")])

# Image generation → vision (cross-model)
resp = lm15.call("gpt-4.1-mini", "Draw a cat.", output="image")
resp2 = lm15.call("claude-sonnet-4-5", ["What's this?", resp.image])

# Document
resp = lm15.call("claude-sonnet-4-5", ["Summarize.", Part.document(url="https://example.com/paper.pdf")])

# Upload via provider file API
doc = lm15.upload("claude-sonnet-4-5", "contract.pdf")
resp = lm15.call("claude-sonnet-4-5", ["Find liability clauses.", doc])
```

### Structured output (JSON)

```python
resp = lm15.call("gpt-4.1-mini", "Extract: 'Alice is 30.'.",
    system="Return JSON: {name, age}", prefill="{")
data = resp.json  # parsed dict — raises ValueError if not valid JSON
print(data["name"], data["age"])  # Alice 30
```

### Image and audio bytes

```python
# Get generated image as raw bytes
resp = lm15.call("gpt-4.1-mini", "Draw a cat.", output="image")
with open("cat.png", "wb") as f:
    f.write(resp.image_bytes)  # decoded bytes, no base64 wrangling

# Same for audio
resp = lm15.call("gpt-4o-mini-tts", "Say hello.", output="audio")
with open("hello.wav", "wb") as f:
    f.write(resp.audio_bytes)
```

### Reasoning

```python
resp = lm15.call("claude-sonnet-4-5", "Prove √2 is irrational.", reasoning=True)
print(resp.thinking)  # chain of thought
print(resp.text)      # final answer
```

### Conversation

```python
gpt = lm15.model("gpt-4.1-mini", system="You remember everything.")

gpt.call("My name is Max.")
gpt.call("I like chess.")
resp = gpt.call("What do you know about me?")
print(resp.text)  # knows both
```

### Prompt caching

Reduces cost and latency for repeated prefixes — system prompts, long documents, agent loops:

```python
agent = lm15.model("claude-sonnet-4-5",
    system="<long system prompt>",
    tools=[read_file, write_file],
    prompt_caching=True,
)

resp = agent.call("Add tests for auth.")
while resp.finish_reason == "tool_call":
    results = execute(resp.tool_calls)
    resp = agent.submit_tools(results)
    print(f"Cache hit: {resp.usage.cache_read_tokens} tokens")
```

### Prefill

```python
resp = lm15.call("claude-sonnet-4-5", "Output JSON for a person.", prefill="{")
```

### Reusable model with config

```python
gpt = lm15.model("gpt-4.1-mini", system="You are terse.", retries=3, cache=True, temperature=0)
resp = gpt.call("Hello.")

# Override per call
resp = gpt.call("Be creative.", temperature=1.5)

# Derive new models
claude = gpt.copy(model="claude-sonnet-4-5")
```

### Config from dicts

```python
config = {"model": "gpt-4.1-mini", "system": "You are terse.", "temperature": 0}
resp = lm15.call(prompt="Summarize DNA.", **config)
```

### Built-in tools

```python
resp = lm15.call("gpt-4.1-mini", "Latest AI news", tools=["web_search"])
for c in resp.citations:
    print(c.title, c.url)
```

## Provider support

| Capability | OpenAI | Anthropic | Gemini |
|---|:---:|:---:|:---:|
| complete | ✅ | ✅ | ✅ |
| stream | ✅ | ✅ | ✅ |
| embeddings | ✅ | — | ✅ |
| files | ✅ | ✅ | ✅ |
| batches | ✅ | ✅ | ✅ |
| images | ✅ | — | ✅ |
| audio | ✅ | — | ✅ |
| prompt caching | auto | ✅ | ✅ |

## Architecture

```
lm15.call / lm15.acall / lm15.model   ← high-level surface
                │
                ▼
          Result / AsyncResult
                │
                ▼
LMRequest ──▶ UniversalLM ──▶ MiddlewarePipeline ──▶ ProviderAdapter ──▶ Transport
                  │                                        │
                  │ resolve_provider(model)                 │ build_request / parse_stream_event
                  ▼                                        ▼
            capabilities.py                         providers/{openai,anthropic,gemini}.py
```

The high-level surface (`lm15.call`, `lm15.acall`, `lm15.model`, `Result`) is a thin layer over `LMRequest`, `UniversalLM`, and provider adapters. Third parties can still build their own surface on top of the same internals.

## Why this exists

- **Stdlib only.** No `requests`, no `httpx`, no `aiohttp`. Transport is `urllib` or optional `pycurl`.
- **Frozen dataclasses all the way down.** High level: `Result` out. Low level: `LMRequest` / `LMResponse` stay fully accessible. No mutable builder chains.
- **Nothing is hidden.** Every internal type is importable. Provider escape hatches are always there.
- **Plugin discovery via entry points.** Third-party providers install and register without touching lm15 core.

## Docs

| Topic | Path |
|---|---|
| **API v2 spec (legacy)** | [`docs/API_SPEC_V2.md`](docs/API_SPEC_V2.md) |
| Getting started | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| Core concepts | [`docs/CONCEPTS.md`](docs/CONCEPTS.md) |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Provider contract | [`docs/CONTRACT.md`](docs/CONTRACT.md) |
| Portability spec | [`docs/PORTABILITY.md`](docs/PORTABILITY.md) |
| Transport design | [`docs/DESIGN_TRANSPORT.md`](docs/DESIGN_TRANSPORT.md) |
| Error handling | [`docs/ERRORS.md`](docs/ERRORS.md) |
| Streaming | [`docs/STREAMING.md`](docs/STREAMING.md) |
| Writing an adapter | [`docs/ADAPTER_GUIDE.md`](docs/ADAPTER_GUIDE.md) |
| Adding a provider | [`docs/ADD_PROVIDER_GUIDE.md`](docs/ADD_PROVIDER_GUIDE.md) |
| Completeness testing | [`docs/COMPLETENESS.md`](docs/COMPLETENESS.md) |
| Production checklist | [`docs/PRODUCTION_CHECKLIST.md`](docs/PRODUCTION_CHECKLIST.md) |

**Cookbooks v2:** [`docs/COOKBOOKS_V2/`](docs/COOKBOOKS_V2/) — practical examples + references:

1. [Hello World](docs/COOKBOOKS_V2/01-hello-world.md)
2. [Streaming](docs/COOKBOOKS_V2/02-streaming.md)
3. [Tools (auto-execute)](docs/COOKBOOKS_V2/03-tools-auto.md)
4. [Tools (manual loop)](docs/COOKBOOKS_V2/04-tools-manual.md)
5. [Multimodal](docs/COOKBOOKS_V2/05-multimodal.md)
6. [Reasoning](docs/COOKBOOKS_V2/06-reasoning.md)
7. [Conversation](docs/COOKBOOKS_V2/07-conversation.md)
8. [Prompt caching](docs/COOKBOOKS_V2/08-prompt-caching.md)
9. [Model config](docs/COOKBOOKS_V2/09-model-config.md)
10. [Building an agent](docs/COOKBOOKS_V2/10-agent.md)
11. [call()/acall()/Result reference](docs/COOKBOOKS_V2/11-complete-reference.md)
12. [Model discovery and provider status](docs/COOKBOOKS_V2/12-model-discovery.md)

**Cookbooks v1 (low-level):** [`docs/COOKBOOKS/`](docs/COOKBOOKS/) — 8 examples using the internal `LMRequest`/`UniversalLM` API directly.

## License

MIT
