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
from lm15 import LMRequest, Message, Part, build_default

lm = build_default()
resp = lm.complete(LMRequest(
    model="claude-sonnet-4-5",
    messages=(Message.user("Hello."),),
))
print(resp.text)
```

Switch models by changing the string. Same types, same streaming protocol, same error hierarchy. That's it.

> Yes, [we know](https://xkcd.com/927/).

## Why this exists

Every LLM wrapper either (a) ships a massive SDK per provider or (b) papers over differences with a lossy abstraction. lm15 takes a different cut:

- **Stdlib only.** No `requests`, no `httpx`, no `aiohttp`. Transport is `urllib` or optional `pycurl`.
- **Frozen dataclasses all the way down.** `LMRequest` in, `LMResponse` out. No mutable builder chains, no hidden state.
- **Nothing is hidden.** Provider-specific options pass through `Config.provider`. The normalized types cover the common surface; the escape hatch is always there.
- **Plugin discovery via entry points.** Third-party providers install and register without touching lm15 core.

## Install

```bash
pip install lm15            # stdlib transport (urllib)
pip install lm15[speed]     # + pycurl transport
```

Set at least one provider key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...         # or GOOGLE_API_KEY
```

## Usage

### Streaming

```python
req = LMRequest(model="gpt-4.1-mini", messages=(...))
for event in lm.stream(req):
    if event.delta_text is not None:
        print(event.delta_text, end="")
```

### Tool calling

```python
from lm15 import Tool, ToolConfig

tools = (Tool(name="get_weather", description="...", parameters={...}),)  # type defaults to "function"
req = LMRequest(model="gemini-2.5-flash", messages=(...), tools=tools)
resp = lm.complete(req)
# resp.message.parts may contain Part(type="tool_call", ...)
```

### Embeddings, images, audio, files, batches

```python
lm.embeddings(EmbeddingRequest(model="text-embedding-3-small", input=["hello"]))
lm.image_generate(ImageGenerationRequest(model="gpt-image-1", prompt="a cat"))
lm.audio_generate(AudioGenerationRequest(model="gpt-4o-mini-tts", input="hi", voice="alloy"))
lm.file_upload(FileUploadRequest(...), provider="openai")
lm.batch_submit(BatchRequest(...))
```

### Middleware

```python
from lm15 import with_retries, with_cache, with_history

lm.middleware.add(with_retries(max_retries=3))
lm.middleware.add(with_cache({}))
lm.middleware.add(with_history([]))
```

### External plugins

Any installed package can register a provider adapter:

```toml
# pyproject.toml of the plugin package
[project.entry-points."lm15.providers"]
myprovider = "lm15_x_myprovider:build_adapter"
```

```python
lm = build_default(discover_plugins=True)
```

### models.dev catalog

Hydrate model capabilities from the [models.dev](https://models.dev) catalog at startup:

```python
lm = build_default(hydrate_models_dev_catalog=True)
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
| live | — | — | — |

## Architecture

```
LMRequest ──▶ UniversalLM ──▶ MiddlewarePipeline ──▶ ProviderAdapter ──▶ Transport
                  │                                        │
                  │ resolve_provider(model)                 │ build_request / parse_response
                  ▼                                        ▼
            capabilities.py                         providers/{openai,anthropic,gemini}.py
```

- **`types.py`** — all request/response/stream types as frozen dataclasses
- **`protocols.py`** — `LMAdapter` and `LiveSession` protocols
- **`providers/base.py`** — `BaseProviderAdapter` with default `complete`/`stream` implementations over raw HTTP
- **`transports/`** — `urllib` and `pycurl` backends behind a `Transport` protocol
- **`capabilities.py`** — model→provider resolution + optional models.dev hydration
- **`middleware.py`** — composable retry / cache / history wrappers
- **`plugins.py`** — entry-point discovery for third-party adapters

## Docs

| Topic | Path |
|---|---|
| Getting started | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| Core concepts | [`docs/CONCEPTS.md`](docs/CONCEPTS.md) |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Provider contract | [`docs/CONTRACT.md`](docs/CONTRACT.md) |
| Error handling | [`docs/ERRORS.md`](docs/ERRORS.md) |
| Streaming | [`docs/STREAMING.md`](docs/STREAMING.md) |
| Writing an adapter | [`docs/ADAPTER_GUIDE.md`](docs/ADAPTER_GUIDE.md) |
| Adding a provider | [`docs/ADD_PROVIDER_GUIDE.md`](docs/ADD_PROVIDER_GUIDE.md) |
| Completeness testing | [`docs/COMPLETENESS.md`](docs/COMPLETENESS.md) |
| Production checklist | [`docs/PRODUCTION_CHECKLIST.md`](docs/PRODUCTION_CHECKLIST.md) |

**Cookbooks:** [`docs/COOKBOOKS/`](docs/COOKBOOKS/) — 8 progressive examples from basic text to plugin development.

## License

MIT
