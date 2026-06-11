# lm15

**lm15** is a small, typed, provider-neutral interface for foundation-model
requests, responses, streams, tools, media parts, endpoint APIs, errors, and
canonical JSON serialization. This repository is its Python reference
implementation.

**What lm15 is — and deliberately is not.** lm15 is a low-level foundation
library: one canonical representation, exact serde for it, and adapters that
translate it to and from each provider's wire format — stdlib-only, with its
own HTTP transport (`websockets` is the single optional extra, for live
sessions). It is NOT an opinionated user-facing API: no magic `call()`, no
automatic tool loops, no DSL. lm15 is meant to be **the dependency** for
libraries that want to build their own take on the right way to talk to AI
systems in Python — you bring the opinions, lm15 brings every provider.

The public API is the top-level package: `from lm15 import AnthropicLM,
Request, Message, ...` (see `lm15/__init__.py` for the full curated surface).
Transport plumbing stays under `lm15.transports`, live sessions under
`lm15.live`, and the conformance shim under `lm15.vet`.

The code blocks below are documentation that runs: every ```output``` block is
the real, captured output of the example above it.

<!-- footprint:generated:start -->
## Footprint

Measured by [`benchmarks/suite/run.py`](benchmarks/suite/run.py) on Python 3.13.3 — methodology and full results in [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md):

| package | install size | transitive deps | cold import | import RSS |
|---|---:|---:|---:|---:|
| **lm15** | 0.5 MiB | 0 | 152 ms | 16.6 MiB |
| openai | 18.0 MiB | 15 | 468 ms | 35.3 MiB |
| anthropic | 17.1 MiB | 15 | 589 ms | 41.2 MiB |
| google-genai | 37.2 MiB | 24 | 934 ms | 60.8 MiB |
| litellm | 133.0 MiB | 54 | 2298 ms | 161.0 MiB |
| langchain-openai | 63.3 MiB | 35 | 930 ms | 61.0 MiB |
<!-- footprint:generated:end -->

## Install

```bash
python3 -m pip install lm15
# Optional extra for websocket live sessions:
python3 -m pip install 'lm15[live]'
```

Or from source, for development:

```bash
git clone https://github.com/lm15-dev/lm15-python && cd lm15-python
python3 -m pip install -e '.[live]'
```

lm15 has zero required dependencies — it is stdlib-only, including its HTTP
transports.

## Quickstart

```python
import os

from lm15 import Config, Message, OpenAILM, Request

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])

response = lm.complete(
    Request(
        model="gpt-4.1-mini",
        system="You are terse.",
        messages=(Message.user("Say hello in three words."),),
        config=Config(max_tokens=50, temperature=0.2),
    )
)

print(response.text)
print(response.finish_reason)
print(response.usage.total_tokens)
```

```output
Hello there, friend.
stop
27
```

The mental model is one straight line:

```text
Message parts → Message → Request → ProviderLM → Response
                              │
                              └── stream() → StreamEvent → materialized Response
```

## One Request, every provider

The exact same `Request` shape drives the three first-party adapters:

```python
import os

from lm15 import AnthropicLM, GeminiLM, Message, Request

providers = [
    AnthropicLM(api_key=os.environ["ANTHROPIC_API_KEY"]),
    GeminiLM(api_key=os.environ["GEMINI_API_KEY"]),
]

for lm in providers:
    response = lm.complete(
        Request(
            model={
                "anthropic": "claude-sonnet-4-5",
                "gemini": "gemini-3-flash-preview",
            }[lm.provider],
            messages=(Message.user("Say hello."),),
        )
    )
    print(lm.provider, response.text)
```

```output
anthropic Hello! How can I help you today?
gemini Hello! How can I help you today?
```

And the same shape reaches every OpenAI-compatible server through
`OpenAIChatLM`, the Chat Completions dialect adapter. A compat preset name —
`"ollama"`, `"groq"`, `"openrouter"`, `"vllm"`, `"sglang"`, ... — bundles
that server's wire-format quirks *and* its default `base_url`, so a local
Ollama is one constructor argument away:

```python
from lm15 import Config, Message, OpenAIChatLM, Request

lm = OpenAIChatLM(api_key="ollama", compat="ollama")  # base_url -> http://localhost:11434/v1

response = lm.complete(
    Request(
        model="qwen3.5:0.8b",
        messages=(Message.user("Say hello in five words or fewer."),),
        config=Config(max_tokens=80, extensions={"reasoning_effort": "none"}),
    )
)

print(response.text)
```

```output
Hello there! I'm ready to help. What would you like me to discuss?
```

Swap `compat="groq"` (plus your Groq key) or `compat="openrouter"` and the
same request hits those servers; pass an explicit `base_url` to point a
preset anywhere. Server-specific knobs ride in `Config.extensions` and pass
through verbatim.

## Streaming

`stream()` yields typed `StreamEvent` objects. Text arrives as
`StreamDeltaEvent(delta=TextDelta(...))`, and the stream is normalized across
providers: exactly one `StreamEndEvent` ends the stream, carrying
`finish_reason` and `usage` (mapping rule MAP-3).

```python
import os

from lm15 import Message, OpenAILM, Request, StreamDeltaEvent, TextDelta

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])
request = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Write one short sentence about Montreal."),),
)

for event in lm.stream(request):
    if isinstance(event, StreamDeltaEvent) and isinstance(event.delta, TextDelta):
        print(event.delta.text, end="", flush=True)
```

```output
Montreal is a vibrant, multicultural city in Canada known for its rich history and festivals.
```

To consume a stream into a full `Response`:

```python
from lm15 import materialize_response

response = materialize_response(lm.stream(request), request)
print(response.text)
```

```output
Montreal is a vibrant, multicultural city in Canada known for its rich history and cuisine.
```

The materialized `Response` is identical in shape to one from `complete()` —
same `message`, `finish_reason`, `usage`, and `provider_data`.

## Tools: the full round-trip

lm15 distinguishes **function tools** that your application executes from
**provider-native built-in tools** like web search. Here is the complete
function-tool round-trip — model asks, you run your function, you answer back:

```python
import os

from lm15 import FunctionTool, Message, OpenAILM, Request

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])

def get_weather(city: str) -> str:
    return f"Sunny and 22°C in {city}."

weather_tool = FunctionTool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

messages = (Message.user("What is the weather in Montreal?"),)
request = Request(model="gpt-4.1-mini", messages=messages, tools=(weather_tool,))

response = lm.complete(request)
for call in response.tool_calls:
    print(call.name, call.input)
```

```output
get_weather {'city': 'Montreal'}
```

Now run your function and hand the result back. The model's tool-call turn is
`response.message`; your answer is `Message.tool({call_id: result})`:

```python
call = response.tool_calls[0]
result = get_weather(**call.input)

messages = (*messages, response.message, Message.tool({call.id: result}))
final = lm.complete(Request(model="gpt-4.1-mini", messages=messages, tools=(weather_tool,)))
print(final.text)
```

```output
The weather in Montreal is sunny with a temperature of 22°C. Would you like to know the forecast for the coming days or any other information?
```

lm15 will never run the loop for you — that's your layer. This is the whole
loop.

Built-in tools are provider-executed; you just declare them and read the
results (citations come back as typed parts):

```python
from lm15 import BuiltinTool, Message, Request

response = lm.complete(
    Request(
        model="gpt-4.1-mini",
        messages=(Message.user("Where will the 2028 Summer Olympics be held? One sentence, cite a source."),),
        tools=(BuiltinTool("web_search"),),
    )
)

print(response.text)
for citation in response.citations:
    print(citation.title, citation.url)
```

```output
The 2028 Summer Olympics are scheduled to be held in Los Angeles, California, United States, from July 14 to 30, 2028. ([britannica.com](https://www.britannica.com/event/Los-Angeles-2028-Summer-Olympic-Games?utm_source=openai))
Los Angeles 2028 Summer Olympic Games | Bidding, Host, Venues, Planning, Sports, Marketing, & Facts | Britannica https://www.britannica.com/event/Los-Angeles-2028-Summer-Olympic-Games?utm_source=openai
```

## Async

Every adapter has an async mirror — `AsyncOpenAILM`, `AsyncAnthropicLM`,
`AsyncGeminiLM`, `AsyncOpenAIChatLM`, `AsyncClaudeCodeLM`,
`AsyncOpenAICodexLM` — with the same constructor fields, the same canonical
`Request` in, and the same `Response`/stream events out. `await` is the only
difference: `complete()` is `async def`, and `stream()` is an
`async for`-able iterator of the same events.

```python
import asyncio

from lm15 import (
    AsyncOpenAIChatLM,
    Config,
    Message,
    Request,
    StreamDeltaEvent,
    TextDelta,
)

async def main() -> None:
    lm = AsyncOpenAIChatLM(api_key="ollama", compat="ollama")
    request = Request(
        model="qwen3.5:0.8b",
        messages=(Message.user("Name two colors."),),
        config=Config(max_tokens=80, extensions={"reasoning_effort": "none"}),
    )

    response = await lm.complete(request)
    print(response.text)

    async for event in lm.stream(request):
        if isinstance(event, StreamDeltaEvent) and isinstance(event.delta, TextDelta):
            print(event.delta.text, end="", flush=True)
    print()

asyncio.run(main())
```

```output
Two examples of natural and artificial colors are **red** and **blue**.
Two common names for a color are **red** (or crimson) and **blue** (often called indigo, cobalt, or azure). Other examples include green, yellow, purple, and brown.
```

The non-chat endpoints (embeddings, files, batch, image, audio, live) are
sync-only for now; the async classes raise `UnsupportedFeatureError` for them
rather than pretending. Async endpoint mirrors are planned.

## Local subscription adapters

The ordinary provider adapters use API keys that callers pass explicitly:
`OpenAILM(api_key=...)`, `AnthropicLM(api_key=...)`, and
`GeminiLM(api_key=...)`.

lm15 also has explicit local-developer subscription adapters for users who are
already signed in to provider CLIs. These adapters do not read API-key
environment variables. They read local OAuth credentials created by the CLI and
send provider-specific OAuth headers.

### Claude Code subscription auth

Use `ClaudeCodeLM.from_claude_code()` when Claude Code is installed and logged
in as the same OS user:

```python
from lm15 import ClaudeCodeLM, Config, Message, Request

lm = ClaudeCodeLM.from_claude_code()

response = lm.complete(
    Request(
        model="claude-fable-5",
        messages=(Message.user("Say hello briefly."),),
        config=Config(max_tokens=128),
    )
)

print(response.text)
```

The default credential path is `~/.claude/.credentials.json`. If the
credential is missing or expired, run Claude Code and log in again (`claude`,
then `/login` if prompted).

`ClaudeCodeLM` always prepends the Claude Code system prompt required by this
OAuth route:

```text
You are Claude Code, Anthropic's official CLI for Claude.
```

If `Request.system` is also provided, lm15 keeps both: the required Claude Code
prompt comes first, then the caller's system instruction.

Fable 5 note: Fable may spend part of `max_tokens` on hidden thinking, so a
too-small budget can return no visible text with `finish_reason="length"`.
Use `Config(max_tokens=128)` or higher for non-trivial prompts.

### OpenAI Codex / ChatGPT subscription auth

Use `OpenAICodexLM.from_codex_cli()` when Codex CLI is installed and signed in
with ChatGPT:

```python
from lm15 import Message, OpenAICodexLM, Request

lm = OpenAICodexLM.from_codex_cli()

response = lm.complete(
    Request(
        model="gpt-5.5",
        messages=(Message.user("Say hello briefly."),),
    )
)

print(response.text)
```

The default credential path is `~/.codex/auth.json`. `OpenAICodexLM` reads the
local ChatGPT OAuth access token and account id from that file, then calls the
Codex subscription endpoint. The Codex subscription backend is
streaming-first, so `complete()` internally streams and materializes a normal
`Response`.

Current Codex route note: lm15 intentionally omits max-token fields here
because the verified local Codex route accepts the request shape without them;
set output limits in your application layer if you need a hard cap.

These subscription adapters are intended for local interactive development, not
server or CI deployments. Treat the credential files as secrets; do not print or
log their bearer tokens.

## Media and non-chat endpoints

Multimodal input uses typed media parts (`ImagePart`, `AudioPart`,
`DocumentPart`, ...):

```python
import os

from lm15 import ImagePart, Message, OpenAILM, Request, TextPart

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])

request = Request(
    model="gpt-4.1-mini",
    messages=(
        Message.user([
            TextPart("Describe this image in a few words."),
            ImagePart(
                url="https://raw.githubusercontent.com/github/explore/main/topics/react/react.png",
                media_type="image/png",
                detail="low",
            ),
        ]),
    ),
)

print(lm.complete(request).text)
```

```output
This image shows a blue atomic symbol, often used to represent an atom or atomic energy.
```

Non-chat endpoints have separate request/response types — `EmbeddingRequest`,
`ImageGenerationRequest`, `AudioGenerationRequest`, `FileUploadRequest`,
`BatchRequest`, `LiveConfig`:

```python
from lm15 import EmbeddingRequest

embeddings = lm.embeddings(
    EmbeddingRequest(
        model="text-embedding-3-small",
        inputs=("hello", "world"),
    )
)
print(len(embeddings.vectors), len(embeddings.vectors[0]))
```

```output
2 1536
```

## Canonical JSON serialization

The serde functions convert every public lm15 type to canonical
JSON-compatible dicts and back, exactly — this is the wire format the
conformance corpus pins:

```python
from lm15 import Message, Request, request_from_dict, request_to_dict

request = Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),))
wire = request_to_dict(request)
round_tripped = request_from_dict(wire)
round_tripped == request
```

```output
True
```

## Error normalization

Provider-specific HTTP/API errors are normalized into one lm15 error
hierarchy, so callers handle `AuthError`, `RateLimitError`,
`ContextLengthError`, ... identically across providers:

```python
import os

from lm15 import AuthError, Message, OpenAILM, ProviderError, RateLimitError, Request

lm = OpenAILM(api_key="not a key")

try:
    lm.complete(Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),)))
except AuthError as exc:
    print("Check API key:", exc.env_keys)
except RateLimitError as exc:
    print("Retry later:", exc.retry_after)
except ProviderError as exc:
    print(exc.provider, exc.provider_code, exc.status, exc.request_id)
```

```output
Check API key: ('OPENAI_API_KEY',)
```

## Model metadata

`ModelRegistry.discover()` hydrates optional, advisory model metadata
(pricing, context windows, capability hints) from installed catalog packages
via the `lm15.model_catalogs` entry-point group — the `aimo` catalog is one
such package. Hydrated metadata never changes what an adapter sends: requests
are byte-identical with or without it. See
[docs/model-hydration.md](docs/model-hydration.md) for the contract.

## Design notes

- [docs/design-rationale.md](docs/design-rationale.md) — why `config=Config(...)`
  instead of kwargs, why there is no automatic tool loop, why request
  `extensions` and response `provider_data` are different names on purpose.
- [docs/serde-rules.md](docs/serde-rules.md) — the canonical JSON omission and
  round-trip rules.
- [docs/mapping-rules.md](docs/mapping-rules.md) — the provider mapping
  invariants (MAP-1, MAP-2, MAP-3, ...).
- Behavior is pinned by a cross-language conformance corpus: the sibling
  `lm15-contract` repository is the spec; this package is the reference
  implementation, not the authority.

## Contributing

Fixture and conformance workflows, the doc-drift checker, the provider
adapter development guide, and the useful-commands cheat sheet live in
[CONTRIBUTING.md](CONTRIBUTING.md).
