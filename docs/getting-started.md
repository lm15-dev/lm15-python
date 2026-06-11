# Getting started

## Install

```bash
python3 -m pip install --pre lm15
# Optional extra for websocket live sessions:
python3 -m pip install --pre 'lm15[live]'
```

lm15 has zero required dependencies — it is stdlib-only, including its
HTTP transports. (`--pre` is needed while the current release is a
pre-release; it will be unnecessary at 1.0 stable.)

## First request

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

print(response.text)            # Hello there, friend.
print(response.finish_reason)   # stop
print(response.usage.total_tokens)
```

## The same Request, other providers

```python
import os

from lm15 import AnthropicLM, GeminiLM, Message, Request

for lm, model in (
    (AnthropicLM(api_key=os.environ["ANTHROPIC_API_KEY"]), "claude-sonnet-4-5"),
    (GeminiLM(api_key=os.environ["GEMINI_API_KEY"]), "gemini-3-flash-preview"),
):
    response = lm.complete(Request(model=model, messages=(Message.user("Say hello."),)))
    print(lm.provider, response.text)
```

And every OpenAI-compatible server through the Chat Completions dialect
adapter — a compat preset bundles the server's wire-format quirks *and*
its default `base_url`:

```python
from lm15 import Message, OpenAIChatLM, Request

lm = OpenAIChatLM(api_key="ollama", compat="ollama")  # -> http://localhost:11434/v1
response = lm.complete(Request(model="qwen3.5:0.8b", messages=(Message.user("Hi!"),)))
```

Presets exist for `"ollama"`, `"groq"`, `"openrouter"`, `"vllm"`,
`"sglang"`, and more — see [Using the providers](using-the-a-provider.md).

## Streaming

```python
for text in lm.stream(Request(model="gpt-4.1-mini", messages=(Message.user("Tell me a story."),))):
    print(text, end="", flush=True)
```

`stream()` returns a lazy `Result`: iterate it for text chunks, iterate
`.events()` for typed chunks (tool calls, thinking, audio…), or read
`.text` to block until the end. Streaming and non-streaming materialize
into the **same** `Response`.

## Async

Every provider has an async mirror — same API, `Async` prefix:

```python
from lm15 import AsyncOpenAILM

lm = AsyncOpenAILM(api_key=os.environ["OPENAI_API_KEY"])
response = await lm.complete(request)
```

## Where next

- Multimodal input, tools, structured output, reasoning, caching, batch,
  embeddings: the [all-features cookbook](cookbook-all-features.md) shows
  all of it, with real captured output.
- The canonical types in depth: [Using the type system](using-the-type-system.md).
- Why the API looks the way it does: [Design rationale](design-rationale.md).
