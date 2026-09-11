# lm15

Talk to every major LLM API through one small, typed interface.

Every provider speaks its own HTTP dialect — different JSON shapes,
auth headers, streaming formats, error codes. lm15 absorbs exactly
that. You write one `Request`, and a per-provider adapter translates
it, byte-for-byte correctly, into the dialect of OpenAI, Anthropic,
Gemini, Groq, OpenRouter, a local ollama/vLLM/SGLang server, or your
Claude/ChatGPT subscription. Same code, same typed `Response` back,
whichever provider answers:

```python
from lm15 import LMRouter, Message, Request

response = LMRouter().complete(
    Request(model="anthropic:claude-haiku-4-5",
            messages=(Message.user("Hello!"),))
)
print(response.text)
```

Three things make it different:

**It is tiny, and measurably so.** Zero dependencies — pure stdlib,
including the HTTP transport with connection pooling. Adding lm15 to
your project costs half a megabyte:

| package | install size | transitive deps | cold import | import RSS |
|---|---:|---:|---:|---:|
| **lm15** | 0.5 MiB | 0 | 152 ms | 16.6 MiB |
| openai | 18.0 MiB | 15 | 468 ms | 35.3 MiB |
| anthropic | 17.1 MiB | 15 | 589 ms | 41.2 MiB |
| google-genai | 37.2 MiB | 24 | 934 ms | 60.8 MiB |
| litellm | 133.0 MiB | 54 | 2298 ms | 161.0 MiB |

*(Machine-measured, re-run mechanically, never hand-edited — see
[Benchmarks](benchmarks.md).)*

**It is exact, and provably so.** The translation layer is not
best-effort: every provider behavior is pinned by fixtures captured
from real providers and enforced by a language-neutral conformance
contract — 477 machine-run checks covering requests, responses,
streams, errors, serde, auth, files, batch, generation, video, and
live sessions, on every commit. When lm15 says two providers behave
the same, that is a tested claim, not a hope.
[How lm15 is specified](how-lm15-is-specified.md) tells that story.

**It has no opinions — on purpose.** No automatic retries, no
tool-execution loop, no cost ledger, no prompt DSL. You can use lm15
directly (the docs show how, start to finish), and it is equally built
to be **the dependency** underneath opinionated libraries — you bring
the opinions, lm15 brings every provider. What it leaves out is a
design decision with a written reason, not a gap
([design rationale](design-rationale.md)).

## The mental model

```text
Message parts → Message → Request → ProviderLM → Response
                              │          ▲
                              │          └── LMRouter("provider:model" → LM)
                              └── stream() → StreamEvent → materialized Response
```

Read it left to right: you compose typed `Message`s into a `Request`,
a provider LM (picked by hand or by the router from a model string)
sends it, and you get back a `Response` — or a stream of typed events
that materializes into the identical `Response`.

## Where to go

- **New here?** [Getting started](getting-started.md) — an API key and
  your first call in about five minutes, every example with real
  captured output.
- **Which providers? What does a model cost?**
  [Providers & models](providers-and-models.md).
- **LM Studio or an unlisted OpenAI-compatible server?**
  [Connect your own server](connecting-openai-compatible-servers.md) — default
  behavior, custom compat objects, and offline request checks.
- **Keys, rotating tokens, subscriptions:**
  [Authentication](authentication.md).
- **Coming from litellm or the OpenAI SDK?**
  [Migrating from an OpenAI-shaped client](migrating-from-openai-chat.md)
  — `request_from_openai_chat` reads the messages you already have into
  a `Request`, and names every key it cannot carry.
- **"How do I do X?"** — nineteen [cookbook recipes](cookbooks/index.md),
  from [first request](cookbooks/01-first-request.md) to
  [video generation](cookbooks/14-video-generation.md) and
  [live sessions](cookbooks/15-live-sessions.md).
- **"Why should I trust the translation?"**
  [How lm15 is specified](how-lm15-is-specified.md) — the contract,
  the authority model, and what "frozen" means here. The part of lm15
  you won't find anywhere else.
- **[API reference](reference/types.md)**, **[Benchmarks](benchmarks.md)**,
  and the **[Roadmap](roadmap.md)**.
