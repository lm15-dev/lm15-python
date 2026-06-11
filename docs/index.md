# lm15

**lm15** is a small, typed, provider-neutral interface for foundation-model
requests, responses, streams, tools, media parts, endpoint APIs, errors, and
canonical JSON serialization — **stdlib-only, zero dependencies**, with its
own HTTP transport.

It is a **low-level foundation library**: one canonical representation,
exact serde for it, and adapters that translate it to and from each
provider's wire format. It is deliberately *not* an opinionated user-facing
API — no magic `call()`, no automatic tool loops, no DSL. lm15 is meant to
be **the dependency** for libraries that want to build their own take on
the right way to talk to AI systems in Python: you bring the opinions,
lm15 brings every provider.

| package | install size | transitive deps | cold import | import RSS |
|---|---:|---:|---:|---:|
| **lm15** | 0.5 MiB | 0 | 152 ms | 16.6 MiB |
| openai | 18.0 MiB | 15 | 468 ms | 35.3 MiB |
| anthropic | 17.1 MiB | 15 | 589 ms | 41.2 MiB |
| google-genai | 37.2 MiB | 24 | 934 ms | 60.8 MiB |
| litellm | 133.0 MiB | 54 | 2298 ms | 161.0 MiB |

*(Machine-measured — see [Benchmarks](benchmarks.md).)*

## The mental model

```text
Message parts → Message → Request → ProviderLM → Response
                              │          ▲
                              │          └── LMRouter("provider:model" → LM)
                              └── stream() → StreamEvent → materialized Response
```

One `Request` shape drives OpenAI (Responses API), Anthropic, Gemini,
Claude Code, OpenAI Codex, and every Chat Completions–compatible server
(Groq, OpenRouter, DeepSeek, vLLM, SGLang, Ollama, …) — with identical
canonical behavior, enforced by a cross-language conformance contract.

## Where to go

- **[Getting started](getting-started.md)** — install and first requests.
- **Guides** — the [router](using-the-router.md), the
  [type system](using-the-type-system.md),
  [tools from functions](tools-from-functions.md),
  [providers](using-the-a-provider.md),
  [model profiles](using-model-profiles.md), and
  [transports](using-the-transports.md).
- **Cookbooks** — [69 features across three providers](cookbook-all-features.md)
  and [Gemini Live sessions](cookbook-gemini-live.md), with real captured
  output.
- **[How lm15 is specified](how-lm15-is-specified.md)** — the
  cross-language contract, the authority model, and what "frozen" means
  here. The part of lm15 you won't find anywhere else.
- **[API reference](reference/types.md)** — generated from the source.
- **[Benchmarks](benchmarks.md)** and the **[Roadmap](roadmap.md)**.
