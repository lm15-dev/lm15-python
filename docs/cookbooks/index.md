# lm15 cookbook

Sixteen recipes. Every code block runs; every output block is a real
capture. Style rules live in [STYLE.md](STYLE.md).

## Essentials

- [01 — Your first request](01-first-request.md) — One model string, one
  `Request`, one response: the router as the front door, keys included.
- [02 — Multi-turn conversations](02-conversations.md) — Conversations
  are tuples you build yourself; nothing is hidden in a session object.
- [03 — System & developer prompts](03-system-prompts.md) — Where
  instructions go and how each provider maps them on the wire.
- [04 — Controlling generation](04-generation-config.md) — `Config`:
  temperature, max tokens, stop sequences, and what each provider ignores.
- [05 — Streaming](05-streaming.md) — Typed delta events from
  `router.stream()`, sync and async, without parsing SSE yourself.

## Tools

- [06 — Function tools: define & dispatch](06-function-tools.md) —
  `tool(fn)` derives the schema; you run the loop. `derive()` explains it.
- [07 — Built-in provider tools](07-builtin-tools.md) — Web search, code
  execution, and friends: server-side tools you enable, not implement.
- [08 — Structured output](08-structured-output.md) — JSON schemas in,
  parsed objects out, across all three providers' strict modes.

## Modalities

- [09 — Images, PDFs & documents](09-images-and-documents.md) — Sending
  bytes and URLs as parts; what each provider accepts.
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md) —
  The expensive modalities, plus thinking deltas from reasoning models.

## Beyond chat

- [11 — Prompt caching](11-caching.md) — Cache breakpoints that cut cost
  on long prefixes, with real usage numbers before and after.
- [12 — Embeddings, batch & media generation](12-embeddings-batch-generation.md)
  — The non-chat endpoints: vectors, async batches, image generation.
- [13 — Live sessions (realtime)](13-live-sessions.md) — Bidirectional
  sessions over websockets: events in, audio and text out.

## Production

- [14 — Local & OpenAI-compatible servers](14-local-and-compatible-servers.md)
  — Ollama, vLLM, and friends via `OpenAIChatLM` with `base_url` and compat.
- [15 — Errors, retries & testing](15-errors-and-testing.md) — The typed
  error tree, why retry policy is yours, and testing without a network.
- [16 — Provider passthrough](16-provider-passthrough.md) — Reaching
  provider-only knobs without forking your code off lm15's types.
