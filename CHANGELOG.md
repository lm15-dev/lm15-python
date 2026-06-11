# Changelog

## 1.0.0a1 — 2026-06-11

**The stability promise.** The chat core — canonical types, serde, errors,
request building, response parsing, streaming — is frozen; all future changes
to it are additive (enforced mechanically by the surface ratchet and spec
drift gate). Non-chat endpoints and live sessions remain provisional; see
`lm15-contract/spec/SCOPE.md`.

What backs the promise:

- **Four independent implementations** — Python (this package), Rust, Go,
  TypeScript — each passing the identical 304-check conformance corpus
  (`lm15-dev/lm15-contract`), each live-tested against real providers
  including the full tool-calling round-trip.
- **A written, ratified spec**: 61 types, 25 vocabularies, 49 numbered
  invariants, mapping rules MAP-1..3, one omission rule, one number rule.
- **Every fixture carries provenance**; wire fixtures change only with live
  receipts; the reference implementation holds no oracle authority.
- **Measured, regenerable benchmarks**: 0 dependencies, 0.5 MiB installed,
  171 ms cold import, and faster than raw stdlib HTTP at steady state
  (connection pooling).

Changes since 0.3.0: prompt-caching fixtures recaptured (GA, no beta
header); OpenAI file inputs send `filename` (provider drift caught by the
live sweep); `FunctionTool.parameters` always emitted, `{}` round-trips
verbatim; malformed nested config objects reject instead of silently
dropping; `Result` and live sessions no longer contain any automatic
tool-execution machinery.

## 0.3.0 — 2026-06-11

Ground-up rewrite. `lm15` is now a **low-level foundation library**: one
canonical representation, exact serde, provider adapters — and nothing
opinionated. The 0.2.x high-level API (`lm15.call()`, `Model`, `Conversation`,
cost tracking, middleware, REPL) is **gone by design**; build it (or your own
take) on top. Pin `lm15==0.2.*` if you depend on the old surface.

### The canonical core
- Typed, frozen, immutable canonical model: `Request`/`Response`, `Message`,
  typed `Part`s (text, thinking, media, tool calls/results, citations),
  `Config`, `Usage`, stream events.
- Exact canonical JSON serde with written rules: one omission rule, opaque
  payloads never mutated, declared number types (`serde-rules.md`).
- Normalized error hierarchy (`AuthError` with key/credential guidance,
  `RateLimitError.retry_after`, `ContextLengthError`, ...).
- Mapping invariants written and pinned: provider-executed tools are not
  parts (MAP-1), response messages are never empty (MAP-2), a stream yields
  exactly one end event carrying finish_reason and usage (MAP-3).

### Providers
- First-party adapters: OpenAI (Responses), Anthropic, Gemini.
- `OpenAIChatLM`: the Chat Completions dialect with compat presets for
  ollama, Groq, OpenRouter, vLLM, SGLang — live-validated against Groq,
  ollama, vLLM, and SGLang.
- Native async mirrors of every adapter (`AsyncOpenAILM`, ...): same
  constructor, same canonical types, no thread-wrapping.
- Local subscription adapters: `ClaudeCodeLM` (Claude Code OAuth) and
  `OpenAICodexLM` (Codex/ChatGPT OAuth).
- Stdlib-only HTTP/1.1 sync + async transports; `websockets` is the single
  optional extra (live sessions).

### Conformance
- Behavior is pinned by the cross-language `lm15-contract` corpus: 108
  request cases, 108 reviewed response/stream goldens, error and serde
  vectors, all live-captured or hand-authored with provenance, verified by a
  language-neutral harness (`python -m lm15.vet`).
- A written spec (types, vocabularies, 48 numbered invariants) with a
  reflection-based drift gate.

### Optional model metadata
- `ModelRegistry.discover()` hydrates advisory pricing/context metadata from
  installed catalogs (entry-point group `lm15.model_catalogs`); never affects
  what adapters send.

## 0.2.0 and earlier

The previous-generation high-level SDK, developed in the `lm15-python`
repository. See its history there.
