# Roadmap

Where lm15-python is, and what is planned before and after the stable 1.0
release. Dates are intentions, not promises; everything here follows the
same discipline as the code — see
[How lm15 is specified](how-lm15-is-specified.md).

## Where we are (June 2026)

- **1.0.0a1** is the current release. The chat core — canonical types,
  serde, errors, request building, response parsing, streaming — is frozen
  by the cross-language contract and enforced mechanically (304 conformance
  checks, spec drift gate, surface ratchet).
- Non-chat endpoints (embeddings, files, batch, image/audio generation) and
  live sessions work and are live-tested, but remain **provisional**: their
  shapes may still change before they are frozen.
- Rust, Go, and TypeScript implementations pass the identical conformance
  corpus. Julia is planned.

## Toward 1.0 stable

The alpha exists so the full documentation can be read end-to-end and the
user experience judged as a whole before the final freeze. In order:

1. **Complete documentation site** (this site) — guides, cookbooks, API
   reference, specification pages, benchmarks.
2. **User-experience review pass** — read the docs as a new user would;
   adjust library ergonomics where the docs reveal friction. Small,
   additive-only changes to the frozen chat core; provisional surfaces may
   still move.
3. **Decide the 1.0 scope** — whether provisional surfaces ship as
   "provisional, clearly labeled" in 1.0 or wait for a later minor.
4. **Release engineering** — tag-driven publishing via PyPI trusted
   publishing (OIDC), CI across Python 3.10–3.14 and Linux/macOS/Windows,
   a type-checking gate alongside the shipped `py.typed`.
5. **Stable release.**

## Provider coverage

Today, with identical canonical behavior and live-receipt fixtures:

- OpenAI (Responses API) and OpenAI Codex
- Anthropic and Claude Code
- Google Gemini, including Live (WebSocket) sessions
- Any Chat Completions–compatible server through one dialect adapter with
  typed compatibility policies — Groq, OpenRouter, DeepSeek, vLLM, SGLang,
  Ollama

Planned: a published, continuously tested compatibility matrix, and
fixture-first coverage of additional hosted endpoints (Azure OpenAI,
Bedrock, Vertex, Mistral, Together, Fireworks, xAI are the candidates).
New providers always land as contract fixtures with live receipts first,
code second.

## Layers above the foundation

lm15 is deliberately low-level: no automatic tool loop, no retries, no
routing, no cost ledger. Several companion pieces are under consideration
once the foundation's user experience is validated — each as a separate
package built on the frozen core, none of them contract-governed:

- **An ergonomic layer** — a concise `call()`-style interface, automatic
  tool loops, retry/fallback patterns, for people who want three lines and
  sensible defaults.
- **A model catalog** — maintained pricing, context-window, and capability
  metadata via the entry-point protocol already specified in
  [model-hydration](model-hydration.md), enabling cost estimation and
  routing.
- **Recipes** — cookbook pages for everything the core deliberately omits
  (retries, fallback, budget caps, proxying), so each "lm15 doesn't do X"
  has a one-page answer.

## Multi-language

- Publish the Rust (crates.io), Go, and TypeScript (npm) implementations,
  each gated on the same 304-check corpus.
- Rebuild the Julia port against the contract.
- The promise stays the same in every language: byte-identical wire
  requests, identical canonical parses, one spec.

## Ecosystem and community

- Integration examples: a FastAPI service, an agent loop, notebooks, and
  migration guides from other clients.
- A fixture-first "add a provider" contributor path (see
  [CONTRIBUTING](https://github.com/lm15-dev/lm15-python/blob/main/CONTRIBUTING.md)).
- Benchmarks stay machine-generated and re-run on a schedule — numbers in
  the README and on this site are never hand-edited.
