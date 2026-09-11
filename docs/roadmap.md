# Roadmap

Where lm15-python is, and what is planned before and after the stable 1.0
release. Dates are intentions, not promises; everything here follows the
same discipline as the code — see
[How lm15 is specified](how-lm15-is-specified.md).

## Where we are (September 2026)

- **1.0.0rc1** is the current release candidate. Install it explicitly with
  `pip install lm15==1.0.0rc1`; it is not the stable 1.0 release. The chat core
  — canonical types, serde, errors, request building, response parsing,
  streaming — is checked against the pinned language-neutral contract.
  Publishing this candidate does not ratify draft contract changes.
- Non-chat endpoints — files, batch, image and speech generation,
  job-shaped video generation (Sora, Veo, grok-imagine) — and live
  sessions (Gemini Live and OpenAI's GA Realtime protocol) work, are
  live-tested, and are covered by contract checks, but remain
  **provisional**: their shapes may still change before they are frozen.
- **Shipped in the alpha** (additive, outside the frozen core, not yet
  contract-governed): the [model-string router](using-the-router.md)
  (`LMRouter`/`AsyncLMRouter`) and
  [tool derivation from functions](tools-from-functions.md)
  (`lm15.tool`/`derive_tool`). A cross-language porting spec is
  [proposed](router-portability.md), pending ratification.
- The earlier Rust, Go, TypeScript, and Julia implementations drifted
  from the contract and were deliberately deleted; rebuilds from the
  contract are underway, module by module (the auth module has landed in
  all four), each module gated on its slice of the conformance corpus.

## Toward 1.0 stable

The release candidate lets users test the package and read the full
documentation before the stable release. Remaining work includes:

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

- OpenAI (Responses API) and OpenAI Codex, including GA Realtime (live)
  sessions, Sora video, images, and speech
- Anthropic and Claude Code
- Google Gemini, including Live (WebSocket) sessions, Veo video, images,
  and speech
- xAI (Grok), including subscription OAuth, images, and grok-imagine
  video
- Any Chat Completions–compatible server through one dialect adapter with
  typed compatibility policies — Groq, OpenRouter, DeepSeek, vLLM, SGLang,
  Ollama

Planned: a published, continuously tested compatibility matrix, and
fixture-first coverage of additional hosted endpoints (Azure OpenAI,
Bedrock, Vertex, Mistral, Together, Fireworks are the candidates).
New providers always land as contract fixtures with live receipts first,
code second.

## Layers above the foundation

lm15 is deliberately low-level: no automatic tool loop, no retries, no
cost ledger, no *policy* routing (the shipped
[router](using-the-router.md) is a lookup table — no fallbacks, no
ranking). Several companion pieces are under consideration
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

- Finish rebuilding the Rust, Go, TypeScript, and Julia implementations
  from the contract, module by module, each module gated on its slice of
  the 477-check corpus (auth has landed in all four).
- Publish them (crates.io, Go module, npm) once they pass the full
  corpus — never before.
- The promise stays the same in every language: byte-identical wire
  requests, identical canonical parses, one spec.

## Ecosystem and community

- Integration examples: a FastAPI service, an agent loop, notebooks, and
  migration guides from other clients.
- A fixture-first "add a provider" contributor path (see
  [CONTRIBUTING](https://github.com/lm15-dev/lm15-python/blob/main/CONTRIBUTING.md)).
- Benchmarks stay machine-generated and re-run on a schedule — numbers in
  the README and on this site are never hand-edited.
