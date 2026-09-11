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
- Non-chat endpoints, stored-cache resources, live sessions and Chat
  Completions ingest ship as **provisional**. Contract tests cover recorded
  behavior; they do not promise that every provider/account works live.
  The stability boundary below applies even after the package reaches 1.0.
- The [model-string router](using-the-router.md) and
  [tool derivation from functions](tools-from-functions.md) are available.
  The contract's [API-family playbook](https://github.com/lm15-dev/lm15-contract/blob/main/playbooks/api-family.md)
  governs the shared public names; an older router-portability proposal is
  not the authority for the current implementation.
- Python, Rust and TypeScript pass the shared corpus at their recorded
  pins. That is evidence of agreement on the recorded cases, not a release
  approval or a claim about untested runtime behavior.

## What ships in 1.0, and what is stable

**Already decided: provisional features ship in 1.0, clearly labeled.**
This follows the ratified contract's
[`spec/SCOPE.md`](https://github.com/lm15-dev/lm15-contract/blob/main/spec/SCOPE.md),
not a new release-policy decision.

- **Frozen:** the canonical chat types, serialization, errors, request and
  response mapping, streaming, credential resolution and model listing.
  Removing or changing frozen behavior requires a major release and
  maintainer ratification. Additions follow the contract's change process.
- **Provisional:** non-chat endpoints (files, batches, image/speech/video
  generation), stored-cache resources, live sessions and Chat Completions
  ingest. These are included, but may change incompatibly during 1.x with
  a contract `changes/` entry. Additive changes are preferred, not guaranteed.
  Pin an exact package version and review change entries before upgrading
  applications that use them.
- **Out of scope:** embeddings and canonical provider-executed computer
  use. The contract describes the limits of provider-specific passthrough;
  it is not a stability promise for those features.

The stability of a feature is separate from its test coverage. In particular,
typed media *inside chat* belongs to the frozen chat model; the standalone
media-generation endpoints are provisional.

## Toward 1.0 stable

The release candidate lets users test the package and read the full
documentation before the stable release. Remaining work includes:

1. **Complete documentation site** (this site) — guides, cookbooks, API
   reference, specification pages, benchmarks.
2. **User-experience review pass** — read the docs as a new user would;
   adjust library ergonomics where the docs reveal friction. Small,
   additive-only changes to the frozen chat core; provisional surfaces may
   still move.
3. **Check scope labeling** — the scope is settled above. Keep the
   provisional notices on the relevant guides and release notes; do not
   advertise the entire exported package as a frozen API.
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

Azure, Bedrock and Vertex already have contract coverage; see
[cloud hosts](cloud-hosts.md) for their distinct credentials and settings.
Provider/model availability and live verification remain separate from
passing recorded cases. Additional providers require evidence-backed
fixtures before implementation.

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

- Keep Python, Rust and TypeScript aligned with their contract pins and
  runtime tests. Other ports must meet the same gates before claiming
  parity; the corpus grows, so use the current reports rather than a
  historical fixed check count.
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
