# Design rationale

Short, honest answers to the questions newcomers ask first. These are
deliberate choices, not accidents; if one stops being right we will change it
and say why here.

## Why `config=Config(...)` instead of kwargs?

`lm15` is the foundation layer, not the DSL. Convenience surfaces like
`complete(model, prompt, temperature=0.7)` belong to the libraries built on
top, and each of them will make different choices about defaults, naming, and
which knobs to expose. Keeping generation settings in one explicit, frozen
`Config` value gives those layers a single stable thing to construct, hash,
compare, serialize, and pass through — and keeps the `Request` signature from
accreting a kwarg per provider feature. Sugar belongs to the layer above.

## Why `extensions` on requests but `provider_data` on responses?

They are different directions with different ownership, so they get different
names on purpose:

- `extensions` (request): user-supplied passthrough. You wrote it, you are
  asking the adapter to forward it to the provider.
- `provider_data` (response): provider-returned raw material. The provider
  produced it, the adapter is preserving it for you verbatim.

A single name (e.g. `extra`) would suggest the two are symmetric or
round-trippable. They are not: echoing `provider_data` back as `extensions`
is almost always a bug.

## Why no `call()`, no `Model` object, no automatic tool loop?

Same boundary. A `call()` helper, a stateful `Model` with memory, and an
agentic execute-tools-until-done loop are all opinionated DSL features: they
require policy decisions (retries, history truncation, tool sandboxing,
loop-termination rules) that the foundation has no business hard-coding.
`lm15` gives you the exact canonical request/response/stream vocabulary and
provider adapters; loops and ergonomics are intentionally left to the layer
above. `Result` exists only to assemble streams, not to run conversations.
`Result` previously contained an automatic tool-execution loop; it was
removed on 2026-06-11 as a positioning violation. Live sessions had their own
copy of that loop (callable registry, tool-call callback, auto-execution); it
was removed for the same reason — a session surfaces tool calls as events and
the caller sends results back.

## Why tuples everywhere, but lists accepted?

Canonical values (`Message.parts`, `Request.messages`, `Request.tools`, ...)
are stored as tuples because the types are frozen: immutable values are
hashable, safe to share across threads and caches, and cannot be mutated
behind an adapter's back after validation. But forcing callers to type
`(Message.user("hi"),)` is hostile, so constructors coerce lists (and, where
unambiguous, a single bare item) into tuples at the boundary. You may pass a
list; you will always read a tuple.

## Async

Async support ships as separate mirror classes — `AsyncOpenAILM`,
`AsyncAnthropicLM`, `AsyncGeminiLM`, `AsyncOpenAIChatLM`,
`AsyncClaudeCodeLM`, `AsyncOpenAICodexLM`, `AsyncXaiLM` — with the same
constructor fields, the same canonical `Request` in, and the same canonical
`Response`/stream events out as their sync siblings. `await` is the only
user-visible difference: `complete()` is `async def`, `stream()` returns an
`AsyncIterator[StreamEvent]` (coalesced per MAP-3 by
`lm15.result.acoalesce_stream`, the async twin of `coalesce_stream`).

They are built by composition, not inheritance. Subclassing the sync adapter
and overriding sync methods with async ones would be a typing violation —
`complete` would no longer be substitutable for the base signature. Instead,
each async class owns the async transport (`lm15.transports
.StdlibAsyncTransport` by default) and delegates every pure transformation —
`build_request`, `parse_response`, `parse_stream_events`, `normalize_error`,
payload/header helpers — to an inner instance of the sync adapter class
constructed with a transport that raises if it is ever used: the inner
adapter must never touch the network, so the contract-pinned mapping code
stays single-sourced and the async classes cannot drift from it. The one
sync method that does need the network, `GeminiLM.resolve_prompt_cache`, is
ported onto `AsyncGeminiLM` against the async transport; `complete()` and
`stream()` invoke it first, mirroring the sync class.

Endpoint status: `complete()` and `stream()` came first; the non-chat
endpoints (files, batch, image and speech generation, video jobs, and
live sessions on the live-capable providers) now have native async
implementations driving the same pure build/parse hooks as the sync
classes. Endpoints a provider does not offer raise
`UnsupportedFeatureError` in both mirrors, so the surface stays honest
rather than silently absent.

## Why does a "no routing" library now ship a router?

Because every program that uses lm15 was writing the same eight lines —
map a model name to a class, find the env var, construct, cache — and a
mapping table is foundation-shaped, while *policy* routing (retries,
fallbacks, cost-based selection) is not. `LMRouter` is deliberately the
former and refuses to be the latter: four fixed resolution rungs (an
object rung for model values that carry their own provider, explicit
`provider:` prefix, opt-in catalog, built-in prefix rules),
first match wins, no callbacks, no fallback chains, no configurable rung
order. `resolve()` is pure and its `Resolution` return value is the
explanation — there is no hidden state to ask about.

The honest trade-offs:

- **The built-in rule table goes stale.** A brand-new model family won't
  match until a release ships — by design. The mitigations are all
  user-visible data: the explicit prefix always works, a catalog
  (e.g. `aimo` via `ModelRegistry.discover()`) is opt-in, and
  `DEFAULT_RULES` is replaceable as plain tuples.
- **Ambiguity is an error, not a preference.** When a catalog offers one
  id under two providers we raise `AmbiguousModelError` instead of
  ranking them. Ranking is policy; the foundation doesn't have one.
- **No `base_url`/transport syntax in model strings.** Encoding endpoint
  configuration into strings is where stringly-typed routers rot.
  `router.lm()` returns the ordinary provider LM, so the escape hatch to
  direct construction is the return value itself — both paths are
  first-class, and the cookbook cases (ollama, vLLM, Azure, OpenRouter)
  stay on the direct path.

Cross-language: the algorithm is pure data + four rungs precisely so
Rust/Go/TS/Julia can port it idiomatically (a struct table, an exported
slice, a sync `resolve()` everywhere). The porting spec is in
[router-portability](router-portability.md) — a proposal until ratified.

## Why `ResponseStream` and not `Result` (the 2026-07-13 API review)

A four-lens fresh-eyes panel (cold learner, write-from-memory test,
API-design critic, downstream library author) reviewed the whole public
surface at 1.0.0a1 — the one moment renames were still legal. The
streaming surface took the biggest change: `Result` was renamed
`ResponseStream` (constructor now positional:
`ResponseStream(events, request)`), `.events()` now yields the same
canonical `StreamEvent`s the raw stream carries instead of a second
chunk vocabulary, and one push-based engine — `StreamAccumulator` —
now backs the sync skin, the async mirror (`AsyncResponseStream`), and
`materialize_response`/`amaterialize_response` alike.

The reasons, briefly: `result` is the most common variable name in
user code, so the class shadowed itself in every lesson; the library
itself already re-exported it under a second name (`Stream`); the
constructor still carried hooks from the tool-execution loop deleted a
month earlier; the second vocabulary (`StreamChunk`) was stringly
typed and not actually streaming for tool calls; and a pull-generator
class cannot be mirrored in Rust or Go, while a push accumulator can —
`Result` also collides with Rust's `std::result::Result` outright.

The same review curated the top level (161 → 107 exports; serde
pairs, adapter machinery, and the router's data tables each live one
module deep), split the two protocols that shared the `ProviderLM`
name (the callable surface keeps it; the wire-mapping seam is
`ProviderDialect`), and made hyphens canonical in provider strings
(`openai-chat`, underscore spelling aliased forever). Full findings:
`architecture-review/api-review-2026-07-13.md` in the workspace repo.

## Why `api_key` accepts a callable (and lm15 still has no auth dependencies)

A credential is not always a static string. Azure Entra tokens expire
hourly; OAuth tokens refresh; enterprises rotate keys. Every serious
client library ends up needing a *token provider* — and the ecosystem
already standardized its shape: `azure.identity.get_bearer_token_provider`
returns exactly a zero-argument callable producing a string.

So `api_key: str | Callable[[], str]`, resolved at request-build time,
once per request. A static string is the degenerate constant provider —
nothing changes for existing callers. The split follows the same line as
the concurrency ruling in the contract: credential **placement** (which
header, what format) is spec'd, testable data; credential **acquisition**
(fetching, refreshing, caching tokens) is behavior, per-language idiom,
and the *caller's* job. lm15 never depends on `azure-identity`, `boto3`,
or `google-auth`; it only places what your callable returns. The cloud
doors added a second, deliberately narrow way in (2026-09-03,
2026-09-19): with no credential at all the router walks the cloud's own
default chain, and `credentials={"azure": "platform"}` runs one named
identity — the metadata server or token endpoint spoken directly, with
the standard library. Both say what they picked (the doctor before, every
auth error after); neither is a second identity framework, and the
explicit callable stays the path an expert reaches for.

The subscription adapters (`ClaudeCodeLM`, `OpenAICodexLM`) use the same
seam internally: they validate the local CLI credential at construction
(typed, re-login-guided errors), then re-resolve per request — a
long-lived client picks up refreshed tokens without being rebuilt.
Credential material never appears in reprs, whichever form it takes.

## Why the router grew an object rung and preset routes

Two seams kept forcing users back into boilerplate the router exists to
remove. First: catalog packages (aimo) ship model ids as `str`
subclasses that *know their provider* — but passing one as a string
threw that knowledge away, turning resolvable models into
`AmbiguousModelError`s. Rung 0 reads a duck-typed `provider` attribute:
no package is named, plain strings never trigger it, and the most
intentional signal available wins. Second: the catalog would resolve
`groq`, `OpenAIChatLM` shipped a live-validated groq preset with the
right base_url — and the router still refused, because the bridge
between "provider string" and "compat preset" wasn't wired. Preset
routes are that bridge, as data: provider string → preset name +
env-key convention + keyless placeholder. Only presets with pinned,
live-validated base URLs qualify; everything else keeps the explicit
`OpenAIChatLM(base_url=...)` escape hatch.

## Why no tools from functions?

A tool is data: a name, a description, and a JSON Schema for its inputs.
lm15 takes it written out, the same way in every language, and does not
read one off a function.

From June to September 2026 the Python package had `tool(fn)`, which
derived the schema from a signature, its type hints and its docstring
(Julia had `@tool`). It was removed before 1.0, on 2026-09-23:

- **It cannot be the same everywhere.** TypeScript, Rust and Go cannot
  read a function's input types while a program runs, so the contract
  already ruled it out for them. A feature that exists in two languages
  out of six breaks "learn it once, use it in any language".
- **It had already drifted.** Python's and Julia's versions built
  different tools from the same function (Julia closed the schema with
  `"additionalProperties": false`). Two answers to one question is what
  the contract exists to prevent.
- **It is a convenience with opinions**: which docstring style, which
  types map to what, strict or open schemas. That is the job of the
  library built on lm15, which can make those choices once for all the
  languages it supports.

A request holds only data, so a function passed in `tools` is refused,
and the error says what to write instead. lm15 never executes tools.
