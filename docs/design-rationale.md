# Design rationale

Short, honest answers to the questions newcomers ask first. These are
deliberate choices, not accidents; if one stops being right we will change it
and say why here.

(Linking this doc from the README is pending — the README is currently under
maintainer revision.)

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
`AsyncAnthropicLM`, `AsyncGeminiLM`, `AsyncOpenAIChatLM` — with the same
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

Endpoint status: `complete()` and `stream()` are the 1.0 async mirror. The
non-chat endpoints (embeddings, file upload, batch, image, audio, live)
remain sync-only for now; the async classes implement them as methods that
raise `UnsupportedFeatureError` ("use the sync adapter for this endpoint
(async endpoints planned)") so the surface is honest rather than silently
absent.
