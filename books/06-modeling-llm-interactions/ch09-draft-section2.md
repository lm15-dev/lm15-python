## Three Categories

Everything a library does falls into one of three concealment categories. The categories aren't theoretical — they determine the API surface, the documentation, and the user's debugging experience.

### Fully Hidden (and Should Be)

These are pure implementation details. The user doesn't want to see them, wouldn't benefit from seeing them, and shouldn't write code that depends on them.

**Wire format translation.** The fact that OpenAI wants `{"role": "system", "content": "..."}` as a message and Anthropic wants `"system": "..."` as a top-level field is invisible to the user. It should be. The user writes `system="Be concise."` and the adapter places it where the provider expects. The translation is mechanical, lossless, and uninteresting. Exposing it would add complexity without adding capability.

**SSE parsing.** The line protocol — `data:` lines, `event:` lines, empty-line delimiters — is invisible. The user iterates `StreamChunk` objects. They never see bytes, never see SSE events, never see the parser. The parsing is a solved problem with no user-facing implications.

**Auth mechanics.** Bearer tokens vs header keys vs query parameters. The user provides a key. The adapter attaches it to the request in whatever format the provider expects. The user doesn't care whether their key travels in the `Authorization` header (OpenAI), the `x-api-key` header (Anthropic), or the URL query string (Gemini). The auth strategy is infrastructure, not interface.

**Provider routing.** `"claude-sonnet-4-5"` routes to Anthropic. `"gpt-4.1-mini"` routes to OpenAI. The prefix-matching logic in `capabilities.py` is invisible. The user provides a model name and trusts the library to find the right provider. The routing is trivial for standard models and has an explicit override (`provider=`) for non-standard ones.

These are correct hiding decisions. The user's code would be worse if they were exposed — more verbose, more fragile, dependent on implementation details that might change. The one-sentence test passes for all four: each can be described without naming a provider.

### Partially Hidden (and Debatable)

These are behaviors where the abstraction is leaky by nature — where hiding the mechanism creates the impression of uniformity that doesn't exist.

**Prompt caching.** `prompt_caching=True` works on all providers. The user doesn't see the mechanism — cache control markers on Anthropic, automatic detection on OpenAI, CachedContent resources on Gemini. This is convenient. But the cost discount differs (90% vs 50% vs 75%), the lifetime differs (minutes vs automatic vs configurable), and the granularity differs (per-message breakpoints vs prefix detection vs entire-resource caching). A user tuning their caching strategy needs to understand the mechanism. The universal parameter hides it.

lm15's answer: hide the mechanism but expose the metrics. `resp.usage.cache_read_tokens` and `resp.usage.cache_write_tokens` tell the user what happened without exposing how. The user can observe the effect without understanding the cause. This is an incomplete answer — the user who needs to *control* the caching (set a TTL, force a breakpoint, invalidate) can't do so through the universal API and must reach for the escape hatch.

**Reasoning semantics.** `reasoning=True` enables chain-of-thought. But Anthropic's reasoning uses a token budget, OpenAI's uses an effort level, and the dict form (`reasoning={"budget": 10000}`) maps cleanly to Anthropic and not at all to OpenAI. The universal parameter hides the parameterization difference. A user who writes `reasoning={"budget": 10000}` and switches to OpenAI gets... nothing. The budget is ignored. No error, no warning, no indication that the parameter was meaningless.

This is a hiding failure. The parameter looks universal. The behavior is provider-specific. The user who switches providers discovers the non-universality by observing unexpected behavior, not by reading a clear error message. A better design might validate the dict form against the current provider and warn when parameters are irrelevant.

**Model capabilities.** Some models support tools. Some support vision. Some support reasoning. The user finds out when they try — `InvalidRequestError` if the model doesn't support the requested feature. The capabilities are partially discoverable through `lm15.models(supports={"tools"})`, but the discovery is optional. A user who doesn't check and calls a non-tool-supporting model with `tools=[...]` gets an opaque provider error, not a helpful "this model doesn't support tools."

The library hides the capability matrix. Whether it *should* is debatable. Checking capabilities at call time (and raising a clear error) would be more helpful. Not checking preserves the simplicity of the dispatch path. lm15 chose not to check, which means the provider's error is the user's first indication of a capability mismatch.

### Deliberately Exposed (and Should Be)

These are transparency points — places where the library explicitly shows its internals, because the user who needs them shouldn't have to hack around the abstraction to reach them.

**The escape hatch.** `config.provider` and `resp.provider` are untyped dicts that carry provider-specific data. They're the windows in the wall. The user who needs Anthropic's `stop_sequence` or OpenAI's `system_fingerprint` reaches through the hatch. The hatch is documented, visible, and opt-in.

**History.** `model.history` is a public list of `HistoryEntry` objects. The user can inspect every request and response, compute total token usage, count turns, and debug conversation flow. The library accumulates history but doesn't hide it.

**The internal API.** `UniversalLM`, the adapter classes, the types — all importable. A user who needs manual control can construct a `UniversalLM`, register adapters, and call them directly, bypassing the sugar and the Model entirely. The library's internals are public, not because they're stable (they might change between versions) but because a user who needs them shouldn't need to monkey-patch or fork to reach them.

**The types.** `LMRequest`, `LMResponse`, `Part`, `Message`, `Tool`, `Usage` — every type is importable and documented. The v1 API (constructing `LMRequest` objects and calling `UniversalLM` directly) is a fully supported, if verbose, way to use lm15. The types are the foundation, not an implementation detail.

The pattern: **expose the data, hide the mechanism.** The user can see what was sent, what was received, and what the token counts were. The user can't see how the request was serialized, how the SSE was parsed, or how the auth header was constructed. Data is useful to the user. Mechanism is not — unless the user is extending the library, in which case they're reading the source anyway.
