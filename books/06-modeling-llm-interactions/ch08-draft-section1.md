# Chapter 8: Layering and Boundaries

Every library has layers. The question isn't whether to have them — code that does more than one thing inevitably separates into parts — but whether the boundaries between the parts are in the right places. A boundary in the right place absorbs change: something shifts on one side and nothing moves on the other. A boundary in the wrong place creates ceremony: two files, two abstractions, two sets of imports, and every change touches both.

There is a simple test for whether a boundary is real.

## The Substitution Test

Can you replace the thing on one side of the boundary without changing the thing on the other side?

If yes, the boundary is doing its job. The two sides are genuinely independent. Changes on one side don't cascade across the boundary.

If no, the boundary is decoration. It looks like separation — different files, different classes, different modules — but the two sides are coupled. A change on one side requires a change on the other, and the boundary adds indirection without adding isolation.

Apply this to lm15's five boundaries:

**Transport ↔ Adapter.** Can you replace `urllib` with `pycurl` without changing any adapter? Yes. The adapters call `self.transport.request(HttpRequest(...))` and receive an `HttpResponse`. They don't know which transport is underneath. lm15 swaps transports at construction time in `build_default()`, and no adapter code changes. The boundary is real.

**Adapter ↔ Client.** Can you add a Mistral adapter without changing `UniversalLM`? Yes. You write a class that satisfies the `LMAdapter` Protocol, register it with `client.register()`, and the client dispatches to it. The client's code — routing, middleware, dispatch — is unchanged. The boundary is real. This is the plugin boundary, and it's the most important one in the architecture.

**Client ↔ Model.** Can you add a retry mechanism without knowing about any provider? Yes. Retries live in `Model._complete_with_cache()` (or in middleware). They wrap the client call without knowing whether it goes to OpenAI, Anthropic, or Gemini. The boundary is real.

**Model ↔ Sugar.** Can you use `lm15.complete()` without knowing that a `Model` object exists? Yes. The function creates a Model internally, calls it, and returns the response. The user never sees the Model. The boundary is real.

Now a more revealing question: **Can you replace the `Part` type without changing anything else?**

No. `Part` is used in `Message` (types), in `Model` (tool execution), in adapters (wire format translation), in `Stream` (event accumulation), in the user's code (constructing prompts). It crosses every boundary. Changing `Part` would require changes to every layer.

But this isn't a layering failure. `Part` isn't behind a boundary — it IS the shared vocabulary. It's the contract that all layers agree on. Boundaries separate things that change independently. Shared types connect things that must agree. Confusing the two — trying to put a boundary where you need a shared type, or sharing types where you need a boundary — is the most common layering mistake.

**Boundaries exist between things that change independently. Shared types exist between things that must agree.** The five layers change independently. The types they communicate through do not — and shouldn't. When `Part` gains a new type value (a new content type), every adapter that handles that type updates. This isn't a leak. It's the system responding to a change in the shared vocabulary, which is exactly what shared vocabulary is for.
