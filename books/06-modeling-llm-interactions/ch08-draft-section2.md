## The Five Boundaries

Chapter 5 examined the adapter boundary in detail — Protocol vs inheritance, per-provider vs per-model, translation vs delegation. This section examines all five boundaries, but through a different lens: not *how* each one is structured, but *what changes on each side* and how the boundary isolates the change.

**Sugar ↔ Model.** The sugar (`api.py`) changes when the user-facing API evolves — a new parameter on `complete()`, a new convenience function like `upload()`. The Model (`model.py`) changes when internal orchestration evolves — tool execution logic, history management, retry mechanics. These are different change vectors. Adding `prefill=` to `complete()` required one line in `api.py` (pass it through) and ten lines in `model.py` (handle it). The sugar didn't need to understand prefill. It forwarded it.

The sugar is 7 lines of delegation. This isn't laziness — it's the boundary doing its job. The sugar's simplicity is evidence that the boundary is in the right place: the sugar changes rarely and minimally, because it does almost nothing.

**Model ↔ Client.** The Model changes when application-level concerns change — how tools are auto-executed, how history accumulates, how retries are managed, how per-call overrides work. The Client changes when infrastructure-level concerns change — how providers are registered, how middleware is composed, how requests are dispatched.

The test: can you change the tool auto-execution loop (Model) without touching provider dispatch (Client)? Yes. Can you add middleware (Client) without touching conversation management (Model)? Yes. The change vectors are independent.

The boundary leaks in one direction: the Model constructs `LMRequest` objects (from the types layer) and passes them to `UniversalLM.complete()`. This is a downward dependency — the Model depends on the Client's interface. This is normal in layered architectures (upper layers depend on lower layers). The reverse dependency doesn't exist — the Client doesn't know about Model, history, or tool execution.

**Client ↔ Adapter.** This is the most important boundary in the system, because adapters change most frequently. When OpenAI adds a new response field, the OpenAI adapter updates. When Anthropic changes their cache control format, the Anthropic adapter updates. When Gemini adds a new model family, the Gemini adapter updates. None of these changes touch the Client, because the Client doesn't know about wire formats. It dispatches `LMRequest` and receives `LMResponse`. The adapter handles everything in between.

This boundary is also the plugin boundary. Third-party adapters (`lm15-x-mistral`) depend on the Protocol and the types. They don't depend on the Client's implementation. The boundary's narrowness — five methods and a set of frozen dataclasses — is what makes plugins viable.

**Adapter ↔ Transport.** The adapter changes when a provider's API changes (new endpoints, new parameters, new error formats). The transport changes when HTTP handling changes (pycurl support, timeout policy, proxy configuration). These are different concerns.

The transport boundary is the least glamorous and most stable. HTTP doesn't change. `urllib` doesn't change. The `HttpRequest`/`HttpResponse` types are trivial. The boundary exists not because the transport is complex, but because it might be *swapped*: urllib for pycurl, real HTTP for a mock in tests, standard transport for a logging wrapper that records traffic. The boundary enables substitution, and substitution enables testing — which is a benefit that's easy to undervalue until you try to test adapters without it.

### What the Boundaries Share

Between every pair of layers, frozen dataclasses flow. `LMRequest` flows from Model to Client to Adapter. `LMResponse` flows back. `StreamEvent` flows from Adapter through Client to Stream. `HttpRequest` and `HttpResponse` flow between Adapter and Transport.

These types are the contracts. They're the narrowest possible interface between layers — a few fields, immutable, with no methods beyond construction and access. The boundaries are narrow because the types are narrow. If `LMRequest` had methods that adapters needed to call — `request.validate()`, `request.optimize()` — the boundary would widen, and changes to those methods would cascade across it. Frozen dataclasses with no behavior are the thinnest possible contract, and thin contracts make boundaries durable.
