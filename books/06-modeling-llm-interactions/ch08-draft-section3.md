## Where Boundaries Leak

Every boundary leaks. A leak is data, behavior, or knowledge crossing a boundary that was supposed to contain it. The question isn't whether leaks exist — they always do — but whether they're contained or corrosive. A contained leak is a pragmatic compromise. A corrosive leak is evidence that the boundary is in the wrong place.

**The escape hatch.** `config.provider` and `resp.provider` carry provider-specific data across the adapter boundary — from the adapter's world (where provider differences are explicit) to the user's world (where they're supposed to be hidden). The boundary says: "adapters are interchangeable, the user doesn't need to know which one ran." The escape hatch says: "actually, sometimes the user does."

This is a contained leak. The data is in an untyped dict, clearly labeled as provider-specific. The user who accesses `resp.provider["stop_sequence"]` knows they're reaching through the abstraction. The leak is explicit and opt-in. It doesn't contaminate the universal types — `LMResponse` has the same fields regardless of provider. The escape hatch is a door in the boundary, not a crack.

**Prompt caching semantics.** `prompt_caching=True` on `Config` triggers different adapter behavior — cache control markers on Anthropic, a CachedContent resource on Gemini, nothing on OpenAI. The boundary between Model and Adapter says: "the Model sends a request, the Adapter handles it." But `prompt_caching=True` is a flag that means different things depending on which adapter receives it. The Model sets the flag. The Adapter interprets it. The interpretation varies. The boundary permits this because `Config.provider` (the dict) carries the flag — but the flag is on the universal `Config`, not in the provider dict, which means the universal type encodes a concept that's semantically non-universal.

This is a mild leak. The flag works — it reduces cost on all providers. But it creates the illusion of behavioral equivalence that Chapter 4 examined in detail. The leak is conceptual, not structural: the boundary holds mechanically (the Model doesn't know which adapter runs), but it leaks semantically (the flag means different things to different adapters).

**Factory coupling.** `factory.py` — `build_default()` — is the most coupled module in lm15. It imports adapter classes, transport classes, auth helpers, env file parsers, plugin loaders, and capability hydrators. It touches every layer. It's the wiring layer — the code that assembles the system at startup.

This coupling is necessary. Something must know about everything in order to connect everything. But `build_default()` is also where accidental complexity accumulates. API key resolution, env file parsing, transport selection, adapter instantiation, plugin discovery, models.dev hydration — these are six concerns in one function. Each concern is simple. Together, they're a 130-line function that's harder to test and harder to modify than any other piece of lm15.

The leak here is gravitational — `build_default()` pulls concerns toward itself because it's the natural place to put "startup logic." Adding a new provider? The factory needs to know about it. New auth method? Factory. New env var format? Factory. The module grows because it's the one place that crosses all boundaries, and crossing all boundaries attracts responsibility.

**The Model and types.** The Model class imports from `types.py` (to construct `LMRequest`, `Config`, `Message`) and from `client.py` (to call `UniversalLM`). It also imports from `stream.py` (to create `Stream` objects) and from `errors.py` (to classify retryable errors). The Model depends on four other modules. Is this a leak?

No — this is the normal shape of a layered architecture. Upper layers depend on lower layers. The Model is the highest-logic layer (below the sugar), and it depends on the types, the client, the stream, and the error hierarchy. The reverse dependencies don't exist: the Client doesn't import Model. The Stream doesn't import Model (it receives a callback). The types don't import anything. The dependency graph is acyclic and top-down.

The test for leaks isn't "does this module import from other modules?" It's "does a change in module A force a change in module B when the boundary between them was supposed to prevent that?" Importing types is using the shared vocabulary. Importing to circumvent a boundary is a leak. The distinction is subtle but structural.

### What Leaks Tell You

A leak is information about the boundary's fitness. A small leak — the escape hatch, the prompt caching flag — means the boundary is mostly right but the world is more complex than the abstraction admits. This is normal. All abstractions simplify, and the gap between the simplification and reality is where leaks live.

A large leak — a module that imports extensively from another layer, a type that carries provider-specific semantics in universal fields, a boundary that requires changes on both sides for most modifications — means the boundary might be in the wrong place. The abstraction isn't simplifying enough to justify its existence.

lm15's leaks are small. The escape hatch is contained. The prompt caching flag is a mild semantic leak. The factory's coupling is necessary (something must wire the system). The Model's imports are normal downward dependencies. None of these are evidence that a boundary should be moved. They're evidence that boundaries are imperfect — which is always true, and usually fine.
