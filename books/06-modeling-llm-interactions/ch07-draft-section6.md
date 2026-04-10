## The Identity Question

Strip away the pragmatics — the version conflicts, the cold start measurements, the security incidents — and a philosophical question remains. Not "how many dependencies should a library have?" but "what is a library?"

There are two views.

**The composition view.** A library is an assembly. It combines specialized components — an HTTP client, a validation framework, an async runtime, a serialization layer — into something greater than the parts. The library's identity is its *API*: the surface it presents to users, the abstractions it offers, the problems it solves. The implementation is an assembly of other people's work, and that's fine — good, even. Change the HTTP client underneath and the library is still the library. The components are interchangeable. The identity is the interface.

LangChain is a composition. It assembles `httpx`, `pydantic`, `numpy`, `tiktoken`, provider SDKs, vector store clients, and dozens of other packages into an orchestration framework. No single team could write all of this. The composition model is the only model that works at LangChain's scope — 200,000+ lines, dozens of integrations, an ecosystem of plugins. The dependencies aren't a choice. They're a structural necessity. The library is too large to be self-contained.

**The artifact view.** A library is a thing. Every line is written, every byte is owned. The library's identity is its *implementation*: the code itself, not just the surface. There is no "underneath." The HTTP code isn't a component that could be swapped — it IS the library. The types aren't validated by a framework — they validate themselves. The library is its code, the way a novel is its words. You don't swap out the sentences of a novel and call it the same book.

lm15 is an artifact. 2,408 lines, 30 files, zero dependencies. The HTTP transport is lm15. The SSE parser is lm15. The types, the adapters, the middleware — all lm15. You can read the entire implementation in an afternoon, because the entire implementation is one thing written by one team with one set of decisions.

**Neither view is wrong.** They optimize for different properties.

The composition optimizes for *capability*. By assembling the best HTTP client, the best validator, the best async runtime, you get a library that's better at HTTP, better at validation, better at async than any single team could build. The cost is coupling — each component's release schedule, version constraints, and breaking changes propagate through the composition.

The artifact optimizes for *reliability*. By owning every line, you control every behavior. No dependency can break you. No transitive package can introduce a vulnerability. No version conflict can prevent installation. The cost is capability — you're limited to what you can build and maintain yourself, which is less than what the ecosystem offers.

The composition scales up gracefully. Adding a new integration — a new provider, a new vector store, a new embedding model — means adding a dependency and writing a thin adapter. The marginal cost of growth is low.

The artifact scales up painfully. Adding a new integration means building it from scratch — the HTTP calls, the error handling, the edge cases. The marginal cost of growth is high. This is why lm15 has three providers and LangChain has dozens. The artifact model works when the scope is small. It becomes unsustainable as the scope grows.

**The scope determines the answer.** The question "how many dependencies should a library have?" reduces to "how large is the library's scope?"

Small scope, narrow needs, hostile deployment environments → zero dependencies. The library is an artifact. It fits in the developer's head and on a serverless function.

Large scope, broad needs, server-side deployment → many dependencies. The library is a composition. It's too large for one team to own every line, and the deployment environment can absorb the dependency cost.

lm15's zero dependencies isn't a principle about dependencies. It's a consequence of being 408K of focused code that does one thing — call LLMs across providers — in environments where cold start matters. If lm15's scope expanded to include vector stores, document loaders, agent frameworks, and embedding pipelines, zero dependencies would be untenable. The scope would demand the composition model.

This is the answer to the chapter's question. Not "are dependencies good or bad?" — they're a tool, and tools are context-dependent. The answer is: **the dependency count follows from the scope.** Know your scope. Know your deployment environment. The dependency decision makes itself.

A library that has zero dependencies and shouldn't will be fragile, limited, and exhausting to maintain — reimplementing solved problems, missing edge cases, falling behind specialized tools. A library that has fifty dependencies and shouldn't will be bloated, conflict-prone, and hostile to deploy — imposing costs on users who can't refuse them. The mistake, in both cases, is a mismatch between the dependency count and the scope. Get the scope right and the dependency question answers itself.
