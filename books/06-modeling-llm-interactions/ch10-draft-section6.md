## Modeling the Unfinished

Every design decision in this book was made with incomplete information.

The content model (Chapter 1) was designed before citations, refusals, and audio content blocks existed. The Part type — one class, twelve optional fields, an open discriminator — was a bet that new content types would arrive and that the type should absorb them without structural changes. The bet paid off: citations, refusals, and audio arrived and were added as new type values. The structure didn't change. But the bet could have failed: if a provider had introduced a content type that couldn't be expressed as a discriminator value plus optional fields — an interactive widget, a live data stream, a 3D object with its own rendering protocol — the Part type would have needed surgery.

The conversation model (Chapter 2) was designed before million-token context windows existed. The decision to send everything and cache the prefix was a bet on window growth. The bet paid off: windows grew from 4K to 2M, and "send everything" became viable for most conversations. But the bet could have failed: if windows had plateaued at 32K, context management — summarization, sliding windows, the memory infrastructure that lm15 chose not to build — would have been necessary, and retrofitting it would have been more expensive than building it from the start.

The tool model (Chapter 3) was designed before multi-modal tool results existed. The decision to represent tool results as tuples of Parts was forward-looking — it supported text results today and could support image or audio results tomorrow. When image results arrived (tools returning screenshots, charts, generated images), the representation handled them without changes. But the auto-execution model (8 hops maximum, no external stopping condition) was designed for simple tool tasks and strains under the 20-40 turn agent sessions that are now common.

Each decision was correct for its moment. Each was incomplete for the moment that followed. This isn't a failure of foresight. It's the condition of designing for a domain that's still revealing itself.

### The Measure of a Design

The question isn't "how do I get the design right?" You can't. The domain hasn't finished becoming itself. The content types of next year don't exist yet. The provider APIs of next year haven't been written. The interaction patterns of next year haven't been invented. Any design that claims to be correct for all time is either trivially generic (so abstract it says nothing) or prematurely specific (so detailed it's wrong tomorrow).

The question is: **how do I design so that being wrong is cheap?**

lm15's answers, accumulated across ten chapters:

**Open type sets.** New Part types without structural changes. New finish reasons without enum updates. New event types without Protocol modifications. The type system grows by addition, not by modification. Being wrong about the set of content types costs one new value — not a restructuring.

**Escape hatches.** Features that can't be modeled today are carried in untyped dicts until they can. The hatch is ugly — no autocompletion, no documentation, no type checking. But it's cheaper than the alternative: a typed field that's wrong, a universal parameter that means different things, a structural commitment to a concept that might not generalize.

**Thin contracts.** Frozen dataclasses with fields, not behavioral interfaces with methods. The adapter Protocol has five methods and a set of passive data types. Adding a field to `LMRequest` is backward-compatible (default value). Adding a method to the Protocol is not (existing adapters don't have it). Thin contracts are easier to evolve because there's less to break.

**Simple layers.** 2,408 lines. 30 files. Each layer small enough to rewrite if the assumptions it was built on turn out to be wrong. The Model class is 465 lines. If the conversation model needs to change fundamentally — add context management, change the tool execution loop, restructure history — the rewrite is contained. A 10,000-line Model class would make the same change a project.

**Convergence tracking.** Model what has converged (universal types). Acknowledge what hasn't (escape hatches). The boundary between them moves as providers align. The architecture doesn't need to predict where the boundary will be — it needs to absorb the movement when it happens.

These strategies don't guarantee correctness. They guarantee that correction is possible without rewriting the system. The Part type might need a new field. The conversation model might need context management. The tool execution loop might need a different stopping mechanism. Each of these changes would be local — contained within one module, affecting one layer — because the architecture was designed for local change, not for permanent correctness.

### The Principle

In a stable domain — SQL, HTTP, POSIX — correctness is the measure of a design. The schema should be right. The API should be complete. The type system should capture the domain. Stability rewards precision, and precision is achievable because the domain isn't moving.

In an unstable domain — LLM APIs today, web frameworks in 2010, mobile platforms in 2008 — correctness is temporary. Today's correct design is tomorrow's legacy. The measure of a design isn't whether it's right, but **whether it's easy to change when it's wrong.** Adaptability, not accuracy. The design that survives isn't the one that predicted the future correctly. It's the one that made the future's arrival affordable.

This book has examined ten design problems. Content representation. Conversation modeling. Tool use. Normalization. Adapters. Streaming. Dependencies. Layering. Concealment. And now, time. Each problem had a solution space. Each solution had tradeoffs. lm15 chose one point in each space, and each choice was shaped by the same meta-principle: make wrongness cheap.

The domain will keep changing. The providers will keep evolving. The interaction patterns will keep expanding. New content types, new capabilities, new providers, new pricing models, new deployment contexts — all arriving on a timeline that no library author can predict. The library that survives is the one that designed for the change, not for the moment.

Twelve optional fields. One type discriminator. Five layers. Six escape hatches. Zero dependencies. 2,408 lines.

That's the model. It's incomplete, and it knows it. That's the point.
