# Chapter 5: The Adapter Pattern

The previous chapter established that normalization distorts — that a universal
API over three provider APIs necessarily introduces gaps between what the
parameter promises and what the provider delivers. This chapter is about the
entity that lives in those gaps: the adapter.

An adapter has one job. It receives a request in the library's universal format,
translates it into the provider's wire format, sends it, receives the response
in the provider's format, and translates it back. `LMRequest` in, HTTP call in
the middle, `LMResponse` out. The adapter is a translator, and like all
translators, its quality is determined by how much meaning survives the
translation.

But "translator" understates the design problem. A human translator between
French and English makes judgment calls — word choices, idiom substitutions,
cultural adaptations — that preserve meaning at the cost of literal accuracy. An
LLM adapter makes the same kind of judgment calls: should `system=` become a
message with role `"system"` (OpenAI) or a top-level field (Anthropic) or a
`systemInstruction` object (Gemini)? Each choice is a translation decision that
preserves the intent while changing the form. The adapter pattern is about
structuring these decisions so they're consistent, extensible, and testable.

This chapter examines four questions about that structure: how you define the
adapter contract, how you decide the adapter's scope, what the adapter loses in
translation, and how other libraries answered the same questions.

## Structuring the Contract

Two decisions shape the adapter before you write a line of translation code: how
you define what an adapter must be, and how many adapters you need.

**Protocol vs inheritance.** The conventional approach is an abstract base class
— `class BaseAdapter` with abstract methods. Each provider subclasses it. The
type system enforces the contract, and IDE support is excellent: subclass the
base, fill in the stubs, done.

lm15 uses a `Protocol` instead — structural typing. The adapter doesn't inherit
from anything. It doesn't import the protocol class. It just needs to have the
right methods and attributes. `UniversalLM.register()` accepts anything that
satisfies the Protocol, verified by the type checker or at runtime.

The benefit is decoupling. A third-party adapter imports lm15's types
(`LMRequest`, `LMResponse`) but nothing else. It doesn't inherit from a base
class that might change between versions. The contract is structural: have these
methods, accept these types, return these types.

The cost is discoverability. No base class to subclass, no stubs to fill in. The
developer reads the Protocol definition and builds from scratch. And Python's
Protocols are checked by type checkers, not by the runtime — a missing method
produces `AttributeError` at call time, not a compile-time error. lm15 mitigates
this with `EndpointSupport`, a dataclass where the adapter declares which
methods it implements. The client checks the declaration before calling.

The choice is language-shaped and philosophy-shaped. In Java, you'd use an
interface. In Rust, a trait. In Go, an implicit interface. In Python, both
options exist, and the choice signals intent: inheritance says "extend me"
(framework); Protocol says "match me" (library). lm15 is a library.

**One per provider vs one per model.** lm15 has three adapters. Each handles all
models from its provider. `OpenAIAdapter` handles `gpt-4.1-mini`, `o1-preview`,
`dall-e-3`, and every other OpenAI model. Model-specific behavior — different
default max_tokens, support for reasoning, vision capabilities — is handled with
conditionals inside the adapter.

The alternative — one adapter per model — is cleaner in theory. Each adapter is
small, focused, free of conditionals. In practice, OpenAI lists over 40 models
that share 95% of their code. The 5% that differs doesn't justify 40 classes.
The per-provider approach handles the 5% with ugly but contained conditionals:
`if model.startswith("o1"): body.pop("temperature", None)`.

The practical heuristic: **one adapter per authentication boundary.** If all
models share the same API key, base URL, and auth header, they share an adapter.

**Translation vs delegation.** The adapter can talk to the provider's HTTP API
directly (translation — build JSON, send via urllib, parse response) or delegate
to the provider's SDK (`openai.chat.completions.create(...)`). Translation is
more work but avoids SDK dependencies. Delegation is easier but inherits the
SDK's dependency tree — `openai` brings `httpx`, `pydantic`, `anyio`, and more.
For a zero-dependency library, delegation is off the table. But the choice
exists for libraries with different priorities, and an interesting intermediate
position is **optional delegation** — raw HTTP by default, SDK if installed.
