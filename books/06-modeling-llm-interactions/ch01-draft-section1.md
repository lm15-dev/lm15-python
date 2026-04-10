# Chapter 1: What Is a Message?

Here is a single response from Claude, returned from one API call:

```
[thinking] The user uploaded a photo of a restaurant receipt and asked
whether the tip was reasonable. Let me look at the total and the tip
line. The subtotal is $84.50, tax is $7.61, total is $92.11, and the
tip written in is $12.00. That's about 13%, which is below the
customary 18-20% in the US.

[text] The tip on this receipt is $12.00 on a pre-tax subtotal of
$84.50, which comes to about 14.2%. In the US, 18-20% is customary
for table service. A 20% tip would have been $16.90.

[tool_call] calculate(expression="12.00 / 84.50 * 100")
→ 14.201183...

[tool_call] calculate(expression="84.50 * 0.20")
→ 16.90
```

This arrived as a single message — one object, from one function call. Inside it: a thinking trace the model produced before composing its visible answer, a text response meant for the user, and two tool calls that were executed during generation to verify the arithmetic. Upstream, attached to the request that prompted this response, was a photograph of a restaurant receipt — an image the model examined to extract the numbers it then reasoned about.

What is the type of this thing?

The question seems pedestrian — a data modeling exercise, the kind of thing you sketch on a whiteboard in the first hour of a project. But I want to argue that it's something more fundamental. The question "what is a message?" is really the question "how do we impose discrete structure on a continuous, probabilistic process?" — and the answer you give determines the shape of everything you build on top.

## The Nature of the Territory

To understand why representing messages is hard, you need to understand what a
language model's output actually is — not at the API level, where it's already
been packaged into JSON, but at the level where it's produced.

A language model generates a sequence of tokens. Each token is selected from a
probability distribution over the entire vocabulary — typically 50,000 to
150,000 entries. The model doesn't produce "text" and "tool calls" and
"thinking" as separate streams. It produces one stream of tokens. Some of those
tokens, when decoded, form natural language. Some form structured JSON (the
arguments to a tool call). Some are special delimiter tokens that the model
learned during training to emit when transitioning between "thinking" and
"answering." But to the model, these are all the same kind of object: entries in
a vocabulary, selected by the same attention mechanism, embedded in the same
continuous space.

The model assigns a probability to every token it generates, and that
probability is information — it tells you how confident the model was, how many
alternatives it considered, how close the second-best token was to the winner.
When you read `resp.text` and get a clean string, that entire probability
landscape is gone. The representation didn't just *organize* the output; it
*discarded* most of it.

This is the first thing to internalize: every message representation is an act
of lossy compression. The territory — the model's actual output — is a rich
probabilistic object. The map — your message type — is a discrete structure with
named fields and clean boundaries. The gap between them is information loss, and
the design question is what to lose.

Consider the thinking trace in our specimen. The model generated it as part of
one token stream — mechanically identical to the visible response, distinguished
only by special delimiter tokens that mark it as "thinking." When your
representation separates thinking from text into different parts, it gains
something valuable: the application can show thinking in a collapsible panel,
omit it entirely, or count its tokens separately. But it also loses the temporal
interleaving. The model didn't first think and then respond in two clean phases.
It thought *while* composing, and the boundary between thinking and answering
was not a wall but a gradient. The representation imposed a boundary that the
output didn't inherently have.

Or consider the tool calls. In the token stream, a tool call is a sequence of
tokens that happen to form valid JSON — `{"name": "calculate", "arguments":
"..."}` — emitted at a specific point in the generation where the model decided
it needed to verify a number. When the representation extracts this as a
structured `Part(type="tool_call", name="calculate", input={...})`, it preserves
the structure but discards the position. The tool call appeared *between* the
thinking trace and the text response, at a precise moment in the model's
reasoning. That position carries information — it tells you the model verified
its arithmetic *before* writing the answer, not after. Most representations
don't preserve it. Most applications don't need it. The loss is real; the loss
is usually acceptable. But it was a choice, and most library authors didn't
realize they were making it.

This is the game every representation plays. You decide which dimensions of the
territory to preserve — type identity, structure, content — and which to discard
— temporal position, token-level probabilities, the model's uncertainty. The
decisions feel obvious in the moment: of course you separate thinking from text,
of course you extract tool calls as structured objects. "Obvious" is the most
dangerous word in design. It means the choice was made unconsciously, and
unconscious choices are the ones that haunt you.

## The Essential Difficulty

Fred Brooks distinguished *essential* difficulties — problems inherent in the
nature of the thing being built — from *accidental* difficulties — problems
arising from the tools and processes used to build it. The representation
problem has both, and confusing them leads to bad designs.

The essential difficulty is this: the model's output is continuous; the
application's types are discrete. The output has no inherent boundaries between
"thinking" and "text." It has no inherent structure called "a tool call" — there
are tokens that form JSON, and a convention that certain JSON shapes constitute
tool requests. The model doesn't know it's producing "parts." It's producing
tokens. The decomposition into parts is something we impose from outside, and
any decomposition we choose will be wrong in some way — will cut the output at a
boundary that doesn't naturally exist, or will merge things that the application
would have benefited from separating.

This difficulty doesn't go away with better APIs or better type systems. It's in
the nature of the thing. As long as we're bridging a probabilistic generator and
a deterministic type system, we'll be making lossy choices about where to draw
boundaries.

The accidental difficulties are the ones that arise from the current state of
provider APIs. OpenAI encodes tool calls as a separate field on the message
object. Anthropic encodes them as content blocks in the content array. Gemini
encodes them as `functionCall` parts. These are three accidental solutions to
the same essential problem, and a multi-provider library must translate between
them. This translation is tedious, but it's not deep — it's plumbing, and it'll
get simpler as providers converge (which they are, slowly).

The distinction matters because it tells you where to spend your design effort.
Accidental difficulties are solved by engineering — better adapters, more
complete translations, broader test suites. Essential difficulties are solved by
*making good choices about what to lose*. The representation chapter of your
library is where the essential choices live, and it deserves more thought than
the adapter chapter.

## What Representations Preserve and Destroy

Let me be specific about what's lost and what's kept, because the abstract claim
"every representation is lossy" is true but unhelpful without an inventory.

**Preserved by all reasonable representations:**
- *Type identity.* This is text. That is a tool call. That is an image. The
  categorical labels survive.
- *Content.* The actual text string. The tool call's name and arguments. The
  image's bytes or URL. The payload survives.
- *Role.* Who produced this message — the user, the model, or a tool execution.
  The conversational role survives.

**Preserved by some representations, lost by others:**
- *Part ordering.* The thinking came before the text, which came before the tool
  call. An ordered container (array/tuple of parts) preserves this. A flat
  structure with `text`, `thinking`, and `tool_calls` as separate fields
  destroys it — you know all three exist, but not in what order they appeared.
- *Co-occurrence.* This thinking trace and this text appeared in the same
  message. A parts-in-message structure preserves this. A normalized database
  schema with separate tables for text, thinking, and tool calls might not.
- *Part boundaries within a type.* Did the model produce one long text block or
  two short ones separated by a tool call? A representation that joins all text
  parts into one string destroys this boundary. lm15's `resp.text` property
  joins them (convenient for display), but the underlying `message.parts`
  preserves them (available for inspection).

**Lost by all standard representations:**
- *Token-level probabilities.* The model's confidence in each token. Some APIs
  offer this through `logprobs`, but no message representation preserves it as
  part of the content model. It's metadata, requested separately, attached as an
  afterthought.
- *Generation order at the sub-part level.* Within a text block, you don't know
  which words the model generated first (in practice, left to right, but with
  speculative decoding this isn't always true). Within a tool call's arguments,
  you don't know the order the fields were generated.
- *Alternative generations.* The model considered other tokens at every step.
  The representation shows only the winners.
- *The model's internal state.* Attention patterns, activation values, embedding
  representations. These exist and carry information, but no API exposes them
  and no representation models them.

This inventory matters because it reveals the design space. A representation
that preserves part ordering is strictly more expressive than one that doesn't —
it can express everything the orderless representation can, plus temporal
relationships. That doesn't mean it's always better — the extra expressiveness
costs complexity — but it means the choice is real.

## The Portability Constraint

If messages only traveled one direction — from provider to application — the
representation problem would be easier. You'd model whatever the provider sends,
in whatever structure is convenient, and be done. But messages travel sideways.
They travel between providers.

This is the defining constraint of a multi-provider library. The message
representation must be universal — not OpenAI-shaped, not Anthropic-shaped. A
format that any provider's adapter can produce and any provider's adapter can
consume.

Consider a concrete pipeline: generate an image on OpenAI, then send that image
to Claude for analysis. The image arrives from OpenAI's adapter as some
representation of image data. It must then enter Anthropic's adapter as input.
If the image is an OpenAI-specific object — carrying OpenAI's URL format,
OpenAI's `detail` parameter, OpenAI's response structure — Anthropic's adapter
can't consume it without a conversion layer. If the image is a universal Part
with a DataSource that holds raw bytes or a URL, both adapters speak the same
language.

But universality leaks. OpenAI's image content has a `detail` field (`"low"`,
`"high"`, `"auto"`) that controls how many tokens the image consumes in the
context window. Anthropic has no equivalent. When a universal Part carries a
`detail` field, it carries information meaningful to one provider and
meaningless to another. Is that field part of the universal representation, or a
provider-specific annotation?

There is no clean answer. Include `detail` in the universal type and every image
part carries a field relevant only to OpenAI — a small lie about universality.
Exclude it and OpenAI users lose a feature. Relegate it to an untyped metadata
dict and it's accessible but invisible to documentation and autocompletion. lm15
puts it on `DataSource` — the sub-object describing the media data — accepting
the leak but containing it. The field is there. It's documented. It does nothing
on Anthropic or Gemini. This is an honest compromise, the kind that universal
representations require constantly.

The deeper issue is that universality is aspirational, never achieved. Every
"universal" representation is universal with respect to the providers that
existed when it was designed. A representation built for today's three providers
will struggle when a provider arrives with a genuinely novel content type —
interactive widgets, 3D objects, executable code blocks, real-time sensor data.
The representation either adds a new part type (easy if the type set is open,
hard if it's closed), or shoves the content into a generic container (losing
structure), or simply can't represent it (losing the provider).

This reveals a connection that isn't immediately obvious: **extensibility and
portability are the same property viewed from different directions.**
Extensibility asks: "can I add a new content type without changing the
structure?" Portability asks: "can I support a new provider without changing the
structure?" Both are asking whether the representation is open or closed to new
variants. An open representation — an ordered container of typed parts where the
type set can grow — is both extensible *and* portable. A closed representation —
a fixed set of classes or fields — is neither.

This is perhaps the single most important insight about message representation:
the choice between open and closed type sets isn't a matter of taste. It's a
choice between a representation that can absorb the future and one that breaks
on contact with it.

## The Shape Radiates

Four consequences flow from the representation you choose, and they pull in
different directions.

**Composability.** Can a thinking trace appear next to a tool call in one
message? Can an image appear in a system prompt? A flat structure with `text`,
`tool_calls`, and `thinking` as separate top-level fields implies these are
independent dimensions. But they're not — they co-occur in specific orders
within a single message, and that order carries meaning. A parts container
preserves co-occurrence and ordering. A flat structure destroys both, and every
downstream piece of code must reconstruct what was lost, or work without it.

**Adapter complexity.** Every adapter translates between the universal type and
a provider's wire format. OpenAI, Anthropic, and Gemini all use some variant of
an ordered array of typed content blocks. The names differ, the type strings
differ, the sub-structures differ — but the shape is the same: an ordered
sequence, a type discriminator per element, type-specific payload fields. A
universal representation that mirrors this shape — an ordered sequence of typed
parts — translates mechanically to all three. One that diverges forces every
adapter to decompose and reconstruct, losing information at each boundary.

**Extensibility.** Content types are arriving faster than library versions.
Refusals, citations, audio parts, video frames — each one is a new entry in the
content taxonomy. A closed representation (a fixed set of classes) requires
surgery for each addition: new class, new serializer, new case in every switch
statement across every adapter. An open representation (a discriminator field
accepting new values) absorbs additions without structural changes. The question
is whether the taxonomy of content types is complete. The answer, empirically,
is: it has never been complete. Every year brings types that didn't exist the
year before.

**Cognition.** This is the force that's easiest to forget and hardest to
measure. A type isn't just a data structure — it's a mental model. When a
developer works with your library, they think in the shapes your types provide.
Twelve subclasses create twelve mental categories. One Part type with a
discriminator creates one: "a message contains parts, each part has a type."
Simpler shapes produce clearer thinking, up to the point where simplicity hides
distinctions the developer needs to make. A single Part type makes it easy to
iterate and switch on `part.type`. It also makes it easy to write code that
ignores types it doesn't recognize — a quiet form of extensibility that class
hierarchies don't support without a catch-all base case.

Types aren't the only way to carry structure. A plain dict — `{"type": "text", "text": "..."}` — is also open,
extensible, and portable. It's what the JSON wire format actually is, before the
library reconstructs it as a typed object. The Part type adds `__post_init__`
validation, IDE autocompletion, and a named class to talk about. It takes away
the radical flexibility of "any key, any value." Whether the typed approach is
worth the cost depends on the ecosystem. In Python, where classes are expected
and dicts lack autocompletion, the tradeoff favors types. In Clojure, where data
orientation is the norm, it would favor maps. In TypeScript, discriminated
unions give you both — open-ended types with compiler-checked exhaustiveness.
The representation problem is universal; the solution is language-shaped.

These four forces pull in different directions. A class hierarchy is excellent
for cognition (clear categories, one class per concept) but terrible for
extensibility (each new concept is a new class, a new import, a new case in
every handler). A flat dict is excellent for extensibility (add any key) but
terrible for cognition (what keys exist? what are their types? what combinations
are valid?). A discriminated union — whether implemented as a single class with
a type field, a TypeScript union, or a Rust enum — is a compromise that balances
all four forces at the cost of being fully optimal for none.

Liskov's objection should be named here too: the single Part type makes all
parts nominally substitutable — any function that accepts a Part accepts both
text parts and image parts — but *semantically* they're not substitutable. Code
that reads `part.text` will get `None` on an image part. Code that reads
`part.source` will get `None` on a text part. The type system permits operations
that are meaningless for a given variant. Python's `match` statement and
TypeScript's narrowing can recover some type safety at the point of use — `match
part.type: case "text": print(part.text)` — but the Part type itself doesn't
enforce which fields are valid for which discriminator values. This is a real
cost. The `__post_init__` validation catches some mistakes at construction time,
but it can't catch code that accesses the wrong field on a correctly-constructed
Part.

Is there a design that preserves both safety and openness? Perhaps. A
protocol-based approach — `TextPart`, `ImagePart`, `ToolCallPart` as separate
classes, all satisfying a common `Part` protocol — would give you type-specific
fields *and* a common iteration interface. Python's structural typing
(`Protocol`) supports this. The cost is more classes, more imports, and adapters
that must use isinstance checks or pattern matching instead of field-based
switching. lm15 decided this cost outweighed the benefit for a library with
three adapters and growing content types. A library with fifty adapters — or a
language with better sum types — might decide differently.

The rest of this chapter examines four specific approaches to the representation
problem, from the simplest to the most flexible, and maps where each one sits in
this tradeoff space. But the tradeoffs are already visible. The question is what
you're optimizing for — type safety, portability, extensibility, cognitive
simplicity — and how fast the territory is changing.

Right now, the territory is changing very fast.
