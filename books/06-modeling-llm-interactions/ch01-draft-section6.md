## The Representation Bet

lm15's message representation is five specific choices. Each one is a point in the tradeoff space this chapter has mapped.

**One Part type, not a hierarchy.** A single frozen dataclass with a `type` discriminator and twelve optional fields. The opening section examined this in detail. It buys: one serialization path, one iteration pattern, graceful handling of unknown types. It costs: weak type safety, optional fields that are meaningless for most variants, no compiler enforcement of which fields are valid for which types. The choice is a bet that extensibility and portability matter more than static type guarantees — a bet shaped by Python's type system, which doesn't offer discriminated unions.

**Ordered tuple of parts, not a flat structure.** `Message.parts` is a `tuple[Part, ...]`, not separate `text`, `tool_calls`, and `thinking` fields. This preserves co-occurrence (which parts appeared together) and ordering (which came first). It costs convenience — accessing the text requires filtering `message.parts` by type, which is why `resp.text` and `resp.tool_calls` exist as computed properties. The underlying tuple carries the structure; the properties provide the shortcuts.

**Frozen dataclasses, not mutable objects.** Every type is `@dataclass(frozen=True)`. You build a new `Part`, you don't modify an existing one. This prevents action-at-a-distance bugs (two references to the same Part, one modifies it, the other is silently corrupted), enables hashing for caching, and provides thread safety. It costs verbosity — constructing a modified copy requires spelling out every field. lm15's factory methods (`Part.text_part()`, `Part.image()`) absorb some of this verbosity.

**DataSource for media, not inline bytes.** Images, audio, video, and documents carry their content in a `DataSource` sub-object that can hold a URL, base64-encoded bytes, or a file ID. This level of indirection means a Part doesn't need to know whether its image is a URL, a local file, or an uploaded reference — the DataSource handles the variants. The cost is one more type to understand; the benefit is that media parts are uniform regardless of how the content is sourced.

**Escape hatch, not universal coverage.** Provider-specific information lives in `part.metadata`, `config.provider`, and `resp.provider` — untyped dicts that carry whatever the universal type can't express. This is an explicit design choice, not an omission. The alternative — adding fields to `Part` for every provider-specific concept — would grow the type indefinitely and make it provider-shaped rather than universal. The escape hatch acknowledges that full normalization is impossible and provides a clean channel for the un-normalizable.

### What Would Change the Bet

Three developments would challenge this representation.

**Python gains sum types.** If a future Python version adds algebraic data types with exhaustive pattern matching — similar to Rust's `enum` or TypeScript's discriminated unions — the argument for a single Part class weakens considerably. Sum types would give you the discriminator pattern's extensibility *and* the hierarchy's type safety. lm15 would likely migrate to them, because the current design is explicitly a workaround for Python's type system limitations.

**The content taxonomy stabilizes.** If, in three years, no new content types have emerged — if text, images, audio, video, documents, tool calls, thinking, citations, and refusals are the final set — then a closed hierarchy becomes viable. The extensibility benefit of the discriminator pattern is only valuable when the type set is growing. A static set favors explicit types. But there is no evidence that the taxonomy is stabilizing. Every six months brings a new content type.

**A provider introduces genuinely alien content.** Interactive widgets. 3D objects. Executable code blocks. Live data streams. Content types that don't fit the `Part` pattern — that can't be expressed as a type string plus optional fields, because they require structure that no existing field can carry. The escape hatch (`metadata`) could handle these, but at that point the universal type is carrying the alien content as an opaque blob, and the universality is nominal rather than real. If this happens often enough, the Part type would need either new fields (growing toward the universe of all possible content) or a redesigned extensibility mechanism (plugin-registered part types with custom fields).

### The Deeper Bet

Underneath the five technical choices is a philosophical one. lm15 bets that the structure of LLM messages is *convergent* — that all providers are moving toward the same shape (ordered arrays of typed parts), and that a universal representation built around that convergence will remain valid as the providers evolve.

The evidence for convergence is strong. OpenAI, Anthropic, and Gemini independently arrived at the same content-block pattern. New providers (Mistral, Cohere, DeepSeek) adopt the same shape. The structure appears to be a natural fit for the problem — the way that relational tables are a natural fit for structured data, or key-value pairs are a natural fit for configuration. If this convergence holds, the universal-parts representation is not just adequate but *correct* — it mirrors the natural structure of the domain.

The evidence against is the pace of change. Two years ago, tool calls didn't exist. One year ago, thinking blocks didn't exist. Six months ago, audio content blocks didn't exist. The convergence is real, but the converged shape keeps expanding. The container is stable; the contents keep growing. Whether the container can absorb every future content type — or whether some future capability will require a fundamentally different container — is the question that the bet rides on.

Every representation is a bet on the future. The safest bet is a container that absorbs new contents without changing its shape. That's what lm15 built. So far, it's held. But the future has a record of surprising the people who bet on it, and the only defense against surprise is a design simple enough to change when the surprise arrives.

Twelve optional fields. One type discriminator. A frozen tuple. An escape hatch. That's the representation. The next chapter asks a different question: now that we know what a message is, who holds the conversation?
