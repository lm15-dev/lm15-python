# Chapter 10: The Design of Time

Every model name in this book is a lie.

Not today — today they work. `claude-sonnet-4-5` resolves to Anthropic. `gpt-4.1-mini` resolves to OpenAI. `gemini-2.5-flash` resolves to Google. The calls go through, the responses come back, the examples run. But `claude-sonnet-4-5` will be replaced by `claude-sonnet-5-0` or something with a naming scheme nobody has announced yet. `gpt-4.1-mini` will become `gpt-5-mini` or `gpt-5-nano` or whatever OpenAI decides to call the next generation of cheap models. The examples in this book, the examples in lm15's documentation, the hardcoded model names in thousands of applications — all of them are ticking clocks.

This is not the normal pace of API evolution. HTTP 1.1 was published in 1997 and is still the dominant protocol. SQL's core syntax hasn't changed meaningfully since the 1990s. POSIX system calls from 1988 still work on every Unix. These are APIs designed for decades. They change on geological timescales.

LLM APIs change on biological timescales. OpenAI has shipped three fundamentally different API surfaces in three years: text completions, chat completions, and the responses API. Anthropic has revised their Messages API format multiple times — adding tool calling, adding extended thinking, adding cache control, each revision changing the wire format that adapters must produce and consume. Gemini has changed endpoint paths between API versions and restructured content formats. Models are deprecated months after launch. Pricing changes quarterly. Capabilities — what a model can do, what modalities it supports, how it handles tools — shift with each model generation.

The previous nine chapters examined how to model LLM interactions as they exist today: content representation, conversation state, tool use, normalization, adapters, streaming, dependencies, boundaries, concealment. Each chapter treated the current state as a design problem and proposed solutions. This chapter asks a different question: what happens to those solutions when the current state changes?

The answer, honestly, is that they break. And the measure of a design is how cheaply they break — whether a change in the domain requires a local fix or a systemic rewrite.

## Everything Is Temporary

The instability runs deeper than model names. It reaches into every layer of the system.

**Wire formats evolve.** When OpenAI added tool calling, the assistant message gained a `tool_calls` field that didn't exist before. When Anthropic added extended thinking, their response included `thinking` content blocks with a new `redacted` field. When Gemini added grounding, their response included citation metadata with a structure unlike either OpenAI's or Anthropic's. Each addition is a wire format change that requires adapter updates — new fields to parse, new response structures to handle, new edge cases to test.

**Capabilities expand.** Two years ago, no model supported tool calling. One year ago, no model supported extended thinking. Six months ago, no model returned citations as structured content. Each new capability is a new Part type, a new event in the streaming protocol, a new parameter on Config, and potentially a new method on the adapter Protocol. The capability set hasn't stabilized — it's growing, and each addition ripples through the type system.

**Providers appear and disappear.** Mistral, Cohere, DeepSeek, Grok — each new provider is a new adapter, a new set of wire format quirks, a new entry in the capability matrix. Some of these providers will thrive. Some will be acquired. Some will shut down. The adapter written for a provider that shuts down doesn't crash — it just stops being useful. But the adapter written for a provider that changes its API without backward compatibility does crash, and the library must respond.

**Pricing shifts.** The cost assumptions from Chapter 2 (quadratic conversation costs) and Chapter 8 (the economics of agent loops) are based on today's pricing. Prices have dropped 10x in two years. If they drop another 10x, some of the cost-optimization strategies — prompt caching, model routing, token budgets — become irrelevant. If they stabilize or increase, they become more important. The library's design doesn't depend on specific prices, but the user's architectural decisions do, and the library's guidance (when to cache, when to route, when to budget) is implicitly tuned for a price range that's moving.

The common thread: **the domain is being invented in real time.** The library is modeling something that hasn't finished becoming itself. Software design is usually about capturing what IS — the database schema reflects the data, the API reflects the operations, the type system reflects the domain. LLM library design is about capturing what's BECOMING — the types must absorb new content types, the adapters must absorb new providers, the parameters must absorb new capabilities, and the architecture must absorb all of this without requiring the user to rewrite their code.

This is the design problem of this chapter: not what the library should look like, but how it should age.
