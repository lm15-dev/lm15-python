# Chapter 4: The Normalization Question

Every map distorts. The Mercator projection preserves direction — a straight
line on the map corresponds to a constant compass bearing, which is why sailors
used it for centuries. But it distorts area. Greenland appears the size of
Africa. In reality, Africa is fourteen times larger. The distortion isn't a bug
in the Mercator projection. It's a mathematical certainty: you cannot flatten a
sphere onto a plane without distorting something. The only question is what you
choose to distort and whether the people reading the map know about the
distortion.

A multi-provider LLM library is a map of three territories — OpenAI, Anthropic,
Gemini — projected onto one surface. The user reads one API. Underneath, three
APIs exist, with different JSON structures, different authentication schemes,
different streaming protocols, different tool-call encodings, different caching
mechanisms, and different error formats. The library's job is to present one
clean surface. The question is what distortions that surface introduces, and
whether the distortions are benign or dangerous.

This is the normalization question, and every multi-provider library answers it
— usually without stating the answer explicitly. This chapter makes the answers
explicit.

## The Map and the Territories

There are four levels of normalization, each accepting a different amount of
distortion.

**Pass-through.** No normalization. Each provider has its own types, its own
methods, its own error handling in the library. The user writes
provider-specific code. This is what the raw SDKs give you —
`openai.chat.completions.create(...)`, `anthropic.messages.create(...)`,
`genai.GenerativeModel(...).generate_content(...)`. Three surfaces for three
territories. No distortion, because there's no projection. The user navigates
each territory directly.

The cost is obvious: every piece of application logic that touches the LLM must
be written three times, or the application must commit to one provider and
accept the lock-in. Pass-through is honest but impractical for applications that
need flexibility.

**Structural normalization.** One set of types — `LMRequest`, `LMResponse`,
`Part`, `Message` — that work across all providers. The types are universal; the
behavior underneath may differ. `prompt_caching=True` exists as a parameter on
all providers, but it does different things on each (explicit markers on
Anthropic, a no-op on OpenAI, a CachedContent resource on Gemini). The structure
is normalized. The semantics are not.

This is where lm15 sits. The user writes one set of code, and it works across
providers — with the caveat that "works" means "produces a response using the
provider's native mechanism," not "produces identical behavior." The distortion
is contained but real: the user who expects `prompt_caching=True` to mean the
same thing on every provider will be surprised.

**Behavioral normalization.** One set of types *and* identical behavior.
`prompt_caching=True` would not only exist on all providers but would behave
identically — same cost reduction, same lifetime, same granularity. This is the
promise of libraries like LiteLLM: "use the OpenAI API, we'll make it work on
every provider."

The promise is stronger, and the lie is deeper when it breaks. Behavioral
normalization works beautifully for features that genuinely behave the same
(text completion, token counting). It's a fiction for features that don't
(caching, reasoning, image generation). The fiction helps the user who doesn't
care about the differences. It traps the user who does.

**Lowest common denominator.** Only expose features that all providers support
identically. No caching (OpenAI's is automatic, Anthropic's is explicit, so
neither is universal). No reasoning (parameterized differently). No image
generation (only two of three providers support it). What remains: text
completion, basic tool calling, streaming, error handling. Roughly 40% of what
modern models can do.

This is the most honest normalization — everything it exposes is genuinely
universal. It's also the most limiting. A library that restricts itself to the
LCD is a library that can't use half the capabilities that make the underlying
models valuable. The honesty comes at the cost of usefulness.

### The Projection Choice

These four levels are not a quality ranking. They're projection choices — each
preserves some property and distorts others.

Pass-through preserves accuracy and distorts usability. LCD preserves honesty
and distorts capability. Structural normalization preserves capability and
distorts behavioral equivalence. Behavioral normalization preserves the
*appearance* of equivalence and distorts truth.

Every multi-provider library sits somewhere on this spectrum, whether its author
articulated the choice or not. LiteLLM sits between structural and behavioral —
OpenAI-shaped types with per-provider translation that mostly works. LangChain
sits between pass-through and structural — provider-specific integrations
wrapped in a common interface that doesn't guarantee behavioral equivalence.
lm15 sits at structural normalization — universal types, provider-specific
behavior, escape hatches for the differences.

The user rarely knows which projection they're reading. They see
`prompt_caching=True` and assume behavioral equivalence, the way a map reader
sees Greenland and assumes it's the size of Africa. The library's documentation
usually doesn't say "this parameter means different things on different
providers." The distortion is silent. This chapter is about making it audible.
