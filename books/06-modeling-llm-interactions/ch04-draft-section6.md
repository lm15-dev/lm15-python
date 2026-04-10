## Is a Universal LLM API Possible?

The question has haunted this chapter, and the honest answer is: it depends on
what you mean by "universal."

If universal means *identical behavior* — the same parameter producing the same
mechanism, the same cost, the same latency, the same edge-case behavior on every
provider — then no. A universal LLM API is not possible. The providers differ in
real ways that aren't syntactic. Anthropic's cache control is a fundamentally
different mechanism from OpenAI's automatic prefix caching. Claude's extended
thinking with a token budget is a fundamentally different concept from GPT's
reasoning effort levels. These aren't naming differences that a translation
layer can resolve. They're conceptual differences that reflect different
engineering decisions, different cost structures, and different product
philosophies. No amount of normalization can make them the same, because they
aren't the same.

If universal means *identical intent* — the same parameter expressing the same
high-level goal (make it cheaper, make it think harder, generate an image), with
each provider fulfilling the goal through its own mechanism — then yes,
approximately, and improving. The intent-level normalization works today for
text, tools, streaming, reasoning, caching, and multimodal input. The developer
who operates at the intent level — "I want the model to think before answering"
rather than "I want to set `budget_tokens` to 10,000" — gets genuine portability
across providers.

If universal means *a surface that absorbs the future* — an API design that
won't need fundamental changes when new providers arrive, new modalities emerge,
or new interaction patterns develop — then it's a bet, not a fact. The bet is on
convergence. The evidence for convergence is strong: all three major providers
independently arrived at the same content-block pattern, the same tool-calling
protocol, the same streaming structure. New providers (Mistral, Cohere,
DeepSeek) adopt the same shapes. The architectural pattern — ordered array of
typed parts, tool calls as structured requests, streaming as server-sent events
— appears to be a natural fit for the problem, the way REST is a natural fit for
web APIs or SQL is a natural fit for relational queries. If this convergence
holds, a universal API built around it will become more accurate over time, not
less.

But convergence has limits. Each provider has features that the others don't and
may never adopt. Anthropic's prefill. OpenAI's response format constraints.
Gemini's grounding with Google Search. A universal API that tries to absorb
every provider-specific feature ceases to be universal — it becomes the union of
all providers, which is the opposite of a shared abstraction. The escape hatch
exists precisely for these features — to carry them without pretending they're
universal.

The practical answer, the one that determines how you design a library, is this:
**a universal LLM API is not a destination but a trajectory.** It becomes more
universal as providers converge, and it remains non-universal where they
diverge. The library's job isn't to achieve universality — that's impossible.
It's to track the convergence, normalizing what has converged, acknowledging
what hasn't, and carrying the rest in escape hatches that empty over time as the
convergence continues.

A library that claims full universality today is lying — the providers differ in
ways the library can't hide. A library that claims no universality is
unnecessary — the providers agree on more than they disagree. The useful library
is the one that knows which is which and tells you honestly.

The map is getting more accurate. The territories are becoming more similar. But
the map is not the territory, and the gap between them — the distortion zone
where normalization promises what it can't deliver — is where the interesting
design decisions live. This chapter has mapped that zone. The next chapter
examines the translation layer itself — the adapter — and how you structure a
system that must live in the gap between universal types and provider-specific
wire formats.
