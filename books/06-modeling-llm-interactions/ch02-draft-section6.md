## The Bet lm15 Made

lm15's conversation model is five decisions, each of which this chapter has
prepared you to evaluate.

**The library accumulates, the user inspects.** Position two on the ownership
spectrum. The `Model` object appends every turn to `_conversation`
automatically. History is visible (`model.history`), clearable
(`model.history.clear()`), and never modified by the library. No summarization,
no truncation, no sliding window. The developer gets zero-boilerplate continuity
and full visibility into what the model sees.

**Send everything, every time.** No context management. Turn 20 sends all 20
turns. The library bets on large context windows and trusts the developer to
clear history when the conversation outgrows the window. Context overflow is a
`ContextLengthError` — a runtime failure, not a graceful degradation.

**Cache the prefix, not the state.** `prompt_caching=True` reduces the cost of
re-sending everything but doesn't change the semantics. The conversation is
still reconstructed from scratch on every call. The cache is a billing
optimization. The architecture doesn't depend on it.

**Transcript only.** No index, no working memory, no RAG integration. The
conversation is a list of messages. If the developer wants vector search or
scratchpads, they build them as tools — external to the conversation model,
managed by the application.

**Stateless function as the entry point.** `lm15.complete()` creates a `Model`,
calls it once, and discards it. No session, no persistence, no implicit state.
The developer who wants statefulness opts into it explicitly by creating a
`Model` object and keeping a reference to it.

These five choices are a coherent position. They optimize for simplicity,
transparency, and composability — at the cost of requiring the developer to
manage context length, build external memory systems, and absorb quadratic cost
on long conversations. The library is a sharp tool, not a safety net.

What bought this position its viability is the 500x growth in context windows.
In 2022, sending everything would have been reckless — a 10-turn conversation
could overflow a 4K window. In 2025, a 50-turn conversation uses 1M tokens and
fits comfortably in Gemini's 2M window. The bet was that the constraint would
loosen fast enough that managing it wouldn't be worth the abstraction cost. So
far, it has.

Three things would break the bet.

**Context windows plateau.** If windows stop growing and models stabilize at
200K, then 50-turn agent sessions with large documents and tool-call histories
will regularly overflow. The library would need to either manage context or
provide explicit support for the developer to do so — truncation helpers,
token-counting utilities, history summarization hooks.

**Attention quality degrades at scale.** If the "lost in the middle" problem
worsens — if larger windows mean *processing* more tokens without *using* them
more effectively — then sending everything becomes not just expensive but
counterproductive. The model would perform better with a curated 20K-token
context than a verbatim 200K-token context. The library would need to take a
position on context quality, not just context fit.

**The economics shift.** If pricing doesn't drop proportionally with window
growth — if 1M tokens of input remains expensive enough that "send everything"
is viable only for well-funded applications — then the library's implicit advice
("don't worry about context length") becomes misleading. Developers building
cost-sensitive products would need the management tools that lm15 chose not to
build.

None of these has happened yet. All three could. The architecture's defense
against them is its simplicity — adding context management to a system that
currently does none is easier than removing it from a system that has baked it
into every layer. The `Model` object is 465 lines. The conversation is a list.
If the bet fails, the fix is local.

This is the deepest argument for lm15's position: not that "send everything" is
the right answer, but that it's the answer with the lowest cost of being wrong.
