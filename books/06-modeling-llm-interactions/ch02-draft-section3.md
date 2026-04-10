## The Physics of Context

Every conversation strategy in the previous section is shaped by a single
number: the context window. It's the maximum number of tokens the model can
process in one call — input and output combined. Exceed it and the call fails.
Stay within it and everything works. There are no partial overflows, no graceful
degradation, no "the model sees the first 4,096 tokens and ignores the rest."
The window is a cliff, and your conversation either fits on the ledge or falls
off.

This makes the context window the most physics-like constraint in LLM
application design. It's not a guideline or a best practice. It's a hard limit,
like the speed of light in distributed systems or the bandwidth of a network
link. You can't negotiate with it, optimize around it, or pretend it doesn't
apply. You can only design within it or change models.

What makes this constraint unusual — and what makes designing for it treacherous
— is that it's been moving. Fast.

GPT-3 had a 4,096-token context window. A four-turn conversation with detailed
responses could fill it. Any serious application needed aggressive context
management — truncation, summarization, sliding windows. An entire category of
LLM tooling (LangChain's memory classes, LlamaIndex's context management, custom
summarization pipelines) exists because 4K tokens wasn't enough for a real
conversation.

GPT-4 raised it to 8,192, then 32,768, then 128,000. Claude went to 100,000,
then 200,000. Gemini offered 1,000,000, then 2,000,000. In two years, context
windows grew by a factor of 500.

This isn't just a quantitative change. It's a qualitative one — a phase
transition. At 4K tokens, context management is the central engineering problem
of any conversational application. At 200K tokens, a typical conversation never
approaches the limit. At 1M tokens, you can send an entire book as context. The
strategies that were essential at 4K — summarization memories, sliding windows,
RAG over conversation history — become unnecessary overhead at 200K. The
engineering problem didn't get easier. It ceased to exist.

Except it didn't, entirely. Three forces keep context management relevant even
in the age of million-token windows.

**Cost scales with input.** Providers charge per input token. A 200K-token
conversation at Claude's pricing ($3/M input tokens) costs $0.60 per turn just
for the input. Over a 20-turn conversation, the cumulative input cost is $0.60 ×
(1 + 2 + ... + 20) / 20 ≈ $6.30 — and that's before output tokens. At GPT-4.1's
pricing ($2/M), the same conversation is $4.20. These aren't hypothetical
numbers; they're what production agent loops actually cost. The context window
may be large enough, but the budget may not be.

**Attention degrades with length.** Transformers process all tokens, but they
don't attend to all tokens equally well. Research on long-context models
consistently shows that information in the middle of a long context is less
reliably retrieved than information at the beginning or end — the "lost in the
middle" phenomenon. A conversation that fits in the context window is not the
same as a conversation where every turn is equally accessible to the model. At
200K tokens, the model can *process* everything. Whether it can *use* everything
effectively is a different question, and the empirical answer is: not always. A
20-turn conversation with the important instruction on turn 3 may not perform as
well as a 5-turn conversation with the same instruction on turn 1.

**Not all tokens are equal.** A conversation is a temporal structure. Early
turns establish context and identity ("My name is Alice, I'm researching protein
folding"). Middle turns do work ("Here's paper A, compare it to paper B"). Late
turns need the conclusions of the middle turns and the identity of the early
turns, but not necessarily the verbatim text of every middle turn. Sending
everything treats all turns as equally valuable. They're not. A context
management strategy that weights recent and foundational turns more heavily than
verbose middle turns can produce better responses in fewer tokens. The
million-token window makes this optimization unnecessary for *fitting*. It
doesn't make it unnecessary for *quality*.

These three forces — cost, attention quality, and information density — mean
that "just send everything" is a viable strategy, not an optimal one. It works.
It's simple. It's what lm15 does. But at high turn counts or on cost-sensitive
applications, it leaves money on the table and quality on the floor.

The strategic question for library designers is: which side of the bet do you
take? Do you build for the world where context windows keep growing and "send
everything" becomes universally correct? Or do you build for the world where
cost and attention quality keep context management relevant?

lm15 bet on the first world. LangChain bet on the second. Both bets were
reasonable when they were made. lm15's bet is aging better — window growth has
outpaced the need for management. But the cost force hasn't disappeared, and the
attention quality force may get worse before it gets better, because attention
doesn't scale linearly with context length in current architectures.

The honest answer is that nobody knows. Context windows might plateau. Attention
mechanisms might improve. Pricing might drop. A library that bets on one future
and gets the other wrong will need to adapt — and the ease of that adaptation
depends on where it drew the ownership boundary from the previous section. A
library that accumulated history without managing it (lm15) can add management
later. A library that baked management into its core abstraction (LangChain) can
remove it, but the abstraction — the `Memory` class, the `ConversationChain`,
the management pipeline — will carry the scar tissue of a problem that may no
longer exist.

This is a general principle in library design, worth stating explicitly: **bet
on the constraint that's loosening, not the one that's tightening.** Context
windows are loosening. Build for abundant context and optimize for cost, rather
than building for scarce context and optimizing for fit. If the constraint
reverses — if windows shrink, or attention degrades faster than expected —
adding management to a simple system is easier than removing it from a complex
one.
