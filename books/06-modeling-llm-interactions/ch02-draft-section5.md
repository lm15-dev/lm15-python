## Transcript and Index

There are two fundamentally different things people mean when they say "the
model should remember."

The first is continuity. "I told you my name five minutes ago, you should still
know it." This is short-range, session-scoped, and complete. The model should
have access to everything said in this conversation, in order, verbatim. Nothing
should be lost, paraphrased, or reranked. The user expects a transcript — a
faithful record that the model reads sequentially, the way a human rereading a
chat log would.

The second is knowledge. "I uploaded a paper last week, you should know what's
in it." This is long-range, cross-session, and selective. The model should have
access to relevant information from past interactions, documents, and external
sources — but not necessarily all of it, not necessarily verbatim, and not
necessarily in temporal order. The user expects an index — a searchable store of
facts, organized by relevance, retrieved on demand.

These two kinds of memory are often conflated, and the conflation leads to
architectures that serve neither well. A system that treats knowledge as a
transcript — stuffing documents into conversation history — wastes context
window space on verbatim text that only needs to be consulted selectively. A
system that treats continuity as an index — retrieving "relevant" turns from the
current conversation — loses the temporal structure that gives conversation its
coherence.

The distinction maps roughly onto what cognitive scientists call episodic and
semantic memory. Episodic memory is autobiographical — "I had coffee at that
café on Tuesday." It's temporal, sequential, contextualized by when and where it
happened. Semantic memory is factual — "Paris is the capital of France." It's
decontextualized, organized by meaning, accessed by association rather than by
timestamp. Humans use both, constantly, and the systems that access them are
different. You don't retrieve your knowledge of French geography by replaying
every conversation you've ever had about France. You access a fact — a
compressed, decontextualized piece of knowledge — and deploy it.

Current LLM architectures rely almost entirely on the episodic model.
Conversation history is a transcript — a complete, ordered recording of
everything that was said, replayed in full on every call. This works because
transformer attention can process the transcript and extract relevant
information from it in context. The model doesn't need a separate semantic
memory because it can perform semantic retrieval *over the transcript itself*,
at inference time. If the answer to "what's my name?" is in the transcript at
turn 3, the attention mechanism will find it.

But this only works when the transcript fits in the context window and when the
attention mechanism can effectively search it. At small scales — a 10-turn
conversation, a 5-page document — in-context retrieval is fast, cheap, and
reliable. At larger scales, the three forces from the previous section reassert
themselves: cost (processing the full transcript on every turn), attention
quality (information in the middle is attended to less effectively), and
information density (most of the transcript is irrelevant to any given
question).

This is where the index model becomes necessary. Instead of sending the full
transcript, you maintain a searchable store — a vector database, a keyword
index, a structured knowledge graph — and retrieve relevant fragments on each
turn. The model sees a small, focused context: the system prompt, the retrieved
fragments, and the current question. The full transcript lives outside the
model's input, consulted but not transmitted.

The practical dividing line is roughly this: if the memory is about *this
conversation*, use a transcript. If the memory is about *everything the system
knows*, use an index. A customer support agent that's been talking to a user for
10 turns needs a transcript — the user expects continuity, and the temporal
structure ("I already tried restarting" → "let's try something else") is
load-bearing. A research assistant that's analyzed 200 papers over three months
needs an index — no single conversation can hold 200 papers, and the user's
question on any given turn is relevant to maybe 3 of them.

But the interesting cases are in the middle. A coding agent on turn 40 of a
refactoring session needs both — transcript-level continuity ("I already
refactored auth.py, now I'm working on views.py") and index-level knowledge
("the codebase has 200 files, here are the ones that import from auth"). The
agent can't send the full transcript (too expensive, too noisy) and can't rely
on retrieval alone (the temporal sequence of changes matters). The agent needs
something in between.

The scratchpad pattern is one answer. Give the agent a tool —
`write_notes(text)` and `read_notes()` — that writes to and reads from a plain
text file. The agent maintains its own running summary: decisions made, files
changed, remaining tasks, important constraints discovered. The conversation
history can be cleared or truncated freely, because the essential state is in
the scratchpad. The model operates with a short transcript (recent turns) plus a
self-maintained index (the notes). The scratchpad is not a transcript — it's
curated, compressed, updated by the agent itself. And it's not a vector index —
it's unstructured text, read in full, written deliberately. It's a third thing:
working memory. The equivalent of a programmer's notepad on the desk — not the
full record of everything they've done, and not a database of facts, but a
running record of what matters right now.

This pattern deserves more attention than it gets. Most conversations about LLM
memory focus on the transcript-vs-index axis and miss the possibility that the
model itself can be the memory manager — not through a library's summarization
algorithm, but through its own judgment about what's worth remembering. The
model is better at deciding what matters than any retrieval algorithm, because
the model understands the task. A `write_notes` tool gives it a place to put
that understanding, and a `read_notes` tool gives it a way to recover it after
history has been cleared.

The tradeoff is trust. You're trusting the model to maintain accurate notes. It
might forget to record something. It might record something incorrectly. It
might lose track of its own notes as they grow. These are real failure modes.
But they're the same failure modes a human programmer has with a notepad — and
the notepad is still the most common tool for managing working memory during
complex tasks. It works not because it's perfect, but because it puts the memory
decisions in the hands of the entity doing the work.

A conversation model that acknowledges these three kinds of memory — transcript
(short-term, complete, ordered), index (long-term, selective, unordered), and
working memory (medium-term, curated, self-maintained) — is richer than one that
offers only a transcript. Most LLM libraries, including lm15, offer only the
transcript. The index and working memory are built by the application, using
tools and external storage. Whether the library should own these abstractions —
or leave them to the application developer — is a reprise of the ownership
question from earlier in this chapter, applied to a harder domain.
