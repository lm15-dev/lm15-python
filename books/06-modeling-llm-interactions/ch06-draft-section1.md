# Chapter 6: Streaming as a Design Problem

The first token arrives in about 200 milliseconds. Then nothing for a beat — a
pause just long enough to notice. Then another token. Then two more, faster.
Then the response begins to flow, word by word, at roughly the speed you'd read
it aloud. The text builds on your screen the way a sentence builds in
conversation: incrementally, with momentum, creating the feeling that something
on the other side is *thinking*.

This feeling is the point. Not the engineering, not the architecture, not the
SSE protocol or the event accumulation or the response materialization. The
point is that a 2-second blocking response feels like waiting for a machine, and
a 2-second streaming response feels like watching a mind work. The tokens are
identical. The cost is identical. The quality is identical. What changes is the
temporality — *when* each piece of the response becomes visible — and that
changes everything about how the human on the other end experiences the
interaction.

Streaming matters for trust. A user staring at a blank screen for three seconds
doesn't know if the system is working. A user watching tokens appear knows
immediately — the system is alive, it's processing, it's making progress. The
difference between "is this thing broken?" and "this is working" is the first
token.

Streaming matters for perceived speed. A blocking call that returns in 2 seconds
feels slow. A stream that shows the first word in 200ms and completes in 2
seconds feels fast — even though the total wait is the same. This isn't
irrational. The user can start reading before the response is complete. They're
consuming the output while it's being produced. Real information arrives
earlier, even though the last token arrives at the same time.

Streaming matters for tool-using agents. When an agent searches the web, reads a
document, writes a file, and then answers — a process that might take 30 seconds
across multiple tool calls — streaming is the difference between a blank screen
for 30 seconds and a live feed: "🔍 searching... 📄 reading... ✏️ writing... ✅
done." The user sees every step. They can interrupt if the agent is going in the
wrong direction. They can trust the process because they can observe it.

And streaming matters for debugging. When a model with `reasoning=True`
generates thinking tokens before answering, streaming lets you watch the
reasoning unfold — see where the model's logic goes right, where it goes wrong,
where it hesitates. With a blocking call, you get `resp.thinking` as a complete
string, after the fact. With streaming, you watch the thought happen. The
difference is the difference between reading a chess game's transcript and
watching it played.

All of this — trust, perceived speed, agent visibility, debugging — comes from
one design decision: show the response as it's generated, not after. The
decision sounds simple. Its implementation is the most complex subsystem in lm15
— more complex than the message representation, more complex than the
conversation model, more complex than the tool execution loop.

Because streaming doesn't just change *when* the user sees the response. It
changes the programming model. A blocking call is a function: input in, output
out, done. The caller's code runs, then the function runs, then the caller's
code runs again. Control flows in a straight line.

A streaming call is an observation: the caller and the library are *both
running*, with the caller consuming data as the library produces it. The
caller's code runs *between* events — processing each one, deciding whether to
continue, potentially doing work (printing, logging, updating a UI) at each
step. Control flow is interleaved. The caller and the library take turns.

This interleaving creates design questions that blocking calls don't face. Who
initiates the interleaving — does the caller pull events, or does the library
push them? What happens when the caller is slow — does the stream buffer, block,
or drop events? What happens when the stream fails mid-way — does the caller get
the partial data? What happens when the stream contains tool calls — does the
stream pause while the tool executes?

Each of these questions has multiple valid answers, and the answer you choose
determines what kind of code your users write. Not what kind of streaming code —
what kind of *all* code. Because the streaming model is the most viral part of a
library's API: if streaming is callback-based, the user's code becomes
event-driven. If streaming is async, the user's code becomes async. If streaming
is iterator-based, the user's code stays synchronous. The streaming model
doesn't just handle streams. It shapes the codebase that uses the library.

The rest of this chapter examines the choices: five consumption models, the
two-level API that lets most users ignore the complexity, the dual nature of a
stream that's both an event source and a response container, the partial failure
problem that has no clean solution, and the temporal complexity of streaming
tools and reasoning. But the foundational claim is here: streaming is not a
feature bolted onto a blocking API. It's a different interaction modality — one
that changes the relationship between the library and the user's code in ways
that outlast any individual stream.
