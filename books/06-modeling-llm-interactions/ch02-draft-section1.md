# Chapter 2: The Conversation Problem

Run this experiment. Call a language model twice:

```python
resp1 = lm15.complete("claude-sonnet-4-5", "My name is Alice.", env=".env")
resp2 = lm15.complete("claude-sonnet-4-5", "What's my name?", env=".env")
print(resp2.text)
```
```
I don't know your name — you haven't told me.
```

The model has no idea that you just told it your name. Not because of a bug, not
because of a configuration error, not because the library discarded something.
The model literally doesn't remember. It never did. It never will. The second
call is a stranger walking up to a machine and asking a question. No context, no
history, no relationship to anything that came before.

This surprises almost everyone the first time they encounter it, because the
chat interfaces — ChatGPT, Claude.ai, Gemini — create a powerful illusion of
continuity. You type a message, the model responds, you type another, the model
responds with full awareness of the conversation. It feels like talking to
something that remembers. But what's actually happening is that the chat
interface re-sends the entire conversation — every previous message, both yours
and the model's — as the input to each new call. The model doesn't *remember*
the earlier turns. It *re-reads* them. Every time.

This distinction — between remembering and re-reading — sounds like a
technicality. It's not. It's the foundation on which every conversation system
is built, and getting it wrong leads to architectures that are either too
expensive, too brittle, or too opaque. This chapter is about modeling
conversation — the act of creating continuity in a system that has none.

## The Stateless Machine

A language model is a function. It takes an input — a sequence of tokens — and
produces an output — a probability distribution over the next token, sampled
repeatedly to generate a response. The function has no side effects. It modifies
no state. It writes to no database. If you call it with the same input twice,
you get the same distribution of outputs (sampling introduces variance, but the
distribution is identical). In the vocabulary of computer science, it's a pure
function. In the vocabulary of web architecture, it's a stateless server.

This isn't an implementation detail that might change with future architectures.
It's a consequence of how transformer models work. The model's parameters — its
weights — are fixed after training. They don't update when you talk to it.
Nothing about your conversation is written into the model's weights, stored in a
buffer, or saved to a session. The model's "knowledge" of your conversation
exists in exactly one place: the input you send it.

This means that what users experience as "conversation" is entirely
reconstructed on every call. The full conversation history — system prompt,
previous user messages, previous assistant responses, tool calls, tool results —
is serialized, sent as the input, and processed from scratch. The model reads
the entire transcript, from the beginning, every single time. Turn 10 of a
conversation is not "turn 10" to the model. It's a single request that happens
to contain 10 turns worth of context in its input.

The analogy to the web is precise and illuminating. HTTP is a stateless
protocol. Each request is independent. But web applications need sessions —
login state, shopping carts, preferences. The solution is to reconstruct state
on each request, typically by sending a session cookie that the server uses to
look up persisted state. The session exists not because HTTP supports it, but
because the application layer simulates it on top of a stateless protocol.

LLM conversations work the same way. The "session" is the message history. The
"cookie" is the conversation context included in the request. The model is the
stateless server that processes each request independently. And just like web
sessions, the conversation state must live *somewhere* — the question is where,
and who manages it.

But the analogy breaks in one important direction. A web server with a session
cookie looks up a small token and accesses a state object. An LLM with
conversation history re-processes the entire state from scratch. Imagine if
every HTTP request included the full contents of the session — every page
visited, every form submitted, every response received — and the server
re-derived the user's current state by replaying the entire history. That's what
an LLM does. The input isn't a pointer to context; it *is* the context. And the
context grows with every turn.

This has a cost consequence that's easy to miss. In a web application, the cost
of a request is roughly constant regardless of session length — the server looks
up the session, processes the request, returns a response. In an LLM
conversation, the cost of a request grows linearly with conversation length —
because the entire conversation is the input, and the provider charges per input
token. A 10-turn conversation doesn't cost 10x a single turn. It costs 1 + 2 + 3
+ ... + 10 = 55 units, because each turn re-sends all previous turns. The cost
is quadratic in the number of turns. (Chapter 8's discussion of prompt caching
addresses this — but it's important to understand the underlying cost structure
before reaching for the optimization.)

And it has a reliability consequence that's even easier to miss. In a web
session, old state doesn't interfere with new requests — the server reads the
current state, processes the current input, and responds. In an LLM
conversation, old turns are live context — the model reads them alongside the
new message, and they influence the response. A misleading early answer, a
failed tool call, a confusing tangent — these stay in the history and can bias
all subsequent turns. The model has no mechanism for "forgetting" or
"de-weighting" old context. It reads the history the way it reads any text:
sequentially, with equal attention to every token. Your conversation history is
not just state. It's an ever-growing prompt that you didn't write.

This last point deserves emphasis, because it's where the HTTP analogy fully
breaks down. A web session is inert state — it records what happened, but it
doesn't influence what happens next except through explicit application logic. A
conversation history is active context — it shapes the model's responses through
the same mechanism that the system prompt uses. An unhelpful response on turn 3
doesn't just waste time; it becomes part of the prompt for turns 4 through N,
and the model may pattern-match on it, extend it, or defer to it. Managing
conversation history isn't just an engineering concern about cost and context
window size. It's a prompt engineering concern about the quality of the context
the model sees.

This is the fundamental tension of conversation modeling: the model is
stateless, but the application needs statefulness; the conversation history
provides that statefulness, but it grows without bound, costs quadratically, and
influences the model in ways the developer doesn't control. Every design
decision in this chapter — who owns the history, how it's managed, when it's
truncated, what's cached — is a response to this tension.
