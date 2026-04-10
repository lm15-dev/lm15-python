# Chapter 3: Modeling Tool Use

A coding agent is refactoring a test suite. It reads the files, understands the
structure, proposes changes. On turn 14, it decides the old test fixtures are no
longer needed. It calls `run_command("rm -rf tests/fixtures")`. The fixtures are
gone. Some of them were hand-written, capturing edge cases that took weeks to
discover. The model, having no concept of "irreversible" or "valuable" or "maybe
check first," proceeds to the next step and reports success.

Everything in the first two chapters of this book was safe. Messages are
representations — data structures that describe content. Conversations are
protocols — sequences of messages managed by the caller. The model generates
text, the application reads it, and if the text is wrong, nothing has happened.
You got a bad answer. You throw it away and try again. The model's mistakes are
contained within the conversation. They don't leak into the world.

Tool calling breaks this containment. The moment a model can call `write_file`,
`send_email`, `run_sql`, or `execute_command`, its output is no longer text
about the world. It's action *on* the world. The difference is the difference
between a weather forecaster who's wrong about tomorrow's rain and a pilot who's
wrong about which button to press. One produces a bad prediction. The other
produces a consequence.

## The Phase Transition

This is not a feature being added to a text generator. It's a change in the kind
of system you're operating.

Before tools, a language model is an **oracle**. You ask a question, it
generates an answer. The answer might be wrong — hallucinated, outdated, subtly
misleading — but the wrongness is inert. It sits in a string. It doesn't do
anything. The worst case is a user who believes a wrong answer, and even that
requires the user to act on it. The model's error and its consequence are
separated by a human decision.

After tools, a language model is an **agent**. It doesn't just answer — it acts.
It requests function calls, receives results, and requests more function calls
based on those results. The model's decisions drive real computation, real I/O,
real mutations. And the model makes these decisions using the same probabilistic
mechanism it uses to write poetry — next-token prediction, softmax sampling,
pattern matching over training data. There is no separate "decision-making"
module that's more careful or more reliable than the text generation module. The
model that hallucinates a fact and the model that requests a tool call are the
same model, using the same weights, applying the same mechanism. The only
difference is that one output is read by a human and the other is executed by a
machine.

This has implications that ripple through every layer of library design.

**The error model changes.** In text generation, errors are soft. A wrong answer
can be corrected in the next turn, or ignored, or caught by the user. In tool
execution, errors can be hard — a deleted file stays deleted, a sent email stays
sent, a committed transaction stays committed. The retry logic from Chapter 2
("if the API call fails, try again") assumed that calls were idempotent — that
retrying a failed request was safe because the request had no side effects. Tool
calls shatter this assumption. Retrying a failed `send_email` might send the
email twice. Retrying a failed `write_file` might overwrite a file that was
modified between attempts. The library's error handling must now distinguish
between "the API call failed" (safe to retry) and "the tool execution produced a
wrong result" (not safe to retry — the wrong result is already in the world).

**The trust model changes.** In text generation, you trust the model to produce
useful text. If it doesn't, you've lost some time and some tokens. In tool
execution, you trust the model to make good decisions about what actions to take
— which files to modify, which commands to run, which APIs to call. The
consequences of misplaced trust are proportional to the power of the tools you
provide. A model with `read_file` can waste your time. A model with `write_file`
can corrupt your data. A model with `run_command` can do anything your user
account can do. The set of tools you provide is, functionally, a permission
system — a declaration of what you trust the model to do. Most developers don't
think of it this way. They think of tools as features. But every tool is a
permission granted to a non-deterministic system.

**The API design changes.** This is the one that isn't obvious until you've
built it. Tool calling requires state between the model's request and the
application's response. The model says "call `search`." The application must
execute the search, then send the result back — which requires knowing the
conversation so far, the pending tool call's ID, and the tool definitions. A
stateless function (`lm15.complete()`) can't do this. The conversation must be
kept alive between the tool request and the tool result. This is the real reason
`lm15.model()` exists — not for conversation history (that's a secondary
benefit), but for the state continuity that tool round-trips require.
`submit_tools()` needs a conversation to submit *into*. The Model object is, at
its core, a tool-call state machine that also happens to manage conversations.

This last point deserves emphasis because it reveals something about how lm15's
design emerged. If you'd asked "what class should manage multi-turn
conversations?", you'd design a conversation manager — a list of messages with
append and clear operations. If you'd asked "what class should manage tool-call
round-trips?", you'd design a state machine — something that tracks pending
calls, accepts results, and continues the conversation. lm15 asked the second
question, and conversation management fell out of the answer. The `Model` class
is shaped by tools. Conversations ride along.

This isn't just a historical anecdote. It's a design principle: **the most
complex interaction pattern should drive the abstraction.** Text generation is
simple — function in, function out. Conversations are medium — state
accumulation. Tool use is hard — multi-step protocol with side effects. If you
design your abstraction for text generation, conversations and tools will be
awkward add-ons. If you design it for tool use, text generation and
conversations will be natural subsets. The complex case subsumes the simple
cases. The reverse isn't true.

The rest of this chapter examines how tool use works as an interaction protocol,
who should control tool execution, how to design tools for a non-human client,
and what happens when the system acts wrongly on the world. But the foundational
claim is here: tool calling is not a feature. It's a phase transition — from
safe to consequential, from oracle to agent, from prediction to action. Every
design decision that follows is shaped by which side of that transition you're
on.
