## The Quadratic Bill

Let's make the cost concrete, because abstractions like "quadratic growth" hide
the numbers that actually determine whether a product is viable.

A single turn of a conversation sends N tokens of input and receives some
output. The next turn sends the original N tokens, plus the output from turn 1,
plus the new message — call it 2N for simplicity. Turn 3 sends 3N. Turn K sends
KN. The total input tokens across a K-turn conversation are N × (1 + 2 + 3 + ...
+ K) = N × K(K+1)/2.

For a research assistant — system prompt of 200 tokens, tools schema of 400
tokens, average turn of 300 tokens (question + answer) — N is about 900 tokens.
Here's what a conversation actually costs at Claude Sonnet's input pricing ($3/M
tokens):

| Turns | Cumulative input tokens | Input cost | Per-turn average |
|---|---|---|---|
| 5 | 13,500 | $0.04 | $0.008 |
| 10 | 49,500 | $0.15 | $0.015 |
| 20 | 189,000 | $0.57 | $0.028 |
| 50 | 1,147,500 | $3.44 | $0.069 |

The per-turn cost nearly doubles every time the conversation doubles in length.
At 50 turns — which is ordinary for a coding agent that reads files, writes
code, runs tests, and iterates — the input alone costs $3.44. Add output tokens
and reasoning, and a single agent task can exceed $5.

This is where prompt caching enters, and it's important to understand what it
actually is, because the name is misleading.

"Caching" suggests that something is stored and reused — that the model
remembers processing the prefix and skips it. That's the operational effect, but
the conceptual model matters. Prompt caching doesn't give the model memory. It
doesn't change what the model sees. The full conversation is still sent on every
call. Every message, every tool definition, every prior turn — all of it,
transmitted and received, exactly as before. What changes is that the provider
recognizes the prefix — "I've processed these tokens before, recently, for this
API key" — and skips the computational work of re-encoding them. The input is
the same. The computation is reduced. The output is identical to what it would
have been without caching.

This distinction matters because it tells you what caching can and can't do.

**Caching reduces cost.** On Anthropic, cached tokens are charged at 10% of the
input rate. The 50-turn conversation above drops from $3.44 to roughly $0.70 —
an 80% reduction. On OpenAI, it's 50% off. On Gemini, 75% off. These are
substantial savings that make long conversations economically viable.

**Caching reduces latency.** Processing 100K tokens of input takes time —
hundreds of milliseconds to seconds, depending on the model. Cached tokens are
processed near-instantly. The user experiences faster responses on every turn
after the first, because only the new tokens require full computation.

**Caching does not improve quality.** The model sees the same input with or
without caching. If the conversation is too long for the model to attend to
effectively — the "lost in the middle" problem from the previous section —
caching doesn't help. It makes the same degraded response cheaper and faster to
produce.

**Caching does not reduce bandwidth.** The full conversation is still
transmitted over the network. At 50 turns and 1M tokens, that's roughly 4MB of
JSON per request. In most scenarios this is negligible. In edge environments
with limited bandwidth, it isn't.

The analogy to HTTP caching is useful and precise. When a browser sends a
conditional GET with `If-None-Match`, the full request travels over the wire.
The server checks its cache, finds a match, and returns `304 Not Modified` —
same headers, no body. The client still made the request. The server still
received it. The savings are in the response body computation and transfer, not
in the request. LLM prompt caching works the same way: the client still sends
the full conversation, the server still receives it, but the server skips the
expensive computation on the cached prefix.

This framing clarifies a mistake I see developers make: treating prompt caching
as a memory architecture. "We'll use caching so the model remembers the
conversation" — no. Caching is a billing optimization. The model "remembers"
because you sent the entire history. Caching makes that sending cheaper. If you
stop sending the history, the model forgets, regardless of what's cached. The
cache is keyed on the token prefix, not on a session ID. It's computation reuse,
not state management.

For library designers, the implication is clean: caching is an implementation
detail of the transport layer, not a feature of the conversation model. It
should be toggleable (`prompt_caching=True`) and observable
(`resp.usage.cache_read_tokens`), but it shouldn't change the conversation's
semantics. The model sees the same messages whether caching is on or off. The
bill is different. The behavior is not.

lm15 treats it exactly this way — one boolean, observable cache metrics, no
semantic side effects. This is right. But it's worth noting what's tempting and
wrong: building a conversation system that *relies* on the cache being warm. "We
can send 200K tokens because the first 195K are cached" is true today, but
caches expire. Anthropic's ephemeral cache lasts minutes. If the user pauses for
lunch and returns, the cache is cold, and the 200K-token request is billed at
full rate. A conversation architecture should work correctly — if expensively —
with a cold cache. The cache is an optimization, not a load-bearing wall.
