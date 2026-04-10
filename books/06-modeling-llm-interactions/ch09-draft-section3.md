## The Convenience-Transparency Tradeoff

Every concealment decision sits on a tradeoff curve between convenience (the user doesn't need to know) and transparency (the user can find out). The ideal is both: convenient by default, transparent on demand. The escape hatch pattern is lm15's primary mechanism for achieving both, but it's not the only mechanism, and it's not always sufficient.

**Convenient by default, transparent on demand** works at three levels in lm15:

The *API level*. `lm15.complete()` is maximally convenient — one function, one response. `lm15.model()` adds configuration reuse and conversation. `UniversalLM` with manual adapter registration is maximally transparent — you wire everything yourself. The user chooses their level. Most stay at the top. The ones who need transparency step down.

The *response level*. `resp.text` is convenient — one string, the model's answer. `resp.message.parts` is transparent — every content block, typed and ordered. `resp.provider` is maximally transparent — the raw JSON from the provider, uninterpreted.

The *stream level*. `stream.text` is convenient — just the words. `for event in stream` is transparent — every event type, including thinking and tool calls. `stream.response` materializes the complete response after streaming.

At each level, the simple surface conceals the complex surface, and the complex surface is accessible without ceremony. No subclassing, no configuration flags, no "enable debug mode." Just access a different property or call a different method.

This works because the levels are compositional — each one is built from the one below. `resp.text` computes from `resp.message.parts`. `stream.text` filters from the event iterator. `lm15.complete()` constructs a `Model` and calls it. The convenient surface isn't a separate codepath — it's a view of the transparent surface. This means the transparent surface is always available, always consistent with the convenient surface, and never requires a different mental model.

### Where the Tradeoff Fails

The tradeoff fails when the convenient surface creates expectations that the transparent surface contradicts.

`prompt_caching=True` is convenient — one boolean, works everywhere. But the transparent surface — `resp.usage.cache_read_tokens` — reveals that the caching behaved differently on different providers. The convenient surface said "caching is on." The transparent surface says "caching wrote 1,250 tokens on Anthropic and zero on OpenAI." The user who only reads the convenient surface believes caching works the same everywhere. The user who reads the transparent surface discovers it doesn't. The two surfaces tell different stories, and neither warns the user about the discrepancy.

A better design would connect the surfaces: when `prompt_caching=True` is a no-op (on OpenAI), the transparent surface should say so — perhaps a field like `resp.usage.cache_mechanism = "automatic"` vs `"explicit"` vs `"none"`. The user who checks would know. The user who doesn't would still get caching (or not) without error. The gap between the surfaces is the gap between the stories they tell.

### What Other Libraries Choose to Hide

The concealment decision varies across libraries, and the variation reveals priorities.

LangChain hides mechanism but exposes framework abstractions — chains, agents, memory systems, output parsers. The user sees LangChain's mental model, not the provider's. This is deep concealment: the user doesn't just avoid seeing the wire format; they avoid seeing the provider at all. The cost is that LangChain's abstractions are an additional layer of concealment between the user and the model. When the model behaves unexpectedly, the user must debug through both LangChain's abstraction and the provider's behavior.

LiteLLM hides almost nothing. The input is OpenAI-format dicts. The output is OpenAI-format dicts. The user sees the provider's mental model (OpenAI's specifically), with translation happening underneath. The concealment is minimal — barely more than a routing layer. The user who can read OpenAI's docs can read LiteLLM's types. The cost is that Anthropic and Gemini-specific features are awkwardly accessed through OpenAI-shaped parameters.

The Vercel AI SDK hides mechanism and exposes a framework-specific surface (React hooks, streaming UI primitives). The concealment is tuned for the deployment context: web applications that stream text to browsers. The user sees `useChat()` and `StreamingTextResponse`, not `LMRequest` and `StreamEvent`. The concealment is deep but appropriate — web developers don't need to see the streaming protocol.

lm15 hides mechanism and exposes data. The wire format is hidden. The SSE parsing is hidden. The auth mechanics are hidden. But the types are public, the history is inspectable, the escape hatch is accessible, and the internal API is importable. The user sees the data (what was sent, what was received, what it cost) and doesn't see the plumbing (how it was sent, how it was parsed, how it was authenticated).

Each library's concealment strategy reflects what it considers important for the user to understand. LangChain says: understand our abstractions. LiteLLM says: understand OpenAI's API. Vercel says: understand your UI framework. lm15 says: understand the data — messages, parts, usage, history — and trust us with the plumbing.

The choice "what to hide" is really the choice "what should the user's mental model be?" The concealment creates the mental model by determining what the user sees, and what they see is what they think about. Hide the providers and the user thinks about your abstractions. Hide the abstractions and the user thinks about the providers. Hide the plumbing and the user thinks about the data. The concealment is the pedagogy.
