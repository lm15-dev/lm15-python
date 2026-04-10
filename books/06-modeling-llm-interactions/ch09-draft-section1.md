# Chapter 9: What to Hide

A library is an act of concealment. You take a complex system — HTTP calls, JSON serialization, wire format translation, SSE parsing, error mapping, auth resolution — and you hide it behind a surface the user can understand. `lm15.complete("claude-sonnet-4-5", "Hello.")` conceals a hundred decisions. The user doesn't see the transport, the adapter, the middleware, the factory, the key resolution, the provider routing. They see a function, a model name, and a prompt.

This concealment is the entire point. A library that hides nothing is a copy of the source code. A library that hides everything is magic — convenient until it does the wrong thing, then impenetrable. The design problem is choosing what to put on each side of the surface: what the user sees, what the user doesn't, and — critically — what the user *can* see if they choose to look.

This chapter examines that choice through a test, three categories of concealment, and the cases where lm15 got it right, got it debatable, and got it wrong.

## The One-Sentence Test

There's a practical test for whether something should be hidden: **can you describe the hidden behavior in one sentence without naming a provider?**

"Sends the request to the provider and parses the response." No provider named. Provider-irrelevant. Hide it. The user gains nothing from seeing the JSON construction, the HTTP call, the response parsing. These are implementation details that would change if the providers changed, and the user's code shouldn't depend on them.

"Caches the conversation prefix using Anthropic's ephemeral cache control markers." Provider named. Provider-specific. The user might need to understand this — the cache lifetime, the breakpoint placement, the cost discount — and hiding it creates the illusion of uniformity where none exists.

The test isn't absolute — there are cases where provider-specific behavior should still be hidden (because the user genuinely doesn't care which mechanism is used). But it catches the obvious cases. If you can't describe the behavior without naming a provider, the behavior is provider-specific, and hiding it is a normalization decision (Chapter 4), not an implementation-hiding decision. The user should at least have access to it, even if it's not on the primary surface.
