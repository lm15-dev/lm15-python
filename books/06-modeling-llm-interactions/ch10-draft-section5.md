## The Convergence Bet

This entire book rests on a bet: that providers are converging. The universal type system (Chapter 1), the normalization layer (Chapter 4), the adapter architecture (Chapter 5), the escape-hatch lifecycle (previous section) — all of them are premised on the idea that OpenAI, Anthropic, and Gemini are becoming more similar over time, and that a universal model built around their shared patterns will become more accurate, not less.

The evidence for convergence is substantial.

Tool calling launched with three wildly different wire formats. Two years later, all three providers use the same protocol: model emits a structured request, application executes, result returns in a follow-up message. The names differ. The JSON structure is almost the same. The semantic protocol is identical. An adapter written today does less translation work than one written at launch.

Content blocks converged similarly. OpenAI introduced typed content arrays. Anthropic adopted the pattern independently. Gemini uses `parts` with the same structure under different names. The ordered-array-of-typed-blocks pattern is now universal — not because the providers coordinated, but because the pattern is a natural fit for heterogeneous content.

Streaming converged. SSE with JSON deltas is now the standard across all three providers. The event names differ. The delta structures differ. But the pattern — server-sent events carrying typed JSON deltas — is shared.

New providers (Mistral, Cohere, DeepSeek) launch with APIs that already resemble the converged pattern. They adopt the chat completions format, the tool calling protocol, the SSE streaming — because that's what the ecosystem expects. Convergence is self-reinforcing: each new provider that adopts the pattern makes it harder for the next one to diverge.

The evidence against convergence is also real.

Each provider is developing unique capabilities that the others don't share and may never adopt. Anthropic's extended thinking with token budgets. OpenAI's structured output with JSON Schema enforcement. Gemini's grounding with Google Search. These are differentiating features — the reason you'd choose one provider over another. They're deliberately non-convergent, because convergence would eliminate the competitive advantage.

The pattern: **infrastructure converges, capabilities diverge.** The wire format, the content model, the tool protocol, the streaming mechanism — these converge because they're infrastructure. They don't differentiate. No user chooses Anthropic because their JSON structure is better. Providers converge on infrastructure because divergence costs them (every unique format is a barrier to adoption) without benefiting them (no user values format uniqueness).

Capabilities diverge because they differentiate. Extended thinking is a reason to choose Claude. Structured output is a reason to choose GPT. Google Search grounding is a reason to choose Gemini. Providers invest in unique capabilities because divergence benefits them (competitive moats) and costs their competitors (harder to replicate).

For the library, this means the universal model captures an expanding infrastructure core and leaks a persistent capability fringe. The core grows as providers align on shared patterns. The fringe persists as providers invest in differentiation. The escape hatches carry the fringe. The universal types carry the core.

Three scenarios for the library's future:

**Convergence accelerates.** The infrastructure core grows. The capability fringe shrinks (providers adopt each other's innovations). The universal model becomes more accurate. The escape hatches empty. The library's value increases — it captures more of what the providers offer through universal types.

**Convergence stalls.** The infrastructure core stabilizes. The capability fringe grows (each provider develops more unique features). The universal model stops growing. The escape hatches fill. The library is useful for the infrastructure core but increasingly requires escape-hatch access for the features that matter most. The library's value plateaus.

**One provider dominates.** One provider captures 80%+ market share. Multi-provider support becomes less valuable — most users only need one provider. The library's value proposition (switch providers by changing a string) becomes academic. The library is still useful as a cleaner API over the dominant provider's SDK, but the multi-provider architecture is overhead.

Which scenario is most likely? The evidence suggests continued convergence on infrastructure with persistent divergence on capabilities — the second scenario with elements of the first. The library remains valuable for the common case (text, tools, streaming, conversations) and increasingly relies on escape hatches for the advanced case (provider-specific features).

This is a stable outcome for the library, not a declining one. The infrastructure core covers 80% of what most applications need. The escape hatches cover the rest. The 80% justifies the library's existence. The 20% justifies the escape hatches' existence. Both can coexist indefinitely.
