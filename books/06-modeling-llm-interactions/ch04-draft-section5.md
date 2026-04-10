## The Moving Boundary

The normalization question has a temporal dimension that most discussions miss.
The boundary between "honestly normalizable" and "distortion required" isn't
fixed. It moves — and it moves in one direction.

Consider tool calling. In early 2023, OpenAI introduced function calling with a
specific JSON format. Anthropic launched tool use months later with a different
format. Gemini followed with a third. At that point, normalizing tool calls
required significant translation between three genuinely different structures —
different field names, different encoding for arguments (JSON string vs JSON
object), different handling of tool results. A library that normalized them was
doing real work and introducing real distortions.

By 2025, the protocols have converged. All three providers use the same
conceptual model: the model emits a structured tool-call object, the application
executes, the result goes back as a message. The field names still differ
(`tool_calls` vs `tool_use` vs `functionCall`), but the structure is the same.
The normalization that was once a significant distortion is now a mechanical
renaming. The map got more accurate because the territories became more similar.

The same trajectory is visible in other features. Streaming started with very
different SSE formats — OpenAI sent `data: [DONE]`, Anthropic sent different
event types, Gemini had yet another format. Over time, the formats have become
more similar (not identical, but closer). Multimodal input started with
completely different encoding for images; now all three use some variant of
"base64 data or URL in a typed content block."

The pattern is **competitive convergence**. Providers watch each other. When
OpenAI introduces a feature (function calling, vision, streaming), Anthropic and
Gemini add their versions. The first implementations diverge. Over 6-18 months,
the implementations converge toward a shared shape — usually the shape of
whoever moved first, with modifications. The convergence isn't complete (it
never is), but the remaining differences shrink from "fundamentally different
concepts" to "different names for the same thing."

This matters for library design because it means the normalization tax — the
effort required to translate between providers, and the distortion introduced by
the translation — decreases over time. Features that were expensive to normalize
last year are cheap to normalize this year. Features that required escape-hatch
workarounds are becoming universal parameters.

The design implication is subtle but important: **don't over-engineer the escape
hatch**. If a feature is in the hatch today because the providers are too
different, it might be ready for the universal API in six months. Build the
hatch to be easy to empty, not comfortable to live in. The hatch should feel
provisional — a temporary holding zone, not a permanent residence.

Conversely, **don't rush to normalize**. A feature that just launched on two
providers with different semantics will look more similar in a year. Normalizing
it now locks in a mapping that might be wrong as the providers evolve.
`prompt_caching=True` was a reasonable normalization decision because caching
had existed long enough for the concepts (if not the mechanisms) to stabilize.
`output="audio"` — which currently works on only one provider — might be
premature.

The temporal dimension also affects the library's documentation obligations. A
normalization that's honest today might become distorted tomorrow (if a provider
changes their mechanism), or might become more honest (if providers converge).
The documentation should state not just "this parameter does X" but "this
parameter does X on provider Y and Z on provider W, and the difference matters
when..." This is more work than documenting a single behavior. It's also more
honest.

The moving boundary is the strongest argument for lm15's structural
normalization position — the middle of the spectrum. Pass-through doesn't
benefit from convergence (the user writes provider-specific code regardless).
Behavioral normalization is strained by convergence (each movement requires
updating the behavioral contract). Structural normalization absorbs convergence
naturally — as providers become more similar, the adapters' translation becomes
simpler, and the universal types become more honest, without any change to the
library's API.
