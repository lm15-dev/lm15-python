## Where the Map Distorts

Three case studies. In each one, a single lm15 parameter maps to three different
provider behaviors. The parameter promises equivalence. The reality delivers
approximation.

### Case Study 1: Prompt Caching

`prompt_caching=True` on a model object. One boolean. Here's what happens on
each provider.

**Anthropic.** The adapter injects `cache_control` markers into the request
body. Specifically, it adds `{"cache_control": {"type": "ephemeral"}}` to the
system prompt content block and to strategically placed message boundaries — the
"advancing breakpoint" pattern, where each turn's breakpoint is placed at the
latest complete exchange. The provider processes the markers, caches everything
up to the breakpoint, and serves cached tokens at 10% of the input price on
subsequent calls. The cache lifetime is short — a few minutes of inactivity, and
the cache expires. The developer must keep calling to keep the cache warm.

**OpenAI.** The adapter does nothing. Literally nothing — `prompt_caching=True`
is a no-op. OpenAI detects and caches repeated prefixes automatically,
server-side, with no client action required. The developer gets caching whether
or not they ask for it. Cached tokens are charged at 50% of the input rate. The
developer can't control what's cached, can't set breakpoints, can't manage cache
lifetime.

**Gemini.** The adapter creates a `CachedContent` resource via a separate API
call to `generativelanguage.googleapis.com`. This is a persistent, server-side
object with its own ID and lifecycle. Subsequent requests reference the cached
content by ID instead of re-sending the prefix. The pricing discount is 75%. The
cached content persists until explicitly deleted or until its TTL expires
(configurable). Creating the resource requires an additional HTTP round-trip
that the other providers don't need.

One boolean. Three mechanisms:

| | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Client action | Inject markers | None | Create resource |
| Server behavior | Cache at markers | Auto-detect prefix | Store resource |
| Discount | 90% off | 50% off | 75% off |
| Lifetime | Minutes (ephemeral) | Automatic | Configurable TTL |
| Extra HTTP calls | 0 | 0 | 1 (creation) |

A developer who writes `prompt_caching=True` and switches from Claude to GPT
will see their caching "still work" — in the sense that the call succeeds and
cached tokens appear in usage. But the discount changed from 90% to 50%, the
lifetime went from minutes to automatic, and the control went from explicit
markers to none. A developer who switches to Gemini will see an extra HTTP call
on the first turn, a persistent resource that needs cleanup, and a 75% discount.

Is this normalization helping or hurting? It depends on the developer's
expectations. If they think of `prompt_caching=True` as "make repeated calls
cheaper" — a high-level intent — the normalization is helpful. The intent is
fulfilled on every provider, and the mechanism differences don't matter. If they
think of it as "enable ephemeral cache control markers" — a specific mechanism —
the normalization is misleading. The mechanism is Anthropic-specific and the
boolean doesn't mean what they think it means on other providers.

Most developers operate at the intent level. They want cheaper calls. They don't
care how the provider achieves it. For them, the distortion is benign — like
Mercator's Greenland, it doesn't affect their navigation. But the developer
who's tuning cache behavior — monitoring hit rates, managing TTLs, optimizing
breakpoint placement — is navigating by the map and will run aground when the
territory doesn't match.

### Case Study 2: Reasoning

`reasoning=True` enables chain-of-thought. Here's the divergence.

Anthropic: the adapter sends `"thinking": {"type": "enabled", "budget_tokens":
N}` — a token budget that limits how much the model can think. The developer can
control spending with `reasoning={"budget": 10000}`. The model's thinking is
returned as content blocks that the developer can read.

OpenAI: the adapter maps reasoning to an effort parameter — `"reasoning_effort":
"high"`. There's no token budget concept. The effort levels are qualitative
(`"low"`, `"medium"`, `"high"`), not quantitative. `reasoning={"budget": 10000}`
is meaningless on OpenAI. `reasoning={"effort": "high"}` maps cleanly, but the
equivalence is approximate — what Anthropic calls 10,000 tokens of thinking and
what OpenAI calls "high effort" are different concepts measured in different
units.

Gemini: reasoning support varies by model, with different parameter names and
semantics again.

The `reasoning=True` boolean works everywhere — it enables thinking. The dict
form (`reasoning={"budget": N}` or `reasoning={"effort": "high"}`) is
provider-shaped. A developer who uses only `reasoning=True` gets honest
normalization. A developer who uses the dict form is writing provider-specific
code in universal syntax — the worst kind of distortion, because it looks
portable and isn't.

### Case Study 3: Image Generation

`output="image"` asks the model to generate an image. On OpenAI, the adapter
routes to DALL-E — a completely different API endpoint
(`/v1/images/generations`) with its own parameters, pricing, and capabilities.
On Gemini, it uses the model's native multimodal generation. On Anthropic, it's
not supported — the call fails with `UnsupportedFeatureError`.

One parameter. Two entirely different generation architectures. One flat-out
unsupported. The normalization here is at its thinnest — `output="image"` is
less a universal parameter and more a convenience alias that happens to route to
whatever image generation capability the provider offers, if any.

### The Pattern

In each case, the distortion follows the same structure. The *intent* normalizes
(make it cheaper, make it think, generate an image). The *mechanism* doesn't
(three caching systems, two reasoning parameterizations, two generation
architectures). The universal parameter captures the intent and hides the
mechanism.

Whether this is good or bad depends on whether the developer needs to understand
the mechanism. For 80% of use cases — "just make it work" — hiding the mechanism
is a service. For the 20% who are optimizing, debugging, or operating at the
edges of a provider's capabilities — hiding the mechanism is an obstacle.

The library's job isn't to eliminate the distortion. It's to make the distortion
visible enough that the developer who needs to see through it can, and invisible
enough that the developer who doesn't need to see through it isn't burdened.
That's what the escape hatch is for.
