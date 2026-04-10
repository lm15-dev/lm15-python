## The Escape Hatch

The `provider` dict on `Config` and `LMResponse` is lm15's answer to the
normalization boundary. Anything that doesn't fit the universal model goes in
the dict. It's untyped, undocumented, and explicitly provider-specific.

```python
# Request: pass provider-specific parameters
resp = lm15.complete("claude-sonnet-4-5", "Hard problem.",
    reasoning={"budget": 10000},  # universal-ish (works on Anthropic, approximate on OpenAI)
    env=".env",
)

# Response: access provider-specific data
raw = resp.provider  # the full JSON from Anthropic's response
```

The escape hatch serves three purposes.

**It carries the un-normalizable.** Anthropic's `cache_control` marker
placement, Gemini's `CachedContent` TTL, OpenAI's `response_format` for JSON
mode — these are provider-specific concepts that don't have equivalents
elsewhere. Adding them to the universal `Config` would grow the type with fields
that are meaningful on one provider and meaningless on two. The escape hatch
carries them without inflating the universal type.

**It enables debugging.** When something goes wrong, the developer needs to see
what the provider actually returned — not what lm15 normalized, but the raw
JSON. `resp.provider` carries the complete provider response. The developer can
inspect the actual error message, the actual model version, the actual rate
limit headers, the actual content filter annotations — whatever the provider
sent that lm15 didn't surface as a universal field.

**It provides forward compatibility.** When a provider adds a new feature — say,
a confidence score on tool calls, or a content moderation annotation on the
response — the feature is immediately accessible through the escape hatch,
before lm15 adds universal support. The developer who needs it today doesn't
wait for a library release. They reach through the hatch, use the raw data, and
move on.

### The Cost

The escape hatch is untyped.
`resp.provider["usage"]["cache_creation_input_tokens"]` is a valid access path
on Anthropic and a `KeyError` on OpenAI. No autocompletion. No compile-time
checking. No documentation in lm15 — you need the provider's API docs to know
what keys exist.

This means code that uses the escape hatch is provider-specific code in
disguise. It compiles on any provider but runs only on the provider it was
written for. The universal syntax provides a false sense of portability — the
code *looks* like it uses lm15's universal API, but it's actually reaching
through to a specific provider's response format.

A more honest design might quarantine the escape:
`resp.anthropic.cache_creation_input_tokens` instead of
`resp.provider["cache_creation_input_tokens"]`. This would make the provider
dependency explicit in the code, not just in the data. lm15 chose the dict for
simplicity — one field that works for all providers, no provider-specific
properties — but the simplicity obscures the coupling.

### When to Add a Universal Parameter

The escape hatch exists because not everything should be universal. But over
time, features migrate from the hatch to the universal API. Prompt caching
started as a provider-specific concept and became `prompt_caching=True`.
Reasoning started as Anthropic-specific and became `reasoning=True`. The
migration criteria are informal:

**Add it to the universal API when** two or more providers support the same
*concept*, even if the *mechanism* differs. Prompt caching is a universal
concept (reduce cost on repeated prefixes) with provider-specific mechanisms
(markers, auto-detection, resources). The concept deserves a universal
parameter. The mechanism differences are handled by the adapter.

**Leave it in the hatch when** only one provider supports the concept. Prefill
(seeding the assistant's response with a prefix) is Anthropic-specific. Gemini's
`CachedContent` TTL is Gemini-specific. These belong in the escape hatch because
adding them to the universal type would create parameters that are meaningless
on most providers.

**The gray zone:** features that two providers support differently and one
doesn't support at all. Image generation is supported by OpenAI and Gemini but
not Anthropic. Audio generation is supported by OpenAI only. These are currently
in the universal API (`output="image"`) but could reasonably be in the escape
hatch. The decision is a judgment call about whether the feature is heading
toward universal support or will remain spotty.

The migration from hatch to universal API is a form of **normalization catching
up with convergence**. As providers adopt each other's features, the concepts
converge, and the universal API can honestly represent them. The hatch is a
holding zone for features that haven't converged yet — a place where
provider-specific information lives until it's ready to be normalized.
~~