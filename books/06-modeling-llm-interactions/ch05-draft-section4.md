## The Boundary That Matters

Every design decision in this chapter — Protocol vs inheritance, per-provider vs
per-model, translation vs delegation, early vs late yield in streaming — serves
one architectural property: **the adapter and the library can change
independently.**

The adapter knows everything about one provider's wire format and nothing about
the library's internals — not how the Model class manages history, not how the
middleware pipeline wraps calls, not how the factory resolves API keys. The
library knows everything about routing, types, and orchestration and nothing
about any provider — not how OpenAI encodes tool calls, not how Anthropic places
cache markers, not how Gemini handles file uploads.

The boundary between them is narrow: a Protocol with five methods, a set of
frozen dataclasses, and an entry-point declaration. Everything else is internal
to one side or the other.

This boundary is what makes the plugin system work. `lm15-x-mistral` is a Python
package with one class, one factory function, and one line in `pyproject.toml`.
It imports lm15's types. It doesn't import lm15's implementation. When lm15's
internals change — new middleware, new factory logic, new Model features — the
plugin doesn't break, because it never depended on the internals. When the
provider's API changes, the plugin updates and lm15 doesn't, because lm15 never
knew the wire format.

```toml
[project.entry-points."lm15.providers"]
mistral = "lm15_x_mistral:build_adapter"
```

One line. `pip install lm15-x-mistral`. `lm15.complete("mistral-large",
"Hello.")` works. No configuration, no imports, no registration code.

This is the payoff of five workmanlike decisions: Protocol for decoupling,
per-provider for simplicity, translation for independence, early-yield for
liveness, entry points for discovery. None is elegant. Each is the least-bad
option for its context. Together, they produce a system where adding a provider
is a small, self-contained project, and where the library and its providers can
evolve at different speeds without breaking each other.

In a domain where both the library and the providers are evolving simultaneously
— new models every month, new API features every quarter, new providers every
year — independent evolution isn't a nice-to-have. It's the property that
determines whether the architecture survives contact with time. Every other
property — elegance, performance, type safety — is secondary to the question of
whether the system can absorb change without cascading breakage.

The adapter pattern's deepest argument isn't that it's the right way to
structure a translation layer. It's that translation layers need boundaries, and
boundaries need to be drawn at the point where change is most frequent. The
change in LLM systems is at the provider level — new models, new features, new
wire formats. The adapter boundary puts that change on one side and everything
else on the other. That's why it works. That's why it matters.
