# Using the router

`LMRouter` turns a model string into the right provider LM. It is the
recommended front door because it removes the one piece of boilerplate
every program repeats — "which class, which env var" — without adding a
framework: three fixed resolution rungs, a printable rule table, and a
`Resolution` value that tells you exactly what happened. The direct LM
classes remain first-class; both paths produce the identical `Request`
and wire bytes.

```python
from lm15 import LMRouter, Message, Request

router = LMRouter()
response = router.complete(
    Request(model="claude-sonnet-4-5", messages=(Message.user("Hi!"),))
)
print(response.text)
```

That is the whole API surface: `resolve()`, `lm()`, `complete()`,
`stream()`. `AsyncLMRouter` is the async mirror (async `complete`, async
iterator from `stream`; `resolve()` stays sync because it is pure).

## The model-string grammar

A model string is split on the **first** `:`. If the head is a known
provider string (a key of `lm15.ADAPTERS`), the remainder is the model id
sent on the wire. Otherwise the whole string — colons and all — is a bare
model id.

```text
"anthropic:claude-sonnet-4-5"   provider prefix + wire id
"gpt-4.1-mini"                  bare id, resolved by catalog or rule
"qwen3.5:0.8b"                  bare id ("qwen3.5" is not a provider)
"openai:ft:gpt-4.1:org"         fine-tune ids need the explicit form
```

Known providers: `openai` (Responses API), `openai_chat` (Chat
Completions), `anthropic`, `gemini`, `claude-code`, `openai-codex`.

## How resolution works, step by step

`resolve()` walks exactly three rungs, in a fixed order you cannot
reconfigure. First match wins; no fallback chains, no plugins.

1. **Explicit prefix.** `"openai:gpt-4.1-mini"` → provider `openai`, wire
   model `gpt-4.1-mini`. Source: `"prefix"`. Always available, always
   unambiguous.
2. **Catalog.** Only if you passed a registry in `RouterConfig`. The model
   string is matched against each `ModelInfo.id` and its `aliases`. An
   exact id match beats an alias match; an alias resolves to the
   canonical id. If more than one provider offers the id — or one
   provider's catalog matches more than one entry — resolution **fails**
   with `AmbiguousModelError` rather than pick one for you — the error
   names every candidate and the explicit form that fixes it.
   Source: `"catalog"`.
3. **Built-in rules.** A small prefix table, `lm15.DEFAULT_RULES`
   (`claude-` → anthropic, `gpt-`/`o1`/`o3`/`o4` → openai, `gemini-` →
   gemini). First match wins. The table is a convenience, not a registry
   of truth: a brand-new model family needs a release, a catalog, or the
   `provider:` prefix. Source: `"rule"`.

Nothing matched? `UnknownModelError`, carrying the rules tried and
whether a catalog was searched, with concrete fixes in the message.

`resolve()` is pure: no network, no secret values read. Its return value
*is* the explanation — there is no separate `explain()`:

```python
res = LMRouter().resolve("claude-sonnet-4-5")
print(res.source)     # rule
print(res)            # 'claude-sonnet-4-5' -> provider 'anthropic' (AnthropicLM);
                      # via built-in rule prefix='claude-' — Anthropic Claude family;
                      # wire model 'claude-sonnet-4-5'; key from $ANTHROPIC_API_KEY.
```

Every field is typed: `requested`, `model` (wire id), `provider`,
`adapter`, `source`, `rule`, `env_key`, `model_info` (catalog metadata
when source is `"catalog"`).

## Credentials

`lm()` constructs the provider LM, looking up the key in this order:

1. `RouterConfig(api_keys={"anthropic": "..."})` — explicit, repr-suppressed,
   beats the environment. Pass `env={}` too for fully hermetic tests.
2. The provider's `ProviderManifest.env_keys`, first set variable wins —
   the same declaration `lm15.auth` and the adapters already use. The
   router adds no new env vars.

No key found → `MissingCredentialError`, which subclasses the existing
`NotConfiguredError` so current handlers keep working. OAuth providers
(`claude-code`, `openai-codex`) declare no env keys; the router calls
their self-resolving constructors and `env_key` is `None`.

`resolve()` only ever records *which* env var would be read (in
`Resolution.env_key`), never the value.

One LM is built per provider, lazily, and reused across calls. The cache
is the router's only mutable state.

## Catalogs: aimo and friends

Catalog use is opt-in. Pass a registry and rung 2 lights up:

```python
from lm15 import LMRouter, ModelRegistry, RouterConfig

registry = ModelRegistry.discover()        # entry-point group "lm15.model_catalogs"
router = LMRouter(config=RouterConfig(registry=registry))

res = router.resolve("sonnet")             # an alias, if your catalog defines one
print(res.model_info.inference.pricing.input_per_million)
```

`discover()` hydrates from installed catalog packages via the entry-point
protocol specified in [Model hydration](model-hydration.md); the `aimo`
package implements it. The guardrail there applies here too: catalog data
is advisory metadata. It selects a provider and canonicalizes an alias —
it never changes what `build_request` produces.

If the catalog resolves to a provider lm15 has no adapter for (say, a
hosted OpenAI-compatible service), the error says so and points you at
constructing `OpenAIChatLM` with a `base_url` directly.

## Without a catalog

The router degrades gracefully: rungs 1 and 3 need nothing installed.
Explicit prefixes always work; the built-in rules cover the mainstream
model families. The only thing you lose is alias/metadata resolution —
and `UnknownModelError` tells you a catalog would have been consulted if
you had supplied one.

## When to use direct LM objects instead

Both paths are first-class. Skip the router and construct the LM
yourself when:

- you need a **custom `base_url`, transport, or compat preset** — ollama,
  vLLM, Azure, OpenRouter. The router deliberately has no syntax for
  this; `OpenAIChatLM(api_key=..., compat="ollama")` is the documented
  path and is one line.
- you are a **library** wrapping lm15: take an LM object from your
  caller; don't impose string parsing on your API.
- you want **zero resolution logic** in the call path, or several
  differently-configured instances of the same provider.

And the escape hatch is built in: `router.lm("gpt-4.1-mini")` returns an
ordinary `OpenAILM`. Keep it, configure it, never call the router again.

```python
lm = LMRouter().lm("gpt-4.1-mini")   # plain OpenAILM
```

## Customizing the rule table

Rules are data, not callbacks:

```python
from lm15 import DEFAULT_RULES, LMRouter, RouteRule, RouterConfig

rules = (RouteRule("glm-", "openai_chat", note="in-house vLLM"), *DEFAULT_RULES)
router = LMRouter(config=RouterConfig(rules=rules))
```

First match wins, so prepend to override. Note that a rule can only name
a provider in `ADAPTERS` — pointing `glm-` at `openai_chat` routes the
request, but a non-default `base_url` still requires constructing the LM
directly (see the escape hatch above).
