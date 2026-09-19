# Using the router

`LMRouter` turns a model string into the right provider LM. It is the
recommended front door because it removes the one piece of boilerplate
every program repeats — "which class, which env var" — without adding a
framework: four fixed resolution rungs, a printable rule table, and a
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

A model string is split on the **first** `:`. If the head is a routable
provider string (a key of `lm15.router.ADAPTERS` or of
`lm15.router.CHAT_PRESET_ROUTES`),
the remainder is the model id sent on the wire. Otherwise the whole
string — colons and all — is a bare model id.

```text
"anthropic:claude-sonnet-4-5"   provider prefix + wire id
"gpt-4.1-mini"                  bare id, resolved by catalog or rule
"qwen3.5:0.8b"                  bare id ("qwen3.5" is not a provider)
"openai:ft:gpt-4.1:org"         fine-tune ids need the explicit form
```

Known providers: `openai` (Responses API), `openai-chat` (Chat
Completions), `anthropic`, `gemini`, `xai`, `claude-code`,
`openai-codex` — plus the Chat Completions preset providers `groq`,
`openrouter`, `deepseek`, `zai`, `moonshotai`, `meta-chat`, `ollama`, `vllm`, and `sglang`, which route to
`OpenAIChatLM(compat=<preset>, access=<policy>)`; `deepseek-anthropic`,
`meta-anthropic`, and `moonshotai-anthropic`, which route to `AnthropicLM(compat=<preset>, access=<policy>)`;
and `meta` and `moonshotai-responses`, which route to `OpenAILM(compat=<preset>, access=<policy>)` — each with
that server's default `base_url` and its own credential. The whole list, with each
provider's dialect, env keys, and console URL, is one table:
`lm15.registry.PROVIDERS`.

## How resolution works, step by step

`resolve()` walks exactly four rungs, in a fixed order you cannot
reconfigure. First match wins; no fallback chains, no plugins.

0. **Object attribute.** If the model *value* carries a non-empty string
   `provider` attribute naming a routable provider, that settles it.
   Catalog packages (aimo, …) ship model ids as `str` subclasses that
   know their provider; the check is duck-typed, so lm15 names no
   package, and a plain string never triggers it. An attribute naming
   nothing routable falls through (and is mentioned if nothing else
   matches either). Source: `"object"`.
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
3. **Built-in rules.** A small prefix table, `lm15.router.DEFAULT_RULES`
   (`claude-` → anthropic, `gpt-`/`o1`/`o3`/`o4`/`sora-` → openai,
   `gemini-`/`veo-` → gemini, `grok-` → xai). First match wins. The
   table is a convenience, not a registry
   of truth: a brand-new model family needs a release, a catalog, or the
   `provider:` prefix. Source: `"rule"`.

Nothing matched? `UnknownModelError`, carrying the rules tried and
whether a catalog was searched, with concrete fixes in the message.

`resolve()` is offline: no network or credential renewal, no secret values
returned. It checks configuration and environment presence. Its return value
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
when source is `"catalog"`), `compat` (the preset name when routed
through `CHAT_PRESET_ROUTES`).

`resolve_openai_chat()` is the same lookup for the OpenAI-shaped door
(`complete_from_openai_chat`): it reads a litellm-style `provider/model`
string and sends a bare OpenAI name to `openai-chat` rather than
`openai`, which is where that call goes — see
[Migrating from the OpenAI SDK or LiteLLM](migrating-from-openai-chat.md).

## Credentials

`lm()` constructs the provider LM, looking up the key in this order:

1. `RouterConfig(api_keys={"anthropic": "..."})` — explicit, repr-suppressed,
   beats the environment. An exact provider entry wins; otherwise a
   single entry with the identical non-empty environment-key list supplies
   the credential. For example, `api_keys={"openai": key}` covers both
   OpenAI APIs. Multiple shared candidates require an exact entry instead
   of choosing an account. Pass `env={}` too for fully hermetic tests.
   A value may also be a zero-argument **credential provider** callable
   (an Azure Entra `get_bearer_token_provider(...)`, your own rotation
   logic); the adapter resolves it per request, so long-lived clients
   never hold a stale token.
2. The provider's `ProviderManifest.env_keys` — or, for preset
   providers, the server's own convention (`GROQ_API_KEY`,
   `OPENROUTER_API_KEY`) — first set variable wins. The router adds no
   new env vars.
3. For keyless local servers (`ollama`, `vllm`, `sglang`), the preset's
   placeholder key. These servers accept any value; override via
   `api_keys` if yours is locked down.

The provider strings that key `api_keys` (and `base_urls`, `settings`)
are the ones `resolve()` reports; either spelling (`openai-chat`,
`openai_chat`) works, but do not specify both spellings in `api_keys`.
Only credentials share across matching environment-key declarations;
URLs and host settings remain endpoint-specific. A string that names no routable provider is
refused when the router is built, with the nearest real name — an
ignored entry would send the request out on the environment's key.

No key found → `MissingCredentialError`, which subclasses the existing
`NotConfiguredError` so current handlers keep working. OAuth providers
(`claude-code`, `openai-codex`) declare no env keys; the router calls
their self-resolving constructors and `env_key` is `None`. `xai` is a
hybrid: an explicit key or `XAI_API_KEY` wins, and when neither is set
the router falls back to `XaiLM()`'s own subscription OAuth credential.

`resolve()` only ever records *which* env var would be read (in
`Resolution.env_key`), never the value. The full credential story —
every provider's env var, rotating token providers, subscriptions —
is on [Authentication](authentication.md).

One LM is built per provider, lazily, and reused across calls. The cache
is the router's only mutable state.

## Addresses

`RouterConfig(base_urls={"openai-chat": "https://gw.example/v1"})` gives
a provider's LM a different URL than its default — a proxy in front of
OpenAI, a vLLM server on another host. The entry is matched by provider
string (either spelling), the preset's dialect and credential rule are
unchanged, and a provider without an entry keeps its default. On a cloud
door (`azure-chat`, `bedrock-*`, `vertex`) the entry is the endpoint
root — what the console shows, a private endpoint, a gateway — and the
door appends its own path and keeps its auth scheme and error mapping;
without an entry the vendor's variable (`AZURE_OPENAI_ENDPOINT`,
`AWS_ENDPOINT_URL_BEDROCK_RUNTIME`) is read, then the URL is built from
`RouterConfig(settings=...)` — see [Cloud hosts](cloud-hosts.md).

## Connections: how long to wait, how many at once

A router owns one transport (one connection pool) shared by every LM it
builds, so two fields on `RouterConfig` are that router's whole connection
budget:

```python
from lm15 import LMRouter, RouterConfig, Timeouts

router = LMRouter(RouterConfig(
    timeouts=Timeouts(read=1800),   # seconds to wait for the next byte; default 600
    max_connections=200,            # concurrent connections; default 100
))
```

The defaults are the provider SDKs' (connect 10 s; read, write and the
wait for a free connection 600 s; 100 connections). Raise `read` for a
slow local model or a long non-streaming answer; raise `max_connections`
for a wide evaluation. `Timeouts(pool=None)` waits for a free connection
indefinitely. Pass `transport=` instead to bring your own transport (the
two fields are then refused: configure the transport you pass).

`router.close()` (or `with LMRouter(...) as router:`) closes every
connection; the router can be used again afterwards and simply builds a
fresh transport. `AsyncLMRouter.aclose()` is the async twin — call it on
the event loop that used the router, before that loop ends (one router
per loop; an asyncio connection belongs to its loop).

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

Catalog model objects that carry a `provider` attribute (aimo's do)
resolve on rung 0 before the catalog is even consulted — pass the object
itself as `Request.model` and ambiguity over bare ids disappears.

If the catalog resolves to a provider lm15 has neither an adapter nor a
compat preset for, the error says so and points you at constructing
`OpenAIChatLM` with a `base_url` directly.

## Without a catalog

The router degrades gracefully: rungs 1 and 3 need nothing installed.
Explicit prefixes always work; the built-in rules cover the mainstream
model families. The only thing you lose is alias/metadata resolution —
and `UnknownModelError` tells you a catalog would have been consulted if
you had supplied one.

## When to use direct LM objects instead

Both paths are first-class. Skip the router and construct the LM
yourself when:

- you need a **custom compat policy or transport per provider** — the
  router takes one `transport` for every LM it builds and binds each
  provider's stock compat preset; `OpenAIChatLM(api_key=...,
  compat="ollama", base_url=...)` is the one-line path when those must
  differ. (A different URL alone does not need this:
  `RouterConfig(base_urls={"vllm": "http://gpu-box:8000/v1"})` replaces
  a provider's default or preset URL, keeping its dialect and
  credential rule; on the cloud doors — Azure, Bedrock, Vertex — it is
  the endpoint root the door completes.)
- you are a **library** wrapping lm15: take an LM object from your
  caller; don't impose string parsing on your API.
- you want **zero resolution logic** in the call path, or several
  differently-configured instances of the same provider.

And the escape hatch is built in: `router.lm("gpt-4.1-mini")` returns an
ordinary `OpenAILM`. Keep it, configure it, never call the router again.

```python
lm = LMRouter().lm("gpt-4.1-mini")   # plain OpenAILM
```

## Declaring a provider the registry does not list

A gateway, a hosting service lm15 has not receipted yet, a second
OpenAI-compatible vendor: declare it, and the router treats it like any
registry entry — by name, with the same credential, address and
connection rules — in every router built with that config.

```python
from lm15 import LMRouter, RouterConfig
from lm15.compat import OpenAIChatCompat
from lm15.features import AccessPolicy, EndpointSupport
from lm15.registry import ProviderDefinition

FIREWORKS = ProviderDefinition.chat(
    AccessPolicy(
        provider="fireworks",
        supports=EndpointSupport(complete=True, stream=True, models=True),
        auth_modes=("bearer",),
        env_keys=("FIREWORKS_API_KEY",),
        base_url="https://api.fireworks.ai/inference/v1",
    ),
    compat=OpenAIChatCompat(max_tokens_field="max_tokens", thinking_format="reasoning_effort"),
    aliases=("fireworks-ai",),          # litellm spells it fireworks_ai/
    note="Fireworks (Chat Completions dialect)",
)

router = LMRouter(config=RouterConfig(providers=(FIREWORKS,)))
router.resolve("fireworks:accounts/fireworks/models/deepseek-v4p1-flash").provider   # "fireworks"
router.resolve_openai_chat("fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash")  # same door
```

What a declaration is:

- **`ProviderDefinition.chat(access, compat=...)`** for the Chat
  Completions wire; `.responses(...)` and `.anthropic(...)` for the other
  two dialects. The access policy names the provider, its key variable(s),
  its address and what it serves; the compat object describes the server's
  spellings (`OpenAIChatCompat` fields — the same knobs the built-in presets
  use). You never name an adapter class.
- **`aliases`** are extra input spellings, hyphenated: accepted as
  `alias:model` and `alias/model` (and their underscore forms), never
  emitted. `RouterConfig(api_keys=...)` and `base_urls` are keyed by the id.
- **One string, one door.** An id or alias that a registry entry or a
  litellm prefix already spells is refused when the config is built, so a
  declaration can never shadow `groq` or `ollama_chat/`.

What it is not: a receipt. lm15's registry entries are pinned from live
captures in the contract; a declared provider is your word. The router says
so — `Resolution.declared` is true and `describe()` ends with
"declared by RouterConfig(providers=...) — no lm15 receipts" — and
`lm15 doctor` and `vet` do not list it. When a declared provider earns its
receipts it becomes a registry entry and the declaration is deleted.

Everything else applies unchanged: `api_keys`/env-key lookup and
`MissingCredentialError` naming the declared variable, `base_urls`
overriding the address, the shared transport and `timeouts`, `plan()` with
no key, `adaptations=`, and the async router building the async mirror.

## Customizing the rule table

Rules are data, not callbacks:

```python
from lm15 import LMRouter, RouterConfig
from lm15.router import DEFAULT_RULES, RouteRule

rules = (RouteRule("glm-", "openai-chat", note="in-house vLLM"), *DEFAULT_RULES)
router = LMRouter(config=RouterConfig(rules=rules))
```

First match wins, so prepend to override. Note that a rule can only name
a routable provider (`lm15.router.ADAPTERS` or `CHAT_PRESET_ROUTES`) —
pointing `glm-` at `openai-chat` routes the request to that provider's
URL, which `RouterConfig(base_urls={"openai-chat": ...})` can move for
every `openai-chat` request. A second, differently-addressed instance
of the same provider still means constructing the LM directly (see the
escape hatch above).
