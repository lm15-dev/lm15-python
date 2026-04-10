# lm15 Routing Design — Model Name → Provider Resolution

## Problem

lm15's core promise is "change the model string, everything works." Routing — mapping a model name to the adapter that serves it — is the front door of the library. If it fails, the promise is broken.

The v2 routing system has three disconnected pieces:
1. A global static registry with three hardcoded prefixes (`gpt`, `claude`, `gemini`)
2. Adapter registration (`UniversalLM.adapters`) that knows which providers are available
3. Live discovery (`discovery.py`) that knows which models exist

These don't talk to each other. The registry doesn't know what adapters are registered. Adapters don't participate in routing. Discovery results aren't used for routing. The result: mainstream models like `o4-mini` fail without `provider=`, and community plugins can't route at all.

---

## Design Principles

1. **Routing is a function of what's registered.** If only Anthropic is configured, nothing routes to OpenAI. Routing reflects reality, not possibility.
2. **Adapters own their routing declarations.** The adapter knows what it serves. Routing asks the adapters, not a separate registry.
3. **No silent wrong routing.** Sending a prompt to the wrong provider is a data leak. Ambiguity is an error, not a guess.
4. **Common models are instant.** No I/O for `gpt-4.1-mini` or `claude-sonnet-4-5`. The fast path is a string scan.
5. **Unusual models self-heal.** When prefix matching fails, lazy discovery catches it without user action.
6. **`provider=` always wins.** Explicit overrides everything. This is the escape hatch and it is always available.

---

## Architecture

### Routing lives in `UniversalLM`

Routing moves out of `capabilities.py` (global static registry) and into `UniversalLM`, where the adapters live. This is the right home because routing depends on what's registered.

The global `REGISTRY` in `capabilities.py` is deleted. `CapabilityResolver` is reduced to capability lookup only (context window, modalities, features) — it no longer does routing.

### Adapter-declared prefixes

Each adapter declares the model name prefixes it handles, as part of its manifest:

```python
@dataclass(slots=True, frozen=True)
class ProviderManifest:
    provider: str
    supports: EndpointSupport
    prefixes: tuple[str, ...]       # NEW — model name prefixes this adapter handles
    env_keys: tuple[str, ...] = ()
    auth_modes: tuple[str, ...] = ()
    enterprise_variants: tuple[str, ...] = ()
```

Core adapters declare their prefixes:

```python
# OpenAI
ProviderManifest(
    provider="openai",
    prefixes=("gpt-", "o1-", "o3-", "o4-", "chatgpt-", "dall-e-", "ft:gpt-"),
    ...
)

# Anthropic
ProviderManifest(
    provider="anthropic",
    prefixes=("claude-",),
    ...
)

# Gemini
ProviderManifest(
    provider="gemini",
    prefixes=("gemini-",),
    ...
)
```

Community plugins declare their own:

```python
# lm15-x-mistral
ProviderManifest(
    provider="mistral",
    prefixes=("mistral-", "codestral-", "pixtral-"),
    ...
)
```

---

## Resolution Algorithm

Three phases, tried in order. First match wins. `provider=` bypasses all of them.

### Phase 1: Prefix scan (instant, no I/O)

Scan registered adapters' prefixes against the model name. This is the fast path — a string comparison loop, zero network calls.

```python
def _resolve_by_prefix(self, model: str) -> str | None:
    lower = model.lower()
    matches: list[str] = []
    for name, adapter in self.adapters.items():
        for prefix in adapter.manifest.prefixes:
            if lower.startswith(prefix):
                matches.append(name)
                break  # one match per adapter is enough

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousModelError(
            f"Model '{model}' matches multiple providers: {matches}\n\n"
            f"  Fix: specify the provider explicitly:\n"
            f"    lm15.call('{model}', ..., provider='{matches[0]}')\n"
        )
    return None  # no match — fall through to phase 2
```

### Phase 2: Lazy live discovery (one-time, cached)

If no prefix matches, query each registered adapter's provider API for its model list. Cache the results for the process lifetime.

```python
def _resolve_by_discovery(self, model: str) -> str | None:
    self._ensure_model_index()  # fetches once, caches forever
    provider = self._model_index.get(model)
    if provider and provider in self.adapters:
        return provider
    return None
```

`_ensure_model_index()` calls each provider's `/models` endpoint (the same fetchers already in `discovery.py`), builds a `{model_id: provider}` dict, and caches it. This runs at most once per process, only when prefix matching fails.

Cost: ~200-500ms total across providers, one time. Subsequent lookups are a dict lookup.

### Phase 3: Error with guidance

If both phases fail, raise with actionable information:

```python
def _resolve_error(self, model: str) -> NoReturn:
    registered = list(self.adapters.keys())
    raise UnsupportedModelError(
        f"Can't determine provider for model '{model}'\n\n"
        f"  Registered providers: {', '.join(registered)}\n\n"
        f"  To fix, do one of:\n"
        f"    1. Specify the provider: lm15.call('{model}', ..., provider='openai')\n"
        f"    2. Check available models: lm15.models()\n"
        f"    3. Verify the model name is correct\n"
    )
```

### Combined

```python
class UniversalLM:
    def resolve_provider(self, model: str) -> str:
        # Phase 1: prefix scan
        result = self._resolve_by_prefix(model)
        if result:
            return result

        # Phase 2: lazy discovery
        result = self._resolve_by_discovery(model)
        if result:
            return result

        # Phase 3: error
        self._resolve_error(model)
```

---

## Collision Handling

### What collisions look like

1. **OpenAI-compatible local servers.** Ollama registers with `prefixes=("gpt-", "llama-")` alongside real OpenAI. Both claim `gpt-*`.
2. **Provider forks.** Azure OpenAI and OpenAI both serve `gpt-4.1-mini`.
3. **Overly broad plugin prefixes.** A plugin declares `prefixes=("o",)` which overlaps with OpenAI's `("o1-", "o3-", "o4-")`.

### Behavior

| Situation | Behavior |
|-----------|----------|
| One prefix match | Route to it |
| Multiple prefix matches | `AmbiguousModelError` — user adds `provider=` |
| No prefix match, discovery match | Route to it |
| No match at all | `UnsupportedModelError` with guidance |
| `provider=` given | Skip all resolution |

### Registration-time warning

Detect overlapping prefixes when an adapter is registered, so the developer sees the issue during setup rather than at call time:

```python
def register(self, adapter: LMAdapter) -> None:
    for prefix in adapter.manifest.prefixes:
        for name, existing in self.adapters.items():
            for existing_prefix in existing.manifest.prefixes:
                if prefix.startswith(existing_prefix) or existing_prefix.startswith(prefix):
                    import warnings
                    warnings.warn(
                        f"Prefix overlap: '{adapter.manifest.provider}' prefix '{prefix}' "
                        f"overlaps with '{name}' prefix '{existing_prefix}'. "
                        f"Models matching both will require provider= to resolve.",
                        stacklevel=2,
                    )
    self.adapters[adapter.manifest.provider] = adapter
```

The warning at registration time is informational. The error at call time is enforced. This means: you can register overlapping adapters (it's not a fatal setup error), but using an ambiguous model name without `provider=` is.

### Why not longest-prefix-match

HTTP routers use longest-prefix-match to resolve overlaps. We don't, because model names aren't hierarchical paths. `"o4-mini"` matching `"o4-"` over `"o"` feels right, but the rule breaks intuition in edge cases and introduces a silent priority system the user can't predict. Errors are clearer than heuristics.

### Why not first-registered-wins

Priority by registration order is invisible. The user doesn't control (or know) the order adapters are registered in, especially when plugins auto-discover via entry points. A silent priority based on load order is a source of bugs that are impossible to diagnose.

---

## Discovery Cache

### Lifecycle

```
First call to resolve_provider("o4-mini")
  → Phase 1: prefix scan — no match
  → Phase 2: _ensure_model_index()
       → For each registered adapter, fetch /models endpoint
       → Build {model_id: provider_name} dict
       → Cache in self._model_index
       → Return "openai"
  → Cached for all future calls

Second call to resolve_provider("o4-mini")
  → Phase 1: prefix scan — no match
  → Phase 2: self._model_index["o4-mini"] → "openai" (cache hit)
```

### Cache invalidation

The cache lives on the `UniversalLM` instance. It's invalidated when:
- A new adapter is registered (`register()` clears `_model_index`)
- The user explicitly requests refresh (`lm15.models(refresh=True)`)
- A new `UniversalLM` instance is created

There is no TTL. Within a single process, provider model lists don't change meaningfully. If a user needs to pick up a newly deployed model mid-process, they call `lm15.models(refresh=True)` or pass `provider=`.

### Timeout

Discovery fetches use a short timeout (3s per provider, concurrent where possible). If a provider is unreachable, its models are simply absent from the index. This is not an error — it means those models require `provider=`, which is already the fallback.

---

## Plugin Integration

A community plugin gets routing for free by declaring prefixes:

```toml
# pyproject.toml
[project.entry-points."lm15.providers"]
mistral = "lm15_x_mistral:build_adapter"
```

```python
# lm15_x_mistral.py
class MistralAdapter:
    manifest = ProviderManifest(
        provider="mistral",
        prefixes=("mistral-", "codestral-", "pixtral-"),
        supports=EndpointSupport(complete=True, stream=True),
        env_keys=("MISTRAL_API_KEY",),
    )
    # ...
```

After installation and registration:

```python
import lm15

lm15.configure(env=".env")

# Just works — "mistral-" prefix routes to the plugin
r = lm15.call("mistral-large-latest", "Hello.")
```

No core code change. No PR to lm15. The plugin brings its own routing.

---

## What This Replaces

| v2 | v3 |
|----|-----|
| `REGISTRY` in `capabilities.py` — global, static, 3 prefixes | Deleted. Routing derived from registered adapters. |
| `CapabilityResolver.resolve_provider()` — standalone function | `UniversalLM.resolve_provider()` — method on the client that owns the adapters |
| `hydrate_models_dev_catalog` — opt-in, off by default, used for routing | models.dev used only for capability enrichment, not routing |
| `resolve_provider()` as a free function in `capabilities.py` | Instance method on `UniversalLM` — routing depends on instance state |

`CapabilityResolver` continues to exist for capability queries (context window, modalities, features) and can still be hydrated from models.dev. It just no longer does routing.

---

## `ProviderManifest.prefixes` Guidelines for Adapter Authors

1. **Use trailing hyphens.** `"gpt-"` not `"gpt"`. This avoids `"gpt"` matching a hypothetical `"gptx-turbo"` from another provider.
2. **Declare all known prefixes.** OpenAI models use `gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt-`, `dall-e-`, `ft:gpt-`. List them all.
3. **Be specific.** `"mistral-"` is better than `"mi"`. Narrower prefixes reduce collision risk.
4. **Update when the provider adds naming conventions.** When OpenAI ships `o5-`, add `"o5-"` to the manifest. This is an adapter-level change, not a routing system change.
5. **For catch-all adapters** (e.g., an OpenAI-compatible proxy that serves anything), use an empty `prefixes=()` and require `provider=`. A catch-all adapter that claims broad prefixes will collide with everything.

---

## Implementation Checklist

1. Add `prefixes: tuple[str, ...]` to `ProviderManifest`
2. Declare prefixes on `OpenAIAdapter.manifest`, `AnthropicAdapter.manifest`, `GeminiAdapter.manifest`
3. Add `resolve_provider()`, `_resolve_by_prefix()`, `_resolve_by_discovery()` to `UniversalLM`
4. Add `_model_index` cache and `_ensure_model_index()` to `UniversalLM`
5. Add overlap warning to `UniversalLM.register()`
6. Add `AmbiguousModelError` to `errors.py`
7. Delete `REGISTRY` from `capabilities.py`
8. Remove routing from `CapabilityResolver` (keep capability lookup)
9. Update `api.py` to use `client.resolve_provider()` instead of the free function
10. Update plugin docs and adapter guide with `prefixes` guidance
11. Tests: prefix match, collision error, lazy discovery fallback, `provider=` override, plugin routing
