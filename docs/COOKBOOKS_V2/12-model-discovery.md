# Cookbook 12 — Model Discovery and Provider Status

Use `lm15.models()` and `lm15.providers_info()` to inspect what is available before calling `call()`.

---

## List all models (merged live + fallback)

```python
import lm15

all_models = lm15.models(env=".env")
print(len(all_models))
print(all_models[0].provider, all_models[0].id)
```
```output | ✓ 1.3s | 2 vars
186
anthropic claude-3-haiku-20240307
```

`lm15.models()` returns a list of `ModelSpec` objects with normalized fields:

- `id`
- `provider`
- `context_window`
- `max_output`
- `input_modalities`
- `output_modalities`
- `tool_call`
- `structured_output`
- `reasoning`

---

## Filter by provider

```python
openai_models = lm15.models(provider="openai", env=".env")
anthropic_models = lm15.models(provider="anthropic", env=".env")
gemini_models = lm15.models(provider="gemini", env=".env")

print(openai_models[:3])
```

```output | ✓ 153ms | 5 vars
[ModelSpec(id='babbage-002', provider='openai', context_window=None, max_output=None, input_modalities=(), output_modalities=(), tool_call=False, structured_output=False, reasoning=False, raw={'id': 'babbage-002', 'object': 'model', 'created': 1692634615, 'owned_by': 'system'}), ModelSpec(id='chatgpt-image-latest', provider='openai', context_window=None, max_output=None, input_modalities=(), output_modalities=(), tool_call=False, structured_output=False, reasoning=False, raw={'id': 'chatgpt-image-latest', 'object': 'model', 'created': 1765925279, 'owned_by': 'system'}), ModelSpec(id='computer-use-preview', provider='openai', context_window=None, max_output=None, input_modalities=(), output_modalities=(), tool_call=False, structured_output=False, reasoning=False, raw={'id': 'computer-use-preview', 'object': 'model', 'created': 1734655677, 'owned_by': 'system'})]
```

---

## Filter by capabilities

```python ✓
# Models that support tools + reasoning
models = lm15.models(
    supports={"tools", "reasoning"},
    env=".env",
)

# Models that accept images as input
vision_models = lm15.models(
    input_modalities={"image"},
    env=".env",
)

# Models that can return audio
audio_models = lm15.models(
    output_modalities={"audio"},
    env=".env",
)
```

---

## Force live refresh

```python
fresh = lm15.models(live=True, refresh=True, timeout=3.0, env=".env")
print(len(fresh))
```

`refresh=True` bypasses cache for this call.

---

## Provider status and configuration

```python
info = lm15.providers_info(env=".env")
for name, meta in info.items():
    print(name, meta)
```

Each provider entry includes:

- `env_keys`: accepted environment variable names
- `configured`: whether an API key is currently available
- `model_count`: number of discoverable models

---

## Practical pattern: pick model from discovered list

```python
import lm15

candidates = lm15.models(provider="openai", supports={"tools"}, env=".env")
model_id = candidates[0].id if candidates else "gpt-4.1-mini"

resp = lm15.call(model_id, "say ok", env=".env")
print(resp.text)
```

This avoids hardcoding stale model IDs.
