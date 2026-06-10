# Model hydration

Normative contract for exchanging model metadata as canonical JSON and for
hydrating a `ModelRegistry` from installed catalog packages.

## Guardrail (rule, not a suggestion)

**Hydrated data is ADVISORY metadata** — pricing, context windows, capability
hints. It must never change what `build_request` produces. A request built
with or without a hydrated registry is byte-identical. lm15's source must
never name any specific catalog package.

## Canonical JSON for ModelInfo

Serde pair: `model_info_to_dict` / `model_info_from_dict` (`lm15/serde.py`).
Field names in the JSON are exactly the dataclass field names
(`lm15/models.py`). Omission follows `docs/serde-rules.md`: each typed object
omits its own empty optional fields (and boolean flags that are `false`);
opaque payloads (`extensions`, `dimensions`, `provider_data`) are embedded
verbatim and never cleaned.

### ModelInfo (top level)

| field        | type            | notes |
|--------------|-----------------|-------|
| `id`         | string          | required, non-empty |
| `provider`   | string          | required, non-empty |
| `api_family` | string          | required, non-empty |
| `aliases`    | array of string | omitted when empty |
| `origin`     | ModelOrigin     | omitted when it is the default `{"type": "provider"}` |
| `inference`  | InferenceModelInfo | omitted when absent |
| `training`   | TrainingModelInfo  | omitted when absent |
| `extensions` | object (opaque) | omitted when absent; contents verbatim |

`ModelInfo.compat` is runtime-only configuration and is NOT part of the
canonical JSON; it never serializes and always deserializes to `None`.

### ModelOrigin

`type` (string, default `"provider"`), `id` (string|null), `base_model`
(string|null), `provider_data` (opaque object).

### InferenceModelInfo

`input_modalities` / `output_modalities` (arrays of string, default
`["text"]`), `context_window` / `max_output_tokens` (positive int|null),
`supports_reasoning` (bool, omitted when false), `reasoning_efforts`
(array of string), `pricing` (InferencePricing), `extensions` (opaque).

### TrainingModelInfo

`supports_lora` / `supports_full_finetune` (bool, omitted when false),
`trainable_modalities` (array of string), `pricing` (TrainingPricing),
`extensions` (opaque).

### InferencePricing

`input_per_million`, `output_per_million`, `cache_read_per_million`,
`cache_write_per_million` (non-negative number|null), `currency` (string,
default `"USD"`), `dimensions` (opaque object).

### TrainingPricing

`training_tokens_per_million`, `gpu_second` (non-negative number|null),
`currency` (string, default `"USD"`), `dimensions` (opaque object).

Validation lives in the constructors: negative prices, empty modality
strings, non-positive context windows, and empty required strings raise
`ValueError`. `model_info_from_dict` is lenient about absent fields
(defaults above) but constructor validation always applies.

## Entry-point catalog protocol

- Group: `lm15.model_catalogs` (override via
  `ModelRegistry.discover(group=...)`).
- Each entry point loads to a **zero-argument callable** returning an
  iterable of canonical ModelInfo dicts (the JSON above).
- Catalogs are processed **sorted by entry-point name**.
- Duplicate models — same `(provider, id)` key — keep the **first**
  occurrence (first-wins merge).
- A catalog whose callable raises (load error, bad dict, validation error)
  is **skipped** with a `warnings.warn` naming the entry point. Discovery
  must never crash the host application.
- With no catalogs installed, `discover()` returns an empty registry.

`ModelRegistry.from_dicts(dicts)` is the non-entry-point path: it validates
each dict through `model_info_from_dict` and rejects junk by raising.
