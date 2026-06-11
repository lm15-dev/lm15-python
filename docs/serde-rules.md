# Canonical JSON serialization rules

Normative rules for the lm15 canonical JSON wire format. Implementations in
every language must produce byte-identical canonical JSON for the same value
(after key sorting), regardless of which serialization entry point is used.

## Omission rule

There is exactly one omission rule, applied at exactly one place:

1. **Each typed serializer omits its own empty optional fields.** When a typed
   object (Part, Tool, Config, Request, Response, StreamEvent, …) is
   serialized, a field whose value is `null`, `""`, `[]`, or `{}` is omitted
   from that object's JSON — by the serializer for that type, at the top level
   of that object only.
2. **Nested values are embedded verbatim.** A serialized value placed inside a
   larger object (a Part inside a Message, a Message inside a Request, a
   Config inside a Request) is embedded exactly as its own serializer produced
   it. Enclosing serializers never re-clean, re-order, or otherwise rewrite
   what an inner serializer emitted.
3. **Opaque payloads are NEVER mutated.** The contents of user- or
   provider-owned JSON — tool call `input`, FunctionTool `parameters`
   (JSON Schema), config `extensions`, `response_format`, builtin tool
   `config`, `provider_data`, continuation `data` — round-trip exactly.
   Empty strings, empty objects, empty arrays, and explicit nulls inside
   these payloads are user data, not noise: `{"properties": {}}` stays
   `{"properties": {}}`.

Consequences:

- The same value has exactly one canonical wire form. Serializing a
  `ToolCallPart` directly and serializing it inside `request_to_dict` produce
  byte-identical JSON (`tests/test_omission_rule.py` pins this).
- Absent and empty are equivalent ONLY for a typed object's own optional
  fields, and only at the moment that object is serialized. Inside opaque
  payloads, absent and empty are distinct and significant.

Required fields are always emitted, even when empty — e.g. `TextPart.text` is
emitted as `""` when empty, `ToolResultPart.content` is emitted as `[]`
when empty, and `FunctionTool.parameters` is emitted as `{}` when it is the
explicit empty object (INV-033: it is an opaque payload, so `{}` round-trips
verbatim) — because those fields are part of the type's shape, not optional.

## Number rule

Every numeric field in the canonical model has a DECLARED JSON number type.
The canonical wire form follows the declaration, never the Python literal the
caller happened to type: a value has exactly one wire form, reproducible from
any language.

1. **Float fields always serialize as JSON floats** (`1.0`, never `1`):
   - `Config.temperature`, `Config.top_p`
   - `InferencePricing.input_per_million`, `.output_per_million`,
     `.cache_read_per_million`, `.cache_write_per_million`
   - `TrainingPricing.training_tokens_per_million`, `.gpu_second`
   - `EmbeddingResponse.vectors` elements
   - `retry_after` on lm15 errors (rate-limit metadata, not serde, but
     float-typed under the same rule)
2. **Int fields always serialize as JSON ints** (`2`, never `2.0`):
   - `Config.max_tokens`, `Config.top_k`
   - `Reasoning.thinking_budget`, `.total_budget`
   - `CacheConfig.prefix_until_index`
   - every `Usage` token counter
   - `InferenceModelInfo.context_window`, `.max_output_tokens`
   - `AudioFormat.sample_rate`, `.channels`
   - every Delta `part_index`
3. **Constructors coerce same-valued cross-type input.** Int `1` for a float
   field becomes `1.0`; float `2.0` for an int field becomes `2`. A
   non-integral float for an int field (`top_k=2.5`) is REJECTED, never
   rounded. `bool` never coerces to either number type — `True` is not `1`.
4. **Opaque payloads are untouched, as always.** Inside `extensions`, tool
   call `input`, FunctionTool `parameters`, `response_format`, builtin tool
   `config`, `provider_data`, continuation `data`, and pricing `dimensions`,
   numbers round-trip exactly as written: `{"x": 1}` stays int `1`. The
   Number rule applies to TYPED fields only.

Provider wire dialects are a different layer: an adapter may format a
canonical float in whatever number form the provider's API expects (e.g. the
Gemini adapter emits integral `generationConfig.temperature`/`topP` values in
proto3-JSON integer form, matching live captures). The canonical JSON form is
fixed by the declarations above.

History: before 2026-06-10, `Config(temperature=1)` serialized as `1` while
`Config(temperature=1.0)` serialized as `1.0` — the canonical form depended on
the caller's Python literal, which non-Python ports cannot reproduce. The rule
above replaces that behavior.

History: before 2026-06-09, `_clean_mapping` recursed into nested structures,
which stripped empties inside opaque payloads at the request/response level
but not at the part level — the same value had two wire forms depending on
entry point, and user data was silently destroyed. The rule above replaces
that behavior.
