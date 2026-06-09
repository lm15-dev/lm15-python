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
emitted as `""` when empty, and `ToolResultPart.content` is emitted as `[]`
when empty, because those fields are part of the type's shape, not optional.

History: before 2026-06-09, `_clean_mapping` recursed into nested structures,
which stripped empties inside opaque payloads at the request/response level
but not at the part level — the same value had two wire forms depending on
entry point, and user data was silently destroyed. The rule above replaces
that behavior.
