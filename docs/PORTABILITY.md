# Portability Spec

LM15 now ships a frozen, language-neutral contract + fixture bundle for ports.

## Files

- Universal contract: `spec/contract/v2.json`
- Complete fixtures: `spec/fixtures/v2/complete.json`
- Tool-call fixtures: `spec/fixtures/v2/tool_call.json`
- Stream fixtures: `spec/fixtures/v2/stream.json`
- Error fixtures: `spec/fixtures/v2/errors.json`
- Live/session fixtures: `spec/fixtures/v2/live.json`

## What is frozen

### 1. Universal contract

`spec/contract/v2.json` defines the portable shape of:

- `DataSource`
- `Part`
- `Message`
- `Tool` as a tagged union of function and builtin tools
- `ToolConfig`
- `ReasoningConfig`
- `Config`
- `LMRequest`
- `Usage`
- `LMResponse`
- `PartDelta`
- `StreamEvent`
- `AudioFormat`
- `LiveConfig`
- `LiveClientEvent`
- `LiveServerEvent`

It also freezes discriminator values:

- roles
- part types
- data source types
- finish reasons
- stream event types
- part delta types

### 2. Canonical fixture JSON

Fixtures use a canonical JSON encoding of the universal types:

- snake_case field names
- tuples serialized as arrays
- optional `null` / empty-string / empty-container fields omitted
- normalized fixture outputs exclude provider passthrough bags unless explicitly under test

### 3. Provider normalization fixtures

The fixture bundle captures:

- provider wire response -> normalized `LMResponse`
- provider stream wire events -> normalized `StreamEvent`
- provider and HTTP error inputs -> canonical lm15 errors
- live/session config and event shapes -> canonical live transport types

For completions, transport is intentionally abstracted away: REST+SSE, blocking REST, and WebSocket completion adapters all normalize to the same `StreamEvent` surface. That means the completion fixtures stay transport-agnostic. The separate live fixture bundle covers the persistent session pattern. See `docs/DESIGN_TRANSPORT.md` for the transport model behind this split.

## Versioning rule

The portability spec is additive-only:

- add optional fields
- add new discriminator values
- never repurpose existing discriminator values

Ports should ignore unknown optional fields for forward compatibility.

## v2 design shifts

Compared to the earlier draft contract, v2 is more portability-first:

- `Part` stays a true discriminated union in the spec
- `Tool` is now a discriminated union of function and builtin variants
- open extension bags use recursive JSON value types instead of implicit `Any`
- `reasoning` is a typed config object, not an unstructured bag
- shared error payloads use a named `ErrorInfo` type

## Recommended port workflow

1. Implement the universal types from `spec/contract/v2.json`
2. Make the port pass all fixture bundles under `spec/fixtures/v2/`
3. Only then add runtime sugar for that language
4. Keep provider-specific wire types out of the portable surface

## Python helpers

Python uses `lm15/serde.py` to serialize and load the canonical fixture JSON.
That file is the reference for fixture encoding, not a second contract.
