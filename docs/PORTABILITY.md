# Portability Spec

LM15 now ships a frozen, language-neutral contract + fixture bundle for ports.

## Files

- Universal contract: `spec/contract/v1.json`
- Complete fixtures: `spec/fixtures/v1/complete.json`
- Tool-call fixtures: `spec/fixtures/v1/tool_call.json`
- Stream fixtures: `spec/fixtures/v1/stream.json`
- Error fixtures: `spec/fixtures/v1/errors.json`

## What is frozen

### 1. Universal contract

`spec/contract/v1.json` defines the portable shape of:

- `DataSource`
- `Part`
- `Message`
- `Tool`
- `ToolConfig`
- `Config`
- `LMRequest`
- `Usage`
- `LMResponse`
- `PartDelta`
- `StreamEvent`

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

## Versioning rule

The portability spec is additive-only:

- add optional fields
- add new discriminator values
- never repurpose existing discriminator values

Ports should ignore unknown optional fields for forward compatibility.

## Recommended port workflow

1. Implement the universal types from `spec/contract/v1.json`
2. Make the port pass all fixture bundles under `spec/fixtures/v1/`
3. Only then add runtime sugar for that language
4. Keep provider-specific wire types out of the portable surface

## Python helpers

Python uses `lm15/serde.py` to serialize and load the canonical fixture JSON.
That file is the reference for fixture encoding, not a second contract.
