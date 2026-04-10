# LM15 Contract

Machine-readable contract: `spec/contract/v1.json`

Portability guide: `docs/PORTABILITY.md`

## Primary Types

- `LMRequest(model, messages, system?, tools?, config?)`
- `LMResponse(id, model, message, finish_reason, usage, provider?)`
- `Message(role, parts, name?)`
- `Part(type, ...)`
- `DataSource(type=base64|url|file, ...)`
- `StreamEvent(type=start|delta|part_start|part_end|end|error, ...)`

## Contract Guarantees

- `messages` non-empty.
- `parts` non-empty.
- Type-specific field validation is enforced.
- `config.provider` and `response.provider` are passthrough bags.

## Versioning Rule

- Additive changes only:
  - Add new `Part.type` values.
  - Add optional fields.
- Never repurpose existing discriminators.

The frozen, language-neutral bundle lives in `spec/contract/v1.json`.
