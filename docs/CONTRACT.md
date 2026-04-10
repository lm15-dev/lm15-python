# LM15 Contract

Machine-readable contract: `spec/contract/v2.json`

Portability guide: `docs/PORTABILITY.md`

## Primary Types

- `LMRequest(model, messages, system?, tools?, config?)`
- `LMResponse(id, model, message, finish_reason, usage, provider?)`
- `Message(role, parts, name?)`
- `Part(type, ...)`
- `DataSource(type=base64|url|file, ...)`
- `StreamEvent(type=start|delta|part_start|part_end|end|error, ...)`
- `AudioFormat(encoding, sample_rate, channels?)`
- `LiveConfig(model, system?, tools?, voice?, input_format?, output_format?, provider?)`
- `LiveClientEvent(type=audio|video|text|tool_result|interrupt|end_audio, ...)`
- `LiveServerEvent(type=audio|text|tool_call|interrupted|turn_end|error, ...)`

## Contract Guarantees

- `messages` non-empty.
- `parts` non-empty.
- Type-specific field validation is enforced.
- `config.provider`, `response.provider`, and `live_config.provider` are passthrough bags.

## Versioning Rule

- Additive changes only:
  - Add new `Part.type` values.
  - Add optional fields.
- Never repurpose existing discriminators.

The current frozen, language-neutral bundle lives in `spec/contract/v2.json`.
