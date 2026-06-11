# Controlling generation

**Problem** — Every provider spells sampling knobs differently:
`max_tokens` vs `max_output_tokens` vs `maxOutputTokens`, `stop` vs
`stop_sequences` vs `stopSequences`. You want one place to set them, and
you want to know which provider ignores which knob instead of finding
out in production.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

`Config` holds the universal generation parameters. The same instance
drives every provider; each adapter translates field names on the wire.

```python
import json

from lm15 import Config, LMRouter, Message, Request, config_to_dict

router = LMRouter()
config = Config(temperature=0.2, max_tokens=200)

for model in ("gpt-4.1-mini", "claude-sonnet-4-5", "gemini-3-flash-preview"):
    request = Request(
        model=model,
        messages=(Message.user("One-line haiku about rain."),),
        config=config,
    )
    print(model, "->", router.complete(request).text)
```
```output
gpt-4.1-mini -> Silent drops descend,
whispering earth’s soft secrets—
rain’s gentle embrace.
claude-sonnet-4-5 -> Autumn rain falling—
puddles mirror gray sky's tears.
…
gemini-3-flash-preview -> Soft drops fall from
```

Gemini's answer is cut short: `gemini-3-flash-preview` is a reasoning
model and its `maxOutputTokens` budget covers thinking tokens too, so
200 tokens can be spent before the visible text finishes. Honest knob,
provider-specific accounting.

An all-default `Config` serializes to `{}` — lm15 sends only what you
set, never a default it invented:

```python
print(config_to_dict(Config()))
print(config_to_dict(config))
print(config_to_dict(Config(top_p=0.9, stop=("END",))))
```
```output
{}
{'max_tokens': 200, 'temperature': 0.2}
{'top_p': 0.9, 'stop': ['END']}
```

When the model hits `max_tokens`, the text stops mid-sentence and
`finish_reason` says so:

```python
response = router.complete(Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Explain rain in two paragraphs."),),
    config=Config(max_tokens=24),
))
print(response.text)
print(response.finish_reason)
print(response.usage)
```
```output
Rain is a form of precipitation that occurs when water vapor in the atmosphere condenses into droplets heavy enough to fall to the
length
Usage(input_tokens=13, output_tokens=24, total_tokens=37, …)
```

Stop sequences end generation the moment a string appears. The match is
consumed — `"5"` never reaches the output:

```python
counting = Request(
    model="gemini-3-flash-preview",
    messages=(Message.user("Count from 1 to 10, one number per line, digits only."),),
    config=Config(stop=("5",), temperature=0.0),
)
response = router.complete(counting)
print(repr(response.text), response.finish_reason)
```
```output
'1\n2\n3\n4\n' stop
```

The same `Config` object works against an entirely different wire
protocol — here Chat Completions instead of Gemini's `generateContent`:

```python
response = router.complete(Request(
    model="openai_chat:gpt-4.1-mini",
    messages=counting.messages,
    config=counting.config,
))
print(repr(response.text), response.finish_reason)
```
```output
'1  \n2  \n3  \n4  \n' stop
```

`response_format` constrains output to a JSON schema. One canonical
shape, three wire translations:

```python
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}
fmt = {"type": "json_schema", "name": "capital", "schema": schema}
for model in ("gpt-4.1-mini", "claude-sonnet-4-5", "gemini-3-flash-preview"):
    response = router.complete(Request(
        model=model,
        messages=(Message.user("Capital of Japan?"),),
        config=Config(response_format=fmt),
    ))
    print(model, json.loads(response.text))
```
```output
gpt-4.1-mini {'city': 'Tokyo', 'country': 'Japan'}
claude-sonnet-4-5 {'city': 'Tokyo', 'country': 'Japan'}
gemini-3-flash-preview {'city': 'Tokyo', 'country': 'Japan'}
```

## How it works

`Config` is a frozen dataclass with typed universal fields: `max_tokens`,
`temperature`, `top_p`, `top_k`, `stop`, `response_format`, plus
`tool_choice`, `reasoning`, `cache`, and `extensions` covered in later
recipes. Validation happens at construction — `temperature=-1` or
`top_p=2` raises before any network call.

Each provider adapter maps fields to its own wire names: `max_tokens`
becomes `max_output_tokens` (OpenAI Responses), `max_tokens`
(Anthropic, Chat Completions), or `maxOutputTokens` (Gemini); `stop`
becomes `stop_sequences` (Anthropic) or `stopSequences` (Gemini).
`response_format` in the canonical
`{"type": "json_schema", "name": …, "schema": …}` shape is translated to
OpenAI's `text.format`, Anthropic's `output_config`, and Gemini's
`responseSchema`/`responseJsonSchema` in `generationConfig`. You can also pass a provider-native shape; it is
forwarded as-is.

Unset fields are not sent. `config_to_dict(Config())` is `{}`, and the
wire payload contains no sampling keys at all — the provider's own
defaults apply, not lm15's opinion of them. The one exception:
Anthropic's API requires `max_tokens`, so the adapter sends 1024 when
you set nothing. That is a wire requirement, not a hidden default for
other providers.

## Variations

- **What each provider ignores.** The OpenAI Responses adapter sends no
  `top_k` and no `stop` — the Responses API has neither parameter. Need
  stop sequences with OpenAI? Use `openai_chat:` as above. `top_k`
  reaches only Anthropic and Gemini.
- **Anthropic always gets `max_tokens`.** The API rejects requests
  without it; lm15 sends 1024 unless you choose. Set it explicitly for
  long answers.
- **Reasoning models bend the budget.** For `gemini-3-flash-preview`
  (and OpenAI o-series), `max_tokens` caps thinking plus visible text.
  See [recipe 10](10-audio-video-reasoning.md) for `Config(reasoning=…)`.
- **`stop` coerces.** `Config(stop="END")` becomes `("END",)`; tuples
  preferred.
- **Async mirror.** Identical `Config`, `await AsyncLMRouter().complete(request)`.
- **Provider-only knobs** (`seed`, `frequency_penalty`, …) go in
  `Config(extensions=…)` — [recipe 16](16-provider-passthrough.md).

## See also

- [03 — System & developer prompts](03-system-prompts.md)
- [05 — Streaming](05-streaming.md)
- [08 — Structured output](08-structured-output.md) — `response_format` in depth
- [16 — Provider passthrough](16-provider-passthrough.md)
- [Using the router](../using-the-router.md)
