# Provider passthrough

**Problem** — A provider ships a knob lm15's canonical `Config` does not
have, or returns metadata the canonical `Response` does not model. You
need both without forking your code off lm15's types — and without the
escape hatch quietly becoming the API.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

`Config.extensions` is a dict of raw provider fields. The adapter merges
it into the request payload after the canonical fields — provider
spelling, your responsibility. You can see exactly what goes on the wire
without spending a token: every LM exposes `build_request`.

```python
import json

from lm15 import Config, LMRouter, Message, Request, tool

router = LMRouter()

config = Config(max_tokens=300, extensions={"max_tool_calls": 1, "service_tier": "default"})
request = Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),), config=config)

wire = router.lm("gpt-4.1-mini").build_request(request, stream=False)
print(wire.url)
print(json.dumps(json.loads(wire.body), indent=2))
```
```output
https://api.openai.com/v1/responses
{
  "model": "gpt-4.1-mini",
  "input": […],
  "stream": false,
  "max_output_tokens": 300,
  "max_tool_calls": 1,
  "service_tier": "default"
}
```

`max_tokens` became `max_output_tokens` — canonical, translated. The two
extension keys went through verbatim. lm15 does not validate them; a
typo'd key is the provider's 400, not lm15's.

`max_tool_calls` is one of the four blessed extension knobs (below). It
caps total tool invocations across a Responses API run. Live:

```python
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "22 C, clear"

def get_time(city: str) -> str:
    """Current local time in a city."""
    return "14:05"

weather, clock = tool(get_weather), tool(get_time)

request = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("What is the weather AND the local time in Oslo?"),),
    tools=(weather, clock),
    config=Config(extensions={"max_tool_calls": 1}),
)
response = router.complete(request)
print([(c.name, c.input) for c in response.tool_calls])
print(response.finish_reason)
```
```output
[('get_weather', {'city': 'Oslo'}), ('get_time', {'city': 'Oslo'})]
tool_call
```

Honest note: this run still emitted two parallel function calls — the
cap governs invocations the provider executes server-side, not how many
function calls the model emits in one turn (that is canonical
`ToolChoice(parallel=False)`). How do you know the knob was accepted at
all? The response echoes it — which is what `Response.provider_data`
is for. It holds the raw provider response body, verbatim:

```python
print(response)
pd = response.provider_data
print(len(pd), "keys")
for key in ("id", "model", "max_tool_calls", "service_tier", "status"):
    print(key, "=", pd.get(key))
```
```output
Response(
    message=Message(role='assistant', parts=(ToolCallPart(…), ToolCallPart(…)), …),
    …
    provider_data=<dict: 35 keys>,
)
35 keys
id = resp_04c181fea396…
model = gpt-4.1-mini-2025-04-14
max_tool_calls = 1
service_tier = default
status = completed
```

The repr elides it (`<dict: 35 keys>`) because raw payloads are noise
until you ask for them; the data itself is complete.

The blessed knob on the Anthropic side is the `thinking` object's
`display` field — canonical `Reasoning` maps to `{type, budget_tokens}`
only, because no other provider has a display concept. The whole wire
object goes in `extensions`:

```python
request = Request(
    model="claude-sonnet-4-5",
    messages=(Message.user("Why is the sky blue? One sentence."),),
    config=Config(
        max_tokens=4096,
        extensions={"thinking": {"type": "enabled", "budget_tokens": 2048, "display": "summarized"}},
    ),
)
wire = router.lm("claude-sonnet-4-5").build_request(request, stream=False)
print(json.dumps(json.loads(wire.body)["thinking"], indent=2))
```
```output
{
  "type": "enabled",
  "budget_tokens": 2048,
  "display": "summarized"
}
```

## How it works

`extensions` and `provider_data` are deliberately asymmetric — different
directions, different ownership, different names:

- `extensions` (request): you wrote it; you are asking the adapter to
  forward it. Merged into the payload *after* canonical fields, so an
  extension key can override what lm15 built — useful (the `thinking`
  example) and sharp.
- `provider_data` (response): the provider produced it; the adapter
  preserves it verbatim. Response fields lm15 could not map are also
  flagged here under `_lm15_unmapped` (path + type), so nothing is
  silently dropped.

A single name like `extra` would suggest the two round-trip. They do
not: echoing `provider_data` back as `extensions` is almost always a
bug. See [design rationale](../design-rationale.md).

The discipline is INV-049 in the spec: exactly four `extensions`
passthrough keys are *blessed* — permanent, not burn-down debt — because
each is provider syntax for a capability no canonical key can express
without inventing semantics the other providers lack: Anthropic
`thinking` display (two spec cases), OpenAI `max_tool_calls`, and OpenAI
`include` for code-interpreter outputs (provider-executed tool traces
live in `provider_data` by mapping rule MAP-1, never in canonical
parts — see [mapping rules](../mapping-rules.md)). Everything else you
put in `extensions` is your contract with the provider, not lm15's.
Promoting a blessed key to canonical later is an additive spec change.

## Variations

- **Async mirror.** `AsyncLMRouter` is identical: `await
  router.complete(request)`; `extensions` and `provider_data` are plain
  data, no sync/async split.
- **Reserved keys.** Each adapter pops the keys it interprets itself
  before forwarding: all three reserve `prompt_caching`; OpenAI also
  reserves `cache` and `compat`/`openai_compat`/`openai_responses_compat`; Gemini
  reserves `output` (modality switch) and `transport` (live sessions).
  Anything else passes through untouched.
- **Shapes differ per provider.** The same recipe, three raw payloads:

  ```python
  for model in ("claude-sonnet-4-5", "gemini-3-flash-preview"):
      r = router.complete(Request(model=model, messages=(Message.user("Say ok."),)))
      print(model, sorted(r.provider_data.keys()))
  ```
  ```output
  claude-sonnet-4-5 ['content', 'id', 'model', 'role', 'stop_details', 'stop_reason', 'stop_sequence', 'type', 'usage']
  gemini-3-flash-preview ['candidates', 'modelVersion', 'responseId', 'usageMetadata']
  ```

  Code that reads `provider_data` is provider-specific by construction.
  Branch on `router.resolve(model).provider`, not on key sniffing.
- **Gemini nesting.** Gemini extensions merge at the payload top level;
  a `generationConfig` sub-field must be passed as the full nested
  object, e.g. `extensions={"generationConfig": {"responseLogprobs": True}}` —
  and it replaces the `generationConfig` lm15 built, canonical knobs
  included. Prefer canonical fields; reach for `extensions` only for
  what they cannot say.
- **Other endpoints.** `EmbeddingRequest`, `BatchRequest`,
  `ImageRequest`, and `LiveConfig` carry the same `extensions` field,
  and their responses the same `provider_data` (see
  [recipe 12](12-embeddings-batch-generation.md)).

## See also

- [04 — Controlling generation](04-generation-config.md) — the canonical
  knobs you should exhaust first.
- [06 — Function tools](06-function-tools.md) — `tool()` and the
  tool-call loop you own.
- [07 — Built-in tools](07-builtin-tools.md) — provider-executed tools,
  whose traces land in `provider_data`.
- [Design rationale](../design-rationale.md) — the asymmetry argument in
  full.
- [Mapping rules](../mapping-rules.md) — what becomes a canonical part
  and what stays raw.
