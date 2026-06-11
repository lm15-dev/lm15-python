# Structured output

**Problem** — You need a dict, not prose. Every provider can constrain
output to a JSON schema, but each calls the knob something different
(`response_format`, `output_config`, `generationConfig`) and each
accepts a slightly different schema dialect. lm15 gives you one field
and maps it; you still own parsing and validation.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

Write the schema once, in the canonical lm15 shape: a dict with
`type: "json_schema"` and a `schema` key holding plain JSON Schema. Put
it on `Config.response_format`.

```python
import json

from lm15 import Config, LMRouter, Message, Request, ToolChoice, tool

invoice_schema = {
    "type": "json_schema",
    "name": "invoice",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "total": {"type": "number"},
            "currency": {"type": "string", "enum": ["USD", "EUR", "CAD"]},
            "line_items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["vendor", "total", "currency", "line_items"],
        "additionalProperties": False,
    },
}

email = (
    "From: billing@northwind.ca\n"
    "Your order is confirmed. 3x Ergo keyboard ($89 ea), 1x USB hub ($35).\n"
    "Total: $302.00 CAD. Thanks for shopping with Northwind Supply."
)

router = LMRouter()
response = router.complete(Request(
    model="gpt-4.1-mini",
    messages=(Message.user(f"Extract the invoice.\n\n{email}"),),
    config=Config(response_format=invoice_schema),
))
print(response.text)
```
```output
{"vendor":"Northwind Supply","total":302,"currency":"CAD","line_items":["3x Ergo keyboard ($89 ea)","1x USB hub ($35)"]}
```

`response.text` is a JSON string. `parse_json()` turns it into Python
and raises `ValueError` with the raw text in the message if the model
sent anything else.

```python
data = response.parse_json()
print(type(data))
print(data["vendor"], data["total"], data["currency"])
```
```output
<class 'dict'>
Northwind Supply 302 CAD
```

The same schema dict travels to every provider. lm15 rewrites it per
wire format; your code does not change.

```python
for model in ("gpt-4.1-mini", "gemini-3-flash-preview", "claude-sonnet-4-5"):
    r = router.complete(Request(
        model=model,
        messages=(Message.user(f"Extract the invoice.\n\n{email}"),),
        config=Config(response_format=invoice_schema),
    ))
    d = r.parse_json()
    print(f"{model:24} -> {d['vendor']!r}, {d['total']} {d['currency']}, {len(d['line_items'])} items")
```
```output
gpt-4.1-mini             -> 'Northwind Supply', 302.0 CAD, 2 items
gemini-3-flash-preview   -> 'Northwind Supply', 302.0 CAD, 2 items
claude-sonnet-4-5        -> 'Northwind Supply', 302.0 CAD, 2 items
```

Schema enforcement constrains shape, not sense. A model can emit a
total of `-5` in perfectly valid JSON. lm15 is stdlib-only and ships no
validator; write the checks that matter to you and run them on every
parse.

```python
def check_invoice(d):
    problems = []
    for key in ("vendor", "total", "currency", "line_items"):
        if key not in d:
            problems.append(f"missing {key}")
    if not isinstance(d.get("total"), (int, float)) or d.get("total", 0) <= 0:
        problems.append(f"bad total: {d.get('total')!r}")
    if d.get("currency") not in ("USD", "EUR", "CAD"):
        problems.append(f"bad currency: {d.get('currency')!r}")
    return problems

print(check_invoice(data))
print(check_invoice({"vendor": "x", "total": -5, "currency": "GBP"}))
```
```output
[]
['missing line_items', 'bad total: -5', "bad currency: 'GBP'"]
```

The other route: tools-as-extraction. Define the target shape as a
function with `lm15.tool()`, force a call with
`ToolChoice(mode="required")`, and read `tool_calls[0].input` — already
a dict, no string parsing. The function body can be empty; nothing
executes it.

```python
def record_invoice(vendor: str, total: float, currency: str, line_items: list[str]) -> None:
    """Record one extracted invoice.

    Args:
        vendor: Company that issued the invoice.
        total: Grand total, numeric.
        currency: ISO 4217 code.
        line_items: One string per line item.
    """

extract = tool(record_invoice)
r = router.complete(Request(
    model="claude-sonnet-4-5",
    messages=(Message.user(f"Record this invoice.\n\n{email}"),),
    tools=(extract,),
    config=Config(tool_choice=ToolChoice(mode="required")),
))
call = r.tool_calls[0]
print(call.name)
print(json.dumps(call.input, indent=2))
```
```output
record_invoice
{
  "vendor": "Northwind Supply",
  "total": 302.0,
  "currency": "CAD",
  "line_items": [
    "3x Ergo keyboard ($89 ea)",
    …
  ]
}
```

## How it works

`Config.response_format` is a plain `JsonObject`; lm15 validates that
it is JSON-serializable and otherwise leaves it to the provider
adapter. On the wire:

- **OpenAI (Responses)** gets `text.format` with
  `{"type": "json_schema", "name": …, "schema": …}`. The chat-completions
  dialect (Groq, Ollama, vLLM via `OpenAIChatLM`) gets the nested
  `response_format.json_schema` shape instead.
- **Anthropic** gets `output_config.format` with the same schema.
- **Gemini** gets `generationConfig` with
  `responseMimeType: "application/json"` plus the schema. Gemini has two
  schema fields: `responseSchema` (OpenAPI-ish, rejects
  `additionalProperties`) and `responseJsonSchema` (real JSON Schema).
  lm15 picks `responseJsonSchema` when your schema contains
  `additionalProperties`, `responseSchema` otherwise.

If you already hold a provider-native config, pass it through: a dict
with a `text`, `output_config`, or `generationConfig` key is forwarded
verbatim to that provider. That escapes portability — use it only when
you need a dialect feature the canonical shape can't express.

`Response.parse_json()` is `json.loads` on `response.text` plus honest
errors: it refuses non-text responses (tool calls, images) by listing
the part types, and includes a preview of unparseable text. The
`response.json` property is the soft version — `None` on any failure,
indistinguishable from a JSON `null`. Prefer `parse_json()` in
pipelines; `parse_json(default=…)` when you have a fallback.

When to prefer tools-as-extraction:

- The model also has real tools in the request — one mechanism, not two.
- You want the schema derived from a typed Python signature
  (see [recipe 06](06-function-tools.md)) instead of hand-written.
- The provider's structured-output support is weaker than its
  tool-calling (common on open models behind chat-completions servers).

When to prefer `response_format`: pure extraction with no tool
machinery, `strict: True` guarantees on OpenAI, or when you need the
text to *be* the JSON document (logging, piping onward).

## Variations

- **Async mirror.** Same field, same parsing:

  ```python
  from lm15 import AsyncLMRouter
  response = await AsyncLMRouter().complete(req)   # req as above
  data = response.parse_json()
  ```

- **JSON mode without a schema.**
  `Config(response_format={"type": "json_object"})` asks for *some*
  valid JSON object. All three providers honor it; you get no field
  guarantees, so `check_invoice`-style validation is mandatory. Mention
  JSON in the prompt — OpenAI rejects `json_object` requests whose
  messages never say "JSON".
- **`strict: True`** is an OpenAI feature: the schema is compiled into
  a grammar and enforced during decoding, but every field must be
  `required` and `additionalProperties: False`. The Anthropic and
  Gemini adapters forward only `schema`; `strict` and `name` never
  reach those providers.
- **Soft parsing.** On a plain prose response,
  `response.parse_json(default=None)` and `response.json` both return
  `None`; bare `parse_json()` raises `ValueError` with the raw text.
- **Streaming.** `response_format` composes with `stream()`
  ([recipe 05](05-streaming.md)); you receive the JSON document as text
  deltas and must buffer to the end before parsing.

## See also

- [06 — Function tools](06-function-tools.md) — `tool()`, `derive()`, and dispatch.
- [05 — Streaming](05-streaming.md) — buffering deltas before `json.loads`.
- [02 — Conversations](02-conversations.md) — feeding tool results back.
- [../tools-from-functions.md](../tools-from-functions.md) — schema derivation rules.
- [../mapping-rules.md](../mapping-rules.md) — full per-provider wire mappings.
