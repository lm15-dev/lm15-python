# Function tools: define & dispatch

**Problem** — You want the model to call your Python functions, but
hand-writing JSON Schema for every signature is tedious and lm15 has no
agent loop to do the calling for you. The recipe: derive the schema from
the function, send the call, run the function yourself, feed the result
back.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

`tool(fn)` reads a function's signature, type hints, and docstring and
returns a frozen `FunctionTool`. It does not wrap, register, or execute
anything — you keep `fn`.

```python
import json
from pprint import pprint
from typing import Literal

from lm15 import (
    Config, FunctionTool, LMRouter, Message, Request, ToolChoice, derive, tool,
)

def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
        unit: Temperature unit.
    """
    return f"22°{unit.upper()} in {city}"

weather = tool(get_weather)
print(weather.name)
print(weather.description)
pprint(weather.parameters)
```
```output
get_weather
Get the current weather for a city.
{'properties': {'city': {'description': 'City name, e.g. "Paris".',
                         'type': 'string'},
                'unit': {'default': 'c',
                         'description': 'Temperature unit.',
                         'enum': ['c', 'f'],
                         'type': 'string'}},
 'required': ['city'],
 'type': 'object'}
```

When the schema looks wrong, ask `derive()` why. It is `tool()` plus a
full typed account: one `DerivedParam` per parameter, with the hint, the
emitted fragment, and where each piece came from.

```python
d = derive(get_weather)
print(d.docstring_style_detected)
for p in d.params:
    print(f"{p.name}: {p.annotation} required={p.required} source={p.source}")
```
```output
google
city: str required=True source=hint+docstring
unit: typing.Literal['c', 'f'] required=False source=hint+docstring
```

When you need schema features Python hints can't express — `pattern`,
`minimum` — write the `FunctionTool` by hand. It is the canonical escape
hatch, not the default:

```python
def search_flights(origin, dest, max_results=5):
    rows = [("AC870", "08:15"), ("AF347", "10:40"), ("TS110", "13:05")]
    return [{"flight": f, "departs": t} for f, t in rows[:max_results]]

flights = FunctionTool(
    name="search_flights",
    description="Search direct flights between two IATA airport codes.",
    parameters={
        "type": "object",
        "properties": {
            "origin": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "dest": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["origin", "dest"],
    },
)
```

Send both tools. `ToolChoice(mode="required")` forces the model to call
at least one tool instead of answering in prose:

```python
router = LMRouter()
request = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("What's the weather in Paris, and any flights from YUL to CDG?"),),
    tools=(weather, flights),
    config=Config(tool_choice=ToolChoice(mode="required")),
)
response = router.complete(request)
print(response.finish_reason)
for call in response.tool_calls:
    print(call.id, call.name, call.input)
```
```output
tool_call
call_E8wA… get_weather {'city': 'Paris', 'unit': 'c'}
call_VM2a… search_flights {'origin': 'YUL', 'dest': 'CDG', 'max_results': 5}
```

The model requested two calls; nothing ran. Dispatch is yours — a plain
dict from tool name to function, with whatever validation or sandboxing
your application needs:

```python
handlers = {fn.__name__: fn for fn in (get_weather, search_flights)}

results = {}
for call in response.tool_calls:
    out = handlers[call.name](**call.input)
    results[call.id] = out if isinstance(out, str) else json.dumps(out)
pprint(results)
```
```output
{'call_E8wA…': '22°C in Paris',
 'call_VM2a…': '[{"flight": "AC870", "departs": "08:15"}, '
               '{"flight": "AF347", "departs": "10:40"}, '
               '{"flight": "TS110", "departs": "13:05"}]'}
```

Close the loop: extend the conversation with the assistant's tool-call
message, then a `tool` message mapping call id → output, and complete
again.

```python
followup = Request(
    model="gpt-4.1-mini",
    messages=(
        *request.messages,
        response.message,
        Message.tool(results),
    ),
    tools=(weather, flights),
)
final = router.complete(followup)
print(final.text)
print(final.finish_reason)
```
```output
The weather in Paris is currently 22°C. There are a few direct flights from YUL (Montreal) to CDG (Paris Charles de Gaulle) available:

- Flight AC870 departs at 08:15
- Flight AF347 departs at 10:40
- Flight TS110 departs at 13:05
…
stop
```

## How it works

`FunctionTool.parameters` is opaque JSON Schema and goes on the wire
unchanged; `tool(fn)` only fills it in. Derivation is eager — a
non-derivable signature raises `ToolDerivationError` at `tool()` time,
never at request time — and the full hint-to-schema table lives in
[tools-from-functions](../tools-from-functions.md). It is deliberately
not a decorator: a decorator would replace or wrap the function.

On the response side, a model that wants tools answers with
`finish_reason='tool_call'` and `ToolCallPart`s in `response.message`;
`response.tool_calls` is the shortcut. lm15 stops there. It does not
execute, validate `call.input` against the schema, retry, or loop —
that is policy, and policy lives in your layer (see the
[design rationale](../design-rationale.md#why-no-call-no-model-object-no-automatic-tool-loop)).
You answer by appending `response.message` verbatim (the provider needs
to see its own calls) followed by `Message.tool({call_id: output})`,
which builds one `ToolResultPart` per entry.

## Variations

- **Async mirror.** Same shapes with `AsyncLMRouter`:
  `response = await router.complete(request)`; handlers can be
  coroutines you `await` in your own loop.
- **Targeting one tool.** `ToolChoice.from_tools(weather, mode="required")`
  converts tool objects to an allowlist of names:
  ```output
  ToolChoice(mode='required', allowed=('get_weather',), parallel=None)
  ```
  `mode="none"` disables tool calls without removing the schemas;
  `parallel=False` asks for at most one call per turn where the
  provider supports it.
- **When derivation refuses.** Hard inputs fail loudly rather than
  guess — here, a fixed-length tuple:
  ```python
  def lookup(record: tuple[str, int]) -> str: ...
  tool(lookup)
  ```
  ```output
  ToolDerivationError: cannot derive 'lookup': parameter 'record' has
  fixed-length tuple annotation tuple; only homogeneous tuple[X, ...] is
  supported; override this parameter via ToolConfig(overrides=...) or
  pass an explicit FunctionTool with hand-written parameters
  ```
  `ToolConfig(overrides=(("record", {...}),))` patches one parameter;
  the hand-written `FunctionTool` remains the full escape hatch.
- **Strict modes.** `ToolConfig(additional_properties_false=True)` emits
  `"additionalProperties": false` for providers' strict tool modes;
  those modes typically also require every property to be required.
- **Provider notes.** Call ids differ in shape (`call_…` on OpenAI,
  `toolu_…` on Anthropic, short opaque ids on Gemini) — treat them as
  opaque and always echo them back in `Message.tool`.

## See also

- [07 — Built-in provider tools](07-builtin-tools.md)
- [08 — Structured output](08-structured-output.md)
- [15 — Errors, retries & testing](15-errors-and-testing.md)
- [Tools from functions](../tools-from-functions.md) — full derivation rules
- [Using the router](../using-the-router.md)
