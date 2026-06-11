# Tools from functions

`lm15.tool(fn)` derives a `FunctionTool` from a plain Python callable —
its signature, type hints, and docstring — so you stop hand-writing
JSON Schema for the easy cases. It is pure sugar over the canonical
path: it returns an ordinary frozen `FunctionTool`, the JSON-Schema
`parameters` field stays primary and opaque on the wire (INV-033), and
nothing is registered, wrapped, cached, or executed.

```python
from typing import Literal

from lm15 import Config, LMRouter, Message, Request, ToolChoice, tool

def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
        unit: Temperature unit.
    """
    return f"22°{unit.upper()} in {city}"

weather = tool(get_weather)
print(weather.parameters)
# {'type': 'object',
#  'properties': {'city': {'type': 'string', 'description': 'City name, e.g. "Paris".'},
#                 'unit': {'enum': ['c', 'f'], 'type': 'string',
#                          'description': 'Temperature unit.', 'default': 'c'}},
#  'required': ['city']}
```

Deliberately **not** a decorator: a decorator would replace or wrap the
function — magic. Call `tool(fn)` and keep `fn` yourself.

## Derivation rules

Derivation is eager — errors surface when you call `tool()`, never at
request time. It is soft where the input is prose (a missing docstring
description is never an error) and hard where the input is types
(anything not obviously JSON-Schema-able raises `ToolDerivationError`
rather than guess).

| Python hint | JSON Schema |
|---|---|
| `str` / `int` / `float` / `bool` | `"string"` / `"integer"` / `"number"` / `"boolean"` |
| `None` / `NoneType` | `{"type": "null"}` |
| `Any` | `{}` (explicitly anything) |
| `list[X]`, `Sequence[X]` | `{"type": "array", "items": X}` |
| `set[X]`, `frozenset[X]` | array + `"uniqueItems": true` |
| `tuple[X, ...]` (homogeneous only) | `{"type": "array", "items": X}` |
| `dict[str, X]`, `Mapping[str, X]` | `{"type": "object", "additionalProperties": X}` |
| `X \| Y`, `Optional[X]` | `{"anyOf": [...]}` (`Optional` adds `{"type": "null"}`) |
| `Literal["a", "b"]` | `{"enum": [...]}`, plus `"type"` when homogeneous |
| `enum.Enum` subclass | `{"enum": [member values]}` (values must be JSON) |
| `Annotated[X, "text"]` | schema for `X` + `"description": "text"` |
| `TypedDict` / `@dataclass` class | inlined nested object (no `$ref` in v1) |
| fixed-length tuple, non-`str` dict keys, recursive types, anything else | `ToolDerivationError` |

Further rules:

- **Required-ness comes solely from defaults.** A parameter is required
  iff it has no default. `Optional[X]` is about *value nullability* and
  maps to `anyOf` with null — the two axes are orthogonal.
- JSON-compatible defaults are emitted as informational `"default"`
  values (disable with `ToolConfig(include_defaults=False)`); non-JSON
  defaults are silently skipped.
- Docstring parsing is best-effort: `"auto"` tries Google, NumPy, then
  Sphinx markers. The summary paragraph becomes the tool description.
- `*args`, `**kwargs`, positional-only parameters, missing annotations,
  and bare lambdas are errors — tool calls dispatch by name, so
  signatures must be fully nameable.
- Name and description are overridable: `tool(fn, name=..., description=...)`.

Two escape hatches when one parameter won't derive:
`ToolConfig(overrides=(("when", {"type": "string", "format": "date-time"}),))`
replaces derivation for just that parameter, or write the whole
`FunctionTool` by hand — that path is and remains canonical.

For provider strict-tool modes, `ToolConfig(additional_properties_false=True)`
emits `"additionalProperties": false`; note that strict modes typically
also require every property to be required.

## Inspecting a derivation

`derive(fn)` is `tool(fn)` plus a full typed account — the `explain()`
analogue. It returns a `ToolDerivation` with one `DerivedParam` per
parameter (`annotation`, `schema`, `required`, `description`, `source`).

```python
from lm15 import derive

d = derive(get_weather)
for p in d.params:
    print(p.name, p.annotation, p.source, p.required)
# city str hint+docstring True
# unit typing.Literal['c', 'f'] hint+docstring False
```

## Dispatch stays in your hands

lm15 never executes tools — `tool()` produces a schema, full stop. You
hold the functions and you run the loop:

```python
import json

handlers = {fn.__name__: fn for fn in (get_weather,)}

request = Request(
    model="claude-sonnet-4-5",
    messages=(Message.user("Weather in Paris?"),),
    tools=(weather,),
    config=Config(tool_choice=ToolChoice.from_tools(weather)),
)
response = LMRouter().complete(request)

for call in response.tool_calls:
    result = handlers[call.name](**call.input)   # YOUR code, YOUR sandboxing
    print(call.name, "->", result)
```

Whether to validate `call.input`, sandbox the handler, retry, or feed a
`ToolResultPart` back into the conversation is policy, and policy lives
in the layer above lm15 — see
[the design rationale](design-rationale.md#why-no-call-no-model-object-no-automatic-tool-loop).
