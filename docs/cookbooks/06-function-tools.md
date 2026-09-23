# Function tools: define & dispatch

**Problem** — You want the model to call your Python functions, and lm15
has no agent loop to do the calling for you. The recipe: describe each
function as a `FunctionTool`, send the call, run the function yourself,
feed the result back.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

A `FunctionTool` is data: a name, a description the model reads, and a
JSON Schema for the inputs. It does not wrap, register, or execute
anything; the function stays yours.

```python
import json
from pprint import pprint

from lm15 import Config, FunctionTool, LMRouter, Message, Request, ToolChoice

def get_weather(city: str, unit: str = "c") -> str:
    return f"22°{unit.upper()} in {city}"

weather = FunctionTool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": 'City name, e.g. "Paris".'},
            "unit": {"enum": ["c", "f"], "type": "string", "description": "Temperature unit.", "default": "c"},
        },
        "required": ["city"],
    },
)
```

The schema can say more than Python's types: a `pattern`, a `minimum`,
an upper bound on a count:

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
unchanged. lm15 does not derive schemas from function signatures: that is
a convenience with opinions (docstring styles, which types map to what,
strict or open schemas), and it belongs to the library you build on lm15,
where it can be the same for every language you support. A request holds
only data; a function passed in `tools` is refused with this recipe named.

On the response side, a model that wants tools answers with
`finish_reason='tool_call'` and `ToolCallPart`s in `response.message`;
`response.tool_calls` is the shortcut. lm15 stops there. It does not
execute, validate `call.input` against the schema, retry, or loop —
that is policy, and policy lives in your layer (see the
[design rationale](../design-rationale.md#why-no-call-no-model-object-no-automatic-tool-loop)).
You answer by appending `response.message` verbatim (the provider needs
to see its own calls) followed by `Message.tool(call_id, output)` — or
`Message.tool({call_id: output, ...})` to answer several calls at once,
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
- **Strict modes.** Add `"additionalProperties": False` to the schema
  for providers' strict tool modes; those modes typically also require
  every property to be required.
- **Provider notes.** Call ids differ in shape (`call_…` on OpenAI,
  `toolu_…` on Anthropic, short opaque ids on Gemini) — treat them as
  opaque and always echo them back in `Message.tool`.

## See also

- [07 — Built-in provider tools](07-builtin-tools.md)
- [08 — Structured output](08-structured-output.md)
- [17 — Errors, retries & testing](17-errors-and-testing.md)
- [Using the router](../using-the-router.md)
