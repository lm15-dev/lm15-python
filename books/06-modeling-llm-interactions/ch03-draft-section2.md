## The Protocol

Let's trace a single tool call through the system, because the abstraction —
"the model calls a function" — hides the mechanical reality, and the mechanical
reality is where the bugs live.

The user asks: "What's the current weather in Montreal?" The model has access to
a tool called `get_weather` that takes a `city` parameter. Here's what actually
happens, step by step.

**Step 1: The model generates a tool request.** Instead of generating text
tokens that form a natural-language answer, the model generates tokens that form
a structured JSON object: `{"name": "get_weather", "arguments": "{\"city\":
\"Montreal\"}"}`. It also generates a tool-call ID — a string like `call_abc123`
— that will be used later to match the result to the request. The model signals
that this is a tool call rather than text by emitting it in a special format
that the provider recognizes, and the response comes back with `finish_reason:
"tool_call"` instead of `finish_reason: "stop"`.

This is the first boundary crossing: the model's token stream is interpreted as
a structured request. Note the nested encoding — `arguments` is a JSON string
*inside* a JSON object. The model generates the arguments as serialized text,
not as a structured object. The provider parses the outer JSON; the library
parses the inner JSON. If the model generates malformed JSON in the arguments —
which happens, especially with complex schemas — the parse fails and the tool
call is broken. The model is, quite literally, writing code (JSON) as text, and
hoping the text is syntactically valid.

**Step 2: The application executes the function.** The library (or the
developer) receives the parsed tool call — name, arguments, ID — and invokes the
corresponding function. `get_weather("Montreal")` runs. Maybe it calls a weather
API. Maybe it reads from a cache. Maybe it fails because the API is down.
Whatever happens, it produces a result: a string like `"22°C, partly cloudy"`,
or an error like `"Error: weather API unavailable"`.

This is the second boundary crossing: the application's world — network calls,
file I/O, database queries, computation — produces a result that must travel
back into the model's world. The result must be serialized as text, because text
is the only thing the model can read. If your function returns a Python dict,
the library calls `str()` on it and the model reads `"{'temp': 22, 'condition':
'cloudy'}"` — syntactically Python, not JSON, with single quotes the model may
struggle to parse. The return value format matters more than most developers
realize, because the model's ability to use the result depends entirely on its
ability to read the result.

**Step 3: The result goes back to the model.** The library constructs a
follow-up request. This request contains the *original* conversation — system
prompt, user message, the assistant's tool-call message — plus a new message
with role `"tool"` carrying the result. The tool result is tagged with the
tool-call ID from step 1, so the model can match it to the request it made. The
entire conversation, now including the tool interaction, is sent as a new API
call. The model reads everything from the beginning, sees that it requested
weather data, sees the result, and generates its final text response: "It's
currently 22°C and partly cloudy in Montreal."

This is the third boundary crossing: the tool result, formatted as a message,
enters the model's context alongside all other messages. The model doesn't
experience the tool result differently from a user message — it's text in the
token stream, processed by the same attention mechanism. The model *trusts* this
text the same way it trusts user input: completely and uncritically. If the
application returns `"The weather in Montreal is 500°C and raining fire"`, the
model will incorporate that into its response without hesitation. The model has
no way to verify tool results. It has no concept of "that doesn't seem right."
It reads and believes.

### What the Protocol Reveals

Three properties of this protocol matter for library design.

**The protocol is multi-turn.** A tool call is not a single request-response.
It's a minimum of two API calls: one that produces the tool request, one that
processes the result. With multiple tool calls, or multi-hop reasoning, it can
be three, five, ten calls. Each call carries the full conversation, including
all previous tool interactions. The protocol is inherently stateful — you can't
process the result without the context of the request. This is why stateless
functions (`lm15.complete()`) handle auto-executed tools internally (making the
multi-turn sequence invisible) but can't handle manual tools (the developer
needs to hold state between calls).

**The protocol is asymmetric.** The model initiates tool calls, but the
application controls execution. The model can ask for anything — it can request
tools that don't exist, pass arguments that don't match the schema, or call the
same tool fifty times in a row. The application decides what to actually do. It
can execute faithfully, modify the arguments, return a fake result, refuse to
execute, or terminate the loop. The model is a requester, not a commander. The
application has veto power at every step.

This asymmetry is a security feature, not an accident. If the model could
execute tools directly — without the application as intermediary — there would
be no opportunity for validation, approval, or intervention. The application's
position between the model and the tools is the only control point. Every safety
mechanism in tool-using systems — approval gates, argument validation, rate
limiting, budget caps — lives at this control point. Remove the intermediary and
you remove the ability to say no.

**The protocol is trust-based, in one direction.** The model trusts tool results
completely. It has no mechanism for skepticism — no way to say "that result
seems wrong, let me verify." If the application returns incorrect data, the
model will use it. If the application lies, the model will believe the lie. This
means the application can manipulate the model's behavior by controlling tool
results — which is useful (injecting context, correcting errors, providing mock
data in tests) and dangerous (silently biasing the model's reasoning by
filtering or modifying tool output).

The reverse trust doesn't exist. The application has no reason to trust the
model's tool requests. The model might hallucinate tool names that don't exist.
It might pass arguments that don't match the schema. It might call a tool when
it shouldn't — requesting a database write when it was only asked to read. The
application must validate every tool request before executing it, because the
model's requests are probabilistic outputs, not verified commands.

This unidirectional trust — the model trusts the application, the application
doesn't trust the model — is the fundamental asymmetry of tool use. It's what
makes the control question (who executes?) so consequential, and it's what makes
the next section's analysis of auto-execute vs manual execution more than an API
convenience question. It's a question about where the trust boundary sits, and
what passes through it unchecked.
