## The Control Problem

The model has requested `write_file(path="config.py", content="...")`. Someone
has to execute it. The question is who, and the answer is the most consequential
API design decision in tool-using systems — more consequential than the message
representation (Chapter 1), more consequential than conversation ownership
(Chapter 2). Because the message representation determines what you can
*express*. Conversation ownership determines what you can *remember*. Tool
execution determines what you can *do*. And "do" is where the irreversible
consequences live.

There are three positions on the spectrum.

### Auto-Execute: The Library Acts

You pass a Python function. The library infers the schema, sends it to the
model, catches the tool call, executes your function, sends the result back, and
returns the final response. One call in, one response out. The tool round-trip
is invisible.

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C in {city}"

resp = lm15.complete("gpt-4.1-mini", "Weather in Montreal?", tools=[get_weather])
print(resp.text)  # "It's 22°C in Montreal."
```

The developer wrote a function and passed it. The developer did not explicitly
authorize any specific call. The model decided to call `get_weather` with
`city="Montreal"`, and the library executed that decision without asking. If the
model had instead decided to call `get_weather` with `city="'; DROP TABLE
cities; --"` — a prompt-injection attempt encoded in the arguments — the library
would have executed that too. The library doesn't inspect arguments. It calls
the function with whatever the model provides.

This is delegation of agency. The developer has said: "I trust the model to make
good decisions about when and how to call this function, and I trust the library
to execute those decisions without review." For `get_weather`, that trust is
well-placed. The function has no side effects, the worst case is a nonsensical
city name, and the model is good at generating reasonable arguments for simple
schemas. The cost of misplaced trust is a useless weather lookup.

For `write_file`, the trust calculation is different. The model might write the
wrong content. It might overwrite a file that shouldn't be overwritten. It might
write to a path that the developer didn't intend. And the developer will only
discover this after it's happened, because auto-execution didn't ask.

For `run_command`, the trust is existential. `run_command("rm -rf /")` is a
valid tool call. The model probably won't generate it — but "probably" is doing
a lot of work in a system that processes thousands of requests. The long tail of
a probability distribution is where disasters live.

### Manual Execute: The Developer Acts

You pass a `Tool` object — a schema without a function. The library sends the
schema to the model, catches the tool call, and returns it to you. You execute
it yourself. You send the result back with `submit_tools()`.

```python
from lm15 import Tool

write_file = Tool(name="write_file", description="Write to a file", parameters={...})
agent = lm15.model("claude-sonnet-4-5", tools=[write_file])
resp = agent("Rewrite config.py to use environment variables.")

for tc in resp.tool_calls:
    print(f"Write to {tc.input['path']}?")
    print(tc.input['content'][:200])
    if input("Approve? [y/n] ").strip() == "y":
        open(tc.input['path'], 'w').write(tc.input['content'])

results = {tc.id: "File written." for tc in resp.tool_calls}
resp = agent.submit_tools(results)
```

The developer sees every tool call before it executes. The developer approves or
rejects. The model's decisions are proposals, not commands. The trust boundary
is explicit: the model proposes, the human disposes.

The cost is boilerplate and latency. Every tool call requires code to handle it.
Multi-hop interactions — model calls tool A, reads result, calls tool B, reads
result, answers — require a loop that the developer writes and maintains. If the
model makes 15 tool calls in a coding session, that's 15 approval prompts. The
developer who started with manual execution for safety often ends up pressing
"y" reflexively by call 8, at which point the approval gate is security theater.

There's a deeper cost: manual execution forces the library to provide stateful
objects. `submit_tools()` needs the conversation context — the previous
messages, the pending tool calls, the tool definitions. A stateless function
can't provide this. This is why `lm15.model()` exists, and why `lm15.complete()`
can auto-execute tools (it handles the state internally) but can't support
manual tool execution (there's no state to submit into). The control choice
shaped the API surface. The library has two calling conventions — function and
object — because tool execution has two modes.

### Hybrid: The Pragmatic Middle

The experienced agent builder converges on a pattern: auto-execute tools that
can't cause harm, manually approve tools that can.

```python
# Auto-execute: no consequences if called incorrectly
def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()

def search_code(pattern: str) -> str:
    """Search for a pattern in the codebase."""
    return subprocess.run(["grep", "-r", pattern, "."], capture_output=True, text=True).stdout

# Manual: consequences if called incorrectly
write_file = Tool(name="write_file", description="Write to a file", parameters={...})
run_command = Tool(name="run_command", description="Run a shell command", parameters={...})

agent = lm15.model("claude-sonnet-4-5",
    tools=[read_file, search_code, write_file, run_command])
```

Read operations are callables — auto-executed, no approval needed. Write
operations are `Tool` objects — the model can request them, but the application
controls execution. The agent navigates the codebase freely (reads, searches)
and proposes changes (writes, commands) that require approval.

This pattern encodes a distinction that the type system doesn't make explicit:
**tools have a risk profile**, and the risk profile should determine the
execution mode. A `read_file` tool with auto-execution is a convenience. A
`write_file` tool with auto-execution is a gamble. A `run_command` tool with
auto-execution is an open door. The hybrid pattern lets the developer calibrate
the trust level per tool rather than making a single library-wide decision.

But lm15 doesn't enforce this pattern — it falls out of the type system
accidentally. Callables auto-execute. `Tool` objects don't. There's no
`risk_level` parameter, no `requires_approval` flag, no formal distinction
between safe and dangerous tools. The developer encodes the distinction by
choosing between two tool representations. This works, but it's implicit. A
developer who doesn't know the convention — who passes all tools as callables
because it's simpler — gets auto-execution on everything, including the
dangerous operations.

A library that took the risk distinction seriously would make it explicit:

```python
# Hypothetical — not lm15's current API
tools=[
    Tool(fn=read_file, auto=True),
    Tool(fn=write_file, auto=False, requires="human_approval"),
    Tool(fn=run_command, auto=False, requires="human_approval"),
]
```

This doesn't exist in lm15, and it doesn't exist in most libraries, because the
line between "safe" and "dangerous" is application-specific. `write_file` is
dangerous in a production agent and harmless in a sandboxed test environment.
`send_email` is dangerous when talking to real users and harmless when talking
to a test inbox. The library can't know the risk because the library doesn't
know the deployment context.

This is the honest conclusion of the control problem: the right execution mode
depends on the specific tool, in the specific application, in the specific
deployment context. No library can get this right by default. The best a library
can do is make the choice explicit — give the developer the tools (literally) to
express their trust decisions — and default to the safer side. lm15 defaults to
auto-execute for callables and manual for schemas. A more cautious library might
default to manual for everything and require explicit opt-in for auto-execution.
The tradeoff is convenience vs safety, and the right point depends on whether
you're building a research notebook or a production system that runs
unsupervised at 3 AM.
