## The Loop

The model calls `search("protein folding advances")`. It reads the results. The
results mention AlphaFold. It calls `search("AlphaFold3 accuracy benchmarks")`.
It reads those results. It finds a reference to a specific paper. It calls
`read_url("https://nature.com/...")`. It reads the paper's abstract. Now it has
enough context to answer the user's question. It generates a three-paragraph
summary with citations.

That was four tool calls across five API requests, driven entirely by the
model's judgment about what information it needed. Nobody programmed the
sequence. Nobody wrote a flow chart. The model decided, at each step, whether it
had enough information to answer or needed to look up something else. When it
decided it had enough, it stopped calling tools and started generating text.

This is the agent loop:

```python
resp = agent("Summarize recent protein folding advances.")

while resp.finish_reason == "tool_call":
    results = execute_tools(resp.tool_calls)
    resp = agent.submit_tools(results)

print(resp.text)
```

Five lines. The model acts. The code executes. The model decides whether to act
again. The loop continues until the model generates text instead of a tool call
— until `finish_reason` switches from `"tool_call"` to `"stop"`.

Every agent framework in existence — LangChain's agents, AutoGPT, CrewAI, custom
orchestrators with planning and reflection and self-critique — is this loop with
additional machinery. The planning is in the system prompt. The reflection is an
extra turn where the model reviews its own output. The self-critique is a second
model call that evaluates the first. The coordination between multiple agents is
multiple loops communicating through shared state. Strip away the abstractions
and there's always a `while` loop checking `finish_reason`.

The loop itself is trivially simple. The hard problem is stopping.

### The Stopping Problem

Who decides when the loop ends? In the code above, the model decides — it stops
calling tools when it believes it has enough information. But the model's belief
is a probability distribution, not a verified fact. The model might stop too
early (it thinks it knows enough but doesn't). It might stop too late (it keeps
searching for information it already has). It might never stop (it enters a
cycle of calling the same tool with slight variations, each result triggering
another search).

lm15 imposes a hard limit: 8 tool-call rounds per `complete()` call. After 8
rounds, it stops the loop and returns whatever the model has generated. This
number is arbitrary. Why not 5? Why not 20? There's no principled answer. The
limit exists to prevent infinite loops, not because 8 is the right number of
steps for any particular task.

The right stopping condition is always task-specific:

- A coding agent should stop when the tests pass. That's a verifiable exit
  condition — run the test suite, check the result, stop if green.
- A research agent should stop when it has enough sources. But "enough" is a
  judgment call that depends on the user's question, the quality of the sources
  found, and the desired depth.
- A customer support agent should stop when the user's question is answered. But
  "answered" is the model's assessment, which might not match the user's.
- A data analysis agent should stop when the analysis is complete. But
  "complete" depends on what the user considers a thorough analysis.

In each case, the stopping condition depends on the *task*, which the library
doesn't know, and on a *quality judgment*, which the library can't make. A
library that claims to solve the stopping problem — through "goal detection" or
"task completion analysis" — is hiding the difficulty behind a heuristic. The
heuristic might work on benchmarks. It will fail on the tasks that matter most,
because the tasks that matter most are the ones that aren't well-defined enough
for a heuristic to evaluate.

The honest design is lm15's: provide a safety limit (the 8-hop cap), expose the
loop to the developer (manual mode with `while finish_reason == "tool_call"`),
and let the developer implement the stopping condition that's appropriate for
their task. The library can't know when to stop. The developer can.

### The Cost of Iteration

Each iteration of the loop is a full API call. The model re-reads the entire
conversation — system prompt, tools, all previous turns, all previous tool calls
and results, and the latest tool result. By iteration 5, the context contains
the original question, the model's reasoning, five tool calls, five tool
results, and whatever text the model has drafted along the way. The quadratic
cost structure from Chapter 2 applies doubly here, because agent loops produce
more turns, with longer turns (tool results can be large), in a shorter time
span than human conversations.

A coding agent that reads a file (4KB), edits it, runs the test suite (2KB of
output), reads the error (1KB), edits again, and runs the suite again has
generated roughly 15KB of context in 6 turns — before counting the system prompt
and tool schemas. At Claude Sonnet pricing, that's about $0.25 for those 6 turns
alone. A 20-turn coding session easily reaches $2-3. Without prompt caching,
it's $5-8. These numbers make the abstract concern about quadratic cost very
concrete: agents are expensive, and the expense comes from the loop.

This cost structure creates a tension between thoroughness and economy. A
thorough agent calls more tools, reads more sources, verifies more assumptions.
An economical agent calls fewer tools and relies more heavily on its training
data. The budget — whether explicit (`TOKEN_BUDGET = 50_000`) or implicit (the
developer's willingness to pay) — is a stopping condition in disguise. When the
developer says "the agent is too expensive," what they mean is "the agent takes
too many turns," which means the stopping condition was wrong — not wrong in the
sense of buggy, but wrong in the sense of misaligned with the developer's cost
tolerance.

This is the deepest version of the stopping problem: the loop isn't just a
control flow question. It's an optimization problem with three competing
objectives — task quality, cost, and latency — and no single correct answer. The
library's job is to make the developer aware of all three (through token usage
reporting, history inspection, and explicit budget parameters) and let the
developer decide the balance.
