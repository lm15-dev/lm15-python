## The Pit of Success and the Cliff of Surprise

A well-designed API makes the right thing easy. The developer does the obvious thing — the thing they'd guess, the thing they'd try first — and it works. This is the pit of success: you fall into the correct behavior without effort. A poorly-designed API makes the right thing possible and the wrong thing equally easy. The developer guesses, the code compiles, the call succeeds, and the result is subtly wrong.

lm15 has several pits of success. It also has cliffs — places where the developer falls off the obvious path and discovers, too late, that the obvious path was wrong.

### The Pits

**Passing a function as a tool.**

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C in {city}"

resp = lm15.complete("gpt-4.1-mini", "Weather in Paris?", tools=[get_weather])
```

The developer's instinct — "I have a function, I'll pass it" — is correct. The function becomes a tool. The schema is inferred. The execution is automatic. The developer didn't learn about `Tool` objects, JSON Schema, or the tool-call protocol. They passed a function and got tool calling for free.

This is a deep pit of success. The obvious thing (pass a function) is the right thing. The mechanism (schema inference, auto-execution) is invisible. The result (the model called the function) is what the developer expected. Zero concepts stand between intent and outcome.

**Switching providers by changing a string.**

```python
resp = lm15.complete("gpt-4.1-mini", "Hello.")     # OpenAI
resp = lm15.complete("claude-sonnet-4-5", "Hello.")  # Anthropic
resp = lm15.complete("gemini-2.5-flash", "Hello.")   # Gemini
```

The developer's instinct — "I'll try a different model" — requires changing one argument. Not a different import, not a different client, not a different response type. One string. The developer who has never thought about multi-provider architecture gets multi-provider architecture by accident.

**Reading the response.**

```python
print(resp.text)
```

The developer's first question — "what did it say?" — is answered by the most obvious attribute. Not `resp.content[0].text`, not `resp.choices[0].message.content`, not `resp.generations[0].text`. `resp.text`. The attribute name is the question.

### The Cliffs

**Auto-execution of dangerous tools.** Chapter 3's central concern, restated as a UX problem. Passing a callable auto-executes it. Passing a `Tool` object doesn't. The distinction between "this will run automatically" and "this will wait for your approval" is encoded in the *type of the argument*, not in a visible parameter. A developer who passes `tools=[write_file, delete_records]` as callables — because callables are the obvious, documented path — gets auto-execution on destructive operations. The pit of success (passing functions is easy) leads over a cliff (the functions execute without review).

The UX failure: the obvious thing (pass functions) and the right thing (gate dangerous operations) are different actions. The API doesn't signal the difference. A developer who's been in the pit of success with `get_weather` applies the same pattern to `delete_records` and is over the cliff before they know it exists.

**The `reasoning` dict form.** `reasoning=True` works everywhere — a pit of success. `reasoning={"budget": 10000}` works on Anthropic and is silently ignored on OpenAI. The developer who fine-tunes their reasoning config on Claude, switches to GPT, and gets different behavior has fallen off a cliff. The API accepted the parameter, executed the call, and returned a response — no error, no warning, no indication that the config was meaningless. The cliff is silent.

Silent cliffs are the worst kind. A cliff with an error message is a wall — the developer hits it, reads the message, adjusts. A silent cliff produces subtly wrong behavior that the developer might not notice for weeks. "The model seems less thorough on GPT" — yes, because the reasoning budget you thought you set was ignored.

**The history accumulation surprise.** A developer creates a `model()` object for configuration reuse — same system prompt, same temperature, same tools. They don't want conversation memory; they want config binding. But the `model()` object accumulates history on every call. After 100 calls, the context contains 100 prior exchanges. The calls get slower, more expensive, and eventually hit the context limit. The developer's intent (reusable config) and the object's behavior (accumulating conversation) are misaligned.

The developer who wanted config reuse should have used a config dict with `complete()`. But the API didn't signal this — `model()` looks like config binding (the constructor takes the same parameters as `complete()`) and behaves like conversation management (it accumulates history). The developer fell into one pit (convenient config) and off a cliff (unwanted history).

**`stream.response` before consumption.** Accessing `stream.response` before iterating the stream silently consumes the entire stream to materialize the response. The developer who writes `resp = stream.response` expecting a property access gets a blocking operation that defeats the purpose of streaming. No error, no warning. The property looks like an accessor; it's actually a consumer.

### The Pattern

The pits share a property: the developer's mental model matches the API's behavior. "Pass a function, get tool calling." "Change the string, change the provider." "Read `.text`, get the text." The API meets the developer where they are.

The cliffs share a different property: the developer's mental model is *almost right* but wrong in one critical dimension. "Pass a function, it auto-executes" — the developer didn't model the auto-execution. "Set a budget, it applies" — the developer didn't model the provider-specificity. "Create a model, configure once" — the developer didn't model the history accumulation. The API is close enough to the developer's expectation that the divergence is invisible until it causes damage.

The design lesson: **pits of success are about matching the developer's first guess. Cliffs are about the developer's second assumption — the thing they didn't check because the first guess worked.** The first guess ("pass a function") works beautifully. The second assumption ("it won't auto-execute destructive operations") is wrong, and the developer had no reason to check it because everything up to that point confirmed their mental model.

Closing the gap between pits and cliffs requires making the second assumption checkable without destroying the first guess's simplicity. Chapter 3's `Tool.auto()` / `Tool.manual()` proposal is one approach. The `reasoning` dict validation warning from the improvement essay is another. Each one adds a small friction to the first guess (an extra word, a logged warning) in exchange for preventing the second assumption from becoming a cliff. The tradeoff is always the same: how much convenience are you willing to sacrifice for how much safety?
