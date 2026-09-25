# Getting started

lm15 is one set of types for every LLM API. You build a `Request`, you
get back a `Response` — the same two objects whichever provider
answers.

This page is the whole core loop: install, one call, switch provider,
stream, use a tool. You need Python 3.10+, about five minutes, and one
API key — we sort the key out first. Every output below is real
captured output, and every block is copy-paste runnable.

## Install

```bash
python3 -m pip install lm15
```

Zero dependencies, stdlib only.

## Set an API key

Grab a key from any **one** provider and set its standard environment
variable — that is the only setup. On macOS or Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # console.anthropic.com
export GEMINI_API_KEY="AIza..."         # aistudio.google.com/apikey
export GROQ_API_KEY="gsk_..."           # console.groq.com/keys
export OPENAI_API_KEY="sk-..."          # platform.openai.com/api-keys
```

??? question "First time setting an environment variable? Open this."

    An environment variable is a named value your terminal hands to
    the programs it starts. Putting the key there means it never has
    to appear in your code. You only need **one** of the variables
    above — pick the provider you got a key from.

    === "macOS / Linux"

        Paste the `export` line into your terminal, then run your
        Python from that same terminal. It lasts until the terminal
        closes. Check it took:

        ```bash
        echo $ANTHROPIC_API_KEY
        ```

        To make it permanent, add the same `export` line to the end
        of `~/.zshrc` (macOS) or `~/.bashrc` (most Linux) and open a
        new terminal.

    === "Windows (PowerShell)"

        ```powershell
        $env:ANTHROPIC_API_KEY = "sk-ant-..."
        ```

        Lasts until the window closes; check with
        `echo $env:ANTHROPIC_API_KEY`. To make it permanent:

        ```powershell
        setx ANTHROPIC_API_KEY "sk-ant-..."
        ```

        `setx` writes it for *future* windows — open a new terminal
        before running Python.

    === "Windows (cmd)"

        ```bat
        set ANTHROPIC_API_KEY=sk-ant-...
        ```

        Lasts until the window closes (note: no quotes, no spaces
        around `=`). Make it permanent with the same `setx` command
        as PowerShell.

    Rather not touch the environment at all? Pass the key in code
    instead — `LMRouter(RouterConfig(api_keys={"anthropic": "sk-ant-..."}))`
    — just keep it out of version control.

The examples below use Anthropic; on a different provider, keep the
code and swap the model string (the tabs below show exactly how).
Running a local ollama? No key needed at all. Every other way to
authenticate — explicit keys, rotating token providers, Claude/ChatGPT
subscriptions — is on [Authentication](authentication.md).

If you forget, the error says precisely what to do:

```output
MissingCredentialError: no API key found for provider 'anthropic'.
Set ANTHROPIC_API_KEY in the environment, or pass
RouterConfig(api_keys={'anthropic': "..."}).
```

## One call

`LMRouter` is the front door: give it a model string and it picks the
right provider, the right adapter, and the right env var. (It is a
lookup table you can inspect, not magic — more on that below.)

```python
from lm15 import LMRouter, Message, Request

router = LMRouter()

response = router.complete(
    Request(model="claude-haiku-4-5",
            messages=(Message.user("Say hello in exactly three words."),))
)
print(response.text)
```

```output
Hello, world friend.
```

`response` is typed all the way down: `response.usage.input_tokens` is
`14`, `response.finish_reason` is `"stop"`, and `response.message`
slots straight into the next request's `messages` to continue the
conversation.

!!! note "Why the trailing comma in `(Message.user(...),)`?"
    `messages` is a tuple, and in Python a one-element tuple needs a
    trailing comma. lm15's types are immutable throughout — a `Request`
    you built is never changed under you — and tuples are the price of
    that guarantee. (Lists are accepted too and converted.)

## Any provider: change one string

The call never changes — only the model string does. Pick a tab:

=== "Anthropic"

    ```python
    response = router.complete(
        Request(model="claude-haiku-4-5",   # needs ANTHROPIC_API_KEY
                messages=(Message.user("The capital of Canada, two words max."),))
    )
    print(response.text)
    ```

    ```output
    Ottawa.
    ```

=== "Gemini"

    ```python
    response = router.complete(
        Request(model="gemini-3-flash-preview",   # needs GEMINI_API_KEY
                messages=(Message.user("The capital of Canada, two words max."),))
    )
    print(response.text)
    ```

    ```output
    Ottawa
    ```

=== "Groq"

    ```python
    response = router.complete(
        Request(model="groq:llama-3.3-70b-versatile",   # needs GROQ_API_KEY
                messages=(Message.user("The capital of Canada, two words max."),))
    )
    print(response.text)
    ```

    ```output
    Ottawa
    ```

=== "Local (ollama)"

    ```python
    response = router.complete(
        Request(model="ollama:qwen3:4b",   # no key; any model you've pulled
                messages=(Message.user("The capital of Canada, two words max."),))
    )
    print(response.text)
    ```

    No key, no config: local servers (`ollama:`, `vllm:`, `sglang:`)
    route by name and use a placeholder credential.

The same code covers OpenAI (`gpt-4.1-mini`) and anything else the
router knows — the full matrix of providers, endpoints, and env vars is
on [Providers & models](providers-and-models.md). And nothing about it
is magic — ask it to explain itself:

```python
print(router.resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

Prefer no router at all? The adapter classes are equally first-class:
`AnthropicLM(api_key="...")` takes the same `Request`. See
[Using the router](using-the-router.md) and
[Using the providers](using-the-a-provider.md).

## Streaming

Two pieces share the work: `router.stream()` yields **typed events**
as the provider sends them, and `ResponseStream` assembles them —
iterate it and you get the text as it arrives:

```python
from lm15 import ResponseStream

req = Request(model="claude-haiku-4-5",
              messages=(Message.user("One short sentence about rivers."),))
result = ResponseStream(router.stream(req), req)
for text in result:
    print(text, end="", flush=True)
```

```output
Rivers flow from high elevations to the sea, shaping landscapes and sustaining life along their paths.
```

When you need more than text — tool calls, thinking, audio — iterate
`result.events()` instead. And after the stream ends,
`result.response` is the very same `Response` a non-streaming call
would have returned: nothing about your downstream code has to care
which way the answer arrived. Full recipe:
[Streaming](cookbooks/05-streaming.md).

## Tools

A tool call is a conversation in two rounds: the model answers your
question with *"call `get_weather` with `{'city': 'Montreal'}`"*, you
run the function yourself, send the result back, and the model writes
the final answer. lm15 hands you each step as plain data — it never
executes anything for you, which means no framework to fight when you
want control over errors, retries, or budgets.

```python
from lm15 import FunctionTool

def get_weather(city: str) -> str:
    return f"18°C and sunny in {city}"

weather = FunctionTool(       # what the model sees: name, description, input schema
    name="get_weather",
    description="Current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)

messages = (Message.user("What's the weather in Montreal right now?"),)
r = router.complete(Request(model="claude-haiku-4-5", messages=messages, tools=(weather,)))

call = r.tool_calls[0]        # ToolCallPart(name='get_weather', input={'city': 'Montreal'})
messages = (*messages, r.message, Message.tool(call.id, get_weather(**call.input)))

final = router.complete(Request(model="claude-haiku-4-5", messages=messages, tools=(weather,)))
print(final.text)
```

```output
The weather in Montreal right now is **18°C (about 64°F) and sunny**. It's a nice day!
```

Full recipe, including hand-written schemas and parallel calls:
[Function tools](cookbooks/06-function-tools.md).

## Async

Working in an async app? Everything has an async mirror with the same
shape — add `Async` to the name and `await` the call:

```python
import asyncio
from lm15 import AsyncLMRouter

async def main() -> None:
    router = AsyncLMRouter()
    response = await router.complete(
        Request(model="groq:llama-3.3-70b-versatile",
                messages=(Message.user("Say ok."),))
    )
    print(response.text)

asyncio.run(main())
```

```output
Ok.
```

## Credentials, the short version

The one-line summary of each option — the full story, with examples,
is on [Authentication](authentication.md):

- **Env vars** — the router reads each provider's standard variable
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
  `GROQ_API_KEY`, …). The adapter classes themselves never touch the
  environment.
- **Explicit** — `LMRouter(RouterConfig(api_keys={"anthropic": "..."}))`,
  or pass `api_key=` when constructing an adapter directly.
- **Rotating** — anywhere a key string goes, a zero-argument callable
  works too (an Azure Entra token provider, your own refresh logic);
  it is resolved once per request, so long-lived clients never go
  stale.
- **Subscriptions** — logged into the Claude or Codex CLI?
  `ClaudeCodeLM()` / `OpenAICodexLM()` use those local credentials
  directly. On a SuperGrok plan, `lm15.auth.login_xai()` sets up
  `XaiLM()` the same way.
- **Local servers** — `ollama:`, `vllm:`, and `sglang:` models need no
  key at all.

## Where next

- Multimodal input, structured output, reasoning, caching, batch,
  media generation, live sessions: eighteen
  [cookbook recipes](cookbooks/index.md), each with real captured
  output.
- The canonical types in depth:
  [Using the type system](using-the-type-system.md).
- Why the API looks the way it does:
  [Design rationale](design-rationale.md).
