# Your first request

**Problem** — You have a model name and a prompt, and three providers
with three SDKs, three env vars, and three wire formats between you and
a response. lm15 gives you one `Request` type and a router that turns
the model string into the right provider — and can tell you exactly how
it decided.

## Recipe

Keys first. lm15 reads nothing implicitly; you put keys in the
environment, the router finds them there. This loader searches the
current directory and its parents for a `.env` file:

```python
import asyncio
import os
from pathlib import Path

from lm15 import (
    AnthropicLM, AsyncLMRouter, GeminiLM, LMRouter, Message, OpenAILM, Request,
)

def load_env(filename=".env"):
    for directory in (Path.cwd(), *Path.cwd().parents):
        path = directory / filename
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            return path
    return None

print(load_env())
```
```output
/home/maxime/Projects/lm15-dev/.env
```

(From a shell, the one-liner is `export $(grep -v "^#" ../.env | xargs)`.)

Now the request. `LMRouter` is the front door: it maps
`"gpt-4.1-mini"` to `OpenAILM`, builds the LM with the key from
`$OPENAI_API_KEY`, and forwards the call.

```python
router = LMRouter()
response = router.complete(
    Request(model="gpt-4.1-mini", messages=(Message.user("Say hello in five words."),))
)
print(response.text)
```
```output
Hello! Hope you're doing well.
```

`response` is a frozen dataclass, not a provider blob. Print it and you
see everything lm15 normalized:

```python
print(response)
```
```output
Response(
    text="Hello! Hope you're doing well.",
    model='gpt-4.1-mini-2025-04-14',
    finish_reason='stop',
    usage=Usage(input_tokens=13, output_tokens=8, …),
    id='resp_09b0…',
    provider_data=<dict: 35 keys>,
)
```

How did the router pick OpenAI? Ask it. `resolve()` is pure — no
network, no key values read — and its return value is the explanation:

```python
res = router.resolve("gpt-4.1-mini")
print(res)
print(res.provider, res.adapter, res.source, res.env_key)
```
```output
'gpt-4.1-mini' -> provider 'openai' (OpenAILM); via built-in rule prefix='gpt-' — OpenAI GPT family (Responses API; use openai_chat: for Chat Completions); wire model 'gpt-4.1-mini'; key from $OPENAI_API_KEY.
openai OpenAILM rule OPENAI_API_KEY
```

A `provider:` prefix bypasses the rule table and is always unambiguous:

```python
print(router.resolve("anthropic:claude-sonnet-4-5"))
```
```output
'anthropic:claude-sonnet-4-5' -> provider 'anthropic' (AnthropicLM); via explicit provider prefix; wire model 'claude-sonnet-4-5'; key from $ANTHROPIC_API_KEY.
```

The router never guesses. A string it cannot place raises
`UnknownModelError` with the fixes spelled out:

```python
router.resolve("qwen3.5:0.8b")
```
```output
Traceback (most recent call last):
  …
lm15.router.UnknownModelError: could not route model 'qwen3.5:0.8b': no provider prefix, no catalog supplied, and none of the 6 built-in rules matched. Use an explicit provider prefix, e.g. "anthropic:qwen3.5:0.8b" (known providers: anthropic, claude-code, gemini, openai, openai-codex, openai_chat). Or pass a model catalog: …
```

The router is sugar, not a layer. The direct LM classes are first-class
and take the same `Request`. The same prompt, three providers:

```python
prompt = (Message.user("Say hello in five words."),)

openai_lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])
print(openai_lm.complete(Request(model="gpt-4.1-mini", messages=prompt)).text)

anthropic_lm = AnthropicLM(api_key=os.environ["ANTHROPIC_API_KEY"])
print(anthropic_lm.complete(Request(model="claude-sonnet-4-5", messages=prompt)).text)

gemini_lm = GeminiLM(api_key=os.environ["GEMINI_API_KEY"])
print(gemini_lm.complete(Request(model="gemini-3-flash-preview", messages=prompt)).text)
```
```output
Hello there! Hope you're well!
Hello, how are you today?
Hello, how are you doing?
```

`AsyncLMRouter` is the async mirror — `complete` is awaitable,
`resolve()` stays sync because it is pure:

```python
async def main():
    arouter = AsyncLMRouter()
    response = await arouter.complete(
        Request(model="gemini-3-flash-preview", messages=prompt)
    )
    print(response.text)

asyncio.run(main())
```
```output
Hello, how are you today?
```

## How it works

A `Request` is a frozen dataclass: a model string and a tuple of
`Message` objects (plus optional `Config`, tools, and more in later
recipes). `router.complete(request)` does three things: resolve the
model string, build (and cache) the provider LM, and call its
`complete()`. Resolution walks three fixed rungs — explicit
`provider:` prefix, optional catalog, built-in prefix rules — first
match wins, no fallback chains. The full grammar and the catalog rung
are in [Using the router](../using-the-router.md).

Credentials follow the provider's manifest: `OpenAILM` declares
`OPENAI_API_KEY`, and that is the only place the router looks unless
you pass `RouterConfig(api_keys=...)`. A missing key raises
`MissingCredentialError` at `lm()` time, not a 401 at request time.
`resolve()` records *which* env var would be read, never its value.

Both paths — router and direct LM — produce the identical `Request`
and identical wire bytes. lm15 deliberately does not retry, does not
pool connections behind your back, and does not pick a model for you:
one call in, one typed `Response` out.

## Variations

- **Keep the LM, drop the router.** `router.lm("gpt-4.1-mini")`
  returns a plain `OpenAILM`; configure it once and never resolve
  again.
- **Direct LMs are for configuration, not preference.** Reach for them
  when you need a custom `base_url` or compat preset (ollama, vLLM —
  recipe [14](14-local-and-compatible-servers.md)) or when you are a
  library taking an LM object from your caller.
- **Async direct LMs exist too**: `AsyncOpenAILM`, `AsyncAnthropicLM`,
  `AsyncGeminiLM`, same constructors, awaitable `complete`.
- **OpenAI has two providers.** Bare `gpt-` routes to the Responses
  API (`OpenAILM`); `"openai_chat:gpt-4.1-mini"` selects Chat
  Completions (`OpenAIChatLM`). The resolve output above says so.

## See also

- [02 — Multi-turn conversations](02-conversations.md)
- [04 — Controlling generation](04-generation-config.md)
- [05 — Streaming](05-streaming.md)
- [Using the router](../using-the-router.md) — grammar, catalogs, credentials
- [15 — Errors, retries & testing](15-errors-and-testing.md)
