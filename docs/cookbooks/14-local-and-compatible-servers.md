# Local & OpenAI-compatible servers

**Problem** — Half the model world speaks the OpenAI Chat Completions
dialect: ollama on your laptop, Groq, OpenRouter, a vLLM box in the
rack. Each one diverges from OpenAI in small wire-format ways — which
max-tokens field, which role for instructions, whether reasoning fields
exist. `OpenAIChatLM` takes a compat preset name and handles the
dialect; the same `Request` works against all of them.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

This is the one recipe where you construct the LM directly instead of
going through the router. `compat="groq"` sets both the wire-format
policy and the default `base_url`:

```python
import os

from lm15 import LMRouter, Message, OpenAIChatCompat, OpenAIChatLM, Request, Result
from lm15.compat import OPENAI_CHAT_PRESET_BASE_URLS

groq = OpenAIChatLM(api_key=os.environ["GROQ_API_KEY"], compat="groq")
print(groq.base_url)
response = groq.complete(
    Request(model="llama-3.3-70b-versatile", messages=(Message.user("Say hello in five words."),))
)
print(response.text)
print(response.model, response.finish_reason, response.usage)
```
```output
https://api.groq.com/openai/v1
Hello, how are you today?
llama-3.3-70b-versatile stop Usage(input_tokens=41, output_tokens=8, …)
```

Local servers work the same way. Ollama wants *an* API key header but
ignores its value. This block requires a running ollama
(`ollama serve`) with the model pulled; everything else on this page
works without it:

```python
ollama = OpenAIChatLM(api_key="ollama", compat="ollama")
print(ollama.base_url)
print(ollama.complete(
    Request(model="llama3.2:1b", messages=(Message.user("Say hello in five words."),))
).text)
```
```output
http://localhost:11434/v1
Good day to you.
```

Each preset name carries a default endpoint:

```python
for name, url in OPENAI_CHAT_PRESET_BASE_URLS.items():
    print(f"{name:11s} {url}")
```
```output
openai      https://api.openai.com/v1
ollama      http://localhost:11434/v1
groq        https://api.groq.com/openai/v1
openrouter  https://openrouter.ai/api/v1
vllm        http://localhost:8000/v1
sglang      http://localhost:30000/v1
```

An explicit `base_url` always wins over the preset's default — the
preset keeps supplying the wire-format policy. This is how you point
the `vllm` preset at your own host:

```python
vllm = OpenAIChatLM(api_key="unused", base_url="http://gpu-box:8000/v1", compat="vllm")
print(vllm.base_url)
```
```output
http://gpu-box:8000/v1
```

Why not the router? The router resolves *model names*, and a name like
`llama-3.3-70b-versatile` names a model, not a server — it runs on
Groq, on ollama, and on your vLLM box, with different URLs and
different keys. The router refuses to guess:

```python
LMRouter().resolve("llama-3.3-70b-versatile")
```
```output
Traceback (most recent call last):
  …
lm15.router.UnknownModelError: could not route model 'llama-3.3-70b-versatile': no provider prefix, no catalog supplied, and none of the 6 built-in rules matched. Use an explicit provider prefix, …
```

The `openai_chat:` prefix exists, but it routes to OpenAI's own Chat
Completions endpoint — there is deliberately no router syntax for a
non-default `base_url`:

```python
print(LMRouter().resolve("openai_chat:gpt-4.1-mini"))
```
```output
'openai_chat:gpt-4.1-mini' -> provider 'openai_chat' (OpenAIChatLM); via explicit provider prefix; wire model 'gpt-4.1-mini'; key from $OPENAI_API_KEY.
```

Everything else from the router recipes carries over to the direct LM —
same `Request`, same `Result`. Streaming against Groq:

```python
req = Request(
    model="llama-3.1-8b-instant",
    messages=(Message.user("Name three rivers in Quebec, one per line, names only."),),
)
result = Result(events=groq.stream(req), request=req)
for text in result:
    print(text, end="", flush=True)
print()
print(result.finish_reason, result.usage)
```
```output
Saint Lawrence River

Magpie River

Matane River
stop Usage(input_tokens=48, output_tokens=12, total_tokens=60, …)
```

A preset is data, not magic. Print one and you see the whole policy:

```python
print(OpenAIChatCompat.preset("groq"))
```
```output
OpenAIChatCompat(instruction_role='system', max_tokens_field='max_tokens', stream_usage='include', tool_result_name='omit', assistant_after_tool_result=None, thinking_format='reasoning_effort', thinking_replay=None, assistant_reasoning_content=None, strict_tools='omit', cache_control='none', routing=None, extensions=None)
```

## How it works

`OpenAIChatLM.compat` accepts a preset name string, an
`OpenAIChatCompat` object, or None (plain OpenAI policy). A name does
two things at construction time: it resolves to a wire-format policy
via `OpenAIChatCompat.preset(name)`, and — only if you left `base_url`
at its default — it substitutes that server's default endpoint from
`OPENAI_CHAT_PRESET_BASE_URLS`.

The policy fields are the divergences that actually bite: Groq and
ollama still want `max_tokens` where OpenAI now wants
`max_completion_tokens`; ollama has no reasoning fields
(`thinking_format="none"`) while OpenRouter has its own
(`"openrouter"`); prompt-cache markers are OpenAI-only
(`cache_control`). Fields left None inherit; `"auto"` is an explicit
"use the adapter heuristic" — the distinction matters because policies
layer (`lm15/compat.py`).

The router stays out of this on purpose. Its job is explainable
name-to-provider resolution from three fixed rungs; a base URL and a
key for *your* server is configuration, not resolution, so the
documented path is one line of direct construction. See
[Using the router](../using-the-router.md), "When to use direct LM
objects instead".

## Variations

- **Async mirror.** `AsyncOpenAIChatLM` — same constructor, same
  `compat`/`base_url` handling, awaitable `complete`, async `stream`.
- **Override one field of a preset.** Pass an `OpenAIChatCompat`
  instead of a name: start from `OpenAIChatCompat.preset("vllm")` and
  rebuild with `dataclasses.replace(...)`. Note that a compat *object*
  does not set `base_url` — only a preset name does.
- **OpenRouter** is the same shape with a hosted twist:
  `OpenAIChatLM(api_key=os.environ["OPENROUTER_API_KEY"],
  compat="openrouter")`. Its preset keeps `cache_control="openai"` and
  speaks OpenRouter's reasoning format; the `routing` field carries
  OpenRouter-specific routing JSON.
- **More presets than shown here**: `lmstudio`, `deepseek`, `qwen`,
  `zai` — see `OpenAIChatCompat.preset` in `lm15/compat.py`. Unknown
  names raise `ValueError` at construction, not at request time.
- **Routing your own prefix.** `RouteRule("glm-", "openai_chat", ...)`
  prepended to `DEFAULT_RULES` makes the router accept your model
  names — but it still builds the LM against api.openai.com. Custom
  `base_url` means direct construction, full stop.

## See also

- [01 — Your first request](01-first-request.md) — keys, router vs direct LMs
- [05 — Streaming](05-streaming.md) — `Result` over typed stream events
- [15 — Errors, retries & testing](15-errors-and-testing.md)
- [16 — Provider passthrough](16-provider-passthrough.md) — server-specific knobs
- [Using the router](../using-the-router.md) — why `base_url` is not router syntax
