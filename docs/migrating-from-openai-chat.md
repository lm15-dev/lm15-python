# Migrating from the OpenAI SDK or LiteLLM

You have code that calls `client.chat.completions.create(...)` or
`litellm.completion(...)`: a model string, a list of `{"role", "content"}`
dictionaries, a few keyword arguments. This page moves that code to
lm15 one line at a time. Each step shows the line you have, the line
it becomes, and what happened underneath — so that when the two differ,
you know why and can decide for yourself.

The rule the whole page follows: **keep your messages and your model
string; change the call.** `LMRouter.complete_from_openai_chat(model,
messages, **kwargs)` takes the same arguments as both libraries and
answers with an lm15 `Response`. Anything lm15 cannot carry is refused
by name; nothing is silently dropped. It is a migration, not a
drop-in: the answer is `response.text`, not `choices[0].message.content`.

You need Python 3.10+, `pip install --pre lm15`, and the API keys you
already have — as you will see in a moment, they do not move. Every
output below is real captured output.

## Setup

One object, made once, reused everywhere. It keeps one connection pool
per provider, the way an `OpenAI()` client does.

```python
import lm15

router = lm15.LMRouter()
```

Every example on this page uses this `router`.

## Your first call, side by side

The running example: a system instruction and a question.

```python
messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]
```

=== "Before — OpenAI SDK"

    ```python
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=100,
    )
    print(response.choices[0].message.content)
    ```

=== "Before — LiteLLM"

    ```python
    import litellm

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=100,
    )
    print(response.choices[0].message.content)
    ```

**After — lm15**

```python
response = router.complete_from_openai_chat(
    "gpt-4o-mini",
    messages,
    max_completion_tokens=100,
)
print(response.text)
```

```output
The sky appears blue due to the scattering of sunlight by the Earth's atmosphere, with shorter blue wavelengths being scattered more than other colors.
```

The `messages` list is the same object. The keyword is the same
keyword. What changed is the shape of the answer: `response` is an
lm15 `Response`, typed all the way down — `response.usage.output_tokens`
is `26`, `response.finish_reason` is `"stop"`, `response.message` is a
message you can put into the next request.

Three things happened under that call, each visible if you want it:

- The bare `gpt-4o-mini` went to OpenAI's **Chat Completions**
  endpoint — the one both libraries were using — not the Responses API
  that `router.complete(Request(model="gpt-4o-mini", …))` would pick.
  Same string, different door, on purpose; write `openai:gpt-4o-mini`
  if you want Responses.
- The list was read into a `Request`: the system row became
  `request.system`, the token limit `request.config.max_tokens`.
- The reply was parsed by lm15's own OpenAI Chat adapter.

To see the `Request` instead of sending it — and the adapter it would
be sent through:

```python
request, lm = router.request_from_openai_chat("gpt-4o-mini", messages, max_completion_tokens=100)
print(type(lm).__name__, lm.base_url)
print(request.system, request.config.max_tokens, request.messages)
```

```output
OpenAIChatLM https://api.openai.com/v1
Answer in one sentence. 100 (Message(role='user', parts=(TextPart(text='Why is the sky blue?', continuation=(), type='text'),), continuation=()),)
```

## Where the key came from

That call worked without you setting anything up, and this is not
luck: the migration changes nothing about keys unless you passed one
in code.

=== "Before — OpenAI SDK"

    ```python
    client = OpenAI()          # reads $OPENAI_API_KEY
    ```

=== "Before — LiteLLM"

    ```python
    litellm.completion(model="gpt-4o-mini", …)   # reads $OPENAI_API_KEY
    ```

**After — lm15**

```python
router = lm15.LMRouter()       # reads $OPENAI_API_KEY
```

All three read the same variable, chosen by the provider the model
string resolves to. That is the whole story for a bare `gpt-…` name.
For every other string, the variable is the one you already have set
for that provider:

| your model string | goes to | key read from |
|---|---|---|
| `gpt-4o-mini`, `openai/…` | OpenAI, Chat Completions | `OPENAI_API_KEY` |
| `anthropic/…`, `claude-…` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini/…` | Gemini | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` |
| `groq/…` | Groq | `GROQ_API_KEY` |
| `openrouter/…` | OpenRouter | `OPENROUTER_API_KEY` |
| `deepseek/…` | DeepSeek | `DEEPSEEK_API_KEY` |
| `xai/…` | xAI | a stored xAI login if you have one, then `XAI_API_KEY` |
| `moonshot/…` | Moonshot | `MOONSHOTAI_API_KEY`, then `MOONSHOT_API_KEY` |
| `ollama/…`, `hosted_vllm/…` | your local server | nothing (a placeholder) |
| `azure/…` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, then the Azure identity chain |

Rather than trust a table, ask. `explain_auth` walks the same chain
`lm()` will walk, marks the rung that wins, and never prints a value:

```python
from lm15.doctor import explain_auth

print(explain_auth("anthropic").describe())
```

```output
auth for provider 'anthropic':
   - explicit api_keys entry: not provided
  => env $ANTHROPIC_API_KEY: set (value never shown)
  configured: yes — env $ANTHROPIC_API_KEY
```

And when nothing is set, the error is the fix:

```python
lm15.LMRouter(lm15.RouterConfig(env={})).lm("claude-sonnet-4-5")
```

```output
MissingCredentialError: no API key found for provider 'anthropic'.
Set ANTHROPIC_API_KEY in the environment, or pass
RouterConfig(api_keys={'anthropic': "..."}).
```

(`MissingCredentialError` is a `NotConfiguredError`; an existing
`except NotConfiguredError` keeps working.)

### A key passed in code

This is the one place keys move. Both libraries take a key on the
client or on the call; lm15 takes it on the router, once per provider.

=== "Before — OpenAI SDK"

    ```python
    client = OpenAI(api_key="sk-…")
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    ```

=== "Before — LiteLLM"

    ```python
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=messages,
        api_key="sk-ant-…",
    )
    ```

**After — lm15**

```python
router = lm15.LMRouter(lm15.RouterConfig(api_keys={
    "openai-chat": "sk-…",
    "anthropic": "sk-ant-…",
}))
response = router.complete_from_openai_chat("anthropic/claude-sonnet-4-5", messages)
```

An explicit entry beats the environment, and `explain_auth` shows the
environment variable being shadowed:

```python
print(explain_auth("openai-chat", config=router.config).describe())
```

```output
auth for provider 'openai-chat':
  => explicit api_keys entry: provided (value never shown)
   ~ env $OPENAI_API_KEY: set (value never shown)
  configured: yes — explicit api_keys entry
```

The provider names are the ones the router reports. Note `openai-chat`,
not `openai`: on this door a bare `gpt-…` string goes to Chat
Completions, and lm15 treats that endpoint as its own provider with its
own entry (`openai_chat` is accepted too). An entry under `openai` would
cover `openai:…` strings — the Responses API — and leave this call
reading the environment. A value may also be a zero-argument callable
that returns a fresh token, for credentials that rotate.

If you keep `api_key=` on the call out of habit, lm15 tells you where it
went rather than guessing:

```python
router.complete_from_openai_chat("gpt-4o-mini", messages, api_key="sk-…")
```

```output
NotConfiguredError: 'api_key' configures the client, not the request; in lm15 it lives in LMRouter(RouterConfig(api_keys={provider: key})) or the environment
```

!!! note "Why not per call?"
    The router builds one adapter per provider and reuses it — that is
    where the connection pool lives. A key on the call would either
    rebuild the adapter every time or silently rebind a shared one to a
    different account. Neither is what you meant. Need two accounts on
    one provider? Two routers.

### A different server

`base_url` on the client, or `api_base` on the litellm call, is a proxy
in front of OpenAI or a server of your own. It moves the same way the
key did:

=== "Before — OpenAI SDK"

    ```python
    client = OpenAI(base_url="https://gw.example/v1")
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    ```

=== "Before — LiteLLM"

    ```python
    response = litellm.completion(
        model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        api_base="http://gpu-box:8000/v1",
    )
    ```

**After — lm15**

```python
router = lm15.LMRouter(lm15.RouterConfig(base_urls={
    "openai-chat": "https://gw.example/v1",
    "vllm": "http://gpu-box:8000/v1",
}))
router.complete_from_openai_chat("gpt-4o-mini", messages)
router.complete_from_openai_chat("hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", messages)
```

The URL replaces the adapter's default (or the preset's: `vllm`,
`ollama`, `groq`, … each know their own) while the preset's dialect
stays. The cloud doors are the exception: Azure's and Bedrock's URLs
are *built* from a resource name or a region, so an entry for `azure-chat`
is refused and points you to `RouterConfig(settings=…)` — see
[Cloud hosts](cloud-hosts.md).

## The same call, from LiteLLM

LiteLLM speaks the OpenAI message format to every provider; so does
this door. Only the model string needs reading, and it is read the way
litellm reads it.

**Before — LiteLLM**

```python
response = litellm.completion(
    model="anthropic/claude-sonnet-4-5",
    messages=messages,
    max_tokens=100,
)
print(response.choices[0].message.content)
```

**After — lm15**

```python
response = router.complete_from_openai_chat(
    "anthropic/claude-sonnet-4-5",
    messages,
    max_tokens=100,
)
print(response.text)
```

```output
The sky is blue because molecules in Earth's atmosphere scatter shorter blue wavelengths of sunlight more than longer red wavelengths, a phenomenon called Rayleigh scattering.
```

That went to Anthropic's Messages API through lm15's Anthropic adapter,
with the key from `ANTHROPIC_API_KEY`. The rule for the string:

- only the **first** segment is the provider; a model id may contain
  slashes itself (`groq/openai/gpt-oss-20b` → Groq, model
  `openai/gpt-oss-20b`);
- an lm15 `provider:model` string is always accepted as-is;
- a litellm prefix lm15 has no door for — or one that names two doors
  (`bedrock/`, `vertex_ai/`: Anthropic or not, by model) — is refused
  by name. Write `bedrock-anthropic:…` or `vertex:…` yourself.

```python
from lm15.router import openai_chat_model_string

openai_chat_model_string("groq/openai/gpt-oss-20b")
openai_chat_model_string("bedrock/anthropic.claude-3")
```

```output
'groq:openai/gpt-oss-20b'
UnknownModelError: could not read 'bedrock/anthropic.claude-3' as a litellm model string: 'bedrock' is not a provider prefix lm15 has a door for (known: anthropic, azure, deepseek, gemini, groq, hosted_vllm, moonshot, ollama, ollama_chat, openai, openrouter, xai); write it as lm15's provider:model instead
```

## The chat loop: where you stop writing dictionaries

A chat loop appends the reply to the history and calls again. On the
dictionary door, the reply is a `Response`, not a dictionary — and lm15
does not turn it back into one (that would be a second, lossy format to
maintain). So the loop is the seam: read the dictionaries **once**, then
stay typed.

**Before**

```python
while True:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    messages.append(response.choices[0].message)
    messages.append({"role": "user", "content": input("> ")})
```

**After**

```python
from dataclasses import replace

request, lm = router.request_from_openai_chat("gpt-4o-mini", messages)
while True:
    response = lm.complete(request)
    request = replace(request, messages=request.messages + (
        response.message,
        lm15.Message.user(input("> ")),
    ))
```

`request_from_openai_chat` hands you the adapter along with the request
because the two belong together: the `Request` carries the bare wire
model (`gpt-4o-mini`), and `router.complete(request)` would route that
bare name to the Responses API. `lm.complete(request)` keeps the door
you started on. (To route through the router instead, set
`model="openai-chat:gpt-4o-mini"` on the request.)

`replace` is `dataclasses.replace`: lm15's types are immutable, so a new
turn is a new `Request` rather than a mutated list. Messages are a
tuple; `+ (…,)` appends.

### Your history probably holds SDK objects

The old loop did `messages.append(response.choices[0].message)`, so an
existing history mixes dictionaries with pydantic objects. Pass their
dictionary form and leave the dictionaries alone:

```python
history = [m if isinstance(m, dict) else m.model_dump() for m in messages]
response = router.complete_from_openai_chat("gpt-4o-mini", history)
```

Their `null`-valued keys and empty `annotations` read as absent;
non-empty `annotations` (web-search citations) become `CitationPart`s.
The exact dumps the SDK and litellm produce are pinned as contract
cases. An old SDK response you still hold on its own:
`lm15.response_from_openai_chat(old.model_dump())`.

## Streaming

`stream=True` is refused on `complete_from_openai_chat` — a boolean that
changes the return type is exactly the kind of thing lm15 makes
explicit. The streaming twin yields lm15's typed events, not
OpenAI-shaped chunks; `ResponseStream` assembles them into text and, at
the end, into the same `Response` the non-streaming call returns.

**Before**

```python
for chunk in client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**After**

```python
request, lm = router.request_from_openai_chat("gpt-4o-mini", messages, max_completion_tokens=100)
result = lm15.ResponseStream(lm.stream(request), request)
for text in result:
    print(text, end="", flush=True)
result.response.usage.output_tokens
```

```output
The sky appears blue due to the scattering of sunlight by the Earth's atmosphere, with shorter blue wavelengths being scattered more than other colors.
26
```

`router.stream_from_openai_chat(model, messages, **kwargs)` is the same
events without the assembly, for when you want tool calls and thinking
as they arrive.

## Client settings: each one has a place

`api_key` and `api_base` you have seen. The other keywords that
configure the *client* rather than the *request* are refused the same
way, each naming its lm15 home:

| you wrote | in lm15 it is |
|---|---|
| `api_key=` | `RouterConfig(api_keys={provider: key})`, or the environment |
| `api_base=`, `base_url=` | `RouterConfig(base_urls={provider: url})` |
| `timeout=`, `headers=`, `extra_headers=`, `extra_query=` | `RouterConfig(transport=…)` — see [Transports](using-the-transports.md) |
| `num_retries=`, `max_retries=` | your own loop over `lm15.RETRYABLE_ERRORS`; lm15 never retries on its own |
| `cache=`, `caching=` | your own cache keyed on the `Request`; lm15 has no response cache |
| `extra_body=` | `config.extensions` on the `Request` (build it, then edit) |
| `mock_response=` | `lm15.testing.FakeLM` |
| `drop_params=` | nothing: lm15 refuses what it cannot carry instead of dropping it |
| `n=` | refused: a `Response` is one message; loop in your code |

Two more differences that are not settings:

- **Provider-specific keywords are read with the destination's
  spelling.** `deepseek/…` reads DeepSeek's `thinking: {"type":
  "disabled"}`; `groq/…` reads `reasoning_format`; sending DeepSeek's
  spelling to OpenAI is refused instead of forwarded to a server that
  ignores it. Anthropic's native `thinking` object is not a Chat
  Completions key at all; say it as `reasoning_effort` (or build the
  `Request` and set `Config.reasoning`).
- **Cached `ModelResponse` objects** read one choice at a time:
  `lm15.response_from_openai_chat(cached.model_dump(), choice=i)`. Each
  carries the request's *total* usage, not a share.

## Structured output: a pydantic class is not JSON

`response_format=Out` (a pydantic class) is the one keyword neither door
can take as-is, because it is not JSON. Convert it to the `json_schema`
object first — and know the difference: the OpenAI SDK *rewrites* your
schema into its strict form (`additionalProperties: false` everywhere,
every property required) before sending; lm15 sends your schema
**verbatim** (INV-050) and lets the provider's 400 be the contract. So
`Out.model_json_schema()` with `strict: true` can be rejected where the
SDK's transformed copy was accepted. Either use the SDK's own converter
if it is installed
(`openai.lib._parsing._completions.type_to_response_format_param(Out)` —
a private path, so it may move), send `strict: false`, or write the
strict-form schema yourself.

---

The rest of this page is the contract of the converters underneath:
what every key does, how rows become messages, and where a round trip
cannot be exact. Read it when a refusal surprises you.

## The converters underneath

`complete_from_openai_chat` is three public pieces you can use on their
own: `lm15.router.openai_chat_model_string` (the model-string rule
above), `lm15.request_from_openai_chat(body, compat=None)` (the body →
`Request`), and the provider's own `parse_response`.

### Tell it which server the body was written for

Several OpenAI-compatible servers spell the same knob differently:
DeepSeek writes `thinking: {"type": "disabled"}` where OpenAI writes
`reasoning_effort: "none"`, Groq has `reasoning_format`, OpenRouter a
`reasoning` object. `compat` names the dialect whose spellings are read —
the same preset names `OpenAIChatLM` takes:

```python
lm15.request_from_openai_chat(body, compat="deepseek")
lm15.request_from_openai_chat(body, compat="groq")
```

A body carrying another server's spelling is **refused**, not forwarded:
a knob a server ignores is a silent paid no-op (hidden reasoning tokens
are billed), and lm15 does not guess which server a body was meant for.
On an adapter, `lm.request_from_openai_chat(body)` uses that adapter's
own compat, including per-model overrides; the router door does this
for you.

### What happens to each key

Every key has exactly one verdict, recorded as data in the contract
(`lm15-contract/tools/openai-chat-ingest-verdicts.json`) and checked
there against the scraped OpenAI reference, so a key nobody decided about
cannot exist quietly. The buckets (mapping rule MAP-12):

| Verdict | What it means | Examples |
|---|---|---|
| **map** | reads into a canonical field | `messages`, `tools`, `tool_choice`, `parallel_tool_calls`, `max_tokens` / `max_completion_tokens` (both, on every preset), `temperature`, `top_p`, `stop`, `logprobs` + `top_logprobs`, `response_format`, `service_tier`, `store`, `user` / `safety_identifier`, the reasoning and prompt-cache spellings of the chosen preset |
| **extensions** | passes verbatim into `config.extensions` and back out unchanged | `seed`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `metadata`, `verbosity`, `moderation` |
| **refuse** | `UnsupportedFeatureError` naming the key and where it belongs | `n` (lm15 reads one choice; fan out in the caller), the deprecated `functions` / `function_call` shape, `audio`, `modalities`, `prediction`, `web_search_options`, `top_k`, a per-message `name`, a `custom` tool, `strict: true` on a tool, a content block with no canonical part |
| **call-mode** | read and dropped — it says how the request is sent, not what is asked | `stream`, `stream_options` (pass `stream` to `complete()` / `stream()` instead) |
| **default** | equal to the wire's default; reads as absent | `response_format {"type": "text"}`, `strict: false`, `logprobs: false` |

A key with no verdict at all is refused. Malformed input — a wrong JSON
type, a tool call whose `arguments` string is not JSON, `max_tokens` and
`max_completion_tokens` that disagree — raises `ValueError` / `TypeError`,
like everything else in lm15's type system.

### Rows and content

- The first row, when `system` or `developer`, becomes `Request.system`;
  a later `system` / `developer` row becomes a `developer` message at
  that position.
- Consecutive `tool` rows become **one** tool message with one
  `ToolResultPart` per row (`tool_call_id` → `id`, `name` → `name`).
- An assistant row's parts come out in a fixed order: `reasoning_content`
  as a `ThinkingPart`, then `content`, then `refusal`, then `tool_calls`
  with `arguments` parsed. `content: null` with nothing else is one empty
  text part (a message is never empty).
- `text` → `TextPart`; `image_url` → `ImagePart` (a data URI becomes
  inline data; a URL stays a URL); `input_audio` → `AudioPart`; `file` →
  `DocumentPart`; `refusal` → `RefusalPart`.
- A `prompt_cache_breakpoint` on the system row is `prefix="stable"`; on
  the last text block of message *N* it is `prefix_until_index=N`.

### Where the round trip is not exact

`request_from_openai_chat(lm.build_request(r).body) == r` for every
request the chat adapter can carry losslessly — the contract checks this
over every recorded chat body in its corpus. Three things the **wire**
cannot carry do not come back, and each is named:

- A `ThinkingPart` replayed as text (`thinking_replay="as_text"`) reads
  back as text. lm15 never parses `<think>` blocks or other markers out of
  prose, in either direction.
- A `ToolResultPart.name` on a preset that omits it from the tool row.
- A leading `developer` message reads back as `Request.system`: the wire
  has one instruction row for both.

Likewise, a tool result whose text begins with `[error] ` (how the chat
wire carries `is_error`) reads back as text with `is_error=False`.
Reversing a prose marker would be a guess.

### The response, the other way

`response_from_openai_chat(body)` reads a Chat Completions **response**
body into a `Response` — the same reader the OpenAI Chat adapter runs on
provider traffic, so what it maps (text, `reasoning_content` as thinking,
tool calls, refusal, usage, logprobs, finish reason) and what it records
as unmapped are pinned by the contract's 302 recorded provider responses.
It takes a plain dict, which is what a cached litellm object gives you:

```python
resp = lm15.response_from_openai_chat(litellm_model_response.model_dump())
resp.text, resp.tool_calls, resp.usage.reasoning_tokens
```

Everything the reader does not know rides in `resp.provider_data`, whole;
an error envelope raises the typed provider error. Two rules to know:

- **More than one choice is refused unless you name one.** A `Response` is
  one message. A body produced with `n=3` raises until you read it three
  times with `choice=0`, `1`, `2` — the twin of the request side refusing
  `n`. (The adapter's own `parse_response` refuses too; a user who
  smuggled `n` through `config.extensions` used to get the first choice
  silently.)
- **`model=` fills in what the body lacks.** Servers echo the model; some
  caches strip it.

There is no `compat` argument: the Chat Completions *response* shape does
not vary by server the way the request does, and a parameter that did
nothing would be a lie.

## For framework authors

If your framework's adapter already produces OpenAI-format messages —
DSPy's `ChatAdapter` and `JSONAdapter` do — your typed-request constructor
is one line:

```python
def from_call(model: str, messages: list[dict], **kwargs) -> lm15.Request:
    return lm15.request_from_openai_chat({"model": model, "messages": messages, **kwargs})
```

Render anything that is not JSON first (a pydantic model class passed as
`response_format` must become the `json_schema` object the OpenAI SDK
would send). The exact bodies DSPy's adapters produce are pinned as
contract cases, so a change in either project that breaks this line is
caught on both sides.

## What this does not do

It reads the Chat Completions format only, in both directions. It does
not read the Responses API, Anthropic or Gemini formats; it does not turn
a `Response` back into an OpenAI-shaped response object for serving
behind an OpenAI-compatible endpoint (a later, separate feature); it does
not read streaming chunks. And it is not a `Request.from_...` constructor:
the canonical types stay vendor-free; the converters live beside the
dialect they invert.

## Where to go next

- [Getting started](getting-started.md) — the `Request` / `Response`
  loop you are now standing in, from the beginning.
- [Using the router](using-the-router.md) — the model-string grammar and
  how resolution picks a door.
- [Authentication](authentication.md) — subscriptions, rotating tokens,
  and every other way a key can arrive.
- [Streaming](cookbooks/05-streaming.md) and
  [Function tools](cookbooks/06-function-tools.md) — the two things the dictionary format
  carries least well, done natively.
