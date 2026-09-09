# Migrating from the OpenAI SDK or LiteLLM

You do not need to write JSON strings or rebuild your message history.
Keep your Python message dictionaries, collect your request arguments into
one dictionary, and let `request_from_openai_chat` turn it into an lm15
`Request`. Unsupported fields raise an error naming what could not be
carried.

The examples below use non-streaming calls. Install lm15 with
`pip install lm15` and keep your provider's API key in the environment.
This is a migration, not a drop-in replacement: requests become lm15
`Request` objects and answers become lm15 `Response` objects.

## 1. From the OpenAI SDK to lm15

This section covers `client.chat.completions.create`, not the OpenAI
Responses API (`client.responses.create`). Set `OPENAI_API_KEY` as usual.

### Before: an OpenAI SDK call

```python
from openai import OpenAI

messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_completion_tokens=100,
)
print(response.choices[0].message.content)
```

### After: keep the messages, change the call

```python
import lm15

messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]

router = lm15.LMRouter()  # reuse this across calls
request = router.complete(lm15.request_from_openai_chat(
    model="gpt-4o-mini",
    messages=messages,
    max_completion_tokens=100,
))
print(response.text)
```

`openai-chat:` explicitly keeps the Chat Completions endpoint; it is a
routing prefix, not part of the model id sent to OpenAI. The converter
moves the first system row to `request.system` and the token limit to
`request.config.max_tokens`. You still hold ordinary lm15 types.

### If your history contains SDK message objects

A chat loop may have appended `old_response.choices[0].message` directly.
Convert those objects to dictionaries before passing the history in;
leave existing dictionaries alone:

```python
message_dicts = [
    msg if isinstance(msg, dict) else msg.model_dump()
    for msg in messages
]
request = lm15.request_from_openai_chat({
    "model": "openai-chat:gpt-4o-mini",
    "messages": message_dicts,
})
```

To read an existing OpenAI response without making another network call:

```python
converted = lm15.response_from_openai_chat(old_response.model_dump())
print(converted.text)
```

Once you use lm15 throughout your loop, keep `response.message` as an
lm15 message rather than trying to call `.model_dump()` on it. lm15
objects are not Pydantic objects.

## 2. From LiteLLM to lm15

LiteLLM also accepts OpenAI-style message dictionaries when calling other
providers. The converter reads that message format; the router chooses
where to send the resulting request. Here is an Anthropic example with
`ANTHROPIC_API_KEY` set.

### Before: a LiteLLM call to Anthropic

```python
import litellm

messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5",
    messages=messages,
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### After: the same messages, sent through lm15

```python
import lm15

messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]

router = lm15.LMRouter()
request = lm15.request_from_openai_chat({
    "model": "anthropic:claude-sonnet-4-5",
    "messages": messages,
    "max_tokens": 100,
})
response = router.complete(request)
print(response.text)
```

Despite the converter's name, this call goes to Anthropic's Messages API,
not OpenAI. No `compat="anthropic"` is needed: the input here is a set
of ordinary OpenAI-style message rows and common request arguments.

### What else changes?

- **Provider prefixes:** replace the known LiteLLM routing prefix, such
  as `anthropic/`, with lm15's `anthropic:`. Do not replace every slash:
  model ids can contain slashes themselves. For example, a Groq call to
  `openai/gpt-oss-20b` uses `groq:openai/gpt-oss-20b` in lm15.
- **Client settings:** keep credentials, timeouts, headers and retries
  out of the request dictionary. Configure credentials and transport
  separately; caching, retries and multiple samples need caller-side
  handling rather than being silently inherited from LiteLLM.
- **Provider-specific arguments:** not every LiteLLM keyword maps.
  For example, Anthropic's native `thinking` object is not an OpenAI
  request field; express that intent through lm15's `Config.reasoning`
  instead. An accepted request can still be refused at send time if the
  destination provider cannot carry one of its fields.
- **Existing objects:** use the same `msg.model_dump()` conversion shown
  above for LiteLLM message objects in history. To read a whole cached
  `ModelResponse`, use the response converter:

```python
converted = lm15.response_from_openai_chat(cached_response.model_dump())
print(converted.text)
```

If that cached response contains multiple choices, explicitly select one
with `choice=0`, `choice=1`, and so on. Each converted response carries
the original request's total usage, not a per-choice share; do not add
those usage values together.

Streaming is a separate migration: lm15 emits typed stream events, not
LiteLLM's OpenAI-shaped chunks. These converters do not convert chunks.

## Handling Python objects and structured output

For either SDK, these are the inputs the converters accept:

- **Keyword arguments** to `create(model=…, messages=…, **kwargs)`: the
  body is `{"model": model, "messages": messages, **kwargs}`. Strip the
  client's transport knobs first (`api_key`, `api_base`, `timeout`,
  `num_retries`, `headers`, …) — they are not request content and would
  be refused by name.
- **Message objects appended back into history** —
  `messages.append(response.choices[0].message)`, the first thing every
  chat loop does. Pass `msg.model_dump()`; its null-valued keys and empty
  `annotations` read as absent, non-empty `annotations` (web-search
  citations) become `CitationPart`s, and litellm's own
  `provider_specific_fields` reads as absent when empty. These exact
  dumps are pinned as contract cases.
- **Response objects**: `response_from_openai_chat(resp.model_dump())`.
- **A pydantic class as `response_format`**: not JSON. Convert it to the
  `json_schema` object first — and know the difference: the OpenAI SDK
  *rewrites* your schema into its strict form (`additionalProperties:
  false` everywhere, every property required) before sending; lm15 sends
  your schema **verbatim** (INV-050) and lets the provider's 400 be the
  contract. So `Out.model_json_schema()` with `strict: true` can be
  rejected where the SDK's transformed copy was accepted. Either use the
  SDK's own converter if it is installed
  (`openai.lib._parsing._completions.type_to_response_format_param(Out)`
  — a private path, so it may move), or send `strict: false`, or write
  the strict-form schema yourself.

## Tell it which server the body was written for

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
own compat, including per-model overrides.

## What happens to each key

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

## Rows and content

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

## Where the round trip is not exact

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

## The response, the other way

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
  `n`. (The adapter's own `parse_response` now refuses too; a user who
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
