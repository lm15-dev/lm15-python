# Migrating from the OpenAI SDK or LiteLLM

Keep your message dictionaries and your model string exactly as they
are. `LMRouter.complete_from_openai_chat(model, messages, **kwargs)` takes
the same arguments as `client.chat.completions.create(...)` and
`litellm.completion(...)` and answers with an lm15 `Response`. A keyword
lm15 cannot carry raises an error naming it; nothing is dropped.

The examples use non-streaming calls and read keys from the environment
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). This is a migration, not a
drop-in: the *answer* is an lm15 `Response` (`response.text`), not the
SDK's object.

## 1. From the OpenAI SDK

### Before

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

### After

```python
import lm15

messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]

router = lm15.LMRouter()  # reuse it; it keeps one connection pool per provider
response = router.complete_from_openai_chat(
    "gpt-4o-mini",
    messages,
    max_completion_tokens=100,
)
print(response.text)
```

Three things happened under that call, all visible if you want them:

- The bare `gpt-4o-mini` went to OpenAI's **Chat Completions** endpoint —
  the one the SDK was using — not the Responses API that
  `router.complete(Request(model="gpt-4o-mini", …))` would pick. Same
  string, different door, on purpose; write `openai:gpt-4o-mini` if you
  want Responses.
- `messages` were read by `request_from_openai_chat` (the system row
  became `request.system`, the token limit `request.config.max_tokens`).
- The reply was parsed by lm15's own OpenAI Chat adapter.

To see the `Request` instead of sending it:

```python
request, lm = router.request_from_openai_chat("gpt-4o-mini", messages, max_completion_tokens=100)
```

### Your history probably holds SDK objects

Every chat loop does `messages.append(response.choices[0].message)`.
Those are pydantic objects; pass their dict form and leave existing
dicts alone:

```python
history = [m if isinstance(m, dict) else m.model_dump() for m in messages]
response = router.complete_from_openai_chat("gpt-4o-mini", history)
```

Their `null`-valued keys and empty `annotations` read as absent;
non-empty `annotations` (web-search citations) become `CitationPart`s.
The exact dumps the SDK produces are pinned as contract cases. Once the
loop runs on lm15, append `response.message` (an lm15 message) instead.

An old SDK response you still hold: `lm15.response_from_openai_chat(old.model_dump())`.

## 2. From LiteLLM

LiteLLM speaks the same message format to every provider; so does this
door. The model string is read the way litellm reads it.

### Before

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

### After

```python
import lm15

router = lm15.LMRouter()
response = router.complete_from_openai_chat(
    "anthropic/claude-sonnet-4-5",
    messages,
    max_tokens=100,
)
print(response.text)
```

That call went to Anthropic's Messages API through lm15's Anthropic
adapter. `gemini/gemini-3.8-flash`, `groq/openai/gpt-oss-20b`,
`openrouter/…`, `deepseek/…`, `xai/…`, `ollama/…` work the same way:
only the **first** segment is the provider (a model id may contain
slashes itself), and an lm15 `provider:model` string is always accepted
as-is. A litellm prefix lm15 has no door for — or one that covers two
doors (`bedrock/`, `vertex_ai/`) — is refused by name; write
`bedrock-anthropic:…` or `vertex:…` yourself.

### What else changes

- **Client settings are refused, with the lm15 place named.** `api_key`,
  `api_base`, `timeout`, `num_retries`, `headers`, `cache`,
  `drop_params`, … configure the client, not the request:
  `LMRouter(RouterConfig(api_keys=…, base_urls=…, transport=…))`. lm15
  never retries or caches on its own; `RETRYABLE_ERRORS` is data for your
  loop.
- **Provider-specific keywords are read with the destination's spelling.**
  `deepseek/…` reads DeepSeek's `thinking: {"type": "disabled"}`;
  `groq/…` reads `reasoning_format`; sending DeepSeek's spelling to
  OpenAI is refused instead of forwarded to a server that ignores it.
  Anthropic's native `thinking` object is not a Chat Completions key at
  all; say it as `reasoning_effort` (or build the `Request` and set
  `Config.reasoning`). A key the destination cannot carry fails at send
  time, loudly — lm15 does not `drop_params`.
- **`n` is refused.** A `Response` is one message; loop in your code.
- **Cached `ModelResponse` objects:**
  `lm15.response_from_openai_chat(cached.model_dump(), choice=i)` per
  choice — each carries the request's *total* usage, not a share.
- **Streaming:** `router.stream_from_openai_chat(...)` yields lm15's typed
  events, not OpenAI-shaped chunks; these doors never convert chunks.

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

## The converters underneath

`complete_from_openai_chat` is three public pieces you can use on their
own: `openai_chat_model_string` (the model-string rule above),
`request_from_openai_chat(body, compat=None)` (the body → `Request`), and
the provider's own `parse_response`. The rest of this page is the
contract of the body converter.

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
