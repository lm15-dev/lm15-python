# Migrating from an OpenAI-shaped client

You already have requests in the format almost every tool speaks: a JSON
object with `model`, a list of `{"role", "content"}` rows, maybe `tools`
and a few knobs. A framework built it, a log recorded it, or the client
library you are leaving expected it. `request_from_openai_chat` reads that
object into a canonical `Request`, or tells you exactly which key it could
not carry.

```python
import lm15

body = {
    "model": "gpt-5-mini",
    "messages": [
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    "temperature": 0.2,
    "max_tokens": 100,
}

request = lm15.request_from_openai_chat(body)
print(request.system)              # 'Answer in one sentence.'
print(request.config.max_tokens)   # 100

lm15.LMRouter().complete(request)  # now any provider can answer it
```

The result is an ordinary `Request`. Nothing is hidden behind the
converter; from here on you are using lm15's types.

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
