# Streaming

**Problem** — You want tokens on screen as the model produces them, but
you also want the complete `Response` at the end — without parsing SSE,
without accumulating deltas by hand, and without writing the code twice
for streaming and non-streaming paths.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

`router.stream()` returns typed `StreamEvent`s. Wrap them in `Result`
and the stream becomes something you can iterate as text:

```python
from lm15 import AsyncLMRouter, LMRouter, Message, Request, Result

router = LMRouter()
req = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Name three rivers in Quebec, one per line, names only."),),
)
result = Result(events=router.stream(req), request=req)
for text in result:
    print(text, end="", flush=True)
```
```output
Saint Lawrence
Rimouski
Saguenay
```

While text streamed past, `Result` was accumulating it. After the loop,
the materialized response is already there — `.text` does not re-call
the API:

```python
print(result.text)
print(result.finish_reason, result.usage)
```
```output
Saint Lawrence
Rimouski
Saguenay
stop Usage(input_tokens=20, output_tokens=12, total_tokens=32, …)
```

Iterating a `Result` yields only text. For everything else — thinking,
tool calls, images, audio — use `.events()`, which yields typed
`StreamChunk`s. The final chunk is always `finished`, carrying the
complete `Response`:

```python
req = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Say 'streams are lazy' and nothing else."),),
)
result = Result(events=router.stream(req), request=req)
for chunk in result.events():
    print(chunk.type, repr(chunk.text))
```
```output
text 'streams'
text ' are'
text ' lazy'
finished None
```

You do not have to iterate at all. Touching `.response` (or `.text`,
`.usage`, …) drains the stream and blocks until it is done. What you
get is an ordinary `Response` — the same type, the same fields, as
`router.complete()` returns:

```python
result = Result(events=router.stream(req), request=req)
response = result.response          # drains the stream, blocks
print(type(response).__name__)
print(response.message.parts)
print(response.finish_reason, response.usage)
```
```output
Response
(TextPart(text='streams are lazy', continuation=(), type='text'),)
stop Usage(input_tokens=17, output_tokens=4, total_tokens=21, …)
```

The stream contract is strict. Every lm15 stream is a `start` event,
zero or more `delta` events, and **exactly one** `end` event, last.
This is rule MAP-3, and it holds even when the provider's wire format
emits several terminal frames:

```python
events = list(router.stream(req))
print("first:", events[0].type, "last:", events[-1].type)
print("end events:", sum(e.type == "end" for e in events))
print(events[-1])
```
```output
first: start last: end
end events: 1
StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=17, output_tokens=4, total_tokens=21, …),
    provider_data=…,
    type='end',
)
```

`finish_reason` and `usage` ride on that single end event, which is why
`Result` can always hand you a complete `Response`.

## How it works

`router.stream()` resolves the model string, opens the provider's SSE
connection, and translates each wire frame into a `StreamEvent` — the
same vocabulary (`TextDelta`, `ThinkingDelta`, `ToolCallDelta`,
`ImageDelta`, …) across OpenAI, Anthropic and Gemini. See
[using the router](../using-the-router.md) for resolution; the event
vocabulary lives in `lm15.types`.

Providers disagree about endings: OpenAI sends a finish-reason chunk,
then a usage-only chunk, then `[DONE]`; Anthropic sends
`message_delta` plus `message_stop`. Adapters are stateless and emit
one end event per terminal frame; `coalesce_stream()` (applied inside
every provider's `stream()`) merges them into the single final
`StreamEndEvent` you observed above. If a stream errors or is cut off
mid-flight, no end event is fabricated — absence of `end` means the
stream did not finish.

`Result` is a pure materializer on top of those events. It buffers
deltas by part index as they pass through, so iterating costs nothing
extra, and `materialize()` reassembles them into `Message` parts. It
executes nothing: tool-call deltas surface as data (recipe
[06](06-function-tools.md)), and any run-tools-and-continue loop is
yours to write. There is no retry, no timeout policy, no reconnection
— lm15 hands you the events; policy is the layer above.

## Variations

- **Async.** `AsyncLMRouter.stream()` returns an async iterator of the
  same events; consume it with `async for`. This ran against Gemini:

  ```python
  import asyncio

  async def main():
      arouter = AsyncLMRouter()
      req = Request(
          model="gemini-3-flash-preview",
          messages=(Message.user("Count from 1 to 5, comma-separated."),),
      )
      async for event in arouter.stream(req):
          if event.type == "delta" and event.delta.type == "text":
              print(event.delta.text, end="", flush=True)
      print()

  asyncio.run(main())
  ```
  ```output
  1, 2, 3, 4, 5
  ```

  The MAP-3 guarantee is identical; `acoalesce_stream` is the async
  mirror of the coalescer.

- **Raw events, no `Result`.** Filtering `router.stream(req)` yourself,
  as in the async example, is fine when you only want one delta type.
  `Result` earns its keep when you want the materialized `Response`
  afterward.

- **Delta granularity differs by provider.** OpenAI streams a few
  tokens per delta; Gemini sends larger sentence-sized deltas;
  Anthropic sits in between. Your code should not depend on chunk
  boundaries.

- **Replay.** `lm15.result.response_to_events(response)` converts a
  complete `Response` back into a stream — useful for testing stream
  consumers without a network (recipe
  [15](15-errors-and-testing.md)).

## See also

- [01 — Your first request](01-first-request.md) — keys and the router front door.
- [06 — Function tools](06-function-tools.md) — tool-call parts surfaced by `Result`.
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md) — thinking deltas.
- [15 — Errors, retries & testing](15-errors-and-testing.md) — stream errors, offline replay.
- [Using the router](../using-the-router.md) — resolution rules and `RouterConfig`.
