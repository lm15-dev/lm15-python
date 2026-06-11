# Multi-turn conversations

**Problem** — you want the model to remember earlier turns, but lm15 has
no session object, no `chat.append()`, no hidden state. The conversation
is a tuple of `Message` values that you build, extend, and persist
yourself — which also means you can inspect, edit, and replay it.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

A conversation starts as a tuple with one user message. `complete()`
returns a `Response`; `response.text` is the assistant's reply as a
string.

```python
import json
import pathlib

from lm15 import LMRouter, Message, Request, messages_from_json, messages_to_json

router = LMRouter()
history = (Message.user("Pick a prime number between 80 and 90. Answer with the number only."),)

response = router.complete(Request(model="gpt-4.1-mini", messages=history))
print(response.text)
```
```output
83
```

To take another turn, build a *new* tuple: the old history, the
assistant's full message (`response.message`, not just its text), and
your next user message. Messages are frozen; `(*history, ...)` is the
append operation.

```python
history = (*history, response.message, Message.user("Add 6 to it. Number only."))

response = router.complete(Request(model="gpt-4.1-mini", messages=history))
print(response.text)
```
```output
89
```

`Message.text` is the per-message convenience: the joined text when the
message is text-only, `None` if it carries any other part (a tool call,
an image). That strictness is deliberate — a transcript printer that
silently drops tool calls is a bug factory.

```python
history = (*history, response.message)
for m in history:
    print(f"{m.role:>9}: {m.text}")
```
```output
     user: Pick a prime number between 80 and 90. Answer with the number only.
assistant: 83
     user: Add 6 to it. Number only.
assistant: 89
```

To persist the conversation, `messages_to_json` produces plain dicts in
lm15's canonical JSON shape — `json.dumps` away from disk.

```python
path = pathlib.Path("conversation.json")
path.write_text(json.dumps(messages_to_json(history), indent=2))
print(path.read_text()[:200])
```
```output
[
  {
    "role": "user",
    "parts": [
      {
        "type": "text",
        "text": "Pick a prime number between 80 and 90. Answer with the number only."
      }
    ]
  },
…
```

Loading it back round-trips to equal `Message` values. The history is
provider-neutral, so the same transcript can continue on a different
model.

```python
restored = tuple(messages_from_json(json.loads(path.read_text())))
print(restored == history)

response = router.complete(Request(
    model="gemini-3-flash-preview",
    messages=(*restored, Message.user("What number did we end on, and is it prime?")),
))
print(response.text)
```
```output
True
We ended on 89, and it is a prime number.
```

## How it works

Chat APIs are stateless: every request carries the whole conversation,
and the provider replays it. lm15 mirrors that on the surface instead of
hiding it. `Request.messages` is a tuple of frozen `Message` values;
each `Message` is a role plus a tuple of typed parts (`TextPart`,
`ToolCallPart`, …). Nothing in the library mutates or remembers them —
the LM objects and the router hold no conversation state, so the same
`history` tuple can be sent to two providers, forked into two branches,
or trimmed for context-window budget, all with ordinary tuple code.

Append `response.message`, never a hand-built
`Message.assistant(response.text)`. The full message preserves
non-text parts — tool calls, thinking blocks, provider continuation
state — that the next turn may depend on. `Response.text` exists for
display; `response.message` is the thing that goes back on the wire.

`messages_to_json` / `messages_from_json` implement lm15's canonical
JSON form for messages — the format specified in
[Serde rules](../serde-rules.md), not any provider's wire shape. That is
why a transcript saved mid-conversation with OpenAI loads and continues
on Gemini: translation to provider wire format happens later, inside
each LM's `build_request`. Binary parts (images, audio) serialize too;
their bytes are base64 inside the JSON, so transcripts with media get
large.

What lm15 deliberately does not do: truncation, summarization, sliding
windows, token counting. Context management is policy, and the tuple is
the whole interface for implementing yours.

## Variations

- **Async mirror.** Identical shapes with `AsyncLMRouter`; only the call
  is awaited:

  ```python
  from lm15 import AsyncLMRouter

  router = AsyncLMRouter()
  response = await router.complete(Request(model="gpt-4.1-mini", messages=history))
  history = (*history, response.message)
  ```

- **Mid-conversation instructions.** `Message.developer("...")` injects
  high-authority instructions at any position without restarting the
  conversation. OpenAI sees a native `developer` role; Anthropic and
  Gemini get a user message with a `[developer]` prefix. See
  [recipe 03](03-system-prompts.md).
- **Assistant prefill.** End `messages` with a partial
  `Message.assistant(...)` and the model continues it — Anthropic honors
  the prefix verbatim; Gemini usually continues it; OpenAI's Responses
  API tends to answer fresh instead:

  ```python
  req = Request(
      model="claude-sonnet-4-5",
      messages=(
          Message.user("What is 2 + 2? Answer in one short sentence."),
          Message.assistant("Arr, as a pirate I"),
      ),
  )
  print(repr(router.complete(req).text))
  ```
  ```output
  "'d say 2 + 2 be 4, matey!"
  ```

- **Branching.** Tuples make forks free: keep `base = history[:4]` and
  extend it two different ways; the shared prefix is the same objects,
  which also keeps provider prompt caches warm
  ([recipe 11](11-caching.md)).
- **`Message.text` vs `Response.text`.** `Message.text` is `None` for
  any mixed-part message. `Response.text` is slightly more forgiving on
  its own `message` — it joins the `TextPart`s of a reply that also
  carries thinking or citations. For transcripts, filter explicitly with
  `m.parts_of(TextPart)`.
- **Lists are rejected loudly.** `Message.parts` must be `Part` objects;
  passing a raw string raises
  `TypeError: Message.parts must be Part objects; use Message.user('text') for strings`.
  The factory methods (`Message.user`, `.assistant`, `.developer`,
  `.tool`) are the intended constructors.

## See also

- [01 — Your first request](01-first-request.md) — router setup and key loading.
- [03 — System & developer prompts](03-system-prompts.md) — where instructions go.
- [06 — Function tools](06-function-tools.md) — tool calls and results in the history tuple.
- [11 — Prompt caching](11-caching.md) — long shared prefixes, paid once.
- [Serde rules](../serde-rules.md) — the canonical JSON format behind `messages_to_json`.
