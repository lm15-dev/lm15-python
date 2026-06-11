# System & developer prompts

**Problem** — Your instructions ("you are X, never do Y") must not mix
with user input, but each provider puts them somewhere different on the
wire: OpenAI has `instructions` and a native `developer` role, Anthropic
has a `system` field and no developer role, Gemini has
`systemInstruction`. lm15 gives you two portable surfaces —
`Request.system` and `Message.developer()` — and maps both.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

`Request.system` is the standing instruction for the whole request. A
plain string is the common case.

```python
from lm15 import LMRouter, Message, Request, TextPart

router = LMRouter()

req = Request(
    model="gpt-4.1-mini",
    system="You are a maritime lawyer. Answer in one sentence.",
    messages=(Message.user("Who owns a shipwreck in international waters?"),),
)
print(router.complete(req).text)
```
```output
In international waters, a shipwreck is generally considered abandoned
property and ownership depends on maritime salvage laws, but the original
owner retains rights unless they explicitly relinquish them or …
```

`system` also accepts a tuple of parts. Useful when you assemble the
prompt from pieces — a persona you reuse, constraints that vary per
deployment — and want to keep them as separate values until send time.

```python
persona = TextPart(text="You are a code reviewer for a Python team.")
rules = TextPart(text="Rules:\n- Flag mutable default arguments.\n- Keep feedback under 50 words.")
req = Request(
    model="gpt-4.1-mini",
    system=(persona, rules),
    messages=(Message.user("def add(item, items=[]): items.append(item); return items"),),
)
print(router.complete(req).text)
```
```output
Avoid using mutable default arguments like `items=[]`; it can cause
unexpected behavior. Use `None` as the default and initialize inside the
function:
…
```

`system` is fixed for the request. To inject an instruction
*mid-conversation* — without rewriting the system prompt and
invalidating any cached prefix built from earlier turns — use a
`Message.developer()` in the message tuple. It carries higher authority
than user messages and can appear anywhere.

```python
history = (
    Message.user("Recommend a sci-fi novel."),
    Message.assistant("Try *A Fire Upon the Deep* by Vernor Vinge — galaxy-spanning, big ideas."),
    Message.developer("From now on, answer in French only."),
    Message.user("Another one?"),
)
req = Request(model="gpt-4.1-mini", messages=history)
print(router.complete(req).text)
```
```output
Bien sûr ! Je recommande *Les Dossiers Dresden* de Jim Butcher, une série
qui mélange science-fiction et fantastique avec un détective spécialisé
dans le surnaturel.
```

On OpenAI that message went out with the native `developer` role.
Anthropic and Gemini have no such role, so their providers rewrite it as
a user message prefixed with `[developer]` — the model still sees the
instruction boundary, and the instruction still lands:

```python
req = Request(model="gemini-3-flash-preview", messages=history)
print(router.complete(req).text)
```
```output
Je vous suggère **"Hypérion" de Dan Simmons**.

C'est un chef-d'œuvre absolu du *space opera*. L'histoire suit six
pèlerins qui voyagent vers la planète Hypérion …
```

## How it works

`Request.system` is normalized at construction: a string stays a string,
anything else goes through part normalization to a
`tuple[PromptPart, ...]`. An empty string raises `ValueError`; protocol
parts (tool calls, tool results, citations) raise `TypeError` — system
content is instructions, not transcript. On the wire:

- **OpenAI (Responses)** — `system` becomes the top-level
  `instructions` field; parts are flattened to text.
- **Anthropic** — the `system` field; with a cache breakpoint it becomes
  a content block carrying `cache_control` (recipe 11).
- **Gemini** — `systemInstruction`, one text part.

`Message.developer()` is a fourth role alongside `user`, `assistant`,
and `tool`. OpenAI keeps it as `role: "developer"` (configurable via
compat — older Chat Completions servers want `"system"`; recipe 14).
Anthropic and Gemini providers convert it to a user message whose text
is prefixed with `[developer]\n`. That is a convention, not an API
guarantee: on those providers a developer message has whatever authority
the model grants a clearly-marked user turn, which is less than the real
system prompt. Put standing rules in `system`; use `developer` for
mid-conversation steering.

lm15 does not merge, template, or dedupe your prompts. One `system` per
request, your messages verbatim — what you build is what is sent.

## Variations

- **Async** is the same shape: `await AsyncLMRouter().complete(req)`.
  `Request` construction and `Message.developer()` are identical.
- **Persona vs. constraints.** A pattern that survives provider swaps:
  persona and immutable rules in `system`; per-turn steering ("answer in
  French", "the user is now an admin") as `developer` messages appended
  to history. The system prefix never changes, so provider-side prompt
  caching keeps hitting (recipe 11).
- **Authority differs by provider.** OpenAI documents
  `developer` > `user` in its instruction hierarchy. Anthropic and
  Gemini see a `[developer]`-prefixed user message — usually followed,
  never privileged. If an instruction must win against adversarial user
  input, put it in `system`.
- **A leading `developer` message is not a `system` substitute.** On
  OpenAI a `Message.developer()` at position 0 reads like instructions;
  on Anthropic and Gemini it is still a `[developer]`-prefixed user
  turn, while `system` maps to the provider's native instruction field.
  Use `system` for the standing prompt on every provider.
- **Fail-fast validation.** `Request(system="")` raises `ValueError`
  and protocol parts (tool calls, citations) in `system` raise
  `TypeError` — both at construction, before any network call
  (recipe 15).
- **System parts are text-only in practice.** All three providers
  flatten system parts to text before sending. Images or documents in
  instructions belong in a `user` or `developer` message instead
  (recipe 09).

## See also

- [02 — Multi-turn conversations](02-conversations.md)
- [04 — Controlling generation](04-generation-config.md)
- [11 — Prompt caching](11-caching.md)
- [14 — Local & OpenAI-compatible servers](14-local-and-compatible-servers.md)
- [Using the router](../using-the-router.md)
