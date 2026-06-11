# Built-in provider tools

**Problem** — All three providers can search the web or run code on
their own servers, but each names the tool differently on the wire:
OpenAI wants `web_search_preview`, Anthropic `web_search_20250305`,
Gemini `googleSearch`. lm15 gives you one `BuiltinTool` with canonical
names — `"web_search"`, `"code_execution"` — and each adapter maps them
to its native spelling. Unlike [function tools](06-function-tools.md),
you enable these; the provider executes them.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

A `BuiltinTool` is a name plus an optional provider-specific `config`
dict. Put it in `Request.tools` like any other tool:

```python
from lm15 import BuiltinTool, LMRouter, Message, Request

router = LMRouter()
search = BuiltinTool("web_search")

response = router.complete(Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Who won the most recent Super Bowl? One sentence."),),
    tools=(search,),
))
print(response.text)
```
```output
The Seattle Seahawks won Super Bowl LX on February 8, 2026, defeating the
New England Patriots 29-13 at Levi's Stadium in Santa Clara, California.
([pro-football-reference.com](https://www.pro-football-reference.com/super-bowl/…))
```

No tool-call loop on your side: the search happened on OpenAI's
servers, and the answer came back grounded. The sources arrive as
`CitationPart` objects in the assistant message; `response.citations`
collects them:

```python
print([p.type for p in response.message.parts])
for c in response.citations:
    print(c.title, "|", c.url)
```
```output
['text', 'citation']
Super Bowl History | Pro-Football-Reference.com | https://www.pro-football-reference.com/super-bowl/?utm_source=openai
```

The same `BuiltinTool` works unchanged on the other two providers —
only the model string changes:

```python
for model in ("claude-sonnet-4-5", "gemini-3-flash-preview"):
    response = router.complete(Request(
        model=model,
        messages=(Message.user("Who won the most recent Super Bowl? One sentence."),),
        tools=(search,),
    ))
    print(model, "->", response.text)
    for c in response.citations[:1]:
        print("  cite:", c.title, "|", c.url, "|", repr(c.text)[:50])
```
```output
claude-sonnet-4-5 -> The Seattle Seahawks defeated the New England Patriots 29-13 in Super Bowl LX on February 8, 2026.
  cite: Super Bowl LX - Wikipedia | https://en.wikipedia.org/wiki/Super_Bowl_LX | 'The National Football Conference (NFC) champion…
gemini-3-flash-preview -> The Seattle Seahawks won the most recent Super Bowl, defeating the New England Patriots 29–13 in Super Bowl LX on February 8, 2026.
  cite: wikipedia.org | https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQES… | 'The Seattle Seahawks won the most recent…
```

A `CitationPart` has three optional fields — `url`, `title`, `text`
(the cited span) — and providers fill them unevenly: Anthropic gives a
page title and direct URL, Gemini gives a domain as title and a
grounding-redirect URL.

`code_execution` runs model-written code in a provider sandbox. OpenAI
requires a `container` parameter for its code interpreter; that is what
`config` is for — keys merge into the wire-format tool object verbatim:

```python
prompt = (Message.user(
    "Run code to compute the sum of the first 1000 primes. Reply with the number only."
),)
runs = (
    ("gpt-4.1-mini", BuiltinTool("code_execution", config={"container": {"type": "auto"}})),
    ("claude-sonnet-4-5", BuiltinTool("code_execution")),
    ("gemini-3-flash-preview", BuiltinTool("code_execution")),
)
for model, sandbox in runs:
    response = router.complete(Request(model=model, messages=prompt, tools=(sandbox,)))
    print(model, "->", response.text)
```
```output
gpt-4.1-mini -> 3682913
claude-sonnet-4-5 -> 3682913
gemini-3-flash-preview -> 3682913
```

Three sandboxes, one answer. Omitting `config` for OpenAI raises
`InvalidRequestError: Missing required parameter: 'tools[0].container'`
— lm15 does not invent defaults for provider-required fields.

`config` also carries provider knobs. Anthropic's web search takes
`max_uses`; the raw server-side machinery stays visible in
`provider_data` if you want it:

```python
capped = BuiltinTool("web_search", config={"max_uses": 2})
response = router.complete(Request(
    model="claude-sonnet-4-5",
    messages=(Message.user("What is the current population of Gatineau, Quebec? One sentence."),),
    tools=(capped,),
))
print(response.text)
print([block["type"] for block in response.provider_data["content"]])
```
```output
The current estimated population of Gatineau, Quebec is 309,193.
['server_tool_use', 'web_search_tool_result', 'text']
```

## How it works

Each adapter holds a small map from canonical names to its native tool
spelling:

| canonical        | OpenAI (Responses)   | Anthropic                 | Gemini          |
| ---------------- | -------------------- | ------------------------- | --------------- |
| `web_search`     | `web_search_preview` | `web_search_20250305`     | `googleSearch`  |
| `code_execution` | `code_interpreter`   | `code_execution_20250522` | `codeExecution` |

A name not in the map passes through verbatim, so new provider tools
work before lm15 learns their canonical alias. `config` keys merge into
the tool object as-is — they are provider vocabulary, not lm15's, which
is why the OpenAI `container` dict above would be rejected by Anthropic
if you reused it there.

On the way back, the provider's intermediate blocks — OpenAI
`web_search_call` items, Anthropic `server_tool_use` /
`*_tool_result` blocks, Gemini `executableCode` parts — are *not*
turned into `ToolCallPart`s. They are server-side execution you cannot
respond to, so lm15 keeps `message.parts` to what you act on: text and
citations. The full raw payload stays in `response.provider_data`.

Two provider quirks lm15 handles for you: Anthropic's `code_execution`
needs the `anthropic-beta: code-execution-2025-05-22` header, added
automatically when the tool is present; Anthropic's wire format also
wants a `name` field alongside `type`, which the adapter fills in.

## Variations

- **Async mirror.** `AsyncLMRouter().complete(req)` with the same
  `Request`; citations and `provider_data` come back identically.
- **Streaming.** Citations stream as `CitationDelta` events on OpenAI
  and Anthropic; Gemini attaches grounding metadata at the end. See
  [05 — Streaming](05-streaming.md).
- **More OpenAI builtins.** `"file_search"` and `"computer_use"`
  (→ `computer_use_preview`) are also mapped; both need `config`
  (vector store ids, display size) per OpenAI's docs.
- **Mixing tool kinds.** `tools=(search, weather)` with a
  `FunctionTool` is legal; the provider executes the search, you
  execute `weather`.
- **Gemini URLs are redirects.** Grounding citations point at
  `vertexaisearch.cloud.google.com/grounding-api-redirect/…`, not the
  source page; the `title` carries the domain. Resolve them only if
  you must — Google rate-limits the redirect endpoint.

## See also

- [06 — Function tools: define & dispatch](06-function-tools.md)
- [05 — Streaming](05-streaming.md)
- [08 — Structured output](08-structured-output.md)
- [Tools from functions](../tools-from-functions.md)
- [Using the router](../using-the-router.md)
