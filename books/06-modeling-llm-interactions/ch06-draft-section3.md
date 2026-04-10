## The Two-Level API

A developer building a chatbot wants text. They want to print each word as it
arrives. They don't care about thinking tokens, tool call fragments, usage data,
or finish reasons — not on every iteration. They care about the words.

A developer building an agent dashboard wants everything. They want to show
thinking in a dimmed panel, tool calls in a sidebar, usage in a footer, and text
in the main area. They need every event type, distinguished, with metadata.

These are the same stream. These are different users. A streaming API that
serves both must offer both levels of access without forcing the simple user
through the complex path or hiding the complex path from the power user.

Most streaming APIs don't do this. OpenAI's SDK requires the user to iterate
chunks and extract `delta.content` manually:

```python
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

The text-only user pays the full price: iterate chunks, access nested fields,
check for None, handle the case where `delta.content` doesn't exist on this
chunk (tool call chunks have `delta.tool_calls` instead). The API offers one
level — the full event level — and leaves the simple case to the user.

Anthropic's SDK is similar. Every consumer handles `event.type`, even if they
only care about text:

```python
for event in stream:
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="")
```

Two type checks, a nested access, and a specific event-type string that the user
must know. To print words.

lm15 offers two interfaces on the same object:

```python
# Simple: text only
for text in stream.text:
    print(text, end="")

# Full: all events
for event in stream:
    match event.type:
        case "text":     print(event.text, end="")
        case "thinking": print(f"💭 {event.text}", end="")
        case "tool_call": print(f"🔧 {event.name}")
        case "finished": print(f"\n📊 {event.response.usage}")
```

`stream.text` is a generator that yields only text strings. It iterates the
underlying event stream internally, filters for text events, and yields the text
content. The user sees a generator of strings. The events, the types, the
metadata — invisible.

The full event interface is one level deeper. Same object, same iteration
protocol. The user who needs events gets events. The user who needs text gets
text. Neither path is a wrapper around the other — they're both native to the
`Stream` class, sharing the same underlying event source.

The design principle: **the common case should be the shortest code path.** Not
the most flexible, not the most powerful — the most common. If 80% of users want
text, the text path should be one line. The 20% who want events pay the
complexity of handling event types. The 80% never encounter that complexity.

This sounds obvious. It's violated by most streaming APIs, because API designers
tend to optimize for completeness ("what if the user needs...?") rather than
frequency ("what does the user usually need?"). Completeness is the right
instinct for a reference API. Frequency is the right instinct for a user-facing
API. The two-level approach serves both: `.text` optimizes for frequency, the
event interface optimizes for completeness.

One subtlety: the two levels are mutually exclusive on a single stream. If you
start iterating via `.text`, you can't switch to the event interface mid-stream
— the text generator has already consumed some events. If you start with the
event interface, `.text` isn't useful — you're already handling events. The user
picks a level at the start and stays there. This is a reasonable constraint
(mixing levels would require buffering and replaying), but it's not documented
as prominently as it should be. A developer who writes `for text in stream.text`
and then tries `for event in stream` gets an already-consumed iterator, not an
error. The failure is silent, which is the worst kind.
