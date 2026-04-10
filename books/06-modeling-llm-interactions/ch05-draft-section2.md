## What Translation Destroys

Chapter 4 asked where normalization distorts — where the same parameter means
different things on different providers. This section asks a different question:
where does the act of translation itself lose information? Not because the
concepts differ, but because the *encoding* differs in ways that destroy
structure, ordering, or metadata that existed in the provider's response.

**Ordering within tool calls.** When a model generates multiple tool calls in
one response, the calls arrive in a specific order. The model decided to search
for topic A before searching for topic B — perhaps because A is the primary
question and B is supplementary. OpenAI preserves this order in its `tool_calls`
array. Anthropic preserves it in the content blocks' sequence. lm15's `Part`
objects in `message.parts` preserve it too. So far, no loss.

But consider what happens during *streaming*. The model generates tool call A
and tool call B as interleaved token sequences — fragments of A's arguments,
then fragments of B's, then more of A's. The adapter must accumulate these
fragments into complete tool calls, tracking which fragment belongs to which
call via `part_index`. The order in which fragments arrive doesn't match the
order of the completed calls. The adapter must reassemble, and the reassembly
discards the interleaving pattern — which contained information about how the
model's attention moved between the two calls. This is a small loss, and no
application uses the interleaving order. But it's a loss, and it illustrates the
principle: every translation that restructures data destroys some property of
the original.

**The tool result round-trip.** This is where translation is hardest and loss is
most consequential. Tool results — the data sent back to the model after
executing a tool — expose encoding differences that aren't visible until
something breaks.

OpenAI expects tool results as a message with role `"tool"`, a `tool_call_id`,
and `content` as a string. Anthropic expects a `tool_result` content block
inside a `user`-role message, with `content` as an array of content blocks.
Gemini expects a `functionResponse` part with the result as a JSON dict.

The adapter translates `Part(type="tool_result", id="call_1",
content=(Part.text_part("22°C"),))` into each format. The translation works for
text results. But what happens when the tool result contains an image — a
screenshot the tool captured, a chart it generated?

On Anthropic, the image becomes an image content block inside the tool result's
content array. The model sees the image. On OpenAI, the tool result's content is
a string. The adapter must either drop the image (the model never sees it),
base64-encode it into the string (the model sees a wall of characters), or
refuse to translate (the developer gets an error). Each choice loses something.
On Gemini, the result is a JSON dict, and an image doesn't fit in a dict without
inventing an encoding.

This isn't a normalization problem — it's a translation problem. The universal
type (`Part.tool_result` with a content tuple) can represent "a tool result
containing text and an image." The Anthropic wire format can express it. The
OpenAI and Gemini wire formats can't. The adapter is the entity that must decide
what to do when the universal representation is more expressive than the wire
format — and "more expressive" means information will be lost.

**Provider-specific response metadata.** Each provider returns metadata that the
others don't. Anthropic returns `stop_sequence` — the specific stop string that
triggered the finish. OpenAI returns `system_fingerprint` — a hash identifying
the backend configuration. Gemini returns `safety_ratings` — content moderation
scores.

lm15's `LMResponse` doesn't have fields for any of these. They're available in
`resp.provider` (the escape hatch), but they're not surfaced as typed fields. A
developer who switches from Anthropic to OpenAI and relied on
`resp.provider["stop_sequence"]` gets a `KeyError`. The adapter didn't translate
the metadata — it carried it in an untyped dict, and the developer's code
assumed the dict's shape.

The pattern across all three examples: **translation loss occurs at capability
boundaries — where one format can express something that another can't.** The
universal type's job is to be the union of what all providers can express. The
adapter's job is to translate into what each specific provider can express. When
the universal type exceeds the wire format's expressiveness, the adapter must
drop, transform, or refuse. Each choice is a judgment call, and the judgment is
often made once, buried in 20 lines of adapter code, and never revisited.

## Streaming: The Temporal Translation

Blocking translation is a spatial problem — transform one structure into
another. Streaming translation is a temporal problem — transform a sequence of
fragments into a sequence of events, in real time, with incomplete information
at every step.

The adapter opens an SSE connection to the provider. Bytes arrive as lines. The
SSE parser groups lines into events. Each event contains a JSON fragment — a
delta with partial text, a piece of a tool call's arguments, a usage report, or
an end signal. The adapter must interpret each fragment, decide whether it
contains enough information to yield a `StreamEvent`, and maintain state across
fragments for things that arrive in pieces.

Text is easy. A text delta — `{"type": "text", "text": "Hello"}` — is
self-contained. The adapter yields `StreamEvent(type="delta",
delta=PartDelta(type="text", text="Hello"))` immediately. No accumulation
needed.

Tool calls are hard. A tool call arrives as a sequence of fragments:

```
{"type": "tool_call", "id": "call_1", "name": "search"}
{"type": "tool_call", "input": "{\"quer"}
{"type": "tool_call", "input": "y\": \"tcp\"}"}
```

The name and ID arrive first. The arguments arrive as JSON string fragments that
must be concatenated before they're parseable. The adapter can't yield the tool
call after the first fragment (no arguments yet) and can't wait for all
fragments (that would defeat streaming). lm15's adapters yield each fragment as
a `StreamEvent`, and the `Stream` class downstream handles accumulation. But the
adapter still must track `part_index` — which fragments belong to which tool
call when multiple calls are in flight — and must decide when a tool call's name
is "known enough" to report.

The yield-timing judgment is the hardest decision in streaming adapter design.
Yield too early and the consumer gets incomplete data. Yield too late and the
stream loses its real-time character. There's no universally correct answer
because the right timing depends on what the consumer does with the events — a
CLI that prints tool call names wants early yields; a tool executor that needs
complete arguments wants late yields. lm15 chose early (yield every fragment,
let the Stream class accumulate), which pushes complexity downstream but
preserves liveness.

Provider differences compound the problem. OpenAI signals stream end with `data:
[DONE]` — a sentinel in the data field. Anthropic signals it with `event:
message_stop` — a named event type. Gemini closes the connection. Each adapter
must detect its provider's end signal and yield a final `StreamEvent(type="end",
usage=..., finish_reason=...)`. If the adapter misses the signal — or if the
connection drops before the signal arrives — the stream hangs or the consumer
gets no usage data.

Error handling is worst. A provider error mid-stream — rate limit hit, server
crash, malformed JSON — can arrive as a well-formed error event, a malformed SSE
line, or a sudden connection close. The adapter must handle all three,
distinguishing "the provider told me about an error" (yield an error event) from
"the connection broke" (raise `TransportError`). The distinction matters to the
consumer: an error event carries diagnostic information; a transport error
carries nothing.

The streaming adapter is the most complex piece of code in lm15 — more complex
than the message representation, more complex than the conversation model, more
complex than the tool execution loop. It's where the translation problem, the
normalization problem, and the temporal problem converge. And it's where the
most bugs live, because each provider's streaming format has undocumented quirks
that only surface under load, with long responses, or during error conditions.
