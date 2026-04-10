## Partial Failure

A blocking call is binary. It succeeds — you get an `LMResponse` — or it fails —
you get an exception. There is no third state. The model either finished
generating or it didn't, and you know which.

A stream exists on a gradient between success and failure. The model might
generate 500 words of careful analysis — the user watches them appear, sentence
by sentence, building toward a conclusion — and then the connection drops. Or
the provider hits an internal error. Or the rate limit kicks in mid-response.
The user has seen 500 words. The application has yielded 500 words worth of
events. And now... an exception.

What happened? Did the call succeed? 500 words arrived. They might be useful —
might be the complete first paragraph of a two-paragraph response, or might be a
sentence cut off mid-word. Did the call fail? An exception was raised. The model
didn't finish. The response is incomplete.

The answer is both. The call partially succeeded. And "partial success" is a
state that most APIs aren't designed to express.

### Three Responses to the Impossible

**Discard the partial data.** Raise the exception. The partial events are gone —
the stream is dead, the generator is exhausted, the accumulated text is
unreachable. The user caught the exception but has no access to what was
generated before the failure.

This is the simplest implementation and the most wasteful. Those 500 words cost
tokens. They took time. They might contain the answer, even if the response was
truncated before the model could write "In conclusion..." Discarding them is
clean but expensive, and in an interactive context — where the user *saw* the
words appear on screen — discarding them is disorienting. The text was there.
Now it's gone. The exception erased it.

**Preserve and raise.** Raise the exception, but make the partial data
accessible. This is lm15's approach. The `Stream` class accumulates text,
thinking tokens, and tool calls internally as the stream progresses. When a
failure occurs, the accumulation state survives. The user catches the exception,
then accesses `stream.response` — which materializes an `LMResponse` from
whatever was accumulated before the failure.

```python
try:
    for text in stream.text:
        print(text, end="")
except TransportError:
    # The stream failed, but partial data is available
    resp = stream.response
    print(f"\n[incomplete: {len(resp.text or '')} chars received]")
```

The response is marked as incomplete — `finish_reason` will be `"error"` or
whatever the adapter could determine. The text is there. The usage data might be
partial or missing (the `end` event, which carries usage, didn't arrive). The
developer can decide: is the partial response useful? Can they display it with a
caveat? Should they retry from scratch?

This is more honest than discarding — the data existed, the user might want it,
so the library preserves it. But it introduces an asymmetry: `stream.response`
after a successful stream returns a complete `LMResponse`. After a failed
stream, it returns a partial one. Same property, different guarantees. The
developer must check `finish_reason` or `usage` to know which case they're in.

**Error as event.** Don't raise an exception. Yield the error as a
`StreamChunk(type="error", ...)` in the event stream. The user's iteration loop
encounters it and decides what to do:

```python
for event in stream:
    match event.type:
        case "text":    print(event.text, end="")
        case "error":   print(f"\n⚠️ {event.error}")
        case "finished": print(f"\n📊 {event.response.usage}")
```

This is the most composable design — the error is just another event type,
handled by the same dispatch logic that handles text and tool calls. No
`try/except`, no separate error path.

The cost: every consumer must handle the error event. A consumer that doesn't —
that only has cases for `text` and `finished` — silently ignores the error. The
stream appears to end normally, with no indication that something went wrong.
This is the opposite failure mode from "discard" — instead of losing data, you
lose the error signal. Both are silent. Both are bad.

### The Completeness Problem

Underneath all three approaches is a deeper issue: the consumer may not know the
response is incomplete.

In HTTP, a `Content-Length` header or chunked transfer encoding lets the client
verify that it received the full response body. If the connection drops
mid-transfer, the client knows — the byte count doesn't match, or the final
chunk marker is missing.

Streaming LLM responses have no equivalent integrity check. The response is a
sequence of SSE events. The final event — `type="finished"` — carries the usage
data and finish reason. If the final event doesn't arrive (because the
connection dropped, the provider crashed, or the rate limit interrupted the
stream), the consumer has received an unknown fraction of the response with no
way to determine how much is missing.

A consumer who reads `stream.response` after a connection drop gets a response
that *looks* complete — it has text, it has a message, it has a role. But it
*is* incomplete — the model had more to say. The text might end mid-sentence.
The thinking might be truncated mid-reasoning. A tool call might have partial
arguments (JSON fragments that couldn't be parsed). Nothing in the response
object screams "I'm incomplete" unless you check `finish_reason` — and if the
`finished` event never arrived, `finish_reason` defaults to `"stop"`, which is
the same value as a successful completion.

This is a genuine design flaw in lm15's current implementation — the default
`finish_reason` for a materialized stream that never received an `end` event
should be something other than `"stop"`. A value like `"unknown"` or
`"interrupted"` would signal incompleteness. The current behavior means a
partial stream that lost its final event is indistinguishable from a complete
stream, which is exactly the kind of silent failure that's hardest to debug.

The broader point stands regardless of the specific default: **streaming
introduces an integrity problem that blocking calls don't have.** A blocking
call either returns a complete response or raises an exception — there's no
state where the caller has a response but doesn't know whether it's complete.
Streaming creates that state, and no library handles it perfectly. The best you
can do is preserve the partial data, signal the failure clearly, and default to
values that make incompleteness visible rather than invisible.
