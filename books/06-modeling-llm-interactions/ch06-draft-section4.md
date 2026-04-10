## The Dual Nature Problem

After the stream ends, you often need the complete response. Token counts for billing. The finish reason for control flow. The full text for logging or for passing to the next pipeline stage. But the stream is consumed — the events are gone. You iterated them, processed them, and they don't exist anymore. You can't rewind a generator.

lm15 solves this by making the `Stream` object serve double duty. During iteration, it's an event source — a lazy, forward-only generator of `StreamChunk` objects. After iteration, it's a response container — `stream.response` returns a materialized `LMResponse`, assembled from all the events that were streamed.

```python
stream = lm15.stream("gpt-4.1-mini", "Explain DNS.", env=".env")

# Phase 1: event source
for text in stream.text:
    print(text, end="")

# Phase 2: response container
resp = stream.response
print(f"\nTokens: {resp.usage.total_tokens}")
```

The object changed character between line 4 and line 7. During the loop, it was a generator. After the loop, it's a data container. Same variable, different behavior. This is convenient — two lines, no extra objects — and surprising, because objects usually don't change what they are after you use them.

The surprise has a sharp edge. If you access `.response` *before* consuming the stream, the stream silently consumes itself:

```python
stream = lm15.stream("gpt-4.1-mini", "Explain DNS.", env=".env")
resp = stream.response  # consumes the entire stream to build the response
# The stream is now exhausted — iterating it yields nothing
```

The user got the response but lost the streaming. No error, no warning. The property looked like a simple accessor; it was actually a consuming operation. This is a foot-gun — subtle, documented, but easy to hit.

The dual-nature pattern appears in other domains. Python's `itertools.tee` creates two iterators from one, but buffers everything — defeating the memory benefit of streaming. Rust's `Peekable` iterator adds look-ahead to an iterator, but it's still forward-only. Java's `Stream` API is explicitly single-use — iterating twice throws `IllegalStateException`. Each language has encountered the same tension: streams are ephemeral by nature, but users want persistent access to the data that flowed through them.

Three alternative designs avoid the dual nature:

**Separate objects.** The library returns a stream *and* a response future:

```python
stream, response_future = model.stream("Explain DNS.")
for text in stream.text:
    print(text, end="")
resp = response_future.result()  # blocks until stream ends
```

Clean separation. The stream is an event source and nothing else. The response comes from a different object. The cost: two variables instead of one, and the user must understand futures (or promises, or callbacks — whatever the mechanism for "the response will be available later").

**Explicit materialization.**

```python
for text in stream.text:
    print(text, end="")
resp = stream.collect()  # explicitly build the response
```

The user triggers materialization explicitly. No surprise, no silent consumption. The cost: the user must remember to call `.collect()`. If they don't, the response data is unavailable — the stream is consumed and the accumulation state is internal. And calling `.collect()` mid-stream has ambiguous semantics: does it consume the rest of the stream? Return what's accumulated so far? Block until the stream ends?

**Accumulator callback.**

```python
response_builder = ResponseBuilder()
for event in stream:
    response_builder.add(event)
    if event.type == "text":
        print(event.text, end="")
resp = response_builder.build()
```

Maximum control. The user decides exactly what to accumulate and when. The cost: maximum boilerplate. Every consumer must create a builder, feed it events, and call build. The common case — "I just want to print text and then check the token count" — requires five lines of ceremony.

lm15 chose the dual nature because it minimizes the code for the common case: iterate, then access `.response`. Two lines. No extra imports, no builders, no futures. The surprise (silent consumption when accessing `.response` early) is the tax on that convenience. The library judged — correctly, I think — that the common path matters more than the edge case, and that documenting the edge case is cheaper than adding ceremony to the common path.

But it's worth stating clearly: the dual nature is a compromise, not a solution. It papers over a real tension — streams are ephemeral, responses are persistent, and one object can't naturally be both. Every design that tries to combine them introduces a surprise somewhere. lm15's surprise (silent consumption) is smaller than the alternatives' costs (extra objects, extra methods, extra boilerplate), but it's still a surprise. A perfect design would make it impossible to access `.response` before consuming the stream. Python doesn't offer the type-system machinery to enforce this, so it's handled with documentation and programmer discipline — the weakest enforcement mechanism available.
