## Five Ways to Consume a Stream

The fundamental question isn't "how do you produce streaming data" — the provider sends SSE, the transport yields bytes, the adapter yields events. That's plumbing, and it works roughly the same way everywhere. The fundamental question is "how does the user receive it?" — and the answer shapes the user's code more deeply than any other API decision.

Five models exist. Each one determines a different control flow in the user's code.

### The Iterator

The user pulls events:

```python
for event in stream:
    if event.type == "text":
        print(event.text, end="")
```

The loop body runs between events. If the body is slow (say, writing to a database), the stream waits — backpressure is automatic. The code reads like any other `for` loop. No concurrency, no registration, no special constructs. The user's existing mental model of "iterate a collection" transfers directly to "iterate a stream."

The cost: the user's thread is blocked. While iterating the stream, you can't serve another HTTP request, update a progress bar, or check whether the user pressed Ctrl+C — unless you involve threading, which the iterator model doesn't require but also doesn't help with. For a CLI tool that does nothing except print the response, this is perfect. For a web server handling concurrent requests, it means one thread per stream.

### The Callback

The library pushes events:

```python
stream.on("text", lambda text: print(text, end=""))
stream.on("tool_call", lambda name, input: print(f"🔧 {name}"))
stream.on("finished", lambda resp: print(f"\n📊 {resp.usage}"))
stream.start()
```

The library controls the loop. The user registers handlers. Events arrive when the library decides to deliver them. This inverts control — the user's code doesn't drive the process; it reacts to it.

Callbacks compose naturally with UI frameworks. React's `useState`, browser event listeners, Node.js event emitters — all callback-based. If your library targets front-end developers or UI-heavy applications, callbacks match the ecosystem's patterns.

The costs are well-documented in the programming literature. Error handling is awkward: if the `text` callback throws, which handler catches it? The `finished` handler? A global error handler? Ordering guarantees are implicit: are callbacks called in the order events arrive? What if two events arrive simultaneously? And callback registration creates a temporal coupling — you must register handlers *before* calling `start()`, and handlers registered after `start()` might miss early events.

### The Async Iterator

The user pulls, but doesn't block:

```python
async for event in stream:
    if event.type == "text":
        print(event.text, end="")
```

Same syntax as the synchronous iterator. Same backpressure semantics. But between events, the async runtime can do other work — serve another request, run another coroutine, check for cancellation. This is the best of iterators (pull-based, readable) and callbacks (non-blocking, concurrent).

The cost is the async tax. `async for` requires an `async` function. An `async` function requires an `async` caller. The asyncness propagates up the call stack until it reaches the event loop at the top. A library that offers only async streaming forces every user to write async code — or to use `asyncio.run()` as a synchronous shim, losing the concurrency benefit.

For libraries with an async runtime already in the dependency tree (`httpx` uses `anyio`, which uses `asyncio` or `trio`), async iterators are natural. For a zero-dependency library that uses `urllib`, there is no async runtime to use.

### The Observable

The library produces an event stream with composition operators:

```python
stream.pipe(
    filter(lambda e: e.type == "text"),
    map(lambda e: e.text),
    buffer_count(10),
).subscribe(lambda batch: print("".join(batch)))
```

Powerful. You can filter, map, merge, debounce, and buffer streams with declarative operators. Complex streaming patterns — "batch text events into groups of 10 and flush every 100ms" — are one-liners.

The cost is conceptual overhead. Reactive programming is a paradigm, not a library feature. The user must understand observables, operators, subscriptions, and disposal. Most developers don't, and the learning curve for printing "hello" one token at a time is absurd. RxPY exists. Nobody uses it for LLM streaming.

### The Channel

The producer writes events to a queue. The consumer reads from the queue. Decoupled by a buffer:

```python
# Producer (library, in background thread or coroutine)
channel.put(StreamEvent(type="text", text="Hello"))

# Consumer (user code)
while event := channel.get():
    print(event.text)
```

Channels decouple production and consumption temporally — the producer can run ahead of the consumer, buffered by the channel. This enables true concurrency without callbacks, and the buffer absorbs timing mismatches.

The cost: channels require concurrency primitives (`queue.Queue`, `asyncio.Queue`), and the buffer semantics are another design decision. Unbounded buffers consume unlimited memory. Bounded buffers create backpressure (the producer blocks when the buffer is full) or drop events (the producer discards when the buffer is full). Either way, the developer must reason about concurrency — which is the complexity that iterators avoid entirely.

### Why lm15 Chose Iterators

Two constraints forced the choice.

**Zero dependencies.** Async iterators require an async runtime. Observables require a reactive library. Channels require threading or async primitives. The synchronous iterator requires nothing — it uses Python's built-in iteration protocol (`__iter__`, `__next__`), which exists in every Python environment from a REPL to a Jupyter notebook to a CGI script.

**Python's synchronous majority.** Most Python code is synchronous. Most Python developers think synchronously. A library that streams only via async iterators excludes them — not technically (they can wrap with `asyncio.run()`), but culturally. The code looks foreign. The mental model doesn't transfer. The synchronous iterator looks like a `for` loop, because it *is* a `for` loop. Nothing new to learn.

The Vercel AI SDK chose differently — async iterators for the server, callbacks for the browser. LangChain offers both callbacks (`CallbackHandler`) and async iterators. OpenAI's SDK uses synchronous iterators for blocking streams and async iterators for async streams. Each choice reflects the ecosystem: JavaScript is async-first, so Vercel uses async. Python is sync-first, so lm15 uses sync.

The choice isn't about streaming. It's about what kind of code your users write — and the best choice is the one that matches what they already write.
