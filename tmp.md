
 Each thread runs the exact same synchronous code that lm15 runs today —
 urllib.request.urlopen(). Each thread independently opens a socket, sends the
 request, and blocks while waiting for the response. The OS kernel handles the
 actual I/O; the thread just sleeps until data arrives.

 Why this works despite the GIL: Python's Global Interpreter Lock prevents two
 threads from running Python code simultaneously. But when a thread calls
 urlopen(), it releases the GIL while waiting for network I/O. So all three
 threads can be waiting on I/O simultaneously — the GIL isn't held by any of
 them. The GIL only matters for CPU-bound work; LLM API calls are 99.9% waiting.

 The mechanics:

 ```python
   # This is literally all that happens:
   import threading

   def call_in_thread():
       urllib.request.urlopen(req)  # GIL released here, thread sleeps

   t1 = threading.Thread(target=call_in_thread)
   t2 = threading.Thread(target=call_in_thread)
   t1.start(); t2.start()  # Both threads waiting on network simultaneously
   t1.join(); t2.join()     # Collect results
 ```

 Cost: Each thread is an OS thread (~8MB stack by default, though Python can use
 less). For 5-20 concurrent LLM calls, this is negligible. For 10,000, you'd run
 out of memory. But you'd also hit API rate limits at 50, so this is never the
 bottleneck.

 ### Approach B: Async (asyncio + aiohttp)

 ```
   Your Python process
   └── Single thread, single event loop
       ├── Coroutine 1: open socket → send HTTP → YIELD → receive
       ├── Coroutine 2: open socket → send HTTP → YIELD → receive
       └── Coroutine 3: open socket → send HTTP → YIELD → receive
 ```

 Instead of multiple threads, there's one thread running an event loop. Each
 "coroutine" (async function) runs until it needs to wait for I/O, then it
 yields control back to the event loop. The event loop checks which sockets have
 data ready and resumes the corresponding coroutine.

 The key difference from threads: No OS threads. No GIL issue (there's only one
 thread). The concurrency is cooperative — each coroutine voluntarily yields
 when it would block. The event loop is a scheduler, like a juggler keeping
 multiple balls in the air by switching between them.

 But here's the catch: urllib.request.urlopen() doesn't know how to yield. It's
 a blocking call — it takes over the thread and doesn't give it back until the
 response arrives. To do true async, you need an HTTP client that's written for
 async I/O — one that uses asyncio's socket layer instead of the standard
 blocking sockets. That's what aiohttp does. And aiohttp is a dependency.

 The alternative — asyncio.to_thread():

 ```python
   async def acall(model, prompt):
       return await asyncio.to_thread(lm15.call, model, prompt)
 ```

 This is not true async. It's Approach A (threads) wearing an async costume.
 Under the hood, asyncio.to_thread() creates an OS thread and runs the blocking
 lm15.call() in it. The only benefit is that your code can use await syntax,
 which is convenient if you're already in an async codebase (FastAPI, etc.).

 ### Approach C: True async with stdlib sockets

 You could use asyncio.open_connection() (stdlib, no deps) to open raw TCP
 sockets and write the HTTP protocol by hand:

 ```python
   reader, writer = await asyncio.open_connection('api.openai.com', 443, ssl=True)
   writer.write(b'POST /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\n...')
   response = await reader.read()
 ```

 This is truly async, zero deps — but you're reimplementing HTTP. Chunked
 transfer encoding, TLS negotiation, keep-alive, redirect following, SSE parsing
 over async streams... it's hundreds of lines of careful protocol code. That's
 what aiohttp is. You'd be writing your own aiohttp.

 Summary Table

 ┌──────────────┬──────────────────┬────────────────┬────────────────────┬──────────────┐
 │              │ Threads (A)      │ True Async (B) │ asyncio.to_thread  │ Raw
 sockets  │ │              │                  │                │ (B') │ (C) │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ How          │ OS manages       │ Event loop     │ Thread per call,   │ Event
 loop + │ │ concurrency  │ multiple         │ switches       │ wrapped in async
 │ hand-written │ │ works        │ threads, each    │ between        │ syntax │
 HTTP         │ │              │ blocking         │ coroutines at  │ │ │ │
 │ independently    │ yield points   │ │              │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ Needs new    │ No               │ Yes (aiohttp)  │ No                 │ No │
 │ dependency   │                  │                │                    │ │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ Changes to   │ None — wraps     │ New transport  │ ~3 lines           │ ~500
 lines   │ │ lm15 code    │ existing sync    │ layer, new     │ │ of HTTP      │
 │              │ calls            │ adapter        │ │ client       │ │ │
 │ methods, new   │ │              │ │              │ │ API surface    │ │
 │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ Import cost  │ 0ms              │ ~200ms         │ 0ms                │ 0ms │
 │              │                  │ (aiohttp)      │                    │ │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ Max          │ ~50-100 (OS      │ ~10,000+ (no   │ ~50-100 (still     │
 ~10,000+     │ │ practical    │ thread limit)    │ thread         │ threads) │
 │ │ concurrency  │                  │ overhead)      │ │              │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ Actual LLM   │ ~5-50 (rate      │ ~5-50 (rate    │ ~5-50 (rate        │ ~5-50
 (rate  │ │ concurrency  │ limits)          │ limits)        │ limits) │ limits)
 │ │ limit        │                  │                │ │              │
 ├──────────────┼──────────────────┼────────────────┼────────────────────┼──────────────┤
 │ User code    │ None             │ Must use       │ Must use           │ Must
 use     │ │ changes      │                  │ async/await    │ async/await │
 async/await  │ │              │                  │ everywhere     │ │ │
 └──────────────┴──────────────────┴────────────────┴────────────────────┴──────────────┘

 The Punchline

 The rate limit is the real ceiling. OpenAI gives you ~500 RPM on tier 1, ~5000
 RPM on tier 4. Anthropic gives ~1000 RPM. These are requests per minute. Even
 if you could have 10,000 concurrent connections (true async), the provider
 would reject most of them.

 For the realistic use case — 3-20 concurrent LLM calls for fan-out, evaluation,
 or parallel agents — threads and true async perform identically. The time is
 dominated by the model thinking (1-30 seconds), not by connection overhead
 (1-5ms). The difference between threads and async is microseconds of scheduling
 overhead vs. seconds of model inference time.

 True async (Approach B) wins when you have thousands of concurrent lightweight
 I/O operations — like a web server handling 10,000 simultaneous HTTP requests.
 For LLM calls, where each request takes seconds and you're rate-limited to
 dozens, threads are the right tool. They require zero dependencies, zero API
 changes, and the performance difference is unmeasurable.