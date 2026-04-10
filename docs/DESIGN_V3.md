# lm15 v3 Design — Unified Result API

## Overview

This document specifies the v3 API redesign of lm15. The core insight: there are three orthogonal interaction axes, and all combinations must compose cleanly. The design unifies `call()` and `stream()` into a single entry point returning a `Result` type, always backed by a provider stream under the hood.

**Status:** Pre-release. Nothing is shipped. Everything can change.

---

## Three Orthogonal Axes

```
         Delivery          State           Concurrency
        ┌─────────┐      ┌──────────┐      ┌───────┐
        │ blocking │      │ stateless│      │ sync  │
        │ streaming│      │ stateful │      │ async │
        └─────────┘      └──────────┘      └───────┘
```

- **Delivery** (blocking vs streaming): How the user consumes the response. NOT how it's requested — always a stream under the hood.
- **State** (stateless vs stateful): Who owns conversation history. Stateless = user manages messages. Stateful = `Model` manages history.
- **Concurrency** (sync vs async): Whether the call is synchronous or asynchronous.

These produce 2 × 2 × 2 = 8 logical combinations, but only **4 entry points** because delivery is a consumption choice, not a function choice:

| State | Concurrency | Entry point | Returns |
|-------|-------------|-------------|---------|
| stateless | sync | `lm15.call()` | `Result` |
| stateful | sync | `agent.call()` | `Result` |
| stateless | async | `lm15.acall()` | `AsyncResult` |
| stateful | async | `agent.acall()` | `AsyncResult` |

All capabilities (tools, reasoning, prefill, output modality, prompt caching, built-in tools) are accepted by all 4 entry points. Delivery mode is determined by how you consume the `Result`.

---

## Always-Stream Architecture

### Principle

Every completion request to a provider uses the streaming endpoint (`stream=True`). There is no non-streaming code path for completions. "Blocking" consumption means the `Result` object internally drains the stream and caches the materialized response before returning a property value.

### Why

1. **One code path.** Adapters only implement `build_request()` and `parse_stream_event()`. No `parse_response()`. Cuts adapter surface in half (~90 lines of duplicated logic removed).
2. **Honest unification.** `Result` is always backed by a real stream. "Blocking is consumed streaming" is the implementation, not a metaphor.
3. **Better timeouts.** Streaming requests get first token in 1-2s. Read timeouts (no data for N seconds) replace response timeouts (whole response must arrive in N seconds).
4. **Free cancellation.** Users can stop consuming mid-response.
5. **Partial failure data.** On connection drop after 90% of tokens, partial data is accessible.

### What stays non-streaming

These are not completions — they have no streaming equivalent:

- `embeddings()` — returns vectors, no token stream
- `file_upload()` — returns a file ID
- `batch_submit()` — returns a batch ID
- `image_generate()` — returns image data
- `audio_generate()` — returns audio data

### Non-streaming providers

Providers that have no SSE/streaming endpoint can still be added to lm15. The adapter protocol requires `stream() -> Iterator[StreamEvent]`, but nothing requires that iterator to be backed by real SSE. A non-streaming adapter makes a normal blocking HTTP request internally and yields synthetic events:

```python
class MyBlockingAdapter(BaseProviderAdapter):
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        # 1. Normal blocking HTTP call
        req = self.build_request(request)
        resp = self.transport.request(req)  # blocks, returns full response
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())

        # 2. Parse the complete response
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = Usage(...)

        # 3. Yield synthetic stream events
        yield StreamEvent(type="start", id=data.get("id"), model=request.model)
        yield StreamEvent(type="delta", part_index=0,
                          delta=PartDelta(type="text", text=text))
        yield StreamEvent(type="end", finish_reason="stop", usage=usage)
```

From `Result`'s perspective, this is indistinguishable from a real SSE stream — it just happens to yield all the text in one chunk instead of token-by-token. The only user-visible difference is latency profile: real streaming gets first tokens in ~1-2s, synthesized streaming blocks for the full response then delivers everything at once. The API contract is identical.

This means the always-stream architecture does not exclude any provider. A provider with only a blocking REST API is a provider whose adapter synthesizes a stream. The `EndpointSupport.stream` field indicates whether the provider has native SSE (which affects timeout strategy and partial failure behavior), not whether the adapter can satisfy the `stream()` protocol — every adapter can.

### Impact on non-streaming users

Users who don't care about streaming see no difference. They access `.text`, it blocks, returns a string:

```python
r = lm15.call("gpt-4.1-mini", "Hello.")
print(r.text)  # blocks, returns str — same experience as old lm15.call()
```

Performance is identical or better (fewer timeout issues). Error types are the same (`AuthError`, `RateLimitError`, etc.).

---

## The `Result` Type

`Result` is the single return type for all sync completion calls. It is lazy — backed by a provider stream that is consumed on demand.

### Consumption modes

```python
r = lm15.call("gpt-4.1-mini", "Write a haiku.")

# --- Blocking (access properties — stream auto-consumed) ---
r.text              # str | None — full text, blocks until complete
r.thinking          # str | None — reasoning text
r.tool_calls        # list[Part] — tool call parts
r.image             # Part | None — first image part
r.images            # list[Part] — all image parts
r.audio             # Part | None — first audio part
r.citations         # list[Part] — all citation parts
r.usage             # Usage — token counts
r.finish_reason     # str — "stop", "tool_call", "length", etc.
r.model             # str — model that responded
r.json              # Any — parsed JSON from text (raises ValueError if invalid)
r.image_bytes       # bytes — decoded image bytes
r.audio_bytes       # bytes — decoded audio bytes
r.response          # LMResponse — the full response object

# --- Streaming text (iterate — the 80% streaming case) ---
for text in r:
    print(text, end="")

# --- Streaming events (full detail) ---
for event in r.events():
    match event.type:
        case "text":        print(event.text, end="")
        case "thinking":    print(f"💭 {event.text}", end="")
        case "tool_call":   print(f"🔧 {event.name}({event.input})")
        case "tool_result": print(f"📎 {event.text}")
        case "finished":    print(f"📊 {event.response.usage}")
```

### Lifecycle

1. On creation, `Result` holds a reference to the provider stream iterator. Nothing has been consumed yet.
2. Iterating (`for text in r` or `for event in r.events()`) pulls events lazily from the stream.
3. Accessing any blocking property (`.text`, `.usage`, etc.) drains the entire stream, caches the materialized `LMResponse`, and returns the value.
4. After the stream is consumed (by either method), all properties return cached values. Iteration raises `StopIteration`.
5. Properties accessed after iteration return accumulated values (e.g., `.text` returns the full text even after streaming).

### Class definition

```python
class Result:
    """Lazy stream-backed response."""

    def __init__(
        self,
        *,
        events: Iterator[StreamEvent],
        request: LMRequest,
        on_finished: Callable[[LMRequest, LMResponse], None] | None = None,
        callable_registry: dict[str, Callable] | None = None,
        on_tool_call: Callable | None = None,
        max_tool_rounds: int = 8,
    ) -> None: ...

    # --- Streaming ---

    def __iter__(self) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        for chunk in self._chunks():
            if chunk.type == "text" and chunk.text is not None:
                yield chunk.text

    def events(self) -> Iterator[StreamChunk]:
        """Yield all events (text, thinking, tool_call, tool_result, finished)."""
        yield from self._chunks()

    # --- Blocking properties ---

    @property
    def text(self) -> str | None: ...

    @property
    def thinking(self) -> str | None: ...

    @property
    def tool_calls(self) -> list[Part]: ...

    @property
    def image(self) -> Part | None: ...

    @property
    def images(self) -> list[Part]: ...

    @property
    def audio(self) -> Part | None: ...

    @property
    def citations(self) -> list[Part]: ...

    @property
    def usage(self) -> Usage: ...

    @property
    def finish_reason(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def json(self) -> Any: ...

    @property
    def image_bytes(self) -> bytes: ...

    @property
    def audio_bytes(self) -> bytes: ...

    @property
    def response(self) -> LMResponse:
        """Full LMResponse. Blocks until stream is consumed."""
        ...

    # --- Internal ---

    def _consume(self) -> LMResponse:
        """Drain the stream, materialize and cache the response."""
        ...

    def _chunks(self) -> Iterator[StreamChunk]:
        """Internal generator that handles tool auto-execution loops."""
        ...
```

### StreamChunk (unchanged from current)

```python
@dataclass(slots=True, frozen=True)
class StreamChunk:
    type: str  # "text" | "thinking" | "tool_call" | "tool_result" | "image" | "audio" | "finished"
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    image: Part | None = None
    audio: Part | None = None
    response: LMResponse | None = None  # only on "finished"
```

---

## Tool Execution

### Tool types

Tools are provided in the `tools=` parameter. Four forms:

| Form | Schema source | Auto-execute | Example |
|------|--------------|-------------|---------|
| Bare callable | Inferred from type hints + docstring | Yes | `tools=[get_weather]` |
| `Tool(fn=callable)` | Explicit `parameters` dict + callable | Yes | `Tool(name="x", parameters={...}, fn=my_fn)` |
| `Tool(...)` (no fn) | Explicit `parameters` dict | No (manual) | `Tool(name="x", parameters={...})` |
| `Tool.from_fn(callable)` | Inferred from callable | No (manual) | `Tool.from_fn(write_file)` |
| `"string"` | Built-in (provider server-side) | Provider handles | `"web_search"` |

### The `on_tool_call` hook

`on_tool_call` is the single interception point for all tool execution. It is called before each tool call is executed, for every function tool (not built-in tools — those execute server-side).

```python
def on_tool_call(call: ToolCallInfo) -> str | None:
    """Called before each tool execution.

    Args:
        call: Object with .name, .input, .id attributes.

    Returns:
        str — Use this string as the tool result. Skip auto-execute.
        None — Proceed with auto-execute (if callable is registered).
               If no callable is registered, the tool call is unanswered
               and the loop stops (finish_reason remains "tool_call").
    """
```

#### ToolCallInfo

```python
@dataclass(slots=True, frozen=True)
class ToolCallInfo:
    id: str
    name: str
    input: dict[str, Any]
```

#### Usage patterns

```python
# 1. Pure auto-execute (default — no on_tool_call)
r = lm15.call("gpt-4.1-mini", "Weather?", tools=[get_weather])

# 2. Logging / visibility
def log(call):
    print(f"🔧 {call.name}({call.input})")
    return None  # proceed with auto-execute
r = lm15.call("gpt-4.1-mini", "Weather?", tools=[get_weather], on_tool_call=log)

# 3. Approval gate
def approve(call):
    if call.name in ("write_file", "run_command"):
        print(f"⚠️ {call.name} wants: {call.input}")
        if input("Approve? [y/n] ") != "y":
            return "User denied this action."
    return None  # auto-execute
r = lm15.call("gpt-4.1-mini", "Edit code", tools=[read_file, write_file], on_tool_call=approve)

# 4. Full override (user provides all results)
def manual(call):
    return my_custom_executor(call.name, call.input)
r = lm15.call("gpt-4.1-mini", "Do work", tools=[read_file, write_file], on_tool_call=manual)
```

### Tool loop mechanics

When the model responds with `finish_reason="tool_call"` and there are auto-executable tools (callables in registry, or `on_tool_call` returning results):

1. For each tool call in the response:
   a. If `on_tool_call` is set, call it with `ToolCallInfo`.
   b. If `on_tool_call` returns a string, use that as the result.
   c. If `on_tool_call` returns `None`, look up the callable in the registry and execute it.
   d. If no callable exists and `on_tool_call` returned `None`, the tool call is unanswered — stop the loop.
2. Construct a follow-up request with the tool results appended.
3. Start a new provider stream for the follow-up.
4. Repeat up to `max_tool_rounds` (default 8).

During streaming, tool execution happens between streaming rounds. The user sees:
```
text chunks → tool_call event → tool_result event → text chunks (from next round) → ... → finished
```

### Tool loop in streaming

The `Result._chunks()` generator handles this internally:

```python
def _chunks(self) -> Iterator[StreamChunk]:
    """Yield chunks, auto-executing tools between rounds."""
    round = 0
    current_events = self._events
    while round < self._max_tool_rounds:
        # Yield all chunks from current stream
        for chunk in self._process_stream(current_events):
            yield chunk

        # Check if we need another round
        if not self._pending_tool_calls or not self._has_executable_tools():
            break

        # Execute tools
        for tc in self._pending_tool_calls:
            result = self._execute_tool(tc)
            if result is not None:
                yield StreamChunk(type="tool_result", text=str(result), name=tc.name)

        # Build follow-up request and start new stream
        request = self._build_followup_request()
        current_events = self._start_new_stream(request)
        round += 1

    yield StreamChunk(type="finished", response=self._materialize_response())
```

### Built-in tools

Built-in tools (`"web_search"`, etc.) execute server-side. They:
- Skip `on_tool_call` entirely
- Don't participate in the client-side tool loop
- Results (text, citations) arrive in the normal response stream
- Coexist with user tools in the same `tools=` list

```python
r = lm15.call("gpt-4.1-mini", "Latest AI news and summarize",
         tools=["web_search", summarize_page])
# web_search runs server-side, summarize_page runs client-side
```

---

## Stateless Mode

### Basic usage

```python
import lm15

# Blocking
print(lm15.call("gpt-4.1-mini", "Hello.").text)

# Streaming
for text in lm15.call("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")

# Events
for e in lm15.call("gpt-4.1-mini", "Explain TCP.", reasoning=True).events():
    match e.type:
        case "thinking": print(f"💭 {e.text}", end="")
        case "text":     print(e.text, end="")
```

### Manual tool loop (user manages messages)

```python
from lm15 import Conversation, Tool

weather = Tool(name="get_weather", description="Get weather by city",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})

conv = Conversation(system="You are helpful.")
conv.user("Weather in Montreal?")

r = lm15.call("gpt-4.1-mini", messages=conv.messages, tools=[weather])
conv.assistant(r.response)

if r.finish_reason == "tool_call":
    # User executes tools
    conv.tool_results({tc.id: fetch_weather(tc.input) for tc in r.tool_calls})
    r2 = lm15.call("gpt-4.1-mini", messages=conv.messages, tools=[weather])
    conv.assistant(r2.response)
    print(r2.text)
```

### Prefill

```python
r = lm15.call("claude-sonnet-4-5", "Output JSON for a person.", prefill="{")
print(r.text)  # starts with "{"
```

Prefill is a message concern — it appends `Message.assistant("{")` to the message list before the API call. It applies to the first turn only. If the model calls tools instead of generating text, the prefill is in history but doesn't affect the tool call.

---

## Stateful Mode

### Basic usage

```python
import lm15

agent = lm15.model("claude-sonnet-4-5",
    system="You are a coding assistant.",
    tools=[read_file, write_file],
    on_tool_call=approve,
    prompt_caching=True,
    retries=2,
)

# Blocking
print(agent.call("Add validation to models.py").text)

# Streaming
for text in agent.call("Now add tests."):
    print(text, end="")

# Events
for e in agent.call("Refactor auth.", reasoning=True).events():
    match e.type:
        case "thinking": print(f"💭 {e.text}", end="")
        case "text":     print(e.text, end="")

# History
print(f"Turns: {len(agent.history)}")
agent.history.clear()
```

`agent.call()` and `lm15.call()` return the same `Result` type. The only difference: `agent.call()` records to history when the stream is consumed.

### History recording

History is appended when the `Result` is consumed (stream drained). This happens:
- Immediately when a blocking property is accessed (`.text`, `.response`, etc.)
- At the end of iteration (`for text in r` or `for e in r.events()`)
- When the `finished` chunk is yielded

Each history entry records the final `LMRequest` (after tool loop expansion) and the `LMResponse`.

### `submit_tools()` for manual tools on model

```python
agent = lm15.model("gpt-4.1-mini")
r = agent.call("Weather?", tools=[weather_tool])

if r.finish_reason == "tool_call":
    results = {tc.id: "22°C, sunny" for tc in r.tool_calls}
    r2 = agent.submit_tools(results)
    print(r2.text)  # r2 is also a Result
```

`submit_tools()` returns a `Result` — same type as `agent.call()`. It can be streamed or consumed blocking.

### Forking models with `.copy()`

`.copy()` is the single method for creating derived models. It replaces `with_model()`, `with_system()`, `with_tools()`, and `with_provider()`. Every kwarg you can pass to `lm15.model()`, you can pass to `.copy()`. Omitted kwargs keep the current value. Conversation history is preserved by default.

```python
# Exact fork (same config, same history)
forked = agent.copy()

# Fork + change one thing
claude = agent.copy(model="claude-sonnet-4-5")
terse = agent.copy(system="You are terse.")
with_tools = agent.copy(tools=[get_weather])
local = agent.copy(provider="openai")

# Fork + change multiple things
fast = agent.copy(model="gpt-4.1-mini", temperature=0, max_tokens=100)

# Fork without history (clean slate, same config)
fresh = agent.copy(history=False)
```

The original is always unchanged. `.copy()` returns a new `Model` with independent mutable state.

#### Signature

```python
def copy(
    self,
    *,
    model: str | _Unset = UNSET,
    system: str | None | _Unset = UNSET,
    tools: list | _Unset = UNSET,
    on_tool_call: Callable | None | _Unset = UNSET,
    provider: str | None | _Unset = UNSET,
    retries: int | _Unset = UNSET,
    cache: bool | dict | _Unset = UNSET,
    prompt_caching: bool | _Unset = UNSET,
    temperature: float | None | _Unset = UNSET,
    max_tokens: int | None | _Unset = UNSET,
    max_tool_rounds: int | _Unset = UNSET,
    history: bool = True,
) -> Model:
```

The `_Unset` sentinel distinguishes "not passed" from `None` (since `system=None` is a valid override meaning "remove the system prompt").

- `history=True` (default): copies `_conversation`, `history`, and `_pending_tool_calls` from the original.
- `history=False`: starts with empty conversation state. Same as creating a new `lm15.model()` with the same config.

#### Use cases

```python
# Swap model, keep conversation
claude = gpt.copy(model="claude-sonnet-4-5")

# Parallel exploration from shared context
agent.call("Here's the dataset schema: ...")
by_region = agent.copy()
by_product = agent.copy()
r1, r2 = await asyncio.gather(
    by_region.acall("Analyze by region."),
    by_product.acall("Analyze by product."),
)

# A/B testing with different configs
a = agent.copy(temperature=0)
b = agent.copy(temperature=1.5)

# Reset conversation, keep config
agent_fresh = agent.copy(history=False)
# Equivalent to: agent.history.clear() but non-destructive
```

### Upload

```python
doc = agent.upload("contract.pdf")
r = agent(["Summarize.", doc])
```

Also available stateless:
```python
doc = lm15.upload("claude-sonnet-4-5", "contract.pdf")
r = lm15.call("claude-sonnet-4-5", ["Summarize.", doc])
```

---

## The `Conversation` Helper

`Conversation` is a dumb message list builder for stateless mode. It has no LLM client, no provider, no state machine. It just constructs message tuples in the right format.

```python
class Conversation:
    def __init__(self, *, system: str | None = None) -> None:
        self.system = system
        self._messages: list[Message] = []

    def user(self, content: str | list[str | Part]) -> None:
        """Append a user message."""
        ...

    def assistant(self, response: LMResponse) -> None:
        """Append the assistant message from a response (includes tool calls if any)."""
        self._messages.append(response.message)

    def tool_results(self, results: dict[str, str | Part | list[Part]]) -> None:
        """Append a tool result message from a {call_id: result} dict."""
        ...

    def prefill(self, text: str) -> None:
        """Append an assistant prefill message."""
        self._messages.append(Message.assistant(text))

    @property
    def messages(self) -> tuple[Message, ...]:
        """The message tuple, ready to pass to lm15.call() or lm15.acall()."""
        return tuple(self._messages)

    def clear(self) -> None:
        """Reset to empty."""
        self._messages.clear()
```

Usage:
```python
conv = Conversation(system="You are helpful.")
conv.user("My name is Max.")

r = lm15.call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
conv.assistant(r.response)

conv.user("What's my name?")
r2 = lm15.call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
print(r2.text)  # "Max"
```

---

## Concurrency

### The protocol reality

All three provider APIs (OpenAI, Anthropic, Gemini) are plain HTTP REST endpoints. There is no multiplexed or async protocol. Each request is one HTTP connection, one response. "Concurrency" means having multiple HTTP connections open simultaneously, each independently waiting for its response.

The provider rate limits are the real ceiling — typically 500–5000 RPM depending on tier. Even with unlimited client concurrency, the provider caps you at dozens of simultaneous requests. This means the difference between OS threads and true async I/O is unmeasurable for LLM workloads: both spend 99.9% of time waiting on network I/O (which releases the GIL), and both are capped at the same rate limit.

### Design decision: threads, not async I/O

lm15 uses `concurrent.futures.ThreadPoolExecutor` for all concurrency. No async runtime, no new dependencies, no import cost.

**Why threads work perfectly here:**
- LLM calls are I/O-bound. `urllib.request.urlopen()` releases the GIL while waiting for the network. Multiple threads wait on I/O simultaneously without contention.
- Practical concurrency is 3–50 calls (rate limits). OS threads handle this trivially. The thread overhead (~8MB stack each) is negligible at this scale.
- The entire existing sync stack (transport → adapter → client → model) works unchanged inside threads.
- `concurrent.futures` is stdlib (Python 3.2+). Zero import cost — already loaded by the runtime.

**Why not true async I/O:**
- Requires an async HTTP client dependency (`aiohttp` = 5 transitive deps, +200ms import time, breaks zero-dep identity).
- Or requires reimplementing HTTP over raw `asyncio.open_connection()` (~500 lines of protocol code — TLS, chunked encoding, keep-alive, redirects).
- True async wins at 10,000+ concurrent lightweight connections (web servers). LLM APIs cap at ~50 concurrent requests. At that scale, threads and async perform identically.
- `async/await` infects the entire call stack. Every function between the user and the network must be async. This conflicts with lm15's synchronous, iterator-based mental model.

---

## Async Mode

`lm15.acall()` and `model.acall()` provide `async/await` syntax for users in async codebases (FastAPI, Starlette, Jupyter notebooks). Under the hood, they run the sync code in a thread via `asyncio.to_thread()`.

This is explicitly **syntactic sugar over threading**, not true async I/O. The documentation must be honest about this. The benefit is interoperability with async frameworks, not performance.

### `AsyncResult`

`AsyncResult` wraps a sync `Result` running in a background thread. It provides the same consumption modes with async syntax:

```python
class AsyncResult:
    """Async wrapper over a thread-backed Result."""

    def __init__(self, sync_fn: Callable[..., Result], *args, **kwargs):
        # The sync call hasn't started yet.
        # It starts when the user awaits or iterates.
        self._sync_fn = sync_fn
        self._args = args
        self._kwargs = kwargs
        self._result: Result | None = None

    def __await__(self):
        """Consume the entire stream in a thread. Return completed Result.

        >>> r = await lm15.acall("gpt-4.1-mini", "Hello.")
        >>> print(r.text)  # str, immediately available
        """
        async def _consume():
            def _run():
                result = self._sync_fn(*self._args, **self._kwargs)
                result._consume()  # drain stream
                return result
            return await asyncio.to_thread(_run)
        return _consume().__await__()

    async def __aiter__(self) -> AsyncIterator[str]:
        """Yield text chunks from a thread-backed stream.

        >>> async for text in lm15.acall("gpt-4.1-mini", "Write a haiku."):
        ...     print(text, end="")
        """
        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()

        def _produce():
            try:
                result = self._sync_fn(*self._args, **self._kwargs)
                for text in result:
                    asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, text)
                asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, None)
                self._result = result
            except Exception as e:
                asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, e)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _produce)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def events(self) -> AsyncIterator[StreamChunk]:
        """Yield all events from a thread-backed stream."""
        # Same queue pattern as __aiter__, but yields StreamChunks
        ...
```

**Key implementation detail:** The thread-to-async bridge uses `asyncio.Queue`. The sync `Result` runs in a thread, pushing items into the queue. The async consumer awaits the queue. This gives true interleaved async behavior — the event loop can handle other coroutines between chunks — while the actual I/O happens in a thread.

### Usage patterns

```python
# Async stateless — blocking (awaits full result)
r = await lm15.acall("gpt-4.1-mini", "Hello.")
print(r.text)   # str, immediately available — no further awaiting

# Async stateless — streaming text
async for text in lm15.acall("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")

# Async stateless — streaming events
async for e in lm15.acall("gpt-4.1-mini", "Explain TCP.", reasoning=True).events():
    match e.type:
        case "thinking": print(f"💭 {e.text}", end="")
        case "text":     print(e.text, end="")

# Async stateful — blocking
agent = lm15.model("claude-sonnet-4-5", system="You are helpful.")
r = await agent.acall("Hello.")
print(r.text)

# Async stateful — streaming
async for text in agent.acall("Write a haiku."):
    print(text, end="")
```

### `await` semantics

`await lm15.acall(...)` runs the entire sync call (including tool loops, stream consumption, retry logic) in a background thread. It returns a **completed `Result`** — all properties (`.text`, `.usage`, etc.) are immediately available as regular attribute access, no further awaiting needed. The `Result` is the same type as the sync version.

### `on_tool_call` in async mode

`on_tool_call` remains a sync callback. It runs inside the background thread alongside the rest of the sync stack. This means it can do blocking I/O (like `input()` for approval gates) without blocking the event loop.

---

## Concurrency Patterns

lm15 does not provide parallel/gather/map helpers. Concurrency is orchestration — it belongs in user code or in frameworks above lm15. The primitives (`lm15.call()`, `lm15.acall()`, `agent.call()`, `agent.acall()`, `agent.copy()`) compose naturally with stdlib concurrency tools.

### Sync parallel (ThreadPoolExecutor)

```python
from concurrent.futures import ThreadPoolExecutor
import lm15

# Fan-out across providers
models = ["gpt-4.1-mini", "claude-sonnet-4-5", "gemini-2.5-flash"]
with ThreadPoolExecutor(3) as pool:
    results = list(pool.map(lambda m: lm15.call(m, "Explain TCP."), models))

# Batch prompts
prompts = ["Summarize DNA.", "Summarize RNA.", "Summarize proteins."]
with ThreadPoolExecutor(3) as pool:
    results = list(pool.map(lambda p: lm15.call("gpt-4.1-mini", p), prompts))
```

### Async parallel (asyncio.gather)

```python
import asyncio, lm15

results = await asyncio.gather(
    lm15.acall("gpt-4.1-mini", "Explain TCP."),
    lm15.acall("claude-sonnet-4-5", "Explain TCP."),
    lm15.acall("gemini-2.5-flash", "Explain TCP."),
)
```

### Stateful parallel (copy + gather)

```python
agent = lm15.model("claude-sonnet-4-5", system="You are a data analyst.")
agent.call("Here's the Q4 dataset: ...")

# Each copy has the full conversation history, runs independently
prompts = ["Analyze by region.", "Analyze by product.", "Analyze by quarter."]
results = await asyncio.gather(
    *(agent.copy().acall(p) for p in prompts)
)
# Original agent is untouched
```

### Why lm15 doesn't provide `parallel()` or `gather()`

These patterns are 1–3 lines of stdlib. Wrapping them in lm15 would impose opinions about error handling, ordering, cancellation, and rate limiting that every orchestration layer will want differently. lm15's job is to make each individual call correct. Composing calls is the user's job (or their framework's).

---

## Thread Safety

- **`lm15.call()` (stateless)** is fully thread-safe. Each call creates its own `Result` backed by its own stream.
- **`lm15.model()` instances** are **NOT** thread-safe. They maintain mutable conversation state (`_conversation`, `history`, `_pending_tool_calls`). Do not call the same `Model` instance from multiple threads. Instead, fork with `agent.copy()` to create independent instances with their own state.
- **`UniversalLM`** (the client) is thread-safe for `stream()` calls. The adapter `stream()` methods create independent HTTP connections per call.
- **`Conversation`** is **NOT** thread-safe. It's a mutable list builder.
- **`Result`** is NOT thread-safe. Each `Result` should be consumed by a single thread or coroutine.

---

## Module-Level API

### `lm15.call()` — the primary entry point

```python
def call(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list[Callable | Tool | str] | None = None,
    on_tool_call: Callable[[ToolCallInfo], str | None] | None = None,
    reasoning: bool | dict | None = None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    max_tool_rounds: int = 8,
    retries: int = 0,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> Result
```

### `lm15.acall()` — async entry point

Same signature as `lm15.call()`, returns `AsyncResult`. Runs the sync call in a background thread via `asyncio.to_thread()`.

### `lm15.model()` — create a stateful model

```python
def model(
    model_name: str,
    *,
    system: str | None = None,
    tools: list[Tool | Callable | str] | None = None,
    on_tool_call: Callable[[ToolCallInfo], str | None] | None = None,
    provider: str | None = None,
    retries: int = 0,
    cache: bool | dict = False,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_tool_rounds: int = 8,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
) -> Model
```

### `lm15.configure()` — set defaults

```python
def configure(*, env: str | None = None, api_key: str | dict[str, str] | None = None) -> None
```

Unchanged from current.

### Other module functions (unchanged)

```python
lm15.upload(model, path, *, media_type=None, provider=None) -> Part
lm15.prepare(model, prompt, **kwargs) -> LMRequest
lm15.send(request, **kwargs) -> Result  # NOTE: now returns Result, not LMResponse
lm15.models(**kwargs) -> list[ModelSpec]
lm15.providers() -> dict[str, tuple[str, ...]]
lm15.providers_info(**kwargs) -> dict[str, dict]
```

---

## `Model` Class

### Methods

```python
class Model:
    def call(self, prompt=None, *, messages=None, tools=None, on_tool_call=None,
            reasoning=None, prefill=None, output=None, prompt_caching=None,
            temperature=None, max_tokens=None, top_p=None, stop=None,
            max_tool_rounds=None, provider=None) -> Result: ...

    def acall(self, prompt=None, **kwargs) -> AsyncResult: ...

    def submit_tools(self, results: dict[str, Any], *, provider=None) -> Result: ...

    def upload(self, path, *, media_type=None) -> Part: ...

    def prepare(self, prompt=None, **kwargs) -> LMRequest: ...

    # Fork (replaces with_model, with_system, with_tools, with_provider)
    def copy(self, *, model=UNSET, system=UNSET, tools=UNSET,
             on_tool_call=UNSET, provider=UNSET, retries=UNSET,
             cache=UNSET, prompt_caching=UNSET, temperature=UNSET,
             max_tokens=UNSET, max_tool_rounds=UNSET,
             history: bool = True) -> Model: ...

    # State
    history: list[HistoryEntry]  # with .clear()
    model: str
    system: str | None
    provider: str | None
```

Key change: `Model.call()` replaces both `Model.__call__()` and `Model.stream()`. It returns `Result`. There is no separate `.stream()` method — `Result` handles both consumption modes.

Similarly, `submit_tools()` returns `Result` instead of `LMResponse`.

---

## Adapter Contract

### Before (v2)

```python
class LMAdapter(Protocol):
    def complete(self, request: LMRequest) -> LMResponse: ...
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]: ...
    # ... other endpoints
```

### After (v3)

```python
class LMAdapter(Protocol):
    provider: str
    capabilities: Capabilities
    supports: EndpointSupport
    manifest: ProviderManifest

    def stream(self, request: LMRequest) -> Iterator[StreamEvent]: ...

    # Non-streaming endpoints (unchanged)
    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse: ...
    def batch_submit(self, request: BatchRequest) -> BatchResponse: ...
    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...
    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse: ...
    def live(self, config: LiveConfig) -> LiveSession: ...
```

`complete()` is removed from the adapter protocol. The client layer constructs a `Result` from `adapter.stream()`.

### `BaseProviderAdapter` changes

```python
class BaseProviderAdapter:
    # REMOVED: def complete(self, request: LMRequest) -> LMResponse

    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        """Build request with stream=True, parse SSE events."""
        req = self.build_request(request)  # always builds a streaming request
        for raw in parse_sse(self.transport.stream(req)):
            evt = self.parse_stream_event(request, raw)
            if evt is not None:
                yield evt

    # REMOVED: def parse_response(self, request, response) -> LMResponse
    # CHANGED: build_request no longer takes a `stream` parameter
    def build_request(self, request: LMRequest) -> HttpRequest: ...
    def parse_stream_event(self, request: LMRequest, raw_event: SSEEvent) -> StreamEvent | None: ...
    def normalize_error(self, status: int, body: str) -> ProviderError: ...
```

Each provider adapter (OpenAI, Anthropic, Gemini) removes its `parse_response()` method and updates `build_request()` to always set stream=True.

### `UniversalLM` changes

```python
class UniversalLM:
    # REMOVED: def complete(self, request, provider=None) -> LMResponse

    def stream(self, request: LMRequest, provider: str | None = None) -> Iterator[StreamEvent]:
        """Route to adapter and yield stream events."""
        adapter = self._adapter(request.model, provider)
        run = self.middleware.wrap_stream(adapter.stream)
        yield from run(request)
```

The `Result` is constructed in the `Model` layer or the `api.py` layer, not in `UniversalLM`. `UniversalLM` only deals with raw `StreamEvent` iterators.

---

## Middleware Changes

### `MiddlewarePipeline`

```python
class MiddlewarePipeline:
    stream_mw: list[StreamMiddleware]  # kept
    # REMOVED: complete_mw: list[CompleteMiddleware]
```

Since all completions go through `stream()`, only `StreamMiddleware` is needed. The existing `with_retries`, `with_cache`, and `with_history` middlewares need to be adapted to work on streams.

However, retry and cache semantics on streams are more complex. Consider:

- **Retries on streams**: If the stream fails mid-way, retry the entire request. The middleware wraps the stream factory, not the stream itself.
- **Cache on streams**: Cache the materialized `LMResponse`. On cache hit, yield synthetic events from the cached response. This is handled at the `Model`/`Result` level, not at the stream middleware level.
- **History**: Recording happens in the `on_finished` callback when the `Result` is consumed, not in middleware.

For simplicity, retries and cache move to the `Model` layer (where they already partially live via `_complete_with_cache`). The `MiddlewarePipeline` can still exist for advanced users who want to inject custom stream processing.

---

## Prompt Caching

Unchanged from v2. `prompt_caching=True` on `lm15.call()` or `lm15.model()` enables provider-side prefix caching. Per-part `cache=True` on `Part` objects for fine-grained control.

The adapters handle cache mechanics:
- **Anthropic**: `cache_control` breakpoints on system prompt and advancing message boundary
- **Gemini**: Creates/reuses `CachedContent` for conversation prefix
- **OpenAI**: No-op (automatic prefix caching)

---

## Provider Built-in Tools

Built-in tools (strings in the `tools=` list) execute server-side:

```python
r = lm15.call("gpt-4.1-mini", "Latest AI news", tools=["web_search"])
print(r.text)
for c in r.citations:
    print(f"  [{c.title}]({c.url})")
```

They:
- Are passed to the provider in the request (adapter translates to provider-specific format)
- Execute entirely server-side — no `on_tool_call`, no client-side loop
- Results (text, citations) arrive in the normal response stream
- Coexist with user function tools

---

## Error Handling

### Error types (unchanged)

```
ULMError
├── TransportError
└── ProviderError
    ├── AuthError
    ├── BillingError
    ├── RateLimitError
    ├── InvalidRequestError
    │   └── ContextLengthError
    ├── ServerError
    ├── TimeoutError
    ├── UnsupportedModelError
    └── UnsupportedFeatureError
```

### Error timing with always-stream

Most errors occur at connection time (before any SSE events) and behave identically to non-streaming:
- `AuthError` — 401/403 before stream starts
- `BillingError` — 402 before stream starts
- `RateLimitError` — 429 before stream starts (sometimes mid-stream)

Errors that can occur mid-stream:
- `ContextLengthError` — detected after initial processing
- `ServerError` — provider failure during generation

Mid-stream errors are raised as exceptions when the user accesses a blocking property or during iteration. The partial data accumulated before the error is accessible via `Result._materialize_response()` for debugging.

### Retries

Retries are handled at the `Model` layer (for stateful mode) and at the `lm15.call()` function level (for stateless mode). Retry logic:

1. Start the stream
2. If a retryable error occurs (before or during streaming), retry the entire request
3. Retryable errors: `RateLimitError`, `TimeoutError`, `ServerError`, `TransportError`
4. Exponential backoff: `0.2 * 2^attempt` seconds
5. Maximum retries specified by `retries=` parameter

---

## Migration from v2

### Removed

| v2 | v3 | Notes |
|----|----|-------|
| `lm15.call() → LMResponse` | `lm15.call() → Result` | Same name, new return type |
| `lm15.stream() → Stream` | `lm15.call() → Result` | Unified into `call()`, consume via iteration |
| `model.stream()` | `model()` | Same entry point, consume via iteration |
| `stream.text` (generator property) | `for text in result` | `__iter__` yields text |
| `stream.response` | `result.response` | Same, but always available |
| `Stream` class | `Result` class | Unified type |
| `adapter.complete()` | Removed | Always stream |
| `adapter.parse_response()` | Removed | Stream events only |
| `MiddlewarePipeline.complete_mw` | Removed | Stream middleware only |
| `model.with_model()` | `model.copy(model=...)` | Single method replaces four |
| `model.with_system()` | `model.copy(system=...)` | Preserves history by default |
| `model.with_tools()` | `model.copy(tools=...)` | |
| `model.with_provider()` | `model.copy(provider=...)` | |

### Added

| v3 | Purpose | Implementation |
|----|---------|----------------|
| `lm15.call() → Result` | Unified entry point | Regular module function |
| `lm15.acall()` | Async entry point | `asyncio.to_thread` over `call()` |
| `model.acall()` | Async on model | `asyncio.to_thread` over `model()` |
| `model.copy(**kwargs)` | Fork model with overrides | Replaces all `with_*` methods |
| `Result` | Unified sync response type | Replaces `Stream` |
| `AsyncResult` | Unified async response type | Thread + `asyncio.Queue` bridge |
| `Conversation` | Message list builder for stateless mode | Pure data, no I/O |
| `ToolCallInfo` | Info passed to `on_tool_call` | Frozen dataclass |
| `on_tool_call=` parameter | Tool interception hook | Called in tool loop |
| `max_tool_rounds=` parameter | Tool loop limit (default 8) | Replaces hardcoded 8 |

### Backward compatibility

For a transition period, keep `lm15.call()` and `lm15.stream()` as aliases:

```python
def call(model, prompt=None, **kwargs):
    """Deprecated: use lm15.call() directly."""
    r = call(model, prompt, **kwargs)
    return r.response  # consume and return LMResponse for compat

def stream(model, prompt=None, **kwargs):
    """Deprecated: use lm15.call() directly."""
    return call(model, prompt, **kwargs)  # return Result (replaces Stream)
```

---

## Complete Examples

### Simplest possible

```python
import lm15
print(lm15.call("gpt-4.1-mini", "Hello.").text)
```

### Streaming

```python
import lm15
for text in lm15.call("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")
```

### Tools with auto-execute

```python
import lm15

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C in {city}"

print(lm15.call("gpt-4.1-mini", "Weather in Montreal?", tools=[get_weather]).text)
```

### Streaming with tools and approval

```python
import lm15

def approve(call):
    if call.name == "write_file":
        return None if input(f"Write {call.input['path']}? ") == "y" else "Denied."
    return None

for e in lm15.call("claude-sonnet-4-5", "Refactor auth.py",
              tools=[read_file, write_file], on_tool_call=approve).events():
    match e.type:
        case "text":        print(e.text, end="")
        case "tool_call":   print(f"\n🔧 {e.name}")
        case "tool_result": print(f"📎 {e.text}")
```

### Stateful agent

```python
import lm15

agent = lm15.model("claude-sonnet-4-5",
    system="You are a coding assistant.",
    tools=[read_file, write_file, run_command],
    on_tool_call=approve,
    prompt_caching=True,
)

for e in agent.call("Add validation to models.py", reasoning=True).events():
    match e.type:
        case "thinking": print(f"💭 {e.text}", end="")
        case "text":     print(e.text, end="")

print(agent.call("Now add tests.").text)
print(f"Turns: {len(agent.history)}")
```

### Stateless multi-turn with Conversation

```python
import lm15
from lm15 import Conversation, Tool

conv = Conversation(system="You are helpful.")
conv.user("My name is Max.")

r = lm15.call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
conv.assistant(r.response)

conv.user("What's my name?")
r2 = lm15.call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
print(r2.text)
```

### Cross-model pipeline

```python
import lm15

cat = lm15.call("gpt-4.1-mini", "Draw a cat.", output="image")
desc = lm15.call("claude-sonnet-4-5", ["Describe this.", cat.image])
print(desc.text)
```

### Image and audio bytes

```python
import lm15

r = lm15.call("gpt-4.1-mini", "Draw a sunset.", output="image")
with open("sunset.png", "wb") as f:
    f.write(r.image_bytes)

r = lm15.call("gpt-4o-mini-tts", "Say hello.", output="audio")
with open("hello.wav", "wb") as f:
    f.write(r.audio_bytes)
```

### Config-driven

```python
import lm15

config = {"model": "gpt-4.1-mini", "system": "You are terse.", "temperature": 0}
r = lm15.call(prompt="Summarize DNA.", **config)
print(r.text)
```

### Async

```python
import lm15

# Blocking
r = await lm15.acall("gpt-4.1-mini", "Hello.")
print(r.text)

# Streaming
async for text in lm15.acall("gpt-4.1-mini", "Write a haiku."):
    print(text, end="")

# Stateful
agent = lm15.model("claude-sonnet-4-5", system="You are helpful.")
r = await agent.acall("Hello.")
print(r.text)

async for text in agent.acall("Write a haiku."):
    print(text, end="")
```

---

## Implementation Order

### Phase 1: Core unification (sync only)
1. Add `ToolCallInfo` dataclass to `types.py`
2. Create `Result` class (replaces `Stream`) in `result.py`
3. Rename `Model.__call__()` to `Model.call()`, return `Result` instead of `LMResponse`
4. Remove `Model.stream()` — `Model.call()` now returns `Result` which handles both
5. Add `on_tool_call` parameter to `Model.__init__` and `Model.call()`
6. Implement tool loop in `Result._chunks()`
7. Update `Model.submit_tools()` to return `Result`
8. Update `lm15.call()` to return `Result` instead of `LMResponse`
9. Remove `lm15.stream()` (or keep as deprecated alias that returns `Result`)

### Phase 2: Always-stream
1. Remove `parse_response()` from all three adapters
2. Remove `build_request(stream=bool)` — always build streaming request
3. Remove `BaseProviderAdapter.complete()` — only `stream()` remains
4. Remove `UniversalLM.complete()` — only `stream()` remains
5. Remove `MiddlewarePipeline.complete_mw` — only stream middleware
6. Move retry logic to `Model` / `Result` layer (retry the stream on failure)
7. Move cache logic to `Model` / `Result` layer

### Phase 3: Conversation helper
1. Create `Conversation` class
2. Add `Message.tool_results()` convenience method (or make `Conversation.tool_results()` handle the message construction)

### Phase 4: Concurrency and Async
1. Create `AsyncResult` class (thread-to-async bridge via `asyncio.Queue`)
2. Add `lm15.acall()` entry point (`asyncio.to_thread` over `lm15.call()`)
3. Add `Model.acall()` method (`asyncio.to_thread` over `Model.call()`)
4. Test: `await lm15.acall(...)` consumes in thread, returns completed `Result`
5. Test: `async for text in lm15.acall(...)` bridges chunks via queue
6. Test: `asyncio.gather(lm15.acall(...), lm15.acall(...))` runs concurrently
7. Test: thread safety of stateless `lm15.call()` from multiple threads
8. Document concurrency patterns (ThreadPoolExecutor, asyncio.gather, copy + gather)

### Phase 5: Cleanup
1. Update all docs and cookbooks
2. Update examples
3. Update tests
4. Remove deprecated aliases if appropriate
5. Add cookbook: concurrency patterns
6. Add cookbook: async integration (FastAPI example)
