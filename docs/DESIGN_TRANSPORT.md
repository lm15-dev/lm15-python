# lm15 Transport Design — One Interface, Any Protocol

## Problem

LLM providers use different protocols to serve completions:

- **REST + SSE** — OpenAI, Anthropic, Gemini (standard completions). HTTP POST, server-sent events for streaming.
- **REST blocking** — Some providers, local servers, simple APIs. HTTP POST, full response in one shot.
- **WebSocket** — OpenAI Realtime, Gemini Live. Persistent connection, bidirectional framing.

Users shouldn't know or care which protocol a model uses. `lm15.call("model-name", "prompt")` should work regardless. The transport is the adapter's problem.

The frozen portability bundle reflects this split:

- completion transports normalize to the existing `LMRequest` / `LMResponse` / `StreamEvent` contract in `spec/contract/v2.json`
- persistent live sessions use `AudioFormat`, `LiveConfig`, `LiveClientEvent`, and `LiveServerEvent`
- frozen fixtures live in `spec/fixtures/v2/`

---

## Design Principle

The adapter contract is `stream(LMRequest) -> Iterator[StreamEvent]`. That's it. How the adapter produces those events is an implementation detail. Three transports, one interface:

```
User sees:          lm15.call() → Result
                         │
Adapter translates:      ▼
                    ┌──────────────────────────────────────────────┐
                    │  REST+SSE adapter  → HTTP POST, parse SSE    │
                    │  Blocking adapter  → HTTP POST, synthesize   │
                    │  WebSocket adapter → WS connect, synthesize  │
                    └──────────────────────────────────────────────┘
                         │
All yield:          Iterator[StreamEvent]
```

`Result` doesn't know what produced its events. It consumes `StreamEvent` objects identically regardless of source. Streaming, blocking access, tool loops, `on_tool_call`, retries — all work the same.

---

## REST + SSE (the common path)

This is what OpenAI, Anthropic, and Gemini use for standard completions. The adapter builds an HTTP request with `stream=True`, parses the SSE line protocol, and yields `StreamEvent` objects.

```python
class SomeRESTAdapter(BaseProviderAdapter):
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        http_req = self.build_request(request)
        for raw_event in parse_sse(self.transport.stream(http_req)):
            event = self.parse_stream_event(request, raw_event)
            if event is not None:
                yield event
```

This is the existing v2 implementation. It stays unchanged.

---

## Blocking REST (synthesized stream)

Providers with no SSE endpoint. The adapter makes a blocking HTTP request and synthesizes `StreamEvent` objects from the complete response.

```python
class BlockingAdapter(BaseProviderAdapter):
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        http_req = self.build_request(request)
        resp = self.transport.request(http_req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = self._parse_usage(data)

        yield StreamEvent(type="start", id=data.get("id"), model=request.model)
        yield StreamEvent(type="delta", part_index=0,
                          delta=PartDelta(type="text", text=text))
        yield StreamEvent(type="end", finish_reason="stop", usage=usage)
```

From `Result`'s perspective, indistinguishable from SSE. The only user-visible difference is latency profile — all text arrives in one chunk instead of token-by-token.

---

## WebSocket (live models in completion mode)

This is the key addition. Models that only have a WebSocket/live API (Gemini Live, OpenAI Realtime) can be used through `lm15.call()` — the adapter opens a WebSocket, sends the input, collects the response, closes the connection, and yields `StreamEvent` objects.

### How the adapter works

```python
class LiveModelAdapter(BaseProviderAdapter):
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        # 1. Open WebSocket
        ws = self._connect(request)

        try:
            # 2. Send session config
            ws.send(json.dumps({
                "setup": {
                    "model": request.model,
                    "system_instruction": request.system,
                    "tools": self._format_tools(request.tools),
                }
            }))

            # 3. Send input content (audio, video, text from messages)
            for message in request.messages:
                for part in message.parts:
                    ws.send(self._encode_part(part))

            # 4. Signal end of turn
            ws.send(json.dumps({"client_content": {"turn_complete": True}}))

            # 5. Yield StreamEvents as the model responds
            yield StreamEvent(type="start", model=request.model)

            while True:
                raw = ws.recv()
                event = self._parse_ws_message(request, raw)
                if event is None:
                    continue
                yield event
                if event.type == "end":
                    break

        finally:
            # 6. Close WebSocket
            ws.close()
```

### What the user sees

Nothing different. Same `Result`, same properties, same streaming:

```python
# STT — user doesn't know this is WebSocket
r = lm15.call("gemini-2.0-flash-live",
    Part.audio(data=recording),
    system="Transcribe this audio.")
print(r.text)

# TTS
r = lm15.call("gemini-2.0-flash-live",
    "Say hello in French.",
    output="audio")
speaker.play(r.audio_bytes)

# Multimodal analysis
r = lm15.call("gemini-2.0-flash-live", [
    Part.video(data=screen_recording),
    Part.audio(data=voice_note),
    "What's wrong with this code?",
])
print(r.text)

# Streaming works
for text in lm15.call("gemini-2.0-flash-live", [recording, "Summarize this."]):
    print(text, end="")

# Tools work
r = lm15.call("gemini-2.0-flash-live",
    Part.audio(data=voice_question),
    tools=[search, get_weather])
print(r.text)

# on_tool_call works
r = lm15.call("gemini-2.0-flash-live",
    Part.audio(data=voice_question),
    tools=[read_file, write_file],
    on_tool_call=approve)
print(r.text)
```

### Tool loops over WebSocket

When the model returns a tool call, the `Result` tool loop works the same as with SSE — but the follow-up round goes over the same (or a new) WebSocket connection instead of a new HTTP request:

```
Round 1:  WS connect → send input → recv tool_call → close
          Result executes tool (or calls on_tool_call)
Round 2:  WS connect → send original + tool_result → recv text → close
          Result yields text chunks → finished
```

Or, if the adapter keeps the WebSocket open across rounds:

```
WS connect → send input → recv tool_call
           → send tool_result → recv text → close
```

The adapter decides which strategy to use. `Result` doesn't care — it yields `StreamEvent` objects from whatever `Iterator[StreamEvent]` the adapter provides.

---

## The persistent session API (`lm15.live()`)

Everything above is about using live models in the **completion pattern** — input in, output out, connection closed. This is the common case and the one that composes with `Result`, tools, DSPy, and everything else.

`lm15.live()` exists for a different interaction pattern: **persistent bidirectional sessions** where the connection stays open and both sides send continuously. Voice assistants with live mic/speaker streaming. Interactive tutoring. Long-running conversations with server-side state.

### The test

**If you know your input before you start, use `call()`.** The adapter handles the WebSocket transparently.

**If you're streaming input continuously and the model responds while you're still sending, use `live()`.** You need the session object.

### Session API

```python
session = lm15.live("gemini-2.0-flash-live",
    system="You are a helpful voice assistant.",
    tools=[get_weather],
    on_tool_call=approve,
    voice="alloy",
)

# Non-blocking send
session.send(audio=mic_chunk)
session.send(text="What's the weather?")
session.send(tool_result={call_id: "22°C, sunny"})
session.interrupt()

# Blocking receive (iterator)
for event in session:
    match event.type:
        case "audio":       speaker.play(event.data)
        case "text":        print(event.text, end="")
        case "tool_call":   print(f"🔧 {event.name}")
        case "tool_result": print(f"📎 {event.text}")
        case "turn_end":    pass
        case "error":       print(event.error)

session.close()
```

### Context manager

```python
with lm15.live("gemini-2.0-flash-live", system="You are helpful.") as session:
    session.send(text="Hello!")
    for event in session:
        if event.type == "text":
            print(event.text)
            break
# WebSocket closed automatically
```

### Async

```python
session = await lm15.alive("gemini-2.0-flash-live",
    system="You are helpful.",
    tools=[get_weather],
)

await session.send(audio=mic_chunk)

async for event in session:
    match event.type:
        case "audio": await speaker.play(event.data)
        case "text":  print(event.text, end="")

await session.close()
```

### Concurrent send/receive

Voice requires talking and listening simultaneously. `send()` is non-blocking (queues to the WebSocket write buffer). The iterator is blocking (waits for the next server event). The user composes with threading:

```python
import threading

session = lm15.live("gemini-2.0-flash-live")

def capture():
    for chunk in mic.stream():
        session.send(audio=chunk)

threading.Thread(target=capture, daemon=True).start()

for event in session:
    if event.type == "audio":
        speaker.play(event.data)
```

lm15 provides `send()` (thread-safe, non-blocking) and `__iter__` (blocking receive). The user composes them. lm15 doesn't impose a concurrency model.

### Tools in sessions

Same `tools=` parameter, same `on_tool_call=` hook. Auto-execute works — the session runs the callable and sends the result back automatically. Manual tools work — the user sees a `tool_call` event and calls `session.send(tool_result=...)`.

```python
# Auto-execute: session handles it
session = lm15.live("gemini-2.0-flash-live", tools=[get_weather])

for event in session:
    match event.type:
        case "audio":       speaker.play(event.data)
        case "tool_call":   print(f"🔧 {event.name}")    # already executed
        case "tool_result": print(f"📎 {event.text}")     # result already sent

# Manual: user handles it
from lm15 import Tool
weather = Tool(name="get_weather", description="Get weather", parameters={...})

session = lm15.live("gemini-2.0-flash-live", tools=[weather])

for event in session:
    match event.type:
        case "tool_call":
            result = my_weather_api(event.input)
            session.send(tool_result={event.id: result})
```

---

## Session types

### `SessionEvent`

```python
@dataclass(slots=True, frozen=True)
class SessionEvent:
    type: str   # "audio" | "text" | "tool_call" | "tool_result" | "turn_end" | "interrupted" | "error"
    data: bytes | None = None           # audio bytes (on "audio")
    text: str | None = None             # text content (on "text", "tool_result")
    id: str | None = None               # tool call ID (on "tool_call", "tool_result")
    name: str | None = None             # tool name (on "tool_call")
    input: dict[str, Any] | None = None # tool arguments (on "tool_call")
    usage: Usage | None = None          # token counts (on "turn_end")
    error: str | None = None            # error message (on "error")
```

### `Session`

```python
class Session:
    def __init__(self, ws, config, callable_registry, on_tool_call): ...

    def send(
        self,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> None:
        """Non-blocking. Queues data to the WebSocket."""
        ...

    def interrupt(self) -> None:
        """Tell the model to stop its current response."""
        ...

    def close(self) -> None:
        """Close the WebSocket connection."""
        ...

    def __iter__(self) -> Iterator[SessionEvent]:
        """Blocking receive. Yields events as they arrive."""
        ...

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
```

### `AsyncSession`

```python
class AsyncSession:
    async def send(self, *, audio=None, text=None, tool_result=None) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[SessionEvent]: ...
    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.close()
```

---

## The adapter contract for live

Adapters that support live models implement two methods:

```python
class LMAdapter(Protocol):
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]:
        """Completion pattern. Open connection, send input, collect response, close.
        Works with REST+SSE, blocking REST, or WebSocket — adapter's choice."""
        ...

    def live(self, config: LiveConfig) -> Session:
        """Persistent session pattern. Open connection, return Session object.
        Only for adapters with WebSocket support."""
        ...
```

A single adapter can support both. The Gemini adapter, for example:

- `stream()` — for `lm15.call("gemini-2.0-flash-live", ...)`: opens WebSocket, sends input, yields events, closes.
- `live()` — for `lm15.live("gemini-2.0-flash-live", ...)`: opens WebSocket, returns `Session`, user drives.

Same WebSocket code underneath. Different interaction patterns on top.

---

## Dependency: `websockets`

WebSocket support requires the `websockets` package. It is an **optional dependency**, installed via extras:

```toml
[project]
dependencies = []  # still zero

[project.optional-dependencies]
live = ["websockets>=13.0"]
```

### Why `websockets`

- Pure Python, no C extensions
- Zero transitive dependencies
- ~150KB installed
- ~20-30ms import time
- Battle-tested, security-audited
- Sync and async client APIs

### Lazy import

`websockets` is only imported when a WebSocket adapter method is called. Users who never use live models never import it, never pay for it.

```python
def _connect(self, ...):
    try:
        from websockets.sync.client import connect
    except ImportError:
        raise ImportError(
            "WebSocket models require the 'websockets' package.\n\n"
            "  Install it with: pip install lm15[live]\n"
        )
    return connect(url, additional_headers=headers)
```

### Impact on non-live users

| Concern | Impact |
|---------|--------|
| `pip install lm15` | Unchanged — 0 deps, 408KB, 72ms |
| `import lm15` | Unchanged — 95ms, no websockets loaded |
| `lm15.call("gpt-4.1-mini", ...)` | Unchanged — REST+SSE, no websockets |
| `lm15.call("gemini-2.0-flash-live", ...)` | Requires `pip install lm15[live]` |
| `lm15.live(...)` | Requires `pip install lm15[live]` |

---

## OAuth

OAuth is an auth concern, not a transport concern. But it's relevant here because some providers require OAuth tokens instead of static API keys, and those tokens need to be acquired and refreshed.

### Why OAuth can be stdlib-only

OAuth flows are HTTP requests with specific parameters. Everything needed is in the standard library:

| Component | Stdlib module |
|-----------|--------------|
| HTTP requests | `urllib.request` |
| JSON parsing | `json` |
| URL encoding | `urllib.parse` |
| SHA256 for PKCE | `hashlib` |
| Base64 for PKCE | `base64` |
| Local redirect server | `http.server` |
| Open browser | `webbrowser` |
| Token storage | `json` + `pathlib` |

No dependency needed.

### Auth flows

```python
# Client credentials (machine-to-machine) — simplest
auth = lm15.OAuth(
    token_url="https://provider.com/oauth/token",
    client_id="...",
    client_secret="...",
)

# Authorization code (user-facing) — opens browser
auth = lm15.OAuth(
    authorize_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token",
    client_id="...",
    redirect_port=8400,  # local server catches redirect
)
auth.login()  # opens browser, waits for redirect, exchanges code

# Device code (CLI tools) — displays code, user visits URL
auth = lm15.OAuth(
    device_url="https://provider.com/oauth/device",
    token_url="https://provider.com/oauth/token",
    client_id="...",
)
auth.login()  # prints "Visit https://... and enter code: ABCD-1234"
```

### Integration with adapters

OAuth produces a bearer token. The token refreshes automatically when expired. From the adapter's perspective, it's just a dynamic `api_key`:

```python
# Static key (current)
lm15.call("gpt-4.1-mini", "Hello.", api_key="sk-...")

# OAuth token (new) — same interface, token refreshes transparently
auth = lm15.OAuth(token_url="...", client_id="...", client_secret="...")
lm15.call("gpt-4.1-mini", "Hello.", auth=auth)

# Or on a model
agent = lm15.model("gpt-4.1-mini", auth=auth)
```

### `OAuthAuth` strategy

Extends the existing auth strategy pattern in `auth.py`:

```python
@dataclass(slots=True)
class OAuthAuth(AuthStrategy):
    token_url: str
    client_id: str
    client_secret: str | None = None
    authorize_url: str | None = None
    device_url: str | None = None
    scopes: tuple[str, ...] = ()
    redirect_port: int = 8400
    token_path: Path | None = None  # ~/.lm15/tokens/{provider}.json

    _access_token: str | None = None
    _refresh_token: str | None = None
    _expires_at: float = 0.0

    def apply_headers(self, headers: dict[str, str]) -> dict[str, str]:
        self._ensure_valid_token()
        out = dict(headers)
        out["Authorization"] = f"Bearer {self._access_token}"
        return out

    def _ensure_valid_token(self) -> None:
        if self._access_token and time.time() < self._expires_at - 30:
            return  # still valid (with 30s buffer)
        if self._refresh_token:
            self._do_refresh()
            return
        if self.client_secret:
            self._do_client_credentials()
            return
        raise AuthError("OAuth token expired and no refresh mechanism available.")

    def login(self) -> None:
        """Interactive login. Opens browser or prints device code."""
        if self.device_url:
            self._do_device_flow()
        elif self.authorize_url:
            self._do_auth_code_flow()
        else:
            self._do_client_credentials()

    def _do_client_credentials(self) -> None:
        """POST to token_url with client_id + client_secret."""
        ...

    def _do_refresh(self) -> None:
        """POST to token_url with refresh_token."""
        ...

    def _do_auth_code_flow(self) -> None:
        """Open browser, run local server, catch redirect, exchange code."""
        ...

    def _do_device_flow(self) -> None:
        """POST to device_url, print code, poll for completion."""
        ...
```

### Token persistence

Tokens are cached to disk so the user doesn't re-authenticate on every script run:

```python
auth = lm15.OAuth(
    token_url="https://provider.com/oauth/token",
    client_id="...",
    client_secret="...",
    token_path="~/.lm15/tokens/provider.json",  # persists across runs
)
```

On first call, acquires token and writes to disk. On subsequent calls, reads from disk, refreshes if expired. The file contains `access_token`, `refresh_token`, `expires_at`.

---

## Summary: What lives where

| Concern | Where it lives | Dependencies |
|---------|---------------|-------------|
| REST + SSE transport | `transports/urllib_transport.py`, `transports/pycurl_transport.py` | stdlib (+ optional pycurl) |
| SSE parsing | `sse.py` | stdlib |
| Blocking → stream synthesis | Individual adapter's `stream()` method | stdlib |
| WebSocket → stream synthesis | Individual adapter's `stream()` method | `websockets` (optional extra) |
| WebSocket persistent session | Individual adapter's `live()` method | `websockets` (optional extra) |
| OAuth flows | `auth.py` | stdlib |
| Token persistence | `auth.py` | stdlib |

### Install footprint

```bash
pip install lm15            # 0 deps, 408KB — REST completions, OAuth
pip install lm15[live]      # 1 dep (websockets), ~560KB — adds WebSocket support
```

---

## Implementation checklist

### WebSocket completion support (live models in `call()`)

1. Add `websockets>=13.0` to `[project.optional-dependencies].live`
2. Implement WebSocket `stream()` in Gemini adapter (open, send, collect, close, yield events)
3. Implement WebSocket `stream()` in OpenAI adapter (same pattern, different protocol)
4. Test: `lm15.call("gemini-...-live", audio)` returns `Result` with text
5. Test: `for text in lm15.call("gemini-...-live", audio)` streams text
6. Test: tool calls work over WebSocket completion
7. Test: `on_tool_call` works over WebSocket completion
8. Test: clear error when `websockets` not installed
9. Add adapter guide section on WebSocket adapters

### Persistent session API (`lm15.live()`)

1. Create `Session` class with `send()`, `interrupt()`, `close()`, `__iter__`
2. Create `AsyncSession` class wrapping `Session` with thread bridge
3. Create `SessionEvent` dataclass
4. Add `lm15.live()` entry point in `api.py`
5. Add `lm15.alive()` async entry point
6. Implement `live()` in Gemini adapter (return `Session` wrapping WebSocket)
7. Implement `live()` in OpenAI adapter
8. Implement tool auto-execute in `Session` receive loop
9. Implement `on_tool_call` in `Session`
10. Test: voice conversation pattern (send audio, receive audio)
11. Test: tools in live session (auto and manual)
12. Test: concurrent send/receive from separate threads
13. Test: context manager (`with lm15.live(...) as session`)
14. Test: async session (`async for event in session`)

### OAuth

1. Add `OAuthAuth` to `auth.py`
2. Implement client credentials flow (`_do_client_credentials`)
3. Implement token refresh (`_do_refresh`)
4. Implement authorization code flow (`_do_auth_code_flow`) with local HTTP server
5. Implement device code flow (`_do_device_flow`)
6. Implement PKCE (SHA256 challenge)
7. Implement token persistence to disk
8. Add `auth=` parameter to `lm15.call()`, `lm15.model()`, adapter constructors
9. Test: client credentials acquire and refresh
10. Test: token persistence across process restarts
11. Test: expired token triggers automatic refresh
12. Test: missing websockets gives clear error
13. Add cookbook: OAuth setup for enterprise providers
