# Using lm15 transports

`lm15.transports` is the bytes-in/bytes-out HTTP layer. It does not know about
models, messages, tools, JSON schemas, or provider error formats. It sends a
transport `TransportRequest` and returns a streaming `TransportResponse`.

The built-in transports are stdlib-only HTTP/1.1 implementations:

```python
from lm15.transports import StdlibTransport, StdlibAsyncTransport
```

Provider LMs create a sync `StdlibTransport` automatically when no transport is
passed, and a router builds one transport that every LM it constructs shares.
For the two everyday knobs — how long to wait and how many connections to
open — you do not need to touch the transport at all:

```python
import lm15

router = lm15.LMRouter(lm15.RouterConfig(
    timeouts=lm15.Timeouts(read=1800),   # a slow local model, or a long non-streaming answer
    max_connections=200,                 # a wide evaluation
))
```

Create a transport explicitly when you want to share a pool across routers,
customize TLS, or pin a proxy. Use the async transport when you are doing
transport-level async I/O yourself.

## Transport request and response models

The transport-level `TransportRequest` is intentionally small:

```python
from lm15.transports import TransportRequest

request = TransportRequest(
    method="POST",
    url="https://api.example.com/v1/messages",
    headers=[("Authorization", "Bearer sk-..."), ("Content-Type", "application/json")],
    body=b'{"hello":"world"}',
    connect_timeout=10.0,
    read_timeout=600.0,
    write_timeout=600.0,
)
```

Important details:

- `headers` is a list of `(name, value)` pairs so order and duplicates can be
  preserved.
- `body` is bytes. JSON encoding belongs in the LM or caller.
- Per-request timeouts are optional. `None` means use the transport default.
- URLs must be `http://` or `https://`.

A `TransportResponse` exposes status, reason, headers, HTTP version, and a streaming byte
iterator.

```python
with transport.stream(request) as response:
    print(response.status, response.reason)
    print(response.header("content-type"))
    body = response.read()
```

Always use a response as a context manager or call `response.close()`. This is
how the underlying connection is returned to the pool or closed safely.

## Basic sync usage

```python
from lm15.transports import TransportRequest, StdlibTransport

with StdlibTransport() as transport:
    req = TransportRequest(method="GET", url="https://example.com/")
    with transport.stream(req) as resp:
        data = resp.read()

print(resp.status)
print(data[:100])
```

The response body streams as chunks. It is not buffered unless you call
`read()`.

```python
with transport.stream(req) as resp:
    for chunk in resp:
        process(chunk)
```

## Basic async usage

```python
from lm15.transports import TransportRequest, StdlibAsyncTransport

async with StdlibAsyncTransport() as transport:
    req = TransportRequest(method="GET", url="https://example.com/")
    async with transport.stream(req) as resp:
        body = await resp.read()
```

Async responses are async iterators:

```python
async with transport.stream(req) as resp:
    async for chunk in resp:
        process(chunk)
```

## JSON requests

The transport does not have a `json=` parameter. Encode JSON explicitly.

```python
import json
from lm15.transports import TransportRequest

payload = {"model": "demo", "input": "hello"}
req = TransportRequest(
    method="POST",
    url="https://api.example.com/v1/responses",
    headers=[("Content-Type", "application/json")],
    body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
)
```

This keeps the transport independent from any provider or serialization policy.

## Streaming lines and SSE

HTTP chunks are arbitrary byte chunks, not necessarily newline-aligned. Use
`TransportResponse.iter_lines()` when you need line-oriented protocols such as
Server-Sent Events.

```python
from lm15.sse import parse_sse

with transport.stream(req) as resp:
    for event in parse_sse(resp.iter_lines()):
        print(event.event, event.data)
```

Async responses expose `aiter_lines()`:

```python
async with transport.stream(req) as resp:
    async for line in resp.aiter_lines():
        print(line)
```

`lm15.sse.parse_sse()` accepts an iterator of byte lines and produces `SSEEvent`
objects. Provider LMs then parse those SSE payloads into typed
`StreamEvent`s.

## Connection pooling

A `StdlibTransport` owns a keep-alive pool keyed by origin `(scheme, host,
port)`. Reuse a transport for many requests to get connection reuse.

```python
transport = StdlibTransport(max_connections=100)
try:
    for url in urls:
        with transport.stream(TransportRequest(method="GET", url=url)) as resp:
            resp.read()
    print(transport.pool_stats())
finally:
    transport.close()
```

`max_connections` (default 100, the httpx and aiohttp norm) is the total pool
slot limit. Concurrent requests beyond it wait for a free slot, up to
`pool_timeout` (default 600 s; `None` waits indefinitely); the error when
that wait runs out names both knobs. A provider's rate limit, not this
number, is the practical ceiling on cloud calls; against a single local
model the server queues what the pool admits.

Close a transport (or use it as a context manager) when you are done with
it. One that is simply dropped closes its idle sockets when garbage-
collected, so a forgotten short-lived client does not leak file descriptors
or warn under `python -W error`.

Idle connections are checked for staleness before reuse. If a server closes a
keep-alive connection while idle, the transport drops it and opens a fresh one.

## Proxies

An explicit `proxy=` on the transport always wins; otherwise, with
`trust_env=True` (the default), the standard environment variables
(`HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`, minus `NO_PROXY`) are
consulted per request. `trust_env=False` makes the transport hermetic.

```python
transport = StdlibTransport(proxy="http://proxy.corp:3128")
transport = StdlibTransport(trust_env=False)   # ignore proxy env vars
```

Only plain-HTTP proxy URLs are supported. HTTPS targets are tunneled
with CONNECT and the TLS handshake runs end-to-end to the origin — the
proxy never sees inside the tunnel.

## Timeouts

Every timeout is per operation, not per request: `read_timeout` bounds the
wait for the *next* byte, so a stream that keeps trickling never times out
and a stalled one fails `read_timeout` seconds after its last byte. The
defaults are the provider SDKs' (OpenAI, Anthropic and litellm all wait
600 s), because a model that thinks for three minutes before its first byte
is ordinary, and a client that gives up sooner reports it as a network
failure — and, under a retry loop, restarts the generation each time.

```python
transport = StdlibTransport(
    connect_timeout=10.0,    # TCP + TLS (+ proxy CONNECT)
    read_timeout=600.0,      # next byte of the reply
    write_timeout=600.0,     # sending the request
    pool_timeout=600.0,      # a free connection; None waits indefinitely
)
```

`lm15.Timeouts(connect=, read=, write=, pool=)` is the same set of numbers
as a value, for `RouterConfig(timeouts=...)`. A read timeout's message says
so — `this is lm15's read timeout, not a server failure — raise it with
Timeouts(read=...)` — so it is not mistaken for a dead server and retried.

Provider builders inherit these settings, including auxiliary model/file/batch/
cache/media and token-scoring calls. They do not impose their own fixed read
limits. Authentication exchanges use separate operation-specific deadlines;
they are not inference transport requests.

A low-level `TransportRequest` can override connect/read/write (not pool).
These are not canonical `Request` or `Config` fields:

```python
req = TransportRequest(
    method="GET",
    url="https://example.com/slow-stream",
    connect_timeout=5.0,
    read_timeout=120.0,
)
```

Timeout errors are typed:

```python
from lm15.transports import ConnectTimeout, ReadTimeout, WriteTimeout
```

Provider LMs catch transport exceptions and translate them to
`lm15.errors.TransportError` for the higher-level API.

## Compressed replies

Requests advertise `Accept-Encoding: identity`: a compressed SSE stream sits
in a proxy's buffer until its window fills, which defeats streaming. Gateways
and CDNs compress anyway; a reply that arrives `Content-Encoding: gzip` (or
`x-gzip`, or `deflate`, zlib-wrapped or raw) is decoded incrementally, so a
compressed stream still streams. `br` and `zstd` have no stdlib codec and
raise a `ProtocolError` that names the coding rather than handing compressed
bytes to a JSON parser.

## Pyodide / FetchTransport

In a page, worker, or Node hosting Pyodide, pass
`FetchTransport(read_timeout=600)` to an async adapter. The read limit bounds
both the initial fetch through response headers and each subsequent body read,
not the total stream duration. A low-level `TransportRequest.read_timeout`
overrides it; `None` inherits it. Timeout raises transport `ReadTimeout`.
Cancellation before headers, a read failure, or closing an unfinished response
aborts the host request. Reader locks are released on completion/close;
`transport.aclose()` also aborts active requests. Aborting does not guarantee
the server has not already served or billed the call.

Fetch does not expose separate socket connect/write deadlines, proxy selection,
a pool-wait budget, or a connection cap. Explicit low-level connect/write
overrides are refused rather than silently ignored. Its header deadline covers
whatever connection/upload work the browser performs before returning headers;
it is not an implementation of the stdlib transport's separate socket budgets.

The host automatically decodes compressed bodies. lm15 checks **visible**
`Content-Encoding` against INV-053 (`identity`, `gzip`, `x-gzip`, `deflate` only),
rejecting even host-supported `br`/`zstd`, but never inflates decoded bytes a
second time. It attempts `Accept-Encoding: identity`; browsers can strip that
forbidden header and negotiate compression themselves. On cross-origin replies
`Content-Encoding` is not CORS-safelisted: servers must expose it with
`Access-Control-Expose-Headers` for this policy check to work. Hidden headers
cannot be checked, and raw compressed bytes/integrity checks belong to the host.
A CORS refusal and a network failure both appear as fetch errors; a CORS refusal
may occur **after** the server has received the request. Host-specific or custom
fetch implementations must honor Fetch's automatically-decoded-body semantics.

## TLS verification

By default, HTTPS uses the system trust store.

```python
transport = StdlibTransport(verify=True)
```

For private test servers, pass a CA bundle path:

```python
transport = StdlibTransport(ca_bundle="./ca.pem")
```

For local development only, verification can be disabled:

```python
transport = StdlibTransport(verify=False)
```

!!! danger "Never ship `verify=False`"
    Disabling verification removes the only proof that you are talking
    to the real server. Anyone on the network path can then read and
    rewrite your traffic, including your API keys. For private test
    CAs, use `ca_bundle=` instead — it keeps verification on.

## Headers and defaults

The HTTP/1.1 codec adds defaults only when the caller did not provide them:

- `Host`
- `User-Agent`
- `Accept: */*`
- `Accept-Encoding: identity`
- `Content-Length` when the request has a body

Caller-provided headers win.

```python
req = TransportRequest(
    method="GET",
    url="https://example.com/",
    headers=[("User-Agent", "my-app/1.0")],
)
```

Header names and values are validated against CR/LF/NUL injection before bytes
are written to the socket.

## Error taxonomy

Transport-level exceptions are exported from `lm15.transports`:

```python
from lm15.transports import (
    ConnectError,
    ConnectTimeout,
    ProtocolError,
    ReadError,
    ReadTimeout,
    TransportError,
    WriteError,
    WriteTimeout,
)
```

HTTP error statuses are not transport exceptions. A `429` or `500` response is
still a valid HTTP response and is returned with its body:

```python
with transport.stream(req) as resp:
    if resp.status >= 400:
        error_body = resp.read().decode("utf-8", errors="replace")
```

Provider LMs are responsible for turning HTTP error statuses into provider
errors such as `RateLimitError` or `ServerError`.

## Cancellation and early close

If a sync response is closed before the body is fully consumed, the connection is
closed instead of returned to the pool.

```python
with transport.stream(req) as resp:
    for i, chunk in enumerate(resp):
        if i == 3:
            break
# connection is discarded, not reused with unread bytes
```

The async transport does the same for early exits and task cancellation.

```python
task = asyncio.create_task(read_stream())
task.cancel()
```

Cancellation closes the underlying writer and frees the pool slot.

## Implement a custom transport

Provider LMs only require a sync object with a `stream(request)` method that
returns a context-managed, iterable response. A minimal response should provide:

```text
status: int
reason: str
headers: list[tuple[str, str]]
http_version: str
__iter__() -> Iterator[bytes]
read() -> bytes
close() -> None
__enter__ / __exit__
```

For testing provider LMs, use the shipped doubles in `lm15.testing`
(`FakeTransport` / `FakeResponse`). Write your own only when you are
building a real custom transport; a small fake shows the required shape:

```python
from dataclasses import dataclass

@dataclass
class FakeResponse:
    status: int
    body: bytes
    headers: list[tuple[str, str]]
    reason: str = "OK"
    http_version: str = "HTTP/1.1"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __iter__(self):
        yield self.body

    def read(self):
        return self.body

    def close(self):
        pass

class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        return self.response
```

Inject it into an LM:

```python
from lm15.providers import OpenAILM

lm = OpenAILM(api_key="test", transport=FakeTransport(...))
```

## Current scope

The stdlib transports are deliberately minimal:

- HTTP/1.1 only.
- No HTTP/2.
- HTTP proxies only (`http://` proxy URLs; HTTPS targets tunneled with
  CONNECT).
- No content-encoding decompression; requests default to
  `Accept-Encoding: identity`.
- No multipart helpers; LMs build multipart bytes when needed.
- No provider-level retry policy; higher layers decide when a retry is safe.

This keeps the transport small, dependency-free, and predictable. Provider
LMs and higher-level clients handle model-specific behavior.
