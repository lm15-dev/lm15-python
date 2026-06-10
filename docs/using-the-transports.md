# Using lm15 transports

`lm15.transports` is the bytes-in/bytes-out HTTP layer. It does not know about
models, messages, tools, JSON schemas, or provider error formats. It sends a
transport `TransportRequest` and returns a streaming `TransportResponse`.

The built-in transports are stdlib-only HTTP/1.1 implementations:

```python
from lm15.transports import StdlibTransport, StdlibAsyncTransport
```

Provider LMs create a sync `StdlibTransport` automatically when no transport is
passed. Create one explicitly when you want to share a pool, tune timeouts, or
customize TLS. Use the async transport when you are doing transport-level async
I/O yourself.

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
    read_timeout=60.0,
    write_timeout=60.0,
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
transport = StdlibTransport(max_connections=10)
try:
    for url in urls:
        with transport.stream(TransportRequest(method="GET", url=url)) as resp:
            resp.read()
    print(transport.pool_stats())
finally:
    transport.close()
```

`max_connections` is the total pool slot limit. Concurrent requests block until
a slot is available or the pool wait times out.

Idle connections are checked for staleness before reuse. If a server closes a
keep-alive connection while idle, the transport drops it and opens a fresh one.

## Timeouts

Transport constructors set default timeouts:

```python
transport = StdlibTransport(
    connect_timeout=10.0,
    read_timeout=60.0,
    write_timeout=60.0,
)
```

A request can override any of them:

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

A small fake transport is enough for tests:

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
- No proxy support.
- No content-encoding decompression; requests default to
  `Accept-Encoding: identity`.
- No multipart helpers; LMs build multipart bytes when needed.
- No provider-level retry policy; higher layers decide when a retry is safe.

This keeps the transport small, dependency-free, and predictable. Provider
LMs and higher-level clients handle model-specific behavior.
