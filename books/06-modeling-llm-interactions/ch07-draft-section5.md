## When Zero-Deps Is Wrong

The previous section argued that lm15's needs are narrow enough for the standard
library. This section draws the line — the specific points where the needs
exceed the stdlib and reimplementation becomes more costly than depending.

**OAuth and token refresh.** lm15's providers use API keys — static strings in
an `Authorization` header. But some providers (and many enterprise deployments)
use OAuth: token exchange flows, redirect URIs, PKCE challenges, refresh tokens
that expire and must be rotated mid-session. Implementing OAuth correctly is a
multi-hundred-line project with security implications at every step — token
storage, CSRF prevention, timing-safe comparison. Getting it wrong doesn't
produce a bug; it produces a vulnerability. This is exactly the case where the
bakery wins: `authlib` or `oauthlib` have been audited, attacked, patched, and
hardened over years. Reimplementing their work to avoid a dependency isn't
principled. It's reckless.

lm15 doesn't face this today because no major LLM provider requires OAuth for
API access. If one does — or if enterprise customers require OAuth-based proxy
authentication — lm15 would need to either add a dependency or build a robust
OAuth implementation from scratch. The honest answer is that it should add the
dependency. Some problems are too security-critical to reimplement.

**WebSocket connections.** Some providers offer real-time APIs over WebSocket —
persistent, bidirectional connections for voice, live transcription, or
interactive sessions. Python's standard library has no WebSocket client.
Reimplementing the WebSocket protocol — frame parsing, masking, ping/pong,
connection upgrade, close handshake — is roughly 500 lines of careful,
spec-compliant code with its own edge cases and security considerations. The
`websockets` library handles this and is well-maintained. Reimplementing it to
stay at zero deps would be spending engineering effort on a solved problem to
satisfy a constraint that exists for a different reason.

lm15 currently handles live/real-time APIs in its type system (`LiveConfig`,
`LiveClientEvent`, `LiveServerEvent`) but the actual WebSocket transport is a
gap. If live APIs become central to lm15's value proposition, a WebSocket
dependency — or an optional one, like pycurl — would be the pragmatic choice.

**Async at scale.** lm15 is synchronous. `urllib.request.urlopen` blocks until
the response arrives. For interactive use — one call at a time, a human waiting
for the result — this is fine. For a batch pipeline processing 500 documents
concurrently, it means 500 threads, each blocking on I/O, each consuming memory
and OS resources for the privilege of waiting.

Async HTTP (via `httpx` or `aiohttp`) solves this with cooperative multitasking
— hundreds of concurrent requests on a single thread, with the event loop
switching between them during I/O waits. But async requires an async runtime
(`asyncio`, `trio`, `uvloop`), which is a dependency, which is a framework,
which is a commitment that propagates through the entire codebase. lm15 can't
add async without either depending on an async runtime or reimplementing one —
and reimplementing an async runtime is not a sentence anyone should write
seriously.

The practical escape is `pycurl`, which lm15 already supports as an optional
dependency. pycurl provides connection reuse and HTTP/2 multiplexing, which
addresses the throughput problem without requiring async. It's not as efficient
as true async (each call still blocks a thread), but it eliminates the TCP/TLS
handshake overhead that makes sequential `urllib` calls slow in batch scenarios.
The optional dependency pattern — zero deps by default, better performance if
you install pycurl — is lm15's compromise between the zero-dep principle and the
practical need for throughput.

**Connection pooling.** Each `urllib` call opens a new TCP connection, performs
a TLS handshake, sends the request, receives the response, and closes the
connection. The handshake alone takes 100-300ms depending on latency to the
provider's servers. For a single call, this overhead is invisible — it's a
fraction of the model's generation time. For 100 sequential calls in a batch,
it's 10-30 seconds of pure handshake overhead. `requests` and `httpx` maintain
connection pools — they keep TCP connections open and reuse them, amortizing the
handshake across many calls. `urllib` doesn't, and can't without reimplementing
connection management.

Again, pycurl partially addresses this — libcurl maintains a connection cache
internally. But the user must opt in with `pip install pycurl`, and pycurl has
build dependencies (libcurl headers, OpenSSL headers) that can fail on minimal
environments. The zero-dep default pays the handshake cost on every call. For
most interactive use, this is negligible. For batch processing, it's the
dominant cost.

**The line.** The principle from the previous section — zero deps works when
needs are narrower than what deps provide — has a corollary: **zero deps fails
when needs exceed what the stdlib provides.** OAuth, WebSocket, async, and
connection pooling are all cases where the stdlib is genuinely inadequate — not
"worse than the specialized library" (that's always true) but "unable to do the
job at all." You can't add OAuth to `urllib` with clever coding. You can't add
WebSocket to the standard library with a wrapper. The need exceeds the tool.

lm15's position is defensible today because its needs are narrow: synchronous
HTTP, API key auth, SSE streaming. If the needs widen — OAuth providers,
WebSocket-based real-time APIs, high-throughput batch processing as a core use
case — the zero-dep position would need to retreat. The retreat wouldn't be a
failure of principle. It would be an acknowledgment that the principle's
precondition — needs within the stdlib's coverage — no longer holds.
