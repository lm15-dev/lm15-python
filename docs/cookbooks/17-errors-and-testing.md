# Errors, retries & testing

**Problem** — Production code needs to tell "bad key" from "slow down"
from "prompt too big", retry the right ones, and run its test suite
without a network. lm15 gives you a typed error tree and an injectable
transport; it deliberately gives you no retry policy.

## Recipe

Keys loaded as in [recipe 01](01-first-request.md).

Every provider failure surfaces as a subclass of `LM15Error`. Start
with the most common one: a wrong key. The router builds the LM, the
provider answers 401, lm15 maps it to `AuthError` — with the fix in
the message:

```python
import json
import time

from lm15 import LMRouter, Message, Request
from lm15.router import RouterConfig
from lm15.errors import RETRYABLE_ERRORS, AuthError, ContextLengthError

bad = LMRouter(config=RouterConfig(api_keys={"openai": "sk-proj-wrong"}))
bad.complete(Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),)))
```
```output
Traceback (most recent call last):
  …
  File ".../lm15/providers/base.py", line 131, in complete
    raise error
lm15.errors.AuthError: Incorrect API key provided: sk-proj-*rong. You can
find your API key at https://platform.openai.com/account/api-keys.
(openai, HTTP 401)

  To fix:
    - Check that your API key is correct and not expired
    - Pass the key explicitly (api_key, or RouterConfig api_keys), or on a host with an environment set OPENAI_API_KEY=...
    - Verify your openai account/project has access
```

The guidance is provider-specific because the error carries structured
metadata, not just a string. `env_keys` names the exact variable to
set — useful when you log errors or build your own "fix it" UI:

```python
try:
    bad.complete(Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),)))
except AuthError as err:
    print(err.code, err.status, err.provider, err.env_keys)
```
```output
auth 401 openai ('OPENAI_API_KEY',)
```

`ContextLengthError` is a subclass of `InvalidRequestError`, detected
from the provider's message. This is a real over-long prompt against
Claude's 200k window (rejected requests are not billed):

```python
router = LMRouter()
big = "word " * 260_000
try:
    router.complete(Request(model="claude-sonnet-4-5", messages=(Message.user(big),)))
except ContextLengthError as err:
    print(err.code, err.status)
    print(str(err).partition("\n")[0])
```
```output
context_length 400
prompt is too long: 260024 tokens > 200000 maximum (anthropic, HTTP 400, request req_011Cczaa6Z4CF5Hr9kLymtjS)
```

### Retries are yours

lm15 never retries. Not on 429, not on 5xx, not on a dropped socket —
one call in, one response or one typed error out. What it gives you
instead is classification: `RETRYABLE_ERRORS` is the tuple of error
classes that are safe to retry.

```python
print(tuple(cls.__name__ for cls in RETRYABLE_ERRORS))
```
```output
('RateLimitError', 'TimeoutError', 'ServerError', 'TransportError')
```

The whole retry policy is a loop you own. `RateLimitError.retry_after`
is the provider's requested wait in seconds when available; fall back
to exponential backoff when it is `None`:

```python
def complete_with_retry(router, request, attempts=5, base=0.5):
    for attempt in range(attempts):
        try:
            return router.complete(request)
        except RETRYABLE_ERRORS as err:
            if attempt == attempts - 1:
                raise
            wait = err.retry_after or base * 2 ** attempt
            print(f"attempt {attempt + 1}: {err.code} "
                  f"(retry_after={err.retry_after}); sleeping {wait}s")
            time.sleep(wait)
```

You cannot summon a 429 on demand, so the demonstration uses a fake
transport — which is also exactly how you test this loop offline.

### Testing offline: lm15.testing

Every provider LM takes a `transport`, and lm15 ships the doubles —
`lm15.testing` has `FakeTransport`/`FakeResponse` (wire-level: the
REAL adapter serde runs against your scripted bytes) and `FakeLM`
(canonical-level: script `Response` objects or plain strings, no wire
JSON at all — the right seam for tool loops and retry logic).

Script the wire double: two 429s — the second with a `Retry-After`
header — then a success in the provider's wire format (here OpenAI
Chat Completions). Hand it to the router via
`RouterConfig(transport=...)` and run the retry loop:

```python
ok = json.dumps({
    "id": "chatcmpl-1", "object": "chat.completion", "model": "fake-model",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Hello from the fake."}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
}).encode()
limited = json.dumps({"error": {"message": "Rate limit reached"}}).encode()

from lm15.testing import FakeResponse, FakeTransport

fake = FakeTransport([
    FakeResponse(429, limited),
    FakeResponse(429, limited, headers=[("content-type", "application/json"),
                                        ("Retry-After", "0.05")]),
    FakeResponse(200, ok),
])
offline = LMRouter(config=RouterConfig(api_keys={"openai-chat": "sk-fake"},
                                       env={}, transport=fake))

req = Request(model="openai-chat:fake-model", messages=(Message.user("Hi"),))
print(complete_with_retry(offline, req, base=0.01).text)
```
```output
attempt 1: rate_limit (retry_after=None); sleeping 0.01s
attempt 2: rate_limit (retry_after=0.05); sleeping 0.05s
Hello from the fake.
```

(The second attempt slept the provider's requested 0.05s, not the
exponential 0.02s: adapters parse the `Retry-After` header — both the
delta-seconds and HTTP-date forms — into `err.retry_after` on live
traffic too.)

The fake also records every request it saw, so the same fixture
asserts what went on the wire — no provider, no key, no flake:

```python
print(len(fake.requests), json.loads(fake.requests[0].body)["model"])
```
```output
3 fake-model
```

## How it works

Every adapter funnels failures through `normalize_error(status, body)`:
it extracts the provider's message and code, then maps the HTTP status
onto the tree in [the errors reference](../reference/errors.md) —
`AuthError` (401/403), `BillingError` (402), `RateLimitError` (429),
`InvalidRequestError` (4xx) with `ContextLengthError` and
`UnsupportedModelError` beneath it, `TimeoutError` (408/504),
`ServerError` (5xx). Each error carries `code` (a stable string for
serialization), `provider`, `provider_code`, `status`, `request_id`,
and `retry_after`. `ContextLengthError` detection is message-based and
per-provider: OpenAI's `context_length_exceeded` code, Anthropic's
"prompt is too long", Gemini's token-limit phrasings.

The fake transport works because providers are pure functions around
their transport: `build_request` produces the wire bytes,
`parse_response` consumes them, and the transport in between is a
constructor argument. A fake returning a canned body exercises the
entire serialization path — the same `Response` parsing real traffic
gets. That is why the test suite is hermetic and why yours can be.

`retry_after` comes from two places, provider body first: an adapter's
own error mapping wins, and the HTTP `Retry-After` header (delta-seconds
or HTTP-date) fills the gap on both the `complete()` and `stream()`
paths. Still write `err.retry_after or backoff` — not every 429 carries
either.

## Variations

- **Canonical-level fakes.** `lm15.testing.FakeLM` scripts `Response`
  objects (or strings, or exceptions) and replays them through
  `complete()`/`stream()` — no wire JSON to hand-craft when what you
  are testing is your loop, not a dialect.
- **Async mirror.** `AsyncLMRouter` raises the same error classes from
  `await router.complete(...)`; the retry loop becomes `async def` with
  `await asyncio.sleep(wait)`. An async fake needs `async def stream`
  returning an async-context-manager response (see
  `tests/test_async_adapters.py`).
- **Streaming fakes.** Set the fake's content-type header to
  `text/event-stream` and put SSE lines in the body; `router.stream()`
  parses them like live traffic (`tests/test_router.py`,
  `TestCompleteStream`).
- **Bare `except TimeoutError` works.** `lm15.errors.TimeoutError`
  subclasses the builtin, with lm15 metadata winning in the MRO.
- **Errors serialize.** `canonical_error_code(err)` and
  `error_class_for_code(code)` round-trip the taxonomy as stable
  strings — for logs, queues, or re-raising across a process boundary.
- **Subscription adapters differ.** `claude-code:` / `openai-codex:`
  auth failures carry a `credential_hint` (re-run the CLI login)
  instead of `env_keys` — there is no env var to set.
- **Missing key vs wrong key.** No key at all fails locally at `lm()`
  time with `MissingCredentialError` / `NotConfiguredError`, before any
  network I/O; `AuthError` means the provider rejected a key you sent.

## See also

- [01 — Your first request](01-first-request.md) — key loading, `MissingCredentialError`
- [05 — Streaming](05-streaming.md) — the events a streaming fake must produce
- [16 — Local & OpenAI-compatible servers](16-local-and-compatible-servers.md) — custom transports and `base_url`
- [18 — Provider passthrough](18-provider-passthrough.md)
- [Using the router](../using-the-router.md) — credential resolution order
