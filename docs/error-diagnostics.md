# Understanding “no capacity” and rate limits

A `429` means the provider rejected the request because of rate or capacity
limits. It does **not** by itself mean your model or server address is wrong.
Azure can report `no_capacity` even when both addresses support the model.

lm15 keeps the provider's message and adds useful evidence when available:

```python
from lm15 import RateLimitError

try:
    response = router.complete(request)
except RateLimitError as error:
    print(error)                      # message, request ID, wait advice, limits
    print(error.retry_after)          # seconds, or None if no usable advice
    print(error.request_id)           # reference for provider support
    print(dict(error.rate_limit_headers))
    # Your application decides whether and when to retry.
```

For the Azure failure that prompted this feature, the headers included:

```text
x-ratelimit-limit-requests: 1
x-ratelimit-remaining-requests: -1
x-ratelimit-reset-requests: 105
retry-after: 39
```

These numbers disagree with a simple “wait exactly one minute” assumption.
They are the provider's reports, not a quota calculation performed by lm15.
In our test, longer gaps restored success on both Azure addresses.

## What the fields mean

- **`retry_after`**: usable provider advice in seconds. A valid error-body
  value wins, then `Retry-After`, then numerical `retry-after-ms` or
  `x-ms-retry-after-ms`. No valid advice means `None`, not zero.
- **`rate_limit_headers`**: a read-only mapping of lowercase header names to
  tuples of original values. Multiple values and negative balances stay
  visible. Reset values stay unconverted: providers use seconds, duration
  strings, or timestamps. Do not treat them all as a number of seconds.
- **`request_id`**: the provider's reference, including Azure's
  `apim-request-id` fallback, useful when reporting an issue.

Wait advice is not a guarantee of success. Neither this feature nor the
error's `retryable` classification adds automatic retries, increases quota,
purchases capacity, or changes the endpoint. Use bounded retries in your
application when appropriate; model calls can have costs and side effects.

## Streaming and saved errors

The same diagnostics work for normal calls, async calls, and auxiliary
operations such as file upload or model listing. A rejected stream carries
metadata on the exception. An error *inside* HTTP 200 carries handshake
evidence in `event.error.http_response`; saving/reloading that event preserves
it, and response materialization carries it onto the exception. HTTP 200 is
not assigned as the error's status. Handshake headers are not a live quota
feed and may be older than the stream failure.

## Privacy and limits

Only a documented set of rate-limit headers is retained—not cookies,
credentials, arbitrary headers, or `x-ratelimit-key`. Values are bounded to
four per name and 256 printable ASCII characters each; malformed/oversized
values are omitted, not turned into plausible numbers. Long error displays
show a bounded preview while retained values remain available on the error.
Provider error content still belongs under your application's logging policy.

The cross-language rules and exact header list live in
[lm15-contract/docs/error-diagnostics.md](https://github.com/lm15-dev/lm15-contract/blob/main/docs/error-diagnostics.md).
