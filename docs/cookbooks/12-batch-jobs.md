# Batch jobs

**Provisional in 1.0:** this API ships, but may change incompatibly in 1.x
with a contract change entry. Pin your package version; see
[release scope](../roadmap.md#what-ships-in-10-and-what-is-stable).

**Problem** — You have five hundred requests that nobody is waiting on.
Running them through `complete()` costs full price and busies your
process; what you want is to hand the whole stack to the provider, walk
away, and pick up the answers later at about half price. Every provider
sells this — behind three different wire shapes, two file formats, and
a status zoo.

```text
complete(request)  -> Response            now,    full price
stream(request)    -> events -> Response  now,    full price, incremental
batch(requests)    -> BatchJob            later,  ~half price
```

Same canonical `Request` in all three. Batch is not a different kind of
asking — it is the same conversation on a different clock.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

### Submit

`batch()` takes ordinary `Request` objects — the same ones you would
pass to `complete()`:

```python
from lm15 import LMRouter, Message, Request

router = LMRouter()
lm = router.lm("claude-sonnet-4-5")

capitals = ["France", "Japan", "Brazil", "Kenya", "Norway"]
job = lm.batch([
    Request(model="claude-sonnet-4-5",
            messages=(Message.user(f"Capital of {country}? One word."),))
    for country in capitals
])
print(job.id, job.status)
```
```output
msgbatch_01XrjBa9sAgJrANzykgQL3qo running
```

That is the whole submission. A batch job is a **ticket**: the provider
holds your requests, and `job.id` is the ticket number — a plain string
you can store anywhere.

### Re-attach, wait, collect

Your process can exit and come back hours later. `batch_job(id)`
rebuilds the handle from the id alone:

```python
job = lm.batch_job("msgbatch_01XrjBa9sAgJrANzykgQL3qo")
print("re-attach:", job.status)

info = job.wait()
print("wait ->", info.status)

for entry in job.results():
    print(entry.index, entry.outcome, entry.response.text)
```
```output
re-attach: running
wait -> completed
0 succeeded Paris
1 succeeded Tokyo
2 succeeded Brasília
3 succeeded Nairobi
4 succeeded Oslo
```

Results come back **in submission order**, one entry per request —
lm15 re-sorts them, because providers may not (Anthropic returned them
out of order in live testing). Each `entry.response` is a full
canonical `Response`, identical in shape to what `complete()` returns.

### Lost the ticket?

The provider is the system of record, not your notebook. Enumerate:

```python
for job in lm.batches(limit=3):
    print(job.id[:22] + "…", job.status, job.created_at)
```
```output
msgbatch_01XrjBa9sAgJr… completed 2026-09-01T12:12:24Z
msgbatch_01LKpzTz5ohcv… completed 2026-08-31T19:22:09Z
msgbatch_01YVTLW9C5yLa… completed 2026-06-11T14:38:54Z
```

A lost id is an inconvenience, never a loss — the same lesson `atq`,
print queues, and cloud job APIs settled decades ago.

### Partial failure is a normal outcome

One bad request does not poison the batch. Each entry carries its own
outcome:

```python
mixed = lm.batch([
    Request(model="claude-sonnet-4-5",
            messages=(Message.user("Capital of Peru? One word."),)),
    Request(model="claude-nonexistent-model",
            messages=(Message.user("Hi"),)),
])
mixed.wait()
for e in mixed.results():
    print(e.index, e.outcome,
          e.response.text if e.response else (e.error.code, e.error.message))
```
```output
0 succeeded Lima
1 errored ('invalid_request', 'model: claude-nonexistent-model')
```

`outcome` is a closed vocabulary: `succeeded` entries carry a
`Response`, `errored` entries carry a typed `ErrorDetail`, and
`cancelled`/`expired` entries carry neither.

## How it works

The router routes chat; batch lives on the provider LM, so
`router.lm()` is the bridge. Under the hood the three wires could not
be more different, and lm15 absorbs all of it:

- **Anthropic** has a true server-side queue (`/v1/messages/batches`).
  Submission reuses the exact frozen chat request mapping.
- **OpenAI** batches are a JSONL file upload plus a batch object.
  lm15 writes and uploads the file inside `batch()`; the file ids ride
  in `provider_data`. Trade-off: each batch leaves a JSONL file on your
  OpenAI account.
- **Gemini** inlines the requests into a long-running batch operation
  (`batchGenerateContent`).

Every per-entry result body is a normal chat response in that
provider's wire format, parsed by the same frozen serde `complete()`
uses. Job status is a closed vocabulary — `queued`, `running`,
`cancelling`, `completed`, `failed`, `cancelled`, `expired` — folded
from each provider's own words, which stay verbatim in
`provider_data`.

Batches complete within a provider window (typically 24 h; usually
minutes in practice) at roughly half the synchronous token price.
Providers without a batch endpoint raise `UnsupportedFeatureError` —
lm15 never falls back to silently looping `complete()` at full price.

## Variations

- **Async**: `AsyncLMRouter` gives the same surface with `await` —
  `await lm.batch(...)`, `await job.wait()`, `await job.results()` —
  driving the same pure request/parse hooks as the sync path.
- **Labels**: `lm.batch(requests, label="nightly-eval")` rides provider
  metadata on OpenAI and Gemini, and helps the list-then-recognize
  pattern above. Anthropic's wire has no metadata slot and **raises** —
  lm15 never silently drops a field it cannot deliver.
- **Cancel**: `job.cancel()` returns the job snapshot; a cancel during
  validation can legitimately end with zero entries.
- **Polling knobs**: `job.wait(poll_every=30.0, timeout=3600.0)`.
  `wait()` returns the terminal snapshot rather than raising on
  `failed`, mirroring the entry-level honesty of `results()`.

## See also

- [13 — Media generation](13-media-generation.md) — the other non-chat
  endpoints.
- [16 — Local & OpenAI-compatible servers](16-local-and-compatible-servers.md)
- [17 — Errors, retries & testing](17-errors-and-testing.md) — the
  typed error tree entries carry.
- [18 — Provider passthrough](18-provider-passthrough.md) — reading
  `provider_data`.
