# Prompt caching

**Problem** — Your requests share a large fixed prefix: a system prompt,
a policy document, tool schemas. Without caching you pay full input price
for that prefix on every call. Each provider has its own caching API —
`cache_control` blocks, `prompt_cache_key`, `cachedContents` — and lm15
folds them into one `CacheConfig`.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

Build a prefix big enough to matter. Anthropic and OpenAI refuse to
cache prompts under 1024 tokens (~2048 for Haiku), so a short system
prompt caches nothing on any provider.

```python
from lm15 import CacheConfig, Config, LMRouter, Message, Request
from lm15.models import InferencePricing

manual = "\n".join(
    f"Policy {i}: refunds over ${i * 10} require approval level {i % 5 + 1}; "
    f"log every decision under ticket class R{i:03d} before replying."
    for i in range(1, 161)
)
router = LMRouter()
print(f"~{len(manual) // 4} tokens of policy manual")
```
```output
~4625 tokens of policy manual
```

Attach `CacheConfig()` to the request. On Anthropic this places a
`cache_control: {"type": "ephemeral"}` marker on the system block; the
first call writes the cache:

```python
req = Request(
    model="claude-sonnet-4-5",
    system=f"You are a support agent. Follow this policy manual.\n{manual}",
    messages=(Message.user("Which approval level for a $90 refund? One sentence."),),
    config=Config(max_tokens=100, cache=CacheConfig()),
)
first = router.complete(req)
print(first.text)
print("write:", first.usage.cache_write_tokens, "read:", first.usage.cache_read_tokens)
```
```output
A $90 refund requires approval level 5 according to Policy 9.
write: 4873 read: 0
```

The second call within the TTL (5 minutes by default) reads the prefix
back at one tenth of input price. Only the 20 uncached tokens bill at
the full rate:

```python
second = router.complete(req)
print("write:", second.usage.cache_write_tokens, "read:", second.usage.cache_read_tokens)
print("input:", second.usage.input_tokens)
```
```output
write: 0 read: 4873
input: 20
```

OpenAI caches automatically and for free — no markers, no write charge.
`CacheConfig(key=...)` sends `prompt_cache_key`, which improves hit
rates by routing same-key requests to the same machine. Note the
differences: `cache_write_tokens` is `None` (OpenAI never reports
writes), cached tokens stay inside `input_tokens` rather than replacing
them, and the cached span rounds down to a 128-token boundary:

```python
oreq = Request(
    model="gpt-4.1-mini",
    system=f"You are a support agent. Follow this policy manual.\n{manual}",
    messages=(Message.user("Which approval level for a $90 refund? One sentence."),),
    config=Config(max_tokens=100, cache=CacheConfig(key="support-agent-v2")),
)
for call in ("first", "second"):
    usage = router.complete(oreq).usage
    print(f"{call}: write={usage.cache_write_tokens} read={usage.cache_read_tokens} input={usage.input_tokens}")
```
```output
first: write=None read=0 input=4095
second: write=None read=3968 input=4095
```

Does it pay? `InferencePricing.estimate()` runs the arithmetic on the
counts captured above. With Anthropic's rates (reads 0.1×, writes 1.25×
input price), the write premium on call one is recovered before call
two finishes:

```python
sonnet = InferencePricing(
    input_per_million=3.0, output_per_million=15.0,
    cache_read_per_million=0.30, cache_write_per_million=3.75,
)
cold = sonnet.estimate(input_tokens=4893, output_tokens=50)
write = sonnet.estimate(input_tokens=20, output_tokens=50, cache_write_tokens=4873)
warm = sonnet.estimate(input_tokens=20, output_tokens=50, cache_read_tokens=4873)
print(f"no cache:    ${cold:.5f} per call")
print(f"cache write: ${write:.5f} first call")
print(f"cache read:  ${warm:.5f} every call after")
```
```output
no cache:    $0.01543 per call
cache write: $0.01908 first call
cache read:  $0.00227 every call after
```

## How it works

`CacheConfig` lives on `Config.cache` and carries four fields: `mode`
(`"auto"` or `"off"`), `retention` (`"short"`/`"long"`), `key`, and
`prefix_until_index`. What goes on the wire is per-provider:

- **Anthropic** — explicit breakpoints. With any non-off `CacheConfig`,
  the adapter wraps `system` in a content block with
  `cache_control: {"type": "ephemeral"}`. `retention="long"` adds
  `"ttl": "1h"` (billed at 2× input instead of 1.25×).
  `prefix_until_index=N` puts a second marker on message `N`, so a
  stable conversation history caches along with the system prompt.
  Usage reports both `cache_read_input_tokens` and
  `cache_creation_input_tokens`, mapped to `Usage.cache_read_tokens` /
  `cache_write_tokens`.
- **OpenAI** — implicit. Caching happens server-side on any prompt over
  1024 tokens whether you send `CacheConfig` or not. lm15 forwards
  `key` as `prompt_cache_key` and `retention="long"` as
  `prompt_cache_retention: "24h"`. Reads surface as
  `input_tokens_details.cached_tokens`; writes are never reported.
- **Gemini** — a separate resource. The adapter hashes the prefix
  (everything but the last message, or up to `prefix_until_index`),
  POSTs it once to `cachedContents`, remembers the returned id in the
  LM instance, and sends `cachedContent: <id>` on subsequent payloads.
  Reads surface as `cachedContentTokenCount`.

Numbers carry the [zeros-vs-absent](01-first-request.md) distinction:
`cache_write_tokens=None` means "not reported", `0` means "reported
zero". `InferencePricing.estimate()` skips `None` dimensions — its
result is a lower bound when counts are unknown.

The cache itself is provider-side state. lm15 holds no cache, sets no
TTLs of its own, and does not retry on cache misses — a miss is not an
error, it is full price.

## Variations

- **Gemini, live.** Gemini needs at least two messages before the
  adapter has a prefix to split off, and the cachedContents entry is
  created *before* the first generate call — so even call one reports a
  read. The Gemini LM is built lazily and cached by the router
  (see [using the router](../using-the-router.md)), so the remembered
  cache id survives across `router.complete()` calls:

  ```python
  greq = Request(
      model="gemini-3-flash-preview",
      system=f"You are a support agent. Follow this policy manual.\n{manual}",
      messages=(
          Message.user("Read the manual and stand by."),
          Message.assistant("Standing by."),
          Message.user("Which approval level for a $90 refund? One sentence."),
      ),
      config=Config(max_tokens=200, cache=CacheConfig()),
  )
  for call in ("first", "second"):
      usage = router.complete(greq).usage
      print(f"{call}: read={usage.cache_read_tokens} input={usage.input_tokens}")
  ```
  ```output
  first: read=5087 input=5101
  second: read=5087 input=5101
  ```

- **Opting out.** `CacheConfig(mode="off")` suppresses all cache hints.
  On Anthropic and Gemini that disables caching; on OpenAI the implicit
  server-side cache still applies — you cannot turn it off from the
  client.
- **Keep the prefix stable.** Caches match byte-identical prefixes. A
  timestamp in the system prompt, reordered tools, or a changed
  temperature (on OpenAI, any payload difference before the suffix)
  breaks the match. Put volatile content in the last user message.
- **When it pays.** Anthropic: two calls within the TTL break even;
  prompts under the 1024-token floor never cache. OpenAI: always —
  reads are discounted and writes are free. Gemini: cached tokens are
  discounted but storage bills per token-hour, so short-lived caches on
  rarely-reused prefixes can cost more than they save.
- **Async mirror.** `AsyncLMRouter` behaves identically —
  `await router.complete(req)` with the same `CacheConfig`; the Gemini
  cachedContents POST goes through the async transport.

## See also

- [01 — Your first request](01-first-request.md) — `Usage` fields and the zeros-vs-absent rule
- [02 — Conversations](02-conversations.md) — stable history prefixes worth caching
- [03 — System prompts](03-system-prompts.md) — the prefix you cache most often
- [06 — Function tools](06-function-tools.md) — tool schemas count toward the cached prefix
- [Using the router](../using-the-router.md) — per-provider LM reuse, which keeps Gemini cache ids alive
