# Error Handling

## Error hierarchy

```
ULMError
├── TransportError              # Network-level: DNS, connection refused, socket timeout
└── ProviderError               # API-level: the provider returned an error
    ├── AuthError               # 401, 403 — bad key, expired key, wrong permissions
    ├── BillingError            # 402 — payment or billing issue
    ├── RateLimitError          # 429 — too many requests or quota exceeded
    ├── InvalidRequestError     # 400, 404, 409, 413, 422 — bad model, bad params, not found
    │   └── ContextLengthError  # input exceeds the model's context window
    ├── TimeoutError            # 408, 504 — request timed out
    ├── ServerError             # 5xx — provider is down or overloaded
    ├── UnsupportedModelError
    ├── UnsupportedFeatureError
    └── NotConfiguredError
```

`ContextLengthError` extends `InvalidRequestError` — existing `except InvalidRequestError` catches still work.

## How errors flow

```
Provider HTTP response (status >= 400)
  → Adapter.normalize_error(status, body)   # each adapter parses its own error format
    → map_http_error(status, message)        # maps HTTP status → error class
      → AuthError / RateLimitError / etc.
```

Each adapter owns its error parsing via `normalize_error`. The base `map_http_error` only maps status codes to error classes — adapters extract the human-readable message.

Streaming follows the same taxonomy. Provider stream errors are normalized to canonical stream error codes (`auth`, `billing`, `rate_limit`, `invalid_request`, `context_length`, `timeout`, `server`, `provider`) and `lm15.stream.Stream` raises the corresponding typed `ProviderError` subclass.

---

## Provider error mapping

### OpenAI

Source of truth: [platform.openai.com/docs/guides/error-codes](https://platform.openai.com/docs/guides/error-codes)

Error shape: `{"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}`

| HTTP | `error.type` | `error.code` | lm15 error | Detection |
|------|-------------|--------------|------------|-----------|
| 400 | `invalid_request_error` | `context_length_exceeded` | `ContextLengthError` | ✅ structured `code` field |
| 400 | `invalid_request_error` | `model_not_found` | `InvalidRequestError` | |
| 400 | `invalid_request_error` | `missing_required_parameter` | `InvalidRequestError` | |
| 400 | `invalid_request_error` | `invalid_value` | `InvalidRequestError` | |
| 401 | `invalid_request_error` | `invalid_api_key` | `AuthError` | |
| 403 | `invalid_request_error` | *(project access denied)* | `AuthError` | |
| 429 | `rate_limit_error` | `rate_limit_exceeded` | `RateLimitError` | |
| 429 | `insufficient_quota` | `insufficient_quota` | `BillingError` | ✅ structured `code`/`type` field |
| 500 | `server_error` | *(varies)* | `ServerError` | |
| 503 | `server_error` | `service_unavailable` | `ServerError` | |

The Responses API can also return **in-band errors** on HTTP 200 via `Response.error` (e.g. background task failures):

| `Response.error.code` | lm15 error |
|-----------------------|------------|
| `server_error` | `ServerError` |
| `rate_limit_exceeded` | `RateLimitError` |
| `invalid_prompt` | `InvalidRequestError` |
| *(other)* | `ServerError` (fallback) |

### Anthropic

Source of truth: [docs.anthropic.com/en/api/errors](https://docs.anthropic.com/en/api/errors)

Error shape: `{"type": "error", "error": {"type": "...", "message": "..."}, "request_id": "..."}`

| HTTP | `error.type` | lm15 error | Notes |
|------|-------------|------------|-------|
| 400 | `invalid_request_error` | `InvalidRequestError` | |
| 400 | `invalid_request_error` | `ContextLengthError` | ⚠️ message match: `"prompt is too long"`. No structured code exists |
| 401 | `authentication_error` | `AuthError` | |
| 402 | `billing_error` | `BillingError` | |
| 403 | `permission_error` | `AuthError` | |
| 404 | `not_found_error` | `InvalidRequestError` | |
| 413 | `request_too_large` | `InvalidRequestError` | Payload > 32 MB (from Cloudflare, before API) |
| 429 | `rate_limit_error` | `RateLimitError` | |
| 500 | `api_error` | `ServerError` | |
| 504 | `timeout_error` | `TimeoutError` | |
| 529 | `overloaded_error` | `ServerError` | API-wide overload |

### Google Gemini

Source of truth: [ai.google.dev/gemini-api/docs/troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting) and [Google API error model](https://cloud.google.com/apis/design/errors)

Error shape: `{"error": {"message": "...", "code": <int>, "status": "...", "details": [...]}}`

| HTTP | `error.status` | lm15 error | Notes |
|------|---------------|------------|-------|
| 400 | `INVALID_ARGUMENT` | `InvalidRequestError` | |
| 400 | `FAILED_PRECONDITION` | `BillingError` | "Free tier not available in your country. Enable billing." |
| 403 | `PERMISSION_DENIED` | `AuthError` | |
| 404 | `NOT_FOUND` | `InvalidRequestError` | |
| 429 | `RESOURCE_EXHAUSTED` | `RateLimitError` | |
| 429 | `RESOURCE_EXHAUSTED` | `ContextLengthError` | ⚠️ message match: token + limit/exceed keywords. Gemini conflates context overflow with rate limiting under the same 429/RESOURCE_EXHAUSTED |
| 500 | `INTERNAL` | `ServerError` | |
| 500 | `INTERNAL` | `ContextLengthError` | ⚠️ message match: "input context is too long" — Gemini docs list this under 500/INTERNAL |
| 503 | `UNAVAILABLE` | `ServerError` | |
| 504 | `DEADLINE_EXCEEDED` | `TimeoutError` | Prompt/context too large to process in time |

Gemini `generateContent` can also return in-band generation failures on HTTP 200:

| Signal | lm15 error |
|--------|------------|
| `promptFeedback.blockReason` set | `InvalidRequestError` |
| `candidate.finishReason` in safety/recitation/blocklist/prohibited/tool-malformed classes | `InvalidRequestError` |

---

## Context length detection reliability

| Provider | Detection method | Reliability |
|----------|-----------------|-------------|
| OpenAI | `error.code == "context_length_exceeded"` | ✅ Structured, reliable |
| Anthropic | Message contains `"prompt is too long"` | ⚠️ String matching — could break if Anthropic changes wording |
| Gemini | Message contains token + limit/exceed keywords | ⚠️ String matching — Gemini conflates context overflow with `RESOURCE_EXHAUSTED` (429), same as rate limiting |

Only OpenAI provides a machine-readable error code for context overflow. Anthropic and Gemini require message parsing. If detection fails, the error falls through to `InvalidRequestError` (Anthropic) or `RateLimitError` (Gemini), which are still catchable.

---

## Catching errors

```python
import lm15
from lm15.errors import AuthError, BillingError, RateLimitError, InvalidRequestError, ContextLengthError, ServerError

try:
    resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
except ContextLengthError as e:
    print(f"Input too long: {e}")
except AuthError as e:
    print(f"Bad credentials: {e}")
except BillingError as e:
    print(f"Payment issue: {e}")
except RateLimitError as e:
    print(f"Slow down or add credits: {e}")
except InvalidRequestError as e:
    print(f"Fix your request: {e}")
except ServerError as e:
    print(f"Provider is down, retry later: {e}")
```

> **Order matters.** Catch `ContextLengthError` before `InvalidRequestError` — it's a subclass.

### Catch all provider errors

```python
from lm15.errors import ProviderError

try:
    resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
except ProviderError as e:
    print(f"Provider error: {e}")
```

### Catch everything (including network issues)

```python
from lm15.errors import ULMError

try:
    resp = lm15.call("gpt-4.1-mini", "Hello.", env=".env")
except ULMError as e:
    print(f"Something went wrong: {e}")
```

---

## For plugin authors

Override `normalize_error` on your adapter to parse your provider's error format:

```python
import json
from lm15.errors import ProviderError, map_http_error
from lm15.providers.base import BaseProviderAdapter

class MyAdapter(BaseProviderAdapter):
    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            msg = data.get("error_description", body[:200])
        except Exception:
            msg = body[:200] or f"HTTP {status}"
        return map_http_error(status, msg)
```

`map_http_error` handles the status → error class mapping. You only need to extract the human-readable message. For provider-specific error types (like context length), detect them in your `normalize_error` and return the appropriate lm15 error class directly.
