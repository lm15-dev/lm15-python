## The Zero-Dependency Position

lm15 depends on nothing. Not "a few small packages." Nothing. The HTTP transport is `urllib.request`. The data types are `dataclasses`. The JSON handling is `json`. The SSE parser is 50 lines of custom code. The entire library is 408 kilobytes on disk.

This is an extreme position. Most Python developers would call it unreasonable. The previous section's bakery argument applies directly: `requests` is better at HTTP than `urllib`. `pydantic` catches more bugs than `__post_init__`. `httpx` offers async that `urllib` never will. By choosing zero dependencies, lm15 chose to be worse at HTTP, worse at validation, and incapable of async. The question is whether what it gained is worth what it gave up.

The answer depends on a single observation: **lm15's needs are narrower than what the dependencies provide.**

Consider HTTP. `requests` handles connection pooling, HTTP/2, redirect chains, cookie management, proxy tunneling, chunked uploads, streaming downloads, digest authentication, client certificates, multipart form encoding, and dozens of other HTTP features. lm15 needs one: POST a JSON body to a URL with an `Authorization` header and receive a JSON body back. That's it. No redirects (LLM APIs don't redirect). No cookies (LLM APIs don't use cookies). No proxies (optional, and handled by pycurl when needed). No uploads (file upload is a JSON body with base64 data, not a multipart form). One HTTP operation — POST with JSON — is the entirety of lm15's HTTP needs.

`urllib.request.urlopen` handles this perfectly. The code is six lines:

```python
req = urllib.request.Request(url=url, data=body, method="POST", headers=headers)
with urllib.request.urlopen(req, timeout=timeout) as r:
    return HttpResponse(status=r.status, headers=dict(r.headers), body=r.read())
```

This is worse than `requests` in every way except the ones that matter to lm15: it has zero dependencies, imports in microseconds, and handles POST-with-JSON without flaw. The 95% of HTTP that `requests` provides and lm15 doesn't need is the 95% that would cost 200ms of import time and six transitive packages.

Consider validation. `pydantic` validates arbitrarily nested data structures against complex schemas with custom types, coercion rules, and detailed error messages. lm15 needs to check that a text part has text, that an image part has a source, that a tool call has an ID and a name. Five checks, each one line:

```python
def __post_init__(self):
    if self.type in {"text", "thinking"} and self.text is None:
        raise ValueError(f"Part(type='{self.type}') requires text")
    if self.type in {"image", "audio", "video", "document"} and self.source is None:
        raise ValueError(f"Part(type='{self.type}') requires source")
```

This misses things `pydantic` would catch — a `DataSource` with an invalid URL, a `Config` with contradictory parameters, a `Usage` with negative token counts. These are real gaps. But lm15's types are constructed by trusted code — the `Model` class, the adapters, the factory — not parsed from untrusted user input. The validation needs are different from a web framework that deserializes arbitrary JSON from the internet. `__post_init__` is inadequate for the web framework. It's adequate for lm15.

Consider SSE parsing. There is no standard, well-maintained, widely-used SSE parser for Python. The candidates are either abandoned, overengineered for a browser context, or part of a larger framework. lm15's parser is 50 lines in `sse.py`. It handles the SSE specification: `data:` lines, `event:` lines, comment lines, empty-line delimiters. It has safety limits for line length and event size. It has never needed a bug fix because the SSE specification is simple and the LLM providers' usage of it is simpler. Fifty lines of obvious code is better than a dependency for a problem this small.

Consider JSON. `json.dumps` and `json.loads` are part of the standard library. There is no argument for depending on a JSON library. This isn't even a decision — it's the absence of one.

The pattern across all four: **the dependency's capability exceeds the need.** The need is narrow — POST JSON, check five fields, parse SSE lines, encode JSON. The dependency's capability is broad — full HTTP, full validation, full async. The gap between the need and the capability is the dependency's cost: the import time, the disk space, the version constraints, the attack surface. When the gap is wide — when you're using 5% of a package's functionality — the cost-to-benefit ratio tips toward reimplementation.

This principle has a limit. When the gap is narrow — when you need 80% of what a package provides — reimplementation is foolish. You'd be rebuilding a mature, tested system to avoid a small dependency. The zero-dependency position is viable *because* lm15's needs are in the 5% zone. A library with broader needs — OAuth flows, WebSocket connections, async batch processing — would be in the 80% zone, and the bakery argument would win.

The honest framing: lm15 didn't choose zero dependencies out of principle. It chose zero dependencies because its needs happened to fall within the standard library's coverage. If `urllib` couldn't POST JSON, lm15 would depend on `requests`. The philosophy followed the pragmatics, not the other way around.
