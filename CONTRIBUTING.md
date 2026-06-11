# Contributing to lm15-python

This document covers the maintainer-facing machinery: the fixture/conformance
system, the doc-drift checker, the provider adapter development guide, and the
day-to-day commands.

## Repository layout

```text
lm15-python/
├── lm15/
│   ├── types.py              # canonical dataclasses: Request, Response, Parts, tools, endpoints
│   ├── providers/            # OpenAI, Anthropic, Gemini adapters (+ compat, subscription, async)
│   ├── compat.py             # typed API-dialect compatibility policies
│   ├── models.py             # optional ModelInfo metadata + ModelRegistry
│   ├── profiles.py           # ProviderProfile/EndpointProfile resolution helpers
│   ├── result.py             # stream materialization + lazy Result helper
│   ├── serde.py              # canonical JSON dictionaries
│   ├── errors.py             # normalized lm15 error hierarchy
│   ├── live.py               # websocket/live-session helpers
│   ├── sse.py                # server-sent event parser
│   └── transports/           # stdlib HTTP/1.1 sync + async transports
├── conformance/              # fixture suite and reports
├── tests/                    # unit + conformance tests wired into pytest
├── benchmarks/               # transport benchmarks
└── pyproject.toml
```

## The contract pin (CONTRACT_PIN)

CI runs the language-neutral lm15-contract harness against this repo at the
exact contract commit recorded in `CONTRACT_PIN` (a full SHA of
`lm15-dev/lm15-contract` main). Discipline: when a change here requires a
contract change (new fixture, spec row, serde kind, protocol op), land the
contract commit first — with the evidence AUTHORITY.md demands — then bump
`CONTRACT_PIN` to that SHA in the same lm15-python commit as the code
change, so both repos stay green at every pinned pair of revisions. Never
bump the pin just to make CI green: the diff between the old and new pin is
part of your review.

## How the fixture/conformance system works

The conformance suite checks that lm15's canonical model stays aligned with real
provider APIs.

```text
logical lm15 case
  conformance/cross_sdk/test_cases.json
        │
        ▼
lm15-python provider adapter builds HTTP request
        │
        ▼
expected provider fixture
  conformance/provider_requests/cases/<provider>/<feature>.json
        │
        ├── check_request_fixtures.py compares request shape
        ├── validate_live.py can send the request to the real API
        ├── check_response_fixtures.py parses saved response bodies/SSE
        ├── check_error_fixtures.py normalizes provider error bodies
        ├── check_endpoint_fixtures.py checks embeddings/files/batch/image/audio/live
        ├── check_serde_fixtures.py checks canonical JSON round trips
        └── check_doc_drift.py checks provider docs against features.yaml
```

Run everything:

```bash
python3 conformance/run_all.py --strict
```

Run one check:

```bash
python3 conformance/check_doc_drift.py --strict
python3 conformance/check_response_fixtures.py --strict
```

Run or preview one live provider fixture, if the relevant API key is set:

```bash
python3 conformance/provider_requests/validate_live.py --dry-run --task openai.basic_text
python3 conformance/provider_requests/validate_live.py --task openai.basic_text
```

Generated reports are written under `conformance/reports/` and are ignored by
git.

### Adding or completing a fixture

1. Add or update the logical case in
   `conformance/cross_sdk/test_cases.json`.
2. Add the expected provider HTTP request in
   `conformance/provider_requests/cases/<provider>/<feature>.json`.
3. Add the feature to `conformance/provider_requests/features.yaml` so doc drift
   can tell whether provider documentation is represented.
4. If response parsing should be checked, add an `expect_lm15` block to the
   provider case and save a real response body under
   `conformance/provider_requests/results/bodies/<provider>.<feature>/`.
5. If the provider has a special error shape, add an error case under
   `conformance/errors/cases/<provider>.json`.
6. Run:

   ```bash
   python3 conformance/run_all.py --strict
   pytest -q
   ```

Example `expect_lm15` block:

```json
{
  "expect_lm15": {
    "parts": {
      "text": {"min": 1},
      "citation": {"min": 1}
    },
    "finish_reason": "stop",
    "usage": {"required": true}
  }
}
```

### Doc-drift fixture check

`conformance/check_doc_drift.py` parses snapshotted provider docs in
`conformance/provider_docs/` and compares top-level request parameters with
`conformance/provider_requests/features.yaml`.

Some always-on lm15 request fields do not need separate feature entries:

```python
IGNORE_PARAMS = {"model", "messages", "contents", "input"}
```

Provider docs often use camelCase/PascalCase while lm15 feature names use
snake_case, so the drift check normalizes names before deciding that a param is
unmapped.

If `check_doc_drift.py --strict` reports an unmapped param, either:

- add a real feature entry to `features.yaml`, or
- add the param to `IGNORE_PARAMS` only if it is a core field that should never
  have a separate fixture.

## Provider adapter development guide

Provider classes live in `lm15/providers/` and inherit `BaseProviderLM`.
A provider adapter is responsible for:

- `build_request(request, stream)` — map canonical `Request` to an HTTP request.
- `parse_response(request, response)` — map provider JSON to canonical
  `Response`.
- `parse_stream_events(...)` — map SSE chunks to `StreamEvent`s (the single
  stream-parse path per provider).
- `normalize_error(status, body)` — map provider errors to `lm15.errors`.
- Optional endpoint methods: `embeddings`, `file_upload`, `batch_submit`,
  `image_generate`, `audio_generate`, `live`.

Keep provider-only options in `Config.extensions` rather than adding universal
fields unless the same concept is supported across providers.

## Cross-language conformance: lm15-contract

The sibling `lm15-contract` repository is where cross-language conformance
lives — it is the authority, and this package is one implementation of it.
Start with its `AUTHORITY.md` (what is normative and what wins on conflict),
`spec/` (the canonical rules: serde, mapping, errors), and `harness/` (the
runner that executes the shared corpus of cases, goldens, bodies, and error
fixtures against each language implementation). Behavior changes here must be
reflected there first, then re-verified through the harness.

## Useful commands

```bash
# Unit and conformance tests through pytest
pytest -q

# Full offline conformance suite
python3 conformance/run_all.py --strict

# Request fixture comparison only
python3 conformance/check_request_fixtures.py --strict

# Response/SSE fixture parser check only
python3 conformance/check_response_fixtures.py --strict

# Canonical JSON round trips only
python3 conformance/check_serde_fixtures.py --strict

# Provider-doc coverage only
python3 conformance/check_doc_drift.py --strict
```
