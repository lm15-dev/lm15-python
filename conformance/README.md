# lm15-python conformance

This directory is the active foundation for lm15 compatibility work.

> **Corpus home: the fixture corpus now lives in `lm15-contract/`** (repo root,
> alongside this repository), governed by its `AUTHORITY.md`. The copies under
> this directory keep the old suite running until the harness cutover; until
> then, fixture edits land in BOTH places or not at all. See
> `lm15-contract/changes/2026-06-09-initial-migration.md`.

For now, **lm15-python is the reference implementation**. Other language ports
are frozen until this conformance suite is stable. Future ports should be built
against the `lm15-contract` fixtures and only considered compatible when they
pass the same checks.

## Layout

```text
conformance/
├── README.md
├── run_all.py                    # runs every check and writes summary.json
├── check_request_fixtures.py     # logical case → provider HTTP request
├── check_response_fixtures.py    # provider response/SSE → canonical Response
├── check_error_fixtures.py       # provider error body → typed lm15 error
├── check_endpoint_fixtures.py    # embeddings/files/batch/image/audio/live
├── check_serde_fixtures.py       # canonical JSON round-trips
├── check_doc_drift.py            # unmapped doc params vs feature inventory
├── cross_sdk/
│   ├── test_cases.json           # canonical logical lm15 cases
│   └── dump_request.py           # lm15-python logical case → HTTP request
├── provider_requests/
│   ├── cases/                    # live-tested provider curl fixtures
│   ├── features.yaml             # feature inventory + lm15/provider scope
│   ├── results/                  # saved live-test summaries and bodies
│   └── validate_live.py          # re-run fixtures against the live API
├── errors/cases/                 # provider error body → expected lm15 error
├── serde/canonical.json          # canonical JSON serde fixtures
├── provider_docs/                # snapshot of upstream API references
└── reports/                      # generated local reports, ignored by git
```

## Run everything

```bash
python3 conformance/run_all.py --strict
```

Individual checks:

```bash
python3 conformance/check_request_fixtures.py --strict
python3 conformance/check_response_fixtures.py --strict
python3 conformance/check_error_fixtures.py --strict
python3 conformance/check_endpoint_fixtures.py --strict
python3 conformance/check_serde_fixtures.py --strict
python3 conformance/check_doc_drift.py --strict
```

Each script writes both JSON and Markdown reports under
`conformance/reports/`.

## What each check covers

- **request_fixtures**
  Compares the provider HTTP request built by lm15-python against the
  live-tested provider curl fixture for every logical case.

- **response_fixtures**
  Replays saved provider response bodies and SSE streams through the lm15
  parsers and checks them against per-fixture `expect_lm15` assertions.

- **error_fixtures**
  Feeds curated provider error bodies into each provider's `normalize_error`
  and asserts the resulting structured lm15 error class, code, provider, and
  metadata match.

- **endpoint_fixtures**
  Offline conformance for non-chat endpoints: embeddings, file upload, batch,
  image generation, audio generation, plus live URL/header and session shape.

- **serde_fixtures**
  Round-trips canonical JSON fixtures for every public lm15 type represented in
  the portable interchange contract.

- **doc_drift**
  Audits the snapshot of provider documentation under `provider_docs/` and
  flags any documented top-level request parameters that have no corresponding
  entry in `provider_requests/features.yaml`.

## Future ports

Future Go/Rust/TypeScript/Julia ports should implement a small dump command:

```bash
<port-dump-command> '<logical-case-json>'
```

It must emit normalized JSON of this shape:

```json
{
  "method": "POST",
  "url": "https://.../path",
  "params": {},
  "headers": {"content-type": "application/json"},
  "body": {}
}
```

The conformance runner can then compare each port against the same provider
fixtures and against lm15-python.

## Canonical JSON

The canonical JSON fixtures in `serde/canonical.json` are the language-neutral
interchange format. Future ports should round-trip these fixtures exactly and
use the provider request/response/error fixtures as the behavioral oracle.
