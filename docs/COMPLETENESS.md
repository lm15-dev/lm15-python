# Completeness Harness

The fixture probes are backed by the frozen portability bundle in `spec/fixtures/v2/`.
That bundle now covers completion normalization, error mapping, and live/session contract fixtures.
See `docs/PORTABILITY.md` for the contract + fixture layout.

## Matrices

- Fixture matrix: `completeness/spec_matrix.json`
- Live matrix: `completeness/live_matrix.json`

## Run fixture completeness

```bash
python3 completeness/runner.py --mode fixture --fail-under 1.0
```

## Run live completeness (requires provider keys)

```bash
python3 completeness/runner.py --mode live --fail-under 0.0
```

## Run all

```bash
python3 completeness/runner.py --mode all --fail-under 1.0
```

## Outputs

- JSON report: `completeness/report.json`
- Markdown report: `completeness/report.md`
- Optional history snapshot:

```bash
python3 completeness/save_history.py
```
