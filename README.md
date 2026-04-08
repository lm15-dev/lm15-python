# lm15 (Universal LM) — thin core, full plugin contract

[![PyPI version](https://img.shields.io/pypi/v/lm15.svg)](https://pypi.org/project/lm15/)
[![Python versions](https://img.shields.io/pypi/pyversions/lm15.svg)](https://pypi.org/project/lm15/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A universal LM core optimized for low import/runtime overhead with provider plugins.

## Core architecture

- **Universal contract** (`types.py`): normalized request/response/stream/live types.
- **Provider plugin contract** (`providers/base.py`): complete/stream/live/embeddings/files/batch/images/audio methods.
- **Transport boundary** (`transports/*`): urllib and pycurl implementations.
- **Capability resolver** (`capabilities.py`): static + optional hydration from models.dev.
- **Model catalog bridge** (`model_catalog.py`): loads `https://models.dev/api.json`.
- **Middleware pipeline** (`middleware.py`): retries/history/cache wrappers.
- **Completeness harness** (`completeness/*`): fixture + live probes, score output.

## Support matrix (current adapters)

| Provider | complete | stream | embeddings | files | batches | images | audio | live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Anthropic | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Gemini | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

## Quick start

```python
from lm15 import Message, LMRequest, Part, build_default

lm = build_default(use_pycurl=True)
req = LMRequest(model="claude-sonnet-4-5", messages=(Message(role="user", parts=(Part.text_part("Say hi."),)),))
resp = lm.complete(req)
print(resp.message.parts[0].text)
```

## External plugin adapters (no core PR)

LM15 auto-discovers installed entry-points in group `lm15.providers`.

```toml
# in external package pyproject.toml
[project.entry-points."lm15.providers"]
myprovider = "lm15_x_myprovider:build_adapter"
```

```python
from lm15 import build_default

lm = build_default(discover_plugins=True)
```

## Optional models.dev hydration

```python
lm = build_default(hydrate_models_dev_catalog=True)
```

## Completeness

```bash
python3 completeness/runner.py --mode fixture --fail-under 1.0
python3 completeness/runner.py --mode live --fail-under 0.0
```

Outputs:
- `completeness/report.json`
- `completeness/report.md`

## Packaging and publishing (uv + twine)

```bash
# build sdist + wheel
uv run python -m build

# upload to TestPyPI
# twine upload --repository testpypi dist/*

# upload to PyPI
# twine upload dist/*
```

## Known limitations

- Realtime/live protocol parity is not fully implemented across all providers.
- Completeness score is tied to the current fixture matrix; it is not a guarantee of full vendor API parity.
- Provider-specific advanced options are partially passthrough (`config.provider`) and not all are normalized.

## Environment variables

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Docs

### Core

- `docs/GETTING_STARTED.md`
- `docs/CONCEPTS.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTRACT.md`
- `docs/ERRORS.md`
- `docs/STREAMING.md`
- `docs/COMPLETENESS.md`
- `docs/PRODUCTION_CHECKLIST.md`

### Provider development

- `docs/ADAPTER_GUIDE.md`
- `docs/ADD_PROVIDER_GUIDE.md`
- `docs/COOKBOOK_TEMPLATE.md`

### Cookbooks (learning order)

- `docs/COOKBOOKS/01-basic-text.md`
- `docs/COOKBOOKS/02-streaming.md`
- `docs/COOKBOOKS/03-tools.md`
- `docs/COOKBOOKS/04-multimodal.md`
- `docs/COOKBOOKS/05-files-batches.md`
- `docs/COOKBOOKS/06-reliability.md`
- `docs/COOKBOOKS/07-external-plugins.md`
- `docs/COOKBOOKS/08-models-dev-hydration.md`
