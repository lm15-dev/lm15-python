# Gemini API Reference (local cache)

Local cache of the Google Gemini REST API reference docs from https://ai.google.dev/api.
Use when working on the Gemini provider adapter, debugging Gemini errors, or checking request/response shapes.

Scraped from native `.md.txt` endpoints on 2026-04-09. ~16k lines, 640K total.

## Pages

Files live in `pages/`. Read whichever page is relevant to the task at hand.

### API Reference (REST endpoints)
| File | Source |
|------|--------|
| `generate-content.md` | https://ai.google.dev/api/generate-content |
| `embeddings.md` | https://ai.google.dev/api/embeddings |
| `models.md` | https://ai.google.dev/api/models |
| `tokens.md` | https://ai.google.dev/api/tokens |
| `caching.md` | https://ai.google.dev/api/caching |
| `files.md` | https://ai.google.dev/api/files |
| `live.md` | https://ai.google.dev/api/live |
| `batch-mode.md` | https://ai.google.dev/api/batch-mode |
| `interactions-api.md` | https://ai.google.dev/api/interactions-api |

### Guides
| File | Source |
|------|--------|
| `troubleshooting.md` | https://ai.google.dev/gemini-api/docs/troubleshooting |
| `api-versions.md` | https://ai.google.dev/gemini-api/docs/api-versions |
| `models-gemini.md` | https://ai.google.dev/gemini-api/docs/models/gemini |
| `rate-limits.md` | https://ai.google.dev/gemini-api/docs/rate-limits |
| `tokens-guide.md` | https://ai.google.dev/gemini-api/docs/tokens |

## Updating

Run the update script:
```bash
bash .pi/skills/gemini-api-reference/update.sh
```
Google dev pages expose native markdown at `<url>.md.txt`.
