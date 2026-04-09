# Anthropic API Reference (local cache)

Local cache of the Anthropic Claude API reference docs from https://platform.claude.com/docs/en/api.
Use when working on the Anthropic provider adapter, debugging Anthropic errors, or checking request/response shapes.

Scraped from native `.md` endpoints on 2026-04-09. 26 pages, ~1MB total.

## Pages

Files live in `pages/`. Read whichever page is relevant to the task at hand.

### Core
| File | Source |
|------|--------|
| `overview.md` | https://platform.claude.com/docs/en/api/overview |
| `errors.md` | https://platform.claude.com/docs/en/api/errors |
| `rate-limits.md` | https://platform.claude.com/docs/en/api/rate-limits |
| `versioning.md` | https://platform.claude.com/docs/en/api/versioning |
| `beta-headers.md` | https://platform.claude.com/docs/en/api/beta-headers |
| `service-tiers.md` | https://platform.claude.com/docs/en/api/service-tiers |

### Messages
| File | Source |
|------|--------|
| `messages--create.md` | https://platform.claude.com/docs/en/api/messages/create |
| `messages--streaming.md` | https://platform.claude.com/docs/en/api/messages/streaming |
| `messages-count-tokens.md` | https://platform.claude.com/docs/en/api/messages-count-tokens |

### Message Batches
| File | Source |
|------|--------|
| `creating-message-batches.md` | https://platform.claude.com/docs/en/api/creating-message-batches |
| `retrieving-message-batches.md` | https://platform.claude.com/docs/en/api/retrieving-message-batches |
| `listing-message-batches.md` | https://platform.claude.com/docs/en/api/listing-message-batches |
| `canceling-message-batches.md` | https://platform.claude.com/docs/en/api/canceling-message-batches |
| `message-batch-results.md` | https://platform.claude.com/docs/en/api/message-batch-results |
| `deleting-message-batches.md` | https://platform.claude.com/docs/en/api/deleting-message-batches |

### Models
| File | Source |
|------|--------|
| `models-list.md` | https://platform.claude.com/docs/en/api/models-list |
| `models-get.md` | https://platform.claude.com/docs/en/api/models-get |

### Files
| File | Source |
|------|--------|
| `files-create.md` | https://platform.claude.com/docs/en/api/files-create |
| `files-list.md` | https://platform.claude.com/docs/en/api/files-list |
| `files-get.md` | https://platform.claude.com/docs/en/api/files-get |
| `files-delete.md` | https://platform.claude.com/docs/en/api/files-delete |
| `files-get-content.md` | https://platform.claude.com/docs/en/api/files-get-content |

### Other
| File | Source |
|------|--------|
| `client-sdks.md` | https://platform.claude.com/docs/en/api/client-sdks |
| `supported-regions.md` | https://platform.claude.com/docs/en/api/supported-regions |
| `ip-addresses.md` | https://platform.claude.com/docs/en/api/ip-addresses |
| `getting-help.md` | https://platform.claude.com/docs/en/api/getting-help |

## Updating

Run the update script:
```bash
bash .pi/skills/anthropic-api-reference/update.sh
```
Anthropic pages expose native markdown at `<url>.md`.
