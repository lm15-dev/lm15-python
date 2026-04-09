# OpenAI API Reference (local cache)

Local cache of the OpenAI API reference docs from https://developers.openai.com.
Use when working on the OpenAI provider adapter, debugging OpenAI errors, or checking request/response shapes.

Scraped on 2026-04-09. ~50k lines, 1.6MB total.

**Note:** OpenAI uses two `.md` endpoint patterns:
- Overview/guide pages: `<url>.md`
- Stainless-generated method pages: `<url>/index.md`
- Streaming events pages: JS-rendered only (scraped via Jina fallback)

## Pages

Files live in `pages/`. Read whichever page is relevant to the task at hand.

### Overview
| File | Source |
|------|--------|
| `overview.md` | https://developers.openai.com/api/reference/overview |
| `responses-overview.md` | https://developers.openai.com/api/reference/responses/overview |
| `chat-completions-overview.md` | https://developers.openai.com/api/reference/chat-completions/overview |

### Responses API (primary — used by adapter)
| File | Source |
|------|--------|
| `responses--create.md` | .../resources/responses/methods/create |
| `responses--retrieve.md` | .../resources/responses/methods/retrieve |
| `responses--cancel.md` | .../resources/responses/methods/cancel |
| `responses--delete.md` | .../resources/responses/methods/delete |
| `responses--input-items.md` | .../resources/responses/subresources/input_items/methods/list |
| `responses--count-tokens.md` | .../resources/responses/subresources/input_tokens/methods/count |

### Chat Completions (legacy)
| File | Source |
|------|--------|
| `chat--create.md` | .../resources/chat/subresources/completions/methods/create |
| `chat--streaming.md` | .../resources/chat/subresources/completions/streaming-events |

### Embeddings
| File | Source |
|------|--------|
| `embeddings--create.md` | .../resources/embeddings/methods/create |

### Models
| File | Source |
|------|--------|
| `models--list.md` | .../resources/models/methods/list |
| `models--retrieve.md` | .../resources/models/methods/retrieve |

### Files
| File | Source |
|------|--------|
| `files--create.md` | .../resources/files/methods/create |
| `files--list.md` | .../resources/files/methods/list |
| `files--retrieve.md` | .../resources/files/methods/retrieve |
| `files--delete.md` | .../resources/files/methods/delete |
| `files--content.md` | .../resources/files/methods/content |

### Batches
| File | Source |
|------|--------|
| `batches--create.md` | .../resources/batches/methods/create |
| `batches--list.md` | .../resources/batches/methods/list |
| `batches--retrieve.md` | .../resources/batches/methods/retrieve |
| `batches--cancel.md` | .../resources/batches/methods/cancel |

### Images
| File | Source |
|------|--------|
| `images--generate.md` | .../resources/images/methods/generate |
| `images--edit.md` | .../resources/images/methods/edit |

### Audio
| File | Source |
|------|--------|
| `audio--speech.md` | .../resources/audio/subresources/speech/methods/create |
| `audio--transcription.md` | .../resources/audio/subresources/transcriptions/methods/create |

### Realtime
| File | Source |
|------|--------|
| `realtime--sessions.md` | .../resources/realtime/subresources/sessions/methods/create |

### Guides
| File | Source |
|------|--------|
| `guide--error-codes.md` | https://developers.openai.com/docs/guides/error-codes |
| `guide--rate-limits.md` | https://developers.openai.com/docs/guides/rate-limits |
| `guide--latency.md` | https://developers.openai.com/docs/guides/latency-optimization |
| `guide--production.md` | https://developers.openai.com/docs/guides/production-best-practices |
| `guide--reasoning.md` | https://developers.openai.com/docs/guides/reasoning |
| `guide--streaming.md` | https://developers.openai.com/docs/guides/streaming-responses |
| `guide--function-calling.md` | https://developers.openai.com/docs/guides/function-calling |
| `guide--structured-output.md` | https://developers.openai.com/docs/guides/structured-outputs |
| `guide--text.md` | https://developers.openai.com/docs/guides/text |
| `guide--audio.md` | https://developers.openai.com/docs/guides/audio |
| `guide--embeddings.md` | https://developers.openai.com/docs/guides/embeddings |

## Updating

Run the update script:
```bash
bash .pi/skills/openai-api-reference/update.sh
```
