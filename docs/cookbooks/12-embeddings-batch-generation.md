# Embeddings, batch & media generation

**Problem** — Not everything is chat. You need vectors for retrieval,
half-price overnight batches, a generated image, or spoken audio — and
each provider hides these behind a different endpoint. lm15 gives them
one request type each: `EmbeddingRequest`, `BatchRequest`,
`ImageGenerationRequest`, `AudioGenerationRequest`.

**These surfaces are provisional.** The chat core is frozen by the
cross-language contract; the non-chat endpoints work and are
live-tested, but their shapes may still change before 1.0
([roadmap](../roadmap.md)). Pin your lm15 version if you build on them.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

The router routes chat; the non-chat methods live on the provider LM.
`router.lm()` is the bridge: resolve once, get the configured LM, call
its endpoint methods directly.

### Embeddings

```python
import base64
import math
from pathlib import Path

from lm15 import (
    AudioGenerationRequest, BatchRequest, EmbeddingRequest,
    ImageGenerationRequest, LMRouter, Message, Request,
)

router = LMRouter()
openai = router.lm("openai:text-embedding-3-small")

emb = openai.embeddings(EmbeddingRequest(
    model="text-embedding-3-small",
    inputs=(
        "the cat sat on the mat",
        "a feline rested on the rug",
        "quarterly revenue grew 12%",
    ),
))
print(emb.model, len(emb.vectors), "vectors of dim", len(emb.vectors[0]))
print(emb.usage)
```
```output
text-embedding-3-small 3 vectors of dim 1536
Usage(input_tokens=20, output_tokens=0, total_tokens=20, …)
```

`vectors` is a tuple of float tuples, validated finite — feed it to
whatever store you use. lm15 does not compute similarity; that is four
lines of stdlib:

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

v = emb.vectors
print(f"cat~feline  {cosine(v[0], v[1]):.3f}")
print(f"cat~revenue {cosine(v[0], v[2]):.3f}")
```
```output
cat~feline  0.647
cat~revenue 0.096
```

### Batch

A `BatchRequest` nests ordinary `Request` objects, each carrying its
own model. Anthropic's Message Batches API is a true server-side queue
— half price, results within 24 hours:

```python
anthropic = router.lm("claude-sonnet-4-5")
sub = anthropic.batch_submit(BatchRequest(requests=(
    Request(model="claude-sonnet-4-5",
            messages=(Message.user("Capital of France? One word."),)),
    Request(model="claude-sonnet-4-5",
            messages=(Message.user("Capital of Japan? One word."),)),
)))
print(sub)
print(sub.provider_data["request_counts"])
```
```output
BatchResponse(id='msgbatch_01YV…', status='running', provider_data={…})
{'processing': 2, 'succeeded': 0, 'errored': 0, 'canceled': 0, 'expired': 0}
```

lm15 submits; it does not poll. `batch_submit` returns the provider's
batch id and a normalized status (`submitted`/`queued`/`running`/
`completed`/`failed`/`cancelled`). Polling and result retrieval are
yours — the raw batch object in `provider_data` has everything the
provider returned, including the URLs to fetch results from.

### Image generation

```python
gen = router.lm("gpt-4.1-mini")
img = gen.image_generate(ImageGenerationRequest(
    model="gpt-image-1",
    prompt="a minimal line drawing of a fox reading a book",
    size="1024x1024",
    extensions={"quality": "low"},
))
part = img.images[0]
png = base64.b64decode(part.data)
Path("/tmp/fox.png").write_bytes(png)
print(part.media_type, len(png), "bytes")
```
```output
image/png 1156899 bytes
```

`images` is a tuple of the same `ImagePart` you send *to* models in
recipe [09](09-images-and-documents.md) — base64 in `data`, or a
`url` when the provider returns one. `extensions` passes
provider-specific knobs (`quality` here) straight through.

### Audio generation

```python
aud = gen.audio_generate(AudioGenerationRequest(
    model="gpt-4o-mini-tts",
    prompt="lm fifteen: one request type, every provider.",
    voice="alloy",
    format="wav",
))
clip = base64.b64decode(aud.audio.data)
Path("/tmp/lm15.audio").write_bytes(clip)
print(aud.audio.media_type, len(clip), "bytes")
```
```output
audio/mpeg 60288 bytes
```

We asked for `wav`; OpenAI returned `audio/mpeg`. The provider
decides; lm15 reports the actual content-type on the `AudioPart`
rather than echoing your request.
Trust `media_type`, not `format`, when naming the file.

## How it works

Each endpoint is a frozen request/response dataclass pair plus one
method on the LM (`embeddings`, `batch_submit`, `image_generate`,
`audio_generate`). The base LM raises `UnsupportedFeatureError` for
endpoints a provider lacks — Anthropic has no embeddings API, so:

```python
from lm15 import UnsupportedFeatureError
try:
    anthropic.embeddings(EmbeddingRequest(model="x", inputs=("hi",)))
except UnsupportedFeatureError as e:
    print(e)
```
```output
anthropic: embeddings not supported
```

(`try/except` here because the error *is* the lesson.) On the wire,
each method maps to the provider's native endpoint: OpenAI
`/embeddings`, `/images/generations`, `/audio/speech`, `/batches`;
Anthropic `/messages/batches`; Gemini `:embedContent` and
`:batchEmbedContents`. `extensions` on every request is the
passthrough valve for fields lm15 does not model (recipe
[16](16-provider-passthrough.md)).

The router stays a chat front door on purpose: `complete()` and
`stream()` only. `router.lm(model)` hands you the resolved, keyed
provider LM, and the provisional methods live there — when these
shapes freeze, the router may grow matching verbs.

## Variations

- **Gemini embeddings** — same request, different model string; one
  input uses `:embedContent`, several use `:batchEmbedContents`:

  ```python
  gem = router.lm("gemini-embedding-001")
  r = gem.embeddings(EmbeddingRequest(
      model="gemini-embedding-001", inputs=("the cat sat on the mat",)))
  print(len(r.vectors), len(r.vectors[0]))
  ```
  ```output
  1 3072
  ```

- **OpenAI batch is two-phase.** Without extensions, lm15 falls back
  to a local sequential fan-out — it calls `complete()` per nested
  request, synchronously, full price, and returns
  `status='completed'` with results inline in `provider_data`. The
  real Batch API needs an uploaded JSONL file first:
  `extensions={"input_file_id": "file-…"}` (upload via
  `FileUploadRequest`). Gemini's `batch_submit` is the same local
  fan-out for now.
- **Gemini image/audio generation** routes through chat under the
  hood: `image_generate` issues a `complete()` with
  `responseModalities: ["IMAGE"]` and collects `ImagePart`s from the
  reply. Pick a model that supports the modality.
- **Async mirror**: the LMs from `AsyncLMRouter().lm(...)` expose
  awaitable `embeddings` / `batch_submit` / `image_generate` /
  `audio_generate` with identical types.
- **Cost**: `gpt-image-1` bills per image even at `quality: "low"`;
  batch APIs are the discount path, local fan-out is not.

## See also

- [09 — Images, PDFs & documents](09-images-and-documents.md) — the
  `ImagePart`/`AudioPart` you get back here, used as input
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md)
- [13 — Live sessions (realtime)](13-live-sessions.md) — the other
  provisional surface
- [16 — Provider passthrough](16-provider-passthrough.md) — the
  `extensions` valve these endpoints lean on
- [Using the router](../using-the-router.md) — `router.lm()` and
  resolution
