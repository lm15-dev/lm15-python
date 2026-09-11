# Video generation

**Provisional in 1.0:** this API ships, but may change incompatibly in 1.x
with a contract change entry. Pin your package version; see
[release scope](../roadmap.md#what-ships-in-10-and-what-is-stable).

**Problem** — You describe eight seconds of motion and want an MP4.
Video is the one modality that is a **job on every wire that sells
it**: Sora (OpenAI), Veo (Gemini), and grok-imagine (xAI) all take your
prompt, hand you a ticket, and cook for seconds to minutes. lm15
already has a ticket idiom — batch jobs ([recipe 12](12-batch-jobs.md))
— and video reuses it instead of inventing a second waiting pattern.

```text
image_generate(request)  -> ImageGenerationResponse   now
video_generate(request)  -> VideoJob                  later (a ticket)
```

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

### Submit, wait, collect

```python
from lm15 import LMRouter, VideoGenerationRequest

router = LMRouter()
lm = router.lm("grok-imagine-video")  # subscription OAuth or XAI_API_KEY

job = lm.video_generate(VideoGenerationRequest(
    model="grok-imagine-video",
    prompt="A paper boat drifts across a puddle in light rain",
))
print(job.id, job.status)
```
```output
7d3786ad-5da2-9195-8d36-c0270f746cb1 queued
```

```python
job.wait(poll_every=5.0)
print(job.status, job.progress)

video = job.result()
print(video.media_type, video.url)
```
```output
completed 100
video/mp4 https://vidgen.x.ai/xai-vidgen-bucket/xai-video-7d3786ad-….mp4
```

Thirty seconds, roughly. `result()` returns a **`VideoPart`** — the
same part type video inputs use — addressed the way the provider
delivers it. xAI hands out a public URL; `video.bytes` downloads it
when you want the file.

### Bytes-delivering providers

Sora has no public URLs: the finished video streams from a content
endpoint, so `result()` arrives bytes-addressed. Same call, different
honest delivery:

```python
import base64
from pathlib import Path

lm = router.lm("sora-2")
job = lm.video_generate(VideoGenerationRequest(
    model="sora-2",
    prompt="A red ball bounces once on a white floor",
    seconds=4,
))
part = job.wait().result()
Path("ball.mp4").write_bytes(base64.b64decode(part.data))
print(part.media_type, len(base64.b64decode(part.data)))
```
```output
video/mp4 1424178
```

Veo (`veo-3.1-lite-generate-preview` and friends) also delivers bytes —
its download URI answers **403 without your API key** (verified live),
and a URL you cannot open is not an honest `VideoPart`, so lm15 fetches
it for you.

`seconds` maps to each wire's duration knob (Sora `seconds`, Veo
`durationSeconds`) and **raises on xAI** — that wire has no slot, and
lm15 never silently drops a field.

### Re-attach and enumerate

The ticket is a plain string. Store it; die; come back:

```python
job = lm.video_job("video_6a96c91899508193993c9696c937873e01945c26a50f6a7a")

for j in lm.video_jobs(limit=3):
    print(j.id[:20] + "…", j.status, j.info.created_at)
```
```output
video_6a96c918995081… completed 2026-09-01T12:46:16Z
```

Enumerability is honest, not uniform: OpenAI lists account-wide, Gemini
lists **per model** (`lm.video_jobs(model="veo-3.1-lite-generate-preview")`),
and xAI has no list endpoint at all:

```python
router.lm("grok-imagine-video").video_jobs()
```
```output
UnsupportedFeatureError: xai: the wire has no video list endpoint (probed
2026-09-01: 404) — the ticket you stored is the only copy
```

No hidden local journal papers over that gap (rejected for batch, same
reasons here): on xAI, store the id.

## How it works

The router routes chat; video lives on the provider LM via
`router.lm()`. Underneath: Sora is a job object at `/v1/videos` with a
separate content endpoint; Veo is a Google long-running operation
(`predictLongRunning`) whose terminal body carries a key-bound file
URI; grok-imagine is a `request_id` you poll at `/v1/videos/{id}` until
`done` yields a public URL. lm15 folds all three into one closed status
vocabulary — `queued`, `running`, `completed`, `failed`, `cancelled` —
with each provider's own words preserved verbatim in `provider_data`.

`wait()` returns the terminal snapshot rather than raising on `failed`
(check `status`), mirroring batch. `result()` raises while the job
still runs, and delivers the video without re-hosting: bytes where the
wire streams them, a URL where the wire publishes one.

Input images (image-to-video) exist on all three wires but are **not
mapped yet** — the field raises. Named reason: xAI's image wire
silently ignores unknown fields (pixel-verified during the edits
campaign), so an unverified mapping could silently produce prompt-only
videos. The mappings land when live-receipted.

## Variations

- **Async**: `AsyncLMRouter` mirrors everything — `await
  lm.video_generate(...)`, `await job.wait()`, `await job.result()` —
  over the same pure hooks.
- **Costs**: video bills per clip and is the most expensive call in
  this cookbook; a 4-second Sora clip or one grok-imagine video runs
  tens of cents.
- **Provider knobs** (Sora `size`, Veo `parameters`, xAI tiers) ride
  `extensions` untranslated.
- **Anthropic** has no video endpoint and raises
  `UnsupportedFeatureError`.

## See also

- [12 — Batch jobs](12-batch-jobs.md) — the ticket pattern this reuses.
- [13 — Media generation](13-media-generation.md) — the synchronous
  media pair.
- [18 — Provider passthrough](18-provider-passthrough.md) — reading
  `provider_data` and `extensions`.
