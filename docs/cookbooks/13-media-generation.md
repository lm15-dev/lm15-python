# Media generation (images & speech)

**Provisional in 1.0:** these generation endpoints ship, but may change
incompatibly in 1.x with a contract change entry. The media parts used in
chat remain part of the frozen core. See
[release scope](../roadmap.md#what-ships-in-10-and-what-is-stable).

**Problem** — Sometimes the thing you want back is not text: a picture
of the diagram you described, or your paragraph read aloud. OpenAI,
Gemini, and xAI sell this; Anthropic does not. The wires could not look
more different — OpenAI has dedicated endpoints, Gemini does everything
through its ordinary chat call, xAI mirrors OpenAI's image endpoint —
but the shape of the exchange is identical: a prompt goes in, media
parts come out.

The outputs are the **same Part types you send as inputs**
([recipe 09](09-images-and-documents.md)). A generated image is an
`ImagePart` you can save, or feed straight back into a chat request.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

### Generate an image

```python
import base64
from pathlib import Path
from lm15 import LMRouter, ImageGenerationRequest, SpeechGenerationRequest, ImagePart

router = LMRouter()
lm = router.lm("gpt-image-1-mini")

resp = lm.image_generate(ImageGenerationRequest(
    model="gpt-image-1-mini",
    prompt="A simple flat red circle centered on a white background",
    size="1024x1024",
))
img = resp.images[0]
print(img.media_type, len(img.data), resp.usage.output_tokens)
Path("circle.png").write_bytes(base64.b64decode(img.data))
```
```output
image/png 1239852 4160
```

`media_type` is not a guess: OpenAI's response states
`"output_format": "png"` and lm15 reads it. The tokens are real too —
image models bill in tokens, and `usage` carries them.

### The same call on Gemini

```python
lm = router.lm("gemini-2.5-flash-image")

resp = lm.image_generate(ImageGenerationRequest(
    model="gemini-2.5-flash-image",
    prompt="A simple flat red circle centered on a white background",
))
print(resp.images[0].media_type, "|", resp.text)
```
```output
image/png | Here's a simple flat red circle centered on a white background for you! 
```

Gemini often **talks while it draws**; that narration arrives in
`resp.text` (sometimes `None` — the model decides). On OpenAI, `text`
is always `None`: the wire has no such thing.

### Edit an image

Input images are ordinary `ImagePart`s in `request.images` — "here is
my picture, change this one thing":

```python
lm = router.lm("gpt-image-1-mini")

edit = lm.image_generate(ImageGenerationRequest(
    model="gpt-image-1-mini",
    prompt="Add one small solid blue square in the bottom-right corner. "
           "Keep everything else exactly the same.",
    images=(ImagePart(media_type="image/png",
                      data=base64.b64encode(Path("circle.png").read_bytes()).decode()),),
))
print(edit.images[0].media_type, len(edit.images[0].data))
```
```output
image/png 1216520
```

Each adapter routes input images to its provider's real edit door —
OpenAI switches to `/images/edits`, Gemini sends parts in the same chat
call, xAI uses its edits endpoint. Where the wire cannot carry your
input (two images on xAI, a URL-addressed image on OpenAI), lm15
**raises** instead of silently ignoring it.

### The same call on Grok

```python
lm = router.lm("grok-imagine-image")  # subscription OAuth or XAI_API_KEY

resp = lm.image_generate(ImageGenerationRequest(
    model="grok-imagine-image",
    prompt="A simple flat red circle centered on a white background",
))
print(resp.images[0].media_type, resp.provider_data["usage"])
```
```output
image/jpeg {'cost_in_usd_ticks': 200000000}
```

Third provider, third truth: xAI returns **JPEG** and says so. It also
bills per image, not per token — `usage` stays honestly empty and the
cost figure rides in `provider_data`.

### Generate speech

```python
lm = router.lm("gpt-4o-mini-tts")

s = lm.speech_generate(SpeechGenerationRequest(
    model="gpt-4o-mini-tts",
    prompt="Hello from the cookbook.",
    voice="alloy",
    format="wav",
))
print(s.audio.media_type)
Path("hello.wav").write_bytes(base64.b64decode(s.audio.data))
```
```output
audio/wav
```

Every field except `model` and `prompt` is optional, and **omitting a
field means the server decides, not lm15**:

```python
s = lm.speech_generate(SpeechGenerationRequest(
    model="gpt-4o-mini-tts", prompt="Hello from the cookbook."))
print(s.audio.media_type)
```
```output
audio/mpeg
```

MP3 is OpenAI's real default, honestly reported — never silently
rewritten to something lm15 prefers. Gemini speaks raw PCM, with the
most honest media type on any wire:

```python
lm = router.lm("gemini-2.5-flash-preview-tts")

s = lm.speech_generate(SpeechGenerationRequest(
    model="gemini-2.5-flash-preview-tts",
    prompt="Hello from the cookbook.",
    voice="Kore",
))
print(s.audio.media_type, s.usage.output_tokens)
```
```output
audio/L16;codec=pcm;rate=24000 45
```

### Providers that cannot draw

```python
router.lm("claude-sonnet-4-5").image_generate(
    ImageGenerationRequest(model="claude-sonnet-4-5", prompt="a cat"))
```
```output
UnsupportedFeatureError: anthropic: image generation not supported
```

Anthropic has no generation endpoint of any kind. lm15 says so instead
of routing your prompt somewhere you did not choose.

## How it works

The router routes chat; generation lives on the provider LM, reached
via `router.lm()`. Under the hood: OpenAI images are
`/images/generations` (or `/images/edits`, multipart, when input images
are present) and speech is `/audio/speech`, which returns raw bytes
typed only by its content-type header. Gemini has no dedicated
endpoints at all — image models and TTS models answer the ordinary
`generateContent` call, so generation composes the same frozen chat
mapping as `complete()`. xAI mirrors OpenAI's image endpoints.

Three rules govern the surface. Media types come from the wire, never
from code — the same prompt produced `image/png` on two providers and
`image/jpeg` on the third. Omitted fields mean server defaults; lm15
injects none. And fields the wire cannot carry raise
`UnsupportedFeatureError`: `format` on Gemini (always PCM), `size` on
xAI (its sizing knobs ride `extensions`), more than one input image on
xAI.

`size` and `voice` are portable fields carrying each provider's own
vocabulary — exactly like `model`. OpenAI sizes in pixels
(`"1024x1024"`); Gemini in aspect ratios (`"16:9"`). Voices are
provider-named (`alloy`, `Kore`).

## Variations

- **Async**: `AsyncLMRouter` mirrors both verbs — `await
  lm.image_generate(...)`, `await lm.speech_generate(...)` — over the
  same request/parse hooks.
- **Chat-door generation**: Gemini image models used in ordinary
  `complete()` calls return image parts inside a normal `Response`.
  The dedicated verbs exist for the dedicated intent, not as the only
  door.
- **OpenAI knobs** (`quality`, `background`, `n`) and xAI's
  quality/resolution tiers ride `extensions` untranslated.
- **Video** (Sora, Veo, grok-imagine) is a job, not a call — it reuses
  the batch ticket pattern: [recipe 14](14-video-generation.md).

## See also

- [09 — Images, PDFs & documents](09-images-and-documents.md) — the
  same Part types as inputs.
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md)
- [12 — Batch jobs](12-batch-jobs.md) — the ticket pattern video reuses.
- [14 — Video generation](14-video-generation.md) — the job-shaped
  modality.
- [18 — Provider passthrough](18-provider-passthrough.md) — reading
  `provider_data` and `extensions`.
