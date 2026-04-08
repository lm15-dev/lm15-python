# Cookbook 05 — Multimodal (Images, Audio, Video, Documents)

## Image from URL

```python
import lm15
from lm15 import Part

resp = lm15.complete("gemini-2.5-flash", [
    "Describe this image.",
    Part.image(url="https://example.com/cat.jpg"),
])
print(resp.text)
```

## Image from bytes

```python
import lm15
from lm15 import Part

data = open("photo.png", "rb").read()
resp = lm15.complete("claude-sonnet-4-5", [
    "What's in this photo?",
    Part.image(data=data, media_type="image/png"),
])
print(resp.text)
```

## Document (PDF)

```python
import lm15
from lm15 import Part

resp = lm15.complete("claude-sonnet-4-5", [
    "Summarize this paper.",
    Part.document(url="https://example.com/paper.pdf"),
])
print(resp.text)
```

## Video

```python
import lm15
from lm15 import Part

resp = lm15.complete("gemini-2.5-flash", [
    "What happens in this video?",
    Part.video(url="https://example.com/clip.mp4"),
])
print(resp.text)
```

## Image generation → vision (cross-model)

```python
import lm15

resp = lm15.complete("gpt-4.1-mini", "Draw a cat wearing a top hat.", output="image")

# Pass the image part directly to another model
resp2 = lm15.complete("claude-sonnet-4-5", ["Describe this image in detail.", resp.image])
print(resp2.text)
```

## Audio generation → transcription

```python
import lm15

resp = lm15.complete("gpt-4o-mini-tts", "Say hello world.", output="audio")
resp2 = lm15.complete("gemini-2.5-flash", ["Transcribe this audio.", resp.audio])
print(resp2.text)
```

## Upload (provider file API)

```python
import lm15

doc = lm15.upload("claude-sonnet-4-5", "contract.pdf")
resp = lm15.complete("claude-sonnet-4-5", ["Find all liability clauses.", doc])
print(resp.text)
```

Upload on a model object:

```python
claude = lm15.model("claude-sonnet-4-5")
doc = claude.upload("contract.pdf")
resp = claude(["Summarize section 3.", doc])
print(resp.text)
```
