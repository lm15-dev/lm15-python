# Cookbook 13 — Live Sessions (Real-Time Audio & Video)

Live models accept audio, video, and text — and can respond with audio or text. lm15 supports them in two ways:

1. **Completion mode** — `lm15.call()` with a live model. Same `Result` as any other model. Use when you have your input ready upfront.
2. **Session mode** — `lm15.live()` for persistent bidirectional sessions. Use for real-time voice conversations.

---

## Setup

```bash
pip install lm15[live]
```

This adds `websockets` for the WebSocket connection. For the audio examples in this cookbook, you'll also want:

```bash
uv pip install sounddevice soundfile numpy
```

These give you cross-platform microphone recording and audio playback (Windows, macOS, Linux). For the video feed examples, you'll also want `Pillow` for image/screenshot handling.

---

## Completion mode (live models via `call()`)

Live models work transparently through `lm15.call()`. The WebSocket connection is handled automatically — you don't see it.

### Speech-to-text

Record from your microphone, then transcribe:

```python
import lm15
import sounddevice as sd
import soundfile as sf
from lm15 import Part

# Record 5 seconds from the default microphone
duration = 5  # seconds
sample_rate = 16000
print(f"🎙️  Recording {duration} seconds...")
audio_data = sd.rec(duration * sample_rate, samplerate=sample_rate,
                     channels=1, dtype="int16")
sd.wait()  # block until recording is done
sf.write("recording.wav", audio_data, sample_rate)
print("✅ Done.")
```
```output | ✓ 5.0s | 14 vars
🎙️  Recording 5 seconds...
✅ Done.
```

```python
# Transcribe
import lm15
from lm15 import Part

audio = Part.audio(data=open("recording.wav", "rb").read(), media_type="audio/wav")
r = lm15.call("gemini-3.1-flash-live-preview", [audio, "Transcribe this audio."])
print(r.text)
```
```output | ✓ 7.4s | 4 vars
Bonjour, ceci est un test 1 2 3 4 5
```

### Text-to-speech

```python ✓
import lm15
import sounddevice as sd
import numpy as np

r = lm15.call("gemini-3.1-flash-live-preview", "Say hello in French.", output="audio")

# r.audio_bytes is a valid WAV file — parse and play it
wav_bytes = r.audio_bytes
# Skip 44-byte WAV header, read as 16-bit PCM
pcm = np.frombuffer(wav_bytes[44:], dtype=np.int16)
sd.play(pcm, samplerate=24000)
sd.wait()  # block until playback finishes
```
Or just save it:

```python ✓
with open("hello.wav", "wb") as f:
    f.write(r.audio_bytes)
# Open with any media player — it's a standard WAV file
```

### Vision: live models see video, not images

Live models process vision as a **continuous stream of video frames** — like a camera feed or screen share. They don't support single-image input via `lm15.call()`. For still image analysis, use a non-live model:

```python
import lm15
from lm15 import Part

image = Part.image(data=open("photo.png", "rb").read(), media_type="image/png")
r = lm15.call("gemini-2.5-flash", [image, "Describe what you see."])
print(r.text)
```

To send video frames to a live model, use a **session** (see [Video feed session](#video-feed-session) below) — that's how camera and screen sharing work.
### Audio + text question

Use the microphone recording from earlier together with a text question:

```python
import lm15
from lm15 import Part

# Uses recording.wav captured in the first example
voice = Part.audio(data=open("recording.wav", "rb").read(), media_type="audio/wav")

r = lm15.call("gemini-3.1-flash-live-preview",
    [voice, "What language is the speaker using? Respond in that language."])
print(r.text)
```
```output | ✓ 8.2s | 14 vars
Bonjour ! Le locuteur utilise le français. Comment puis-je t'aider ?
```

### Streaming the response

```python
import lm15
from lm15 import Part

audio = Part.audio(data=open("recording.wav", "rb").read(), media_type="audio/wav")

for text in lm15.call("gemini-3.1-flash-live-preview",
                       [audio, "Summarize what was said."]):
    print(text, end="", flush=True)
print()
```
```output | ✓ 10.4s | 15 vars
Vous avez simplement dit "Bonjour, ceci est un test 1 2 3 4 5". Est-ce qu'il y a autre chose à résumer?
```

### Tools work the same

```python
import lm15
from lm15 import Part

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"

audio = Part.audio(data=open("recording.wav", "rb").read(), media_type="audio/wav")
r = lm15.call("gemini-3.1-flash-live-preview",
    [audio, "Answer the question in this audio."],
    tools=[get_weather])
print(r.text)
```

```output | ✓ 17.0s | 16 vars
The user is saying "Bonjour, ceci est un test 12345," which means "Hello, this is a test 12345" in French. They are likely testing audio input or quality. Is there anything specific you'd like to say back to them?
```

### Force WebSocket transport on any Gemini model

If the model name doesn't contain `-live`, you can force the WebSocket transport:

```python
r = lm15.call("gemini-2.5-flash", "Hello.",
    provider={"transport": "live"})
```
```output | ✗ 4ms | 2 vars
Traceback (most recent call last):
  File "/home/maxime/.cache/rat/kernels/py@lm15/python-kernel.py", line 808, in run_code
    exec(compile(tree, "<rat>", "exec"), namespace, namespace)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<rat>", line 1, in <module>
  File "/home/maxime/Projects/lm15/lm15/api.py", line 214, in call
    m = globals()["model"](
        model,
    ...<8 lines>...
        env=env,
    )
  File "/home/maxime/Projects/lm15/lm15/api.py", line 99, in model
    lm = _get_client(
        api_key=_resolve("api_key", api_key),
        provider_hint=provider,
        env=_resolve("env", env),
    )
  File "/home/maxime/Projects/lm15/lm15/api.py", line 52, in _get_client
    client = _client_cache.get(cache_key)
TypeError: unhashable type: 'dict'
```

---

## Session mode (`lm15.live()`)

For persistent bidirectional sessions — real-time voice assistants, continuous mic/speaker streaming, interactive conversations.

### Text conversation

```python
import lm15

with lm15.live("gemini-3.1-flash-live-preview",
               system="You are a helpful assistant.") as session:
    session.send(text="What is the capital of France? Write it to me and say it outloud")

    for event in session:
        match event.type:
            case "text":     print(event.text, end="")
            case "turn_end": break
    print()
```
```output | ✓ 4.0s | 3 vars
The capital of France is Paris. Paris
```

### Voice assistant (turn-based)

Record a question, send it, play the audio response. Repeat.

```python
import lm15
import sounddevice as sd
import numpy as np
import base64

session = lm15.live("gemini-3.1-flash-live-preview",
    system="You are a voice assistant. Keep answers short.",
    voice="Kore",
)

SAMPLE_RATE = 16000
DURATION = 5  # seconds per recording

try:
    while True:
        input("Press Enter to speak (Ctrl+C to quit)...")

        # Record
        print(f"🎙️  Listening for {DURATION}s...")
        audio_data = sd.rec(DURATION * SAMPLE_RATE, samplerate=SAMPLE_RATE,
                            channels=1, dtype="int16")
        sd.wait()
        session.send(audio=audio_data.tobytes())

        # Collect response
        response_audio = bytearray()
        for event in session:
            match event.type:
                case "audio":
                    response_audio.extend(base64.b64decode(event.data))
                case "text":
                    print(event.text, end="")
                case "turn_end":
                    break
        print()

        # Play response
        if response_audio:
            pcm = np.frombuffer(bytes(response_audio), dtype=np.int16)
            sd.play(pcm, samplerate=24000)
            sd.wait()

except KeyboardInterrupt:
    print("\nGoodbye!")
finally:
    session.close()
```

### Full-duplex voice (talk while listening)

For real-time voice where you talk and listen simultaneously. This uses two threads — one for the microphone, one for the speaker.

```python
import threading
import base64
import lm15
import sounddevice as sd
import numpy as np

session = lm15.live("gemini-3.1-flash-live-preview", voice="Kore")
stop = threading.Event()

# Background thread: stream mic audio to the model
def mic_thread():
    MIC_RATE, CHUNK_SAMPLES = 16000, 4000  # 250ms chunks
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16") as stream:
        while not stop.is_set():
            data, _ = stream.read(CHUNK_SAMPLES)
            session.send(audio=data.tobytes())
    session.send(end_audio=True)

threading.Thread(target=mic_thread, daemon=True).start()

# Main thread: receive and play response audio
SPEAKER_RATE = 24000
print("🎙️  Speak! Press Ctrl+C to stop.")
try:
    for event in session:
        match event.type:
            case "audio":
                pcm = np.frombuffer(base64.b64decode(event.data), dtype=np.int16)
                sd.play(pcm, samplerate=SPEAKER_RATE)
            case "text":
                print(event.text, end="", flush=True)
            case "turn_end":
                print()
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    session.close()
    print("👋 Done.")
```

`send()` is thread-safe and non-blocking. The event iterator is blocking. This is the natural split for voice applications.

### Multi-turn conversation

The session maintains context server-side — each turn sees all previous turns:

```python
import lm15

with lm15.live("gemini-3.1-flash-live-preview",
               system="You remember everything.") as session:
    for question in ["My name is Max.", "I like chess.", "What do you know about me?"]:
        session.send(text=question)
        for event in session:
            if event.type == "text":
                print(event.text, end="")
            if event.type == "turn_end":
                break
        print()
```

### Video feed session

Live models see video as a continuous stream of JPEG frames — like a webcam or screen share. Send frames via `session.send(video=...)` and ask questions with text:

```python
import lm15
import base64, io, time
from PIL import Image

# Capture a screenshot (or load any image)
img = Image.open("screenshot.png").convert("RGB")
img.thumbnail([1024, 1024])  # resize for the API
buf = io.BytesIO()
img.save(buf, format="JPEG")
jpeg_bytes = buf.getvalue()

session = lm15.live("gemini-3.1-flash-live-preview",
                    system="You are a helpful visual assistant.")

try:
    # Send a few frames so the model registers the visual input
    for _ in range(3):
        session.send(video=jpeg_bytes)
        time.sleep(0.5)

    # Now ask about what it sees
    session.send(text="What do you see on screen?")

    for event in session:
        match event.type:
            case "text":     print(event.text, end="")
            case "turn_end": break
    print()
finally:
    session.close()
```

For continuous feeds (webcam, screen share), stream frames in a background thread:

```python
import lm15
import threading, time, io, base64
from PIL import ImageGrab  # Windows/macOS screenshot

session = lm15.live("gemini-3.1-flash-live-preview")
stop = threading.Event()

def screen_feed():
    while not stop.is_set():
        img = ImageGrab.grab()
        img.thumbnail([1024, 1024])
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        session.send(video=buf.getvalue())
        time.sleep(1.0)  # ~1 fps

threading.Thread(target=screen_feed, daemon=True).start()

try:
    session.send(text="Describe what's on my screen.")
    for event in session:
        if event.type == "text": print(event.text, end="")
        if event.type == "turn_end": break
    print()
finally:
    stop.set()
    session.close()
```

`send(video=...)` accepts raw bytes (which are base64-encoded automatically) or a base64 string. Each call sends one frame.

### Tools in a session

Auto-execute — pass callables and lm15 runs them when the model requests:

```python
import lm15

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"

with lm15.live("gemini-3.1-flash-live-preview", tools=[get_weather]) as session:
    session.send(text="What's the weather in Montreal?")

    for event in session:
        match event.type:
            case "text":        print(event.text, end="")
            case "tool_call":   print(f"\n🔧 {event.name}({event.input})")
            case "turn_end":    break
    print()
```

### Approval gate with `on_tool_call`

Intercept tool calls before they execute — useful for dangerous operations:

```python
import lm15
from lm15 import FunctionTool

write_file = FunctionTool(
    name="write_file",
    description="Write content to a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)

def approve(call):
    print(f"\n⚠️  {call.name} wants to write to: {call.input.get('path')}")
    answer = input("Approve? [y/n] ")
    if answer.lower() == "y":
        return None  # proceed with auto-execute (or manual if no fn)
    return "User denied this action."

with lm15.live("gemini-3.1-flash-live-preview",
               tools=[write_file], on_tool_call=approve) as session:
    session.send(text="Create a config.py file with database settings.")
    for event in session:
        if event.type == "text": print(event.text, end="")
        if event.type == "turn_end": break
    print()
```

### Manual tool execution

Handle tool calls entirely yourself:

```python
import lm15
from lm15 import FunctionTool

search = FunctionTool(
    name="search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

with lm15.live("gemini-3.1-flash-live-preview", tools=[search]) as session:
    session.send(text="Find the latest news about AI.")

    for event in session:
        match event.type:
            case "tool_call":
                # Your code runs here — call a real API, database, etc.
                result = f"Top result for '{event.input.get('query', '')}': AI advances continue in 2026."
                session.send(tool_result={event.id: result})
            case "text":
                print(event.text, end="")
            case "turn_end":
                break
    print()
```

### Interrupting the model

```python
session.send(interrupt=True)  # model stops its current response
```

---

## Session on a model object

Bind your config once, open sessions from it:

```python
import lm15

agent = lm15.model("gemini-3.1-flash-live-preview",
    system="You are a coding assistant.",
    tools=[read_file, write_file],
)

# Open a live session with the model's config
with agent.live() as session:
    session.send(text="What files are in the project?")
    for event in session:
        if event.type == "text": print(event.text, end="")
        if event.type == "turn_end": break
    print()

# Override voice or tools for a specific session
with agent.live(voice="Puck", tools=[read_file]) as session:
    session.send(text="Read the README.")
    for event in session:
        if event.type == "text": print(event.text, end="")
        if event.type == "turn_end": break
    print()
```

---

## Async sessions

```python
import lm15

session = await lm15.alive("gemini-3.1-flash-live-preview",
                            system="You are helpful.")

await session.send(text="Hello!")

async for event in session:
    match event.type:
        case "text":     print(event.text, end="")
        case "turn_end": break

await session.close()
```

Also on model objects:

```python
agent = lm15.model("gemini-3.1-flash-live-preview", system="You are helpful.")
session = await agent.alive()

await session.send(text="Hello!")
async for event in session:
    if event.type == "text": print(event.text, end="")
    if event.type == "turn_end": break

await session.close()
```

---

## Reference

### Event types

| `event.type` | Fields | When |
|---|---|---|
| `"text"` | `event.text` | Model generated text |
| `"audio"` | `event.data` (base64 string) | Model generated an audio chunk |
| `"tool_call"` | `event.id`, `event.name`, `event.input` | Model wants to call a tool |
| `"interrupted"` | — | Model was interrupted |
| `"turn_end"` | `event.usage` | Model finished this turn |
| `"error"` | `event.error` | Something went wrong |

### Send methods

| Keyword | What it sends |
|---|---|
| `text="..."` | Text message |
| `audio=bytes` | Audio chunk (raw bytes or base64 string) |
| `video=bytes` | Video frame (raw bytes or base64 string) |
| `tool_result={id: result}` | Tool result(s) back to the model |
| `interrupt=True` | Stop the model's current response |
| `end_audio=True` | Signal end of audio stream |

You can also pass a `LiveClientEvent` directly for full control:

```python
from lm15 import LiveClientEvent
session.send(LiveClientEvent(type="text", text="Hello"))
```

### Completion mode vs session mode

| | `lm15.call()` | `lm15.live()` |
|---|---|---|
| **Connection** | Opened and closed per call | Persistent until `close()` |
| **Return type** | `Result` (same as all models) | `Session` (iterator + `send()`) |
| **State** | Client-managed | Server-managed |
| **Vision** | Audio only (no image input) | Video frames via `send(video=...)` |
| **Best for** | STT, TTS, analysis of recordings | Voice assistants, video feeds, screen share |
| **Tools** | Full support | Full support |
| **Streaming** | `for text in result` | `for event in session` |

**Rule of thumb:** If you know your input before you start, use `call()`. If you're streaming input continuously (audio, video, screen share) and the model responds while you're still sending, use `live()`.
