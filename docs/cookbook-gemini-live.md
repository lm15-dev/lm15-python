# lm15 Cookbook: Gemini Live (Runnable)

This notebook demonstrates Gemini Live with `lm15-python` using the lower-level provider API directly. It follows the same runnable Markdown style as `docs/cookbook-all-features.md`, but focuses on live WebSocket interactions:

- one-shot live completion with `GeminiLM.stream(Request(...))`
- persistent bidirectional sessions with `GeminiLM.live(LiveConfig(...))`
- audio output and audio input
- video frames
- function calling/tool results
- interruption

`lm15-python` intentionally exposes the live API close to the provider layer. The examples below do **not** use the older high-level `lm15.call()` / `lm15.live()` interface from `lm15-python`.

## Setup & Initialization

Live sessions use the optional `websockets` dependency. In this repo's venv it is already installed; in a fresh environment use:

```bash
pip install -e '.[live]'
```
```output
Defaulting to user installation because normal site-packages is not writeable
Obtaining file:///home/maxime/Projects/lm15-dev/lm15-python
  Installing build dependencies ... - \ | / - \ | done
  Checking if build backend supports build_editable ... done
ERROR: Project file:///home/maxime/Projects/lm15-dev/lm15-python has a 'pyproject.toml' and its build backend is missing the 'build_editable' hook. Since it does not have a 'setup.py' nor a 'setup.cfg', it cannot be installed in editable mode. Consider using a build backend that supports PEP 660.
Command exited with code 1
```

The setup cell loads `GEMINI_API_KEY` from a `.env` file in the current directory or a parent directory, then creates a `GeminiLM` provider. Override the model with `GEMINI_LIVE_MODEL` if needed.

```py
import os
import shlex
import json
import base64
import wave
import io
import time
import shutil
import subprocess
import pprint
from pathlib import Path
from collections import Counter

from lm15.providers import GeminiLM
from lm15.types import (
    Request,
    Message,
    Config,
    LiveConfig,
    LiveClientImageEvent,
    LiveClientTurnEvent,
    FunctionTool,
    StreamDeltaEvent,
    StreamEndEvent,
    TextDelta,
    AudioDelta,
    audio,
    image,
    text,
)


def find_dotenv(filename=".env"):
    start = Path.cwd().resolve()
    candidates = [start, *start.parents]
    seen = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        path = directory / filename
        if path.exists():
            return path
    return None


def load_env_file(path, *, override=True):
    """Load shell-style .env lines like: export GEMINI_API_KEY="AIza..."."""
    loaded = {}
    if path is None:
        return loaded

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            parsed = shlex.split(value, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip('"\'')

        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, "")
    return loaded


def mask(value):
    if not value:
        return "missing"
    if len(value) <= 12:
        return "set"
    return f"{value[:7]}...{value[-4:]}"


env_path = find_dotenv()
loaded = load_env_file(env_path)
print(f"Loaded .env from: {env_path or 'not found'}")
print(f"  GEMINI_API_KEY: {mask(os.environ.get('GEMINI_API_KEY'))}")

MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
lm = GeminiLM(api_key=os.environ.get("GEMINI_API_KEY", ""))
print("MODEL:", MODEL)
```

```output:exec-1777843832485-lxq1i
Loaded .env from: /home/maxime/Projects/lm15-dev/.env
  GEMINI_API_KEY: AIzaSyB...nH1w
MODEL: gemini-3.1-flash-live-preview

✓ 56ms | 65 vars
```

Helper functions used throughout the notebook:

```python ✓
# Rat Markdown notebooks cannot render IPython audio widgets, so default to an
# external local player. In Jupyter, set AUDIO_BACKEND = "auto" for inline audio.
AUTO_PLAY_AUDIO = True
AUDIO_BACKEND = os.environ.get("LM15_AUDIO_BACKEND", "external")


def collect_turn(session, *, max_events=80, play_audio=None, audio_path="live-turn-audio.wav"):
    """Read one live session turn into a compact summary, and play audio if present."""
    events = []
    text_chunks = []
    audio_bytes = bytearray()
    tool_calls = []
    usage = None

    for event in session:
        events.append(event)
        if event.type == "text":
            text_chunks.append(event.text)
        elif event.type == "audio":
            audio_bytes.extend(base64.b64decode(event.data))
        elif event.type == "tool_call":
            tool_calls.append(event)
        elif event.type == "turn_end":
            usage = event.usage
            break
        elif event.type == "error":
            break
        if len(events) >= max_events:
            break

    audio_bytes_value = bytes(audio_bytes)
    if play_audio is None:
        play_audio = AUTO_PLAY_AUDIO
    if play_audio and audio_bytes_value:
        play_pcm_audio(audio_bytes_value, path=audio_path, backend=AUDIO_BACKEND)

    return {
        "text": "".join(text_chunks),
        "audio_bytes": audio_bytes_value,
        "audio_byte_count": len(audio_bytes_value),
        "tool_calls": tool_calls,
        "usage": usage,
        "events": events,
    }


def summarize_stream(events, *, play_audio=None, audio_path="live-stream-audio.wav"):
    """Summarize lm15 StreamEvent values, and play audio deltas if present."""
    text_value = "".join(
        event.delta.text
        for event in events
        if event.type == "delta" and isinstance(event.delta, TextDelta)
    )
    audio_bytes = b"".join(
        base64.b64decode(event.delta.data)
        for event in events
        if event.type == "delta" and isinstance(event.delta, AudioDelta)
    )
    ends = [event for event in events if event.type == "end"]
    errors = [event for event in events if event.type == "error"]
    if play_audio is None:
        play_audio = AUTO_PLAY_AUDIO
    if play_audio and audio_bytes:
        play_pcm_audio(audio_bytes, path=audio_path, backend=AUDIO_BACKEND)

    return {
        "text": text_value,
        "audio_byte_count": len(audio_bytes),
        "event_types": [event.type for event in events],
        "finish_reason": ends[-1].finish_reason if ends else None,
        "usage": ends[-1].usage if ends else None,
        "errors": errors,
    }


def pcm_to_wav_bytes(pcm_bytes, *, sample_rate=24000, channels=1, sample_width=2):
    """Wrap raw PCM bytes in an in-memory WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


def write_pcm_wav(path, pcm_bytes, *, sample_rate=24000, channels=1, sample_width=2):
    """Gemini Live audio deltas are raw PCM; wrap them in a WAV file."""
    Path(path).write_bytes(
        pcm_to_wav_bytes(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
    )
    return Path(path)


def _external_audio_player_command(path):
    """Return a local audio-player command if one is available."""
    candidates = [
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]),
        ("paplay", ["paplay", str(path)]),
        ("pw-play", ["pw-play", str(path)]),
        ("aplay", ["aplay", str(path)]),
        ("afplay", ["afplay", str(path)]),  # macOS
    ]
    for binary, command in candidates:
        if shutil.which(binary):
            return command
    return None


def play_pcm_audio(
    pcm_bytes,
    *,
    sample_rate=24000,
    path="live-audio-preview.wav",
    autoplay=False,
    backend="auto",
    block=True,
):
    """Play raw PCM bytes from Gemini Live.

    In Jupyter/IPython, this displays an inline audio widget. In Rat Markdown
    notebooks, rich IPython display is not available, so the helper writes a
    WAV file and falls back to a local player such as ffplay/paplay/pw-play.

    Args:
        pcm_bytes: Raw signed 16-bit PCM bytes.
        sample_rate: Gemini Live usually returns 24 kHz audio.
        path: WAV file path used by the external-player fallback.
        autoplay: Passed to IPython.display.Audio when rich display is available.
        backend: "auto", "ipython", "external", or "file".
        block: If True, wait until the external player exits.
    """
    if not pcm_bytes:
        print("No audio bytes to play.")
        return None

    path = Path(path)
    wav_bytes = pcm_to_wav_bytes(pcm_bytes, sample_rate=sample_rate)

    if backend in {"auto", "ipython"}:
        try:
            from IPython import get_ipython
            from IPython.display import Audio, display
            if get_ipython() is None:
                raise RuntimeError("not running inside IPython")
        except Exception:
            if backend == "ipython":
                raise
        else:
            widget = Audio(data=wav_bytes, autoplay=autoplay)
            display(widget)
            return widget

    path.write_bytes(wav_bytes)
    print(f"Wrote {path}")

    if backend in {"auto", "external"}:
        command = _external_audio_player_command(path)
        if command is None:
            print("No external audio player found. Open the WAV file manually.")
            return path

        print(f"Playing with {command[0]}...")
        # Important for Rat notebooks: the Python runtime uses stdout as a
        # JSON protocol channel. External players can print non-JSON output,
        # so silence and detach them from the kernel's standard streams.
        if block:
            subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        return path

    return path


def audio_bytes_from_live_events(events):
    """Collect raw PCM bytes from LiveServerAudioEvent values."""
    return b"".join(
        base64.b64decode(event.data)
        for event in events
        if getattr(event, "type", None) == "audio"
    )


def play_live_event_audio(events, *, path="live-turn-audio.wav", play_audio=None):
    """Play audio contained in a list of live-session server events."""
    audio_bytes_value = audio_bytes_from_live_events(events)
    if play_audio is None:
        play_audio = AUTO_PLAY_AUDIO
    if play_audio and audio_bytes_value:
        play_pcm_audio(audio_bytes_value, path=path, backend=AUDIO_BACKEND)
    return audio_bytes_value
```

```output:exec-1777843836808-7lt9b
✓ 46ms | 65 vars
```
---

## 1. Persistent Live Session: Text In, Audio + Text Out

`GeminiLM.live()` opens a persistent WebSocket session. A live session is a context manager; send client events with `session.send(...)`, then iterate server events until `turn_end`.

For the tested live model, a text prompt produces both audio chunks (`LiveServerAudioEvent`) and output transcription text (`LiveServerTextEvent`). The text below is the transcription of the audio response.

```python
with lm.live(LiveConfig(model=MODEL, system="Be concise.")) as session:
    session.send(text="Reply with exactly: live hello")
    turn1 = collect_turn(session)

print("text:", repr(turn1["text"]))
print("audio_byte_count:", turn1["audio_byte_count"])
print("event_types:", [event.type for event in turn1["events"]])
print("usage:", turn1["usage"])
```

```output:exec-1777843842087-k35em
Wrote live-turn-audio.wav
Playing with ffplay...
text: 'live hello'
audio_byte_count: 48964
event_types: ['audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'turn_end']
usage: Usage(input_tokens=0, output_tokens=0, total_tokens=0, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None)

✓ 3.4s | 65 vars
```
```output | ✓ 4.5s | 45 vars
Wrote live-turn-audio.wav
Playing with ffplay...
text: 'live hello'
audio_byte_count: 83042
event_types: ['audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'turn_end']
usage: Usage(input_tokens=0, output_tokens=0, total_tokens=0, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None)
```

`collect_turn()` auto-plays any audio it collected. To disable playback, set `AUTO_PLAY_AUDIO = False`; to only write WAV files, set `AUDIO_BACKEND = "file"`; in Jupyter, set `AUDIO_BACKEND = "auto"` for an inline widget.

---

## 2. Session State: Multi-turn Memory

The session keeps context server-side. You do not resend the whole chat transcript for each turn.

```python
with lm.live(LiveConfig(model=MODEL, system="Be concise.")) as session:
    session.send(text="Remember this code word: amber-lime. Reply OK.")
    remember_1 = collect_turn(session)

    session.send(text="What code word did I ask you to remember? Reply with only the code word.")
    remember_2 = collect_turn(session)

print("turn 1:", repr(remember_1["text"]))
print("turn 2:", repr(remember_2["text"]))
print("turn 1 event types:", [event.type for event in remember_1["events"]])
print("turn 2 event types:", [event.type for event in remember_2["events"]])
```
```output | ✓ 5.5s | 47 vars
Wrote live-turn-audio.wav
Playing with ffplay...
Wrote live-turn-audio.wav
Playing with ffplay...
turn 1: 'OK.'
turn 2: 'amber-lime'
turn 1 event types: ['audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'turn_end']
turn 2 event types: ['audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'turn_end']
```

---

## 3. One-shot Live Completion via `stream(Request)`

For one-shot live calls, use the normal `lm.stream(Request(...))` surface. `GeminiLM.stream()` automatically routes models whose names contain `-live` through Gemini's Live WebSocket endpoint.

This is the live analogue of a completion call: create a `Request`, stream typed `StreamEvent` values, then summarize/materialize them yourself.

```python
req_live_stream = Request(
    model=MODEL,
    messages=(Message.user("Reply with exactly: stream hello"),),
    config=Config(max_tokens=40),
)

live_stream_events = list(lm.stream(req_live_stream))
live_stream_summary = summarize_stream(live_stream_events)
pprint.pp(live_stream_summary, width=100, sort_dicts=False)
print("first events:", live_stream_events[:4])
print("last event:", live_stream_events[-1])
```
```output | ✓ 3.6s | 50 vars
Wrote live-stream-audio.wav
Playing with ffplay...
{'text': 'stream hello',
 'audio_byte_count': 64352,
 'event_types': ['start',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'end'],
 'finish_reason': 'stop',
 'usage': Usage(input_tokens=145,
                output_tokens=7,
                total_tokens=145,
                cache_read_tokens=None,
                cache_write_tokens=None,
                reasoning_tokens=None,
                input_audio_tokens=None,
                output_audio_tokens=None),
 'errors': []}
first events: [StreamStartEvent(id=None, model='gemini-3.1-flash-live-preview', type='start'), StreamDeltaEvent(delta=AudioDelta(data='<base64: 4 chars>', url=None, file_id=None, part_index=0, media_type='audio/pcm;rate=24000'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='stream hello', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=AudioDelta(data='<base64: 11560 chars>', url=None, file_id=None, part_index=0, media_type='audio/pcm;rate=24000'), type='delta')]
last event: StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=145, output_tokens=7, total_tokens=145, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    type='end',
)
```

The audio deltas are raw PCM at 24 kHz. Wrap them in a WAV container if you want to save or play them with common audio tools.

```python
generated_pcm = b"".join(
    base64.b64decode(event.delta.data)
    for event in live_stream_events
    if event.type == "delta" and isinstance(event.delta, AudioDelta)
)

wav_path = write_pcm_wav("live-stream-hello.wav", generated_pcm)
print(wav_path, wav_path.stat().st_size)
```
```output | ✓ 52ms | 52 vars
live-stream-hello.wav 64396
```

---

## 4. Audio Input: Transcribe the Audio Generated Above

The previous section generated raw 24 kHz PCM audio. We can feed it back as an `AudioPart` with an explicit media type and ask the live model to transcribe it.

Notice that this uses `lm15.types.audio(...)` and a normal `Request`; `GeminiLM.stream()` still routes through the live WebSocket transport because the model is a live model.

```python
req_transcribe_generated_audio = Request(
    model=MODEL,
    messages=(
        Message.user([
            audio(data=generated_pcm, media_type="audio/pcm;rate=24000"),
            text("Transcribe this audio. Return only the spoken words."),
        ]),
    ),
)

transcribe_events = list(lm.stream(req_transcribe_generated_audio))
transcribe_summary = summarize_stream(transcribe_events)
pprint.pp(transcribe_summary, width=100, sort_dicts=False)
```
```output | ✓ 4.0s | 55 vars
Wrote live-stream-audio.wav
Playing with ffplay...
{'text': 'Stream Hello',
 'audio_byte_count': 51844,
 'event_types': ['start',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'delta',
                 'end'],
 'finish_reason': 'stop',
 'usage': Usage(input_tokens=186,
                output_tokens=6,
                total_tokens=186,
                cache_read_tokens=None,
                cache_write_tokens=None,
                reasoning_tokens=None,
                input_audio_tokens=None,
                output_audio_tokens=None),
 'errors': []}
```

---

## 5. Image Turns and Realtime Image Frames

Live media now distinguishes a **turn** (ordinary prompt content made of existing `Part` objects) from realtime media chunks.

Use `session.send_turn(...)` when you want to send normal prompt content such as a still image plus text. Use `session.send_image(...)` for realtime camera/screen frames. Gemini's wire protocol calls those frame chunks `realtimeInput.video`, but the universal lm15 event is an image event with an explicit `media_type`.

This diagnostic cell shows both encodings without needing to call the model:

```python
from lm15.types import LiveClientImageEvent, LiveClientTurnEvent, image, text

# A valid 64x64 red JPEG frame, embedded to keep the notebook self-contained.
red_jpeg = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD50ooor8MP9UwooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD/2Q=="
)

turn_payload = lm._encode_live_client_event(
    LiveClientTurnEvent(parts=(image(data=red_jpeg, media_type="image/jpeg"), text("What color is this?")))
)[0]
frame_payload = lm._encode_live_client_event(
    LiveClientImageEvent(data=base64.b64encode(red_jpeg).decode("ascii"), media_type="image/jpeg")
)[0]

print("red_jpeg bytes:", len(red_jpeg), red_jpeg[:4])
print("turn part mime:", turn_payload["clientContent"]["turns"][0]["parts"][0]["inlineData"]["mimeType"])
print("frame mime:", frame_payload["realtimeInput"]["video"]["mimeType"])
```
```output | ✓ 43ms | 58 vars
red_jpeg bytes: 694 b'\xff\xd8\xff\xe0'
turn part mime: image/jpeg
frame mime: image/jpeg
```

Now actually send frames over the live realtime path and listen to the response. Gemini includes realtime video frames in the next turn only when the live session is configured with the right turn coverage; `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO` means "include all video frames since the last turn."

`send_realtime_input`-style video is asynchronous, so send a few frames with a short delay before asking the question. This is closer to a real webcam/screen-share feed than sending a single still frame and immediately asking about it.

```python
live_video_config = LiveConfig(
    model=MODEL,
    system="Be concise.",
    extensions={
        "realtimeInputConfig": {
            "turnCoverage": "TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO",
        },
    },
)

with lm.live(live_video_config) as session:
    # This is Gemini Live realtime video. In lm15's universal API each frame is
    # an image event, because a live "video" feed is a stream of image frames.
    for _ in range(6):
        session.send_image(red_jpeg, media_type="image/jpeg")
        time.sleep(0.25)

    # Give Gemini a moment to ingest the realtime frame buffer before the turn.
    time.sleep(0.75)
    session.send_text("What color dominates the video frames I just sent? be verbose")
    live_video_turn = collect_turn(session, audio_path="live-video-frame-audio.wav")

print("text:", repr(live_video_turn["text"]))
print("event_types:", [event.type for event in live_video_turn["events"]])
```
```output | ✓ 25.8s | 60 vars
Wrote live-video-frame-audio.wav
Playing with ffplay...
text: 'The color that absolutely dominates the video frames you sent is a vibrant, deeply saturated red. Every pixel in both frames appears to be filled with this single,'
event_types: ['audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'text']
```

---

## 6. Tools: Manual Function Calling in a Live Session

Declare tools with `FunctionTool`, pass them in `LiveConfig`, listen for `tool_call`, then send results back with `session.send(tool_result={...})`.

This is intentionally lower-level than `lm15-python`'s older convenience wrappers: you receive typed tool-call events and decide when/how to answer them.

```python
weather_tool = FunctionTool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

with lm.live(
    LiveConfig(
        model=MODEL,
        system="Use tools when useful. Be concise.",
        tools=(weather_tool,),
    )
) as session:
    session.send(text="What is the weather in Montreal? Use the tool.")
    tool_demo_events = []
    tool_demo_text = []
    tool_demo_calls = []

    for event in session:
        tool_demo_events.append(event)
        if event.type == "text":
            tool_demo_text.append(event.text)
        elif event.type == "tool_call":
            tool_demo_calls.append(event)
            session.send(tool_result={event.id: "21°C and sunny in Montreal"})
        elif event.type == "turn_end":
            break
        elif event.type == "error":
            break
        if len(tool_demo_events) > 100:
            break

print("tool calls:", tool_demo_calls)
print("text:", repr("".join(tool_demo_text)))
print("event types:", [event.type for event in tool_demo_events])
_ = play_live_event_audio(tool_demo_events, path="live-tool-audio.wav")
```
```output | ✓ 9.3s | 74 vars
tool calls: [LiveServerToolCallEvent(id='fc_153691463906953224', name='get_weather', input={'city': 'Montreal'}, type='tool_call')]
text: "Right now in Montreal, it's 21°C and sunny."
event types: ['tool_call', 'audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'text', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'audio', 'text', 'audio', 'audio', 'turn_end']
Wrote live-tool-audio.wav
Playing with ffplay...
```

---

## 7. Interrupting a Response

`session.send(interrupt=True)` asks the model to stop the current response. In this run, we interrupt as soon as the first text transcription chunk arrives.

```python
with lm.live(LiveConfig(model=MODEL, system="Speak at normal speed.")) as session:
    session.send(text="Count from 1 to 100, separated by commas.")
    interrupt_events = []
    interrupt_text = []
    sent_interrupt = False

    for event in session:
        interrupt_events.append(event)
        if event.type == "text":
            interrupt_text.append(event.text)
            if not sent_interrupt:
                session.send(interrupt=True)
                sent_interrupt = True
        if event.type in {"interrupted", "turn_end", "error"}:
            break
        if len(interrupt_events) >= 30:
            break

print("sent_interrupt:", sent_interrupt)
print("text prefix:", repr("".join(interrupt_text)[:120]))
print("event_types:", [event.type for event in interrupt_events])
_ = play_live_event_audio(interrupt_events, path="live-interrupt-audio.wav")
```
```output | ✓ 1.0s | 77 vars
sent_interrupt: True
text prefix: '1, 2,'
event_types: ['audio', 'text', 'interrupted']
Wrote live-interrupt-audio.wav
Playing with ffplay...
```

---

## Reference Notes

### Session event types

| `event.type` | Meaning |
|---|---|
| `"text"` | Output transcription or text from the model |
| `"audio"` | Base64-encoded PCM audio chunk |
| `"tool_call"` | Model requested a declared function tool |
| `"interrupted"` | Model acknowledged interruption |
| `"turn_end"` | Current turn is complete; `event.usage` is attached |
| `"error"` | Provider-reported live error |

### Sending data to a live session

| Call | Sends |
|---|---|
| `session.send(text="...")` | Text input |
| `session.send(audio=pcm_bytes)` / `session.send_audio(...)` | Raw/base64 PCM audio input with an explicit audio media type |
| `session.send(image=jpeg_bytes)` / `session.send_image(...)` | One realtime image frame with an explicit image media type |
| `session.send(turn=[image(...), text(...)])` / `session.send_turn(...)` | Ordinary turn content using existing `Part` objects |
| `session.send(tool_result={call_id: result})` / `session.send_tool_result(...)` | Tool result(s) |
| `session.send(interrupt=True)` | Interrupt current generation |
| `session.send(end_audio=True)` | Signal end of an audio stream |

### Practical details observed in this run

- `gemini-3.1-flash-live-preview` returned audio by default, with text transcription chunks alongside the audio.
- The notebook auto-plays audio whenever `collect_turn()`, `summarize_stream()`, or `play_live_event_audio()` sees audio bytes. Set `AUTO_PLAY_AUDIO = False` to disable this.
- Audio deltas used `audio/pcm;rate=24000`. Saving them as WAV just requires a small container wrapper; no extra Python packages are needed.
- For audio input with an explicit sample rate, use `audio(data=..., media_type="audio/pcm;rate=24000")` inside a `Request`.
- Live tool calls are regular typed events. The notebook manually sends `tool_result` rather than relying on a high-level auto-run wrapper.
