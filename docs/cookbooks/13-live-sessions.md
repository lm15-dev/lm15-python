# Live sessions (realtime)

**Problem** — You want a bidirectional, low-latency conversation with a
model: send text or microphone audio whenever you like, get audio and
transcription back as it is generated, interrupt mid-sentence. That is
a persistent websocket, not a request/response call, and it needs a
different surface than `complete()`.

Keys loaded as in [recipe 01](01-first-request.md). Live sessions need
the optional `websockets` dependency: `pip install 'lm15[live]'`.

## Recipe

The router resolves the live model like any other Gemini model, but
sessions are opened on the provider LM, not the router — `router.lm()`
hands it to you:

```python
import base64
import os
import wave

from lm15 import LiveConfig, LMRouter, tool

MODEL = "gemini-3.1-flash-live-preview"
router = LMRouter()
print(router.resolve(MODEL))
lm = router.lm(MODEL)
```
```output
'gemini-3.1-flash-live-preview' -> provider 'gemini' (GeminiLM); via built-in rule prefix='gemini-' — Google Gemini family; wire model 'gemini-3.1-flash-live-preview'; key from $GEMINI_API_KEY.
```

`lm.live(LiveConfig(...))` opens the websocket. The session is a
context manager: send with `session.send_text(...)`, then iterate
server events until `turn_end`. An audio-native live model answers a
text prompt with audio chunks plus text events carrying the
transcription of that audio:

```python
with lm.live(LiveConfig(model=MODEL, system="Be concise.")) as session:
    session.send_text("Reply with exactly: live hello")
    pcm, transcript = bytearray(), []
    for event in session:
        if event.type == "audio":
            pcm.extend(base64.b64decode(event.data))
        elif event.type == "text":
            transcript.append(event.text)
        elif event.type in ("turn_end", "error"):
            print("final:", event)
            break

print("transcript:", repr("".join(transcript)))
print("pcm bytes:", len(pcm))
```
```output
final: LiveServerTurnEndEvent(usage=Usage(input_tokens=153, output_tokens=40, …), type='turn_end')
transcript: 'live hello'
pcm bytes: 64830
```

Audio events carry base64 raw PCM, 16-bit mono at 24 kHz. Wrap it in a
WAV container to play or save it — stdlib only:

```python
with wave.open("live-hello.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(bytes(pcm))

print("live-hello.wav", os.path.getsize("live-hello.wav"), "bytes")
```
```output
live-hello.wav 64874 bytes
```

Audio goes the other way too. `session.send_audio()` streams microphone
(here: the PCM we just generated) over the realtime input channel;
`end_audio()` marks the stream done and voice activity detection turns
it into a turn. No text prompt needed:

```python
with lm.live(LiveConfig(model=MODEL, system="Repeat back exactly the words you hear.")) as session:
    session.send_audio(bytes(pcm), media_type="audio/pcm;rate=24000")
    session.end_audio()
    heard = []
    for event in session:
        if event.type == "text":
            heard.append(event.text)
        elif event.type in ("turn_end", "error"):
            break

print("heard:", repr("".join(heard)))
```
```output
heard: 'live hello'
```

Function tools work in live sessions. Derive one with `tool()`, pass it
in `LiveConfig`, answer `tool_call` events with `send_tool_result()` —
you run the function, exactly as in recipe
[06](06-function-tools.md):

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"21°C and sunny in {city}"

weather = tool(get_weather)

cfg = LiveConfig(model=MODEL, system="Use tools when useful. Be concise.", tools=(weather,))
with lm.live(cfg) as session:
    session.send_text("What is the weather in Montreal? Use the tool.")
    answer = []
    for event in session:
        if event.type == "tool_call":
            print(event)
            session.send_tool_result({event.id: get_weather(**event.input)})
        elif event.type == "text":
            answer.append(event.text)
        elif event.type in ("turn_end", "error"):
            break

print("answer:", repr("".join(answer)))
```
```output
LiveServerToolCallEvent(id='fc_7965…', name='get_weather', input={'city': 'Montreal'}, type='tool_call')
answer: '21°C and sunny in Montreal.'
```

Interruption is the point of realtime. `session.interrupt()` (barge-in)
stops the current response; the server acknowledges with an
`interrupted` event instead of `turn_end`:

```python
with lm.live(LiveConfig(model=MODEL)) as session:
    session.send_text("Count from 1 to 100, separated by commas.")
    spoken, seen = [], []
    for event in session:
        seen.append(event.type)
        if event.type == "text":
            spoken.append(event.text)
            session.interrupt()
        if event.type in ("interrupted", "turn_end", "error"):
            break

print("text before interrupt:", repr("".join(spoken)))
print("event types:", seen)
```
```output
text before interrupt: '1, 2,'
event types: ['audio', 'text', 'audio', 'interrupted']
```

## How it works

`lm.live()` opens a websocket to Gemini's BidiGenerateContent endpoint
(the only provider with live support in lm15 today), sends the setup
frame built from `LiveConfig`, and returns a `WebSocketLiveSession`.
The session is a thin codec: `send_*` methods encode typed
`LiveClient*Event`s to wire JSON, iteration decodes wire frames into
typed `LiveServer*Event`s — `text`, `audio`, `tool_call`,
`interrupted`, `turn_end` (with `usage`), `error`. State lives
server-side: the session remembers earlier turns, so you never resend
the transcript.

lm15 deliberately stops there. No audio capture or playback, no
reconnection, no voice-activity logic of its own, no tool execution —
the `tool_call`/`send_tool_result` loop above is yours, same as the
non-live dispatch loop in [tools from
functions](../tools-from-functions.md). Model resolution is the
ordinary router walk described in [using the
router](../using-the-router.md); `live()` itself takes the wire model
in `LiveConfig`.

## Variations

- **One-shot live completion.** `router.stream(Request(...))` with a
  `-live` model routes through the same websocket and yields ordinary
  `StreamEvent`s — `AudioDelta` plus `TextDelta` — so recipe
  [05](05-streaming.md) code works unchanged for a single turn. Text
  input only today: audio input on this path crashes in the Gemini
  provider's usage accounting (`TypeError` on `None` tokens).
- **Multi-turn memory.** Send a second `send_text()` on the same
  session after `turn_end`; context carries over server-side.
- **Realtime image frames.** `session.send_image(jpeg_bytes,
  media_type="image/jpeg")` feeds camera/screen frames over the
  realtime channel (Gemini's wire name for the frame stream is
  `realtimeInput.video`). Frames are only attached to the next turn
  when the session sets
  `extensions={"realtimeInputConfig": {"turnCoverage":
  "TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO"}}`.
- **Mixed turns.** `session.send_turn((image(...), text(...)))` sends
  ordinary `Part` content as one turn — prompt content, not the
  realtime channel. Audio parts are the exception: Gemini rejects
  inline audio in live turn content; use `send_audio()` as above.
- **Async.** `AsyncGeminiLM.live()` raises `UnsupportedFeatureError`
  ("use the sync adapter") — async live is planned, not shipped. The
  sync session blocks on `recv()`; put it on a thread if you need an
  event loop alongside it.
- **Knobs.** `LiveConfig` also takes `voice`, `input_format`/
  `output_format` (`AudioFormat`), and provider-specific `extensions`.
  Usage arrives once per turn on the `turn_end` event, not per chunk.

## See also

- [05 — Streaming](05-streaming.md) — the one-shot stream surface the live transport reuses.
- [06 — Function tools](06-function-tools.md) — the same dispatch loop, non-live.
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md) — request/response audio.
- [Using the router](../using-the-router.md) — resolution and `router.lm()`.
- [Tools from functions](../tools-from-functions.md) — `tool()` derivation rules.
