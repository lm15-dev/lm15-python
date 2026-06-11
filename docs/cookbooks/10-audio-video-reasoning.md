# Audio, video & reasoning models

**Problem** — You want to send sound and video to a model, and you want
models that think before they answer — with the thinking visible, capped,
and billed transparently. Provider support is uneven: lm15 gives you one
`audio()`/`video()` part and one `Reasoning` config, and is explicit
about where each one lands.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

Audio input is a part, like an image. `audio(path=...)` reads the file,
base64-encodes it, and infers the media type. Gemini is the provider
that accepts it. This block synthesizes a beep WAV with the stdlib so
the page has no asset dependency:

```python
import math, struct, urllib.request, wave

from lm15 import LMRouter, Message, Request
from lm15.types import Config, Reasoning, ThinkingPart, audio, text, video

with wave.open("beeps.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    for freq in (440, 660, 880):
        for i in range(8000):
            w.writeframes(struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq * i / 16000))))

router = LMRouter()
response = router.complete(Request(
    model="gemini-3-flash-preview",
    messages=(Message.user((
        text("Describe this audio clip in one sentence."),
        audio(path="beeps.wav"),
    )),),
))
print(response.text)
print(response.usage)
```
```output
A series of high-pitched electronic beeps rings out in rhythmic bursts,
sounding like a digital alarm or notification.
Usage(input_tokens=47, output_tokens=23, total_tokens=438, …, reasoning_tokens=368, …)
```

Note `reasoning_tokens=368`: Gemini 3 thinks by default, even about
beeps.

Video is the same shape. `video(path=...)` inlines the bytes; keep clips
small (inline base64, no chunked upload):

```python
urllib.request.urlretrieve("https://www.w3schools.com/html/mov_bbb.mp4", "clip.mp4")
response = router.complete(Request(
    model="gemini-3-flash-preview",
    messages=(Message.user((
        text("What happens in this clip? Two sentences."),
        video(path="clip.mp4"),
    )),),
))
print(response.text)
print(response.usage)
```
```output
A large, white rabbit enjoys playing with butterflies in a scenic grassy
field. As he stands up and begins to sing, a red apple falls from the
sky and hits him on the head.
Usage(input_tokens=920, output_tokens=38, …, reasoning_tokens=195, …)
```

Now reasoning. `Config(reasoning=Reasoning(...))` turns thinking on;
`thinking_budget` caps it in tokens. Anthropic returns the trace itself
as a `ThinkingPart` in the message, before the answer text:

```python
riddle = "A bat and a ball cost $1.10; the bat costs $1.00 more than the ball. What does the ball cost?"
response = router.complete(Request(
    model="claude-sonnet-4-5",
    messages=(Message.user(riddle),),
    config=Config(reasoning=Reasoning(effort="medium", thinking_budget=2048)),
))
trace = response.message.first(ThinkingPart)
print(trace.text.splitlines()[0])
print("—")
print(response.text.splitlines()[0])
print(response.usage)
```
```output
Let me set up equations for this problem.
—
Looking at this problem, I need to set up equations.
Usage(input_tokens=66, output_tokens=474, total_tokens=540, …, reasoning_tokens=None, …)
```

`reasoning_tokens=None` is deliberate: Anthropic bills thinking inside
`output_tokens` and reports no separate count, so lm15 reports none.

OpenAI is the inverse. It never exposes raw chain-of-thought; it counts
`reasoning_tokens` exactly, and on request (`summary=`) returns a
provider-written summary — parsed as a `ThinkingPart` too:

```python
response = router.complete(Request(
    model="gpt-5.4-mini",
    messages=(Message.user(riddle),),
    config=Config(reasoning=Reasoning(effort="medium", summary="auto")),
))
summary = response.message.first(ThinkingPart)
print(summary.text.splitlines()[0])
print("—")
print(response.text)
print(response.usage)
```
```output
**Explaining the classic trick**
—
The ball costs **$0.05**.

Quick check:
- Ball = $0.05
…
Usage(input_tokens=35, output_tokens=69, total_tokens=104, …, reasoning_tokens=17, …)
```

The off switch matters for models that think by default. On Gemini,
`Reasoning(effort="off")` maps to `thinkingBudget: 0`:

```python
response = router.complete(Request(
    model="gemini-3-flash-preview",
    messages=(Message.user(riddle),),
    config=Config(reasoning=Reasoning(effort="off")),
))
print(response.text.splitlines()[0])
print(response.usage)
```
```output
The ball costs **5 cents** ($0.05).
Usage(input_tokens=32, output_tokens=79, total_tokens=111, …, reasoning_tokens=None, …)
```

Compare with the audio call above: 368 reasoning tokens there, none
here. (And the model still dodges the trap — it is a famous riddle.)

Thinking streams. `ThinkingDelta` events arrive before `TextDelta`
events, distinguished by `delta.type`:

```python
req = Request(
    model="claude-sonnet-4-5",
    messages=(Message.user(riddle),),
    config=Config(reasoning=Reasoning(effort="medium", thinking_budget=1024)),
)
for event in router.stream(req):
    if event.type == "delta" and event.delta.type in ("thinking", "text"):
        print(event.delta.type, repr(event.delta.text[:48]))
    elif event.type == "end":
        print("end", event.usage)
```
```output
thinking 'Let me set'
thinking " up equations for this problem.\n\nLet's say:\n- Th"
…
thinking '✓\n- Bat costs $1.00 more than ball: $1.05 - $0.0'
text 'The ball costs **'
text "$0.05** (5 cents).\n\nHere's why:\n\nLet me call the"
…
end Usage(input_tokens=66, output_tokens=454, total_tokens=520, …)
```

## How it works

`audio()` and `video()` are factories for `AudioPart` and `VideoPart` —
frozen dataclasses holding base64 `data` (or a `url`/`file_id`) plus a
media type, inferred from the file extension when you pass `path=`. They
go in a user message's parts tuple next to `text(...)`; nothing is
uploaded out of band. Gemini maps both to `inlineData`. OpenAI's
Responses mapping emits `input_audio` for `AudioPart` — but only its
audio-native models accept it, and there is no `VideoPart` mapping at
all. Anthropic accepts neither (see Variations for the real error).

`Reasoning` lives on `Config`. `effort` is a portable ladder — `"off"`,
`"adaptive"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` —
and each provider takes the part it understands. `thinking_budget` is a
hard token cap; `total_budget` caps thinking plus visible output (on
Anthropic it becomes `max_tokens`, and must exceed `thinking_budget` —
lm15 raises `ValueError` otherwise). `summary` is OpenAI-only.
`Reasoning(effort="off")` with a budget or summary is a `ValueError` at
construction: lm15 refuses to silently discard config.

On the way back, thinking appears as `ThinkingPart` in
`response.message.parts` — a real trace (Anthropic, Gemini) or a
summary (OpenAI). `message.first(ThinkingPart)` returns the first or
`None`; `message.parts_of(ThinkingPart)` returns all.
`usage.reasoning_tokens` is populated when the provider reports a count
(OpenAI, Gemini) and stays `None` when it does not (Anthropic). lm15
does not estimate.

## Variations

- **Async mirror.** `AsyncLMRouter().complete(...)` is awaitable and
  `stream()` returns an async iterator; the part factories and
  `Reasoning` are identical.
- **Effort ladder per provider.** OpenAI maps the ladder directly
  (`adaptive`→`medium`, `xhigh`→`high` where unsupported). Anthropic
  ignores the rung: any non-off effort enables thinking with
  `thinking_budget` (default 1024 if you give none). Gemini uses on/off
  plus `thinking_budget`; the rung itself is not sent.
- **Low effort may mean no summary.** At `effort="low"` OpenAI
  sometimes skips the summary entirely; `message.first(ThinkingPart)`
  returns `None`. Check before dereferencing.
- **Unsupported media fails loudly, not silently.** Sending
  `audio(...)` to `claude-sonnet-4-5` raises
  `InvalidRequestError: messages: text content blocks must be non-empty`
  (real capture) — Anthropic has no audio block type. `gpt-4.1-mini`
  accepts the request but cannot hear: it answers that it is unable to
  listen to audio files.
- **Gemini streams thoughts unreliably at small budgets.** With
  `thinking_budget=512` we observed runs with and without `thinking`
  deltas, while `usage.reasoning_tokens` was nonzero either way.
  Anthropic streams its trace consistently.
- **Redacted thinking.** Anthropic may return encrypted traces;
  these surface as `ThinkingPart(redacted=True)` with the signature
  preserved in `continuation` so multi-turn requests replay them intact.

## See also

- [05 — Streaming](05-streaming.md) — `Result.events()` and typed deltas.
- [09 — Images, PDFs & documents](09-images-and-documents.md) — the
  other media parts, with broader provider support.
- [13 — Live sessions (realtime)](13-live-sessions.md) — audio out, not
  just in.
- [../using-the-type-system.md](../using-the-type-system.md) — parts,
  `Reasoning`, and `Usage` in full.
- [../mapping-rules.md](../mapping-rules.md) — exactly what each
  provider receives on the wire.
