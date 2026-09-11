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
beeps. And note the arithmetic: 47 + 23 + 368 = 438. On Gemini the
thinking is *outside* `output_tokens`; on OpenAI and Anthropic it is
*inside*. lm15 reports every counter exactly as the provider does, so the
numbers reconcile with the bill; the per-provider rules are in the
contract's `spec/types.md` under Usage.

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

Every adapter puts the explicit off on the wire in its provider's native
form — or fails loudly (mapping rule MAP-5). OpenAI gets
`reasoning: {"effort": "none"}` (gpt-5.1 honors it with zero reasoning
tokens; older models whose floor is `"minimal"` reject it with a clear
400). Chat-dialect servers with `reasoning_effort` (Groq, vLLM, SGLang)
get `reasoning_effort: "none"`; OpenRouter gets
`reasoning: {"enabled": false}`; DeepSeek, Qwen and Z.AI get their
native disable fields. Anthropic sends nothing — thinking is opt-in
there, so absence is the off switch. xAI raises
`UnsupportedFeatureError`: Grok reasoning models have no off switch, and
the server silently ignores disable-shaped fields. What never happens is
a silent no-op that bills you for reasoning you turned off.

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

## Groq/Qwen: keep thinking out of the answer

Groq can return Qwen reasoning as literal `<think>...</think>` text in
`message.content`. lm15 preserves that text; it never guesses which tags
are reasoning rather than text the user asked the model to quote.

Ask Groq to separate the fields instead. On `qwen/qwen3.6-27b`, use the
existing provider extension, without setting a reasoning-effort level:

```python
from lm15 import Config, LMRouter, Message, Request, ResponseStream, ThinkingPart

router = LMRouter()  # GROQ_API_KEY
request = Request(
    model="groq:qwen/qwen3.6-27b",
    messages=(Message.user("What is 17 times 23?"),),
    config=Config(extensions={"reasoning_format": "parsed"}),
)
response = router.complete(request)
print(response.text)  # answer text, not the separate reasoning field
reasoning = response.message.parts_of(ThinkingPart)

# A second request, this time streamed. Text iteration excludes ThinkingDelta.
with ResponseStream(router.stream(request), request) as stream:
    for chunk in stream:
        print(chunk, end="")
    completed = stream.response
```

Both calls can incur provider charges. The model id is a documented example,
not a promise that Groq will keep serving it; check your account's model list.

`parsed` changes the *format*, not the amount of reasoning or its price.
Groq's `message.reasoning` / `delta.reasoning` becomes `ThinkingPart` /
`ThinkingDelta`; answer `content` stays text. A literal `<think>` in answer
content still stays literal. Qwen 3.6 accepts `none|default` for its effort
parameter, not the usual `low|medium|high` ladder, so do not add a guessed
`Reasoning(effort="medium")` merely to request separation. To disable
reasoning on a model that supports it, use `Reasoning(effort="off")` instead.

This is the existing policy in contract MAP-7 rules 7 and 12 and the
[reasoning decision's Groq amendment](https://github.com/lm15-dev/lm15-contract/blob/main/changes/2026-09-02-reasoning-design.md#amendments-during-implementation-2026-09-02).
The Groq-wide default is deliberately unchanged: a Qwen-specific option
must not silently become a request parameter for unrelated models. See
[Groq's reasoning formats](https://console.groq.com/docs/reasoning).

## How it works

`audio()` and `video()` are factories for `AudioPart` and `VideoPart` —
frozen dataclasses holding base64 `data` (or a `url`/`file_id`) plus a
media type, inferred from the file extension when you pass `path=`. They
go in a user message's parts tuple next to `text(...)`; nothing is
uploaded out of band. Gemini maps both to `inlineData`. OpenAI's
Responses mapping emits `input_audio` for `AudioPart` — but only its
audio-native models accept it, and there is no `VideoPart` mapping at
all. Anthropic accepts neither (see Variations for the real error).

`Reasoning` lives on `Config`. `effort` is the one dial and is required
— `"off"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`,
`"max"`. Every provider has this dial with these words; each model
accepts a subset and rejects the rest with a clear 400, and lm15 raises
where a word has no native level rather than downgrade it. Leaving
`reasoning` unset lets the model decide, on every provider.
`thinking_budget` is a token cap on providers that count thinking
separately (Anthropic's 4.5 line, Gemini); on providers without one it
raises. `summary="auto"` asks to see the thinking where a knob exists.
`Reasoning(effort="off")` with a budget or summary is a `ValueError` at
construction: lm15 refuses to silently discard config. The full
per-provider table is MAP-7 in [mapping rules](../mapping-rules.md).

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
- **Hidden thinking.** Anthropic may return encrypted traces
  (`redacted_thinking`); these surface as a `ThinkingPart` with empty
  `text` and the blob preserved in `continuation`
  (`anthropic:redacted_thinking`) so multi-turn requests replay them
  intact. There is no flag and no placeholder text: hidden thinking is
  empty text plus replay state, the same shape as an OpenAI reasoning
  item with no summary (MAP-7 rules 9 and 11).

## See also

- [05 — Streaming](05-streaming.md) — `ResponseStream.events()` and typed deltas.
- [09 — Images, PDFs & documents](09-images-and-documents.md) — the
  other media parts, with broader provider support.
- [15 — Live sessions (realtime)](15-live-sessions.md) — audio out, not
  just in.
- [../using-the-type-system.md](../using-the-type-system.md) — parts,
  `Reasoning`, and `Usage` in full.
- [../mapping-rules.md](../mapping-rules.md) — exactly what each
  provider receives on the wire.
