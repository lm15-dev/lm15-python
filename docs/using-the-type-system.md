# Using the lm15 type system

`lm15.types` is the provider-independent vocabulary used by LMs, streams,
tools, and serialization. The main shape is:

```text
Part -> Message -> Request -> Response
```

A **Part** is one typed block of content. A **Message** is one speaker's list of
parts. A **Request** is the full model call. A **Response** is the assistant
message plus finish reason, usage, and optional provider metadata.

## Build messages from parts

For normal prompts, use the role constructors. Plain strings become `TextPart`s.

```python
from lm15.types import Message, Request, image

messages = (
    Message.user("Describe this image."),
    Message.user([
        image(url="https://example.com/cat.png", detail="auto"),
        image(path="./diagram.png"),
    ]),
)

request = Request(
    model="gpt-4.1-mini",
    messages=messages,
    system="Be concise.",
)
```

Media factories are available for `image()`, `audio()`, `video()`, `document()`,
and `binary()`. Each media part must have exactly one source: `data`, `url`,
`file_id`, or `path`. Bytes passed to `data=` are base64-encoded for you.

```python
from lm15.types import audio, binary

clip = audio(data=b"...wav bytes...", media_type="audio/wav")
blob = binary(path="./archive.zip", media_type="application/zip")

raw_bytes = clip.bytes  # works for inline data or path-backed media
```

## Roles enforce valid content

The constructors keep protocol-only parts out of the wrong roles:

- `Message.user(...)` and `Message.developer(...)` are for prompt content.
- `system=` accepts text or prompt parts.
- `Message.assistant(...)` is for model output, including text, media,
  thinking traces, tool calls, refusals, and citations.
- `Message.tool(...)` may contain only tool results.

```python
from lm15.types import Message, tool_result

tool_msg = Message.tool({
    "call_123": "The current temperature is 19 C.",
})

same_tool_msg = Message.tool(
    tool_result("call_123", "The current temperature is 19 C.", name="weather")
)
```

## Configure a request

Universal generation knobs live in `Config`. Provider-specific options belong in
`extensions`, which must be JSON-compatible.

```python
from lm15.types import Config, FunctionTool, Message, Request, ToolChoice

weather = FunctionTool(
    name="weather",
    description="Get the weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

request = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Weather in Paris?"),),
    tools=(weather,),
    config=Config(
        max_tokens=500,
        temperature=0.2,
        tool_choice=ToolChoice.from_tools(weather),
        extensions={"provider_option": True},
    ),
)
```

`parameters` is opaque JSON Schema, always written by you or derived for
you — `lm15.tool(fn)` produces the same `FunctionTool` from a typed
Python function; see [Tools from functions](tools-from-functions.md).

Reasoning is explicit and optional: `Config(reasoning=None)` means "do not send
a preference"; `Config(reasoning=Reasoning())` explicitly turns reasoning off.

```python
from lm15.types import Config, Reasoning

config = Config(reasoning=Reasoning(effort="medium", thinking_budget=1024))
```

## Read responses by variant

`Response.text` is only set when the assistant message is pure text. For mixed
content, use `message.first()` and `message.parts_of()` with concrete part
classes.

```python
from lm15.types import CitationPart, ImagePart, TextPart

text = response.text  # str | None
first_image = response.message.first(ImagePart)
all_citations = response.message.parts_of(CitationPart)
all_text_parts = response.message.parts_of(TextPart)
```

For JSON responses, `response.parse_json()` parses exact JSON text and raises a
helpful error on failure. `response.json` returns the parsed value or `None`.

## Streaming uses typed events

Streams are represented as `StreamEvent`s:

- `StreamStartEvent`
- `StreamDeltaEvent(delta=...)`
- `StreamEndEvent`
- `StreamErrorEvent`

Deltas are also typed (`TextDelta`, `ImageDelta`, `ToolCallDelta`, etc.). The
helpers in `lm15.result` convert between complete responses and streams.

```python
from lm15.result import materialize_response, response_to_events

for event in response_to_events(response):
    ...

response = materialize_response(events, request)
```

Not every part is streamable. Text, thinking, image, audio, tool calls, and
citations have delta variants. Video, document, binary, tool result, and refusal
parts have no delta representation; they can appear in prompts or final
materialized messages but cannot be emitted incrementally.

## Serialize when crossing process boundaries

Use `lm15.serde` for the canonical JSON-shaped dictionaries:

```python
from lm15.serde import request_from_dict, request_to_dict

payload = request_to_dict(request)
request2 = request_from_dict(payload)
```

Canonical JSON is the portable interchange format used by the conformance
fixtures and future language ports.

## Validation model

The dataclasses are frozen and slotted. Constructors validate the invariants that
make objects meaningful: required identities, legal literal values, role/content
compatibility, one media source per media part, non-negative token counts, and
strict JSON-compatible config/metadata. LMs should normalize provider quirks
before constructing these types.
