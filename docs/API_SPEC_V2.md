# lm15 API v2

## Principles

1. Parts are the universal currency. They come out of responses, they go into prompts.
2. Two tiers: module functions for one-shots, model objects for conversation.
3. Streaming yields high-level events, materializes a full response, and records to history.
4. Callables as tools auto-execute. Tool objects give manual control.
5. Nothing is hidden. Every internal type is importable. None is required.

---

## Quick reference

```python
import lm15
from lm15 import Part, Tool

# text
resp = lm15.complete("gpt-4.1-mini", "Hello.")
resp.text

# stream
for text in lm15.stream("gpt-4.1-mini", "Hello.").text:
    print(text, end="")

# model object
gpt = lm15.model("gpt-4.1-mini", system="You are helpful.")
resp = gpt("Hello.")
```

---

## Module functions

### `lm15.complete`

```python
lm15.complete(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list[Tool | Callable | str] | None = None,
    reasoning: bool | dict | None = None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    provider: str | None = None,
) -> LMResponse
```

- `prompt` is a string or mixed list of strings and parts. Strings become text parts. All wrapped in one user message.
- `messages` is for multi-turn and manual control. Mutually exclusive with `prompt`.
- `tools` accepts callables (auto-schema, auto-execute), `Tool` objects (manual loop), or strings (built-in tools like `"web_search"`).
- `reasoning=True` enables thinking. `reasoning={"effort": "high", "budget": 10000}` for fine control.
- `prefill` seeds the assistant response.
- `output` hints desired output modality: `"image"`, `"audio"`.
- `prompt_caching` enables provider-side prompt caching. See [Prompt caching](#prompt-caching).

### `lm15.stream`

```python
lm15.stream(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    # same kwargs as complete
) -> Stream
```

### `lm15.model`

```python
lm15.model(
    model: str,
    *,
    system: str | None = None,
    tools: list[Tool | Callable | str] | None = None,
    provider: str | None = None,
    retries: int = 0,
    cache: bool | dict = False,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Model
```

### `lm15.upload`

```python
lm15.upload(
    model: str,
    path: str | Path | bytes,
    *,
    media_type: str | None = None,
    provider: str | None = None,
) -> Part
```

Uploads a file via the provider's file API. Returns a `Part` that can be passed directly into prompts. Part type (`image`, `document`, `video`, `audio`) inferred from media type.

```python
doc = lm15.upload("claude-sonnet-4-5", "contract.pdf")
resp = lm15.complete("claude-sonnet-4-5", ["Summarize.", doc])
```

---

## Model

A callable with bound config and conversation history.

### Calling

```python
resp = model(prompt, **kwargs)       # same kwargs as lm15.complete minus model
resp = model(messages=[...])         # multi-turn override
```

Per-call kwargs override bound defaults.

### Streaming

```python
stream = model.stream(prompt, **kwargs)
```

### Upload

```python
file = model.upload("paper.pdf")
resp = model(["Summarize.", file])
```

### Derived models

Return new `Model` instances. Original unchanged.

```python
model.with_model(name) -> Model
model.with_system(system) -> Model
model.with_tools(tools) -> Model
model.with_provider(provider, base_url=None) -> Model
```

### History

```python
model.history -> list[HistoryEntry]   # HistoryEntry has .request, .response
model.history.clear()                 # resets conversation context
```

History is the only mutable state. Every `__call__` and consumed `stream` appends to it. Multi-turn context is built from history automatically.

### Tool result submission

```python
resp = model("Weather?", tools=[weather_tool])
results = {tc.id: "22°C" for tc in resp.tool_calls}
resp = model.submit_tools(results)
```

Reads the last exchange from history, appends tool results, calls the model.

---

## Stream

Returned by `lm15.stream()` and `model.stream()`.

### As text (common case)

```python
for text in stream.text:
    print(text, end="")
```

### As high-level events (full control)

```python
for event in stream:
    match event.type:
        case "thinking":
            print(f"[thought] {event.text}", end="")
        case "text":
            print(event.text, end="")
        case "tool_call":
            print(f"[call {event.name}({event.input})]")
        case "tool_result":
            print(f"[result: {event.text}]")
        case "image":
            save(event.image)
        case "audio":
            play(event.audio)
        case "finished":
            print(f"\n{event.response.usage}")
```

### Materialized response

```python
stream = model.stream("Hello.")
for text in stream.text:
    print(text, end="")
resp = stream.response   # LMResponse — available after stream consumed
print(resp.usage)
```

Accessing `.response` before exhaustion forces consumption.

### StreamChunk

```python
@dataclass
class StreamChunk:
    type: "thinking" | "text" | "tool_call" | "tool_result" | "image" | "audio" | "finished"
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    image: Part | None = None
    audio: Part | None = None
    response: LMResponse | None = None    # only on "finished"
```

### Auto-execute during streaming

When tools are callables, the stream executes them and yields both `tool_call` and `tool_result` events:

```python
stream = model.stream("Weather?", tools=[get_weather])
# yields: tool_call → tool_result → text → finished
```

When tools are `Tool` objects, the stream yields `tool_call` then `finished`. Use `model.submit_tools()` to continue.

---

## LMResponse

### Convenience properties

| Property | Type | Description |
|---|---|---|
| `text` | `str \| None` | Joined text parts. |
| `image` | `Part \| None` | First image part. |
| `images` | `list[Part]` | All image parts. |
| `audio` | `Part \| None` | First audio part. |
| `tool_calls` | `list[Part]` | All tool_call parts. |
| `thinking` | `str \| None` | Joined thinking text. |
| `citations` | `list[Part]` | All citation parts. |

### Fields

| Field | Type |
|---|---|
| `id` | `str` |
| `model` | `str` |
| `message` | `Message` |
| `finish_reason` | `"stop" \| "length" \| "tool_call" \| "content_filter" \| "error"` |
| `usage` | `Usage` |
| `provider` | `dict \| None` |

---

## Part

### Constructors

```python
Part.text_part(text)
Part.image(url=...)
Part.image(data=..., media_type=...)
Part.audio(url=...)
Part.audio(data=..., media_type=...)
Part.video(url=...)
Part.video(data=..., media_type=...)
Part.document(url=...)
Part.document(data=..., media_type=...)
Part.tool_call(id, name, input)
Part.tool_result(id, content, is_error=False)
Part.thinking(text)
Part.refusal(text)
Part.citation(text=None, url=None, title=None)
```

### Cache hint

Any Part can carry a cache hint for provider-side prompt caching:

```python
Part.document(url="...", cache=True)                          # simple
Part.document(url="...", cache={"type": "ephemeral", "ttl": 300})  # full control
```

See [Prompt caching](#prompt-caching).

### Cross-model passing

Parts from responses pass directly into prompts:

```python
resp = lm15.complete("gpt-4.1-mini", "Draw a cat.", output="image")
resp2 = lm15.complete("claude-sonnet-4-5", ["What's this?", resp.image])
```

---

## Tool

### As a Tool object (manual control)

```python
Tool(
    name: str,
    type: "function" | "builtin" = "function",
    description: str | None = None,
    parameters: dict | None = None,
    builtin_config: dict | None = None,
)
```

### As a callable (auto-schema, auto-execute)

```python
def get_weather(city: str) -> str:
    """Get weather by city."""
    return f"22°C in {city}"

resp = lm15.complete("gpt-4.1-mini", "Weather?", tools=[get_weather])
resp.text  # tool was called automatically
```

Schema inferred from type hints and docstring. Name from function name.

### Built-in tools (provider server-side)

```python
resp = lm15.complete("gpt-4.1-mini", "Latest news", tools=["web_search"])
```

String in tools list → `Tool(name=..., type="builtin")`.

---

## Prompt caching

Provider-side prompt caching reduces cost and latency by caching repeated prefixes server-side. Three levels of control.

### Level 1: Automatic (recommended for agent loops)

```python
agent = lm15.model("claude-sonnet-4-5",
    system="<long system prompt>",
    tools=[read_file, write_file, run_command],
    prompt_caching=True,
)

resp = agent("Add tests for the auth module.")
while resp.finish_reason == "tool_call":
    results = execute_tools(resp.tool_calls)
    resp = agent.submit_tools(results)
    print(f"cache hit: {resp.usage.cache_read_tokens} tokens")
```

When `prompt_caching=True`, the adapter automatically places cache boundaries:

- **Anthropic**: adds `cache_control` breakpoints on the system prompt and on the second-to-last message. As the conversation grows, the breakpoint advances so the entire prior history is cached. Each turn only pays for the new message.
- **Gemini**: creates a `CachedContent` object from the system prompt and large content on first call. Updates it as the conversation grows.
- **OpenAI**: no-op. Prefix caching is automatic.

This is the right default for agent loops, RAG over long documents, and any repeated-prefix workflow.

### Level 2: Per-part cache hints

For fine-grained control over what gets cached:

```python
contract = Part.document(data=open("contract.pdf", "rb").read(),
    media_type="application/pdf", cache=True)

resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 1.", contract])
resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 2.", contract])
# second call hits cache on the document
```

Works on any Part — documents, images, long text blocks. The adapter translates `cache=True` to the provider's native mechanism.

### Level 3: Provider-specific control

For TTL, named caches, or provider-specific options:

```python
Part.document(url="...", cache={"type": "ephemeral", "ttl": 300})
```

The dict passes through to the provider adapter raw.

### Cache visibility

Cache usage is always reported in the response:

```python
resp.usage.cache_read_tokens     # tokens served from cache
resp.usage.cache_write_tokens    # tokens written to cache
```

### How automatic caching works in agent loops

```
Turn 1: [system ✎ | user₁]           → cache write on system
Turn 2: [system ✓ | user₁ ✎ | asst₁ | tool₁ | user₂]    → cache hit on system, write on prefix
Turn 3: [system ✓ | user₁ ✓ | asst₁ ✓ | tool₁ ✓ | asst₂ ✎ | tool₂ | user₃]  → cache hit on all prior turns
```

`✓` = cache hit, `✎` = cache breakpoint (written), everything after breakpoint is uncached.

The adapter moves the breakpoint forward each turn. The user never manages this — `prompt_caching=True` is the only knob.

---

## Provider resolution

1. Auto from model name: `claude-*` → anthropic, `gemini-*` → gemini, `gpt-*` → openai.
2. Per-call: `provider="openai"`.
3. On model: `lm15.model("x", provider="openai")`.
4. Custom endpoint: `model.with_provider("openai", base_url="http://localhost:8080/v1")`.

---

## Escape hatches

All internal types remain importable for advanced use:

```python
from lm15 import LMRequest, Config, DataSource, UniversalLM, MiddlewarePipeline, build_default, TransportPolicy
```

For truly custom middleware, transport, or plugin development. Not needed for any of the above examples.

---

## Complete examples

### Coding agent with prompt caching

```python
import lm15

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    return open(path).read()

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    open(path, "w").write(content)
    return f"wrote {len(content)} bytes to {path}"

def run_command(command: str) -> str:
    """Run a shell command."""
    import subprocess
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout

agent = lm15.model("claude-sonnet-4-5",
    system="You are a coding assistant. Read files, make changes, run tests.",
    tools=[read_file, write_file, run_command],
    prompt_caching=True,
    retries=2,
)

# Each turn caches all prior context automatically
resp = agent("Add input validation to the User model in models.py")
while resp.finish_reason == "tool_call":
    results = execute_tools(resp.tool_calls)
    resp = agent.submit_tools(results)

print(resp.text)
print(f"Total turns: {len(agent.history)}")
```

### Streaming with reasoning and tool calls

```python
import lm15

def lookup(topic: str) -> str:
    """Look up a topic."""
    return "The answer is 42."

stream = lm15.stream("claude-sonnet-4-5", "Research and explain quantum computing.",
    tools=[lookup], reasoning=True)

for event in stream:
    match event.type:
        case "thinking":    print(f"💭 {event.text}", end="")
        case "text":        print(event.text, end="")
        case "tool_call":   print(f"\n🔧 {event.name}({event.input})")
        case "tool_result": print(f"📎 {event.text}")
        case "finished":    print(f"\n📊 {event.response.usage}")
```

### Document analysis with caching

```python
import lm15
from lm15 import Part

claude = lm15.model("claude-sonnet-4-5", prompt_caching=True)

contract = Part.document(data=open("contract.pdf", "rb").read(),
    media_type="application/pdf", cache=True)

resp = claude(["Summarize section 1.", contract])     # cache write
resp = claude(["Summarize section 2.", contract])     # cache hit
resp = claude(["Find liability clauses.", contract])  # cache hit

print(f"Cache savings: {resp.usage.cache_read_tokens} tokens read from cache")
```

### Vision pipeline across models

```python
import lm15
from lm15 import Part

gemini = lm15.model("gemini-2.5-flash")
claude = lm15.model("claude-sonnet-4-5")

resp = gemini(["What objects are in this photo?", Part.image(url="https://example.com/room.jpg")])
resp2 = claude(f"Critique this image analysis:\n\n{resp.text}")
print(resp2.text)
```

### Image generation → vision

```python
import lm15

resp = lm15.complete("gpt-4.1-mini", "Draw a cat wearing a top hat.", output="image")
resp2 = lm15.complete("claude-sonnet-4-5", ["Describe this image in detail.", resp.image])
print(resp2.text)
```

### Conversational agent with memory

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", system="You remember everything.")

gpt("My name is Max.")
gpt("I work on developer tools.")
gpt("I like chess and climbing.")

resp = gpt("Write a brief bio about me.")
print(resp.text)

# History tracks everything
for entry in gpt.history:
    print(f"{entry.request.messages[-1].parts[0].text[:50]}...")
```

### Config-driven batch

```python
import lm15

base = {"model": "gpt-4.1-mini", "system": "You are terse.", "temperature": 0}
tasks = [
    {"prompt": "Summarize DNA.", "max_tokens": 50},
    {"prompt": "Summarize RNA.", "max_tokens": 50},
    {"prompt": "Summarize proteins.", "max_tokens": 100},
]

responses = [lm15.complete(**base, **t) for t in tasks]
for r in responses:
    print(r.text)
```
