# Using a provider LM

Provider LMs translate between the provider-independent `lm15.types`
vocabulary and one provider's HTTP or WebSocket API. They are the layer above
`lm15.transports` and below higher-level client/result helpers.

The built-in LMs are:

```python
from lm15.providers import (
    AnthropicLM, ClaudeCodeLM, GeminiLM, OpenAIChatLM, OpenAICodexLM,
    OpenAILM, XaiLM,
)
```

`OpenAIChatLM` speaks the Chat Completions dialect (and, via compat
presets, Groq, OpenRouter, ollama, vLLM, and SGLang); `ClaudeCodeLM`,
`OpenAICodexLM`, and `XaiLM` add subscription-credential handling on
top of the Anthropic and Chat Completions dialects. Each LM exposes
the same endpoint-oriented surface:

```text
complete(Request) -> Response
stream(Request) -> Iterator[StreamEvent]
live(LiveConfig) -> LiveSession
file_upload(FileUploadRequest) -> FileInfo
file_get(file_id) -> FileInfo
file_list(limit, cursor) -> FilePage
file_delete(file_id) -> None
file_download(file_id) -> bytes
batch(requests) -> BatchJob            # plus batch_job(id), batches()
image_generate(ImageGenerationRequest) -> ImageGenerationResponse
speech_generate(SpeechGenerationRequest) -> SpeechGenerationResponse
video_generate(VideoGenerationRequest) -> VideoJob   # plus video_job(id), video_jobs()
list_models() -> tuple[ModelInfo, ...]
```

Unsupported endpoints raise `UnsupportedFeatureError`.

## Create an LM

LMs need an API key. If you do not pass a transport, the LM creates a
`StdlibTransport` by default.

```python
from lm15.providers import OpenAILM

lm = OpenAILM(api_key="sk-...")
```

Use the LM as a context manager when you want its default transport closed for
you. If you need to share a connection pool across several LMs, pass an explicit
long-lived transport.

```python
from lm15.providers import AnthropicLM, OpenAILM
from lm15.transports import StdlibTransport

with OpenAILM(api_key="sk-...") as lm:
    ...

with StdlibTransport(max_connections=10) as transport:
    openai = OpenAILM(api_key="sk-...", transport=transport)
    anthropic = AnthropicLM(api_key="sk-ant-...", transport=transport)
    ...
```

## Local and OpenAI-compatible endpoints

Name the server; the preset supplies its wire policy and its address:

```python
from lm15.providers import OpenAILM

lm = OpenAILM(api_key="ollama", compat="ollama")                       # http://localhost:11434/v1
lm = OpenAILM(api_key="ollama", compat="ollama", base_url="http://gpu-box:11434/v1")
```

A preset name lm15 has no address for in the chosen dialect is refused with
`NotConfiguredError` until you pass `base_url=`; it is never sent to the
OpenAI cloud. (`ProviderProfile` / `OpenAILM.from_profile`, the older way to
say the same, are deprecated and removed in 1.0.0.)

See [Using model profiles and compatibility policies](using-model-profiles.md)
for model metadata, registries, presets, and request-level escape hatches.

## Make a complete call

LMs consume `lm15.types.Request` and return `lm15.types.Response`.

```python
from lm15.providers import OpenAILM
from lm15.types import Config, Message, Request

with OpenAILM(api_key="sk-...") as lm:
    request = Request(
        model="gpt-4.1-mini",
        messages=(Message.user("Write one sentence about TCP."),),
        config=Config(max_tokens=100, temperature=0.2),
    )

    response = lm.complete(request)

print(response.text)
print(response.usage.total_tokens)
```

`response.provider_data` contains the raw provider response dictionary when the
LM has one. Keep application logic on the typed fields when possible and
use provider data only for provider-specific diagnostics or features.

## Stream typed events

`lm.stream()` yields typed stream events, not provider JSON dictionaries.

```python
from lm15.types import StreamDeltaEvent, StreamEndEvent, TextDelta

for event in lm.stream(request):
    if isinstance(event, StreamDeltaEvent) and isinstance(event.delta, TextDelta):
        print(event.delta.text, end="", flush=True)
    elif isinstance(event, StreamEndEvent):
        print("\nfinish:", event.finish_reason)
```

Stream errors are emitted as `StreamErrorEvent` when the provider reports an
in-band streaming error. HTTP errors before the stream starts are raised as
provider exceptions such as `AuthError`, `RateLimitError`, or `ServerError`.

## Tools

Function tools are provider-independent. The LM maps them to the provider's
wire shape.

```python
from lm15.types import Config, FunctionTool, Message, Request, ToolChoice

weather = FunctionTool(
    name="weather",
    description="Get weather for a city.",
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
    config=Config(tool_choice=ToolChoice.from_tools(weather)),
)

response = lm.complete(request)
for call in response.tool_calls:
    print(call.id, call.name, call.input)
```

Tool results go back as `Message.tool(...)` in a follow-up `Request`.

```python
from lm15.types import Message

follow_up = Request(
    model=request.model,
    messages=request.messages + (
        response.message,
        Message.tool(response.tool_calls[0].id, "19 C and cloudy"),
    ),
    tools=request.tools,
    config=request.config,
)
```

Builtin tools are declared with `BuiltinTool`. The LM maps canonical names
like `web_search` and `code_execution` to each provider's current native tool
identifier where one is known.

```python
from lm15.types import BuiltinTool

request = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Search for the latest release notes."),),
    tools=(BuiltinTool("web_search"),),
)
```

## Provider-specific options

Universal knobs live in `Config`. Provider-specific options go in
`Config.extensions` and are passed through by LMs after reserving a few
lm15 keys such as `prompt_caching`, `output`, and `transport`.

```python
from lm15.types import Config, Reasoning

request = Request(
    model="claude-3-7-sonnet-latest",
    messages=(Message.user("Think carefully, then answer."),),
    config=Config(
        max_tokens=1000,
        reasoning=Reasoning(effort="medium", thinking_budget=1024),
        extensions={
            "prompt_caching": True,
            "metadata": {"user_id": "user-123"},
        },
    ),
)
```

Common extension keys used by built-in LMs:

- `prompt_caching`: enables LM-specific prompt/cache wiring where
  supported.
- `output`: set to `"image"` or `"audio"` for providers that use chat/generate
  endpoints for non-text output.
- `transport`: set to `"live"`, `"websocket"`, or `"ws"` to force live
  WebSocket completion for OpenAI/Gemini live-capable models.

## Files, images, and audio

Endpoint-specific request types also use the same LM surface. A stored
file's `id` goes straight into a media Part's `file_id`; the provider is
the system of record, so a lost id is recovered with `file_list`, never
with client-side bookkeeping.

```python
from lm15.types import FileUploadRequest

info = lm.file_upload(
    FileUploadRequest(filename="notes.txt", bytes_data=b"hello", media_type="text/plain")
)
info = lm.file_wait_ready(info.id)   # only Gemini uploads are ever pending
lm.file_list(limit=10)               # the queue remembers
lm.file_delete(info.id)              # returning without an exception IS the confirmation
```

Download works only for files the provider marks downloadable (for
example tool-generated output); other files raise the provider's typed
refusal.

Generated media responses return typed media parts.

```python
from lm15.types import ImageGenerationRequest

result = lm.image_generate(
    ImageGenerationRequest(model="gpt-image-1", prompt="A small red cube")
)
image_bytes = result.images[0].bytes
```

## Error normalization

LMs convert provider-specific error shapes into the canonical lm15 error
hierarchy:

```python
from lm15.errors import AuthError, RateLimitError, ServerError

try:
    lm.complete(request)
except AuthError:
    print("check API key")
except RateLimitError:
    print("retry later")
except ServerError:
    print("provider-side issue")
```

The structured stream equivalent is `ErrorDetail`, carried by
`StreamErrorEvent`.

## Inspect the exact HTTP request

For tests and fixtures, call `build_request()` directly. It returns a
transport-level `lm15.transports.Request` with bytes ready to send.

```python
http_request = lm.build_request(request, stream=False)
print(http_request.method)
print(http_request.url)
print(http_request.headers)
print(http_request.body.decode("utf-8"))
```

This is the easiest way to snapshot provider wire shapes without doing network
I/O.

## Test an LM with a fake transport

A provider LM only needs a transport object with `stream(request)` returning
a context-managed response — and lm15 ships that double in `lm15.testing`,
so you do not write it yourself:

```python
from lm15.testing import FakeResponse, FakeTransport

fake = FakeTransport([FakeResponse(200, b'{"id":"r","output":[]}')])
lm = OpenAILM(api_key="test", transport=fake)
```

The fake records every `TransportRequest` in `fake.requests`, so one
fixture asserts both behavior and wire format. `lm15.testing.FakeLM` is
the canonical-level double (script `Response` objects, no wire JSON) —
see [recipe 17](cookbooks/17-errors-and-testing.md).

## Implement a new provider

Subclass `BaseProviderLM` and implement the four core translation methods:

```python
from lm15.providers import BaseProviderLM

class MyLM(BaseProviderLM):
    provider = "my-provider"

    def build_request(self, request, stream):
        ...  # Request -> lm15.transports.TransportRequest

    def parse_response(self, request, response):
        ...  # buffered HTTP response -> lm15.types.Response

    def parse_stream_events(self, request, raw_event):
        ...  # SSEEvent -> iterator of StreamEvent

    def normalize_error(self, status, body):
        ...  # HTTP error -> ProviderError
```

Guidelines:

- Normalize provider quirks before constructing `lm15.types` objects.
- Return typed parts and deltas, never provider dictionaries.
- Put raw provider telemetry in `provider_data`.
- Keep provider-only request options in `Config.extensions`.
- Let the transport handle sockets, TLS, pooling, and streaming bytes.
