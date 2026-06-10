# lm15-python2

`lm15-python2` is the Python reference implementation of **lm15**: a small,
typed, provider-neutral interface for foundation-model requests, responses,
streams, tools, media parts, endpoint APIs, errors, and canonical JSON
serialization.

**What lm15 is — and deliberately is not.** lm15 is a low-level foundation
library: one canonical representation, exact serde for it, and adapters that
translate it to and from each provider's wire format — stdlib-only, with its
own HTTP transport (`websockets` is the single optional extra, for live
sessions). It is NOT an opinionated user-facing API: no magic `call()`, no
automatic tool loops, no DSL. lm15 is meant to be **the dependency** for
libraries that want to build their own take on the right way to talk to AI
systems in Python — you bring the opinions, lm15 brings every provider.

The public API is the top-level package: `from lm15 import AnthropicLM,
Request, Message, ...` (see `lm15/__init__.py` for the full curated surface).
Transport plumbing stays under `lm15.transports`, live sessions under
`lm15.live`, and the conformance shim under `lm15.vet`.

Behavior is pinned by the `lm15-contract` corpus (sibling repo): this package
is the reference implementation, not the spec.

## Install for local development

From this directory:

```bash
python3 -m pip install -e .
# Optional extras:
python3 -m pip install -e '.[live]'
```

Run the tests:

```bash
pytest -q
python3 conformance/run_all.py --strict
```

## Repository layout

```text
lm15-python2/
├── lm15/
│   ├── types.py              # canonical dataclasses: Request, Response, Parts, tools, endpoints
│   ├── providers/            # OpenAI, Anthropic, Gemini adapters
│   ├── compat.py             # typed API-dialect compatibility policies
│   ├── models.py             # optional ModelInfo metadata + ModelRegistry
│   ├── profiles.py           # ProviderProfile/EndpointProfile resolution helpers
│   ├── result.py             # stream materialization + lazy Result helper
│   ├── serde.py              # canonical JSON dictionaries
│   ├── errors.py             # normalized lm15 error hierarchy
│   ├── live.py               # websocket/live-session helpers
│   ├── sse.py                # server-sent event parser
│   └── transports/           # stdlib HTTP/1.1 sync + async transports
├── conformance/              # fixture suite and reports
├── tests/                    # unit + conformance tests wired into pytest
├── benchmarks/               # transport benchmarks
└── pyproject.toml
```

## Mental model

The central object flow is:

```text
Message parts → Message → Request → ProviderLM → Response
                              │
                              └── stream() → StreamEvent → Result/materialized Response
```

- **Parts** are typed content blocks: `TextPart`, `ImagePart`, `AudioPart`,
  `DocumentPart`, `ToolCallPart`, `ToolResultPart`, `ThinkingPart`,
  `CitationPart`, etc.
- **Messages** group parts under a role: `user`, `assistant`, `tool`, or
  `developer`.
- **Requests** contain the model, messages, tools, and `Config`.
- **Providers** map canonical lm15 requests to provider HTTP requests and map
  provider responses back to canonical lm15 responses.
- **Conformance fixtures** make sure the mapping stays stable across providers
  and future SDK ports.

## Basic completion example

```python
import os

from lm15.providers import OpenAILM
from lm15.types import Config, Message, Request

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])

response = lm.complete(
    Request(
        model="gpt-4.1-mini",
        system="You are terse.",
        messages=(Message.user("Say hello in three words."),),
        config=Config(max_tokens=20, temperature=0.2),
    )
)

print(response.text)
print(response.finish_reason)
print(response.usage.total_tokens)
```

```output | ✓ 1.3s | 37 vars
Hello, how are?
stop
27
```

The same `Request` shape is used for Anthropic and Gemini:

```python
import os

from lm15.providers import AnthropicLM, GeminiLM
from lm15.types import Message, Request

providers = [
    AnthropicLM(api_key=os.environ["ANTHROPIC_API_KEY"]),
    GeminiLM(api_key=os.environ["GEMINI_API_KEY"]),
]

for lm in providers:
    response = lm.complete(
        Request(
            model={
                "anthropic": "claude-sonnet-4-5",
                "gemini": "gemini-3-flash-preview",
            }[lm.provider],
            messages=(Message.user("Say hello."),),
        )
    )
    print(lm.provider, response.text)
```
```output | ✓ 2.0s | 37 vars
anthropic Hello! How can I help you today?
gemini Hello! How can I help you today?
```

## Streaming example

`stream()` yields typed `StreamEvent` objects. Text arrives as
`StreamDeltaEvent(delta=TextDelta(...))`; the final event carries finish reason
and usage.

```python
import os

from lm15.providers import OpenAILM
from lm15.types import Message, Request, StreamDeltaEvent, TextDelta

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])
request = Request(
    model="gpt-4.1-mini",
    messages=(Message.user("Write one short sentence about Montreal."),),
)

for event in lm.stream(request):
    if isinstance(event, StreamDeltaEvent) and isinstance(event.delta, TextDelta):
        print(event.delta.text, end="", flush=True)
```
```output | ✓ 2.3s | 39 vars
Montreal is a vibrant, multicultural city in Canada known for its rich history and lively arts scene.
```

To consume a stream into a full `Response`:

```python
from lm15.result import materialize_response

response = materialize_response(lm.stream(request), request)
print(response.text)
```

```output | ✓ 4.5s | 40 vars
Montreal is a vibrant, multicultural city in Canada known for its rich history and delicious cuisine.
```

```python
response
```

```output | ✓ 2ms | 40 vars
Response(
    text='Montreal is a vibrant, multicultural city in Canada known for its rich history and delicious cuisine.',
    model='gpt-4.1-mini-2025-04-14',
    finish_reason='stop',
    usage=Usage(input_tokens=14, output_tokens=20, total_tokens=34, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_059dae56d8c4f9fa0069f25d77579881949db1ac4a47a9c16b',
    provider_data=<dict: 34 keys>,
)
```

## Tools example

lm15 distinguishes **function tools** that your application executes from
**provider-native built-in tools** like web search.

```python
import os

from lm15.providers import OpenAILM
from lm15.types import FunctionTool, Message, Request, ToolCallPart

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])

weather_tool = FunctionTool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

response = lm.complete(
    Request(
        model="gpt-4.1-mini",
        messages=(Message.user("What is the weather in Montreal?"),),
        tools=(weather_tool,),
    )
)

for call in response.tool_calls:
    assert isinstance(call, ToolCallPart)
    print(call.name, call.input)
```

```output | ✓ 845ms | 44 vars
get_weather {'city': 'Montreal'}
```

Built-in tool example:

```python
from lm15.types import BuiltinTool, Message, Request

response = lm.complete(
    Request(
        model="gpt-4.1-mini",
        messages=(Message.user("Find a recent Python release note and cite it."),),
        tools=(BuiltinTool("web_search"),),
    )
)

print(response.text)
for citation in response.citations:
    print(citation.title, citation.url)
```

```output | ✓ 6.8s | 46 vars
The most recent Python release is version 3.11.15, which supersedes Python 3.11.4. The release notes for Python 3.11.4, published on June 6, 2023, are available on the official Python website. ([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))

Python 3.11.4 introduced several significant features and optimizations, including:

- **PEP 657**: Include Fine-Grained Error Locations in Tracebacks
- **PEP 654**: Exception Groups and `except*`
- **PEP 680**: `tomllib`: Support for Parsing TOML in the Standard Library
- **gh-90908**: Introduce task groups to `asyncio`
- **gh-34627**: Support for atomic grouping (`(?>...)`) and possessive quantifiers (`*+`, `++`, `?+`, `{m,n}+`) in regular expressions

Additionally, the Faster CPython Project yielded exciting results, with Python 3.11 being up to 10-60% faster than Python 3.10. On average, a 1.22x speedup was measured on the standard benchmark suite. ([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))

For a comprehensive list of changes and improvements, you can refer to the full changelog provided in the release notes. ([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))
Python Release Python 3.11.4 | Python.org https://www.python.org/downloads/release/python-3114/?utm_source=openai
Python Release Python 3.11.4 | Python.org https://www.python.org/downloads/release/python-3114/?utm_source=openai
Python Release Python 3.11.4 | Python.org https://www.python.org/downloads/release/python-3114/?utm_source=openai
```

```python
response
```

```output | ✓ 3ms | 46 vars
Response(
    text='The most recent Python release is version 3.11.15, which supersedes Python 3.11.4. The release notes for Python 3.11.4, published on June 6, 2023, are available on the official Python website. ([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))\n\nPython 3.11.4 introduced several significant features and optimizations, including:\n\n- **PEP 657**: Include Fine-Grained Error Locations in Tracebacks\n- **PEP 654**: Exception Groups and `except*`\n- **PEP 680**: `tomllib`: Support for Parsing TOML in the Standard Library\n- **gh-90908**: Introduce task groups to `asyncio`\n- **gh-34627**: Support for atomic grouping (`(?>...)`) and possessive quantifiers (`*+`, `++`, `?+`, `{m,n}+`) in regular expressions\n\nAdditionally, the Faster CPython Project yielded exciting results, with Python 3.11 being up to 10-60% faster than Python 3.10. On average, a 1.22x speedup was measured on the standard benchmark suite. ([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))\n\nFor a comprehensive list of changes and improvements, you can refer to the full changelog provided in the release notes. ([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai)) ',
    model='gpt-4.1-mini-2025-04-14',
    finish_reason='stop',
    usage=Usage(input_tokens=311, output_tokens=346, total_tokens=657, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    citations=[CitationPart(url='https://www.python.org/downloads/release/python-3114/?utm_source=openai', title='Python Release Python 3.11.4 | Python.org', text='([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))', type='citation'), CitationPart(url='https://www.python.org/downloads/release/python-3114/?utm_source=openai', title='Python Release Python 3.11.4 | Python.org', text='([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))', type='citation'), CitationPart(url='https://www.python.org/downloads/release/python-3114/?utm_source=openai', title='Python Release Python 3.11.4 | Python.org', text='([python.org](https://www.python.org/downloads/release/python-3114/?utm_source=openai))', type='citation')],
    id='resp_022eca20811cb3450069f25dc2cd6c8195b881e701d5e89926',
    provider_data=<dict: 35 keys>,
)
```

## Media and endpoint examples

Multimodal input uses typed media parts:

```python
from lm15.types import ImagePart, Message, Request, TextPart

request = Request(
    model="gpt-4.1-mini",
    messages=(
        Message.user([
            TextPart("Describe this image in a few words."),
            ImagePart(
                url="https://raw.githubusercontent.com/github/explore/main/topics/react/react.png",
                media_type="image/png",
                detail="low",
            ),
        ]),
    ),
)

lm.complete(request)
```
```output
Response(
    text='The image is a stylized, light blue atomic symbol with a central nucleus and three elliptical orbits.',
    model='gpt-4.1-mini-2025-04-14',
    finish_reason='stop',
    usage=Usage(input_tokens=148, output_tokens=22, total_tokens=170, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0b292d5cfae1bc4b0069f25e9b30c881949d202000d6a4af89',
    provider_data=<dict: 35 keys>,
)
```

Non-chat endpoints have separate request/response types:

```python
from lm15.types import EmbeddingRequest, ImageGenerationRequest

embeddings = lm.embeddings(
    EmbeddingRequest(
        model="text-embedding-3-small",
        inputs=("hello", "world"),
    )
)
print(len(embeddings.vectors), len(embeddings.vectors[0]))

image_response = lm.image_generate(
    ImageGenerationRequest(
        model="gpt-image-1",
        prompt="A tiny watercolor robot reading a book.",
        size="1024x1024",
    )
)
print(image_response.images[0])
```
```output | ✓ 20.2s | 52 vars
2 1536
ImagePart(media_type='image/png', data='<base64: 2461668 chars>', url=None, file_id=None, path=None)
```

## Canonical JSON serialization

`lm15.serde` converts public lm15 types to canonical JSON-compatible dicts.
This is what conformance fixtures use.

```python
from lm15.serde import request_from_dict, request_to_dict
from lm15.types import Message, Request

request = Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),))
wire = request_to_dict(request)
round_tripped = request_from_dict(wire)
round_tripped == request
```

```output | ✓ 41ms | 56 vars
True
```

## Error normalization

Provider-specific HTTP/API errors are normalized into lm15 error classes:

```python
from lm15.errors import AuthError, RateLimitError, ProviderError
from lm15.types import Message, Request
from lm15.providers import OpenAILM

lm = OpenAILM(api_key = 'not a key')

try:
    lm.complete(Request(model="gpt-4.1-mini", messages=(Message.user("Hi"),)))
except AuthError as exc:
    print("Check API key:", exc.env_keys)
except RateLimitError as exc:
    print("Retry later:", exc.retry_after)
except ProviderError as exc:
    print(exc.provider, exc.provider_code, exc.status, exc.request_id)
```
```output | ✓ 189ms | 59 vars
Check API key: ('OPENAI_API_KEY',)
```

## How the fixture/conformance system works

The conformance suite checks that lm15's canonical model stays aligned with real
provider APIs.

```text
logical lm15 case
  conformance/cross_sdk/test_cases.json
        │
        ▼
lm15-python2 provider adapter builds HTTP request
        │
        ▼
expected provider fixture
  conformance/provider_requests/cases/<provider>/<feature>.json
        │
        ├── check_request_fixtures.py compares request shape
        ├── validate_live.py can send the request to the real API
        ├── check_response_fixtures.py parses saved response bodies/SSE
        ├── check_error_fixtures.py normalizes provider error bodies
        ├── check_endpoint_fixtures.py checks embeddings/files/batch/image/audio/live
        ├── check_serde_fixtures.py checks canonical JSON round trips
        └── check_doc_drift.py checks provider docs against features.yaml
```

Run everything:

```bash
python3 conformance/run_all.py --strict
```

Run one check:

```bash
python3 conformance/check_doc_drift.py --strict
python3 conformance/check_response_fixtures.py --strict
```

Run or preview one live provider fixture, if the relevant API key is set:

```bash
python3 conformance/provider_requests/validate_live.py --dry-run --task openai.basic_text
python3 conformance/provider_requests/validate_live.py --task openai.basic_text
```

Generated reports are written under `conformance/reports/` and are ignored by
git.

### Adding or completing a fixture

1. Add or update the logical case in
   `conformance/cross_sdk/test_cases.json`.
2. Add the expected provider HTTP request in
   `conformance/provider_requests/cases/<provider>/<feature>.json`.
3. Add the feature to `conformance/provider_requests/features.yaml` so doc drift
   can tell whether provider documentation is represented.
4. If response parsing should be checked, add an `expect_lm15` block to the
   provider case and save a real response body under
   `conformance/provider_requests/results/bodies/<provider>.<feature>/`.
5. If the provider has a special error shape, add an error case under
   `conformance/errors/cases/<provider>.json`.
6. Run:

   ```bash
   python3 conformance/run_all.py --strict
   pytest -q
   ```

Example `expect_lm15` block:

```json
{
  "expect_lm15": {
    "parts": {
      "text": {"min": 1},
      "citation": {"min": 1}
    },
    "finish_reason": "stop",
    "usage": {"required": true}
  }
}
```

### Doc-drift fixture check

`conformance/check_doc_drift.py` parses snapshotted provider docs in
`conformance/provider_docs/` and compares top-level request parameters with
`conformance/provider_requests/features.yaml`.

Some always-on lm15 request fields do not need separate feature entries:

```python
IGNORE_PARAMS = {"model", "messages", "contents", "input"}
```

Provider docs often use camelCase/PascalCase while lm15 feature names use
snake_case, so the drift check normalizes names before deciding that a param is
unmapped.

If `check_doc_drift.py --strict` reports an unmapped param, either:

- add a real feature entry to `features.yaml`, or
- add the param to `IGNORE_PARAMS` only if it is a core field that should never
  have a separate fixture.

## Provider adapter development guide

Provider classes live in `lm15/providers/` and inherit `BaseProviderLM`.
A provider adapter is responsible for:

- `build_request(request, stream)` — map canonical `Request` to an HTTP request.
- `parse_response(request, response)` — map provider JSON to canonical
  `Response`.
- `parse_stream_events(...)` — map SSE chunks to `StreamEvent`s (the single
  stream-parse path per provider).
- `normalize_error(status, body)` — map provider errors to `lm15.errors`.
- Optional endpoint methods: `embeddings`, `file_upload`, `batch_submit`,
  `image_generate`, `audio_generate`, `live`.

Keep provider-only options in `Config.extensions` rather than adding universal
fields unless the same concept is supported across providers.

## Useful commands

```bash
# Unit and conformance tests through pytest
pytest -q

# Full offline conformance suite
python3 conformance/run_all.py --strict

# Request fixture comparison only
python3 conformance/check_request_fixtures.py --strict

# Response/SSE fixture parser check only
python3 conformance/check_response_fixtures.py --strict

# Canonical JSON round trips only
python3 conformance/check_serde_fixtures.py --strict

# Provider-doc coverage only
python3 conformance/check_doc_drift.py --strict
```
