# Using model profiles and compatibility policies

`lm15` keeps inference requests simple:

```python
Request(model="gpt-4.1-mini", messages=(...))
```

Model profiles and compatibility policies are optional helpers for applications
that need model metadata, local OpenAI-compatible servers, routing, or exact
provider dialect control.

They answer different questions:

```text
Request              What the caller wants the model to do.
ModelInfo            What a model can do and what it costs.
ProviderProfile      Where/how to call provider endpoints.
Compat policy        How to serialize for a provider API dialect.
ModelRegistry        How to index and resolve model metadata.
```

## Why profiles are separate from Request

Provider compatibility is not a semantic user request. For example, whether an
OpenAI-compatible server expects `max_tokens` or `max_output_tokens` is a wire
serialization detail, not a different task for the model.

For that reason, these settings live outside `lm15.types.Request`:

- `lm15.compat` — typed compatibility policy objects.
- `lm15.models` — model metadata, pricing, and `ModelRegistry`.
- `lm15.profiles` — provider endpoint profiles and compatibility resolution.

## Minimal use: no profile

Existing code does not change.

```python
import os

from lm15.providers import OpenAILM
from lm15.types import Message, Request

lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"])
response = lm.complete(
    Request(
        model="gpt-4.1-mini",
        messages=(Message.user("Say hello."),),
    )
)
```

## Local OpenAI-compatible server

For **Chat Completions** (`/chat/completions`), start with
[Connect an unlisted OpenAI-compatible server](connecting-openai-compatible-servers.md).
It shows `OpenAIChatLM` with no preset, an unknown-name error, and a custom
`OpenAIChatCompat` object. You do not need a profile or a registry entry.

The example below is for **Responses** (`/responses`), a different API.
Use it only if your server implements that endpoint. Use
`ProviderProfile.inference()` with an `OpenAIResponsesCompat` preset.

```python
from lm15.compat import OpenAIResponsesCompat
from lm15.profiles import ProviderProfile
from lm15.providers import OpenAILM

profile = ProviderProfile.inference(
    provider="ollama",
    api_family="openai_responses",
    base_url="http://localhost:11434/v1",
    compat=OpenAIResponsesCompat.preset("ollama"),
)

lm = OpenAILM.from_profile(api_key="ollama", profile=profile)
```

The preset tells the OpenAI Responses serializer to use local-server-friendly
choices such as `max_tokens`, `system` for developer instructions, no strict
schema field, and no prompt-cache fields.

## Compatibility policy fields

`OpenAIResponsesCompat` is a **partial** policy. Fields default to `None`, which
means "inherit from the parent profile". The string value `"auto"` is different:
it explicitly asks the adapter to use its automatic default for that field.

```python
from lm15.compat import OpenAIResponsesCompat

compat = OpenAIResponsesCompat(
    developer_role="system",
    max_output_tokens_field="max_tokens",
    reasoning_format="qwen",
    strict_tools="omit",
    cache_control="none",
)
```

Important fields:

| Field | Meaning |
|---|---|
| `developer_role` | Serialize `Message.developer(...)` as `developer` or `system`. |
| `max_output_tokens_field` | Emit `max_output_tokens`, `max_completion_tokens`, or `max_tokens`. |
| `reasoning_format` | Map `Config.reasoning` to Responses reasoning, `reasoning_effort`, Qwen, DeepSeek, etc. |
| `tool_result_name` | Include or omit `name` on function call outputs. |
| `strict_tools` | Include or omit the OpenAI `strict` field on function tools. |
| `cache_control` | Emit OpenAI prompt-cache fields, no cache fields, or future dialects. |
| `routing` | Provider routing object, for example OpenRouter provider selection. |

Available presets include:

```python
OpenAIResponsesCompat.preset("openai")
OpenAIResponsesCompat.preset("openrouter")
OpenAIResponsesCompat.preset("ollama")
OpenAIResponsesCompat.preset("vllm")
OpenAIResponsesCompat.preset("qwen")
OpenAIResponsesCompat.preset("deepseek")
OpenAIResponsesCompat.preset("zai")
```

## Provider-level plus model-level overrides

Provider profiles can carry default compatibility, and individual models can
override it.

```python
from lm15.compat import OpenAIResponsesCompat
from lm15.models import InferenceModelInfo, ModelInfo
from lm15.profiles import ProviderProfile

profile = ProviderProfile.inference(
    provider="local",
    api_family="openai_responses",
    base_url="http://localhost:8000/v1",
    compat=OpenAIResponsesCompat.preset("vllm"),
    models=(
        ModelInfo(
            id="qwen3-coder",
            provider="local",
            api_family="openai_responses",
            aliases=("qwen",),
            inference=InferenceModelInfo(
                input_modalities=("text",),
                output_modalities=("text",),
                context_window=128000,
                max_output_tokens=8192,
                supports_reasoning=True,
                reasoning_efforts=("off", "low", "medium", "high"),
            ),
            compat=OpenAIResponsesCompat(reasoning_format="qwen"),
        ),
    ),
)
```

Resolution order for OpenAI Responses requests is:

```text
base URL default
  < ProviderProfile endpoint compat
  < ModelInfo compat for Request.model
  < Config.extensions request-level override
```

`None` fields inherit. Non-`None` fields override, including `"auto"`.

## Request-level escape hatch

Request-level compatibility is supported for experiments and temporary provider
features, but it should not be the normal configuration path.

```python
from lm15.types import Config, Message, Request

request = Request(
    model="weird-local-model",
    messages=(Message.user("Hi"),),
    config=Config(
        max_tokens=200,
        extensions={
            "openai_responses_compat": {
                "max_output_tokens_field": "max_tokens",
                "reasoning_format": "qwen",
            },
            "metadata": {"user_id": "u1"},
        },
    ),
)
```

Compatibility keys are consumed by the adapter and are not passed through to the
provider payload. Other extension keys, such as `metadata`, still pass through.

Supported request-level shapes:

```python
{"openai_responses_compat": {...}}
{"openai_compat": {...}}
{"compat": {"openai_responses": {...}}}
{"compat": {"openai": {...}}}
```

## Model metadata

`ModelInfo` describes a model without changing the request shape.

```python
from lm15.models import InferenceModelInfo, InferencePricing, ModelInfo

model = ModelInfo(
    id="gpt-4.1-mini",
    provider="openai",
    api_family="openai_responses",
    aliases=("mini",),
    inference=InferenceModelInfo(
        input_modalities=("text", "image"),
        output_modalities=("text",),
        context_window=1_000_000,
        max_output_tokens=32768,
        supports_reasoning=False,
        pricing=InferencePricing(
            input_per_million=0.40,
            output_per_million=1.60,
            cache_read_per_million=0.10,
        ),
    ),
)
```

Capabilities are endpoint-specific. `ModelInfo.inference` describes inference
capabilities; new endpoint families can be described later without touching
it. Fine-tune provenance lives on `ModelOrigin` (`type`, `base_model`) —
lm15 does inference with tuned models, not training.

## Model registry

`ModelRegistry` is optional. Use it when your application needs discovery,
validation, aliases, or cost estimation.

```python
from lm15.models import ModelRegistry

registry = ModelRegistry.from_profiles([profile])

info = registry.resolve("qwen", provider="local")
assert info is not None
assert info.inference is not None
print(info.inference.context_window)
```

You can also add generated or fine-tuned models at runtime:

```python
registry.add(
    ModelInfo(
        id="ft:gpt-4.1-mini:org:project:abc123",
        provider="openai",
        api_family="openai_responses",
        origin=ModelOrigin(
            type="fine_tune",
            id="ftjob-abc123",
            base_model="gpt-4.1-mini",
        ),
    )
)
```

## Cost estimation

Pricing is deliberately separate from `Usage`, because price tables change and
some providers price non-token dimensions differently.

```python
pricing = info.inference.pricing
if pricing is not None:
    cost = pricing.estimate(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=response.usage.cache_read_tokens,
        cache_write_tokens=response.usage.cache_write_tokens,
    )
    print(cost, pricing.currency)
```

## Design rules

- Keep `Request` semantic and provider-neutral.
- Put provider wire quirks in compat policies.
- Put endpoint URLs and defaults in `ProviderProfile`.
- Put model capabilities and prices in `ModelInfo`.
- Use `ModelRegistry` only when you need discovery or validation.
- Prefer provider/model profiles over request-level compat overrides.
