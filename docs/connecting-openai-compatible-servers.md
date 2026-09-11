# Connect an unlisted OpenAI-compatible server

You do not need a named lm15 preset to use a server. This tutorial shows how to
connect through OpenAI **Chat Completions**, choose compatibility settings from
actual server requirements, and inspect the resulting request before sending it.

Use the Python package from [Getting started](getting-started.md). Steps 1–4
run without a server or a real API key. Step 5 makes real requests.

## What you need from the server

Keep these three choices separate:

| Setting | What to supply |
|---|---|
| `base_url` | The API root, such as `http://localhost:1234/v1`, **not** the full `/chat/completions` URL. |
| `api_key` | That server's token. For a local server with authentication disabled, these examples pass the explicit placeholder `"unused"`. |
| `Request.model` | The exact model ID the server exposes, not an lm15 routing prefix. |

For LM Studio, load a model and start its local server. Copy the address and
model ID from its interface; `http://localhost:1234/v1` is the example address
used here. Configure a different address if your server runs elsewhere.

Use HTTPS for remote servers. Do not copy an OpenAI account key into a custom
server's configuration. A compatibility policy does not choose credentials.

This guide uses `OpenAIChatLM`, which sends requests to `/chat/completions`.
A server exposing only `/responses` needs `OpenAILM` and
`OpenAIResponsesCompat` instead. A preset cannot switch between those APIs.

## 1. Understand what a preset name means

A `compat` string is a lookup in lm15's existing presets, not an arbitrary
provider label and not automatic server detection. Unknown names fail during
client construction, before a chat request is sent:

```python
import json
import os
from dataclasses import replace

from lm15 import Config, Message, OpenAIChatLM, Request, ResponseStream
from lm15.compat import OpenAIChatCompat

base_url = "http://localhost:1234/v1"

try:
    OpenAIChatLM(
        base_url=base_url,
        api_key="unused",
        compat="your-unlisted-server",
    )
except ValueError as error:
    print(error)
```
```output
unknown OpenAIChatCompat preset: 'your-unlisted-server'
```

There are three valid choices:

| `compat` value | Meaning |
|---|---|
| Omitted or `None` | Use the adapter's default Chat Completions policy. No server detection. |
| A recognized string, such as `"vllm"` | Use an existing preset; a server preset can also supply a default address. |
| An `OpenAIChatCompat(...)` object | Use your own settings. Supply `base_url` explicitly; the object has no address. |

Changing `base_url` alone does **not** change the compatibility policy.

## 2. Start without a preset

For a server matching the default OpenAI behavior, omit `compat` entirely.
`Config.max_tokens` expresses your intent; the adapter chooses the JSON field
name. Build a request to see that choice:

```python
request = Request(
    model="your-model-id",
    system="Answer briefly.",
    messages=(Message.user("Explain rainbows in one sentence."),),
    config=Config(max_tokens=64),
)

with OpenAIChatLM(base_url=base_url, api_key="unused") as lm:
    prepared = lm.build_request(request, stream=False)
    payload = json.loads(prepared.body)
    print(prepared.method, prepared.url)
    print("model:", payload["model"])
    print("max_completion_tokens:", payload["max_completion_tokens"])
```
```output
POST http://localhost:1234/v1/chat/completions
model: your-model-id
max_completion_tokens: 64
```

With the literal placeholder credential used here, `build_request` constructs
and authenticates the request locally; it does not send it. Do not print the
whole prepared request or its headers: they can contain credentials. The body
can contain private prompts, so inspect it only with safe test input.

## 3. Configure a real-world compatibility difference

Suppose your deployment's documentation or an actual error establishes that it:

- Uses `system` for instructions.
- Expects `max_tokens`, not `max_completion_tokens`.
- Accepts streaming but rejects the `stream_options` field.

This is a deployment scenario, **not** a claim that every local server or every
LM Studio version has these restrictions. Change only settings you have evidence
for; do not copy another server's entire preset because it runs the same model.

Pass an object instead of inventing a preset name:

```python
chat_compat = OpenAIChatCompat(
    instruction_role="system",
    max_tokens_field="max_tokens",
    stream_usage="omit",
)

with OpenAIChatLM(
    base_url=base_url,
    api_key="unused",
    compat=chat_compat,
) as lm:
    prepared = lm.build_request(request, stream=False)
    payload = json.loads(prepared.body)
    streaming = json.loads(lm.build_request(request, stream=True).body)
    print("instruction role:", payload["messages"][0]["role"])
    print("max_tokens:", payload["max_tokens"])
    print("max_completion_tokens present:", "max_completion_tokens" in payload)
    print("stream:", streaming["stream"])
    print("stream_options present:", "stream_options" in streaming)
```
```output
instruction role: system
max_tokens: 64
max_completion_tokens present: False
stream: True
stream_options present: False
```

The original `Request` did not change. You changed its translation into the
server's JSON format, not the prompt or the requested token limit.

**Trade-off:** `stream_usage="omit"` removes the request for a streaming usage
report. Streaming text still works if the server supports it, but token counts
may be unavailable. It does not disable streaming or calculate missing counts.

Fields left at `None` inherit the adapter's defaults in this direct-construction
example; they do not mean "disable" or "detect". In layered profiles, `None`
means inherit from the preceding layer. `"auto"`, on fields that accept it,
selects the adapter's heuristic, not a network capability probe.

### LM Studio's existing name

Currently `compat="lmstudio"` is recognized as an alias for Ollama's policy,
not an independent, verified LM Studio preset. It has no matching default URL
entry, so using that name without `base_url` can leave the destination as OpenAI.
**Always supply the explicit LM Studio address.** Using your own compat object
also avoids depending on that alias. Test your server version's behavior before
choosing settings; the policy above is not a blanket LM Studio recommendation.

## 4. Adapt a known preset when it really is the same server

For example, if your deployment is vLLM but its gateway rejects `stream_options`,
start with the vLLM preset and change that one field:

```python
custom_vllm = replace(
    OpenAIChatCompat.preset("vllm"),
    stream_usage="omit",
)

with OpenAIChatLM(
    base_url="http://gpu-box:8000/v1",
    api_key="unused",
    compat=custom_vllm,
) as lm:
    prepared = lm.build_request(request, stream=True)
    payload = json.loads(prepared.body)
    print(prepared.url)
    print("stream_options present:", "stream_options" in payload)
```
```output
http://gpu-box:8000/v1/chat/completions
stream_options present: False
```

The object retains the other vLLM settings, but **does not carry vLLM's default
address**. Do not mutate lm15's preset tables to register your deployment.

## 5. Send a request, then test streaming

The following blocks make real requests. Run the preceding blocks first.
Set `LOCAL_MODEL_ID` to your server's model ID, and `LOCAL_LLM_API_KEY` if it
requires authentication. Replace `base_url` as needed. Use `chat_compat` only
if the requirements in step 3 match your deployment; otherwise adjust it or use
`None` for the default policy.

```python
# Live: requires a running server and LOCAL_MODEL_ID.
live_request = replace(request, model=os.environ["LOCAL_MODEL_ID"])

with OpenAIChatLM(
    base_url=base_url,
    api_key=os.environ.get("LOCAL_LLM_API_KEY", "unused"),
    compat=chat_compat,
) as lm:
    response = lm.complete(live_request)
    print(response.text)
```

Then check streaming separately:

```python
# Live: requires a running server and the preceding blocks.
with OpenAIChatLM(
    base_url=base_url,
    api_key=os.environ.get("LOCAL_LLM_API_KEY", "unused"),
    compat=chat_compat,
) as lm:
    result = ResponseStream(lm.stream(live_request), live_request)
    for text in result:
        print(text, end="", flush=True)
    print()
```

No model answer is shown here: offline request inspection verifies serialization,
not successful inference against a particular server. Tests also run these live
blocks against scripted responses, not a real LM Studio installation.

## Troubleshooting and integration checklist

| Symptom or requirement | What to check |
|---|---|
| `unknown OpenAIChatCompat preset` | Omit `compat` or pass an object; an unlisted brand is not a valid preset name. |
| Connection refused or HTTP 404 | Is the server running? Is the address an API root? Does it expose `/chat/completions`? |
| Authentication error | Use the target server's token and authentication settings, not an unrelated provider key. |
| Model not found | Copy the exact server model ID. Direct clients do not strip router prefixes. |
| Rejected token-limit field | Set `max_tokens_field` to the documented spelling. |
| Rejected `stream_options` | Set `stream_usage="omit"` if the server cannot accept it; expect possibly missing usage. |
| Instructions need `system` instead of `developer` | Set `instruction_role="system"`. |
| Works for text, fails for tools or JSON output | Verify those features with that model and server. Text success does not prove feature support. |

For each integration:

1. Record the server version, model ID, and documentation or observed error
   behind each non-default setting.
2. Inspect the destination and both ordinary and streaming request bodies.
3. Test a short text reply, then streaming, then each feature your application
   needs. Check results, not just HTTP success: servers can ignore fields.
4. Keep a small regression test for the fields you depend on. Use
   [`FakeTransport`](cookbooks/17-errors-and-testing.md) for offline tests.
5. Do not silently remove requested features to obtain a successful response.
   A compat object controls known format differences; it cannot add capabilities
   to a server or make an unrelated response format parse as Chat Completions.

Use direct clients for custom policies or multiple differently configured
servers. If only the address differs and a registered provider's policy fits,
[`RouterConfig(base_urls=...)`](using-the-router.md#addresses) is also supported.
An unknown `compat` name and an unknown router provider are separate lookups;
neither registers the other.

## Related documentation

- [Local & OpenAI-compatible servers](cookbooks/16-local-and-compatible-servers.md) — named presets.
- [Model profiles & compat](using-model-profiles.md) — profiles and policy layering.
- [Compatibility API reference](reference/profiles.md#compatibility-policies) — available fields and values.
- [Provider passthrough](cookbooks/18-provider-passthrough.md) — additional server-specific fields.
- [Authentication](authentication.md) — credentials and access policies.
