# Providers & models

You usually arrive knowing which model you want — Claude, GPT, Gemini,
a Llama on Groq, something running on your own machine. This page gets
you from that to a working call: what a "provider" means in lm15,
which ones are supported, how to write the model string, and what the
library can tell you about a model before you spend a token.

## The mental model

Every provider speaks its own HTTP dialect: different URLs, different
JSON shapes, different auth headers, different streaming formats. lm15
hides exactly that — you write one `Request`, and a per-provider
**adapter** translates it into that provider's dialect, byte-for-byte
correctly.

So in lm15, a *provider* is just a short string naming a dialect:
`anthropic`, `openai`, `gemini`, `groq`. And the simplest way to pick
one is to put it in front of the model name:

```python
router.complete(Request(model="anthropic:claude-haiku-4-5", ...))
router.complete(Request(model="groq:llama-3.3-70b-versatile", ...))
```

That is genuinely the whole trick. Everything below is the supporting
detail: the list of providers, the shortcuts, and the metadata.

## Which providers are supported?

| provider | string | key you need | beyond chat |
|---|---|---|---|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | files, batch |
| OpenAI | `openai` | `OPENAI_API_KEY` | live voice, files, batch, images, speech, video |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | live voice, files, batch, images, speech, video |
| xAI | `xai` | `XAI_API_KEY` or Grok subscription | images, video |
| Groq | `groq` | `GROQ_API_KEY` | — |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | — |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | — (see note) |
| DeepSeek, Anthropic wire | `deepseek-anthropic` | `DEEPSEEK_API_KEY` | — (see note) |
| Z.AI (GLM) | `zai` | `ZAI_API_KEY` | — (see note) |
| Moonshot AI (Kimi) | `moonshotai` | `MOONSHOTAI_API_KEY`, then `MOONSHOT_API_KEY` | — (see note) |
| Moonshot AI, Responses wire | `moonshotai-responses` | same key | web search built-in (see note) |
| Moonshot AI, Anthropic wire | `moonshotai-anthropic` | same key | — (see note) |
| Meta (Muse Spark, Muse Image) | `meta` | `META_API_KEY` | files, images (see note) |
| Meta, Chat Completions wire | `meta-chat` | `META_API_KEY` | — (see note) |
| Meta, Anthropic wire | `meta-anthropic` | `META_API_KEY` | — (see note) |
| ollama (local) | `ollama` | none | — |
| vLLM (local) | `vllm` | none | — |
| SGLang (local) | `sglang` | none | — |
| Claude subscription | `claude-code` | your `claude` CLI login | — |
| ChatGPT subscription | `openai-codex` | your `codex` CLI login | — |

Every provider does chat with streaming and tools — that column would
be all checkmarks, so the table only lists what each offers *beyond*
it. Setting the key is one `export`; every other way to authenticate
is on [Authentication](authentication.md).

!!! note "DeepSeek: thinking is on by default"
    DeepSeek models reason before they answer unless you say otherwise
    (`Config(reasoning=Reasoning(effort="off"))`). While thinking is on,
    the server accepts `temperature` and `top_p` and silently ignores
    them (its documentation says so; verified live 2026-09-03) — lm15
    does not raise, because that is the provider's default mode.
    `Config.user_id` is sent as DeepSeek's `user_id` field. Structured
    output is `json_object` only; a JSON schema is refused by the server
    with a clear 400. Requests and responses are processed and stored in
    the People's Republic of China (DeepSeek privacy policy); prepaid
    balance, so a drained account is a 402, never a surprise bill.
    The same key also opens DeepSeek's Anthropic-format endpoint as
    `deepseek-anthropic` — a second provider string because it is a
    different wire. On that wire lm15 refuses `claude-*` model names
    before sending: the endpoint would silently answer with a DeepSeek
    model (`claude-opus*` → `deepseek-v4-pro`, verified live 2026-09-03).
    Name the DeepSeek model you want. No model listing there; use
    `deepseek` for that.

!!! note "Z.AI: thinking is on by default at maximum effort"
    GLM-5.3 models always reason and default to `reasoning_effort: max` —
    "Say ok." cost 127 reasoning tokens in the pinned capture. Pass
    `Config(reasoning=Reasoning(effort="low"))` for cheap calls; an
    explicit `effort="off"` is refused by the server with a clear 400
    (GLM-5.2 still honours it). Only `tool_choice` mode `auto` and
    `response_format` `json_object` are honoured; lm15 raises
    `UnsupportedFeatureError` before the wire for the other forms because
    the server accepts them and silently ignores them (verified live
    2026-09-03). `Config.user_id` is sent as Z.AI's `user_id` and must be
    6–128 characters. A drained balance is a `BillingError` (Z.AI reports
    it on HTTP 429). Data is processed in Singapore; the GLM Coding Plan is
    a separate product with its own endpoint that lm15 does not name.

!!! note "Moonshot AI (Kimi): two reasoning families, one dial"
    `moonshotai` is the Chat Completions wire of the Kimi API Platform
    (`https://api.moonshot.ai/v1`). `kimi-k3` always reasons and takes
    `Reasoning(effort=...)` as `low`, `high`, or `max` (default `max` —
    "Say ok." cost 134 reasoning tokens in the pinned capture; pass `low`
    for cheap calls). Other effort words raise `UnsupportedFeatureError`
    before the wire: the server accepts *any* word without validating it
    (`medium` and `bogus` both answered 200, verified live 2026-09-03), so
    a downgrade would be silent. `effort="off"` sends `thinking:
    {type: disabled}` — the documented switch for `kimi-k2.6`, and one
    `kimi-k3` honours too although its docs say it cannot. Stated
    trade-off: `kimi-k2.6` and `kimi-k2.7-code` have no effort levels and
    the server ignores an effort word on them silently (verified live);
    lm15 does not sniff model names, so do not set an effort word on K2.x.
    `reasoning_content` is replayed natively on every assistant turn
    (the docs require it in tool loops). `temperature` is fixed per model
    and any other value is a loud 400; `tool_choice` `required` is a 400
    on K2.x with thinking on. `Config.user_id` rides `safety_identifier`;
    `Config.max_tokens` rides `max_completion_tokens`. Caching is
    automatic (`cache_read_tokens` from `prompt_tokens_details`); a
    drained balance is a `BillingError` on HTTP 429. Files (content
    extraction, vision input), batches, the Responses wire (`kimi-k3`
    only) and the Anthropic wire (`/anthropic`) exist on the service and
    are not registered. The counterparty is Moonshot AI PTE. LTD.,
    Singapore; data is stored in Singapore. The terms let Moonshot use
    API content to improve its services unless an enterprise agreement
    says otherwise — read them before sending private data. The API
    platform key is distinct from a Kimi Code subscription.

    The same key opens two more wires. `moonshotai-responses` is the
    Responses wire (`kimi-k3` documented; `kimi-k2.6` also answers): it is
    stateless — `store` is always false — so lm15 replays the reasoning
    item with its summary text; the `web_search` built-in runs server-side
    (`BuiltinTool(name="web_search")`, ~4.6K input tokens for one search in
    the pinned capture); this wire validates strictly and answers a loud
    400 to `effort="off"`, `tool_choice` beyond `auto`, `json_object`,
    `temperature`, and `CacheConfig(retention="long")`. `moonshotai-anthropic`
    is the Anthropic Messages wire at `/anthropic` (bearer token, what
    Claude Code's `ANTHROPIC_AUTH_TOKEN` sends): the dial is
    `output_config.effort` alone, `effort="off"` is honoured, thinking
    blocks come back **unsigned** and lm15 replays them as `thinking`
    blocks anyway (the server accepts them), and lm15 raises before the
    wire for `temperature`/`top_p`/`top_k`, `parallel=False`, and effort
    words outside `low|high|max` because that wire swallows them silently.
    No model listing there (`/anthropic/v1/models` is 404); use `moonshotai`.

!!! note "Meta: one key, three wires, and the model always reasons"
    Meta Model API (dev.meta.ai) serves Muse Spark over three request
    formats with one key. `meta` is the Responses wire — Meta's
    recommended default and the only one that carries reasoning across
    turns; it also owns the account surfaces: Files (`file_upload` …),
    image generation and editing with `muse-image-1.0`
    (`image_generate`, $0.01 per image), and model listing. `meta-chat`
    (Chat Completions) and `meta-anthropic` (Anthropic Messages) are the
    same model for code that already speaks those wires. Muse Spark always
    reasons: `effort="off"` is refused by the server with a clear 400 on
    every wire, and the default effort spent 597 hidden reasoning tokens
    on "Say ok." in the first pinned capture — set
    `Config(reasoning=Reasoning(effort="low"))` for cheap calls and give
    `max_tokens` room for the reasoning. `tool_choice` beyond `auto`,
    `logprobs`, `stop`, and `n > 1` are refused by the server (loud, so
    lm15 sends them as asked). Prompt caching is automatic; `cache.key`
    and `retention="long"` are forwarded. Meta's own docs name the env
    variable `MODEL_API_KEY`; lm15 reads only `META_API_KEY`, because a
    vendor-less name could be another tool's secret.
    Standard-tier models (`muse-spark-1.3`) are never trained on; the
    `-contributor` models are cheaper because they are. Speech-to-text
    (`muse-voice-transcribe-1.0`) is on Meta's own WebSocket wire, which
    lm15 does not have a surface for yet.

!!! note "You may also see `openai-chat`"
    OpenAI has two wire dialects: its current **Responses API** (what
    the `openai` string uses) and the older **Chat Completions**
    dialect that half the industry adopted as a de-facto standard.
    lm15 ships both. The `openai-chat` adapter speaks Chat
    Completions — and it is the same adapter that powers the Groq,
    OpenRouter, DeepSeek, Z.AI, ollama, vLLM, and SGLang rows above, each via
    a preset that knows that server's URL and quirks. You rarely type
    `openai-chat` yourself; the presets do.

Chat — the part all of this rests on — is **stable**: it is frozen by
a cross-language contract and only changes additively. The
beyond-chat endpoints work and are live-tested, but their shapes may
still move before 1.0 stable ([roadmap](roadmap.md)).

## What if my provider isn't listed?

For a server implementing OpenAI Chat Completions, start with the
existing adapter; no provider registration is required:

```python
from lm15 import OpenAIChatLM

lm = OpenAIChatLM(api_key="...", base_url="https://your-gateway/v1")
```

Omitting `compat` uses the default Chat Completions policy. An unknown
`compat="server-name"` raises `ValueError`; pass an `OpenAIChatCompat`
object if your server needs different settings.

[Connect an unlisted OpenAI-compatible server](connecting-openai-compatible-servers.md)
walks through a custom policy, LM Studio's preset and address, and checking
requests without sending them. Compatibility depends on the server and
features you use, not just its advertised API family.
[Model profiles & compat](using-model-profiles.md) covers policy layering;
[Cloud hosts](cloud-hosts.md) covers Azure, Bedrock, and Vertex.

## How do I write the model string?

Start with the prefix form — `"provider:model-id"`. It always works
and can never be ambiguous:

```python
"anthropic:claude-haiku-4-5"
"groq:llama-3.3-70b-versatile"
"ollama:qwen3:4b"                 # only the FIRST colon splits
```

For well-known families you can drop the prefix; a small built-in rule
table recognizes `claude-*`, `gpt-*`, `gemini-*`, `grok-*`,
`o1/o3/o4*`, and the video families `sora-*` and `veo-*`:

```python
"claude-haiku-4-5"                # resolves to anthropic on its own
```

And if you ever wonder what happened, ask — resolution is a lookup you
can read, not magic:

```python
print(LMRouter().resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

One honest gotcha: the string splits on the *first* `:`, so a
fine-tune id like `ft:gpt-4.1:org` needs the explicit
`openai:ft:gpt-4.1:org`. The full resolution ladder — including model
*objects* that carry their own provider, from catalog packages like
aimo — is in [the router guide](using-the-router.md).

## Will it work before I try it?

Adapters describe themselves. Instead of hunting through docs for
"does Anthropic support batch?", ask the object. `supports` is the one
endpoint declaration per adapter, pinned by the contract's support matrix;
per-model facts such as modalities and prices live on `ModelInfo`
([model profiles](using-model-profiles.md)).

```python
lm = AnthropicLM(api_key="...")
print(lm.supports.batches)
print(lm.supports.live)
```

```output
True
False
```

Asking for something a provider can't do raises a typed
`UnsupportedFeatureError` immediately — you never find out via a
confusing HTTP 400.

## What does it cost? What's the context window?

Here lm15 is deliberately humble: the core library ships **no** model
list, no context windows, no price table. Those numbers change weekly,
and a frozen library that pretended to know them would quietly rot.

Instead, *catalog packages* supply the numbers through a
[specified protocol](model-hydration.md), and lm15 gives them one
home. Install one — `pip install aimo-registry` — and ask:

```python
from lm15 import ModelRegistry

registry = ModelRegistry.discover()   # finds installed catalog packages
info = registry.resolve("claude-3-5-sonnet-20240620", provider="anthropic")

print(info.inference.context_window)
print(info.inference.pricing.input_per_million,
      info.inference.pricing.output_per_million,
      info.inference.pricing.currency)
print(info.inference.pricing.estimate(input_tokens=1200, output_tokens=350))
```

```output
200000
3.0 15.0 USD
0.00885
```

That run knew 6,993 models across 209 providers. Hand the registry to
the router (`RouterConfig(registry=...)`) and bare model ids and
aliases resolve through it too — with a typed `AmbiguousModelError`
naming the fix when two providers offer the same id, never a silent
guess.

One rule keeps this trustworthy: catalog data is **advisory**. It
informs routing, cost estimates, and your own logic — it never changes
the bytes lm15 puts on the wire.

## What lm15 won't do for you (on purpose)

If you are coming from LiteLLM or LangChain, you may expect built-in
retries, an automatic tool-execution loop, a cost ledger, or fallback
routing. lm15 has none of these — deliberately. It is the foundation
layer; policy belongs to your application or to the opinionated
libraries built on top of it. Every omission has a one-page answer in
the [cookbooks](cookbooks/index.md) (retries, budgets, fallbacks are a
few lines each) and a written reason in the
[design rationale](design-rationale.md).

## Where next

- Set up credentials properly: [Authentication](authentication.md)
- The resolution ladder in full: [Using the router](using-the-router.md)
- Actually calling models: back to [Getting started](getting-started.md),
  or the [cookbooks](cookbooks/index.md)
