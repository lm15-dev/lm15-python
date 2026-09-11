# Changelog

## 1.0.0rc1 — 2026-09-11

First release candidate for 1.0. Install explicitly with
`pip install lm15==1.0.0rc1`; this is a prerelease, not the stable 1.0 release.
The changes below include breaking changes since the earlier public releases.
The contract pin is unchanged; publishing this candidate does not ratify draft
contract material or freeze provisional endpoints.

- Add a tested tutorial for unlisted OpenAI-compatible servers: default policy,
  custom `OpenAIChatCompat` objects, adapting presets, and offline request checks.
- Clarify custom router addresses and the difference between Chat Completions
  and Responses compatibility policies.
- Publish from the single source version in `lm15/_version.py`; check release
  artifacts and the installed wheel's version before upload.
- **A named server never resolves to the OpenAI cloud.** `compat="lmstudio"`
  is its own preset (ollama's wire policy, LM Studio's documented address
  `http://localhost:1234/v1`); before, the name missed the address table and
  the request went to `api.openai.com`. The Responses door gains the local
  engines' roots (`ollama`, `lmstudio`, `vllm`, `sglang`). A preset name with
  no known address in the chosen dialect (`qwen` on Chat; `qwen`, `deepseek`,
  `zai` on Responses; the host-templated `bedrock*` outside the registry) now
  raises `NotConfiguredError` at construction unless `base_url=` is given.
  `preset_base_url` in `lm15.compat` is the one lookup all three dialect
  adapters use.

**Error and completion hardening.** HTTP errors retain request IDs from headers
when the provider body did not supply one; invalid/non-finite retry hints are
ignored. `stream_assembly` now maps back to `StreamAssemblyError` consistently.
Stream-to-response helpers require a final end event and reject trailing events
(both `StreamAssemblyError`, with `partial`), close their sources, and preserve
the primary error or cancellation if cleanup also fails. A failure that follows
the end event — the source raising while it drains, or `close()` raising — never
withholds the completed Response: it is returned, the failure is emitted as a
`StreamCleanupWarning` and recorded on `ResponseStream.cleanup_errors` /
`AsyncResponseStream.cleanup_errors` (an earlier revision on `main` re-raised it
as a `StreamAssemblyError` carrying the complete response in `partial`; that
withheld a billed answer over a connection's afterlife and is gone). Use
`StreamAccumulator.response()` explicitly when inspecting an unfinished stream;
materializers no longer report it as success. Contract:
`lm15-contract/changes/2026-09-11-stream-completion-and-error-metadata.md`.

**Three defects found by the Rust port (2026-09-07).** `Config.stop` and `ToolChoice.allowed` given as a bare string were iterated into characters by `from_dict` (INV-020 says one element); `response.usage: null` crashed `response_from_dict` (INV-042 says a telemetry nest reads as absent); the vet shim could not build the Vertex adapter (it passed `compat=None` to the Gemini dialect). All three fixed; no fixture changed.

**Baseline review (2026-09-06).** These changes are not a frozen release.
The shared contract still contains draft amendments and draft canonical
expectations. `CONTRACT_PIN` must move only after that review is complete.

**Ratified 2026-09-06** (`lm15-contract/lm15-contract/changes/2026-09-06-decisions.md`, `changes/2026-09-06-ratification.md`). The paragraphs below implement the decisions by number.

**D1 — AUTH-2 scheme selection.** `select_scheme` implements the ratified table: an `ApiKey` takes the policy's first header-carrying scheme in policy order (`bearer`, `x-api-key`, `api-key`, `query-key`); a `BearerToken` takes `bearer` if the policy lists it, else `x-api-key` if it lists it; `AwsCredentials` take `sigv4` only. Anything else raises `NotConfiguredError`, and the message now names the schemes the credential kind accepts next to the schemes the door offers. Cost, stated: a token handed to first-party `anthropic` goes in `x-api-key` and gets the provider's 401, not a local error.

**D5 — hidden thinking. BREAKING: `ThinkingPart.redacted` is removed**, and `thinking(content, *, continuation=None)` loses the `redacted` keyword; serde neither reads nor writes the key. Hidden thinking is a `ThinkingPart` with empty `text` and its replay state in `continuation` (MAP-7 rule 11): Anthropic `redacted_thinking` becomes `ThinkingPart(text="", continuation=[anthropic:redacted_thinking {"data": blob}])`; in a stream, `ThinkingDelta(text="")` opens the block and the `ContinuationDelta` carries the blob. The `[redacted]` placeholder text is gone everywhere. Replay is unchanged: the blob goes back as `redacted_thinking`.

**D6 — FileReadiness fold.** One table for every OpenAI-shaped file object (`openai`, `azure`, `meta`), in `lm15.providers.common.openai_file_readiness`: `uploaded | pending → pending`, `error | failed → failed`, `processed | absent | unknown → ready`. `failed` is new; the rest is unchanged.

**D7 — continuation namespace.** Verified: every adapter names the dialect (`openai`, `anthropic`, `gemini`) as `ContinuationState.provider`, never the door; a Meta, Azure or Moonshot reasoning item is `openai:reasoning_item`. Pinned by tests; no code change.

**D8 — message-level id continuation. BREAKING:** no dialect emits `openai:response_id`, `gemini:response_id` or `anthropic:message_id` continuation state on the assistant message, on either the complete or the stream path. `Response.id` and `StreamStartEvent.id` carry the id, as before. Nothing in lm15 consumed those states. INV-051 (stream/complete parity) is pinned by a test over every contract stream body: the Response materialized from the stream equals the complete parse of the body the frames fold into, except the fields the wire withholds on one path — the chat dialect's per-chunk `id` and served `model` snapshot, and Gemini's per-chunk `responseId` (no start frame, MAP-4/MAP-9.6). One known gap is recorded as a strict xfail, not repaired: on `gemini.streaming_tool_call` the complete path mints `fc_0` for the id-less call and the assembler mints `tool_call_0` (MAP-9).

**D9 — `StreamEndEvent.provider_data`.** Every dialect's end event carries, verbatim, the wire frame that supplied `usage`, else the frame that supplied `finish_reason`; bare terminators (`[DONE]`, `message_stop`) contribute nothing. New on the chat dialect and its bound providers (the usage chunk, else the finish chunk), on xAI, and on the Anthropic dialect (the `message_delta` frame); unchanged on Responses (`response.completed`'s `response` object) and Gemini (the last chunk). The MAP-3 coalescer now ranks adapter end events so the usage frame's `provider_data` wins over a finish-only frame in either order. It is an escape hatch, not a canonical fact: the harness compares it by presence and type.

**D10 — no delimiter parsing.** Verified: no adapter parses `<think>` / `<reasoning>` tags out of provider text; such text stays literal on every dialect, pinned by tests. Where a server knob separates reasoning (Groq `reasoning_format: parsed`), the preset sends the knob.

**D16 — reference-only.** Async adapters run a credential provider callable through `asyncio.to_thread`, so a cloud-chain refresh does not block the event loop; a static credential is read inline with no thread. `docs/cloud-hosts.md` names the mitigation for the stdlib RSA signer's timing limit: an external token provider (`BearerToken` from your own signer or the cloud CLI) for high-value keys. The reasoning cookbook describes hidden thinking in the D5 shape.

- Preserve old `AccessPolicy` and compatibility-policy constructor calls.
  New options are keyword-only. An explicit legacy `auth_header` overrides
  `auth_scheme`, including in `dataclasses.replace`.
- Fix Vertex routing, direct Responses compatibility binding, and unsupported
  live-endpoint checks.
- Preserve empty thinking items and final replay state in Responses streams
  under MAP-7.8–9. The independent correction covers Meta and Moonshot;
  `changes/2026-09-06-streamed-reasoning-state.md` records the fixture evidence.
- Keep credentials out of redirects, error messages, object displays, and
  unintended subprocess environments. Validate cloud host names, metadata URLs,
  credential response types, and cached expiry.
- Use the configured HOME for both cloud resolution and the doctor.
  Follow AUTH-1: continue through Azure developer-command failures, but stop on
  a failed deployed credential.
- Check out the pinned contract before CI test collection. Reject a changed
  contract checkout, including untracked files.

**Bedrock mantle Chat Completions** (`lm15-contract/changes/2026-09-04-bedrock-mantle-chat-live.md`). New door `bedrock-mantle-chat`: `https://bedrock-mantle.{region}.api.aws/v1`, SigV4 service `bedrock-mantle`, un-versioned model ids, `list_models()` (55 ids, live). Not a rename of `bedrock-chat` (runtime host). gpt-oss returns `message.reasoning` (a thinking part). Gemma honours `tool_choice` here (it ignores it on runtime). gpt-oss still ignores `tool_choice` and `json_schema` — refused per model. Claude on this door is a loud 400 (wrong API); Nova is 404. Trade-off, stated: a tenth cloud door; the original design named nine.

**Bedrock short-term keys** (`lm15-contract/changes/2026-09-04-bedrock-bearer.md`). Fix: with `AWS_BEARER_TOKEN_BEDROCK` set, `bedrock-anthropic` and `aws-anthropic` raised `NotConfiguredError` ("cannot travel") because a `BearerToken` could only go under `Authorization`; those doors carry it as `x-api-key`. A `BearerToken` now travels under `bearer` where the door offers it, else under `x-api-key`; an Entra token on `azure-anthropic` still goes as `bearer`. Trade-off, stated: a token handed to a key-header-only door that does not take tokens (first-party `anthropic`) now gets the provider's 401 instead of a local error. Live: the bearer rung on `bedrock-chat` (HTTP 200, pinned as `cases/bedrock-chat/bearer_basic_text.json`); `GET /openai/v1/models` is 404 under a key too, so `supports.models` stays false.

**Azure OpenAI live** (`lm15-contract/changes/2026-09-04-azure-live.md`, `…-azure-chat-live.md`). `azure` and `azure-chat` are live-verified against a real resource: text, streaming, tools, strict schema, `gpt-5-mini` reasoning with nonzero hidden-token usage, warm prompt-cache hits, both prompt/completion content-filter paths, async parity, and model listing. `azure` also gains live-pinned Files, Batch, WAV speech, and GA Realtime text WebSockets (`supports.files/batches/speech/live = true`); Azure's 201 + `pending` file state now maps honestly and `file_wait_ready()` waits before a useful download. Realtime auth no longer hard-codes OpenAI bearer: Azure API keys use `api-key`, Entra uses bearer. The `azure-chain` passed live by client secret, certificate (standard-library RS256 accepted by Entra), token-provider callable, both Entra scopes, and its actual `az` rung. `GET /openai/v1/models` answers 200 and lists the resource catalog, so `supports.models` is true on both doors. `DeploymentNotFound` maps to `UnsupportedModelError`. `explain_auth(provider, config=router.config)` now describes the router's own env, keys and settings; a JWT handed to a key-first door as a plain string names the required `BearerToken` wrap before the wire. Two quota-gated proofs remain: images (`gpt-image-1-mini` request pending) and successful `azure-anthropic` inference (eastus2 capacity request denied; host, corrected `x-api-key`, all Entra paths, `/models` refusal and errors are live-proven in `changes/2026-09-04-azure-anthropic-partial.md`). Sora/video is deliberately excluded. The batch capture proves submission, status, and cancellation, not a completed result.

**Cloud hosts — AWS, Azure, Google Cloud** (`lm15-contract/changes/2026-09-03-cloud-hosts.md`, documentation-evidenced; no live receipt yet). Nine doors to existing dialects: `azure`, `azure-chat`, `azure-anthropic`, `aws-anthropic` (Claude Platform on AWS), `bedrock-anthropic` (Bedrock mantle, Opus 4.7+), `bedrock-chat`, `vertex`, `vertex-express`, `vertex-anthropic`. A credential is now a value (`lm15.credentials.ApiKey` / `BearerToken` / `AwsCredentials`); a plain string still reads as an API key. Three credential chains in the cloud SDKs' own order (`aws-chain`, `azure-chain`, `gcp-chain`) with an offline doctor that reports `unprobed` for network rungs and prints the resolved host settings. Standard-library SigV4 (all ten AWS test-suite vectors) and RS256 (Google service accounts, Entra certificates; ~75 ms per signature). `AccessPolicy.auth_header` becomes `auth_scheme` (the old constructor argument and property remain supported); every adapter takes `settings=` and `clock=`; `RouterConfig.settings`. Behaviour change, stated: the auth header is applied once per request in `BaseProviderLM._emit` for every dialect — a credential provider callable is now invoked exactly once per request on every surface, including multipart uploads. Stated gaps, each a typed error naming the fix: `aws login` refresh, Azure Service Fabric identity, PKCS#12/encrypted keys, Google `external_account` with an AWS source, Bedrock Converse (phase 2). See `docs/cloud-hosts.md`.

**Moonshot AI (Kimi)** (`moonshotai:` prefix, `MOONSHOTAI_API_KEY` then `MOONSHOT_API_KEY` — the second is the name Moonshot's own docs use; both are vendor-named so both are read — `https://api.moonshot.ai/v1`) joins as a registry provider on the Chat Completions wire, live-verified: 15 cases, 1 pinned refusal, 4 error envelopes, 19 probes (`lm15-contract/changes/2026-09-03-moonshotai-live.md`). New `OpenAIChatThinkingFormat` value `"kimi"`: an effort word goes out as top-level `reasoning_effort` alone (what `kimi-k3` documents) and `effort="off"` as `thinking: {type: disabled}` alone (what `kimi-k2.6` documents; `kimi-k3` honours it too, contrary to its docs). New `OpenAIChatCompat.reasoning_efforts` (a tuple of native levels, default `None`): when a server accepts any effort word without validating it — `kimi-k3` answered 200 to `medium` and to `bogus` alike — a word outside the list raises `UnsupportedFeatureError` before the wire (MAP-7 rule 2); the `moonshotai` preset lists `low, high, max`. Moonshot's `exceeded_current_quota_error` (insufficient balance, HTTP 429) maps to `BillingError`, not the retryable `RateLimitError`.

**Moonshot's other two wires** (`lm15-contract/changes/2026-09-03-moonshotai-wires.md`). `moonshotai-responses` (Responses wire, same root and key): a `moonshotai` Responses preset — stateless server, reasoning replayed as summary text (already what the dialect sends), `web_search` built-in through `builtin_tools="verbatim"` (renamed from `"meta"` the same day: the canonical name is the wire type on both servers, and a shape is not named after its first server), `prompt_cache_key` forwarded; 11 cases, 9 error envelopes. `moonshotai-anthropic` (Anthropic Messages wire at `/anthropic`, bearer token): 10 cases, 3 pinned refusals, 3 error envelopes. Four new `AnthropicCompat` fields, all from that wire's live behaviour: `thinking_format="effort"` (no `thinking` object exists there — `output_config.effort` alone; off is sent as `disabled`, which the server honours), `thinking_replay="unsigned"` (the server returns `signature: ""` and accepts unsigned `thinking` blocks back; the dialect's default still replays unsigned thinking as text), `sampling_params="reject"` (`temperature`/`top_p`/`top_k` are swallowed silently there while the same server's chat wire refuses them), and `reasoning_efforts` (the allowlist, as on the chat compat — `medium` and `bogus` are HTTP 200 there). Anthropic-dialect error mapping learns Moonshot's `resource_not_found_error` (model → `UnsupportedModelError`, as on the chat wire) and `invalid_authentication_error`.

**Meta Model API** (dev.meta.ai) joins as three provider strings on one key (`META_API_KEY`; Meta's vendor-less `MODEL_API_KEY` is deliberately not read), all at `https://api.meta.ai/v1`: `meta` is the Responses wire (Muse Spark with reasoning replay, the `web_search` built-in, and the account surfaces — Files, image generation and editing with `muse-image-1.0`, model listing); `meta-chat` is the Chat Completions wire; `meta-anthropic` the Anthropic Messages wire (bearer token, `thinking: adaptive` on every model). The Responses dialect now binds like the other two: `OpenAILM(compat=…)` takes a preset name or an `OpenAIResponsesCompat`, `OpenAIResponsesCompat.preset` reads the new `OPENAI_RESPONSES_PRESETS` table, and `lm15.registry` can bind that dialect. Three new Responses compat fields: `commentary_phase` (`"tag"` replays assistant text that precedes a function_call with Meta's `phase: "commentary"`), `edit_image_field` (`"indexed"` sends `image[0]`, `image[1]` on `/images/edits`; Meta rejects OpenAI's `image[]`), and `builtin_tools` (`"verbatim"`: the canonical `web_search` goes out unchanged, not as OpenAI's `web_search_preview`). `OpenAICacheControl` gains `"openai_implicit"` (both OpenAI dialects): `prompt_cache_key` and `prompt_cache_retention` are forwarded, the explicit breakpoint mark is never placed — Meta caches every prefix automatically and swallows the mark silently, and lm15 does not send a field a server ignores. `OpenAIChatCompat.user_field` gains `"safety_identifier"`; `AnthropicCompat.thinking_format` gains `"adaptive"` (every model is the adaptive class; an explicit off is sent as `disabled` so the server refuses it loudly). 33 cases across the three wires, 19 error envelopes, images pixel-verified (`lm15-contract/changes/2026-09-03-meta-live.md`). Speech-to-text (Muse Voice Transcribe) is not wired: lm15 has no transcription surface, and Meta's is its own WebSocket protocol.

**One provider registry.** `lm15.registry.PROVIDERS` is the one table of named providers: id, wire dialect, access policy (`lm15.access`), Chat Completions compat preset, keyless placeholder, console URL. `ADAPTERS`, `ASYNC_ADAPTERS`, and `CHAT_PRESET_ROUTES` are now read-only views of it (same keys, same values, now `MappingProxyType`); `OpenAIChatCompat.preset` reads the `OPENAI_CHAT_PRESETS` table instead of an if-chain; the doctor and the vet surface dump walk the registry, so the contract's support matrix now pins every routable provider, not only the seven adapter classes. Behavior change for routed chat presets: `router.lm("groq:…")` binds the `groq` access policy, so the LM's `provider` is `"groq"` (was `"openai_chat"`) in errors and `ModelInfo.provider`, and `lm.access.env_keys` names `GROQ_API_KEY`. Constructing `OpenAIChatLM(compat="groq")` directly is unchanged.

**DeepSeek over the Anthropic wire** (`deepseek-anthropic:` prefix, same `DEEPSEEK_API_KEY`, `https://api.deepseek.com/anthropic/v1`) is the registry's first binding on a second dialect. New `AnthropicCompat` (small: `thinking_format` anthropic | deepseek, `cache_control`, `structured_output`, `parallel_tool_calls`, `model_prefixes`), `AnthropicLM(compat=…)`, presets `anthropic` and `deepseek`. On this endpoint an explicit `effort="off"` is sent as `thinking: {type: disabled}` (absence means on there), effort rides `output_config.effort` (budget_tokens is ignored, so a `thinking_budget` raises), a JSON schema and `tool_choice.parallel` raise (silently ignored server-side), and `claude-*` model names raise `UnsupportedModelError` because the endpoint silently serves them with DeepSeek models (live 2026-09-03). Plain `AnthropicLM` is unchanged. 9 cases, 4 pinned refusals, 3 error envelopes (`lm15-contract/changes/2026-09-03-deepseek-anthropic-live.md`).

**Z.AI (GLM)** (`zai:` prefix, `ZAI_API_KEY`, `https://api.z.ai/api/paas/v4`) is live-verified: 10 cases, 2 pinned refusals, 4 error envelopes (`lm15-contract/changes/2026-09-03-zai-live.md`). The `zai` compat preset was wrong and is rewritten from the wire: Z.AI's thinking switch is `thinking: {type}` + `reasoning_effort` (the `deepseek` shape), not Qwen's `enable_thinking`; the never-validated `"zai"` value is removed from `OpenAIChatThinkingFormat`. Two new compat fields, both `"send" | "reject"`: `forced_tool_choice` (Z.AI honours `auto` only and silently ignores the rest — `required` answered text, `none` called the tool) and `json_schema` (Z.AI returns 200 with free-form fenced JSON). The `zai` preset rejects both, so lm15 raises `UnsupportedFeatureError` before the wire. Z.AI code `1113` ("insufficient balance", sent on HTTP 429) now maps to `BillingError`; it was `RateLimitError`, which is retryable.

**DeepSeek** (`deepseek:` prefix, `DEEPSEEK_API_KEY`, `https://api.deepseek.com`) joins as the first registry-only provider: a declaration plus a compat preset, no class. The preset now replays `reasoning_content` natively and sends it on every assistant turn when tools are present — DeepSeek answers 400 otherwise (captured verbatim, `errors/cases/deepseek.json`). Live-verified 2026-09-03: eleven cases, four error envelopes, thirteen probes (`lm15-contract/changes/ 2026-09-03-deepseek-live.md`). New `OpenAIChatCompat.user_field` (`"user"` | `"user_id"`, default `user`): `Config.user_id` rides the server's documented field name; the `deepseek` preset sends `user_id`.

**One truth per support fact.** The adapter-level `Capabilities` object (`lm.capabilities`: free-text `features`, adapter-wide modalities) is removed. Nothing in lm15 read it, it was not in the contract, and it had already drifted (Gemini gained reasoning under MAP-7; its `features` set said otherwise). Endpoint support lives in `lm.supports` / `lm.manifest`, pinned by `spec/support-matrix.json`; modalities and prices are per model on `ModelInfo`. Callers reading `lm.capabilities` get `AttributeError`.

**Three follow-ups from the independent review.**
- `LiveServerUsageEvent` (`type: "usage"`): billed tokens of a live response that does not end the turn — a tool-call response (75 tokens were vanishing in the pinned transcript) or a cancelled one (143). `Turn.usage` now sums every `usage` and `turn_end` event it saw; an interrupted turn keeps its usage instead of `None`.
- Gemini modality breakdowns (`promptTokensDetails`, `candidatesTokensDetails`, `responseTokensDetails`, modality `AUDIO`) now fill `input_audio_tokens` / `output_audio_tokens`, as OpenAI's do.
- Usage counters are declared provider-verbatim in the contract, with a per-provider table of what `input_tokens` and `output_tokens` include. Nothing changed in the numbers; the rule that they differ is now text.

**Caching on gpt-5.6+: two amendments from live probes** (MAP-6 rules 4 and 5). `retention="long"` no longer raises on the 5.6 class; it sends `prompt_cache_retention: "24h"` like every other class (the server accepts and echoes it; every 5.6 body already echoed 24h as default). A placed breakpoint (`prefix="stable"`, `prefix_until_index`) now travels with `prompt_cache_options: {mode: "explicit"}` on 5.6+: without it the warm call still wrote the volatile suffix at 1.25x; with it the warm call writes 0 and the cold write is exactly the marked prefix.

**Anthropic streamed tool calls assembled an unparseable input.** The `content_block_start.input: {}` placeholder was serialised and glued in front of the `input_json_delta` fragments. Fixed; the first streaming tool-call body in the corpus caught it.

**Auth by composition** (contract AUTH-10, proposed). An adapter is a dialect bound to an `AccessPolicy` value (`lm15.access`; `ProviderManifest` is the same class under its earlier name): credential policy, auth header, static headers, login hint, endpoint surfaces, backend variant, system prefix, base URL. `ClaudeCodeLM` and `OpenAICodexLM` are now names for `AnthropicLM(access=CLAUDE_CODE)` and `OpenAILM(access=OPENAI_CODEX)` and define nothing but constructors; `XaiLM` composes its credential path and keeps its provider wire (images, video, refusals). Every dialect and async mirror takes `access=` and `credentials_path=`; `api_key` is optional and a `key` policy with no key raises `NotConfiguredError` naming the env keys (was `TypeError`). Endpoint surfaces are gated on the bound policy in the shared drivers, so a subscription login that lacks files/batch raises before any hook. Wire output is byte-identical (harness 13/13).

**Stream assembly never invents a tool-call name** (MAP-9, ErrorCode `stream_assembly`). When a streamed tool call's fragments never carried a name, the accumulator used to guess one from `Request.tools` by position and could dispatch the wrong function silently. It now raises `StreamAssemblyError`, carrying `partial` (the Response assembled from everything else) and `part_index`. `ResponseStream` raises at the end of iteration; text already yielded stays yielded. No shipped dialect triggers this — every one names a call on its first fragment — so the change is visible only to code that fed hand-built events into the accumulator.

**Honest usage counters and a silent `[DONE]`** (INV-029, MAP-3):

- Adapters no longer write `0` for `input_tokens`/`output_tokens` the provider did not report; absent stays `None` and `total_tokens` is summed only when both primaries are present. Callers that summed usage across calls and relied on the invented zeros will now see `None` where the provider said nothing. Gemini is the one stated exception: proto3-JSON omits zero-valued fields, so an absent primary inside a present `usageMetadata` is a reported `0` (pinned by the reviewed golden `gemini.max_output_tokens`); a missing `usageMetadata` is all `None`.
- The Responses dialect's bare `[DONE]` terminator no longer claims `finish_reason="stop"`. Before, on the Codex backend it overwrote the `tool_call` from `response.completed` in the coalesced end event, so the event trace contradicted the materialized `Response`.

**Auth hardening + login primitives + doctor** (contract `lm15-contract/spec/auth.md`, ratified 2026-08-31; fixtures `auth/resolution.json`):

- `lm15.auth`: credential-file writes are now atomic (temp + rename,
  0600) and serialized by a cross-process advisory lock; token refresh is double-checked under the lock, so a refresh completed by another process is used instead of repeated (repeating it loses rotated refresh tokens). Lock contention raises the new `CredentialLockTimeout` (a `TimeoutError`, deliberately not `AuthError`). Locks live in `$XDG_CACHE_HOME/lm15/locks` (`$LM15_LOCK_DIR` overrides), never inside `~/.claude`/`~/.codex`. Stated trade-offs: the lock is advisory and lm15-cooperative only (foreign CLIs do not take it; the double-checked re-read is the mitigation), and refresh holds the lock across the network call (a slow refresh can stall sibling lm15 processes; the alternative double-spends rotated refresh tokens).
- New `lm15.authkit`: login-flow primitives for apps that own a login UX — PKCE (S256 only, RFC 7636 vector pinned), the RFC 8628 device-code polling state machine (injectable clock/sleep), `OAuthCallbackListener` (one-shot loopback listener, 127.0.0.1 only), and `CredentialFileStore` (locked, atomic, 0600, keyed by provider, serialized `mutate`; default `$XDG_CONFIG_HOME/lm15/credentials.json`, `$LM15_CREDENTIALS_PATH` overrides).
- New `lm15.doctor.explain_auth`: rung-by-rung credential-resolution report (selected / shadowed / absent) mirroring the router's exact chain; no network, secret values never rendered. Purity trade-off vs `resolve()`: it tests env vars for presence, so values transit memory but are never retained or shown.
- Contract: `spec/auth.md` (AUTH-1..9, ratified 2026-08-31), language-neutral resolution fixtures (mirrored at `conformance/auth_resolution.json`, run by `tests/test_auth_resolution_contract.py`), and a corpus-wide secrecy CI gate (`tools/check_secrecy.py`). Ports are not yet updated; they are formally behind the contract on this surface until they implement AUTH-1..AUTH-9 against `auth/resolution.json`.

**The API review (breaking — the alpha's one-time window).** A four-lens fresh-eyes panel reviewed the public surface; findings in `architecture-review/api-review-2026-07-13.md`. The breaking set, batched here so the alpha churns exactly once:

- **`Result` → `ResponseStream`**, constructor positional: `ResponseStream(router.stream(req), req)` — no more keyword-only double threading. `.events()` yields canonical `StreamEvent`s (the `StreamChunk` second vocabulary is gone); accessors mirror Response's minimal set. New: `StreamAccumulator` (the shared push-based engine), `AsyncResponseStream` (a true async mirror — the old `AsyncResult`, which could not consume async streams, is deleted), `amaterialize_response`. `lm15/stream.py` (which aliased `Result` as `Stream`) is gone; both MAP-3 coalescer twins live in `lm15.result`.
- **Namespace curation, 161 → 107 top-level exports.** Serde pairs → `lm15.serde`; adapter machinery (BaseProviderLM, transports protocols, Credential, HttpResponse) → `lm15.providers`; error machinery → `lm15.errors`; router data tables (DEFAULT_RULES, ADAPTERS, CHAT_PRESET_ROUTES, RouteRule, PresetRoute) → `lm15.router`; profile/compat/SSE machinery → their modules. Promoted: `RETRYABLE_ERRORS` and `tool_result` to the top level. `derive` is exported as `derive_tool` (collision doctrine).
- **`ProviderLM` now names the callable surface** (complete/stream/…); the wire-mapping protocol formerly exported under that name is `lm15.providers.ProviderDialect`.
- **Provider strings are hyphenated**: `openai-chat` is canonical, `openai_chat` remains a permanent alias everywhere; `Resolution.provider` reports the canonical spelling.

Additive, same review:

- `Message.tool(call_id, output, is_error=False)` positional spelling; every wrong shape now raises a TypeError listing the accepted forms.
- `Request.tools` accepts a bare tool (1-tuple coercion).
- Errors state their cure: the messages TypeError names `Message.user`; `ProviderError.__str__` appends `(provider, HTTP status, request id)`; `UnknownModelError` uses a neutral prefix example and hints near-miss provider heads ("did you mean…").
- **`lm15.testing`**: `FakeLM` (canonical-level double), `FakeTransport`/`FakeResponse` (wire-level, promoted from the test suite). `RouterConfig(transport=...)` injects a transport into every LM the router builds.
- **`Retry-After` is parsed** (delta-seconds and HTTP-date) into `error.retry_after` on both `complete()` and `stream()` paths; provider-body values win.

**One front door: credential providers + universal routing.**

- **Credential providers.** `api_key` on every adapter (sync and async) now accepts a zero-argument callable as well as a string, resolved at request-build time, once per request — Azure Entra `get_bearer_token_provider(...)` output plugs in verbatim; rotating keys never go stale in long-lived clients. Acquisition stays the caller's job: lm15 gains no auth dependencies. `RouterConfig.api_keys` values may be credential providers too.
- **Subscription freshness.** `ClaudeCodeLM`/`OpenAICodexLM` (and async mirrors) validate the local CLI credential at construction, then re-resolve it per request — tokens refreshed on disk are picked up without rebuilding the client.
- **Credential hygiene.** `api_key` is repr-suppressed on all adapters (previously the plain adapters' dataclass repr included it).
- **Router rung 0 (object attribute).** A model value carrying a non-empty string `provider` attribute (catalog packages ship these — aimo's model objects) resolves directly when the provider is routable; duck-typed, no package named. Bare-id ambiguity disappears when the model object knows its provider.
- **Router preset routes.** `groq`, `openrouter`, `ollama`, `vllm`, and `sglang` are now routable provider strings — prefix, catalog, object attribute, or rules — landing on `OpenAIChatLM(compat=<preset>)` with the preset's pinned base_url, the server's own env-key convention (`GROQ_API_KEY`, `OPENROUTER_API_KEY`), and keyless placeholders for local servers. `Resolution` gains a `compat` field; `describe()` narrates the new rungs.
- New exports: `Credential`, `resolve_credential`, `PresetRoute`, `CHAT_PRESET_ROUTES`.

## 1.0.0a1 — 2026-06-11

**The stability promise.** The chat core — canonical types, serde, errors, request building, response parsing, streaming — is frozen; all future changes to it are additive (enforced mechanically by the surface ratchet and spec drift gate). Non-chat endpoints and live sessions remain provisional; see `lm15-contract/spec/SCOPE.md`.

What backs the promise:

- **Four independent implementations** — Python (this package), Rust, Go, TypeScript — each passing the identical 304-check conformance corpus (`lm15-dev/lm15-contract`), each live-tested against real providers including the full tool-calling round-trip.
- **A written, ratified spec**: 61 types, 25 vocabularies, 49 numbered invariants, mapping rules MAP-1..3, one omission rule, one number rule.
- **Every fixture carries provenance**; wire fixtures change only with live receipts; the reference implementation holds no oracle authority.
- **Measured, regenerable benchmarks**: 0 dependencies, 0.5 MiB installed, 171 ms cold import, and faster than raw stdlib HTTP at steady state (connection pooling).

Changes since 0.3.0: prompt-caching fixtures recaptured (GA, no beta header); OpenAI file inputs send `filename` (provider drift caught by the live sweep); `FunctionTool.parameters` always emitted, `{}` round-trips verbatim; malformed nested config objects reject instead of silently dropping; `Result` and live sessions no longer contain any automatic tool-execution machinery.

## 0.3.0 — 2026-06-11

Ground-up rewrite. `lm15` is now a **low-level foundation library**: one canonical representation, exact serde, provider adapters — and nothing opinionated. The 0.2.x high-level API (`lm15.call()`, `Model`, `Conversation`, cost tracking, middleware, REPL) is **gone by design**; build it (or your own take) on top. Pin `lm15==0.2.*` if you depend on the old surface.

### The canonical core
- Typed, frozen, immutable canonical model: `Request`/`Response`, `Message`, typed `Part`s (text, thinking, media, tool calls/results, citations), `Config`, `Usage`, stream events.
- Exact canonical JSON serde with written rules: one omission rule, opaque payloads never mutated, declared number types (`serde-rules.md`).
- Normalized error hierarchy (`AuthError` with key/credential guidance, `RateLimitError.retry_after`, `ContextLengthError`, ...).
- Mapping invariants written and pinned: provider-executed tools are not parts (MAP-1), response messages are never empty (MAP-2), a stream yields exactly one end event carrying finish_reason and usage (MAP-3).

### Providers
- First-party adapters: OpenAI (Responses), Anthropic, Gemini.
- `OpenAIChatLM`: the Chat Completions dialect with compat presets for ollama, Groq, OpenRouter, vLLM, SGLang — live-validated against Groq, ollama, vLLM, and SGLang.
- Native async mirrors of every adapter (`AsyncOpenAILM`, ...): same constructor, same canonical types, no thread-wrapping.
- Local subscription adapters: `ClaudeCodeLM` (Claude Code OAuth) and `OpenAICodexLM` (Codex/ChatGPT OAuth).
- Stdlib-only HTTP/1.1 sync + async transports; `websockets` is the single optional extra (live sessions).

### Conformance
- Behavior is pinned by the cross-language `lm15-contract` corpus: 108 request cases, 108 reviewed response/stream goldens, error and serde vectors, all live-captured or hand-authored with provenance, verified by a language-neutral harness (`python -m lm15.vet`).
- A written spec (types, vocabularies, 48 numbered invariants) with a reflection-based drift gate.

### Optional model metadata
- `ModelRegistry.discover()` hydrates advisory pricing/context metadata from installed catalogs (entry-point group `lm15.model_catalogs`); never affects what adapters send.

## 0.2.0 and earlier

The previous-generation high-level SDK, developed in the `lm15-python` repository. See its history there.
