# Canonical mapping rules

Normative rules for mapping provider responses into the canonical lm15
representation. Companion to `serde-rules.md` (which governs the JSON wire
format); these govern WHAT becomes a canonical part. Goldens and conformance
fixtures cite these rules by number.

## MAP-1 — Parts are what the application must act on

A canonical message part is something the application must handle or display:
text, thinking, citations, media, a client-side tool call (which obligates the
caller to execute it and return a tool_result), or a tool result.

Provider-executed builtin tool activity is **not** represented as parts. This
includes Anthropic `server_tool_use` / `code_execution_tool_result` blocks and
container metadata, OpenAI `code_interpreter_call` / `web_search_call` items,
and Gemini `executableCode` / `codeExecutionResult` parts. The user-relevant
*outputs* of such tools (answer text, citations, generated media) are mapped
to parts as usual; the execution mechanics remain available verbatim in
`provider_data`.

**Why:** a `tool_call` part is a contract — "the caller must execute this."
Agent loops iterate tool_call parts and run them. Surfacing provider-executed
calls as tool_call parts would cause every agent loop to re-execute work the
provider already performed. If canonical access to execution traces is needed
later, it must be a NEW part type (additive), never a reinterpretation of
tool_call.

## MAP-2 — A response message is never empty

When a provider response yields no canonical parts (e.g. the model spent its
entire output budget on hidden reasoning and was truncated), the canonical
message is a single empty `TextPart` (`text: ""`).

**Why:** `Message.parts` is non-empty by invariant, everywhere, for every
producer — relaxing it for one edge case would weaken a guarantee all ports
and consumers rely on. Erroring would turn a legitimate provider response into
a crash. With the empty part, `response.text == ""` plus the finish_reason
(e.g. `"length"`) reads as exactly what happened.

## MAP-3 — A stream yields exactly one end event, and it is final

An lm15 stream yields EXACTLY ONE `StreamEndEvent`, as the final event of the
stream, carrying the `finish_reason` and `usage` accumulated across all of the
provider's terminal frames.

Providers split their terminal data across multiple frames: OpenAI-compatible
servers (vLLM, SGLang, ollama, Groq) send a `finish_reason`-bearing chunk,
then — with `stream_options.include_usage` — a usage-only chunk, then
`[DONE]`; Anthropic sends `message_delta` (stop_reason + usage) followed by a
bare `message_stop`. Adapters stay stateless and may emit one per-frame end
event for each such terminal frame, but that is an internal detail: a
provider-agnostic coalescer (`lm15.result.coalesce_stream`) absorbs every
adapter end event — later non-`None` fields fill gaps, a non-`None` field is
never overwritten by `None` — and emits the single merged end event once the
underlying iterator is exhausted. The canonical event trace (goldens, the vet
shim's `replay_stream`, conformance `parse_stream`) is the POST-coalesce
trace.

**Why:** multiple end events made every consumer's merge semantics
load-bearing, and they failed in live testing. `Result` treated the first end
event as terminal (`break` on `type == "end"`), so the post-finish usage-only
chunk that vLLM/SGLang/ollama send was never applied and the materialized
`Response.usage` came out all zeros (pinned as a known-bug baseline in the
streaming_vllm/streaming_sglang draft goldens before this rule). With exactly
one final end event, "the end event" and "the stream's finish_reason and
usage" are the same thing by construction, in every port.

---

History: MAP-1 and MAP-2 were implicit in the reference adapters; they were
ratified as written rules on 2026-06-10 after the adversarial golden review
flagged anthropic.container, openai.code_interpreter (MAP-1) and
gemini.max_output_tokens (MAP-2) — see
`lm15-contract/goldens/REVIEW-2026-06-10.md`. MAP-3 was written on 2026-06-10
after live vLLM/SGLang/ollama testing showed the multi-end merge losing usage.
