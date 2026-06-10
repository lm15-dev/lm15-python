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

---

History: both rules were implicit in the reference adapters; they were
ratified as written rules on 2026-06-10 after the adversarial golden review
flagged anthropic.container, openai.code_interpreter (MAP-1) and
gemini.max_output_tokens (MAP-2) — see
`lm15-contract/goldens/REVIEW-2026-06-10.md`.
