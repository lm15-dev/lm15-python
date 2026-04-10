## Where the Map Is Accurate

Before cataloging the distortions, it's worth recognizing how much of the map is
genuinely accurate. There's a large zone where normalization works honestly —
where the same parameter produces the same behavior across providers, and the
adapter's translation is lossless. This zone is larger than most critics of
multi-provider libraries acknowledge, and it covers the majority of what most
applications need.

**Text completion.** Send a system prompt and a user message, receive text. The
request structure differs across providers — OpenAI puts the system prompt in
the messages array as a `system` role, Anthropic puts it as a top-level `system`
field, Gemini wraps it in a `systemInstruction` object. But the adapter handles
this mapping completely. The user writes `system="Be concise."` and it reaches
the model correctly, on all three providers, with no behavioral difference. The
response is text in a message. The text is the same type on all providers. The
normalization is real.

This isn't trivial. It means that the most common operation — send a prompt, get
text — works identically across providers with zero asterisks. A developer who
uses only `complete(model, prompt, system=...)` can switch providers by changing
the model string and changing nothing else. This is the genuine value
proposition of a multi-provider library, and it's fully delivered.

**Tool call round-trips.** The protocol — model requests a call, application
executes, result goes back — is the same on all three providers. The wire format
differs (OpenAI uses `tool_calls` on the message with arguments as a JSON
string; Anthropic uses `tool_use` content blocks with arguments as a dict;
Gemini uses `functionCall` parts), but the semantic structure is identical:
name, arguments, ID. The adapter translates cleanly. The user's tool function
runs the same way regardless of provider. The result goes back the same way. A
tool-using application works across providers without modification.

The reason this normalizes well is **structural convergence**. All three
providers independently arrived at the same tool-calling protocol. Not the same
wire format — the same *protocol*. Model requests action, application acts,
result returns. When the underlying concepts align, the normalization is
mechanical: rename fields, restructure JSON, done. No information is lost, no
behavior changes, no lies are told.

**Token usage.** All three providers report `input_tokens`, `output_tokens`, and
the total. The numbers mean the same thing — how many tokens were consumed in
each direction. The `Usage` object maps 1:1 across providers. A developer
computing cost across providers can use the same formula.

**Error classification.** HTTP 429 is a rate limit everywhere. 401 is bad
authentication everywhere. 400 is a malformed request everywhere.
`map_http_error(status, message)` produces the same error class regardless of
provider, because HTTP status codes are the one truly universal standard in the
entire stack. The error *messages* differ (each provider phrases their errors
differently), but the *classification* is consistent.

These four features — text, tools, usage, errors — account for roughly 80% of
what most applications do with an LLM. A multi-provider library that only
normalized these four things would be genuinely useful, because the
normalization would be genuinely honest. The map would match the territory for
the features that matter most.

The trouble starts with the other 20%.
