## Where This Breaks Down

The universal part type is a useful fiction. It says: "a text part from OpenAI and a text part from Anthropic are the same thing." For text, this is true enough. For tool calls, it's mostly true. For images, it's approximately true. For certain provider-specific features, it's a lie.

The lies are worth cataloging, because each one reveals the limits of normalization — the point where "universal" stops meaning "the same" and starts meaning "close enough."

### Cache Control Markers

Anthropic's API allows `cache_control` annotations on individual content blocks:

```json
{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}
```

This tells the provider to cache this block's content for reuse across calls. It's not a content type — it's metadata *on* a content type. lm15 models this as a `metadata` dict on `Part`:

```python
Part(type="text", text="...", metadata={"cache": True})
```

The Anthropic adapter reads `part.metadata` and emits `cache_control` in the wire format. The OpenAI adapter ignores it (OpenAI caches automatically). The Gemini adapter translates it into a `CachedContent` API call (a completely different mechanism).

The universal type says "cache=True." The reality is three different caching mechanisms with different behaviors, different costs, different lifetimes, and different granularity. The `Part` representation hides the *mechanism* but can't hide the *behavior*. A developer who sets `cache=True` and expects identical behavior on all three providers will be surprised. The universal type promised equivalence it can't deliver.

### Image Detail Levels

OpenAI's image content blocks have a `detail` parameter:

```json
{"type": "image_url", "image_url": {"url": "...", "detail": "low"}}
```

`"low"` uses fewer tokens (faster, cheaper). `"high"` uses more (better analysis). Anthropic and Gemini have no equivalent — images are processed at a single, non-configurable resolution.

lm15 puts `detail` on `DataSource` — the sub-object describing media data. This is honest but awkward. The field exists on every `DataSource`, including ones destined for providers that ignore it. A developer setting `detail="low"` on an image sent to Claude is writing dead code that compiles, runs, and does nothing. No error, no warning, no indication that the parameter was ignored.

### Tool Call Structure

OpenAI's tool calls encode arguments as a JSON *string*:

```json
{"function": {"name": "search", "arguments": "{\"query\": \"tcp\"}"}}
```

Anthropic's tool calls encode arguments as a JSON *object*:

```json
{"name": "search", "input": {"query": "tcp"}}
```

lm15 normalizes both to `Part(type="tool_call", input={"query": "tcp"})` — a parsed dict. The adapters handle the string-vs-object difference internally. This normalization is clean and lossless. But it hides a behavioral difference: OpenAI's JSON-string encoding means the model can produce malformed JSON (unbalanced braces, trailing commas), which the adapter must handle. Anthropic's object encoding is parsed by the provider before it reaches the adapter, so malformed arguments never arrive. The universal type presents both as a clean dict, but the failure modes differ — and the developer debugging a malformed tool call on OpenAI won't find the problem in lm15's types, because lm15 already parsed the string into a dict (or failed to, silently).

### Thinking Blocks

Anthropic returns `thinking` content blocks with a `redacted` field — indicating that the thinking was too sensitive to expose and was replaced with a summary:

```json
{"type": "thinking", "text": "[redacted]", "redacted": true}
```

OpenAI doesn't expose thinking blocks at all in some models, and when it does, there's no redaction concept. Gemini returns thinking differently again.

lm15's `Part(type="thinking", text="...", redacted=True|None)` carries the redaction flag. It's meaningless on OpenAI. It's potentially meaningful on future Gemini models. It's a field that exists for one provider today and might matter for others tomorrow — or might never matter for anyone else. Including it is a bet that the concept generalizes. Excluding it loses information from Anthropic's responses.

### The Pattern

Each of these cases follows the same pattern: the universal type includes a field or behavior that's meaningful on one provider and meaningless (or different) on others. The universal type pretends the field is general. The reality is that it's provider-specific information traveling in a universal container.

This is not a failure of lm15's design. It's a fundamental limitation of universal representations. The providers differ in real ways — not just in JSON syntax, but in capabilities, behaviors, and concepts. A universal type can normalize the syntax. It can't normalize the semantics. `cache=True` means one thing on Anthropic and a different thing on Gemini, and no amount of type design can make them the same thing, because they *aren't* the same thing.

The escape hatch — `config.provider` for requests, `resp.provider` for responses, `part.metadata` for content — is the library's acknowledgment of this limitation. Provider-specific information exists. The universal type can't express it without lying. The escape hatch carries it without lying — as untyped, undocumented, explicitly-provider-specific data. The developer who needs it reaches through the universal abstraction to the provider-specific reality. The developer who doesn't never sees it.

This is the right design, but it deserves a name: **honest normalization**. Normalize what can be normalized (text, ordering, role structure). Acknowledge what can't (caching mechanisms, resolution controls, encoding differences). Carry the un-normalizable in an escape hatch rather than pretending it doesn't exist or forcing it into the universal type. The result is a universal type that's genuinely universal for the things it models, and explicitly non-universal for the things it can't.
