## Four Approaches

The representation problem has been solved at least four times, each solution adequate for its era and broken by the next. Seeing them in sequence is instructive — not as history, but as a series of design bets encountering reality.

### String In, String Out

The original OpenAI completions API:

```
prompt: "Summarize TCP in one sentence."
→ completion: "TCP is a reliable transport protocol that ensures ordered delivery."
```

Two strings. No roles, no structure, no metadata. The representation is the content. This works when the interaction is one-shot question-answering and the content is text. It breaks the moment you need multi-turn conversation (who said what?), or the moment the content isn't text (images, tool calls).

No modern library uses this representation, but every modern library *starts* here conceptually. `lm15.complete("gpt-4.1-mini", "Hello.")` looks like string-in, string-out. The simplicity of the surface hides the structured types underneath. This is intentional — most calls *are* string-in, string-out, and the representation shouldn't force complexity on the common case.

### Role + Content String

The chat completions revolution:

```json
{"messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Summarize TCP."},
    {"role": "assistant", "content": "TCP is a reliable..."}
]}
```

Three roles. Ordered messages. Content is a string. This representation supports multi-turn conversation — roles distinguish speakers, ordering preserves temporal structure. It was clean and sufficient for text-only chat.

It broke on two fronts simultaneously. First, images: a user message might contain text *and* a photo. The content string can't hold both. Second, tool calls: an assistant message might contain text *and* a structured function request. The content string can't hold both either. The single-string assumption — that content is one thing of one type — shattered when content became heterogeneous.

The failure mode is revealing. You could patch the representation by adding parallel fields — `content` for text, `image_url` for images, `tool_calls` for tool requests. OpenAI did exactly this in an intermediate version of their API: `tool_calls` became a separate field on the assistant message, alongside `content`. This works, but it destroys ordering. If the model generates text, then a tool call, then more text, the flat structure can't express the sequence. You know all three exist. You don't know the order they appeared.

### Role + Content Blocks

The current generation. Content becomes an array of typed blocks:

```json
{"role": "assistant", "content": [
    {"type": "thinking", "text": "Let me verify..."},
    {"type": "text", "text": "The tip is 14.2%."},
    {"type": "tool_use", "id": "call_1", "name": "calculate",
     "input": {"expression": "12.00 / 84.50 * 100"}}
]}
```

Each block has a `type` discriminator and type-specific fields. The array preserves ordering. New content types — thinking, citations, refusals, audio — are new block types, added to the array without changing the message structure. This representation can express anything, in any order, in any combination.

OpenAI, Anthropic, and Gemini all converged on this shape independently. The names differ (`content` vs `parts`, `tool_use` vs `functionCall`), the type strings differ, the sub-structures differ — but the pattern is identical: an ordered array of discriminated objects. This convergence is evidence that the representation is close to the natural structure of the problem. When three independent teams solve the same problem the same way, the solution has something right.

But the content-blocks representation is provider-specific. OpenAI's blocks have different type names, different field names, and different structural conventions than Anthropic's. A `tool_use` block from Anthropic can't be dropped into an OpenAI request without translation. The representation solved the heterogeneous-content problem but didn't solve the cross-provider problem.

### Role + Universal Parts

The multi-provider library's answer: a universal part type that mirrors the content-block pattern but normalizes across providers.

```python
Message(role="assistant", parts=(
    Part(type="thinking", text="Let me verify..."),
    Part(type="text", text="The tip is 14.2%."),
    Part(type="tool_call", id="call_1", name="calculate",
         input={"expression": "12.00 / 84.50 * 100"}),
))
```

This is lm15's representation. It has the same shape as the provider-level content blocks — ordered sequence, discriminator field, type-specific payload — but the types are provider-agnostic. A `Part(type="text")` means the same thing regardless of whether it came from OpenAI, Anthropic, or Gemini. A `Part(type="image")` carries its data in a universal `DataSource` that any adapter can translate to the provider's format.

The translation to any provider's wire format is mechanical: iterate the parts, switch on the type, emit the provider's equivalent block. `Part(type="text")` → `{"type": "text", "text": "..."}` on all three. `Part(type="tool_call")` → `{"type": "function", "function": {...}}` on OpenAI, `{"type": "tool_use", ...}` on Anthropic, `{"functionCall": {...}}` on Gemini. The structure matches; only the names change.

### What the Progression Shows

Each representation was adequate for its era:

- Strings handled text-only, one-shot interactions.
- Role + string handled multi-turn text conversations.
- Role + content blocks handled multimodal, multi-type messages.
- Role + universal parts handled cross-provider portability.

Each one broke when the demands exceeded its assumptions: strings broke on roles, roles broke on images, content blocks broke on portability. The progression isn't random — it's driven by a monotonic increase in what messages need to express. At each step, the representation absorbed more structure to handle more complexity, and the absorption always took the same form: wrapping the content in one more layer of typed container.

The bet implicit in the universal-parts approach is that this progression is over — that an ordered sequence of discriminated parts is the terminal representation, capable of absorbing whatever content types arrive next. That bet might be right. It's hard to imagine a content type that can't be expressed as a Part with a new type value. But it's worth noticing that every previous representation made the same bet ("this is enough"), and every previous representation was wrong.
