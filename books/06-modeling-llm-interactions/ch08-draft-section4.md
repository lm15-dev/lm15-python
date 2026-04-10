## Fewer Layers

Five layers is not the only viable architecture. LiteLLM has three: a routing function, translation functions, and HTTP calls. No model objects, no middleware pipeline, no transport abstraction. The user calls `completion()`, the function routes to a provider-specific translator, the translator makes the HTTP call, and the response comes back.

```python
# LiteLLM's effective architecture
completion("claude-sonnet-4-5", messages=[...])
    │
    ├── route by model name
    ├── anthropic_chat_completions(messages, ...) → HTTP → parse
    └── return response
```

Three layers. One function entry point. One translation layer (per provider). One HTTP mechanism (inside the translation functions). Everything that lm15 distributes across five layers — state management, middleware, transport abstraction — either doesn't exist (no conversation state, no middleware) or is inlined (HTTP calls inside the translation functions).

This is simpler. Fewer files, fewer modules, fewer concepts. A developer reading LiteLLM for the first time understands the architecture faster than one reading lm15, because there's less architecture to understand. The routing function calls a translation function. The translation function calls the API. Done.

What LiteLLM gives up:

**No stateful objects.** The user manages conversation history. There's no `model()` that accumulates turns. No `submit_tools()` that continues a tool round-trip. The user builds the message list, passes it on every call, and appends the response themselves. This is the OpenAI SDK pattern — maximum control, maximum boilerplate. It works, but the absence of state management means the user reinvents it in every project.

**No middleware.** Retries, caching, and logging live in the user's code or in the translation functions. There's no composable pipeline where you add `with_retries()` and get transparent retry logic. If LiteLLM wants retries, it builds them into the routing function. If the user wants different retry logic, they wrap the call themselves.

**No transport swapping.** The HTTP implementation is inside the translation functions. You can't swap `urllib` for `pycurl` without modifying every translator. You can't inject a mock transport for testing without mocking at a higher level (patching the HTTP call inside the function). The transport isn't a separable concern.

Each of these is a feature that lm15 puts in a separate layer and LiteLLM puts... nowhere. Or in user code. Or inlined. The choice isn't "five layers vs three layers." It's "five concerns separated vs three concerns separated and two pushed elsewhere."

### The Layer Count Principle

The right number of layers is the number of independent change vectors in the system.

If you have three things that change independently — the user-facing API, the provider translation, and the HTTP mechanism — you need three boundaries. LiteLLM has these three, and the boundaries work: you can add a provider without changing the routing function, and you can change the routing without changing the translators.

If you have five things that change independently — user API, conversation state, provider routing, provider translation, and HTTP — you need five boundaries. lm15 has these five, and each boundary isolates its change vector.

If you have two things that change independently, three layers is too many — one boundary is decoration, adding indirection without adding isolation. If you have seven, five layers is too few — two change vectors will be coupled across a shared boundary, and changes will cascade.

The mistake isn't having too many or too few layers. It's having a number that doesn't match the number of independent change vectors. The cost of too many: unnecessary indirection, files that exist for structural reasons but contain trivial delegation. The cost of too few: changes that cascade across boundaries because two concerns share a layer.

lm15's five layers match its five change vectors. LiteLLM's three layers match its three. Neither is wrong. The difference is that lm15 has two concerns (conversation state, middleware/transport swapping) that LiteLLM doesn't address — not because LiteLLM decided three was the right number, but because LiteLLM's scope is narrower. The layer count follows from the scope, the same way Chapter 7's dependency count followed from the scope. Small scope, few concerns, few layers. Large scope, many concerns, more layers.
