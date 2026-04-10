## What Other Libraries Chose

The same problem, solved four ways. Each choice reveals something about what the library values.

### LangChain: The Statement Model

LangChain represents messages as typed classes:

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

messages = [
    SystemMessage(content="You are helpful."),
    HumanMessage(content="What's the weather?"),
    AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Montreal"}}]),
    ToolMessage(content="22°C", tool_call_id="call_1"),
]
```

Each message *is* its type. A `HumanMessage` is not a `Message` with `role="user"` — it's a distinct class with distinct fields. Tool calls live on `AIMessage` as a separate field, not as parts in a content array. The representation is statement-based: each message class is a statement about what kind of utterance occurred.

This buys clarity. When you see `HumanMessage`, you know exactly what you're dealing with. The class name *is* the documentation. IDE autocompletion shows only the fields relevant to that message type.

It costs extensibility. When LangChain needed to add `ToolMessage`, they added a class. When they needed to add `FunctionMessage` (since deprecated), they added another. When they needed multimodal content, they faced a choice: add fields to `HumanMessage` (breaking the "each class is a statement" pattern) or create `HumanMessageWithImage` (combinatorial explosion). They chose to allow `content` to be either a string or a list of typed dicts — essentially embedding the content-blocks pattern inside the statement model. The result is a hybrid that has the hierarchy's verbosity and the content-blocks' flexibility, without the full benefits of either.

The deeper cost is the closed type set. LangChain controls the message classes. A plugin that needs a new message type — say, `ModeratorMessage` for content-filtered responses — must either subclass an existing type (inheriting fields it doesn't need) or ask LangChain to add one (waiting for a release). The type set is closed because it's a class hierarchy, and class hierarchies are closed by nature.

### LiteLLM: The Passthrough

LiteLLM takes the opposite approach — minimal normalization, maximum compatibility:

```python
response = completion(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The input is an OpenAI-shaped dict. The library translates it to whatever the target provider expects. The user thinks in OpenAI's format regardless of which provider they're calling.

This buys adoption. If you know OpenAI's API, you know LiteLLM. The learning curve is zero for OpenAI users. The documentation is OpenAI's documentation.

It costs universality. The representation is OpenAI-shaped, not universal. Anthropic-specific features (cache control, extended thinking) are accessed through provider-specific parameters or don't map cleanly. The dict structure carries no type safety — `{"role": "usr", "content": "Hello"}` (note the typo) compiles and runs and produces a confusing error from the provider, not from the library.

More subtly, it bets on OpenAI's format being the stable standard. If Anthropic's format becomes dominant, or if a new provider introduces concepts that OpenAI's structure can't express, the passthrough model requires either extending the OpenAI format (which OpenAI might not do) or adding parallel input formats (which defeats the purpose). The passthrough is simple because it delegates the representation decision to OpenAI. That delegation is a dependency.

### Vercel AI SDK: The Parts Model

The Vercel AI SDK (TypeScript) uses a parts-based representation that's closest to lm15's:

```typescript
const result = await generateText({
    model: openai("gpt-4.1-mini"),
    messages: [
        { role: "user", content: [
            { type: "text", text: "What's in this image?" },
            { type: "image", image: imageBuffer },
        ]},
    ],
})
```

Content is an array of typed parts. Each part has a `type` discriminator. The type set is defined by the SDK but the structure mirrors the provider-level content blocks. TypeScript's discriminated unions provide type safety within switch statements — inside `case "text"`, `part.text` is `string` and `part.image` doesn't exist.

This is the cleanest implementation of the parts model, because TypeScript's type system supports it natively. The discriminated union is the language-level construct that Python lacks and that lm15 simulates with a single class and optional fields. If lm15 were written in TypeScript, it would look like this.

### The Raw SDKs: No Abstraction

OpenAI's SDK, Anthropic's SDK, and Google's SDK each define their own message types that mirror the wire format exactly. No normalization, no portability. The representation *is* the provider's API.

This is the baseline. Everything above it is an opinion about how much to abstract. The raw SDKs are useful for evaluating that opinion — if your universal representation is harder to use than the raw SDK for the most common case, the abstraction isn't earning its keep.

### What the Comparison Shows

The four approaches map to different priorities:

| Library | Model | Optimizes for | Sacrifices |
|---|---|---|---|
| LangChain | Class hierarchy | Clarity, type safety | Extensibility, simplicity |
| LiteLLM | OpenAI passthrough | Adoption, compatibility | Universality, type safety |
| Vercel AI SDK | Typed parts | Safety, extensibility | (Less — TypeScript helps) |
| lm15 | Untyped parts | Extensibility, portability | Type safety |
| Raw SDKs | Wire format | Accuracy | Portability |

No library is wrong. Each made a defensible choice for its context. LangChain serves a framework-oriented community that values explicit types. LiteLLM serves an OpenAI-first community that values drop-in compatibility. Vercel serves a TypeScript community with language-level discriminated unions. lm15 serves a Python community that values zero dependencies and cross-provider portability.

The lesson isn't "which is best" — it's that the representation decision is downstream of a prior question: *what does this library value?* The representation follows the values. If you value type safety above all, you build a hierarchy. If you value compatibility, you pass through. If you value extensibility, you use a discriminator. The representation looks like a data modeling choice. It's actually a statement of priorities.
