## Progressive Disclosure

lm15 has three ways to call a model. This is either a virtue or a liability, depending on whether the developer perceives them as a ladder or a labyrinth.

**Level 1: The function.**

```python
resp = lm15.complete("gpt-4.1-mini", "What is TCP?", env=".env")
print(resp.text)
```

One function. No state. No setup. The developer who's never used lm15 can start here and produce a working call in thirty seconds. The function handles everything — key resolution, provider routing, adapter construction, request building, HTTP, response parsing. The developer sees none of it. They see a function that takes a model name and a prompt and returns text.

**Level 2: The object.**

```python
gpt = lm15.model("gpt-4.1-mini", system="You are concise.", env=".env")
gpt("My name is Alice.")
resp = gpt("What's my name?")
```

An object with configuration and conversation history. The developer creates it once and calls it like a function. Each call accumulates history. Tools bind to the object. Config is set once and overridden per call.

**Level 3: The machinery.**

```python
client = UniversalLM()
client.register(OpenAIAdapter(api_key="sk-...", transport=UrlLibTransport()))
request = LMRequest(model="gpt-4.1-mini", messages=(Message.user("Hello."),))
response = client.complete(request)
```

Manual wiring. The developer constructs the client, registers adapters, builds requests, and calls the client directly. Nothing is automatic. Everything is visible and controllable.

These three levels form a progressive disclosure — the API reveals complexity as the developer needs it. The genius of progressive disclosure is that each level feels complete. A developer at Level 1 doesn't feel like they're using a simplified version of the "real" API. They're using the API. The function does everything they need. Only when they need conversation memory do they discover Level 2. Only when they need custom transports or test harnesses do they discover Level 3.

The design challenge is the transitions. How does a developer at Level 1 discover Level 2? And when they transition, how much of their mental model transfers?

### Discovering the Next Level

The transition from Level 1 to Level 2 is triggered by a need: conversation. The developer who calls `complete()` twice and discovers the model doesn't remember (Chapter 2's opening experiment) has just discovered the limitation of Level 1. The documentation says "use `model()` for conversations." The developer creates a `model()` object and their code changes shape — from a function call to an object that accumulates state.

The transfer is smooth. `lm15.model("gpt-4.1-mini")` echoes `lm15.complete("gpt-4.1-mini", ...)` — same model string, same provider, same parameters. The object is called like a function: `gpt("Hello.")` mirrors `lm15.complete("gpt-4.1-mini", "Hello.")`. The response is the same `LMResponse` with the same `.text`, `.usage`, `.finish_reason`. The developer's existing knowledge transfers almost completely. The only new concepts are the object itself and the fact that it remembers.

The transition from Level 2 to Level 3 is triggered by a need: control. Custom transports, multiple API keys for the same provider, mock adapters for testing, manual middleware composition. The documentation says "use `UniversalLM` directly." The developer constructs a client and their code changes shape dramatically — from a two-line sugar call to a ten-line manual wiring.

The transfer is rough. `UniversalLM`, `LMRequest`, `Message`, `Config` — four new types. The developer must construct messages from `Part` objects, configure with `Config` objects, and call `client.complete(request)` instead of `gpt("Hello.")`. The mental model shifts from "call a function with text" to "build typed request objects and dispatch them." This is a different programming paradigm — data construction vs function invocation — and the developer's Level 1/2 intuitions don't fully transfer.

The gap between Level 2 and Level 3 is lm15's biggest UX problem. There's no Level 2.5 — no intermediate that gives you more control than `model()` without dropping to raw `LMRequest` construction. A developer who needs, say, to add custom headers to the HTTP request, or to use two different API keys for the same provider, must jump from the convenience layer to the machinery layer. The jump is large, and many developers who need one feature from Level 3 end up learning the entire Level 3 surface — more cognitive investment than the feature requires.

### The Right Number of Levels

Is three the right number? Compared to other libraries:

OpenAI's SDK has one level. `client.chat.completions.create(...)`. There's no sugar level above it and no machinery level below it. The developer always works with the same types — `ChatCompletion`, `ChatCompletionMessage`, `ChatCompletionChunk`. The API is consistent but unforgiving: the simple case (ask a question) requires the same ceremony as the complex case (multi-turn with tools).

LangChain has many levels. `ConversationChain` → `LLMChain` → `BaseChatModel` → `ChatOpenAI` → `openai.ChatCompletion`. Five levels of abstraction, each wrapping the one below. The developer can operate at any level, but must understand how the levels relate, because each level adds concepts (chains, memory, output parsers) that the levels below don't have. The progressive disclosure is more granular but also more complex — five levels to learn vs three.

Vercel AI SDK has two: `generateText()` / `streamText()` (the functions) and the `LanguageModelV1` interface (the provider layer). The functions are the convenience level. The interface is the extension level. There's no intermediate stateful object — conversation management is handled by the UI framework (React state), not by the SDK.

Three levels feels right for lm15's scope. One level (like OpenAI) would sacrifice the convenience of `complete()`. Many levels (like LangChain) would add abstractions the library doesn't need. Two levels (like Vercel) would sacrifice either the stateful object (`model()`) or the manual wiring (`UniversalLM`). Three is the minimum number that gives you stateless convenience, stateful conversation, and full manual control.

But the gap between Level 2 and Level 3 is real, and closing it — adding a Level 2.5 that gives partial control without full machinery — is an open design problem. `model.with_provider()` is one attempt (change the provider routing without dropping to `UniversalLM`). `model.with_tools()` is another (change the tool set without reconstructing the object). These `with_*` methods are Level 2.5 features — they give control over specific aspects without requiring the developer to leave the `model()` paradigm. More of them would narrow the gap further.
