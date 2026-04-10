# How the API Could Improve

The previous essay examined internal improvements — type guards, token estimation, stream completeness signals. This essay examines the surface — the part the developer touches every day. Chapter 11 gave us the vocabulary to evaluate it: naming as cognition, progressive disclosure, pits and cliffs, visible data. These are the tools. Here's what they reveal.

---

## 1. The Name `complete` Has Expired

The function is called `complete` because OpenAI's first API was called "completions" — the model completes your text. But `complete` hasn't described the operation for two years. When you write:

```python
resp = lm15.complete("claude-sonnet-4-5", [
    "Analyze this contract.",
    Part.document(data=pdf_bytes, media_type="application/pdf"),
], system="You are a legal analyst.", tools=[search_case_law], reasoning=True)
```

You are not "completing" anything. You're instructing, providing a document, assigning a persona, granting tool access, and requesting chain-of-thought reasoning. The verb doesn't match the action. Every developer absorbs this mismatch unconsciously, and it creates a subtle cognitive narrowing — `complete` suggests "fill in the blank," which is a smaller idea than what the function actually does.

**The better name:** `call`. It's accurate for every use case: you're calling a model. It's familiar from function-calling terminology. It's short. It doesn't carry a metaphor that's wrong for any case. `lm15.call("claude-sonnet-4-5", "Hello.")` reads as naturally as `complete` and doesn't mislead.

`ask` was tempting but wrong for non-question use cases ("write a poem" isn't asking). `run` implies execution, which is appropriate for agents but strange for one-shot generation. `generate` implies creation, which is wrong for analysis and classification. `call` is neutral enough to be correct everywhere and specific enough to mean something — you're calling the model the way you'd call a function or call a service.

The migration path: add `lm15.call()` as an alias for `lm15.complete()`. Deprecate `complete` gradually. The alias costs one line; the cognitive benefit compounds on every use.

---

## 2. Set Credentials Once

Every example in Book 1 passes `env=".env"`. Every call in a real application passes `env=".env"`. The repetition isn't just verbose — it's a signal that the API is making the developer specify something that should be ambient.

```python
# Current: repeat on every entry point
lm15.complete("gpt-4.1-mini", "Hello.", env=".env")
gpt = lm15.model("gpt-4.1-mini", env=".env")
models = lm15.models(env=".env")
```

The developer sets credentials once, mentally. The API makes them say it on every call. This is a mismatch between the developer's mental model ("I've configured my keys") and the API's reality ("each call resolves keys independently").

**The improvement:**

```python
# Set once
lm15.configure(env=".env")

# Use everywhere — no env= needed
resp = lm15.complete("gpt-4.1-mini", "Hello.")
gpt = lm15.model("gpt-4.1-mini")
models = lm15.models()
```

`lm15.configure()` sets module-level defaults. Subsequent calls use them. Per-call `env=` or `api_key=` overrides the default for that call only. The developer types the credentials once and forgets them — which matches how they think about credentials.

The objection: module-level state is impure. Two tests running in parallel with different credentials would collide. The answer: `configure()` is a convenience for the common case (one set of credentials, one application). The per-call override exists for the uncommon case (testing, multi-tenant, credential rotation). The convenience shouldn't be sacrificed for the edge case; the edge case should have an escape.

---

## 3. Show Me What You're About to Send

Chapter 11's principle: magic should have a reveal. The biggest invisible magic in lm15 is the request construction — the translation from `complete(model, prompt, system=, tools=, ...)` to the `LMRequest` that actually goes to the adapter. The developer can't see this intermediate step. When something goes wrong — the model ignores the system prompt, the tool isn't called, the response is unexpected — the developer can't inspect what was sent.

**The improvement:**

```python
# See the request without sending it
req = lm15.prepare("gpt-4.1-mini", "Hello.",
    system="Be concise.", tools=[get_weather])

print(req.model)        # "gpt-4.1-mini"
print(req.system)       # "Be concise."
print(req.messages)     # (Message(role='user', parts=(Part(type='text', text='Hello.'),)),)
print(req.tools)        # (Tool(name='get_weather', description='...', parameters={...}),)

# Now send it
resp = lm15.send(req, env=".env")
```

`prepare()` builds the `LMRequest` without sending it. The developer can inspect every field — the messages, the inferred tool schemas, the config, the system prompt. If the tool schema looks wrong, they can see it before the API call. If the system prompt was placed incorrectly, they can see it before waiting 2 seconds for a wrong response.

This also solves the schema inspection problem from Chapter 11 §4. `prepare()` with `tools=[get_weather]` shows the inferred schema as part of the request. No separate `callable_to_tool()` export needed — the schema is visible in context, alongside all the other request fields.

The `send()` function accepts a prepared `LMRequest` and sends it. This makes the two-step process (build, then send) as natural as the one-step process (`complete()` = build + send). Most developers use `complete()`. Developers who need visibility use `prepare()` + `send()`. Progressive disclosure: the simple path is still one line; the inspection path is two.

---

## 4. Responses Should Match the Question

`resp.text` answers "what did the model say?" That's the right first question for conversational calls. But for other use cases, different questions are primary:

- **JSON extraction:** "What's the parsed data?" → `resp.json` (parse `resp.text` as JSON, raise `ValueError` if it fails)
- **Classification:** "What label?" → `resp.text.strip()` works, but the developer does it every time
- **Tool calls:** "What tools were called?" → `resp.tool_calls` exists but returns `Part` objects, not a convenient summary
- **Image generation:** "Where's the image?" → `resp.image` exists but its data is buried in `Part.source.data`

The response object has a text-centric hierarchy. Convenience properties for non-text use cases are either missing (`json`) or require the developer to navigate the Part structure to get to the actual data.

**The improvement:**

```python
# Parse response as JSON (common for extraction/classification)
data = resp.json  # parsed dict, or raises ValueError

# Tool calls as a readable summary
for tc in resp.tool_calls:
    print(tc.name, tc.input)  # already works — but...
    print(tc.result)           # doesn't exist — the result from auto-execution isn't accessible

# Image data without navigating Part internals
image_bytes = resp.image.data  # currently requires resp.image.source.data and base64 decoding
```

`resp.json` is the highest-value addition. An enormous fraction of LLM API usage is structured extraction — "return JSON with these fields." Every developer writes the same boilerplate: `json.loads(resp.text)` with a try/except. A property that does this — and raises a clear error when the response isn't valid JSON, including the text that failed to parse — would save thousands of developers from writing the same five lines.

---

## 5. Error Messages Should Be Instructions

lm15's errors are typed (good) and terse (bad). The developer in trouble sees:

```
NotConfiguredError: no API key for provider 'openai'
```

This says what's wrong. It doesn't say how to fix it. The developer must know that the key comes from an environment variable, that the variable is called `OPENAI_API_KEY`, that it can be set in a `.env` file, and that the `.env` file is passed via `env=`. Four pieces of knowledge, none provided by the error.

**The improvement:**

```
NotConfiguredError: no API key for provider 'openai'

  To fix, do one of:
    1. Set OPENAI_API_KEY in your environment
    2. Create a .env file with OPENAI_API_KEY=sk-... and pass env=".env"
    3. Pass api_key="sk-..." directly

  Docs: https://lm15.dev/api-keys
```

Three options, ordered by recommendation. A link to the docs. The developer who reads this error can fix the problem without leaving their terminal. The error message *is* the documentation, delivered at the moment the developer needs it most.

This pattern should apply to every common error:

- `UnsupportedModelError` → "Model 'my-finetune' not recognized. Use `provider='openai'` for custom models, or `lm15.models()` to list available models."
- `ContextLengthError` → "Input exceeds context window (132,000 tokens, limit 128,000). Reduce the prompt, clear conversation history, or use a model with a larger window (e.g. gemini-2.5-flash, 1M tokens)."
- `InvalidRequestError` with tool-calling failure → "Tool call failed: model requested tool 'search' but no tool with that name is registered. Available tools: get_weather, calculator."

Each error message is an instruction, not just a diagnosis. The diagnosis tells you what went wrong. The instruction tells you what to do. Developers read error messages in their most frustrated state. Meeting them with actionable information instead of terse descriptions is the highest-leverage UX improvement a library can make.

---

## 6. The Prompt Parameter Is Doing Too Much

`prompt` accepts three shapes: a string, a list of strings and Parts, or `None` (when using `messages=` instead). This overloading is convenient but creates ambiguity at the type level and in the developer's mental model.

```python
# A string
lm15.complete(model, "Hello.")

# A list of strings and Parts
lm15.complete(model, ["Describe this.", Part.image(url="...")])

# None, with messages instead
lm15.complete(model, messages=[Message.user("Hello.")])
```

Three calling conventions for the same parameter. The developer must learn which to use when, and the mental model is: "prompt is text OR a multimodal list, OR you skip it and use messages." The `prompt` / `messages` mutual exclusion is a constraint that exists in the function signature as a runtime check, not as a type signature. The developer who passes both gets a runtime error, not a type error.

**The improvement is not to change the API** — all three forms are genuinely useful, and combining them into one parameter is a net convenience. The improvement is to **name the pattern explicitly** in documentation and error messages:

- "prompt" is for single-turn input (a string or a multimodal list → becomes one user message)
- "messages" is for multi-turn input (an explicit conversation history)
- They're mutually exclusive because they're two ways to express the same thing: what the model should read

The error message for passing both should teach this:

```
ValueError: prompt and messages are mutually exclusive

  Use prompt= for a single question:
    lm15.complete(model, "What is TCP?")

  Use messages= for a conversation:
    lm15.complete(model, messages=[Message.user("Hi"), Message.assistant("Hello!"), Message.user("How are you?")])
```

The code doesn't need to change. The *explanation* of the code does. The developer's confusion isn't about the API — it's about the mental model behind the API, and the mental model needs to be taught at the point of confusion (the error message), not only in the docs.

---

## 7. The Model Object Should Warn About Unintended History

Chapter 11's cliff: a developer creates `model()` for config reuse, not for conversation, and accidentally accumulates history across 100 calls.

**The improvement:** A `stateless` parameter that creates a model object without history accumulation:

```python
# Stateful (current behavior) — for conversations
gpt = lm15.model("gpt-4.1-mini", system="You are helpful.")
gpt("My name is Alice.")
resp = gpt("What's my name?")  # knows "Alice"

# Stateless — for config reuse without conversation
gpt = lm15.model("gpt-4.1-mini", system="You are helpful.", stateless=True)
gpt("My name is Alice.")
resp = gpt("What's my name?")  # doesn't know — each call is independent
```

`stateless=True` means the model object carries config (system prompt, tools, temperature) but doesn't accumulate history. Each call starts fresh. The developer who wants config reuse without conversation gets it explicitly, instead of accidentally accumulating history and wondering why their calls are getting slower.

The alternative — logging a warning when history exceeds a threshold ("Model has accumulated 50 turns of history. Use model.history.clear() to reset, or stateless=True to disable accumulation") — would catch the accidental case without adding a parameter. Both approaches are valid. The parameter is more explicit; the warning is less intrusive.

---

## The Common Thread

These seven improvements share a principle that Chapter 11 established: **the API should match the developer's mental model, and where it can't, it should teach the correct model at the point of divergence.**

`complete` → `call` — match the developer's concept of what they're doing. `env=".env"` everywhere → `configure()` once — match how the developer thinks about credentials. No request preview → `prepare()` — show the developer what's happening when they need to see it. Terse errors → instructional errors — teach at the moment of confusion. Accidental history → `stateless=True` or a warning — prevent the cliff before the developer falls.

Each improvement is small — an alias, a function, a parameter, a better error message. None requires architectural changes. None breaks existing code. The total effort is perhaps 200 lines across the codebase. The impact is disproportionate, because the surface is where the developer lives. The internals matter to the library author. The surface matters to everyone.

The deepest improvement isn't any individual change. It's the habit of asking, at every API decision: "What will the developer think this does?" Not "what does it do" — that's the implementation question. "What will they think it does" — that's the UX question. The gap between those two answers is where cliffs live. Close the gap and the cliffs become pits of success.
