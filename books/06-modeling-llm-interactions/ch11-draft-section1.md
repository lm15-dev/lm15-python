# Chapter 11: The Language of the API

There's a moment in the first hour of using a new library where something either clicks or doesn't. You read an example, you type it into a REPL, and the shape of the code either fits the shape of your thought or fights it. When it fits — when the code you'd *want* to write turns out to be the code that works — the library disappears. You stop thinking about the library and start thinking about your problem. When it doesn't fit — when the code that works is not the code you'd have guessed — the library is present in every line, a translation layer between your intent and the machine.

```python
resp = lm15.complete("claude-sonnet-4-5", "What is the capital of France?")
```

This fits. The developer who has never seen lm15 can read this line and know what it does. The function name says what happens (`complete` — finish this prompt). The first argument is which model. The second is what to say. The response is in `resp`. The developer's guess matches the reality. There's nothing to learn.

```python
client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(message.content[0].text)
```

This fights. Not badly — it's readable, it works, it's well-designed for its purpose. But the developer who wants to ask a question must first construct a client, then learn that "asking" is called "messages.create," then wrap their question in a dict with a role, then wrap that dict in a list, then set a max_tokens they don't yet have an opinion about, and then navigate the response through `.content[0].text` — indexing into a list they didn't know was a list. Seven concepts (client, messages, create, role, content list, max_tokens, response structure) stand between the developer and their question.

The difference between these two examples isn't just verbosity. It's cognitive load. The first example requires one concept: call a function with a model and a prompt. The second requires seven. Each concept is a thing the developer must learn, remember, and get right. Each concept is a potential mistake — wrong role string, missing max_tokens, indexing `content[1]` instead of `content[0]`. The concepts accumulate, and their accumulation is the distance between the developer's intent ("ask the model a question") and the code that expresses it.

This chapter is about that distance. Not about what the library does internally — the previous ten chapters covered that. About what the library *feels like* to the developer who uses it. The API is not a surface over the internals. It's a language — a vocabulary and a grammar that the developer thinks in. Every naming choice, every parameter, every level of progressive disclosure, every convenience property, every error message is a word in that language. And the language determines what thoughts are easy to think and what thoughts require translation.

## The API as Vocabulary

When you write `lm15.complete(...)`, you're using a verb. The verb says what's happening: completing. When you write `lm15.model(...)`, you're creating a noun. The noun is a thing that persists: a model. When you write `resp.text`, you're reading an attribute of the result. The attribute is the most natural question about the result: what did the model say?

These choices feel inevitable after you've used the library, but they're not. Each one was a decision, and each decision has alternatives that would shape the developer's thinking differently.

**Why `complete`, not `chat` or `generate` or `ask`?** `complete` comes from OpenAI's original completions API — the developer supplies a prefix, the model supplies the continuation. But lm15 calls are not completions in any meaningful sense. When you pass a system prompt, tools, reasoning, and a multimodal prompt list, the model isn't "completing" your text. It's interpreting a complex instruction, reasoning about it, calling functions, and composing a response. "Complete" is a fossil — a name from an earlier, simpler era that survived into a more complex one.

`chat` would imply conversation — appropriate for `model()` objects with history, misleading for stateless `complete()` calls. `generate` would imply creation — appropriate for text and image generation, misleading for classification or extraction where the model isn't generating so much as analyzing. `ask` would imply a question — appropriate for Q&A, misleading for "write a poem" or "translate this." Every verb carries a metaphor, and every metaphor is wrong for some use cases.

lm15 kept `complete` because it was familiar — the majority of developers encountered LLM APIs through OpenAI's completions endpoint, and `complete` connects to that existing mental model. Familiarity is a feature. The developer who's used OpenAI's API doesn't need to learn a new verb; they recognize `complete` and transfer their understanding. The transfer is imperfect (lm15's `complete` does more than OpenAI's `completions.create`), but imperfect transfer is better than no transfer.

The cost: developers who haven't used OpenAI's API — who are encountering LLMs for the first time — get no signal from the verb. "Complete" doesn't tell them what's happening. "Ask" would. The library optimized for the experienced developer's familiarity over the beginner's comprehension. Whether this is right depends on who the library's primary audience is.

**Why `model`, not `session` or `conversation` or `agent`?** `model("gpt-4.1-mini")` creates an object that wraps a specific model with configuration and conversation state. The name says "this is a model." But the object isn't the model — the model is running on OpenAI's servers. The object is a client-side wrapper that manages state. Calling it `model` conflates the remote model (the neural network) with the local object (the state manager).

`session` would be more accurate — the object manages a session with a remote model. But "session" implies ephemerality and network state, neither of which applies. `conversation` would emphasize the multi-turn aspect, but the object also handles single-turn calls with tools and configuration. `agent` would emphasize the tool-using aspect, but the object is also used for simple Q&A without tools.

`model` won because it's the developer's primary mental model of what they're interacting with. When a developer says "I'm using GPT-4," they mean the model — not the HTTP client, not the session, not the conversation. The object named `model` matches the developer's concept of what they're working with, even though the object and the concept are different things. Names don't need to be technically precise. They need to match the developer's thinking.

**Why `resp.text`, not `resp.content` or `resp.answer` or `resp.output`?** `text` is the most literal — it's the text content of the response. `content` would be more general (a response might contain images, audio, tool calls — "content" covers all of these). `answer` would imply a question was asked (misleading for tool calls or reasoning traces). `output` would imply a process (correct but clinical).

lm15 chose `text` because it's what the developer reaches for. After calling a model, the first question is "what did it say?" — and the answer is text. The property name matches the question. `resp.text` reads as natural English: "the response's text." `resp.content` would read as jargon: "the response's content" — content of what kind?

But `text` is also a lie for responses that contain more than text. A response with text and tool calls has `resp.text` (the text) and `resp.tool_calls` (the calls). `resp.text` says "here's the text" and implies "that's everything," but it's not everything — the tool calls are equally important. The name creates a hierarchy that doesn't exist in the response. Text is not more important than tool calls. The property's convenience makes it *feel* more important — it's the first thing the developer checks, the thing they print, the thing they pass downstream. The API's vocabulary elevated text above other content types, not because of a design decision but because `.text` is a nicer attribute name than `.tool_calls`.

This is how API vocabulary shapes cognition. The developer who writes `resp.text` 500 times develops an unconscious mental model: the response IS its text. The tool calls, the thinking trace, the citations, the images — these are secondary, accessed through less prominent attributes. The Part type (Chapter 1) treats all content types equally. The API surface does not. The vocabulary creates a hierarchy the type system doesn't endorse.

### What Names Teach

Every name in an API is a micro-lesson. `Part.image(url=...)` teaches: images are parts, they come from URLs. `Message.user("Hello")` teaches: messages have roles, "user" is one of them. `resp.finish_reason` teaches: responses have reasons for ending, and the reason matters.

The names are the API's documentation in the place where developers actually read it — in the code, at the point of use. A developer who's never read lm15's docs but sees `Part.image(url=...)` in someone else's code learns, from the name alone, that images are part objects constructed with URLs. The name taught the concept.

Bad names teach the wrong concept or no concept. `Part._media_part("image", url=...)` — the internal factory method — teaches nothing. It's a private implementation detail that reveals mechanism (there's a generic media-part constructor) instead of concept (you're creating an image). The public `Part.image()` was a naming decision that prioritized teaching over implementation.

The lesson for API design: **names are the most-read documentation.** More developers will read `lm15.complete()` in code than will read the docstring, the README, or the tutorial. The name must teach the concept at the call site, in one word, to a developer who may never read any other documentation.
