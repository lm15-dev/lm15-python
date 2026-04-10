## How Other Libraries Structure the Translation Layer

The adapter's design decisions — contract shape, granularity, translation vs
delegation — exist in every multi-provider library, answered differently based
on the library's priorities.

**LangChain: SDK delegation with class hierarchy.** LangChain's provider
integrations are separate packages — `langchain-openai`, `langchain-anthropic`,
`langchain-google-genai` — each depending on the provider's SDK. `ChatOpenAI`
imports `openai` and delegates to it. `ChatAnthropic` imports `anthropic` and
delegates. The adapter translates between LangChain's `HumanMessage`/`AIMessage`
types and the SDK's types, not between LangChain and the wire format.

This is the delegation approach carried to its logical conclusion. Each adapter
is thin — 100-200 lines of type mapping. The hard work (HTTP, auth, retries,
streaming) is in the SDK. The cost: `langchain-openai` depends on `openai`,
which depends on `httpx`, `pydantic`, and more. `langchain-anthropic` has its
own tree. `langchain-google-genai` brings 41MB. A project using all three
LangChain integrations installs over 200MB of dependencies, most of which are
HTTP and validation libraries duplicated across SDKs.

The adapters inherit from `BaseChatModel`, an abstract base class with methods
like `_generate()` and `_stream()`. This is the inheritance approach: clear
contract, easy discoverability, but tight coupling to LangChain's internal
abstractions. When `BaseChatModel` changes — which it has, several times — every
adapter must update.

**LiteLLM: Translation with OpenAI-shaped types.** LiteLLM translates directly
between OpenAI-format dicts and each provider's wire format. No SDK delegation —
LiteLLM makes raw HTTP calls. The adapters are functions, not classes:
`completion()` routes to `openai_chat_completions()` or
`anthropic_chat_completions()` or `gemini_chat_completions()` based on the model
name.

The adapters are large — `anthropic_chat_completions.py` is over 1,000 lines —
because they handle the full translation without SDK help. But there's no class
hierarchy, no Protocol, no formal contract. The routing function
(`get_llm_provider()`) decides which translation function to call, and each
translation function has its own signature and behavior. Adding a provider means
adding a function and a routing rule. It's informal, procedural, and pragmatic.

LiteLLM's approach works because it's OpenAI-shaped. The "universal type" is
OpenAI's dict format, so the OpenAI adapter is a passthrough and every other
adapter translates *to and from* OpenAI's format. This simplifies the design —
there's one canonical format, not a separate universal type — but it bets on
OpenAI's format remaining the standard. When Anthropic adds a feature that
OpenAI doesn't have (extended thinking, cache control), LiteLLM must either
extend the OpenAI format (adding fields OpenAI doesn't recognize) or handle it
out-of-band.

**Vercel AI SDK: Provider interface with registry.** The Vercel AI SDK defines a
`LanguageModelV1` interface (TypeScript) with methods like `doGenerate()` and
`doStream()`. Each provider implements the interface in a separate package
(`@ai-sdk/openai`, `@ai-sdk/anthropic`). The packages translate between the
SDK's types and the provider's wire format — the translation approach, not
delegation.

TypeScript's type system makes the contract explicit and compiler-enforced. A
provider that misses a method fails at compile time, not at runtime. The
packages are small (no SDK dependency, just HTTP calls) and independently
versioned. The architecture is the closest to lm15's: Protocol-style contract,
translation approach, independent packages. The main difference is that
TypeScript's discriminated unions give the Vercel SDK type safety in the content
model that lm15 can't achieve in Python.

### What the Comparison Shows

| | LangChain | LiteLLM | Vercel AI SDK | lm15 |
|---|---|---|---|---|
| **Contract** | Abstract base class | Informal (functions) | TypeScript interface | Python Protocol |
| **Translation** | SDK delegation | Raw HTTP | Raw HTTP | Raw HTTP |
| **Dependencies per adapter** | Provider SDK (heavy) | None (self-contained) | None (self-contained) | None (self-contained) |
| **Granularity** | One class per provider | One function per provider | One package per provider | One class per provider |
| **Extensibility** | Subclass BaseChatModel | Add function + routing | Implement interface | Satisfy Protocol + entry point |

The horizontal axis of this table is formality. LiteLLM's functions are the
least formal — no contract, no type checking, just "call this function and it
works." LangChain's abstract base class is the most formal — the contract is
enforced by the type system and the runtime. lm15 and Vercel sit in between —
structural contracts that are checked by type checkers but not by the runtime
(in Python) or enforced by the compiler (in TypeScript).

The vertical axis is independence. LangChain's adapters depend on provider SDKs
— they inherit the SDKs' versions, bugs, and dependency trees. The other three
make raw HTTP calls — they own the translation and depend on nothing except the
wire format.

No position is wrong. LangChain's delegation is the right choice for a framework
whose users want rich SDK features (retry policies, connection pooling, async)
without implementing them. LiteLLM's informality is the right choice for a
routing layer that prioritizes compatibility over structure. The Vercel SDK's
interface is the right choice for a TypeScript ecosystem with compiler-enforced
contracts. lm15's Protocol is the right choice for a zero-dependency Python
library that values decoupling over discoverability.

The lesson, again, is that the design decision is downstream of the library's
values. The adapter pattern isn't one pattern — it's a family of related
patterns, and the choice between them is a statement about what the library
considers most important: developer experience (LangChain), compatibility
(LiteLLM), type safety (Vercel), or independence (lm15).
