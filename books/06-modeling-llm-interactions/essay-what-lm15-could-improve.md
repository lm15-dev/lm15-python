# What lm15 Could Improve

## A Retrospective in Light of What We Learned

Writing a book about the design problems of modeling LLM interactions — while using lm15 as the primary case study — produces an uncomfortable clarity. You see your own decisions from the outside. You map the tradeoff space and discover that you're not always where you thought you were. You articulate principles and notice where your own code violates them.

This essay examines what lm15 would do differently if it were rewritten today, informed by the ten chapters of *Modeling LLM Interactions*. Not everything here is a mistake — some are tradeoffs that were right at the time and are now worth revisiting. Some are genuine design flaws revealed by the book's analysis. Some are opportunities that didn't exist when lm15 was first written.

---

## 1. The Part Type Needs Narrowing, Not Widening

Chapter 1 defended the single `Part` type with twelve optional fields as the right tradeoff for Python — extensible, portable, simple to iterate. The defense holds. But the *implementation* of that tradeoff is sloppier than necessary.

Twelve optional fields where most are `None` for any given instance is a code smell. The `__post_init__` validation catches construction mistakes ("text part without text") but not access mistakes ("reading `.text` on an image part"). Liskov's objection — that all parts are nominally substitutable but semantically they're not — is valid and addressable without abandoning the discriminator pattern.

**What to change:** Add runtime-enforced field access tied to the type discriminator. Not a class hierarchy — that ship has sailed and the reasons against it (Chapter 1, §3) are correct. Instead, properties that raise informative errors:

```python
@property
def text(self) -> str:
    if self.type not in ("text", "thinking", "refusal"):
        raise TypeError(f"Part(type='{self.type}') has no text field")
    if self._text is None:
        raise ValueError(f"Part(type='{self.type}') has text=None")
    return self._text
```

The field still exists on every Part. But accessing it on the wrong type produces a clear error instead of a silent `None`. The discriminator pattern is preserved. The type safety gap is narrowed. The twelve optional fields remain, but they're guarded by runtime checks that catch the access mistakes `__post_init__` can't.

The cost is a small performance hit on property access and a breaking change for code that checks `part.text is None` to detect non-text parts (which should be `part.type != "text"` anyway). The benefit is that an entire category of silent bugs — reading the wrong field on the wrong Part variant — becomes a loud, debuggable error.

---

## 2. The Conversation Model Needs a Position on Context Management

Chapter 2 argued that lm15 bet on large context windows — send everything, cache the prefix, let context overflow be a runtime error. The bet is aging well: windows grew to 2M tokens, and "send everything" works for most conversations.

But the bet has a blind spot. lm15 offers *no tools* for the developer who hits the limit. No token-counting utilities for the conversation so far. No helpers for truncating history while preserving the system prompt and recent turns. No hooks for summarizing old turns. When context overflows, the developer gets `ContextLengthError` and is on their own.

This isn't honest minimalism — it's a missing feature. The library that says "we don't manage context for you" should at least say "here's how many tokens your conversation currently uses, and here's a utility for trimming it." The absence of these tools forces the developer to either estimate (unreliable) or wait for the error (wasteful — the failed call costs the tokens of the entire context that was sent).

**What to change:** Add a `token_estimate()` method on the Model object that returns the approximate token count of the current conversation (system prompt + tools + history). Not a tokenizer — that would be a dependency. An estimate based on character count and a configurable ratio (default ~0.75 words per token). Add a `trim_history(max_tokens=)` method that removes the oldest turns until the estimate is under the limit, preserving the system prompt and the most recent N turns.

These are simple utilities — maybe 30 lines each. They don't violate the "send everything" philosophy. They just give the developer a gauge and a valve for the cases where "everything" doesn't fit. The chapter's principle — "bet on the constraint that's loosening" — still holds. The utilities are for the cases where the constraint hasn't loosened enough.

---

## 3. Tool Execution Needs Explicit Risk Levels

Chapter 3 identified that tools have risk profiles — `read_file` is safe, `write_file` is consequential, `run_command` is existential — and that lm15's current design encodes the distinction *accidentally*. Callables auto-execute. `Tool` objects don't. The developer expresses their trust decision by choosing between two representations, not by declaring a risk level.

This is accidental design masquerading as intentional. A developer who doesn't know the convention — who passes all tools as callables because the documentation shows callables first — gets auto-execution on everything, including dangerous operations. There's no warning, no confirmation, no "are you sure you want to auto-execute `run_command`?"

**What to change:** Add an explicit `auto_execute` parameter to the tool specification:

```python
# Explicit about what auto-executes
agent = lm15.model("claude-sonnet-4-5",
    tools=[
        Tool.auto(read_file),       # auto-execute: safe
        Tool.auto(search_code),     # auto-execute: safe
        Tool.manual(write_file),    # manual: requires submit_tools()
        Tool.manual(run_command),   # manual: requires submit_tools()
    ],
)
```

The developer declares the execution mode per tool. The declaration is visible in the code. A reviewer can see, at a glance, which tools auto-execute and which don't. The current behavior (callables auto-execute, `Tool` objects don't) remains as a default for backward compatibility, but the explicit form is encouraged.

The deeper change: lm15 should log a warning when a callable tool has a name that matches common write patterns — `write`, `delete`, `remove`, `execute`, `run`, `send`, `deploy`. Not a block — a warning. "Tool `run_command` will auto-execute. Use `Tool.manual(run_command)` for approval-gated execution." The warning is a nudge, not a restriction. It catches the accidental auto-execution case without annoying the developer who's intentionally auto-executing in a sandboxed environment.

---

## 4. Streaming Needs a Completeness Signal

Chapter 6, §5 identified a genuine design flaw: when a stream fails before the `finished` event arrives, `_materialize_response()` builds a partial `LMResponse` with `finish_reason="stop"` — the same value as a successful completion. A partial response is indistinguishable from a complete one.

This isn't a theoretical concern. A network blip that drops the last SSE event — the one carrying usage data and finish reason — produces a response that looks complete, has text, and says `finish_reason="stop"`. The developer who checks `finish_reason` to verify completion is silently misled. The developer who doesn't check doesn't know there's something to check.

**What to change:** When a stream ends without receiving an explicit `end` event, set `finish_reason` to `"incomplete"` rather than `"stop"`. The value is self-documenting — any code that checks `finish_reason == "stop"` as a success signal will correctly exclude incomplete streams. The value is new (not in the current `FinishReason` literal), which means existing code that exhaustively matches on `finish_reason` will encounter an unhandled case — and an unhandled case is the right failure mode. It's better to force the developer to decide how to handle incomplete responses than to silently pretend they're complete.

Add an `is_complete` property on `LMResponse` that checks `finish_reason in ("stop", "length", "tool_call", "content_filter")` — the values that indicate the model finished generating, even if the finish was triggered by a limit or filter rather than natural completion. `"incomplete"` and `"error"` would return `False`. The property gives the developer a single, readable check for "did I get a real response?"

---

## 5. The Escape Hatch Needs Structure

Chapters 4, 9, and 10 repeatedly examined the `provider` dict — the escape hatch that carries provider-specific data. The diagnosis was consistent: the hatch is necessary (full normalization is impossible), honest (it doesn't pretend to be universal), and costly (no autocompletion, no documentation, no type checking).

Six chapters of analysis suggest the hatch deserves more design effort than "a dict of Any."

**What to change:** Provide typed escape hatches per provider, alongside the untyped dict:

```python
# Current: untyped, undocumented
resp.provider["stop_sequence"]  # works on Anthropic, KeyError on OpenAI

# Proposed: typed, documented, per-provider
resp.anthropic.stop_sequence    # typed, documented, exists only for Anthropic responses
resp.openai.system_fingerprint  # typed, documented, exists only for OpenAI responses
```

The per-provider accessor is optional — `resp.provider` (the dict) still works for forward compatibility and for providers that don't have typed accessors. But for the three core providers, the typed accessors would provide autocompletion, documentation, and type checking — the three things the escape hatch sacrifices.

The implementation: each adapter returns a provider-specific typed object alongside the generic dict. The `LMResponse` carries both. The user who needs provider-specific data can access it through a typed interface instead of fishing through an untyped dict. The user who's writing provider-agnostic code never touches either.

This doesn't violate the universal type principle — `LMResponse` is still universal. The provider-specific accessors are *additional*, not *replacement*. They're the windows in the wall (Chapter 9), but with glass instead of open air.

---

## 6. Config Should Validate Against the Provider

Chapter 9, §2 flagged a specific failure: `reasoning={"budget": 10000}` is meaningful on Anthropic and meaningless on OpenAI. The parameter is silently ignored. No error, no warning. The developer who switches providers and keeps the config discovers the non-universality through unexpected behavior, not through a clear signal.

**What to change:** When a `Config` contains provider-specific parameters that don't apply to the current provider, emit a warning:

```
UserWarning: reasoning={"budget": 10000} is Anthropic-specific and will be
ignored by provider 'openai'. Use reasoning={"effort": "high"} for OpenAI,
or reasoning=True for provider-agnostic behavior.
```

Not an error — the call should still succeed. A warning, via Python's `warnings` module, that surfaces in development and can be silenced in production. The warning tells the developer three things: what parameter is being ignored, why (it's provider-specific), and what to do instead (the universal form or the other provider's form).

The adapter already knows which provider it is. The Config already contains the parameters. The validation is a straightforward check: "does this config contain parameters that are meaningless for my provider?" The 30 lines of validation code would prevent an entire category of silent misconfiguration.

---

## 7. The Factory Is Overloaded

Chapter 8, §3 identified `build_default()` as an accidental layer — a function that started as a convenience and grew to handle env file parsing, API key resolution, transport selection, adapter instantiation, and plugin discovery. 130 lines that know about every layer.

**What to change:** Split `build_default()` into composable pieces:

```python
# Current: one function does everything
lm = build_default(use_pycurl=True, env=".env", discover_plugins=True)

# Proposed: composable setup
keys = lm15.resolve_keys(env=".env")
transport = lm15.create_transport(policy=TransportPolicy(...))
lm = lm15.create_client(keys=keys, transport=transport, discover_plugins=True)
```

Each function does one thing. The developer who wants the convenience can still call `build_default()` (which calls all three internally). The developer who wants control — a custom transport, keys from a vault, selective plugin loading — can compose the pieces.

This doesn't add lines to the library. It moves the lines from one function to three, each with a clear responsibility. The total complexity is the same. The surface area is more modular.

---

## 8. History Should Track Token Counts Per Turn

Chapter 2's analysis of quadratic cost and Chapter 8's agent economics both depend on understanding token usage across turns. `model.history` contains `HistoryEntry` objects with the full request and response, including `resp.usage`. But computing "how many tokens has this conversation used so far?" requires iterating the history and summing.

**What to change:** Track cumulative token counts on the Model object:

```python
print(agent.total_tokens)           # cumulative across all turns
print(agent.total_input_tokens)     # cumulative input
print(agent.total_cached_tokens)    # cumulative cache hits
print(agent.total_cost_estimate)    # estimated cost at current pricing
```

These are trivially computed from the history — running sums updated after each call. The cost estimate requires knowing the model's pricing, which could come from ModelSpec or from a simple lookup table. The value is immediacy: the developer can check `agent.total_tokens` at any point in an agent loop without writing a summation loop.

For the budget guard pattern from Book 2 (agent loops with token ceilings), this turns:

```python
total = sum(e.response.usage.total_tokens for e in agent.history)
if total > budget:
    break
```

into:

```python
if agent.total_tokens > budget:
    break
```

One line. Readable. The information was always available; the improvement is making it immediate.

---

## The Common Thread

These eight improvements share a pattern: **make the implicit explicit.**

The Part type implicitly allows accessing the wrong field (→ add runtime guards). The conversation model implicitly assumes context will fit (→ add a token gauge). Tool risk is implicitly encoded in the representation choice (→ add explicit risk levels). Stream completeness is implicitly assumed (→ add an explicit signal). The escape hatch is implicitly untyped (→ add typed accessors). Config compatibility is implicitly ignored (→ add explicit warnings). The factory implicitly bundles six concerns (→ split them explicitly). Token usage is implicitly in the history (→ surface it explicitly).

Each implicit behavior was a reasonable shortcut when lm15 was first written. Each became a gap as the library matured and usage patterns revealed what developers actually needed. The book's analysis didn't discover these gaps — developers hitting them in production did. What the book did was provide the vocabulary to name them, the framework to evaluate their severity, and the principles to design the fixes.

The deepest lesson: a library's first version is always shaped by what the author *can foresee*. Its maturity is shaped by what users *actually encounter*. The gap between the two is the backlog, and the measure of the library's design — Chapter 10's central claim — is how cheaply the gap can be closed.

lm15 is 2,408 lines. Every improvement described here is local — contained within one module, affecting one concern. The total work is perhaps 300 additional lines and zero structural changes. That's the payoff of the design decisions the book examined: open type sets, thin contracts, simple layers, contained escape hatches. The design isn't perfect. But being wrong is cheap.

That was the point all along.
