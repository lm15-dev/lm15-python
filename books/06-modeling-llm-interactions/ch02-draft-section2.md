## Who Holds the Conversation

If every call reconstructs the conversation from a list of messages, someone has
to maintain that list. This is the ownership question, and it's the most
consequential API design decision in an LLM wrapper — more consequential than
the message representation (Chapter 1), more consequential than the streaming
model (Chapter 6), more consequential than the error hierarchy. Because whoever
holds the conversation controls what the model sees, and what the model sees
determines what it does.

The options form a spectrum. At one end, the user holds everything. At the
other, the library holds everything. Most libraries land somewhere in between,
and where they land defines what kind of library they are.

### The User Holds Everything

The rawest approach: the library is a function. The user builds the message
list, passes it in, gets a response, appends the response to the list, and
passes the extended list on the next call.

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "My name is Alice."},
]
resp = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
messages.append({"role": "assistant", "content": resp.choices[0].message.content})
messages.append({"role": "user", "content": "What's my name?"})
resp = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
```

This is how OpenAI's SDK works. Anthropic's SDK works the same way. They are not
conversation libraries — they are HTTP clients with typed request builders. The
conversation is a list that the user mutates, and the SDK never touches it.

The benefits are clarity and control. The developer sees exactly what the model
will see, because they built it. There's no hidden state, no surprise
accumulation, no "where did that old message come from?" debugging. If the
conversation needs to be truncated, summarized, or branched, the developer does
it explicitly — because the list is theirs.

The costs are boilerplate and discipline. Every call site must remember to
append the response. Every multi-turn interaction must manage the list. The
developer must track which list is active if there are parallel conversations.
Forgetting to append the assistant's response — a one-line omission — silently
breaks the conversation without an error. The model simply doesn't see the
response and loses continuity. This is a bug that's trivial to write, produces
no error message, and manifests as the model seeming "forgetful" — which the
developer might attribute to the model rather than to their code.

Anthropic and OpenAI made this choice deliberately. They're SDK vendors, not
application framework vendors. Their job is to provide a faithful client for the
HTTP API, and the HTTP API is stateless. Adding conversation management would be
adding opinions to a layer that benefits from having none.

### The Library Accumulates

The next step on the spectrum: the library keeps the list for you. You call a
function or object with a prompt, and the library automatically appends both
your message and the model's response to an internal history. The next call
includes all previous turns.

```python
gpt = lm15.model("gpt-4.1-mini")
gpt("My name is Alice.")
resp = gpt("What's my name?")
print(resp.text)  # "Your name is Alice."
```

This is where lm15 sits. The `Model` object owns a `_conversation` list. Each
call appends the user message and the assistant response. The next call sends
the full list plus the new message. The developer doesn't manage the list,
doesn't append responses, doesn't pass message arrays.

The benefit is friction reduction. The most common pattern — sequential
conversation — works with zero boilerplate. The developer calls the model like a
function and gets continuity for free.

But the word "free" should have an asterisk the size of a billboard. The library
is now making decisions the developer doesn't see. It decides what to keep
(everything). It decides when to discard (never — until the developer calls
`history.clear()`). It decides how to structure the list (append-only, no
editing, no reordering). Each of these decisions is reasonable in isolation, but
the developer who doesn't know they're being made will eventually be surprised.

lm15 mitigates this by making the accumulation visible. `model.history` is a
public list of `HistoryEntry` objects. Each entry contains the full request and
response. The developer can inspect it, count it, compute total token usage
across it, and clear it. The accumulation is automatic but not opaque — it's a
glass box, not a black one.

The critical design question at this level is what to do when the context
overflows. lm15's answer is: nothing. It sends everything, and if the context
window is exceeded, the provider returns a `ContextLengthError`. The developer
must handle this — truncate history, clear and restart, switch to a model with a
larger context window. The library doesn't truncate for you, summarize for you,
or silently drop old messages. This is honest but demanding. The developer who
forgets to manage context length discovers the problem as a runtime error, not
as a graceful degradation.

### The Library Manages

Further along the spectrum: the library doesn't just accumulate — it actively
manages the conversation. It decides what to keep, what to summarize, what to
discard. The developer provides a policy, and the library executes it.

LangChain's memory system is the canonical example:

```python
memory = ConversationSummaryMemory(llm=ChatOpenAI())
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
chain.predict(input="My name is Alice.")
chain.predict(input="What's my name?")
```

Behind the scenes, `ConversationSummaryMemory` periodically summarizes old turns
into a compressed form, replacing the verbatim history with a summary. The model
sees the summary plus recent turns, not the full transcript. Other memory types
exist: `ConversationBufferWindowMemory` keeps the last N turns,
`ConversationTokenBufferMemory` keeps turns up to a token budget,
`VectorStoreRetrieverMemory` embeds turns and retrieves relevant ones.

This is the framework approach. The library is no longer a tool — it's a system
with policies, strategies, and its own decision-making. The benefit is that
context overflow is handled automatically. The developer doesn't think about it.

But the cost is profound: the developer no longer knows what the model sees.
When the model gives a wrong answer, is it because the model is wrong, or
because the summary omitted a crucial detail? When the model seems to forget
something, is it because it actually forgot, or because the memory strategy
discarded that turn? The debugging surface area has expanded from "what did I
send?" to "what did the framework decide to send, and why?"

This is not a theoretical concern. I've watched developers spend hours debugging
a conversational agent that was giving inconsistent answers, only to discover
that the memory system was summarizing away the inconsistency. The model wasn't
wrong. The model never saw the conflicting information — the memory system had
compressed it into a consistent summary. The framework was, in effect, lying to
the model on the developer's behalf.

There's a deeper issue with managed memory, and it connects back to the
observation from the previous section: conversation history is active context,
not inert state. A memory system that summarizes old turns is not just
compressing data — it's rewriting the prompt. The summary becomes part of what
the model reads, and the model's responses are influenced by the summary's
wording, emphasis, and omissions. A skilled prompt engineer chooses words
carefully because they know the model is sensitive to phrasing. A memory system
that auto-summarizes is making those word choices algorithmically, with no human
oversight, on every turn. The prompt the model sees is a collaboration between
the developer (who wrote the system prompt), the user (who wrote the messages),
and the memory system (who decided which parts to keep and how to rephrase
them).

### External State

At the far end of the spectrum, the conversation is no longer a list at all.
It's a database, a vector store, a knowledge graph — an external system that the
model queries through tools rather than receiving as context.

```python
# The model doesn't carry history — it searches for relevant context
def search_memory(query: str) -> str:
    """Search conversation history for relevant information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join(r.text for r in results)

agent = lm15.model("claude-sonnet-4-5", tools=[search_memory])
```

This is RAG (Retrieval-Augmented Generation) applied to conversation history.
Instead of sending the full transcript, the model asks "what do I know about X?"
and gets back the relevant fragments. The context window holds only the current
question and the retrieved context, not the full history.

This approach solves the context overflow problem completely — the history can
be arbitrarily long because only a small, relevant slice is retrieved per turn.
But it introduces a new problem: retrieval quality. If the retrieval misses a
relevant turn, the model doesn't see it. The model's "memory" is only as good as
the search, and semantic search over conversation turns is harder than it sounds
— "what's my name?" requires matching against "My name is Alice," which is
semantically distant but informationally critical. Every failure of retrieval is
a failure of memory, and the failure is silent — the model doesn't know what it
doesn't know.

External memory also changes the nature of the conversation. A transcript-based
conversation has a linear, temporal structure — the model reads turns in order
and experiences the conversation as a narrative. A retrieval-based conversation
has no order — the model sees fragments, selected by relevance, stripped of
temporal context. "You mentioned Alice on turn 3 and then corrected to Alicia on
turn 7" is temporal information that a transcript preserves and a retrieval
system doesn't (unless you embed timestamps and the retrieval surface can reason
about them, which most can't).

### The Spectrum as a Design Choice

These four positions are not a progression from worse to better. They're points
on a tradeoff curve:

| | User holds | Library accumulates | Library manages | External state |
|---|---|---|---|---|
| **Visibility** | Total | High | Low | Varies |
| **Boilerplate** | High | Low | Low | Medium |
| **Context overflow** | User's problem | User's problem | Handled | Handled |
| **Debugging** | Easy | Easy | Hard | Hard |
| **Prompt integrity** | Guaranteed | Guaranteed | Not guaranteed | Not guaranteed |
| **Examples** | OpenAI SDK, Anthropic SDK | lm15 | LangChain | Custom RAG systems |

The column labeled "prompt integrity" is the one that gets the least attention
and matters the most. At the left side of the spectrum, the developer knows
exactly what the model reads, because the developer wrote it. At the right side,
the developer has delegated composition of the model's input to a system that
makes its own decisions about what to include, how to phrase it, and what to
omit. The developer has gained convenience and lost control over the most
important input to the most unpredictable part of their system.

This is not an argument for the left side. Applications with 500-turn
conversations, customer support bots with month-long histories, research agents
that accumulate hundreds of documents — these can't send everything. They need
management. The argument is that the choice should be conscious, that the
developer should understand what they're delegating, and that "the library
handles memory" should be a decision, not a default.

lm15 sits at position two — library accumulates, developer controls — and this
chapter's remaining sections examine the forces that shaped that choice: the
physics of context windows, the economics of re-sending history, the difference
between conversation memory and knowledge memory, and the bet that large windows
make the simplest approach viable.
