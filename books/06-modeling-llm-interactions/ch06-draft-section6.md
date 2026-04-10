## Streaming in Time: Tools and Thinking

Everything so far has treated the stream as a single, continuous flow of events — text tokens arriving one after another until the stream ends. In practice, a stream from a reasoning, tool-using agent is something more complex: a multi-phase protocol with pauses, resumptions, and events of radically different types interleaved in a temporal sequence.

Here's what a real stream looks like, laid out as a timeline:

```
  0ms  💭 "Let me check the current weather..."
 50ms  💭 "I should search for Montreal weather data."
120ms  💭 "I'll use the get_weather tool."
200ms  🔧 tool_call: get_weather(city="Montreal")
       ── stream pauses ──
       ── tool executes (300ms) ──
       ── new API call begins ──
550ms  📝 "The current weather in Montreal is"
600ms  📝 " 22°C and sunny."
650ms  ✅ finished (usage: 145 tokens)
```

Seven hundred milliseconds, five event types, one tool execution, two API calls. From the user's perspective, a single flowing interaction. From the implementation's perspective, a complex orchestration that crosses the stream boundary.

### Three Phases

A stream from a reasoning, tool-using model has three distinct phases, and they don't always all appear.

**Phase 1: Thinking.** If `reasoning=True`, the model emits thinking tokens first. These are `StreamChunk(type="thinking")` events — the model's chain of thought, visible to the application but typically hidden from the end user. The thinking phase completes before any text or tool calls begin. The transition from thinking to the next phase is implicit — the event type changes, and the consumer infers the phase change.

**Phase 2: Acting.** The model emits tool call events — `StreamChunk(type="tool_call")` with the tool name and argument fragments. When the tool call is complete (all argument fragments received), the auto-execution machinery kicks in: the tool function runs, the result is collected, and a follow-up API call is made. The stream *pauses* during tool execution — not in the sense of a network delay, but in the sense that the current SSE connection has ended, work is happening off-stream, and a new SSE connection will begin when the follow-up call starts.

**Phase 3: Speaking.** The model emits text tokens — the visible response, incorporating the tool results. This is the phase the end user sees. If the model needs more tools, Phase 2 repeats before Phase 3 completes. The phases can cycle: think → act → speak → act → speak → finish.

### The Pause

The pause between phases 2 and 3 is the most architecturally significant moment in a streaming tool interaction, because it's where the stream stops being a stream.

During Phase 1 and Phase 3, the library is yielding events from an SSE connection. The user's iterator advances with each event. The control flow is simple: `next()` returns the next event from the open connection.

During Phase 2, the SSE connection has ended (the model's response is complete — it said "call this tool"). The tool is executing. A new API call is being constructed. A new SSE connection is being opened. None of this is visible to the user's iterator. The user calls `next()` and it blocks — not because the network is slow, but because the library is executing a tool and starting a new request.

From the user's code, the pause is invisible:

```python
for event in stream:
    match event.type:
        case "thinking":  print(f"💭 {event.text}", end="")
        case "tool_call": print(f"\n🔧 {event.name}")
        case "text":      print(event.text, end="")
        case "finished":  print(f"\n📊 {event.response.usage}")
```

The `tool_call` event appears. The next iteration of the loop blocks while the tool runs and the follow-up call starts. Then text events begin arriving from the new connection. The loop continues as if nothing happened. The multi-call orchestration is hidden inside `__next__`.

This is where the iterator model shines. The pause between tool execution and response resumption is naturally expressed as blocking. `next()` doesn't return until there's an event to return. The user's code doesn't need to handle the pause explicitly — it's just a slow iteration. With callbacks, the pause would be a gap between callback invocations, and the user would need to distinguish "the stream is paused while a tool runs" from "the stream has ended." With async iterators, the pause would be an `await` that resolves when the next event arrives — correct, but the user must understand that the await is hiding an entire tool-execution-and-re-request cycle.

The iterator model's blocking semantics match the temporal structure of tool use: do something, wait, continue. This match is not a coincidence — it's the reason lm15 chose iterators. The most complex streaming scenario (multi-phase tool use with thinking) is the one where iterators produce the simplest user code.

### What the User Actually Sees

In a well-designed streaming agent interface, the timeline *is* the UX:

```
💭 Let me analyze this receipt...
💭 I need to calculate the tip percentage.
🔧 Calling calculate(12.00 / 84.50 * 100)...
📎 Result: 14.2
📝 The tip on this receipt is 14.2%, which is below the customary 18-20%.
📊 145 tokens (98 thinking, 47 output)
```

Each line appeared in real time. The user watched the model think, saw it reach for a calculator, saw the result, and watched the answer compose itself from the result. The temporal structure — think, act, speak — was visible at every step.

This visibility is streaming's deepest value for tool-using agents. Not speed (the total time is the same). Not cost (the tokens are the same). Visibility. The user can see what the agent is doing, can verify that the tool call makes sense, can interrupt if the reasoning goes off track, can trust the answer because they watched it being assembled from real data.

A blocking call returns the same answer. But the answer arrives as a finished product, with no evidence of the process that produced it. The user must trust the output on faith. The streaming user trusts it on evidence — they saw the reasoning, the tool call, the result, and the synthesis. Streaming doesn't just change when the user sees the response. It changes *whether the user can evaluate the response's reliability*. For an agent that acts on the world — writes files, sends emails, executes commands — that evaluability might be the difference between catching a mistake and deploying it.

This is where the chapter's opening claim becomes concrete. Streaming is not a feature bolted onto a blocking API. It's a different relationship between the user and the model — one where the process is as visible as the product, and where the user's ability to observe, evaluate, and intervene is part of the design, not an afterthought. The consumption model (iterators), the two-level API (text vs events), the dual nature (stream + response), the partial failure semantics (preserve and raise), and the temporal complexity (multi-phase tool use) are all in service of that relationship.

A library that treats streaming as "the same thing, but incremental" will get the incremental part right and miss everything else. The everything else is what this chapter is about.
