# Cookbook 07 — Multi-Turn Conversation

There are two ways to do multi-turn work in v3:

1. **Stateful** — use `lm15.model()` and let the model object manage history
2. **Stateless** — use `Conversation` to build message lists yourself

## Automatic history on model objects

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", system="You remember everything.")

gpt.call("My name is Max.")
gpt.call("I work on developer tools.")
gpt.call("I like chess and climbing.")

resp = gpt.call("Write a brief bio about me.")
print(resp.text)

print(f"Turns: {len(gpt.history)}")
```

## Inspecting history

```python
for entry in gpt.history:
    req_text = entry.request.messages[-1].parts[0].text
    resp_text = entry.response.text
    print(f"Q: {req_text[:60]}...")
    print(f"A: {(resp_text or '')[:60]}...")
    print()
```

## Resetting conversation

```python
gpt.history.clear()
resp = gpt.call("What's my name?")
print(resp.text)
```

## Stateless multi-turn with `Conversation`

```python
import lm15
from lm15 import Conversation

conv = Conversation(system="You are helpful.")
conv.user("My name is Max.")

resp = lm15.call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
conv.assistant(resp.response)

conv.user("What's my name?")
resp = lm15.call("gpt-4.1-mini", messages=conv.messages, system=conv.system)
print(resp.text)
```

## Prefill

```python
import lm15
from lm15 import Conversation

conv = Conversation(system="Return JSON only.")
conv.user("Output JSON for a person.")
conv.prefill("{")

resp = lm15.call("claude-sonnet-4-5", messages=conv.messages, system=conv.system)
print(resp.text)
```

## Explicit messages without `Conversation`

```python
import lm15
from lm15 import Message

resp = lm15.call(
    "gpt-4.1-mini",
    messages=[
        Message.user("My name is Max."),
        Message.assistant("Nice to meet you, Max!"),
        Message.user("What's my name?"),
    ],
)
print(resp.text)
```
