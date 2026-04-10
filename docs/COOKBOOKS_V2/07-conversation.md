# Cookbook 07 — Multi-Turn Conversation

## Automatic history on model objects

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", system="You remember everything.")

gpt("My name is Max.")
gpt("I work on developer tools.")
gpt("I like chess and climbing.")

resp = gpt("Write a brief bio about me.")
print(resp.text)  # knows name, work, and hobbies

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
resp = gpt("What's my name?")
print(resp.text)  # doesn't know anymore
```

## Prefill (seed the assistant response)

```python
import lm15

resp = lm15.call("claude-sonnet-4-5", "Output JSON for a person.", prefill="{")
print(resp.text)  # starts with {
```

## Explicit multi-turn (without model object)

```python
import lm15
from lm15 import Message

resp = lm15.call("gpt-4.1-mini", messages=[
    Message.user("My name is Max."),
    Message.assistant("Nice to meet you, Max!"),
    Message.user("What's my name?"),
])
print(resp.text)
```
