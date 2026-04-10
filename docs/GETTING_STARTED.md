# Getting Started

## 1) Install and verify environment

```bash
# from repository root
python -m unittest discover -s tests -v
python completeness/runner.py --mode fixture --fail-under 1.0
```

## 2) Configure API keys

Set one or more:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## 3) First request

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "Reply with exactly: ok")
print(resp.text)
```

## 4) Stream text or events

```python
import lm15

for text in lm15.call("gpt-4.1-mini", "Write 'ok' and stop."):
    print(text, end="")
```

```python
import lm15

for event in lm15.call("gpt-4.1-mini", "Write 'ok' and stop.").events():
    if event.type == "text":
        print(event.text, end="")
```

## 5) Reuse config with a model object

```python
import lm15

agent = lm15.model("gpt-4.1-mini", system="You are terse.")
print(agent.call("Hello.").text)
print(agent.call("Write a haiku.").text)
```

## 6) Low-level client (advanced)

If you want manual adapter wiring and raw `LMRequest` objects:

```python
from lm15 import Message, LMRequest, Part, build_default

lm = build_default(use_pycurl=True)
req = LMRequest(
    model="gpt-4.1-mini",
    messages=(Message(role="user", parts=(Part.text_part("Reply with exactly: ok"),)),),
)
resp = lm.complete(req)
print(resp.message.parts[0].text)
```

## 7) Run cookbook examples

```bash
python examples/01_basic_text.py
python examples/02_streaming.py
python examples/03_tools.py
```
