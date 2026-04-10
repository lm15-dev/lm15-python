# Cookbook 10 — Building an Agent

## Coding agent

```python
import lm15

def read_file(path: str) -> str:
    return open(path).read()

def write_file(path: str, content: str) -> str:
    open(path, "w").write(content)
    return f"Wrote {len(content)} bytes to {path}"

def run_command(command: str) -> str:
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr

agent = lm15.model(
    "claude-sonnet-4-5",
    system="You are a coding assistant. Read files, make changes, run tests. Be concise.",
    tools=[read_file, write_file, run_command],
    prompt_caching=True,
    retries=2,
)

resp = agent.call("Add input validation to the User model in models.py, then run the tests.")
print(resp.text)
print(f"Turns: {len(agent.history)}")
```

## Streaming agent with visibility

```python
import lm15

agent = lm15.model(
    "claude-sonnet-4-5",
    system="You are a coding assistant.",
    tools=[read_file, write_file, run_command],
    prompt_caching=True,
)

for event in agent.call("Refactor the auth module to use JWT.", reasoning=True).events():
    match event.type:
        case "thinking":
            print(f"💭 {event.text}", end="")
        case "text":
            print(event.text, end="")
        case "tool_call":
            print(f"\n🔧 {event.name}({event.input})")
        case "tool_result":
            print(f"📎 {event.text}")
        case "finished":
            u = event.response.usage
            print(f"\n📊 tokens: {u.total_tokens} (cached: {u.cache_read_tokens})")
```

## Research agent

```python
import lm15

def search(query: str) -> str:
    return "Result: ..."

def read_url(url: str) -> str:
    import urllib.request
    return urllib.request.urlopen(url).read().decode()[:5000]

researcher = lm15.model(
    "gpt-4.1-mini",
    system="You are a research assistant. Search for information, read sources, synthesize findings.",
    tools=[search, read_url],
)

resp = researcher.call("Write a 3-paragraph summary of recent advances in protein folding.")
print(resp.text)
```

## Vision analysis pipeline

```python
import lm15
from lm15 import Part

gemini = lm15.model("gemini-2.5-flash")
claude = lm15.model("claude-sonnet-4-5")

resp = gemini.call(["List all objects in this photo.", Part.image(url="https://example.com/room.jpg")])
resp2 = claude.call(f"Critique this image analysis for accuracy:\n\n{resp.text}")
print(resp2.text)
```

## Agent with approval gate

```python
import lm15

def approve(call):
    if call.name == "write_file":
        print(f"\n--- {call.name}({call.input}) ---")
        return None if input("Approve? [y/n] ").lower() == "y" else "Denied by user."
    return None

agent = lm15.model(
    "claude-sonnet-4-5",
    system="You are a careful coding assistant.",
    tools=[read_file, write_file],
    on_tool_call=approve,
)

resp = agent.call("Rewrite config.py to use environment variables.")
print(resp.text)
```

## Parallel exploration with `.copy()`

```python
import asyncio, lm15

agent = lm15.model("claude-sonnet-4-5", system="You are a data analyst.")
agent.call("Here is the schema: ...")

results = await asyncio.gather(
    agent.copy().acall("Analyze by region."),
    agent.copy().acall("Analyze by product."),
)

for r in results:
    print(r.text)
```
