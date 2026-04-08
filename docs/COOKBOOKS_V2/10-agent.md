# Cookbook 10 — Building an Agent

## Coding agent

```python
import lm15

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    return open(path).read()

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    open(path, "w").write(content)
    return f"Wrote {len(content)} bytes to {path}"

def run_command(command: str) -> str:
    """Run a shell command and return stdout."""
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr

agent = lm15.model("claude-sonnet-4-5",
    system="You are a coding assistant. Read files, make changes, run tests. Be concise.",
    tools=[read_file, write_file, run_command],
    prompt_caching=True,
    retries=2,
)

resp = agent("Add input validation to the User model in models.py, then run the tests.")

# Tools auto-execute. If the model needs multiple rounds, the loop runs automatically.
print(resp.text)
print(f"Turns: {len(agent.history)}")
```

## Streaming agent with visibility

```python
import lm15

agent = lm15.model("claude-sonnet-4-5",
    system="You are a coding assistant.",
    tools=[read_file, write_file, run_command],
    prompt_caching=True,
)

stream = agent.stream("Refactor the auth module to use JWT.", reasoning=True)

for event in stream:
    match event.type:
        case "thinking":    print(f"💭 {event.text}", end="")
        case "text":        print(event.text, end="")
        case "tool_call":   print(f"\n🔧 {event.name}({event.input})")
        case "tool_result": print(f"📎 {event.text}")
        case "finished":
            u = event.response.usage
            print(f"\n📊 tokens: {u.total_tokens} (cached: {u.cache_read_tokens})")
```

## Research agent

```python
import lm15

def search(query: str) -> str:
    """Search the web."""
    # your search implementation
    return "Result: ..."

def read_url(url: str) -> str:
    """Read a webpage."""
    import urllib.request
    return urllib.request.urlopen(url).read().decode()[:5000]

researcher = lm15.model("gpt-4.1-mini",
    system="You are a research assistant. Search for information, read sources, synthesize findings.",
    tools=[search, read_url],
)

resp = researcher("Write a 3-paragraph summary of recent advances in protein folding.")
print(resp.text)
```

## Vision analysis pipeline

```python
import lm15
from lm15 import Part

gemini = lm15.model("gemini-2.5-flash")
claude = lm15.model("claude-sonnet-4-5")

# Step 1: Gemini describes the image
resp = gemini(["List all objects in this photo.", Part.image(url="https://example.com/room.jpg")])

# Step 2: Claude critiques the analysis
resp2 = claude(f"Critique this image analysis for accuracy:\n\n{resp.text}")
print(resp2.text)
```

## Agent with approval gate (manual tools)

```python
import lm15
from lm15 import Tool

write_file = Tool(
    name="write_file",
    description="Write content to a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
)

agent = lm15.model("claude-sonnet-4-5",
    system="You are a careful coding assistant.",
    tools=[write_file],
)

resp = agent("Rewrite config.py to use environment variables.")

while resp.finish_reason == "tool_call":
    for tc in resp.tool_calls:
        print(f"\n--- {tc.name}({tc.input}) ---")
        approve = input("Approve? [y/n] ")
        if approve.lower() == "y":
            # actually execute
            open(tc.input["path"], "w").write(tc.input["content"])

    results = {tc.id: "Done." for tc in resp.tool_calls}
    resp = agent.submit_tools(results)

print(resp.text)
```
