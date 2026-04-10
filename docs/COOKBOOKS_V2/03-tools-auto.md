# Cookbook 03 — Tools (Auto-Execute)

Pass Python functions as tools. lm15 infers the JSON schema from type hints and docstrings, and auto-executes them when the model calls them.

## Basic

```python
import lm15

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"

resp = lm15.call("gpt-4.1-mini", "What's the weather in Montreal?", tools=[get_weather])
print(resp.text)
```

## Multiple tools

```python
import lm15

def search(query: str) -> str:
    """Search the web for information."""
    return f"Top result for '{query}': Python 4.0 released."

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

resp = lm15.call(
    "gpt-4.1-mini",
    "What's 2^16 and what's the latest Python news?",
    tools=[search, calculator],
)
print(resp.text)
```

## Streaming with auto-execute

```python
import lm15

def lookup(topic: str) -> str:
    """Look up a topic."""
    return "42 is the answer."

for event in lm15.call(
    "claude-sonnet-4-5",
    "Research quantum computing.",
    tools=[lookup],
    reasoning=True,
).events():
    match event.type:
        case "thinking":    print(f"💭 {event.text}", end="")
        case "tool_call":   print(f"\n🔧 calling {event.name}...")
        case "tool_result": print(f"📎 got: {event.text}")
        case "text":        print(event.text, end="")
        case "finished":    print(f"\n📊 {event.response.usage}")
```

## Bound tools on a model

```python
import lm15

def get_weather(city: str) -> str:
    """Get weather by city."""
    return f"22°C in {city}"

agent = lm15.model("gpt-4.1-mini", tools=[get_weather])

resp = agent.call("Weather in Paris?")
print(resp.text)

resp = agent.call("What about London?")
print(resp.text)
```

## Inspect or intercept tool execution with `on_tool_call`

```python
import lm15

def get_weather(city: str) -> str:
    """Get weather by city."""
    return f"22°C in {city}"

def log_tool(call):
    print(f"tool: {call.name}({call.input})")
    return None  # continue with normal auto-execution

resp = lm15.call(
    "gpt-4.1-mini",
    "Weather in Montreal?",
    tools=[get_weather],
    on_tool_call=log_tool,
)
print(resp.text)
```
