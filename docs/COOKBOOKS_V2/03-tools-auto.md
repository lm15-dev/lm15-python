# Cookbook 03 — Tools (Auto-Execute)

Pass Python functions as tools. lm15 infers the schema from type hints and docstring, and executes the function automatically when the model calls it.

## Basic

```python
import lm15

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"

resp = lm15.call("gpt-4.1-mini", "What's the weather in Montreal?", tools=[get_weather])
print(resp.text)  # "It's 22°C and sunny in Montreal."
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

resp = lm15.call("gpt-4.1-mini", "What's 2^16 and what's the latest Python news?",
    tools=[search, calculator])
print(resp.text)
```

## Streaming with auto-execute

```python
import lm15

def lookup(topic: str) -> str:
    """Look up a topic."""
    return "42 is the answer."

stream = lm15.stream("claude-sonnet-4-5", "Research quantum computing.",
    tools=[lookup], reasoning=True)

for event in stream:
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

resp = agent("Weather in Paris?")
print(resp.text)

resp = agent("What about London?")
print(resp.text)
```
