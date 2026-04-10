# Cookbook 04 — Tools (Manual Loop)

Use `Tool` objects when you need full control over tool execution — async tools, side effects, approval gates, or tools that return multimodal content.

## Basic manual loop

```python
import lm15
from lm15 import Tool

weather = Tool(
    name="get_weather",
    description="Get weather by city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

gpt = lm15.model("gpt-4.1-mini")

# Step 1: model requests tool call
resp = gpt("Weather in Montreal?", tools=[weather])

# Step 2: execute tools yourself
results = {}
for tc in resp.tool_calls:
    # your code — could be async, call an API, check permissions, etc.
    results[tc.id] = f"22°C, sunny in {tc.input.get('city', '?')}"

# Step 3: submit results back
resp = gpt.submit_tools(results)
print(resp.text)
```

## Built-in tools (provider server-side)

```python
import lm15

resp = lm15.call("gpt-4.1-mini", "Latest AI news", tools=["web_search"])
print(resp.text)

for c in resp.citations:
    print(f"  [{c.title}]({c.url})")
```

## Multi-hop tool loop

```python
import lm15
from lm15 import Tool

search = Tool(name="search", description="Search the web", parameters={
    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
})

agent = lm15.model("gpt-4.1-mini")
resp = agent("Find the population of Tokyo and the area of Japan.", tools=[search])

while resp.finish_reason == "tool_call":
    results = {}
    for tc in resp.tool_calls:
        # simulate search
        results[tc.id] = f"Result for '{tc.input.get('query', '')}': 42 million / 377,975 km²"
    resp = agent.submit_tools(results)

print(resp.text)
```
