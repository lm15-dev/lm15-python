# Cookbook 09 — Model Objects and Configuration

## Create a reusable model

```python
import lm15

gpt = lm15.model("gpt-4.1-mini",
    system="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=500,
    retries=2,
)

resp = gpt("Hello!")
print(resp.text)
```

## Override per call

```python
resp = gpt("Be creative.", temperature=1.5)
resp = gpt("Be precise.", temperature=0.0, max_tokens=100)
```

## Derive new models (immutable)

```python
# Swap model, keep everything else
claude = gpt.with_model("claude-sonnet-4-5")

# Swap system prompt
terse = gpt.with_system("You are terse. One sentence max.")

# Bind tools
weather_gpt = gpt.with_tools([get_weather])

# Swap provider
local = gpt.with_provider("openai", base_url="http://localhost:8080/v1")
```

Original is unchanged:

```python
print(gpt.model)      # still "gpt-4.1-mini"
print(claude.model)    # "claude-sonnet-4-5"
```

## Response cache (local)

```python
gpt = lm15.model("gpt-4.1-mini", cache=True)

resp1 = gpt("What is 2+2?")  # hits API
resp2 = gpt("What is 2+2?")  # returns cached response
```

## Config-driven setup

```python
import yaml, lm15

config = yaml.safe_load(open("agent.yaml"))
# {"model": "claude-sonnet-4-5", "system": "...", "retries": 3, "prompt_caching": true}

agent = lm15.model(**config)
```

## Batch from dicts

```python
import lm15

base = {"model": "gpt-4.1-mini", "system": "You are terse.", "temperature": 0}
tasks = [
    {"prompt": "Summarize DNA.", "max_tokens": 50},
    {"prompt": "Summarize RNA.", "max_tokens": 50},
    {"prompt": "Summarize proteins.", "max_tokens": 100},
]

responses = [lm15.call(**base, **t) for t in tasks]
for r in responses:
    print(r.text)
```
