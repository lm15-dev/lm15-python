# Cookbook 01 — Hello World

## One-liner

```python
import lm15

resp = lm15.complete("gpt-4.1-mini", "Hello.")
print(resp.text)
```

## With system prompt

```python
resp = lm15.complete("claude-sonnet-4-5", "Summarize DNA.", system="You are terse.")
print(resp.text)
```

## Config from a dict

```python
config = {
    "model": "gpt-4.1-mini",
    "system": "You are terse.",
    "temperature": 0.7,
    "max_tokens": 100,
}

resp = lm15.complete(prompt="Summarize RNA.", **config)
print(resp.text)
```

## Explicit provider

```python
resp = lm15.complete("my-fine-tune", "Hello.", provider="openai")
```

## Usage

```python
resp = lm15.complete("gpt-4.1-mini", "Hello.")
print(resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.total_tokens)
```
