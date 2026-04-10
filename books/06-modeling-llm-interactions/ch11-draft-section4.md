## Invisible Magic and Visible Data

When you write `tools=[get_weather]`, something happens that you don't see. lm15 inspects your function — reads its name, its docstring, its parameter types, its defaults — and constructs a JSON Schema:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a city.",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  }
}
```

This schema is what the model actually reads. The model never sees your function. It sees the schema — the name, the description, the parameter types. The schema is the tool's entire identity to the model. And the developer never sees it either.

This is invisible magic. The function goes in. The tool behavior comes out. The intermediate representation — the schema that determines whether the model calls the tool correctly — is hidden from both the developer who created the tool and the model that uses it. The developer can't inspect what was inferred. They can't compare it to what they intended. If the inference is wrong — a complex type mapped to `"string"`, a missing docstring producing `null` description, a parameter name that's misleading in isolation — the developer discovers the problem through the model's behavior ("the model keeps calling my tool with weird arguments"), not through the schema.

Chapter 9's principle was "expose the data, hide the mechanism." The callable-to-tool inference hides both. The mechanism (introspecting the function's signature) is hidden — reasonable, the developer doesn't need to know about `inspect.get_annotations`. But the data (the inferred schema) is also hidden — unreasonable, because the schema is the single most important factor in whether the tool works correctly.

**What visibility would look like:**

```python
tool = lm15.callable_to_tool(get_weather)
print(tool.name)         # "get_weather"
print(tool.description)  # "Get the current weather for a city."
print(tool.parameters)   # {"type": "object", "properties": {...}, "required": [...]}
```

`callable_to_tool` exists — it's in `model.py`, it's what the library calls internally. But it's not documented as a user-facing function. It's not exported from `lm15`. A developer who wants to inspect the inferred schema must import it from `lm15.model` — a private-feeling path that most users would never discover.

Making this visible — exporting `callable_to_tool` as a public function, documenting it, and suggesting that developers inspect their schemas — would add zero complexity to the library and eliminate an entire category of debugging. The developer who writes a tool that the model misuses could check the schema first: "Oh — the docstring is `None` because I used a comment instead of a docstring. The model has no description for this tool." Five seconds of inspection instead of thirty minutes of behavioral debugging.

The broader principle: **magic should have a reveal.** Invisible magic is wonderful when it works. When it doesn't, the developer needs to see the intermediate state — the point where the magic transformed their input into something they didn't expect. Every magical convenience should have a corresponding inspection function that shows what the magic produced. `callable_to_tool()` for tool inference. A hypothetical `build_request()` for request construction. A schema viewer, a request viewer, a "show me what you're about to send" mode.

The libraries that handle this best offer a debug surface parallel to the convenience surface. Pydantic has `model.model_json_schema()` — show me the schema you inferred from my class. SQLAlchemy has `query.statement` — show me the SQL you're about to execute. Django has `queryset.query` — same idea. Each one lets the developer see through the magic at the moment they need to, without requiring them to use the verbose API all the time.

lm15's convenience surface is strong. Its debug surface is almost nonexistent. The developer who needs to see what's happening under the convenience must drop to Level 3 (manual `LMRequest` construction) or read the source code. There's no intermediate — no "show me what `complete()` would send, without actually sending it." This gap is the difference between magic that the developer trusts (because they can verify it) and magic that the developer fears (because they can't).

### The Emergent Pattern

The config-from-dict pattern — `lm15.complete(**config, prompt="...")` — is visible data at work. The developer creates a dict:

```python
researcher = {
    "model": "gemini-2.5-flash",
    "system": "You are a research assistant.",
    "temperature": 0,
    "max_tokens": 500,
    "env": ".env",
}
```

The dict is inspectable, printable, serializable, diffable, versionable. The developer can look at it and see exactly what configuration will be used. They can compare two configs side by side. They can load configs from YAML and know exactly what they contain. The data is visible because it's *data* — a plain dict, not an opaque object with internal state.

This pattern wasn't designed. lm15's API happens to accept all configuration as keyword arguments, and Python's `**` unpacking happens to support dicts. The emergent result — config-as-visible-data — is one of the most useful patterns in practice. Developers build config files, experiment-tracking systems, and A/B testing frameworks on top of it. None of this was planned. It was enabled by an API shape that kept the data visible.

The lesson: **APIs that keep data visible enable uses the designer didn't imagine.** APIs that hide data behind objects limit uses to what the designer provided. The config dict is powerful because it's not a `Config` object — it's a plain dict that the developer controls completely. If `complete()` required a `Config` object, the developer would need to learn the object's constructor, its validation rules, and its serialization format. The dict needs none of that. It's just data, and data is the most composable thing in programming.

This connects to the book's recurring theme: the representation choices (Chapter 1), the conversation model (Chapter 2), the streaming interface (Chapter 6), and now the API surface — at every level, the design that exposes data and hides mechanism produces better outcomes than the design that hides both. The developer doesn't need to see how the HTTP request is constructed (mechanism). They do need to see what the HTTP request contains (data). The developer doesn't need to see how the schema was inferred (mechanism). They do need to see what schema was inferred (data). Mechanism is the library's business. Data is the developer's.
