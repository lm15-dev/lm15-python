## The Interface

Every API has documentation. The developer reads the docs, understands the
parameters, and calls the function correctly. Tool calling inverts this: the
"developer" is the model, and the "documentation" is whatever you put in the
tool's name, description, and parameter schema. The model can't read your source
code. It can't hover over a function to see its implementation. It can't ask a
colleague what `do_stuff` means. It reads the name, the docstring, and the JSON
Schema. That's everything. That's the entire documentation surface.

This makes tool design a form of prompt engineering — but for a stranger
audience than most prompt engineering addresses. You're writing for a reader
that is intelligent but has no context about your codebase, that interprets
descriptions literally, and that makes its own autonomous decisions about when
to call your tool based entirely on what you told it. You're designing an API
for a single consumer that you can't train, can't pair-program with, and can't
ask "did you understand?"

### The Input Side

Here is a bad tool:

```python
def process(data: str) -> str:
    """Handle the input."""
    return backend.run(data)
```

The model sees: a function called `process` that "handles the input" and takes a
string called `data`. When should it call this? What should it pass? What will
it get back? The model has no idea. It might call it on every turn (the name and
description are vague enough to match any request). It might never call it (the
description is too vague to match any *specific* request). Both failure modes
come from the same cause: the model can't distinguish this tool from any other,
because the name and description carry no information.

Here is the same tool, redesigned for its actual consumer:

```python
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for articles matching the query.
    Returns the title and first paragraph of the most relevant article.
    If no article matches, returns 'No results found.'"""
    return backend.run(query)
```

Same implementation. Different name, different description. The model now knows:
this tool searches a knowledge base, it takes a search query, it returns a title
and paragraph, and it has a defined behavior for no results. The model can
decide with high confidence whether this tool is relevant to the user's
question. It knows what to pass (a search query, not raw data). It knows what to
expect back (a title and paragraph, not an opaque string). And it knows the
failure mode (explicit "no results" rather than an empty string or an
exception).

The difference between these two tools is not implementation quality. It's
*description* quality. The model's behavior is determined by what you tell it,
not by what your code does. A beautifully implemented tool with a vague
description will be called incorrectly. A trivial tool with a precise
description will be called perfectly. This inversion — the documentation matters
more than the implementation — is unique to tool design. In human-facing APIs,
developers can compensate for bad docs by reading source code, experimenting in
a REPL, or asking for help. The model can do none of these things.

Three principles follow:

**Name the action, not the abstraction.** `search_knowledge_base` tells the
model what happens. `process` tells it nothing. `get_weather` tells it the
domain. `fetch_data` tells it the mechanism but not the purpose. The model
decides whether to call a tool by matching its understanding of the user's
request against its understanding of the tool's purpose. A name that describes
purpose matches well. A name that describes mechanism doesn't.

**Describe the output, not just the input.** Most tool descriptions say what the
tool does ("Search the knowledge base") but not what it returns ("Returns the
title and first paragraph"). The model needs both, because it plans its response
around the expected return value. If the model expects a URL and gets a
paragraph, it may hallucinate a URL to go with it. If the model expects
structured data and gets prose, it may try to parse the prose as structure.
Telling the model what to expect prevents a category of failures where the model
misinterprets the tool result because it was guessing about the format.

**Document the failure mode.** What happens when the tool can't find anything?
When the API is down? When the input is malformed? If the description says
"Returns 'No results found' when no articles match," the model can handle that
case gracefully — "I searched the knowledge base but couldn't find anything
about that topic." If the description says nothing about failure, the model will
be confused by an unexpected return value and may hallucinate an interpretation.

### The Output Side

The return value problem is the most underexplored aspect of tool design, and
it's where the most preventable bugs live.

The model reads the return value as text. Not as structured data, not as a typed
object — as a sequence of tokens derived from whatever string representation the
library produces. This means the format of the return value directly affects the
model's ability to use it.

```python
# What the developer returns
return {"temperature": 22, "condition": "cloudy", "wind_kph": 15}

# What the model reads (Python's str() of a dict)
"{'temperature': 22, 'condition': 'cloudy', 'wind_kph': 15}"
```

The model reads Python dict syntax — single quotes, no whitespace after colons.
It can probably parse this, but it's working harder than it needs to. The model
was trained overwhelmingly on JSON (double quotes, standard formatting) and
natural language. Python dict repr is a third format that the model handles less
reliably.

```python
# Better: format for the model's consumption
return "Temperature: 22°C, Condition: Cloudy, Wind: 15 km/h"
```

Natural language. The model can read it instantly, extract any value, and
incorporate it into a natural-language response without reformatting. The return
value isn't data to be parsed — it's text to be read. Format it for reading.

Even more important than format is what happens on failure. Consider these two
implementations of a file-reading tool:

```python
# Version 1: raises on error
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    return open(path).read()  # raises FileNotFoundError

# Version 2: returns the error
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return open(path).read()
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: permission denied for '{path}'"
```

In version 1, if the file doesn't exist, `open()` raises `FileNotFoundError`.
The exception propagates through the tool execution machinery. The model
receives... nothing. No result, no error message, no indication of what went
wrong. The tool call simply failed, and the model must decide what to do with no
information. It might retry with the same path. It might hallucinate the file's
contents. It might give up and apologize for an unspecified error.

In version 2, the model receives `"Error: file 'config.py' not found"`. It reads
this, understands the problem, and can adapt — try a different path, ask the
user where the file is, or list the directory contents with another tool to find
the right filename. The error message is a communication from the application to
the model, and the model is remarkably good at recovering from errors when it
knows what went wrong.

This is the principle: **errors are return values, not exceptions**. In
human-facing APIs, exceptions are appropriate — the developer catches them,
handles them, moves on. In model-facing APIs, exceptions are silence. The model
can't catch exceptions. It can only read return values. An error that the model
can read is an error the model can handle. An exception is information that
vanishes at the tool boundary.

The combination of both sides — a descriptive name and docstring on input, a
well-formatted and error-aware return value on output — is what separates tools
that work reliably from tools that work sometimes. The model is a capable but
blind consumer. It makes good decisions when it has good information. The tool
interface is where you provide that information or fail to.
