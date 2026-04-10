## Discriminator vs Hierarchy

The universal-parts approach leaves an implementation question open: how do you represent the set of part types in your language's type system? Two options dominate, and the choice between them is one of the most reliably contentious decisions in library design.

### The Class Hierarchy

Define a base class. Define a subclass for each content type:

```python
class Part:
    """Base class for all content parts."""

class TextPart(Part):
    text: str

class ImagePart(Part):
    source: DataSource

class ToolCallPart(Part):
    id: str
    name: str
    input: dict

class ThinkingPart(Part):
    text: str
    redacted: bool | None
```

Each subclass has exactly the fields it needs. No optional fields, no discriminator. A `TextPart` has `text`. An `ImagePart` has `source`. The compiler (or type checker) can verify that you're accessing the right field on the right type. `text_part.source` is a type error. `image_part.text` is a type error. You get safety.

You also get a proliferation of types. There are currently ten part types in lm15. That's ten classes to import, ten cases in every handler, ten serialization paths, ten entries in the documentation. When content type eleven arrives — and it will — you add a class, update every match statement, extend every serializer, and release a new version.

The deeper problem isn't the number of classes — it's the pattern of use. Code that handles messages almost always iterates over `message.parts` and switches on the type. With a class hierarchy, that switch is `isinstance`:

```python
for part in message.parts:
    if isinstance(part, TextPart):
        print(part.text)
    elif isinstance(part, ToolCallPart):
        execute(part.name, part.input)
    elif isinstance(part, ThinkingPart):
        log_thinking(part.text)
    # ... seven more
```

This is verbose but type-safe. The type checker verifies that `part.text` is accessed only when `part` is a `TextPart`. The problem is the `else` — what happens when a part doesn't match any known type? In a closed hierarchy (you control all subclasses), this can't happen. In an open hierarchy (plugins can add part types), it can, and the missing `else` is a silent bug.

### The Discriminator

Define a single class with a `type` field and optional fields for each variant:

```python
@dataclass(frozen=True)
class Part:
    type: str          # "text", "image", "tool_call", "thinking", ...
    text: str | None = None
    source: DataSource | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # ...
```

One class. One import. Every handler switches on `part.type`:

```python
for part in message.parts:
    match part.type:
        case "text":      print(part.text)
        case "tool_call": execute(part.name, part.input)
        case "thinking":  log_thinking(part.text)
        case _:           pass  # unknown type — ignore gracefully
```

The `case _` is the key. Unknown types are handled by the default case, not by a missing `isinstance` branch. A plugin that adds `Part(type="3d_model")` works with existing code — the code ignores it instead of crashing. The type set is open by default.

The cost is that `part.text` exists on every `Part`, whether or not it's meaningful. You can write `Part(type="image").text` and get `None` — no type error, no warning, just a value that shouldn't have been accessed. The `__post_init__` validation catches construction mistakes ("text part without text"), but it can't catch access mistakes ("reading text on an image part"). The type checker sees `text: str | None` on every `Part` and permits accessing it everywhere.

Liskov would note that this violates the substitution principle in spirit. All parts are nominally substitutable — any function that accepts a `Part` accepts a text part and an image part. But they're not semantically substitutable — operations meaningful for one variant are meaningless for another. The type system permits what the semantics forbid.

### The Middle Path

Some languages offer a clean resolution. TypeScript's discriminated unions give you both safety and openness:

```typescript
type Part =
    | { type: "text"; text: string }
    | { type: "image"; source: DataSource }
    | { type: "tool_call"; id: string; name: string; input: Record<string, unknown> }
```

The type checker narrows within a switch: inside `case "text"`, `part.text` is `string` and `part.source` doesn't exist. You get the hierarchy's safety and the discriminator's pattern. Rust's enums offer the same thing. Python doesn't — not natively.

Python's closest approximation is `Protocol` plus runtime checking. Define protocols for each variant, define a union type, and use `match` with guards:

```python
class TextContent(Protocol):
    type: Literal["text"]
    text: str

class ImageContent(Protocol):
    type: Literal["image"]
    source: DataSource

PartUnion = TextContent | ImageContent | ToolCallContent | ...
```

This gives you type-checker narrowing in theory — `mypy` and `pyright` can narrow on `part.type` checks. In practice, the ergonomics are poor. You need a separate protocol per variant, a union type that enumerates all variants (making it closed, unless you use extensibility tricks), and careful use of `Literal` types for the discriminator values. The boilerplate exceeds what most library authors will tolerate, and the benefit — type-checker narrowing that most users don't enable — doesn't justify it.

### Why lm15 Chose the Discriminator

The practical answer is that lm15 has three adapters that each handle ten part types. A class hierarchy would mean thirty `isinstance` chains (ten types × three adapters), plus additional chains in the `Stream` class, the `Model` class, and user-facing convenience properties. The discriminator means thirty `match` statements, which are structurally identical but handle unknown types gracefully — a match with a default case degrades; an isinstance chain with a missing branch crashes.

The deeper answer is that the part type set is growing. When lm15 was first written, there was no `thinking` type, no `citation` type, no `refusal` type. Adding each one required adding a value to the `PartType` literal and writing the factory method. It did not require adding a class, updating imports, modifying serializers, or touching adapters that don't handle that type. In a class hierarchy, each addition would have touched every file that pattern-matches on parts. In the discriminator model, each addition touched one file (`types.py`) and the adapters that produce or consume the new type.

The honest answer is that it's a language limitation. If Python had Rust's `enum` or TypeScript's discriminated unions, lm15 would use them. The discriminator pattern is the best Python can do for open, extensible, typed variant sets — not because it's the best design, but because it's the best design that Python supports without external dependencies.
