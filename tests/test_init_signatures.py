"""The constructor signatures type checkers see (tools/gen_init_signatures.py).

A dataclass derives __init__ from its stored field types; lm15's value types
store tuples but accept lists (and sometimes one bare item). The generated
``if TYPE_CHECKING: def __init__`` blocks say so to type checkers. These tests
keep them true: up to date with the fields, present on every type that needs
one, and backed by what the constructors really accept.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_init_signatures as gen  # noqa: E402

from lm15 import (  # noqa: E402
    Config, FunctionTool, Message, Request, TextPart, ToolChoice, ToolResultPart,
)


def test_generated_signatures_are_up_to_date():
    done = subprocess.run([sys.executable, str(ROOT / "tools/gen_init_signatures.py"), "--check"],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_type_that_normalizes_an_input_has_a_signature():
    """A new dataclass whose __post_init__ stores a normalized sequence or
    mapping needs an entry in gen_init_signatures.TYPES."""
    listed = {(m, c) for m, names in gen.TYPES.items() for c in names}
    missing = []
    for path in sorted((ROOT / "lm15").rglob("*.py")):
        module_name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            post = next((f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__post_init__"), None)
            if post is None:
                continue
            module = importlib.import_module(module_name)
            obj = getattr(module, cls.name, None)
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            stored = {f.name: str(f.type) for f in dataclasses.fields(obj) if f.init}
            source = ast.get_source_segment(path.read_text(encoding="utf-8"), post) or ""
            normalizes = [
                name for name, ann in stored.items()
                if f'object.__setattr__(self, "{name}"' in source and ann.startswith(("tuple[", "dict[", "Mapping["))
            ]
            if normalizes and gen._hand_written(obj):
                continue  # its own __init__ is what a type checker reads
            if normalizes and (module_name, cls.name) not in listed:
                missing.append(f"{module_name}.{cls.name} ({', '.join(normalizes)})")
    assert not missing, "add to tools/gen_init_signatures.py TYPES: " + "; ".join(missing)


def test_the_signature_matches_the_runtime_init():
    """Same parameter names, order, kinds and defaulted-ness as the dataclass's own __init__."""
    for module_name, names in gen.TYPES.items():
        module = importlib.import_module(module_name)
        for name in names:
            cls = getattr(module, name)
            runtime = inspect.signature(cls.__init__)
            text = gen.signature(cls).replace(" -> None: ...", ":\n    pass")
            ns: dict = {}
            # Annotations stay strings (never evaluated), so no name has to resolve.
            exec(compile("from __future__ import annotations\n" + text, "<sig>", "exec"), ns)
            stub = inspect.signature(ns["__init__"])
            shape = lambda sig: [(p.name, p.kind, p.default is inspect.Parameter.empty) for p in sig.parameters.values()]
            assert shape(stub) == shape(runtime), name


def test_the_wider_inputs_are_really_accepted():
    """What the signatures promise, the constructors do: lists become tuples, one item becomes a one-tuple."""
    hello = Message.user("hello")
    assert Request(model="m", messages=[hello]).messages == (hello,)
    assert Request(model="m", messages=hello).messages == (hello,)
    tool = FunctionTool(name="f", parameters={"type": "object", "properties": {}})
    assert Request(model="m", messages=[hello], tools=[tool]).tools == (tool,)
    assert Request(model="m", messages=[hello], tools=tool).tools == (tool,)
    assert Config(stop=["x", "y"]).stop == ("x", "y")
    assert Config(stop="x").stop == ("x",)
    assert ToolChoice(mode="required", allowed=["f"]).allowed == ("f",)
    assert ToolChoice(mode="required", allowed="f").allowed == ("f",)
    part = TextPart("hi")
    assert Message(role="user", parts=[part]).parts == (part,)
    assert Message(role="user", parts=part).parts == (part,)
    assert ToolResultPart(id="c", content=[part]).content == (part,)
    # Message.tool takes any mapping of call id to output, not only a dict.
    reply = Message.tool(types.MappingProxyType({"call_1": "done"}))
    assert reply.parts[0].id == "call_1"
    # A schema held in an ordinary variable (dict[str, object] to a type checker).
    schema: dict[str, object] = {"type": "object", "properties": {"n": {"type": "integer"}}}
    Config(response_format={"type": "json_schema", "name": "n", "schema": schema})


def test_the_json_the_signatures_admit_is_still_checked_at_runtime():
    with pytest.raises(TypeError):
        Config(response_format={"type": "json_schema", "name": "n", "schema": {"x": object()}})
