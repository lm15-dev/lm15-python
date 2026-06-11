"""Tests for lm15.tools — the pure tool(fn) / derive(fn) factory."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import functools
import typing
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, TypedDict, Union

import pytest

import lm15
from lm15 import FunctionTool
from lm15.errors import LM15Error
from lm15.tools import (
    DerivedParam,
    ToolConfig,
    ToolDerivation,
    ToolDerivationError,
    derive,
    tool,
)


# ─── Reference example from the design ───────────────────────────────

def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Look up current weather.

    Args:
        city: City name, e.g. "Gatineau".
        unit: Temperature unit.
    """
    return f"22{unit} in {city}"


def test_reference_example_produces_plain_function_tool() -> None:
    t = tool(get_weather)
    assert t == FunctionTool(
        name="get_weather",
        description="Look up current weather.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": 'City name, e.g. "Gatineau".',
                },
                "unit": {
                    "enum": ["c", "f"],
                    "type": "string",
                    "description": "Temperature unit.",
                    "default": "c",
                },
            },
            "required": ["city"],
        },
    )


def test_tool_returns_ordinary_function_tool_no_callable_attached() -> None:
    t = tool(get_weather)
    assert type(t) is FunctionTool
    assert lm15.tool_to_dict(t) == {
        "type": "function",
        "name": "get_weather",
        "description": "Look up current weather.",
        "parameters": t.parameters,
    }
    # No callable, no extra state: FunctionTool is slotted.
    assert not hasattr(t, "fn")


def test_tool_is_sugar_for_derive() -> None:
    assert tool(get_weather) == derive(get_weather).tool


# ─── Name and description overrides ──────────────────────────────────

def test_name_and_description_overrides() -> None:
    t = tool(get_weather, name="weather", description="custom")
    assert t.name == "weather"
    assert t.description == "custom"


def test_no_docstring_means_no_description() -> None:
    def f(x: int) -> int:
        return x

    t = tool(f)
    assert t.description is None
    assert t.parameters == {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }


def test_multiline_summary_paragraph() -> None:
    def f(x: int) -> int:
        """First line
        continues here.

        Body paragraph ignored.
        """
        return x

    assert tool(f).description == "First line continues here."


def test_lambda_rejected_without_name() -> None:
    with pytest.raises(ToolDerivationError, match=r"pass name="):
        tool(lambda x: x)


def test_lambda_allowed_with_explicit_name() -> None:
    fn = lambda: None  # noqa: E731
    fn.__annotations__ = {}
    t = tool(fn, name="noop")
    assert t.name == "noop"


def test_partial_without_dunder_name_rejected_with_guidance() -> None:
    p = functools.partial(get_weather, unit="c")
    with pytest.raises(ToolDerivationError, match=r"pass name="):
        tool(p)


def test_non_callable_rejected() -> None:
    with pytest.raises(ToolDerivationError):
        tool(42)  # type: ignore[arg-type]


# ─── Hint mapping table ──────────────────────────────────────────────

def test_primitive_hints() -> None:
    def f(a: str, b: int, c: float, d: bool) -> None: ...

    props = tool(f).parameters["properties"]
    assert props == {
        "a": {"type": "string"},
        "b": {"type": "integer"},
        "c": {"type": "number"},
        "d": {"type": "boolean"},
    }


def test_container_hints() -> None:
    def f(
        a: list[int],
        b: tuple[str, ...],
        c: set[int],
        d: frozenset[str],
        e: dict[str, float],
        g: Sequence[bool],
        h: Mapping[str, int],
    ) -> None: ...

    props = tool(f).parameters["properties"]
    assert props["a"] == {"type": "array", "items": {"type": "integer"}}
    assert props["b"] == {"type": "array", "items": {"type": "string"}}
    assert props["c"] == {
        "type": "array", "items": {"type": "integer"}, "uniqueItems": True,
    }
    assert props["d"] == {
        "type": "array", "items": {"type": "string"}, "uniqueItems": True,
    }
    assert props["e"] == {
        "type": "object", "additionalProperties": {"type": "number"},
    }
    assert props["g"] == {"type": "array", "items": {"type": "boolean"}}
    assert props["h"] == {
        "type": "object", "additionalProperties": {"type": "integer"},
    }


def test_bare_containers() -> None:
    def f(a: list, b: dict) -> None: ...

    props = tool(f).parameters["properties"]
    assert props["a"] == {"type": "array"}
    assert props["b"] == {"type": "object"}


def test_literal_homogeneous_and_mixed() -> None:
    def f(a: Literal["x", "y"], b: Literal[1, 2, 3], c: Literal["x", 3]) -> None: ...

    props = tool(f).parameters["properties"]
    assert props["a"] == {"enum": ["x", "y"], "type": "string"}
    assert props["b"] == {"enum": [1, 2, 3], "type": "integer"}
    assert props["c"] == {"enum": ["x", 3]}  # mixed: no "type"


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class BadEnum(enum.Enum):
    A = object()


def test_enum_subclass() -> None:
    def f(color: Color) -> None: ...

    assert tool(f).parameters["properties"]["color"] == {"enum": ["red", "blue"]}


def test_enum_with_non_json_values_rejected() -> None:
    def f(x: BadEnum) -> None: ...

    with pytest.raises(ToolDerivationError, match=r"'f'.*'x'"):
        tool(f)


def test_union_and_optional_map_to_anyof() -> None:
    def f(a: int | str, b: Optional[int], c: Union[str, None]) -> None: ...

    props = tool(f).parameters["properties"]
    assert props["a"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert props["b"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert props["c"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_optional_does_not_affect_required() -> None:
    # Orthogonality rule: Optional is value nullability; required-ness
    # comes solely from defaults.
    def f(a: int | None, b: int | None = None) -> None: ...

    params = tool(f).parameters
    assert params["required"] == ["a"]
    assert params["properties"]["b"]["default"] is None


def test_annotated_description_beats_docstring() -> None:
    def f(x: Annotated[int, "from annotation"]) -> None:
        """Summary.

        Args:
            x: from docstring.
        """

    assert tool(f).parameters["properties"]["x"] == {
        "type": "integer", "description": "from annotation",
    }


def test_any_maps_to_empty_schema() -> None:
    def f(x: Any) -> None: ...

    assert tool(f).parameters["properties"]["x"] == {}


class Point(TypedDict):
    x: int
    y: int


class Opts(TypedDict, total=False):
    verbose: bool


@dataclasses.dataclass
class Query:
    text: str
    limit: int = 10


@dataclasses.dataclass
class Node:
    value: int
    child: "Node | None" = None


def test_typed_dict_nested_object() -> None:
    def f(p: Point) -> None: ...

    assert tool(f).parameters["properties"]["p"] == {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        "required": ["x", "y"],
    }


def test_typed_dict_total_false_keys_not_required() -> None:
    def f(o: Opts) -> None: ...

    schema = tool(f).parameters["properties"]["o"]
    assert schema == {
        "type": "object",
        "properties": {"verbose": {"type": "boolean"}},
    }


def test_dataclass_nested_object() -> None:
    def f(q: Query) -> None: ...

    assert tool(f).parameters["properties"]["q"] == {
        "type": "object",
        "properties": {"text": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["text"],
    }


def test_recursive_dataclass_rejected_no_ref_in_v1() -> None:
    def f(n: Node) -> None: ...

    with pytest.raises(ToolDerivationError, match=r"recursive"):
        tool(f)


# ─── Rejections ──────────────────────────────────────────────────────

def test_missing_annotation_rejected() -> None:
    def f(x) -> None: ...  # type: ignore[no-untyped-def]

    with pytest.raises(ToolDerivationError, match=r"'f'.*'x'.*no type annotation"):
        tool(f)


def test_var_args_and_kwargs_rejected() -> None:
    def f(*args: int) -> None: ...
    def g(**kwargs: int) -> None: ...

    with pytest.raises(ToolDerivationError, match=r"\*args"):
        tool(f)
    with pytest.raises(ToolDerivationError, match=r"\*\*kwargs"):
        tool(g)


def test_positional_only_rejected() -> None:
    def f(x: int, /) -> None: ...

    with pytest.raises(ToolDerivationError, match=r"positional-only"):
        tool(f)


def test_unsupported_annotation_rejected_with_both_escape_hatches() -> None:
    def f(when: datetime.datetime) -> None: ...

    with pytest.raises(ToolDerivationError) as excinfo:
        tool(f)
    message = str(excinfo.value)
    assert "'f'" in message
    assert "'when'" in message
    assert "datetime.datetime" in message
    assert "overrides" in message
    assert "hand-written" in message


def test_fixed_length_tuple_rejected() -> None:
    def f(pair: tuple[int, str]) -> None: ...

    with pytest.raises(ToolDerivationError, match=r"tuple"):
        tool(f)


def test_dict_with_non_str_keys_rejected() -> None:
    def f(d: dict[int, str]) -> None: ...

    with pytest.raises(ToolDerivationError, match=r"non-str keys"):
        tool(f)


def test_builtin_without_signature_rejected() -> None:
    with pytest.raises(ToolDerivationError, match=r"not introspectable"):
        tool(map)


def test_unresolvable_forward_reference_rewrapped() -> None:
    def f(x: "NoSuchType") -> None: ...  # noqa: F821

    with pytest.raises(ToolDerivationError, match=r"forward"):
        tool(f)


def test_tool_derivation_error_is_lm15_error() -> None:
    assert issubclass(ToolDerivationError, LM15Error)
    err = ToolDerivationError("boom")
    assert err.code == "tool_derivation"


# ─── Defaults ────────────────────────────────────────────────────────

def test_json_defaults_emitted_and_required_from_defaults_only() -> None:
    def f(a: int, b: int = 3, c: list[int] = [], d: str | None = None) -> None: ...

    params = tool(f).parameters
    assert params["required"] == ["a"]
    assert params["properties"]["b"]["default"] == 3
    assert params["properties"]["c"]["default"] == []
    assert params["properties"]["d"]["default"] is None


def test_non_json_default_silently_skipped() -> None:
    sentinel = object()

    def f(x: Any = sentinel) -> None: ...

    params = tool(f).parameters
    assert "default" not in params["properties"]["x"]
    assert "required" not in params  # x has a default, so it is optional


def test_include_defaults_false() -> None:
    def f(b: int = 3) -> None: ...

    t = tool(f, config=ToolConfig(include_defaults=False))
    assert "default" not in t.parameters["properties"]["b"]


def test_tuple_default_emitted_as_list() -> None:
    def f(xs: list[int] = (1, 2)) -> None: ...  # type: ignore[assignment]

    assert tool(f).parameters["properties"]["xs"]["default"] == [1, 2]


# ─── ToolConfig knobs ────────────────────────────────────────────────

def test_additional_properties_false_knob() -> None:
    def f(x: int) -> None: ...

    t = tool(f, config=ToolConfig(additional_properties_false=True))
    assert t.parameters["additionalProperties"] is False
    assert "additionalProperties" not in tool(f).parameters


def test_overrides_replace_derivation_surgically() -> None:
    def f(city: str, when: datetime.datetime) -> None:
        """Summary.

        Args:
            city: Where.
            when: When, ISO 8601.
        """

    cfg = ToolConfig(
        overrides=(("when", {"type": "string", "format": "date-time"}),)
    )
    t = tool(f, config=cfg)
    assert t.parameters["properties"]["when"] == {
        "type": "string",
        "format": "date-time",
        "description": "When, ISO 8601.",  # docstring merged in
    }
    assert t.parameters["properties"]["city"]["type"] == "string"
    assert t.parameters["required"] == ["city", "when"]


def test_override_description_wins_over_docstring() -> None:
    def f(x: int) -> None:
        """Summary.

        Args:
            x: from docstring.
        """

    cfg = ToolConfig(overrides=(("x", {"type": "string", "description": "ovr"}),))
    assert tool(f, config=cfg).parameters["properties"]["x"]["description"] == "ovr"


def test_override_allows_missing_annotation() -> None:
    def f(x) -> None: ...  # type: ignore[no-untyped-def]

    cfg = ToolConfig(overrides=(("x", {"type": "integer"}),))
    assert tool(f, config=cfg).parameters["properties"]["x"] == {"type": "integer"}


def test_override_for_unknown_parameter_rejected() -> None:
    def f(x: int) -> None: ...

    cfg = ToolConfig(overrides=(("nope", {"type": "string"}),))
    with pytest.raises(ToolDerivationError, match=r"nope"):
        tool(f, config=cfg)


def test_tool_config_is_frozen_and_validates() -> None:
    cfg = ToolConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.include_defaults = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        ToolConfig(docstring_style="restructured")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ToolConfig(overrides={"x": {"type": "string"}})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ToolConfig(overrides=(("x",),))  # type: ignore[arg-type]


# ─── Docstring styles ────────────────────────────────────────────────

def test_numpy_style_detected() -> None:
    def f(city: str, unit: str = "c") -> None:
        """Look up weather.

        Parameters
        ----------
        city : str
            City name.
        unit : str
            Temperature unit.
        """

    d = derive(f)
    assert d.docstring_style_detected == "numpy"
    props = d.tool.parameters["properties"]
    assert props["city"]["description"] == "City name."
    assert props["unit"]["description"] == "Temperature unit."


def test_sphinx_style_detected() -> None:
    def f(city: str) -> None:
        """Look up weather.

        :param city: City name.
        :returns: weather string.
        """

    d = derive(f)
    assert d.docstring_style_detected == "sphinx"
    assert d.tool.parameters["properties"]["city"]["description"] == "City name."


def test_google_multiline_param_description() -> None:
    def f(city: str) -> None:
        """Summary.

        Args:
            city: City name,
                continued on a second line.
        """

    assert tool(f).parameters["properties"]["city"]["description"] == (
        "City name, continued on a second line."
    )


def test_docstring_style_none_skips_param_descriptions() -> None:
    t = tool(get_weather, config=ToolConfig(docstring_style="none"))
    assert "description" not in t.parameters["properties"]["city"]
    assert t.description == "Look up current weather."  # summary still used


def test_pinned_style_does_not_fall_back() -> None:
    # get_weather is Google-style; pinning sphinx finds nothing.
    d = derive(get_weather, config=ToolConfig(docstring_style="sphinx"))
    assert d.docstring_style_detected == "plain"
    assert "description" not in d.tool.parameters["properties"]["city"]


def test_plain_docstring_detected() -> None:
    def f(x: int) -> None:
        """Just a summary."""

    assert derive(f).docstring_style_detected == "plain"


def test_no_docstring_detected_as_none() -> None:
    def f(x: int) -> None: ...

    assert derive(f).docstring_style_detected == "none"


# ─── derive() explainability ─────────────────────────────────────────

def test_derive_returns_full_typed_account() -> None:
    d = derive(get_weather)
    assert isinstance(d, ToolDerivation)
    assert d.fn_qualname.endswith("get_weather")
    assert d.docstring_style_detected == "google"
    assert len(d.params) == 2

    city, unit = d.params
    assert isinstance(city, DerivedParam)
    assert city.name == "city"
    assert city.annotation == "str"
    assert city.required is True
    assert city.source == "hint+docstring"
    assert city.description == 'City name, e.g. "Gatineau".'

    assert unit.required is False
    assert unit.source == "hint+docstring"


def test_derive_records_override_source() -> None:
    def f(x: int, y: int) -> None: ...

    cfg = ToolConfig(overrides=(("y", {"type": "string"}),))
    d = derive(f, config=cfg)
    assert d.params[0].source == "hint"
    assert d.params[1].source == "override"


def test_derived_param_annotation_repr_for_unions() -> None:
    def f(xs: list[int] | None) -> None: ...

    d = derive(f)
    assert "list[int]" in d.params[0].annotation
    assert "None" in d.params[0].annotation


# ─── Methods ─────────────────────────────────────────────────────────

class _Service:
    def lookup(self, key: str) -> str:
        """Look up a key."""
        return key

    @classmethod
    def create(cls, label: str) -> "_Service":
        """Create a service."""
        return cls()


def test_bound_method_skips_self() -> None:
    t = tool(_Service().lookup)
    assert t.name == "lookup"
    assert list(t.parameters["properties"]) == ["key"]


def test_classmethod_skips_cls() -> None:
    t = tool(_Service.create)
    assert list(t.parameters["properties"]) == ["label"]


# ─── Purity and top-level exports ────────────────────────────────────

def test_derivation_is_pure_and_repeatable() -> None:
    assert tool(get_weather) == tool(get_weather)
    # the source function is untouched
    assert get_weather("Gatineau") == "22c in Gatineau"


def test_no_parameters_function() -> None:
    def ping() -> str:
        """Ping."""
        return "pong"

    t = tool(ping)
    assert t.parameters == {"type": "object", "properties": {}}


def test_top_level_exports() -> None:
    for name in (
        "tool", "derive", "ToolConfig", "ToolDerivation", "DerivedParam",
        "ToolDerivationError",
    ):
        assert hasattr(lm15, name)
        assert name in lm15.__all__


def test_bare_typing_aliases_map_to_untyped_containers() -> None:
    def f(a: typing.List, b: typing.Dict, c: typing.Sequence,
          d: typing.Mapping, e: typing.Set, g: typing.Tuple) -> None:
        ...

    props = tool(f).parameters["properties"]
    assert props["a"] == {"type": "array"}
    assert props["b"] == {"type": "object"}
    assert props["c"] == {"type": "array"}
    assert props["d"] == {"type": "object"}
    assert props["e"] == {"type": "array", "uniqueItems": True}
    assert props["g"] == {"type": "array"}


def test_non_finite_float_defaults_are_silently_skipped() -> None:
    def f(x: float = float("nan"), y: float = float("inf"), z: float = 1.5) -> None:
        ...

    props = tool(f).parameters["properties"]
    assert "default" not in props["x"]
    assert "default" not in props["y"]
    assert props["z"]["default"] == 1.5
    assert "required" not in tool(f).parameters
