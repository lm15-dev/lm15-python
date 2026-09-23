"""A tool is data: a request refuses a function and says what to write instead.

lm15 does not derive tools from functions (removed 2026-09-23; see
docs/design-rationale.md, "Why no tools from functions?").
"""

from __future__ import annotations

import pytest

import lm15
from lm15 import FunctionTool


def get_weather(city: str) -> str:
    return f"22°C in {city}"


WEATHER = FunctionTool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
HI = lm15.Message.user("hi")


def test_there_is_no_derivation_api() -> None:
    for name in ("tool", "derive_tool", "ToolConfig", "ToolDerivation", "ToolDerivationError", "DerivedParam"):
        assert not hasattr(lm15, name), name


def test_request_refuses_a_function_and_names_the_fix() -> None:
    for tools in ((get_weather,), [get_weather], get_weather):
        with pytest.raises(TypeError, match=r"not the function 'get_weather': describe it as FunctionTool\(name=\.\.\., description=\.\.\., parameters="):
            lm15.Request(model="m", messages=(HI,), tools=tools)
    with pytest.raises(TypeError, match=r"LiveConfig\.tools .* describe it as FunctionTool"):
        lm15.LiveConfig(model="m", tools=[get_weather])


def test_request_refuses_an_unnamed_callable_without_inventing_a_name() -> None:
    with pytest.raises(TypeError, match=r"not a callable function object: describe it as FunctionTool"):
        lm15.Request(model="m", messages=(HI,), tools=[lambda city: city])
    with pytest.raises(TypeError, match=r"must contain Tool objects, not str$"):
        lm15.Request(model="m", messages=(HI,), tools=["get_weather"])


def test_request_accepts_one_tool_or_a_list() -> None:
    assert lm15.Request(model="m", messages=(HI,), tools=WEATHER).tools == (WEATHER,)
    assert lm15.Request(model="m", messages=[HI], tools=[WEATHER]).tools == (WEATHER,)
