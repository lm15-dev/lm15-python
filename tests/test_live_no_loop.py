"""Live sessions are transport, not executors.

The maintainer's positioning ruling (2026-06-11, "no loop... remove anything
that's not coherent") removed Result's tool-execution loop; live sessions get
the same treatment. WebSocketLiveSession must surface
LiveServerToolCallEvent as data and accept LiveClientToolResultEvent from the
caller — it must not own a callable registry, a tool-call callback, or any
tool-invocation machinery.
"""

import pytest

import lm15.live as live_mod
import lm15.types as types_mod
from lm15.live import WebSocketLiveSession


class _FakeWS:
    def send(self, _data):  # pragma: no cover - never reached
        pass

    def recv(self):  # pragma: no cover - never reached
        raise RuntimeError

    def close(self):
        pass


def _session_kwargs():
    return dict(ws=_FakeWS(), encode_event=lambda e: [], decode_event=lambda r: [])


def test_live_session_rejects_tool_loop_parameters() -> None:
    for kwarg in ("callable_registry", "on_tool_call"):
        with pytest.raises(TypeError):
            WebSocketLiveSession(**_session_kwargs(), **{kwarg: None})


def test_live_session_has_no_tool_execution_surface() -> None:
    session = WebSocketLiveSession(**_session_kwargs())
    for name in ("set_on_tool_call", "_maybe_auto_execute_tool",
                 "_callable_registry", "_on_tool_call"):
        assert not hasattr(session, name)


def test_live_module_has_no_tool_invocation_helpers() -> None:
    assert not hasattr(live_mod, "_invoke_tool")


def test_types_has_no_tool_registry_alias() -> None:
    assert not hasattr(types_mod, "ToolRegistry")
    import lm15
    assert not hasattr(lm15, "ToolRegistry")
