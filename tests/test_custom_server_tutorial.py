"""Execute the custom-server guide with real serializers and no network."""

import json
from pathlib import Path
import re
import socket

import pytest

import lm15
from lm15 import OpenAIChatLM
from lm15.testing import FakeResponse, FakeTransport


DOC = Path(__file__).resolve().parents[1] / "docs/connecting-openai-compatible-servers.md"
BLOCKS = re.compile(
    r"^```python\n(?P<code>.*?)^```(?:\n```output\n(?P<output>.*?)^```)?",
    re.MULTILINE | re.DOTALL,
)


@pytest.mark.parametrize("token", [None, "local-server-test-token"])
def test_custom_server_tutorial(monkeypatch, capsys, token):
    """Check captured output and exercise both live examples with scripted replies."""
    def no_network(*args, **kwargs):
        pytest.fail("Tutorial tests must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("LOCAL_MODEL_ID", "example/loaded-model")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-cloud-test-token")
    if token is None:
        monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LOCAL_LLM_API_KEY", token)

    chat = {
        "model": "example/loaded-model",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    }
    frames = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
        {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
    ]
    sse = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
    transport = FakeTransport([
        FakeResponse(200, json.dumps(chat).encode()),
        FakeResponse(200, sse + b"data: [DONE]\n\n", headers=[("content-type", "text/event-stream")]),
    ])

    def client(**kwargs):
        # Only replace the transport. Use the actual constructor and adapters.
        return OpenAIChatLM(transport=transport, **kwargs)

    monkeypatch.setattr(lm15, "OpenAIChatLM", client)
    scope = {}
    blocks = list(BLOCKS.finditer(DOC.read_text()))
    assert len(blocks) == 6, "Review new examples and their expected output"
    live_count = 0
    for block in blocks:
        code = block["code"]
        live = code.startswith("# Live:")
        before = len(transport.requests)
        exec(compile(code, str(DOC), "exec"), scope)
        output = capsys.readouterr().out
        if live:
            live_count += 1
            assert block["output"] is None
            assert len(transport.requests) == before + 1
            assert output == "Hello!\n"
        else:
            assert block["output"] is not None, "Offline examples need captured output"
            assert output == block["output"]
            assert len(transport.requests) == before, "Request inspection must not send"

    assert live_count == 2
    assert not transport.responses
    for index, prepared in enumerate(transport.requests):
        assert prepared.method == "POST"
        assert prepared.url == "http://localhost:1234/v1/chat/completions"
        assert dict(prepared.headers)["Authorization"] == f"Bearer {token or 'unused'}"
        payload = json.loads(prepared.body)
        assert payload["model"] == "example/loaded-model"
        assert payload["max_tokens"] == 64
        assert "max_completion_tokens" not in payload
        assert "stream_options" not in payload
        if index == 0:
            assert "stream" not in payload
        else:
            assert payload["stream"] is True
        assert payload["messages"][0] == {"role": "system", "content": "Answer briefly."}
