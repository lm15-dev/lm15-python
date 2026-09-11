"""Run every README Python block through real adapters, with offline wire replies."""

import base64
import contextlib
import io
import json
from pathlib import Path
import re
import socket

import lm15.auth
from lm15.testing import FakeResponse
from lm15.transports import AsyncTransportResponse
from lm15.transports._async import StdlibAsyncTransport
from lm15.transports._sync import StdlibTransport


README = Path(__file__).resolve().parents[1] / "README.md"
TEXT = "Offline example reply."
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jG7sAAAAASUVORK5CYII="


def _json_response(body, status=200):
    return FakeResponse(status, json.dumps(body).encode())


def _sse_response(frames, done=False):
    body = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
    if done:
        body += b"data: [DONE]\n\n"
    return FakeResponse(200, body, headers=[("content-type", "text/event-stream")])


def _reply(request):
    """Script only the service response; exercise real request building and parsing."""
    payload = json.loads(request.body)
    if dict(request.headers).get("Authorization") == "Bearer not a key":
        return _json_response({"error": {"code": "invalid_api_key", "message": "Invalid key"}}, 401)
    if request.url.endswith("/images/generations"):
        return _json_response({"output_format": "png", "data": [{"b64_json": PNG}]})
    if request.url.endswith("/chat/completions"):
        if payload.get("stream"):
            return _sse_response([
                {"id": "chat", "model": payload["model"], "choices": [{"delta": {"role": "assistant", "content": TEXT}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ], done=True)
        return _json_response({
            "id": "chat", "model": payload["model"],
            "choices": [{"message": {"role": "assistant", "content": TEXT}, "finish_reason": "stop"}],
        })
    if request.url.endswith("/messages"):
        return _json_response({
            "id": "anthropic", "model": payload["model"], "role": "assistant",
            "content": [{"type": "text", "text": TEXT}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        })
    if ":generateContent" in request.url:
        return _json_response({
            "candidates": [{"content": {"role": "model", "parts": [{"text": TEXT}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
        })
    assert request.url.endswith("/responses"), request.url
    tools = payload.get("tools", [])
    function_tools = [t for t in tools if t["type"] == "function"]
    tool_results = [i for i in payload.get("input", []) if i.get("type") == "function_call_output"]
    if function_tools and not tool_results:
        assert function_tools[0]["name"] == "get_weather"
        assert payload["tool_choice"] == "required"
        assert payload["parallel_tool_calls"] is False
        output = [{"type": "function_call", "call_id": "weather_call", "name": "get_weather", "arguments": '{"city":"Montreal"}'}]
    else:
        if tool_results:
            assert tool_results == [{"type": "function_call_output", "call_id": "weather_call", "output": "Sunny and 22°C in Montreal."}]
            assert payload["tool_choice"] == "none"
        annotations = []
        if any(t["type"].startswith("web_search") for t in tools):
            annotations = [{"type": "url_citation", "url": "https://example.com/source", "title": "Example source", "start_index": 0, "end_index": len(TEXT)}]
        output = [{"type": "message", "id": "msg", "role": "assistant", "content": [{"type": "output_text", "text": TEXT, "annotations": annotations}]}]
    response = {
        "id": "response", "model": payload["model"], "status": "completed", "output": output,
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }
    if payload.get("stream"):
        return _sse_response([
            {"type": "response.created", "response": {"id": "response", "model": payload["model"]}},
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": TEXT},
            {"type": "response.completed", "response": response},
        ])
    return _json_response(response)


def test_all_readme_python_examples(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("README tests must never access real services")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "offline-test-key")

    # Use fresh dummy CLI credentials, never the developer's files or refresh tokens.
    claude = tmp_path / "claude.json"
    claude.write_text(json.dumps({"claudeAiOauth": {"accessToken": "offline-claude", "expiresAt": 4102444800000}}))
    codex = tmp_path / "codex.json"
    def segment(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()
    token = ".".join([segment({"alg": "none"}), segment({"exp": 4102444800, "https://api.openai.com/auth": {"chatgpt_account_id": "offline-account"}}), "signature"])
    codex.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": token}}))
    monkeypatch.setattr(lm15.auth, "CLAUDE_CODE_CREDENTIALS_PATH", claude)
    monkeypatch.setattr(lm15.auth, "CODEX_CLI_AUTH_PATH", codex)

    requests = []
    def sync_stream(self, request):
        requests.append(request)
        return _reply(request)

    def async_stream(self, request):
        requests.append(request)
        response = _reply(request)
        async def chunks():
            yield response.body
        async def release(body_consumed):
            pass
        return AsyncTransportResponse(
            status=response.status, reason=response.reason, headers=response.headers,
            http_version=response.http_version, chunks=chunks(), release=release,
        )

    monkeypatch.setattr(StdlibTransport, "stream", sync_stream)
    monkeypatch.setattr(StdlibAsyncTransport, "stream", async_stream)
    blocks = re.findall(r"^```python\n(.*?)^```", README.read_text(), re.M | re.S)
    assert len(blocks) == 15, "Review coverage when adding or removing README examples"
    outputs = []
    scope = {}
    for index, code in enumerate(blocks, 1):
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            exec(compile(code, f"README.md Python block {index}", "exec"), scope)
        outputs.append(shown.getvalue())

    assert len(requests) == 16
    assert outputs == [
        f"{TEXT}\nstop\n5\n",
        f"anthropic {TEXT}\ngemini {TEXT}\n",
        f"{TEXT}\n",
        TEXT,
        f"{TEXT}\n",
        "get_weather {'city': 'Montreal'}\n",
        f"{TEXT}\n",
        f"{TEXT}\nExample source https://example.com/source\n",
        f"{TEXT}\n{TEXT}\n",
        f"{TEXT}\n",
        f"{TEXT}\n",
        f"{TEXT}\n",
        f"image/png {len(base64.b64decode(PNG))} bytes\n",
        "True\n",
        "Check API key: ('OPENAI_API_KEY',)\n",
    ]
