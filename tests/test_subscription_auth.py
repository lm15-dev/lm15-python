"""Subscription adapters: Claude Code OAuth and Codex CLI OAuth.

Same quality bar as the API-key siblings (test_providers.py):

- credential loading from local CLI files only — typed, helpful failures
  (missing file, malformed JSON, expired token), never a raw JSON traceback;
- secrets never appear in reprs;
- build_request is pure (no sockets), MAP-2/MAP-3 hold on synthetic bodies;
- AuthError guidance says how to re-login, not which env var to set;
- async mirrors match the sync constructor surface;
- ONE tiny live smoke per provider, skipped unless local credentials exist
  and are fresh.
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import inspect
import json
import socket
import time
from pathlib import Path

import pytest

from lm15.auth import (
    LocalOAuthCredential,
    get_claude_code_access_token,
    get_codex_cli_access_token,
    read_claude_code_credential,
    read_codex_cli_credential,
)
from lm15.errors import AuthError, NotConfiguredError
from lm15.providers import (
    AsyncClaudeCodeLM,
    AsyncOpenAICodexLM,
    ClaudeCodeLM,
    OpenAICodexLM,
)
from lm15.types import Config, Message, Request, TextPart

from .test_providers import _FakeResponse, _FakeTransport


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _fake_codex_jwt(account_id: str = "acct_test", *, exp: int | None = None) -> str:
    exp = exp or int(time.time()) + 3600
    return ".".join(
        [
            _b64url({"alg": "none", "typ": "JWT"}),
            _b64url({"exp": exp, "https://api.openai.com/auth": {"chatgpt_account_id": account_id}}),
            "signature",
        ]
    )


def _write_claude_creds(
    path: Path,
    *,
    access: str = "sk-ant-oat-local",
    refresh: str | None = "sk-ant-ort-local",
    expires_at: int | None = None,
) -> None:
    oauth: dict = {"accessToken": access}
    if refresh is not None:
        oauth["refreshToken"] = refresh
    oauth["expiresAt"] = expires_at if expires_at is not None else int(time.time() * 1000) + 60_000
    path.write_text(json.dumps({"claudeAiOauth": oauth}), encoding="utf-8")


def _write_codex_auth(path: Path, *, access: str, refresh: str | None = "rt-local") -> None:
    tokens: dict = {"access_token": access}
    if refresh is not None:
        tokens["refresh_token"] = refresh
    path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": tokens}), encoding="utf-8")


# ─── credential loading: valid files ─────────────────────────────────


def test_claude_code_reads_local_credentials_file(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path)

    credential = read_claude_code_credential(path)

    assert credential is not None
    assert credential.access_token == "sk-ant-oat-local"
    assert get_claude_code_access_token(path) == "sk-ant-oat-local"


def test_codex_reads_local_cli_auth_file(tmp_path) -> None:
    path = tmp_path / "auth.json"
    access = _fake_codex_jwt("acct_local")
    _write_codex_auth(path, access=access)

    credential = read_codex_cli_credential(path)

    assert credential is not None
    assert credential.access_token == access
    assert credential.account_id == "acct_local"
    assert get_codex_cli_access_token(path) == credential


# ─── credential loading: typed, helpful failures ─────────────────────


def test_claude_code_missing_file_is_typed_with_login_hint(tmp_path) -> None:
    with pytest.raises(NotConfiguredError) as exc_info:
        get_claude_code_access_token(tmp_path / "nope.json")
    msg = str(exc_info.value)
    assert "claude" in msg.lower()
    assert "log in" in msg.lower() or "/login" in msg
    assert exc_info.value.provider == "claude-code"


def test_claude_code_malformed_json_is_typed_not_a_json_crash(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(NotConfiguredError) as exc_info:
        get_claude_code_access_token(path)
    assert "/login" in str(exc_info.value) or "log in" in str(exc_info.value).lower()


def test_claude_code_wrong_shape_is_typed(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"somethingElse": True}), encoding="utf-8")
    with pytest.raises(NotConfiguredError):
        get_claude_code_access_token(path)


def test_claude_code_expired_without_refresh_token_says_relogin(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path, refresh=None, expires_at=int(time.time() * 1000) - 1)
    with pytest.raises(AuthError) as exc_info:
        get_claude_code_access_token(path)
    msg = str(exc_info.value)
    assert "expired" in msg.lower()
    assert "/login" in msg or "log in" in msg.lower()
    assert "sk-ant" not in msg  # never leak token material


def test_claude_code_expired_refresh_failure_says_relogin(tmp_path, monkeypatch) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path, expires_at=int(time.time() * 1000) - 1)

    def boom(url, payload):
        raise OSError("connection refused")

    monkeypatch.setattr("lm15.auth._post_json", boom)
    with pytest.raises(AuthError) as exc_info:
        get_claude_code_access_token(path)
    msg = str(exc_info.value)
    assert "refresh" in msg.lower()
    assert "/login" in msg or "log in" in msg.lower()
    assert "sk-ant" not in msg


def test_claude_code_expired_refresh_success_rotates_and_persists(tmp_path, monkeypatch) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path, expires_at=int(time.time() * 1000) - 1)

    def fake_post(url, payload):
        assert payload["grant_type"] == "refresh_token"
        return {"access_token": "sk-ant-oat-new", "refresh_token": "sk-ant-ort-new", "expires_in": 3600}

    monkeypatch.setattr("lm15.auth._post_json", fake_post)
    assert get_claude_code_access_token(path) == "sk-ant-oat-new"
    on_disk = json.loads(path.read_text(encoding="utf-8"))["claudeAiOauth"]
    assert on_disk["accessToken"] == "sk-ant-oat-new"
    assert on_disk["refreshToken"] == "sk-ant-ort-new"


def test_codex_missing_file_is_typed_with_login_hint(tmp_path) -> None:
    with pytest.raises(NotConfiguredError) as exc_info:
        get_codex_cli_access_token(tmp_path / "nope.json")
    msg = str(exc_info.value)
    assert "codex login" in msg.lower()
    assert exc_info.value.provider == "openai-codex"


def test_codex_malformed_json_is_typed(tmp_path) -> None:
    path = tmp_path / "auth.json"
    path.write_text("][", encoding="utf-8")
    with pytest.raises(NotConfiguredError) as exc_info:
        get_codex_cli_access_token(path)
    assert "codex login" in str(exc_info.value).lower()


def test_codex_expired_without_refresh_token_says_relogin(tmp_path) -> None:
    path = tmp_path / "auth.json"
    _write_codex_auth(path, access=_fake_codex_jwt(exp=int(time.time()) - 10), refresh=None)
    with pytest.raises(AuthError) as exc_info:
        get_codex_cli_access_token(path)
    msg = str(exc_info.value)
    assert "expired" in msg.lower()
    assert "codex login" in msg.lower()


def test_codex_expired_refresh_success_rotates_and_persists(tmp_path, monkeypatch) -> None:
    path = tmp_path / "auth.json"
    _write_codex_auth(path, access=_fake_codex_jwt(exp=int(time.time()) - 10))
    new_access = _fake_codex_jwt("acct_new")

    def fake_post(url, payload):
        assert payload["grant_type"] == "refresh_token"
        return {"access_token": new_access, "refresh_token": "rt-new"}

    monkeypatch.setattr("lm15.auth._post_form", fake_post)
    credential = get_codex_cli_access_token(path)
    assert credential.access_token == new_access
    assert credential.account_id == "acct_new"
    on_disk = json.loads(path.read_text(encoding="utf-8"))["tokens"]
    assert on_disk["access_token"] == new_access
    assert on_disk["refresh_token"] == "rt-new"


def test_missing_credentials_fail_adapter_construction_with_typed_error(tmp_path) -> None:
    with pytest.raises(NotConfiguredError):
        ClaudeCodeLM(credentials_path=tmp_path / "nope.json", transport=_FakeTransport())
    with pytest.raises(NotConfiguredError):
        OpenAICodexLM(auth_path=tmp_path / "nope.json", transport=_FakeTransport())


# ─── secrets never appear in reprs ───────────────────────────────────


def test_credential_repr_never_contains_token_material() -> None:
    credential = LocalOAuthCredential(access_token="sk-ant-oat-secret", refresh_token="sk-ant-ort-secret")
    text = repr(credential)
    assert "sk-ant-oat-secret" not in text
    assert "sk-ant-ort-secret" not in text


def test_adapter_reprs_never_contain_token_material(tmp_path) -> None:
    claude_path = tmp_path / "credentials.json"
    _write_claude_creds(claude_path, access="sk-ant-oat-secret")
    codex_access = _fake_codex_jwt("acct_local")
    codex_path = tmp_path / "auth.json"
    _write_codex_auth(codex_path, access=codex_access)

    lms = [
        ClaudeCodeLM(credentials_path=claude_path, transport=_FakeTransport()),
        OpenAICodexLM(auth_path=codex_path, transport=_FakeTransport()),
        AsyncClaudeCodeLM(credentials_path=claude_path),
        AsyncOpenAICodexLM(auth_path=codex_path),
    ]
    for lm in lms:
        text = repr(lm)
        assert "sk-ant-oat-secret" not in text, type(lm).__name__
        assert codex_access not in text, type(lm).__name__


# ─── build_request: shape + purity (no sockets) ──────────────────────


def test_claude_code_lm_builds_oauth_anthropic_request(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path, refresh=None)
    lm = ClaudeCodeLM(credentials_path=path, transport=_FakeTransport())
    request = Request(model="claude-fable-5", messages=(Message.user("hi"),))

    http = lm.build_request(request, stream=False)
    headers = {key.lower(): value for key, value in http.headers}
    payload = json.loads(http.body)

    assert http.url == "https://api.anthropic.com/v1/messages"
    assert headers["authorization"] == "Bearer sk-ant-oat-local"
    assert "x-api-key" not in headers
    assert headers["x-app"] == "cli"
    assert "oauth-2025-04-20" in headers["anthropic-beta"]
    assert payload["model"] == "claude-fable-5"
    assert payload["system"] == [
        {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}
    ]
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_openai_codex_lm_builds_subscription_headers(tmp_path) -> None:
    path = tmp_path / "auth.json"
    access = _fake_codex_jwt("acct_local")
    _write_codex_auth(path, access=access)
    lm = OpenAICodexLM(auth_path=path, transport=_FakeTransport(), originator="test-suite")
    request = Request(model="gpt-5.5", messages=(Message.user("hi"),))

    http = lm.build_request(request, stream=True)
    headers = {key.lower(): value for key, value in http.headers}
    payload = json.loads(http.body)

    assert http.url == "https://chatgpt.com/backend-api/codex/responses"
    assert headers["authorization"] == f"Bearer {access}"
    assert headers["chatgpt-account-id"] == "acct_local"
    assert headers["openai-beta"] == "responses=experimental"
    assert headers["originator"] == "test-suite"
    assert payload["model"] == "gpt-5.5"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["instructions"] == "You are a helpful assistant."


def test_build_request_opens_no_socket(tmp_path, monkeypatch) -> None:
    claude_path = tmp_path / "credentials.json"
    _write_claude_creds(claude_path)
    codex_path = tmp_path / "auth.json"
    _write_codex_auth(codex_path, access=_fake_codex_jwt("acct_local"))
    claude = ClaudeCodeLM(credentials_path=claude_path, transport=_FakeTransport())
    codex = OpenAICodexLM(auth_path=codex_path, transport=_FakeTransport())

    def guard(*args, **kwargs):
        raise AssertionError("build_request must not open sockets")

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    request = Request(model="m", messages=(Message.user("hi"),))
    claude.build_request(request, stream=True)
    codex.build_request(request, stream=False)


# ─── error normalization: re-login guidance, not env vars ────────────


def test_claude_code_auth_error_guides_relogin_not_env_vars(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path)
    lm = ClaudeCodeLM(credentials_path=path, transport=_FakeTransport())

    body = json.dumps({"type": "error", "error": {"type": "authentication_error", "message": "OAuth token revoked"}})
    err = lm.normalize_error(401, body)

    assert isinstance(err, AuthError)
    assert "OAuth token revoked" in str(err)
    assert "/login" in str(err)
    assert "environment" not in str(err)


def test_codex_auth_error_guides_relogin_not_env_vars(tmp_path) -> None:
    path = tmp_path / "auth.json"
    _write_codex_auth(path, access=_fake_codex_jwt("acct_local"))
    lm = OpenAICodexLM(auth_path=path, transport=_FakeTransport())

    err = lm.normalize_error(401, json.dumps({"error": {"code": "invalid_api_key", "message": "token expired"}}))

    assert isinstance(err, AuthError)
    assert "codex login" in str(err).lower()
    assert "environment" not in str(err)


def test_claude_code_non_auth_errors_keep_sibling_mapping(tmp_path) -> None:
    from lm15.errors import RateLimitError

    path = tmp_path / "credentials.json"
    _write_claude_creds(path)
    lm = ClaudeCodeLM(credentials_path=path, transport=_FakeTransport())
    body = json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    assert isinstance(lm.normalize_error(429, body), RateLimitError)


# ─── streaming: MAP-3 single end event, usage intact ─────────────────


def _codex_lm(transport: _FakeTransport) -> OpenAICodexLM:
    return OpenAICodexLM(api_key=_fake_codex_jwt("acct_local"), account_id="acct_local", transport=transport)


_CODEX_STREAM_BODY = b"".join(
    [
        b'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}\n\n',
        b'data: {"type":"response.output_text.delta","delta":"pout","output_index":0}\n\n',
        b'data: {"type":"response.output_text.delta","delta":"ine","output_index":0}\n\n',
        b'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3},"output":[]}}\n\n',
        b"data: [DONE]\n\n",
    ]
)


def test_openai_codex_stream_emits_exactly_one_end_with_usage() -> None:
    # MAP-3: response.completed carries usage, the trailing [DONE] is a second
    # adapter-level end frame. The public stream must coalesce them into one
    # final end event that keeps the usage.
    transport = _FakeTransport([_FakeResponse(200, _CODEX_STREAM_BODY, headers=[("content-type", "text/event-stream")])])
    events = list(_codex_lm(transport).stream(Request(model="gpt-5.5", messages=(Message.user("hi"),))))

    ends = [event for event in events if event.type == "end"]
    assert len(ends) == 1
    assert events[-1] is ends[0]
    assert ends[0].usage is not None
    assert ends[0].usage.total_tokens == 3
    assert ends[0].finish_reason == "stop"


def test_claude_code_stream_emits_exactly_one_end_with_usage(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path)
    body = b"".join(
        [
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"claude-fable-5"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":4,"output_tokens":5}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
    )
    transport = _FakeTransport([_FakeResponse(200, body, headers=[("content-type", "text/event-stream")])])
    lm = ClaudeCodeLM(credentials_path=path, transport=transport)
    events = list(lm.stream(Request(model="claude-fable-5", messages=(Message.user("hi"),))))

    ends = [event for event in events if event.type == "end"]
    assert len(ends) == 1
    assert events[-1] is ends[0]
    assert ends[0].usage is not None
    assert ends[0].usage.total_tokens == 9
    assert ends[0].finish_reason == "stop"


def test_openai_codex_complete_materializes_streaming_response() -> None:
    transport = _FakeTransport([_FakeResponse(200, _CODEX_STREAM_BODY, headers=[("content-type", "text/event-stream")])])
    lm = _codex_lm(transport)

    response = lm.complete(Request(model="gpt-5.5", messages=(Message.user("say poutine"),)))

    assert response.text == "poutine"
    assert response.message.parts == (TextPart("poutine"),)
    assert response.usage is not None and response.usage.total_tokens == 3
    assert json.loads(transport.requests[0].body)["stream"] is True


def test_openai_codex_empty_stream_yields_empty_text_part_map2() -> None:
    # MAP-2: a response message is never empty.
    body = b"".join(
        [
            b'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}\n\n',
            b'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1},"output":[]}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    transport = _FakeTransport([_FakeResponse(200, body, headers=[("content-type", "text/event-stream")])])
    response = _codex_lm(transport).complete(Request(model="gpt-5.5", messages=(Message.user("hi"),)))
    assert response.message.parts == (TextPart(""),)


def test_claude_code_parse_records_unmapped_canary(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path)
    lm = ClaudeCodeLM(credentials_path=path, transport=_FakeTransport())
    from lm15.providers.base import HttpResponse

    body = json.dumps(
        {
            "id": "msg_1",
            "model": "claude-fable-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "text", "text": "hi"}, {"type": "mystery_block", "stuff": 1}],
        }
    ).encode("utf-8")
    response = lm.parse_response(
        Request(model="claude-fable-5", messages=(Message.user("hi"),)),
        HttpResponse(status=200, reason="OK", headers=[("content-type", "application/json")], body=body),
    )
    assert response.provider_data["_lm15_unmapped"] == [{"path": "content[1]", "type": "mystery_block"}]


def test_subscription_adapters_expose_plural_stream_parser_only() -> None:
    for cls in (ClaudeCodeLM, OpenAICodexLM):
        assert hasattr(cls, "parse_stream_events")
        assert not hasattr(cls, "parse_stream_event")


# ─── async mirrors ───────────────────────────────────────────────────

_SUBSCRIPTION_PAIRS = [
    (ClaudeCodeLM, AsyncClaudeCodeLM),
    (OpenAICodexLM, AsyncOpenAICodexLM),
]


@pytest.mark.parametrize(("sync_cls", "async_cls"), _SUBSCRIPTION_PAIRS)
def test_async_mirror_constructor_field_parity(sync_cls, async_cls) -> None:
    sync_params = {name for name in inspect.signature(sync_cls.__init__).parameters if name != "self"}
    async_fields = {f.name for f in dataclasses.fields(async_cls) if f.init}
    assert async_fields == sync_params, (
        f"{async_cls.__name__} drifted from {sync_cls.__name__}: "
        f"{sorted(async_fields)} != {sorted(sync_params)}"
    )


class _FakeAsyncTransport:
    """Replays one body; never opens a socket."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests = []

    def stream(self, request):
        from lm15.transports import AsyncTransportResponse

        self.requests.append(request)

        async def chunks():
            yield self.body

        async def release(body_consumed: bool) -> None:
            return None

        return AsyncTransportResponse(
            status=self.status,
            reason="OK",
            headers=[("content-type", "text/event-stream")],
            http_version="HTTP/1.1",
            chunks=chunks(),
            release=release,
        )


def test_async_claude_code_stream_mirrors_sync(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path)
    body = b"".join(
        [
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"claude-fable-5"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":4,"output_tokens":5}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
    )
    request = Request(model="claude-fable-5", messages=(Message.user("hi"),))

    sync_transport = _FakeTransport([_FakeResponse(200, body, headers=[("content-type", "text/event-stream")])])
    sync_events = list(ClaudeCodeLM(credentials_path=path, transport=sync_transport).stream(request))

    async_lm = AsyncClaudeCodeLM(credentials_path=path, transport=_FakeAsyncTransport(body))

    async def collect():
        return [event async for event in async_lm.stream(request)]

    assert asyncio.run(collect()) == sync_events


def test_async_codex_complete_materializes_streaming_response() -> None:
    async_lm = AsyncOpenAICodexLM(
        api_key=_fake_codex_jwt("acct_local"),
        account_id="acct_local",
        transport=_FakeAsyncTransport(_CODEX_STREAM_BODY),
    )
    response = asyncio.run(async_lm.complete(Request(model="gpt-5.5", messages=(Message.user("hi"),))))
    assert response.text == "poutine"
    assert response.usage is not None and response.usage.total_tokens == 3


def test_async_mirror_inner_adapter_never_touches_network(tmp_path) -> None:
    path = tmp_path / "credentials.json"
    _write_claude_creds(path)
    async_lm = AsyncClaudeCodeLM(credentials_path=path)
    with pytest.raises(RuntimeError, match="never touch the network"):
        async_lm._inner.complete(Request(model="m", messages=(Message.user("hi"),)))


# ─── live smokes: one tiny call each, skipped without fresh creds ────


def _fresh(read, *args) -> bool:
    try:
        credential = read(*args)
    except Exception:
        return False
    return credential is not None and not credential.expired


_CLAUDE_LIVE = Path.home() / ".claude" / ".credentials.json"
_CODEX_LIVE = Path.home() / ".codex" / "auth.json"


@pytest.mark.skipif(
    not (_CLAUDE_LIVE.exists() and _fresh(read_claude_code_credential)),
    reason="no fresh ~/.claude/.credentials.json",
)
def test_live_claude_code_smoke() -> None:
    lm = ClaudeCodeLM()
    response = lm.complete(
        Request(
            model="claude-haiku-4-5",
            messages=(Message.user("Reply with the single word: ok"),),
            config=Config(max_tokens=16),
        )
    )
    assert response.message.parts
    assert response.usage is not None and response.usage.output_tokens > 0


@pytest.mark.skipif(
    not (_CODEX_LIVE.exists() and _fresh(read_codex_cli_credential)),
    reason="no fresh ~/.codex/auth.json",
)
def test_live_openai_codex_smoke() -> None:
    lm = OpenAICodexLM()
    response = lm.complete(
        Request(
            model="gpt-5.5",
            messages=(Message.user("Reply with the single word: ok"),),
        )
    )
    assert response.message.parts
    assert response.text != ""
