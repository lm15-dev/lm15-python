"""Auth by composition (spec/auth.md AUTH-9).

An adapter is a dialect bound to an access policy value. The subscription
adapters are names for a binding, not behaviours: the same wire must come
out of ``AnthropicLM(access=CLAUDE_CODE)`` and ``ClaudeCodeLM()``, and the
subclass must define nothing but constructors. Every difference between an
API-key client and a login client is a field of the policy, consulted by
the dialect at stated points.
"""

from __future__ import annotations

import json

import pytest

from lm15 import AnthropicLM, AuthError, Config, GeminiLM, Message, NotConfiguredError, OpenAIChatLM, OpenAILM, Request, UnsupportedFeatureError
from lm15.access import (
    ANTHROPIC_API,
    CLAUDE_CODE,
    GEMINI_API,
    OPENAI_API,
    OPENAI_CODEX,
    XAI,
    AccessPolicy,
    has_stored_credential,
    load_credential,
)
from lm15.features import EndpointSupport, ProviderManifest
from lm15.providers.claude_code import ClaudeCodeLM
from lm15.providers.openai_codex import OpenAICodexLM
from lm15.providers.xai import XaiLM
from lm15.testing import FakeTransport
from lm15.types import BuiltinTool, FileUploadRequest

_REQ = Request(model="m", messages=(Message.user("hi"),), system="be brief")


def _wire(lm, request=_REQ, stream=False):
    r = lm.build_request(request, stream=stream)
    return r.method, r.url, dict(r.headers), json.loads(r.body)


# ─── The subscription adapters are names for a binding ───────────────


def test_claude_code_is_anthropic_bound_to_the_policy() -> None:
    named = ClaudeCodeLM(api_key="tok", transport=FakeTransport([]))
    composed = AnthropicLM(api_key="tok", access=CLAUDE_CODE, transport=FakeTransport([]))
    assert _wire(named) == _wire(composed)
    assert named.provider == composed.provider == "claude-code"
    assert named.supports == composed.supports == CLAUDE_CODE.supports


def test_codex_is_openai_bound_to_the_policy() -> None:
    named = OpenAICodexLM(api_key="tok", account_id="acct", transport=FakeTransport([]))
    composed = OpenAILM(api_key="tok", access=OPENAI_CODEX, account_id="acct", transport=FakeTransport([]))
    assert _wire(named) == _wire(composed)
    assert named.base_url == composed.base_url == OPENAI_CODEX.base_url


@pytest.mark.parametrize("cls", [ClaudeCodeLM, OpenAICodexLM])
def test_named_bindings_carry_no_behaviour(cls) -> None:
    # A port needs the policy table and the dialect, not this class: it may
    # hold the class-level policy and constructors, nothing that touches
    # the wire.
    allowed = {"manifest", "__init__", "from_claude_code", "from_codex_cli"}
    defined = {name for name in vars(cls) if not name.startswith("__") or name == "__init__"}
    assert defined - allowed == set(), defined - allowed


def test_xai_composes_only_its_credential_path() -> None:
    # xAI is a provider (image/video wire, refusals live in the class); the
    # policy carries the chain, the hint, and the surfaces.
    assert XaiLM.manifest is XAI
    lm = XaiLM(api_key="k", transport=FakeTransport([]))
    assert lm.access.credential_policy == "oauth-unless-explicit"
    assert lm.supports.images and lm.supports.video and not lm.supports.files


# ─── Consult points ──────────────────────────────────────────────────


def test_anthropic_auth_header_follows_the_policy() -> None:
    _, _, api_headers, _ = _wire(AnthropicLM(api_key="k", transport=FakeTransport([])))
    _, _, login_headers, _ = _wire(AnthropicLM(api_key="k", access=CLAUDE_CODE, transport=FakeTransport([])))
    assert api_headers["x-api-key"] == "k" and "Authorization" not in api_headers
    assert login_headers["Authorization"] == "Bearer k" and "x-api-key" not in login_headers
    for name, value in CLAUDE_CODE.headers:
        assert login_headers[name] == value


def test_anthropic_beta_headers_are_joined_not_replaced() -> None:
    req = Request(model="m", messages=(Message.user("hi"),), tools=(BuiltinTool(name="code_execution"),))
    _, _, headers, _ = _wire(AnthropicLM(api_key="k", access=CLAUDE_CODE, transport=FakeTransport([])), req)
    assert headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20,code-execution-2025-05-22"
    _, _, api_headers, _ = _wire(AnthropicLM(api_key="k", transport=FakeTransport([])), req)
    assert api_headers["anthropic-beta"] == "code-execution-2025-05-22"


def test_system_prefix_goes_first_and_keeps_the_callers_system() -> None:
    _, _, _, body = _wire(AnthropicLM(api_key="k", access=CLAUDE_CODE, transport=FakeTransport([])))
    assert body["system"][0]["text"] == CLAUDE_CODE.system_prefix
    assert body["system"][1]["text"] == "be brief"
    _, _, _, api_body = _wire(AnthropicLM(api_key="k", transport=FakeTransport([])))
    assert api_body["system"] == "be brief"


def test_codex_backend_consult_points() -> None:
    lm = OpenAILM(api_key="tok", access=OPENAI_CODEX, account_id="acct", transport=FakeTransport([]))
    _, url, headers, body = _wire(lm, Request(model="gpt-5.5", messages=(Message.user("hi"),)))
    assert url.startswith(OPENAI_CODEX.base_url)
    assert headers["chatgpt-account-id"] == "acct"
    assert headers["originator"] == "lm15" and headers["OpenAI-Beta"] == "responses=experimental"
    assert body["instructions"] == OPENAI_CODEX.system_prefix
    assert body["store"] is False and body["stream"] is True
    assert "max_output_tokens" not in body
    # A cap or store=True the backend cannot honour is refused, never stripped (MAP-13 rule 4).
    for config, feature in ((Config(max_tokens=5), "config.max_tokens"), (Config(store=True), "config.store")):
        with pytest.raises(UnsupportedFeatureError) as refused:
            _wire(lm, Request(model="gpt-5.5", messages=(Message.user("hi"),), config=config))
        assert refused.value.feature == feature and "openai-codex" in str(refused.value)
    # The detail envelope is the backend's, classified before the OpenAI shape.
    err = lm.normalize_error(400, json.dumps({"detail": "model gpt-x does not exist"}))
    assert type(err).__name__ == "UnsupportedModelError"
    # The models endpoint takes the policy's client_version.
    mreq = lm._models_request()
    assert mreq.url.endswith("/models?client_version=" + OPENAI_CODEX.backend_options["client_version"])


def test_codex_requires_an_account_id() -> None:
    with pytest.raises(NotConfiguredError, match="account id"):
        OpenAILM(api_key="not-a-jwt", access=OPENAI_CODEX, transport=FakeTransport([]))


def test_policy_gates_surfaces_the_dialect_implements() -> None:
    # Anthropic has files; the Claude Code login does not carry them.
    lm = AnthropicLM(api_key="k", access=CLAUDE_CODE, transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="claude-code: files not supported"):
        lm.file_upload(FileUploadRequest(filename="a", bytes_data=b"x"))
    with pytest.raises(UnsupportedFeatureError, match="batch not supported"):
        lm.batch_list()
    assert AnthropicLM(api_key="k", transport=FakeTransport([])).supports.files is True


def test_login_hint_only_when_the_login_won() -> None:
    body = json.dumps({"error": {"type": "authentication_error", "message": "bad token"}})
    # oauth policy: always the login hint (there is no env var).
    err = AnthropicLM(api_key="k", access=CLAUDE_CODE, transport=FakeTransport([])).normalize_error(401, body)
    assert isinstance(err, AuthError) and CLAUDE_CODE.login_hint in str(err)
    # oauth-unless-explicit with an explicit key: generic guidance, not the login.
    err = XaiLM(api_key="k", transport=FakeTransport([])).normalize_error(401, json.dumps({"code": "x", "error": "bad"}))
    assert isinstance(err, AuthError) and XAI.login_hint not in str(err) and "XAI_API_KEY" in str(err)


def test_key_policy_without_a_key_is_a_typed_error() -> None:
    with pytest.raises(NotConfiguredError, match="ANTHROPIC_API_KEY"):
        AnthropicLM(transport=FakeTransport([]))
    with pytest.raises(NotConfiguredError, match="GEMINI_API_KEY"):
        GeminiLM(transport=FakeTransport([]))
    with pytest.raises(NotConfiguredError, match="OPENAI_API_KEY"):
        OpenAIChatLM(transport=FakeTransport([]))


def test_explicit_key_always_wins() -> None:
    loaded = load_credential(XAI, "explicit")
    assert loaded.credential == "explicit" and loaded.source == "explicit"


def test_stored_probe_comes_from_the_policy(monkeypatch) -> None:
    import lm15.access as access

    monkeypatch.setitem(access._STORED_PROBES, "xai", lambda: True)
    assert has_stored_credential(XAI) is True
    assert XaiLM.has_stored_credential() is True
    assert has_stored_credential(ANTHROPIC_API) is False


# ─── The value itself ────────────────────────────────────────────────


def test_policy_is_the_manifest() -> None:
    assert ProviderManifest is AccessPolicy
    assert AnthropicLM.manifest is ANTHROPIC_API
    assert OpenAILM.manifest is OPENAI_API
    assert GeminiLM.manifest is GEMINI_API
    assert ClaudeCodeLM.manifest is CLAUDE_CODE
    assert OpenAICodexLM.manifest is OPENAI_CODEX


def test_oauth_policy_declares_no_env_keys() -> None:
    with pytest.raises(ValueError, match="env_keys"):
        AccessPolicy(provider="p", supports=EndpointSupport(), credential_policy="oauth", env_keys=("K",))


def test_with_headers_replaces_case_insensitively() -> None:
    p = CLAUDE_CODE.with_headers({"User-Agent": "claude-cli/9.9.9"})
    names = [k for k, _ in p.headers]
    assert names.count("User-Agent") == 1 and "user-agent" not in names
    assert dict(p.headers)["User-Agent"] == "claude-cli/9.9.9"
    assert ClaudeCodeLM(api_key="k", claude_code_version="9.9.9", transport=FakeTransport([]))._headers()["user-agent"] == "claude-cli/9.9.9"
