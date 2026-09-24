"""xAI adapter and subscription-auth tests.

Wire facts pinned live 2026-09-01 (grok-4.6 via subscription OAuth):
max_tokens accepted, reasoning_content in the deepseek shape,
stream_options.include_usage honored, token refresh rotates the
refresh token.
"""

from __future__ import annotations

import json
import time

import pytest

import lm15.auth as auth
from lm15.auth import (
    LocalOAuthCredential,
    XaiDeviceAuthorization,
    _parse_xai_device_authorization,
    get_xai_access_token,
    load_xai_credential,
    poll_xai_device_login,
    read_xai_credential,
    write_xai_credential,
)
from lm15.errors import AuthError, NotConfiguredError, UnsupportedFeatureError
from lm15.providers import AsyncXaiLM, XaiLM
from lm15.router import ADAPTERS, ASYNC_ADAPTERS, LMRouter, RouterConfig

NOW_MS = int(time.time() * 1000)
VALID_ENTRY = {"type": "oauth", "access": "at-1", "refresh": "rt-1", "expires": NOW_MS + 3_600_000}
EXPIRED_ENTRY = {"type": "oauth", "access": "at-old", "refresh": "rt-old", "expires": NOW_MS - 1}


def _store(tmp_path, entry, name="credentials.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"xai": entry}), encoding="utf-8")
    return path


# ─── credential loading ─────────────────────────────────────────────


def test_load_from_explicit_path(tmp_path):
    cred = load_xai_credential(_store(tmp_path, VALID_ENTRY))
    assert cred.access_token == "at-1"
    assert cred.refresh_token == "rt-1"
    assert not cred.expired


def test_missing_credential_raises_typed_hint(tmp_path):
    with pytest.raises(NotConfiguredError) as exc:
        load_xai_credential(tmp_path / "nope.json")
    assert "login_xai" in str(exc.value.credential_hint)


def test_read_returns_none_when_missing(tmp_path):
    assert read_xai_credential(tmp_path / "nope.json") is None


def test_default_paths_prefer_lm15_store_then_pi(tmp_path, monkeypatch):
    lm15_path = _store(tmp_path, VALID_ENTRY, "lm15.json")
    pi_path = _store(tmp_path, {**VALID_ENTRY, "access": "at-pi"}, "pi.json")
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(lm15_path))
    monkeypatch.setattr(auth, "PI_AGENT_AUTH_PATH", pi_path)
    assert load_xai_credential().access_token == "at-1"
    lm15_path.unlink()
    assert load_xai_credential().access_token == "at-pi"


def test_fresh_token_returned_without_refresh(tmp_path):
    assert get_xai_access_token(_store(tmp_path, VALID_ENTRY)) == "at-1"


# ─── refresh: rotation must be persisted to the source file ─────────


def test_expired_token_refreshes_and_writes_back(tmp_path, monkeypatch):
    path = _store(tmp_path, EXPIRED_ENTRY)

    def fake_post_form(url, payload):
        assert url == auth.XAI_TOKEN_URL
        assert payload["grant_type"] == "refresh_token"
        assert payload["refresh_token"] == "rt-old"
        return {"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 21600}

    monkeypatch.setattr(auth, "_post_form", fake_post_form)
    assert get_xai_access_token(path) == "at-new"
    stored = json.loads(path.read_text(encoding="utf-8"))["xai"]
    assert stored["access"] == "at-new"
    assert stored["refresh"] == "rt-new"  # rotated token persisted
    assert stored["expires"] > NOW_MS


def test_refresh_without_rotation_keeps_old_refresh_token(tmp_path, monkeypatch):
    path = _store(tmp_path, EXPIRED_ENTRY)
    monkeypatch.setattr(auth, "_post_form", lambda url, payload: {"access_token": "at-new"})
    assert get_xai_access_token(path) == "at-new"
    assert json.loads(path.read_text(encoding="utf-8"))["xai"]["refresh"] == "rt-old"


def test_expired_without_refresh_token_raises(tmp_path):
    path = _store(tmp_path, {"type": "oauth", "access": "at", "expires": NOW_MS - 1})
    with pytest.raises(AuthError):
        get_xai_access_token(path)


def test_write_then_load_roundtrip(tmp_path):
    path = tmp_path / "creds.json"
    write_xai_credential(
        LocalOAuthCredential(access_token="at-w", refresh_token="rt-w", expires_at=NOW_MS + 1000),
        path,
    )
    cred = load_xai_credential(path)
    assert (cred.access_token, cred.refresh_token) == ("at-w", "rt-w")


# ─── device login ───────────────────────────────────────────────────


def test_parse_device_authorization():
    device = _parse_xai_device_authorization(
        {
            "device_code": "dc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://x.ai/device",
            "verification_uri_complete": "https://x.ai/device?code=ABCD-1234",
            "interval": 2,
            "expires_in": 600,
        }
    )
    assert device.user_code == "ABCD-1234"
    assert device.interval_s == 2.0
    assert "dc" not in repr(device)  # device code never leaks


def test_parse_device_authorization_rejects_non_https():
    with pytest.raises(AuthError):
        _parse_xai_device_authorization(
            {"device_code": "dc", "user_code": "u", "verification_uri": "javascript:alert(1)", "expires_in": 600}
        )


def test_poll_device_login_pending_then_complete(monkeypatch):
    responses = iter(
        [
            (False, {"error": "authorization_pending"}),
            (True, {"access_token": "at-dev", "refresh_token": "rt-dev", "expires_in": 3600}),
        ]
    )
    monkeypatch.setattr(auth, "_post_form_tolerant", lambda url, payload: next(responses))
    device = XaiDeviceAuthorization(
        user_code="u", verification_uri="https://x.ai/d", verification_uri_complete=None,
        interval_s=0.0, expires_in_s=60.0, device_code="dc",
    )
    cred = poll_xai_device_login(device, sleep=lambda _s: None)
    assert cred.access_token == "at-dev"


def test_poll_device_login_denied_raises(monkeypatch):
    monkeypatch.setattr(auth, "_post_form_tolerant", lambda url, payload: (False, {"error": "access_denied"}))
    device = XaiDeviceAuthorization(
        user_code="u", verification_uri="https://x.ai/d", verification_uri_complete=None,
        interval_s=0.0, expires_in_s=60.0, device_code="dc",
    )
    with pytest.raises(AuthError):
        poll_xai_device_login(device, sleep=lambda _s: None)


# ─── adapter and routing ────────────────────────────────────────────


def test_adapter_registered_and_rule_routes_grok():
    assert ADAPTERS["xai"] is XaiLM
    assert ASYNC_ADAPTERS["xai"] is AsyncXaiLM
    resolution = LMRouter().resolve("grok-4.6")
    assert (resolution.provider, resolution.model) == ("xai", "grok-4.6")


def test_adapter_compat_pinned_to_live_wire():
    lm = XaiLM(api_key="k")
    assert lm.base_url == "https://api.x.ai/v1"
    assert lm._resolved_compat.max_tokens_field == "max_tokens"
    assert lm._resolved_compat.thinking_format == "deepseek"
    assert lm._resolved_compat.stream_usage == "include"


def test_env_key_used_only_without_stored_login(monkeypatch, tmp_path):
    # oauth-unless-explicit: the env key is the last rung, reached only
    # when no usable subscription login is stored anywhere.
    monkeypatch.setenv("XAI_API_KEY", "env-key")
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(tmp_path / "empty.json"))
    monkeypatch.setattr(auth, "PI_AGENT_AUTH_PATH", tmp_path / "no-pi.json")
    lm = LMRouter().lm("grok-4.6")
    assert isinstance(lm, XaiLM)
    assert lm.api_key == "env-key"


def test_stored_login_shadows_env_key(monkeypatch, tmp_path):
    # Subscription beats ambient env state: it spends no money per token.
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(_store(tmp_path, VALID_ENTRY)))
    monkeypatch.setattr(auth, "PI_AGENT_AUTH_PATH", tmp_path / "no-pi.json")
    lm = LMRouter(RouterConfig(env={"XAI_API_KEY": "env-key"})).lm("grok-4.6")
    assert isinstance(lm, XaiLM)
    assert callable(lm.api_key)
    assert lm.api_key() == "at-1"


def test_explicit_api_keys_entry_shadows_stored_login(monkeypatch, tmp_path):
    # Deliberate in-process configuration always wins.
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(_store(tmp_path, VALID_ENTRY)))
    monkeypatch.setattr(auth, "PI_AGENT_AUTH_PATH", tmp_path / "no-pi.json")
    lm = LMRouter(RouterConfig(env={}, api_keys={"xai": "config-key"})).lm("grok-4.6")
    assert isinstance(lm, XaiLM)
    assert lm.api_key == "config-key"


def test_oauth_used_when_no_env_key(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(_store(tmp_path, VALID_ENTRY)))
    monkeypatch.setattr(auth, "PI_AGENT_AUTH_PATH", tmp_path / "no-pi.json")
    lm = LMRouter(RouterConfig(env={})).lm("grok-4.6")
    assert isinstance(lm, XaiLM)
    assert callable(lm.api_key)
    assert lm.api_key() == "at-1"


def test_oauth_missing_everywhere_raises_login_hint(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(tmp_path / "empty.json"))
    monkeypatch.setattr(auth, "PI_AGENT_AUTH_PATH", tmp_path / "no-pi.json")
    with pytest.raises(NotConfiguredError):
        LMRouter(RouterConfig(env={})).lm("grok-4.6")


def test_unsupported_endpoints_raise():
    lm = XaiLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm.batch_submit(None)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedFeatureError):
        lm.file_upload(None)  # type: ignore[arg-type]


def test_async_mirror_shares_credential_resolution(tmp_path):
    alm = AsyncXaiLM(api_key="k")
    assert alm.provider == "xai"
    assert alm.base_url == "https://api.x.ai/v1"
    assert alm._inner.api_key == "k"


def test_reasoning_off_is_substituted_with_the_lowest_level():
    # Grok reasoning models have no off switch; api.x.ai accepts
    # thinking={"type": "disabled"} but ignores it (live 2026-09-01:
    # grok-4.6 still spent 158 reasoning tokens).  MAP-13 (decision
    # 2026-09-14 §4.2): the lowest level goes instead and is recorded;
    # the spend is visible in usage.  "refuse" keeps the old behaviour.
    from lm15.types import Config, Message, Reasoning, Request
    from tests._adapt import adapted, refuses

    lm = XaiLM(api_key="xai-test")
    request = Request(
        model="grok-4.6",
        messages=(Message.user("12*13?"),),
        config=Config(reasoning=Reasoning(effort="off")),
    )
    out = adapted(lm, request)
    a = out["config.reasoning.effort"]
    assert (a.action, a.asked, a.applied) == ("substituted", "off", "low")
    assert out["__body__"]["reasoning_effort"] == "low"
    refuses(XaiLM(api_key="xai-test", adaptations="refuse"), request, "config.reasoning.effort")


def test_logprobs_is_dropped_and_recorded():
    # docs.x.ai (2026-09-01): "logprobs and top_logprobs are not supported
    # by models grok-4.20 and newer. These fields will be silently ignored
    # if set."  MAP-13: the request is dropped and recorded;
    # Response.logprobs is then absent (the program sees None, not a crash).
    from lm15.types import Config, Message, Request
    from tests._adapt import adapted

    lm = XaiLM(api_key="xai-test")
    request = Request(
        model="grok-4.6",
        messages=(Message.user("hello"),),
        config=Config(logprobs=0),
    )
    out = adapted(lm, request)
    assert out["config.logprobs"].action == "dropped" and out["config.logprobs"].asked == 0
    assert "logprobs" not in out["__body__"]


