"""Auth hardening: atomic writes, locking, double-checked refresh, authkit
primitives, the doctor report, and the secrecy invariant.

Everything here is hermetic: no network, no real home-directory files, lock
files redirected into tmp via LM15_LOCK_DIR.
"""
from __future__ import annotations

import json
import os
import re
import stat
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import lm15.auth as auth_module
from lm15._authlock import (
    CredentialLockTimeout,
    hold_file_lock,
    lock_path_for,
    write_private_json_atomic,
)
from lm15.auth import (
    LocalOAuthCredential,
    get_claude_code_access_token,
    write_claude_code_credential,
)
from lm15.authkit import (
    CredentialFileStore,
    DeviceCodeExpiredError,
    DeviceComplete,
    DeviceFailed,
    DevicePending,
    DeviceSlowDown,
    OAuthCallbackListener,
    OAuthCallbackResult,
    PKCEPair,
    default_credentials_path,
    generate_pkce,
    pkce_challenge,
    poll_device_code,
)
from lm15.doctor import explain_auth
from lm15.errors import AuthError


def _assert_owner_only(path) -> None:
    """0600 where the file system has POSIX modes. Windows has none: chmod
    can only set read-only, and a file in the user's profile inherits its
    owner-only ACL. The atomic-write checks around each call still run."""
    if os.name == "nt":
        return
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

SENTINEL = "SECRET-SENTINEL-DO-NOT-PRINT"


@pytest.fixture(autouse=True)
def _isolated_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LM15_LOCK_DIR", str(tmp_path / "locks"))


def _write_claude_creds(path: Path, *, access: str, refresh: str | None, expires_at: int) -> None:
    oauth: dict = {"accessToken": access, "expiresAt": expires_at}
    if refresh is not None:
        oauth["refreshToken"] = refresh
    path.write_text(json.dumps({"claudeAiOauth": oauth}), encoding="utf-8")


# ─── atomic writes ───────────────────────────────────────────────────


class TestAtomicWrite:
    def test_writes_content_with_0600_and_no_temp_leftovers(self, tmp_path) -> None:
        target = tmp_path / "creds.json"
        write_private_json_atomic(target, {"k": "v"})
        assert json.loads(target.read_text(encoding="utf-8")) == {"k": "v"}
        _assert_owner_only(target)
        assert [p.name for p in tmp_path.iterdir()] == ["creds.json"]

    def test_failed_replace_preserves_old_file_and_cleans_temp(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "creds.json"
        write_private_json_atomic(target, {"old": True})

        def boom(src, dst):
            raise OSError("simulated crash at replace")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="simulated crash"):
            write_private_json_atomic(target, {"new": True})
        monkeypatch.undo()
        assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
        assert [p.name for p in tmp_path.iterdir()] == ["creds.json"]

    def test_creates_parent_directories(self, tmp_path) -> None:
        target = tmp_path / "nested" / "dir" / "creds.json"
        write_private_json_atomic(target, {})
        assert target.exists()


# ─── locking ─────────────────────────────────────────────────────────


class TestFileLock:
    def test_lock_is_mutually_exclusive(self, tmp_path) -> None:
        target = tmp_path / "creds.json"
        order: list[str] = []
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with hold_file_lock(target):
                order.append("holder-in")
                entered.set()
                release.wait(timeout=5)
                order.append("holder-out")

        thread = threading.Thread(target=holder)
        thread.start()
        assert entered.wait(timeout=5)
        release.set()
        with hold_file_lock(target, timeout_s=5):
            order.append("second-in")
        thread.join(timeout=5)
        assert order == ["holder-in", "holder-out", "second-in"]

    def test_lock_timeout_raises_typed_error_naming_the_file(self, tmp_path) -> None:
        target = tmp_path / "creds.json"
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with hold_file_lock(target):
                entered.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=holder)
        thread.start()
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(CredentialLockTimeout, match=re.escape(str(target))) as caught:
                with hold_file_lock(target, timeout_s=0.2):
                    pass
            assert caught.value.code == "lock_timeout"
            assert caught.value.path == str(target)
            assert caught.value.lock_path.endswith(".lock")
        finally:
            release.set()
            thread.join(timeout=5)

    def test_lock_lives_in_lm15_dir_not_next_to_foreign_file(self, tmp_path) -> None:
        target = tmp_path / "claude-home" / ".credentials.json"
        target.parent.mkdir()
        lock = lock_path_for(target)
        assert lock.parent == (tmp_path / "locks")
        with hold_file_lock(target):
            assert [p.name for p in target.parent.iterdir()] == []


# ─── double-checked refresh ──────────────────────────────────────────


class TestDoubleCheckedRefresh:
    def test_refresh_skipped_when_file_refreshed_while_waiting(self, tmp_path, monkeypatch) -> None:
        """Another process refreshed while we waited for the lock: use its
        token, never spend our (stale, rotated-away) refresh token."""
        path = tmp_path / "credentials.json"
        _write_claude_creds(path, access="stale", refresh="rt-old", expires_at=1)

        def must_not_refresh(_refresh_token: str) -> LocalOAuthCredential:
            raise AssertionError("network refresh must not run after a foreign refresh")

        monkeypatch.setattr(auth_module, "refresh_claude_code_credential", must_not_refresh)

        from contextlib import contextmanager

        real_lock = auth_module.hold_file_lock

        @contextmanager
        def lock_with_foreign_refresh(lock_target, **kwargs):
            with real_lock(lock_target, **kwargs):
                _write_claude_creds(
                    path,
                    access="fresh-from-other-process",
                    refresh="rt-new",
                    expires_at=int(time.time() * 1000) + 3_600_000,
                )
                yield

        monkeypatch.setattr(auth_module, "hold_file_lock", lock_with_foreign_refresh)
        assert get_claude_code_access_token(path) == "fresh-from-other-process"

    def test_refresh_runs_under_lock_and_persists(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "credentials.json"
        _write_claude_creds(path, access="stale", refresh="rt-old", expires_at=1)
        calls: list[str] = []

        def fake_refresh(refresh_token: str) -> LocalOAuthCredential:
            calls.append(refresh_token)
            with pytest.raises(CredentialLockTimeout):
                # Proof we are inside the lock: re-acquiring must time out.
                with hold_file_lock(path, timeout_s=0.1):
                    pass
            return LocalOAuthCredential(
                access_token="fresh",
                refresh_token="rt-rotated",
                expires_at=int(time.time() * 1000) + 3_600_000,
            )

        monkeypatch.setattr(auth_module, "refresh_claude_code_credential", fake_refresh)
        assert get_claude_code_access_token(path) == "fresh"
        assert calls == ["rt-old"]
        stored = json.loads(path.read_text(encoding="utf-8"))["claudeAiOauth"]
        assert stored["accessToken"] == "fresh"
        assert stored["refreshToken"] == "rt-rotated"

    def test_public_write_is_atomic_and_private(self, tmp_path) -> None:
        path = tmp_path / "credentials.json"
        write_claude_code_credential(
            LocalOAuthCredential(access_token="a", refresh_token="r", expires_at=123), path
        )
        _assert_owner_only(path)
        assert json.loads(path.read_text(encoding="utf-8"))["claudeAiOauth"]["accessToken"] == "a"


# ─── PKCE ────────────────────────────────────────────────────────────


class TestPKCE:
    def test_rfc7636_appendix_b_vector(self) -> None:
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        assert pkce_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

    def test_generated_pair_is_valid_and_unique(self) -> None:
        one, two = generate_pkce(), generate_pkce()
        assert one.verifier != two.verifier
        assert 43 <= len(one.verifier) <= 128
        assert re.fullmatch(r"[A-Za-z0-9\-._~]+", one.verifier)
        assert one.challenge == pkce_challenge(one.verifier)
        assert one.method == "S256"

    def test_verifier_not_in_repr(self) -> None:
        pair = PKCEPair(verifier=SENTINEL, challenge="c")
        assert SENTINEL not in repr(pair)


# ─── device-code polling ─────────────────────────────────────────────


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TestDeviceCodePolling:
    def test_completes_after_pending_polls(self) -> None:
        fake = _FakeTime()
        results = iter([DevicePending(), DevicePending(), DeviceComplete({"access": "tok"})])
        value = poll_device_code(
            lambda: next(results), interval_s=5, expires_in_s=900, sleep=fake.sleep, clock=fake.clock
        )
        assert value == {"access": "tok"}
        assert fake.sleeps == [5, 5]  # first poll is immediate by default

    def test_slow_down_grows_interval_by_5s_per_rfc8628(self) -> None:
        fake = _FakeTime()
        results = iter([DeviceSlowDown(), DevicePending(), DeviceComplete("ok")])
        poll_device_code(
            lambda: next(results), interval_s=5, expires_in_s=900, sleep=fake.sleep, clock=fake.clock
        )
        assert fake.sleeps == [10, 10]

    def test_server_supplied_interval_wins_over_step(self) -> None:
        fake = _FakeTime()
        results = iter([DeviceSlowDown(interval_s=30), DeviceComplete("ok")])
        poll_device_code(
            lambda: next(results), interval_s=5, expires_in_s=900, sleep=fake.sleep, clock=fake.clock
        )
        assert fake.sleeps == [30]

    def test_expiry_raises_typed_error(self) -> None:
        fake = _FakeTime()
        with pytest.raises(DeviceCodeExpiredError):
            poll_device_code(
                lambda: DevicePending(), interval_s=5, expires_in_s=12, sleep=fake.sleep, clock=fake.clock
            )

    def test_terminal_failure_raises_auth_error_with_provider(self) -> None:
        with pytest.raises(AuthError) as excinfo:
            poll_device_code(lambda: DeviceFailed("denied by user"), provider="xai")
        assert "denied by user" in str(excinfo.value)
        assert excinfo.value.provider == "xai"

    def test_wait_before_first_poll(self) -> None:
        fake = _FakeTime()
        poll_device_code(
            lambda: DeviceComplete("ok"),
            interval_s=7,
            expires_in_s=900,
            wait_before_first_poll=True,
            sleep=fake.sleep,
            clock=fake.clock,
        )
        assert fake.sleeps == [7]

    def test_token_value_not_in_result_repr(self) -> None:
        assert SENTINEL not in repr(DeviceComplete({"access": SENTINEL}))


# ─── loopback callback listener ──────────────────────────────────────


def _get(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


class TestOAuthCallbackListener:
    def test_receives_code_and_checks_state(self) -> None:
        with OAuthCallbackListener(expected_state="s123") as listener:
            statuses: list[int] = []

            def drive() -> None:
                statuses.append(_get(listener.redirect_uri.replace("/callback", "/wrong")))
                statuses.append(_get(listener.redirect_uri + "?code=c&state=WRONG"))
                statuses.append(_get(listener.redirect_uri + "?code=the-code&state=s123"))

            thread = threading.Thread(target=drive)
            thread.start()
            result = listener.wait(timeout_s=10)
            thread.join(timeout=5)
        assert result == OAuthCallbackResult(code="the-code", state="s123")
        assert statuses == [404, 400, 200]

    def test_provider_error_param_raises_without_code(self) -> None:
        with OAuthCallbackListener() as listener:
            thread = threading.Thread(
                target=_get, args=(listener.redirect_uri + "?error=access_denied",)
            )
            thread.start()
            with pytest.raises(AuthError, match="access_denied"):
                listener.wait(timeout_s=10)
            thread.join(timeout=5)

    def test_binds_loopback_only(self) -> None:
        with OAuthCallbackListener() as listener:
            assert listener.redirect_uri.startswith("http://127.0.0.1:")

    def test_code_not_in_result_repr(self) -> None:
        assert SENTINEL not in repr(OAuthCallbackResult(code=SENTINEL, state="s"))


# ─── credential file store ───────────────────────────────────────────


class TestCredentialFileStore:
    def test_write_read_list_delete_roundtrip(self, tmp_path) -> None:
        store = CredentialFileStore(tmp_path / "credentials.json")
        store.write("anthropic", {"type": "api_key", "key": "k1"})
        store.write("openai", {"type": "api_key", "key": "k2"})
        assert store.read("anthropic") == {"type": "api_key", "key": "k1"}
        assert store.list() == ("anthropic", "openai")
        store.delete("anthropic")
        assert store.read("anthropic") is None
        assert store.list() == ("openai",)

    def test_file_is_private_and_atomic(self, tmp_path) -> None:
        store = CredentialFileStore(tmp_path / "credentials.json")
        store.write("p", {"key": "v"})
        _assert_owner_only(store.path)
        assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["credentials.json"]

    def test_mutate_sees_current_value_inside_lock(self, tmp_path) -> None:
        store = CredentialFileStore(tmp_path / "credentials.json")
        store.write("p", {"n": 1})
        seen: list[dict | None] = []

        def bump(current: dict | None) -> dict:
            seen.append(current)
            return {"n": (current or {"n": 0})["n"] + 1}

        assert store.mutate("p", bump) == {"n": 2}
        assert seen == [{"n": 1}]

    def test_mutate_none_leaves_entry_unchanged(self, tmp_path) -> None:
        store = CredentialFileStore(tmp_path / "credentials.json")
        store.write("p", {"n": 1})
        assert store.mutate("p", lambda _current: None) == {"n": 1}
        assert store.read("p") == {"n": 1}

    def test_read_returns_copy_not_alias(self, tmp_path) -> None:
        store = CredentialFileStore(tmp_path / "credentials.json")
        store.write("p", {"nested": {"a": 1}})
        first = store.read("p")
        assert first is not None
        first["nested"]["a"] = 999
        assert store.read("p") == {"nested": {"a": 1}}

    def test_default_path_honors_env_override(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(tmp_path / "override.json"))
        assert default_credentials_path() == tmp_path / "override.json"
        monkeypatch.delenv("LM15_CREDENTIALS_PATH")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert default_credentials_path() == tmp_path / "xdg" / "lm15" / "credentials.json"

    def test_repr_never_contains_stored_secrets(self, tmp_path) -> None:
        store = CredentialFileStore(tmp_path / "credentials.json")
        store.write("p", {"key": SENTINEL})
        assert SENTINEL not in repr(store)

    def test_corrupt_store_is_a_typed_error_not_a_json_traceback(self, tmp_path) -> None:
        path = tmp_path / "credentials.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape(str(path))):
            CredentialFileStore(path).read("p")


# ─── doctor ──────────────────────────────────────────────────────────


class TestExplainAuth:
    def test_env_key_selected(self) -> None:
        report = explain_auth("anthropic", env={"ANTHROPIC_API_KEY": SENTINEL})
        assert report.configured
        assert report.selected is not None and report.selected.source == "env $ANTHROPIC_API_KEY"

    def test_api_keys_entry_shadows_env(self) -> None:
        report = explain_auth(
            "openai", env={"OPENAI_API_KEY": SENTINEL}, api_keys={"openai": "explicit"}
        )
        assert report.selected is not None and report.selected.source == "explicit api_keys entry"
        assert any(step.state == "shadowed" for step in report.steps)

    def test_router_config_supplies_env_keys_and_settings(self) -> None:
        # explain_auth(config=router.config) describes the router that will
        # send: the provider's settings entry resolves the host, the api_keys
        # entry is rung 0.  Without config the same call names the missing
        # AZURE_OPENAI_RESOURCE instead.
        from lm15.credentials import BearerToken
        from lm15.router import RouterConfig

        config = RouterConfig(
            env={"HOME": "/nonexistent", "PATH": ""},
            api_keys={"azure": lambda: BearerToken("t")},
            settings={"azure": {"resource": "lab"}},
        )
        report = explain_auth("azure", config=config)
        assert report.configured
        assert report.selected is not None and report.selected.source == "explicit api_keys entry"
        assert ("resource", "lab") in report.settings
        assert not any(k == "error" for k, _ in report.settings)

        bare = explain_auth("azure", env=config.env)
        assert any(k == "error" and "AZURE_OPENAI_RESOURCE" in v for k, v in bare.settings)

        # an explicit argument wins over the config's field
        report = explain_auth("azure", config=config, settings={"resource": "other"})
        assert ("resource", "other") in report.settings

    def test_unconfigured_provider_lists_every_absent_rung(self) -> None:
        report = explain_auth("groq", env={})
        assert not report.configured
        assert all(step.state == "absent" for step in report.steps)
        assert "env $GROQ_API_KEY" in report.describe()

    def test_local_server_placeholder_is_last_rung(self) -> None:
        report = explain_auth("ollama", env={})
        assert report.configured
        assert report.selected is not None and "placeholder" in report.selected.source

    def test_oauth_provider_reports_borrowed_file(self, tmp_path) -> None:
        creds = tmp_path / "creds.json"
        _write_claude_creds(
            creds, access=SENTINEL, refresh="rt", expires_at=int(time.time() * 1000) + 3_600_000
        )
        report = explain_auth("claude-code", claude_credentials_path=str(creds))
        assert report.configured
        assert "fresh" in report.describe()

    def test_oauth_provider_missing_file_is_absent(self, tmp_path) -> None:
        report = explain_auth("claude-code", claude_credentials_path=str(tmp_path / "nope.json"))
        assert not report.configured

    def test_unknown_provider_names_known_ones(self) -> None:
        with pytest.raises(ValueError, match="anthropic"):
            explain_auth("not-a-provider")

    def test_underscore_alias_is_accepted(self) -> None:
        assert explain_auth("openai_chat", env={}).provider == "openai-chat"

    def test_report_never_contains_secret_values(self, tmp_path) -> None:
        creds = tmp_path / "creds.json"
        _write_claude_creds(
            creds, access=SENTINEL, refresh=SENTINEL, expires_at=int(time.time() * 1000) + 60_000
        )
        reports = [
            explain_auth("anthropic", env={"ANTHROPIC_API_KEY": SENTINEL}),
            explain_auth("openai", env={}, api_keys={"openai": SENTINEL}),
            explain_auth("claude-code", claude_credentials_path=str(creds)),
        ]
        for report in reports:
            assert SENTINEL not in report.describe()
            assert SENTINEL not in repr(report)


# ─── secrecy invariant across auth error paths ───────────────────────


class TestSecrecyInvariant:
    def test_expired_refresh_failure_error_carries_no_token_material(
        self, tmp_path, monkeypatch
    ) -> None:
        path = tmp_path / "credentials.json"
        _write_claude_creds(path, access=SENTINEL, refresh=SENTINEL, expires_at=1)

        def failing_refresh(_token: str) -> LocalOAuthCredential:
            raise RuntimeError("refresh endpoint said no")

        monkeypatch.setattr(auth_module, "refresh_claude_code_credential", failing_refresh)
        with pytest.raises(AuthError) as excinfo:
            get_claude_code_access_token(path)
        assert SENTINEL not in str(excinfo.value)
        assert SENTINEL not in repr(excinfo.value)

    def test_lock_timeout_message_carries_no_token_material(self, tmp_path) -> None:
        target = tmp_path / "credentials.json"
        _write_claude_creds(target, access=SENTINEL, refresh=SENTINEL, expires_at=1)
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with hold_file_lock(target):
                entered.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=holder)
        thread.start()
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(CredentialLockTimeout) as excinfo:
                with hold_file_lock(target, timeout_s=0.1):
                    pass
            assert SENTINEL not in str(excinfo.value)
        finally:
            release.set()
            thread.join(timeout=5)
