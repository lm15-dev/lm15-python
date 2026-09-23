"""Managed authentication (lm15.login, lm15.interactive): spec/auth.md
AUTH-12–26 core scenarios (lm15-contract/auth/managed/scenarios.md).

Every test is offline: a fake auth server behind ``Auth(opener=...)``, a
sandbox store under a temporary directory, a fixed clock, a scripted UI.
Scenario ids (MA-nnn) name the contract scenario each test carries.
Secrets are sentinels; every public rendering is checked for their absence.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from lm15 import Message
from lm15.errors import AuthOperationError, NotConfiguredError, ServerError, TransportError
from lm15.login import Auth, FileStore, LoginCancelled, MemoryStore, TerminalUI, providers
from lm15.login.bound import BoundClient, model_choices
from lm15.login.engine import CallbackListener, LoginContext, parse_manual_return, run_device_flow, DeviceStep
from lm15.login.types import (
    AuthUrlNotice,
    DeviceCodeNotice,
    ManualCodePrompt,
    ModelSelection,
    SecretPrompt,
    SelectPrompt,
    TextPrompt,
)
from lm15.router import LMRouter, RouterConfig
from lm15.testing import FakeResponse, FakeTransport

SENTINEL = "SECRET-SENTINEL-DO-NOT-PRINT"
ACCESS = f"{SENTINEL}-access"
REFRESH = f"{SENTINEL}-refresh"


# ─── fakes ────────────────────────────────────────────────────────────


class _Reply:
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self._body = json.dumps(body).encode()

    def read(self, n: int = -1) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _http_error(url: str, status: int, body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, status, "err", {}, io.BytesIO(json.dumps(body).encode()))


class FakeXai:
    """The xAI device-code endpoints.  ``script`` is the list of poll
    outcomes; ``refresh`` the refresh outcome."""

    def __init__(self, polls=None, refresh=None) -> None:
        self.polls = list(polls or ["pending", "ok"])
        self.refresh = refresh or "ok"
        self.calls: list[tuple[str, dict]] = []
        self.token_n = 0

    def __call__(self, request: urllib.request.Request, timeout: float):
        url = request.full_url
        data = dict(urllib.parse.parse_qsl(request.data.decode())) if request.data else {}
        self.calls.append((url, data))
        if url.endswith("/device/code"):
            return _Reply(200, {"device_code": f"{SENTINEL}-device", "user_code": "ABCD-1234",
                                "verification_uri": "https://auth.x.ai/device", "interval": 1, "expires_in": 600})
        if url.endswith("/token") and data.get("grant_type", "").endswith("device_code"):
            step = self.polls.pop(0) if self.polls else "ok"
            if step == "pending":
                raise _http_error(url, 400, {"error": "authorization_pending"})
            if step == "slow":
                raise _http_error(url, 400, {"error": "slow_down"})
            if step == "denied":
                raise _http_error(url, 400, {"error": "access_denied", "error_description": SENTINEL + "-desc"})
            if step == "expired":
                raise _http_error(url, 400, {"error": "expired_token"})
            if step == "timeout":
                raise urllib.error.URLError(TimeoutError("timed out"))
            self.token_n += 1
            return _Reply(200, {"access_token": f"{ACCESS}{self.token_n}", "refresh_token": f"{REFRESH}{self.token_n}",
                                "expires_in": 3600})
        if url.endswith("/token") and data.get("grant_type") == "refresh_token":
            if self.refresh == "rejected":
                raise _http_error(url, 401, {"error": "invalid_grant", "error_description": SENTINEL})
            if self.refresh == "server":
                raise _http_error(url, 503, {})
            if self.refresh == "timeout":
                raise urllib.error.URLError(TimeoutError("timed out"))
            if self.refresh == "refused":
                raise urllib.error.URLError(ConnectionRefusedError("refused"))
            self.token_n += 1
            return _Reply(200, {"access_token": f"{ACCESS}{self.token_n}", "refresh_token": f"{REFRESH}{self.token_n}",
                                "expires_in": 3600})
        raise AssertionError(f"unexpected request {url}")


class ScriptUI:
    def __init__(self, answers=()) -> None:
        self.answers = list(answers)
        self.prompts: list = []
        self.notices: list = []

    def prompt(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected prompt {prompt!r}")
        answer = self.answers.pop(0)
        return answer(prompt) if callable(answer) else answer

    def notify(self, notice):
        self.notices.append(notice)


def pick(label):
    def choose(prompt):
        assert isinstance(prompt, SelectPrompt), prompt
        for option in prompt.options:
            if label in (option.id, option.label):
                return option.id
        raise AssertionError((label, [o.id for o in prompt.options]))

    return choose


class Clock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.t = start
        self.m = 0.0

    def wall(self) -> float:
        return self.t

    def mono(self) -> float:
        return self.m

    def advance(self, seconds: float) -> None:
        self.t += seconds
        self.m += seconds


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("LM15_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    # No Pi store, no CLI files: mode A must see only the sandbox.
    import lm15.auth as legacy

    monkeypatch.setattr(legacy, "PI_AGENT_AUTH_PATH", tmp_path / "no-pi.json")
    monkeypatch.setattr(legacy, "CLAUDE_CODE_CREDENTIALS_PATH", tmp_path / "no-claude.json")
    monkeypatch.setattr(legacy, "CODEX_CLI_AUTH_PATH", tmp_path / "no-codex.json")
    return tmp_path


def make_auth(sandbox: Path, clock: Clock, server=None, *, memory=False, real_sleep=False) -> Auth:
    store = MemoryStore() if memory else FileStore(sandbox / "credentials.json")
    sleep = None if real_sleep else clock.advance
    return Auth(store, clock=clock.wall, monotonic=clock.mono, opener=server or FakeXai(), sleep=sleep)


def no_secret(*texts: object) -> None:
    for text in texts:
        assert SENTINEL not in repr(text)
        assert SENTINEL not in str(text)


# ─── MA-001 construction is inert; MA-002 discovery ───────────────────


def test_construction_reads_nothing(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: calls.append(self) or "")
    auth = Auth.local(sandbox / "credentials.json")
    auth.providers()
    auth.methods("xai")
    assert providers.xai.id == "xai" and providers.openai_codex.id == "openai-codex"
    assert calls == []
    assert not (sandbox / "credentials.json").exists()


def test_descriptors_are_data_not_authority(sandbox: Path) -> None:
    auth = Auth.memory()
    with pytest.raises(AttributeError):
        providers.not_a_provider
    with pytest.raises(AuthOperationError) as info:
        auth.login("not-a-provider", ui=ScriptUI())
    assert info.value.reason == "method_unavailable"
    claude = [m for m in auth.methods("claude-code")]
    assert {m.id: m.availability for m in claude} == {
        "browser": "unverified", "loopback": "unverified", "external:claude-code-cli": "supported",
    }
    radius = auth.descriptor("radius")
    assert all(m.availability == "unavailable" for m in radius.methods)


# ─── MA-003 a missing choice asks; unverified is opt-in ───────────────


def test_missing_method_asks_never_guesses(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    ui = ScriptUI([pick("device")])
    connection = auth.login("xai", ui=ui)
    assert isinstance(ui.prompts[0], SelectPrompt)
    assert [o.id for o in ui.prompts[0].options] == ["device", "external:pi-xai", "api_key", "env"]  # all supported; unverified/unavailable not offered
    assert connection.kind == "account" and connection.method_id == "device"
    with pytest.raises(AuthOperationError) as info:
        auth.login("claude-code", "browser", ui=ScriptUI())
    assert info.value.reason == "method_unavailable" and "allow_unverified" in str(info.value)
    with pytest.raises(AuthOperationError) as info:
        auth.login("radius", "browser", ui=ScriptUI())
    assert info.value.reason == "method_unavailable"


# ─── MA-016 device pacing; MA-017 the login itself ────────────────────


def test_device_login_saves_actual_expiry_and_one_owner(sandbox: Path) -> None:
    clock = Clock()
    server = FakeXai(polls=["pending", "slow", "ok"])
    auth = make_auth(sandbox, clock, server)
    ui = ScriptUI()
    connection = auth.login("xai", "device", ui=ui)
    notice = next(n for n in ui.notices if isinstance(n, DeviceCodeNotice))
    assert notice.user_code == "ABCD-1234" and notice.verification_url.startswith("https://")
    no_secret(notice, connection, ui.notices, auth.status("xai"))
    document = json.loads((sandbox / "credentials.json").read_text())
    entry = document["xai"]
    assert entry["type"] == "oauth" and entry["expires"] == int(clock.t * 1000) + 3600 * 1000
    assert entry["issued_at"] == int(clock.t * 1000) and entry["lifetime_s"] == 3600.0
    assert document["_lm15"]["slots"]["xai"]["generation"] == "1"
    # The legacy loader reads the same entry: one file, one owner.
    import lm15.auth as legacy

    assert legacy.xai_stored_state() == "usable"
    # Pacing: the slow_down step raised the interval to >= 6 s (1 + 5).
    assert auth.status("xai").usability == "ready"


def test_device_pacing_never_decreases_and_deadline_never_extends() -> None:
    waits: list[float] = []
    clock = Clock()

    class _UI(ScriptUI):
        pass

    ctx = LoginContext(ui=_UI(), deadline=clock.mono() + 100, clock=clock.mono, wall_clock=clock.wall)
    original_wait = ctx.wait

    def wait(seconds: float) -> None:
        waits.append(seconds)
        clock.advance(seconds)

    ctx.wait = wait  # type: ignore[method-assign]
    steps = iter([DeviceStep("pending"), DeviceStep("slow_down", interval_s=2.0), DeviceStep("slow_down"),
                  DeviceStep("complete", "done")])
    assert run_device_flow(ctx, lambda: next(steps), interval_s=5.0, expires_in_s=50.0) == "done"
    assert waits == [5.0, 5.0, 10.0, 15.0]  # provider 5; slow_down(2) -> max(5+5, 2); slow_down -> +5
    assert ctx.deadline == pytest.approx(50.0)  # bounded by the provider's expiry, not extended


def test_device_denied_and_expired_are_typed_without_provider_text(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock, FakeXai(polls=["denied"]))
    with pytest.raises(AuthOperationError) as info:
        auth.login("xai", "device", ui=ScriptUI())
    assert info.value.reason == "login_denied" and info.value.commit_state == "not_committed"
    no_secret(info.value, info.value.__cause__)
    auth2 = make_auth(sandbox, clock, FakeXai(polls=["expired"]))
    with pytest.raises(AuthOperationError) as info:
        auth2.login("xai", "device", ui=ScriptUI())
    assert info.value.reason == "login_expired"
    assert auth.status("xai").presence == "absent"  # nothing half-saved


# ─── MA-011 PKCE; MA-012/013/014 listener hardening ───────────────────


def test_pkce_is_s256_and_fresh() -> None:
    from lm15.authkit import generate_pkce, pkce_challenge

    a, b = generate_pkce(), generate_pkce()
    assert a.verifier != b.verifier and a.method == "S256"
    # RFC 7636 appendix B vector
    assert pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def _get(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - loopback test
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_listener_validates_state_on_success_and_error_and_binds_loopback() -> None:
    clock = Clock()
    ctx = LoginContext(ui=ScriptUI(), deadline=time.monotonic() + 10, provider="t")
    with CallbackListener(path="/cb", expected_state="S", port=0, redirect_host="localhost") as listener:
        base = f"http://127.0.0.1:{listener.port}"
        assert listener.redirect_uri == f"http://localhost:{listener.port}/cb"  # registered host != bind host
        assert _get(f"{base}/other?code=x&state=S") == 404
        assert _get(f"{base}/cb?code=x&state=WRONG") == 400  # wrong state: generic, keeps waiting
        assert _get(f"{base}/cb?error=access_denied&state=WRONG") == 400  # error with wrong state: ignored
        assert _get(f"{base}/cb?code=x&error=y&state=S") == 400  # both: invalid
        assert _get(f"{base}/cb?code=x&code=y&state=S") == 400  # duplicate params
        assert not listener._done.is_set()
        assert _get(f"{base}/cb?code={SENTINEL}&state=S") == 200
        result = listener.wait(ctx)
        assert result is not None and result.code == SENTINEL
        no_secret(result)
        assert listener._done.is_set()  # one-shot: the serve loop ends with the first accepted return
    with pytest.raises(AuthOperationError):
        CallbackListener(path="/cb", expected_state=None, bind_host="0.0.0.0")


def test_listener_rejects_oversized_target() -> None:
    with CallbackListener(path="/cb", expected_state=None, port=0) as listener:
        assert _get(f"http://127.0.0.1:{listener.port}/cb?code=" + "a" * 9000) == 414


def test_manual_return_parsing_checks_context() -> None:
    ok = parse_manual_return("http://localhost:1/cb?code=C&state=S", expected_state="S", allow_bare_code=False,
                             registered_path="/cb")
    assert ok.code == "C" and ok.state == "S"
    with pytest.raises(AuthOperationError) as info:
        parse_manual_return("http://localhost:1/cb?code=C&state=X", expected_state="S", allow_bare_code=False,
                            registered_path="/cb")
    assert info.value.reason == "invalid_login_state"
    with pytest.raises(AuthOperationError):
        parse_manual_return("http://localhost:1/wrong?code=C&state=S", expected_state="S", allow_bare_code=False,
                            registered_path="/cb")
    with pytest.raises(AuthOperationError):  # a bare code cannot supply state
        parse_manual_return("C", expected_state="S", allow_bare_code=False, registered_path="/cb")
    assert parse_manual_return("C#S", expected_state="S", allow_bare_code=False, registered_path="/cb").code == "C"
    assert parse_manual_return("C", expected_state=None, allow_bare_code=True, registered_path="/x").code == "C"


# ─── MA-019 literal keys; MA-054 no mystery strings ───────────────────


def test_keys_are_literal_text(sandbox: Path) -> None:
    auth = Auth.memory()
    connection = auth.set_api_key("groq", "!shell echo $HOME")
    assert auth.request_auth("groq").credential.value == "!shell echo $HOME"
    no_secret(connection)
    with pytest.raises(AuthOperationError) as info:
        auth.set_api_key("groq", "   ")
    assert info.value.reason == "interaction_required"


# ─── MA-025/026 replacement; MA-029 logout; MA-028 cancel vs commit ───


def test_replacement_is_conditional_and_old_stays_usable(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    first = auth.login("xai", "device", ui=ScriptUI())
    with pytest.raises(AuthOperationError) as info:
        auth.login("xai", "device", ui=ScriptUI())
    assert info.value.reason == "connection_exists" and info.value.connection_id == first.id
    assert auth.request_auth("xai").credential.value == f"{ACCESS}1"  # still usable
    with pytest.raises(AuthOperationError) as info:
        auth.login("xai", "device", ui=ScriptUI(), replace="cn_wrong")
    assert info.value.reason == "connection_changed"
    second = auth.login("xai", "device", ui=ScriptUI(), replace=first.id)
    assert second.id != first.id and second.identity_generation == "2"
    assert auth.request_auth("xai").credential.value == f"{ACCESS}2"


def test_logout_is_scoped_idempotent_and_leaves_a_marker(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth.set_api_key("groq", "k1")
    xai = auth.login("xai", "device", ui=ScriptUI())
    result = auth.logout("xai")
    assert result.forgot and result.identity_generation == "2"
    assert auth.logout("xai").forgot is False  # idempotent
    assert auth.logout(xai.id).forgot is False  # the old id does not remove anything newer
    status = auth.status("xai")
    assert status.presence == "absent" and status.logged_out
    assert "xai" not in json.loads((sandbox / "credentials.json").read_text())  # material gone
    assert auth.request_auth("groq").credential.value == "k1"  # other slots untouched
    with pytest.raises(AuthOperationError) as info:
        auth.request_auth("xai")
    assert info.value.reason == "login_required" and "signed out" in str(info.value)


def test_cancel_before_commit_wins_and_after_commit_does_not(sandbox: Path) -> None:
    clock = Clock()
    server = FakeXai(polls=["pending"] * 50)
    auth = make_auth(sandbox, clock, server, real_sleep=True)
    cancel = threading.Event()
    started = threading.Event()

    class _UI(ScriptUI):
        def notify(self, notice):
            super().notify(notice)
            started.set()

    outcome: dict = {}

    def run() -> None:
        try:
            auth.login("xai", "device", ui=_UI(), cancel=cancel)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    started.wait(5)
    assert auth.cancel_login("xai") == "cancelled"
    thread.join(10)
    assert isinstance(outcome.get("error"), LoginCancelled)
    assert auth.status("xai").presence == "absent"
    document = json.loads((sandbox / "credentials.json").read_text())
    assert "attempt" not in document["_lm15"]["slots"].get("xai", {})
    # After a commit, cancel reports complete and undo is logout.
    auth._sleep = clock.advance
    auth._opener = FakeXai()
    auth.login("xai", "device", ui=ScriptUI())
    assert auth.cancel_login("xai") == "complete"


def test_one_active_attempt_per_slot(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth._reserve("xai", "at_other", None, 900.0)
    with pytest.raises(AuthOperationError) as info:
        auth.login("xai", "device", ui=ScriptUI())
    assert info.value.reason == "login_in_progress"
    with pytest.raises(AuthOperationError) as info:
        auth.set_api_key("xai", "k")
    assert info.value.reason == "login_in_progress"
    clock.advance(20 * 60)  # a stale reservation expires
    auth.login("xai", "device", ui=ScriptUI())


def test_an_interrupt_ends_the_attempt_and_frees_the_slot(sandbox: Path) -> None:
    # Ctrl-C — a notebook's cancel — while waiting for the provider, or at
    # a prompt: the attempt is over, so its reservation must not block the
    # next sign-in for the rest of its lifetime.
    clock = Clock()

    def interrupted(_seconds: float) -> None:
        raise KeyboardInterrupt

    auth = Auth(FileStore(sandbox / "credentials.json"), clock=clock.wall, monotonic=clock.mono,
                opener=FakeXai(polls=["pending"] * 5), sleep=interrupted)
    with pytest.raises(KeyboardInterrupt):
        auth.login("xai", "device", ui=ScriptUI())
    document = json.loads((sandbox / "credentials.json").read_text())
    assert "attempt" not in document["_lm15"]["slots"].get("xai", {})
    auth._sleep = clock.advance
    auth._opener = FakeXai()
    assert auth.login("xai", "device", ui=ScriptUI()).provider == "xai"


# ─── MA-032/033/036/037 renewal ────────────────────────────────────────


def test_renewal_is_due_at_lead_and_bumps_revision(sandbox: Path) -> None:
    clock = Clock()
    server = FakeXai()
    auth = make_auth(sandbox, clock, server)
    auth.login("xai", "device", ui=ScriptUI())
    clock.advance(3600 - 301)  # one second before the 5-minute lead: still fresh
    assert auth.request_auth("xai").credential.value == f"{ACCESS}1"
    assert auth.status("xai").usability == "ready"
    clock.advance(2)
    assert auth.status("xai").usability == "renewal_due"
    assert auth.request_auth("xai").credential.value == f"{ACCESS}2"
    assert auth.status("xai").connection.credential_revision == "2"
    document = json.loads((sandbox / "credentials.json").read_text())
    assert document["xai"]["refresh"] == f"{REFRESH}2" and "renewal_in_flight" not in document["_lm15"]["slots"]["xai"]
    refreshes = [d for u, d in server.calls if d.get("grant_type") == "refresh_token"]
    assert len(refreshes) == 1


def test_short_token_lead_is_lifetime_over_ten(sandbox: Path) -> None:
    auth = Auth.memory()
    assert auth._lead_ms(60.0) == 6000 and auth._lead_ms(3600.0) == 300_000 and auth._lead_ms(None) == 300_000


def test_sibling_refresh_is_reused_not_repeated(sandbox: Path) -> None:
    clock = Clock()
    server = FakeXai()
    auth = make_auth(sandbox, clock, server)
    auth.login("xai", "device", ui=ScriptUI())
    clock.advance(3400)
    other = make_auth(sandbox, clock, server)  # another process, same file
    other.request_auth("xai")  # renews
    auth.request_auth("xai")  # re-reads under the lock: fresh now, no second exchange
    refreshes = [d for u, d in server.calls if d.get("grant_type") == "refresh_token"]
    assert len(refreshes) == 1


def test_permanent_rejection_marks_needs_login_and_never_falls_back(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock, FakeXai(refresh="rejected"))
    auth.login("xai", "device", ui=ScriptUI())
    clock.advance(3400)
    monkeypatch.setenv("XAI_API_KEY", SENTINEL)
    with pytest.raises(AuthOperationError) as info:
        auth.request_auth("xai")
    assert info.value.reason == "credential_rejected" and info.value.commit_state == "committed"
    no_secret(info.value)
    assert auth.status("xai").usability == "needs_login"
    with pytest.raises(AuthOperationError) as info:
        auth.request_auth("xai")
    assert info.value.reason == "login_required"
    router = LMRouter(RouterConfig(auth=auth))
    with pytest.raises(AuthOperationError):
        router.lm("xai:grok-4")  # no env fallback in managed mode


def test_transient_failure_keeps_credentials(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock, FakeXai(refresh="server"))
    auth.login("xai", "device", ui=ScriptUI())
    clock.advance(3400)
    with pytest.raises(ServerError):
        auth.request_auth("xai")
    assert auth.status("xai").usability == "renewal_due"  # nothing lost, nothing marked
    auth2 = make_auth(sandbox, clock, FakeXai(refresh="refused"))
    with pytest.raises(TransportError):
        auth2.request_auth("xai")
    assert auth2.status("xai").usability == "renewal_due"


def test_uncertain_exchange_is_indeterminate_not_retried(sandbox: Path) -> None:
    clock = Clock()
    server = FakeXai(refresh="timeout")
    auth = make_auth(sandbox, clock, server)
    auth.login("xai", "device", ui=ScriptUI())
    clock.advance(3400)
    with pytest.raises(AuthOperationError) as info:
        auth.request_auth("xai")
    assert info.value.reason == "indeterminate" and info.value.commit_state == "unknown"
    assert auth.status("xai").usability == "indeterminate"
    with pytest.raises(AuthOperationError) as info:
        auth.request_auth("xai")  # never a blind retry
    assert info.value.reason == "indeterminate"
    assert len([d for u, d in server.calls if d.get("grant_type") == "refresh_token"]) == 1
    auth.login("xai", "device", ui=ScriptUI(), replace=auth.status("xai").connection.id)  # recovery
    assert auth.status("xai").usability == "ready"


# ─── MA-040 (structural) / MA-041 storage ─────────────────────────────


def test_unreadable_store_is_not_absence(sandbox: Path) -> None:
    path = sandbox / "credentials.json"
    path.write_text("{not json")
    auth = Auth.local(path)
    with pytest.raises(AuthOperationError) as info:
        auth.status("xai")
    assert info.value.reason == "storage_unavailable"
    assert path.read_text() == "{not json"  # never overwritten
    path.write_text(json.dumps({"_lm15": {"version": 99, "slots": {}}}))
    with pytest.raises(AuthOperationError) as info:
        auth.connections()
    assert info.value.reason == "unsupported_store_version"
    path.write_text('{"xai": {"a": 1}, "xai": {"b": 2}}')
    with pytest.raises(AuthOperationError):
        auth.status("xai")


def test_generations_are_decimal_strings_and_survive_reload(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth.set_api_key("groq", "k")
    document = json.loads((sandbox / "credentials.json").read_text())
    document["_lm15"]["slots"]["groq"]["generation"] = str(2**60)
    (sandbox / "credentials.json").write_text(json.dumps(document))
    assert auth.status("groq").connection.identity_generation == str(2**60)
    auth.logout("groq")
    assert auth.status("groq").presence == "absent"
    assert json.loads((sandbox / "credentials.json").read_text())["_lm15"]["slots"]["groq"]["generation"] == str(2**60 + 1)


# ─── MA-005/007/060 identity selection and the legacy boundary ────────


def test_managed_router_order_explicit_then_connection_never_env(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth.login("xai", "device", ui=ScriptUI())
    env = {"XAI_API_KEY": SENTINEL, "GEMINI_API_KEY": SENTINEL}
    router = LMRouter(RouterConfig(auth=auth, env=env))
    assert "managed connection" in router.lm("xai:grok-4").credential_origin()
    with pytest.raises(AuthOperationError) as info:
        router.lm("gemini:gemini-2.5-flash")
    assert info.value.reason == "login_required"
    no_secret(info.value)
    explicit = LMRouter(RouterConfig(auth=auth, env=env, api_keys={"xai": "explicit"}))
    assert "explicit" in explicit.lm("xai:grok-4").credential_origin()
    assert "placeholder" in router.lm("ollama:llama3").credential_origin()  # keyless local still works
    with pytest.raises(AuthOperationError) as info:
        router.lm("azure:gpt-4o")  # the machine's cloud identity is not discovered
    assert info.value.reason == "login_required"


def test_explicit_failure_is_not_a_fallback(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth.login("xai", "device", ui=ScriptUI())
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("callable failed")

    transport = FakeTransport([_chat_reply()])
    router = LMRouter(RouterConfig(auth=auth, api_keys={"xai": broken}, env={"XAI_API_KEY": SENTINEL}, transport=transport))
    lm = router.lm("xai:grok-4")
    with pytest.raises(RuntimeError):
        lm.complete(__import__("lm15").Request(model="grok-4", messages=(Message.user("hi"),)))
    assert calls == [1] and transport.requests == []  # the callable was tried once; nothing else was consulted or sent


def test_mode_a_subscription_first_then_blocks_after_failure(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    env = {"XAI_API_KEY": SENTINEL}
    # never connected: the env key works (R2: key-only users keep working)
    assert "env" in LMRouter(RouterConfig(env=env)).lm("xai:grok-4").credential_origin()
    auth.login("xai", "device", ui=ScriptUI())
    # subscription beats the ambient key
    assert "stored local login" in LMRouter(RouterConfig(env=env)).lm("xai:grok-4").credential_origin()
    # signed out: the ambient key is blocked and the error says so
    auth.logout("xai")
    with pytest.raises(NotConfiguredError) as info:
        LMRouter(RouterConfig(env=env)).lm("xai:grok-4")
    message = str(info.value)
    assert "signed out" in message and "$XAI_API_KEY is set but is used only when passed explicitly" in message
    no_secret(info.value)
    # explicit key after logout is deliberate
    assert "explicit" in LMRouter(RouterConfig(env=env, api_keys={"xai": "k"})).lm("xai:grok-4").credential_origin()
    # expired without refresh: unusable, blocks too
    (sandbox / "credentials.json").write_text(json.dumps({"xai": {"type": "oauth", "access": SENTINEL, "expires": 1}}))
    with pytest.raises(NotConfiguredError) as info:
        LMRouter(RouterConfig(env=env)).lm("xai:grok-4")
    assert "expired and cannot be renewed" in str(info.value)
    no_secret(info.value)


def test_doctor_mirrors_managed_and_legacy_selection(sandbox: Path) -> None:
    from lm15.doctor import explain_auth

    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth.login("xai", "device", ui=ScriptUI())
    report = explain_auth("xai", config=RouterConfig(auth=auth, env={"XAI_API_KEY": SENTINEL}))
    assert [s.kind for s in report.steps] == ["api_keys", "connection", "env:XAI_API_KEY"]
    assert [s.state for s in report.steps] == ["absent", "selected", "shadowed"] and report.configured
    no_secret(report)
    auth.logout("xai")
    legacy = explain_auth("xai", env={"XAI_API_KEY": SENTINEL})
    assert [s.state for s in legacy.steps] == ["absent", "absent", "shadowed"] and not legacy.configured


def test_legacy_login_xai_uses_the_managed_flow(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lm15.auth as legacy
    import lm15.login.manager as manager

    server = FakeXai()
    monkeypatch.setattr(manager.Auth, "__init__", _patched_init(server))
    echoed: list[str] = []
    credential = legacy.login_xai(echo=echoed.append)
    assert credential.access_token == f"{ACCESS}1" and echoed == ["Open https://auth.x.ai/device and enter code: ABCD-1234"]
    assert legacy.xai_stored_state() == "usable"


def _patched_init(server):
    original = Auth.__init__

    def init(self, store, **kwargs):
        kwargs.setdefault("opener", server)
        original(self, store, **kwargs)

    return init


# ─── MA-058/059/060 connect() and the bound client ────────────────────


def _models_reply() -> FakeResponse:
    body = {"object": "list", "data": [{"id": "grok-4", "object": "model"}, {"id": "grok-3-mini", "object": "model"}]}
    return FakeResponse(200, body=json.dumps(body).encode())


def _chat_reply(text: str = "hello") -> FakeResponse:
    body = {"id": "x", "object": "chat.completion", "model": "grok-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    return FakeResponse(200, body=json.dumps(body).encode())


def test_connect_saves_login_before_picker_and_binds(sandbox: Path) -> None:
    from lm15.interactive import connect

    clock = Clock()
    auth = make_auth(sandbox, clock)
    transport = FakeTransport([_models_reply(), _chat_reply()])
    ui = ScriptUI([pick("xai"), pick("device"), pick("grok-4")])
    lm = connect(auth=auth, ui=ui, router_config=RouterConfig(transport=transport))
    assert lm.selection.routed == "xai:grok-4" and lm.connection.kind == "account"
    reply = lm.complete(messages=[Message.user("hi")])
    assert reply.text == "hello"
    sent = dict(transport.requests[-1].headers)
    assert sent["Authorization"] == f"Bearer {ACCESS}1"
    # the picker was cancelled on a second connect: the login stays saved
    ui2 = ScriptUI([pick(lm.connection.id), lambda p: (_ for _ in ()).throw(KeyboardInterrupt())])
    with pytest.raises(AuthOperationError):
        connect(auth=auth, ui=ui2, router_config=RouterConfig(transport=FakeTransport([_models_reply()])))
    assert auth.status("xai").presence == "saved"


def test_connect_offers_env_key_as_explicit_choice_only(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lm15.interactive import connect

    monkeypatch.setenv("GROQ_API_KEY", "gsk-env")
    auth = Auth.memory()
    transport = FakeTransport([_models_reply(), _chat_reply()])
    ui = ScriptUI([pick("groq"), pick("env"), pick("llama")])
    ui.answers[-1] = lambda p: p.options[0].id
    lm = connect(auth=auth, ui=ui, router_config=RouterConfig(transport=transport))
    method_prompt = ui.prompts[1]
    assert isinstance(method_prompt, SelectPrompt) and {o.id for o in method_prompt.options} == {"api_key", "env"}
    assert lm.connection.method_id == "env"
    document = auth.store.read()
    assert document["groq"] == {"type": "env", "name": "GROQ_API_KEY"}  # the name, never the value
    lm.complete(messages=[Message.user("hi")])
    assert dict(transport.requests[-1].headers)["Authorization"] == "Bearer gsk-env"


def test_connect_refuses_without_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    from lm15.interactive import connect

    monkeypatch.setattr("lm15.interactive._interactive_terminal", lambda: False)
    with pytest.raises(AuthOperationError) as info:
        connect(auth=Auth.memory())
    assert info.value.reason == "interaction_required"


def test_bound_client_is_pinned_and_thin(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    first = auth.login("xai", "device", ui=ScriptUI())
    selection = ModelSelection(provider="xai", model="grok-4", connection_id=first.id,
                               identity_generation=first.identity_generation)
    transport = FakeTransport([_chat_reply("one"), _chat_reply("two")])
    client = BoundClient(auth, selection, router_config=RouterConfig(transport=transport))
    request = client.request("hi")
    assert request.model == "xai:grok-4" and request.messages[0].role == "user"
    assert client.complete(request).text == "one"
    with pytest.raises(AuthOperationError) as info:
        client.complete(request.__class__(model="xai:grok-3", messages=request.messages))
    assert info.value.reason == "selection_mismatch"
    # renewal follows the same id; replacement does not
    clock.advance(3400)
    assert client.complete(messages=[Message.user("again")]).text == "two"
    auth.login("xai", "device", ui=ScriptUI(), replace=first.id)
    with pytest.raises(AuthOperationError) as info:
        client.complete(messages=[Message.user("x")])
    assert info.value.reason == "connection_changed"
    auth.logout("xai")
    with pytest.raises(AuthOperationError) as info:
        client.complete(messages=[Message.user("x")])
    assert info.value.reason == "login_required"
    client.close()
    assert auth.status("xai").presence == "absent"  # close is not logout; logout was explicit above


def test_model_choices_carry_provenance_and_capability_filter(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    auth.login("xai", "device", ui=ScriptUI())
    choices = model_choices(auth, "xai", refresh=True, router_config=RouterConfig(transport=FakeTransport([_models_reply()])))
    assert [c.model for c in choices] == ["grok-4", "grok-3-mini"] and all(c.source == "provider" for c in choices)
    assert model_choices(auth, "xai") == ()  # no application registry: nothing invented
    strict = model_choices(auth, "xai", refresh=True, capability="structured-output",
                           router_config=RouterConfig(transport=FakeTransport([_models_reply()])))
    assert strict == ()  # unknown is not offered without opt-in
    with pytest.raises(AuthOperationError):
        model_choices(auth, "gemini")


# ─── MA-043/044 secrecy ───────────────────────────────────────────────


def test_no_secret_in_any_public_rendering(sandbox: Path) -> None:
    clock = Clock()
    auth = make_auth(sandbox, clock)
    connection = auth.login("xai", "device", ui=ScriptUI())
    no_secret(auth, auth.store, connection, auth.status("xai"), auth.connections(), auth.request_auth("xai"))
    error = AuthOperationError("x", reason="login_required")
    no_secret(error)
    assert error.code == "auth_operation" and error.reason == "login_required"
    with pytest.raises(ValueError):
        AuthOperationError("x", reason="made-up")


def test_terminal_ui_prompts_and_does_not_open_browser_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    import io as _io

    out = _io.StringIO()
    answers = iter(["2", "pasted"])
    ui = TerminalUI(out=out, inp=lambda _prompt: next(answers))
    ui.notify(AuthUrlNotice(url="https://example.com/auth", instructions="open it"))
    ui.notify(DeviceCodeNotice(user_code="AB-12", verification_url="https://example.com/d", expires_in_s=600, interval_s=5))
    assert ui.prompt(SelectPrompt("m", "pick", (__import__("lm15.login.types", fromlist=["SelectOption"]).SelectOption("a", "A"),
                                                __import__("lm15.login.types", fromlist=["SelectOption"]).SelectOption("b", "B")))) == "b"
    assert ui.prompt(ManualCodePrompt("r", "paste")) == "pasted"
    text = out.getvalue()
    assert "https://example.com/auth" in text and "AB-12" in text


# ─── external sources (R1): read in place, never copied ───────────────


def test_external_claude_cli_source_reads_in_place(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lm15.auth as legacy

    claude = sandbox / "claude.json"
    claude.write_text(json.dumps({"claudeAiOauth": {"accessToken": ACCESS, "refreshToken": REFRESH,
                                                    "expiresAt": int(time.time() * 1000) + 3_600_000}}))
    monkeypatch.setattr(legacy, "CLAUDE_CODE_CREDENTIALS_PATH", claude)
    auth = Auth.memory()
    with pytest.raises(AuthOperationError):
        auth.request_auth("claude-code")  # a managed scope does not borrow implicitly
    connection = auth.configure("claude-code", method="external:claude-code-cli")
    assert connection.kind == "account"
    assert auth.store.read()["claude-code"] == {"type": "external", "source": "claude-code-cli"}  # no token copied
    assert auth.request_auth("claude-code").credential.value == ACCESS
    auth.logout("claude-code")
    assert json.loads(claude.read_text())["claudeAiOauth"]["accessToken"] == ACCESS  # the tool's file is untouched
