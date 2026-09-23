"""Claude login construction and manual-return validation, entirely offline.

Hosted endpoint/redirect/PKCE length: native Claude Code 2.1.280 capture,
2026-09-23, run 20260923-072216-l1ioisy1. Client ID also matches Earendil
Pi v0.87.0 (16787ad5b2dc748047f314ca1bfe7708f30f54f3), oauth/anthropic.ts.
These checks do not establish provider permission, billing or live LM15 login.
"""

import base64
import hashlib
import json
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from lm15 import auth as legacy
from lm15.errors import AuthOperationError
from lm15.login.engine import CallbackReturn, LoginCancelled, LoginContext, LoginDenied, parse_manual_return
from lm15.login.flows import claude
from lm15.login.types import AuthUrlNotice, ManualCodePrompt

# Independent expected facts, not aliases of the implementation constants.
EXPECTED_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
HOSTED_RETURN = "https://platform.claude.com/oauth/code/callback"


class Reply:
    status = 200

    def read(self, limit=-1):
        return json.dumps({"access_token": "test-access", "refresh_token": "test-refresh", "expires_in": 3600}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class UI:
    def __init__(self, form="code_state"):
        self.authorization = None
        self.form = form

    def notify(self, notice):
        if isinstance(notice, AuthUrlNotice):
            self.authorization = notice.url

    def prompt(self, prompt):
        assert isinstance(prompt, ManualCodePrompt)
        state = parse_qs(urlsplit(self.authorization).query)["state"][0]
        if self.form == "code_state":
            return "test-code#" + state
        if self.form == "url":
            return HOSTED_RETURN + "?" + urlencode({"code": "test-code", "state": state})
        return self.form.replace("EXPECTED_STATE", state)


def forbid_listener(**kwargs):
    raise AssertionError("Hosted login must not open a localhost listener")


@pytest.mark.parametrize("form", ["code_state", "url"])
def test_hosted_flow_has_no_listener_and_matches_captured_registration(monkeypatch, form):
    monkeypatch.setattr(claude, "CallbackListener", forbid_listener)
    requests = []

    def opener(request, timeout):
        assert request.full_url == "https://platform.claude.com/v1/oauth/token"
        assert request.get_header("Content-type") == "application/json"
        payload = json.loads(request.data)
        assert payload["client_id"] == EXPECTED_CLIENT_ID
        requests.append(payload)
        return Reply()

    ui = UI(form)
    ctx = LoginContext(ui=ui, deadline=time.monotonic() + 30, provider="claude-code", opener=opener)
    flow = claude.ClaudeFlow()
    result = flow.login(ctx, claude.METHOD_BROWSER, {}, {})
    url = urlsplit(ui.authorization)
    assert (url.scheme, url.netloc, url.path) == ("https", "claude.com", "/cai/oauth/authorize")
    query = parse_qs(url.query)
    assert query["client_id"] == [EXPECTED_CLIENT_ID]
    assert query["redirect_uri"] == [HOSTED_RETURN]
    assert query["code_challenge_method"] == ["S256"]
    token_request = requests[0]
    assert set(token_request) == {"grant_type", "code", "redirect_uri", "client_id", "code_verifier", "state"}
    assert token_request["grant_type"] == "authorization_code"
    assert token_request["code"] == "test-code"
    assert token_request["redirect_uri"] == HOSTED_RETURN
    assert token_request["state"] == query["state"][0]
    verifier = token_request["code_verifier"]
    assert len(verifier) == 43
    assert len(token_request["state"]) == 43
    assert token_request["state"] != verifier
    expected_challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == [expected_challenge]
    assert claude.METHOD_BROWSER.availability == "unverified"
    assert claude.METHOD_BROWSER.delivery == ("manual",)
    flow.renew(ctx, result.material, {})
    assert len(requests) == 2
    assert requests[1]["grant_type"] == "refresh_token"


@pytest.mark.parametrize("pasted", [
    "test-code", "test-code#wrong-state", "test-code#EXPECTED_STATE#extra",
    "https://wrong.invalid/oauth/code/callback?code=test-code&state=EXPECTED_STATE",
    "http://platform.claude.com/oauth/code/callback?code=test-code&state=EXPECTED_STATE",
    "https://platform.claude.com:444/oauth/code/callback?code=test-code&state=EXPECTED_STATE",
    "https://platform.claude.com:0/oauth/code/callback?code=test-code&state=EXPECTED_STATE",
    "https://platform.claude.com/wrong?code=test-code&state=EXPECTED_STATE",
    "https://user@platform.claude.com/oauth/code/callback?code=test-code&state=EXPECTED_STATE",
    HOSTED_RETURN + "?code=test-code&state=EXPECTED_STATE#fragment",
    HOSTED_RETURN + "?code=test-code&state=EXPECTED_STATE&state=EXPECTED_STATE",
    HOSTED_RETURN + "?code=test-code&error=access_denied&state=EXPECTED_STATE",
    HOSTED_RETURN + "?error=access_denied&state=wrong",
    "code=test-code&code=another&state=EXPECTED_STATE",
])
def test_invalid_hosted_return_never_reaches_token_endpoint(monkeypatch, pasted):
    monkeypatch.setattr(claude, "CallbackListener", forbid_listener)

    def opener(request, timeout):
        raise AssertionError("Invalid return must not be exchanged")

    ctx = LoginContext(ui=UI(pasted), deadline=time.monotonic() + 30, provider="claude-code", opener=opener)
    with pytest.raises(AuthOperationError) as caught:
        claude.ClaudeFlow().login(ctx, claude.METHOD_BROWSER, {}, {})
    assert caught.value.reason == "invalid_login_state"
    assert "test-code" not in str(caught.value)
    assert "wrong.invalid" not in str(caught.value)


def test_validated_error_return_is_denied_not_echoed():
    with pytest.raises(LoginDenied) as caught:
        parse_manual_return(HOSTED_RETURN + "?error=access_denied&error_description=SECRET&state=S",
                            expected_state="S", allow_bare_code=False,
                            registered_path="/oauth/code/callback", registered_uri=HOSTED_RETURN)
    assert "SECRET" not in str(caught.value)


def test_cancel_while_pasting_prevents_exchange(monkeypatch):
    monkeypatch.setattr(claude, "CallbackListener", forbid_listener)
    cancel = threading.Event()

    class CancelUI(UI):
        def prompt(self, prompt):
            result = super().prompt(prompt)
            cancel.set()
            return result

    def opener(request, timeout):
        raise AssertionError("Cancelled attempt must not be exchanged")

    ctx = LoginContext(ui=CancelUI(), cancel=cancel, deadline=time.monotonic() + 30, opener=opener)
    with pytest.raises(LoginCancelled):
        claude.ClaudeFlow().login(ctx, claude.METHOD_BROWSER, {}, {})


def test_loopback_is_still_explicitly_available(monkeypatch):
    class Listener:
        def __init__(self, **kwargs):
            assert kwargs["port"] == 53692
            self.state = kwargs["expected_state"]

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(claude, "CallbackListener", Listener)
    monkeypatch.setattr(claude, "race_callback_and_manual", lambda ctx, listener, prompt: (
        CallbackReturn(code="test-code", state=listener.state), None,
    ))
    requests = []

    def opener(request, timeout):
        requests.append(json.loads(request.data))
        return Reply()

    ui = UI()
    ctx = LoginContext(ui=ui, deadline=time.monotonic() + 30, provider="claude-code", opener=opener)
    claude.ClaudeFlow().login(ctx, claude.METHOD_LOOPBACK, {}, {})
    assert ui.authorization.startswith("https://claude.ai/oauth/authorize?")
    assert requests[0]["redirect_uri"] == "http://localhost:53692/callback"
    assert claude.METHOD_LOOPBACK.availability == "unverified"


@pytest.mark.parametrize("post", [legacy._post_json, legacy._post_form])
def test_legacy_auth_http_identifies_lm15(monkeypatch, post):
    from lm15._version import __version__

    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return Reply()

    monkeypatch.setattr(legacy.urllib.request, "urlopen", urlopen)
    post("https://example.invalid/token", {"grant_type": "refresh_token", "refresh_token": "test-token"})
    assert len(requests) == 1
    assert requests[0].get_header("User-agent") == f"lm15/{__version__}"


def test_legacy_renewal_uses_reference_client_id(monkeypatch):
    requests = []

    def post(url, payload):
        assert url == "https://platform.claude.com/v1/oauth/token"
        requests.append(payload)
        return {"access_token": "test-access", "refresh_token": "test-refresh", "expires_in": 3600}

    monkeypatch.setattr(legacy, "_post_json", post)
    legacy.refresh_claude_code_credential("test-old-refresh")
    assert requests == [{"grant_type": "refresh_token", "client_id": EXPECTED_CLIENT_ID,
                         "refresh_token": "test-old-refresh"}]
