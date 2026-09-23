"""Auth failure diagnostics expose fixed categories, never raw provider text."""

import io
import json
import time
import traceback
import urllib.error
from email.message import Message
from urllib.parse import parse_qs, urlsplit

import pytest

from lm15.errors import AuthOperationError
from lm15.login import Auth, MemoryStore
from lm15.login.engine import LoginContext, http_json
from lm15.login.flows import claude

SECRET = "SECRET-SENTINEL-DO-NOT-PRINT"


class UI:
    def notify(self, notice):
        pass

    def prompt(self, prompt):
        raise AssertionError("No interactive prompts in tests")


def rejected_reply(raw, content_type, *, challenge=False):
    headers = Message()
    headers["Content-Type"] = content_type
    headers["Set-Cookie"] = SECRET
    headers["Location"] = "https://example.invalid/?code=" + SECRET
    headers["request-id"] = SECRET
    if challenge:
        headers["cf-mitigated"] = "challenge"
    calls = []
    bodies = []

    def opener(request, timeout):
        calls.append(request)
        body = io.BytesIO(raw)
        bodies.append(body)
        raise urllib.error.HTTPError(request.full_url, 403, SECRET, headers, body)

    return opener, calls, bodies


@pytest.mark.parametrize("raw,content_type,kind,code", [
    (json.dumps({"error": "invalid_client", "error_description": SECRET}).encode(),
     "application/json", "json", "invalid_client"),
    (json.dumps({"error": {"type": "invalid_grant", "message": SECRET}}).encode(),
     "application/json", "json", "invalid_grant"),
    (json.dumps({"error": SECRET, "code": SECRET, "access_token": SECRET}).encode(),
     "application/json", "json", None),
    (json.dumps({"error": [SECRET]}).encode(), "application/json", "json", None),
    (b'{"error":"invalid_client", "error":"access_denied"}', "application/json", "invalid_json", None),
    (b'{"error":"invalid_client", "number":NaN}', "application/json", "invalid_json", None),
    (("<html>" + SECRET + "</html>").encode(), "text/html; charset=utf-8", "html", None),
    (SECRET.encode(), "application/json", "invalid_json", None),
    (SECRET.encode(), SECRET, "text_or_binary", None),
    (b"", "application/json", "empty", None),
])
def test_response_categories_do_not_reflect_secrets(raw, content_type, kind, code):
    opener, calls, bodies = rejected_reply(raw, content_type)
    ctx = LoginContext(ui=UI(), deadline=time.monotonic() + 30, opener=opener)
    reply = http_json(ctx, "https://example.invalid/token", {"code": SECRET})
    assert reply.status == 403
    assert reply.response_format == kind
    assert reply.oauth_error == code
    assert SECRET not in repr(reply)
    assert SECRET not in reply.failure_summary()
    assert "HTTP 403" in reply.failure_summary()
    assert len(calls) == 1
    assert bodies[0].closed


@pytest.mark.parametrize("headers", [None, {"User-Agent": "provider-specific-client/1.0"}])
def test_auth_requests_identify_lm15_without_overwriting_provider_headers(headers):
    from lm15._version import __version__

    opener, calls, _ = rejected_reply(b'{"error":"invalid_grant"}', "application/json")
    ctx = LoginContext(ui=UI(), deadline=time.monotonic() + 30, opener=opener)
    reply = http_json(ctx, "https://example.invalid/token", {"code": SECRET}, headers=headers)
    expected = headers["User-Agent"] if headers else f"lm15/{__version__}"
    assert calls[0].get_header("User-agent") == expected
    assert reply.oauth_error == "invalid_grant"
    assert len(calls) == 1  # no header-guessing retry on failure


def test_html_does_not_imply_security_challenge():
    for challenge in (False, True):
        opener, _, _ = rejected_reply(b"<html>challenge</html>", "text/html", challenge=challenge)
        ctx = LoginContext(ui=UI(), deadline=time.monotonic() + 30, opener=opener)
        reply = http_json(ctx, "https://example.invalid/token", {})
        assert reply.security_challenge is challenge
        assert ("explicitly marked" in reply.failure_summary()) is challenge


def test_managed_error_retains_only_safe_diagnostics_and_old_connection(monkeypatch):
    class HostedUI(UI):
        def notify(self, notice):
            if getattr(notice, "type", None) == "auth_url":
                self.state = parse_qs(urlsplit(notice.url).query)["state"][0]

        def prompt(self, prompt):
            return SECRET + "#" + self.state

    def forbid_listener(**kwargs):
        raise AssertionError("Hosted login must not open a listener")

    monkeypatch.setattr(claude, "CallbackListener", forbid_listener)
    opener, calls, _ = rejected_reply(
        json.dumps({"error": "invalid_client", "error_description": SECRET,
                    "refresh_token": SECRET}).encode(), "application/json",
    )
    auth = Auth(MemoryStore(), opener=opener)
    # A pre-existing connection must survive a failed replacement.
    auth.store.mutate(lambda document: {"claude-code": {"type": "oauth", "access": SECRET,
                                                       "refresh": SECRET, "expires": 1}})
    previous = auth.status("claude-code").connection
    with pytest.raises(AuthOperationError) as caught:
        auth.login("claude-code", "browser", ui=HostedUI(), replace=previous.id, allow_unverified=True)
    error = caught.value
    assert error.status == 403
    assert error.provider_code == "invalid_client"
    assert error.stage == "exchange"
    assert "OAuth error=invalid_client" in str(error)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert SECRET not in rendered
    assert SECRET not in repr(error)
    assert len(calls) == 1  # a one-use authorization code is never replayed
    assert auth.status("claude-code").connection.id == previous.id
    assert auth.cancel_login("claude-code") == "complete"  # no stuck reservation
