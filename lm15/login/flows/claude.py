"""
lm15.login.flows.claude — Claude subscription login owned by LM15.

Authorization-code + PKCE (S256) against ``claude.ai/oauth/authorize``,
loopback return on the registered redirect URI, manual paste as the
supported alternative when the browser is elsewhere.  The token endpoint
and client registration are the ones Claude Code uses; Pi 0.87.0 is the
implementation reference (``packages/ai/src/auth/oauth/anthropic.ts``).

**Availability: unverified** (AUTH-13.5, R1).  The code is complete, but
LM15 has no live receipt showing that a login made through this
registration is permitted for third-party clients, gives the same account
access as Claude Code, and bills the way the user expects.  Until that
receipt exists the default picker offers the proven path instead — the
user's existing Claude Code login as an external source
(``lm15.login.flows.external``) — and this method needs explicit opt-in.
"""

from __future__ import annotations

import secrets
from typing import Any

from ...authkit import generate_pkce
from ...credentials import BearerToken
from ..engine import (
    CallbackListener,
    LoginContext,
    LoginDenied,
    ManualCodePrompt,
    http_json,
    parse_manual_return,
    race_callback_and_manual,
)
from ..types import AuthUrlNotice, InfoNotice, LoginMethod, ProgressNotice, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth, oauth_material

CLIENT_ID = "9d1c250a-e61b-44d5-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"  # the registered value; bind is 127.0.0.1
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

METHOD_BROWSER = LoginMethod(
    id="browser", label="Sign in with your Claude account (Pro/Max)", kind="account", flow="authorization_code",
    availability="unverified",
    reason="no live receipt yet that this registration is permitted for LM15 and bills as a subscription",
    delivery=("loopback", "manual"), subscription=True,
    billing_note="Anthropic's current terms route third-party usage of a subscription through extra usage; "
                 "verify on your account before relying on it.",
)

DESCRIPTOR = ProviderDescriptor(
    id="claude-code", label="Claude (subscription)", service="Anthropic", routes=("claude-code",),
    methods=(METHOD_BROWSER,),
    docs_url="https://docs.claude.com",
)


def _tokens(body: dict[str, Any], *, now_ms: int) -> Material:
    access, refresh, expires_in = body.get("access_token"), body.get("refresh_token"), body.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise LoginDenied("Claude token response is missing required fields")
    lifetime = float(expires_in) if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0 else None
    return oauth_material(access=access, refresh=refresh, expires_in_s=lifetime, now_ms=now_ms)


class ClaudeFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        pkce = generate_pkce()
        state = secrets.token_urlsafe(32)
        listener: CallbackListener | None
        try:
            listener = CallbackListener(path=CALLBACK_PATH, expected_state=state, port=CALLBACK_PORT,
                                        redirect_host="localhost")
            listener.start()
        except Exception as exc:  # port busy: manual return is the supported alternative
            listener = None
            ctx.notify(InfoNotice(f"Could not listen on port {CALLBACK_PORT} ({type(exc).__name__}); "
                                  "paste the redirect URL when the browser finishes."))
        try:
            query = {
                "code": "true", "client_id": CLIENT_ID, "response_type": "code", "redirect_uri": REDIRECT_URI,
                "scope": SCOPES, "code_challenge": pkce.challenge, "code_challenge_method": "S256", "state": state,
            }
            import urllib.parse

            url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"
            ctx.notify(AuthUrlNotice(url=url, instructions=(
                "Sign in to Claude in your browser. If the browser is on another machine, paste the final "
                "redirect URL (or the code#state it shows) back here.")))
            prompt = ManualCodePrompt(field_id="return", label="Paste the redirect URL or code here (or wait for the browser)",
                                      accepted="the full redirect URL, or code#state")
            returned, pasted = race_callback_and_manual(ctx, listener, prompt)
            if returned is None:
                returned = parse_manual_return(pasted or "", expected_state=state, allow_bare_code=False,
                                               registered_path=CALLBACK_PATH)
            ctx.notify(ProgressNotice(stage="exchange", message="Exchanging the authorization code…"))
            reply = http_json(ctx, TOKEN_URL, {
                "grant_type": "authorization_code", "client_id": CLIENT_ID, "code": returned.code,
                "state": returned.state or state, "redirect_uri": REDIRECT_URI, "code_verifier": pkce.verifier,
            })
            if not reply.ok:
                raise LoginDenied(f"Claude rejected the authorization code (HTTP {reply.status})")
            return LoginResult(material=_tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000)),
                               label="Claude subscription", renewal="refresh_token")
        finally:
            if listener is not None:
                listener.stop()

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        refresh = material.get("refresh")
        if not isinstance(refresh, str) or not refresh:
            raise LoginDenied("Claude credential has no refresh token")
        reply = http_json(ctx, TOKEN_URL, {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh})
        if not reply.ok:
            raise LoginDenied(f"Claude rejected the refresh token (HTTP {reply.status})")
        return LoginResult(material=_tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000)),
                           label="Claude subscription", renewal="refresh_token")

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=BearerToken(material["access"]))
