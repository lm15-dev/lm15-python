"""
lm15.auth — Local subscription credential helpers.

These helpers intentionally do not read ordinary provider API keys from the
environment.  They only support explicit local developer credentials created
by provider CLIs: Claude Code (``~/.claude/.credentials.json``) and the OpenAI
Codex CLI (``~/.codex/auth.json``).

Failure behavior is typed and helpful, never a raw JSON traceback:

- missing / unreadable / malformed credential files raise
  :class:`lm15.errors.NotConfiguredError` telling the user which CLI login to
  run;
- an expired token that cannot be refreshed (no refresh token, or the refresh
  call fails) raises :class:`lm15.errors.AuthError` with the same re-login
  guidance.

Token material never appears in error messages or reprs.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AuthError, NotConfiguredError

CLAUDE_CODE_CREDENTIALS_PATH = Path("~/.claude/.credentials.json").expanduser()
CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d5-88ed-5944d1962f5e"
CLAUDE_CODE_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLAUDE_CODE_LOGIN_HINT = "Log in again: run `claude` and use /login (Claude subscription auth)"

CODEX_CLI_AUTH_PATH = Path("~/.codex/auth.json").expanduser()
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"
OPENAI_CODEX_LOGIN_HINT = "Log in again: run `codex login` (ChatGPT subscription auth)"

_REFRESH_SKEW_MS = 5 * 60 * 1000


@dataclass(frozen=True, slots=True)
class LocalOAuthCredential:
    """A locally stored OAuth credential.  Token fields are repr-suppressed."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: int | None = None
    account_id: str | None = None

    @property
    def expired(self) -> bool:
        return isinstance(self.expires_at, int) and int(time.time() * 1000) >= self.expires_at


def _not_configured(provider: str, message: str, hint: str) -> NotConfiguredError:
    return NotConfiguredError(message, provider=provider, credential_hint=hint)


def _read_json_file(path: Path, *, provider: str, hint: str) -> dict[str, Any]:
    """Read a credential file, raising typed errors instead of crashing."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise _not_configured(provider, f"No credentials file at {path}.", hint) from exc
    except OSError as exc:
        raise _not_configured(provider, f"Could not read credentials file at {path}: {exc}", hint) from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise _not_configured(provider, f"Credentials file at {path} is not valid JSON.", hint) from exc
    if not isinstance(data, dict):
        raise _not_configured(provider, f"Credentials file at {path} has an unexpected shape.", hint)
    return data


def _read_json_file_or_none(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _base64url_json(segment: str) -> dict[str, Any]:
    padding = "=" * (-len(segment) % 4)
    decoded = base64.urlsafe_b64decode(segment + padding)
    data = json.loads(decoded.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT")
    return _base64url_json(parts[1])


def jwt_expires_at_ms(token: str) -> int | None:
    try:
        exp = decode_jwt_payload(token).get("exp")
    except Exception:
        return None
    if isinstance(exp, (int, float)):
        return int(exp * 1000) - _REFRESH_SKEW_MS
    return None


def extract_chatgpt_account_id(token: str) -> str | None:
    try:
        payload = decode_jwt_payload(token)
    except Exception:
        return None
    auth_claim = payload.get(OPENAI_CODEX_JWT_CLAIM_PATH)
    if isinstance(auth_claim, dict):
        account_id = auth_claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - provider token endpoint
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - provider token endpoint
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _coerce_path(path: str | os.PathLike[str] | None, default: Path) -> Path:
    return Path(path).expanduser() if path is not None else default


# ─── Claude Code (~/.claude/.credentials.json) ───────────────────────


def load_claude_code_credential(
    credentials_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential:
    """Load the Claude Code OAuth credential, raising typed errors."""
    path = _coerce_path(credentials_path, CLAUDE_CODE_CREDENTIALS_PATH)
    data = _read_json_file(path, provider="claude-code", hint=CLAUDE_CODE_LOGIN_HINT)
    raw = data.get("claudeAiOauth")
    if not isinstance(raw, dict):
        raise _not_configured(
            "claude-code",
            f"Credentials file at {path} has no claudeAiOauth section.",
            CLAUDE_CODE_LOGIN_HINT,
        )
    access = raw.get("accessToken")
    if not isinstance(access, str) or not access:
        raise _not_configured(
            "claude-code",
            f"Credentials file at {path} has no access token.",
            CLAUDE_CODE_LOGIN_HINT,
        )
    refresh = raw.get("refreshToken")
    expires = raw.get("expiresAt")
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        expires_at=int(expires) if isinstance(expires, (int, float)) else None,
    )


def read_claude_code_credential(
    credentials_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential | None:
    """Optional-style loader: None when no usable credential exists."""
    try:
        return load_claude_code_credential(credentials_path)
    except NotConfiguredError:
        return None


def refresh_claude_code_credential(refresh_token: str) -> LocalOAuthCredential:
    payload = _post_json(
        CLAUDE_CODE_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": CLAUDE_CODE_CLIENT_ID,
            "refresh_token": refresh_token,
        },
    )
    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not isinstance(refresh, str) or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Claude Code token refresh response is missing required fields")
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh,
        expires_at=int(time.time() * 1000 + expires_in * 1000 - _REFRESH_SKEW_MS),
    )


def write_claude_code_credential(
    credential: LocalOAuthCredential,
    credentials_path: str | os.PathLike[str] | None = None,
) -> None:
    path = _coerce_path(credentials_path, CLAUDE_CODE_CREDENTIALS_PATH)
    data = _read_json_file_or_none(path) or {}
    raw = data.get("claudeAiOauth")
    current = raw if isinstance(raw, dict) else {}
    current["accessToken"] = credential.access_token
    if credential.refresh_token:
        current["refreshToken"] = credential.refresh_token
    if credential.expires_at is not None:
        current["expiresAt"] = credential.expires_at
    data["claudeAiOauth"] = current
    _write_private_json(path, data)


def get_claude_code_access_token(
    credentials_path: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = True,
) -> str:
    """Return a usable Claude Code access token, refreshing it if expired.

    Raises NotConfiguredError when no credential exists, AuthError when the
    credential is expired and cannot be refreshed.
    """
    credential = load_claude_code_credential(credentials_path)
    if not credential.expired:
        return credential.access_token
    if not refresh or not credential.refresh_token:
        raise AuthError(
            "Claude Code OAuth token is expired and no refresh token is available.",
            provider="claude-code",
            credential_hint=CLAUDE_CODE_LOGIN_HINT,
        )
    try:
        refreshed = refresh_claude_code_credential(credential.refresh_token)
    except Exception as exc:
        raise AuthError(
            "Claude Code OAuth token is expired and the refresh attempt failed.",
            provider="claude-code",
            credential_hint=CLAUDE_CODE_LOGIN_HINT,
        ) from exc
    write_claude_code_credential(refreshed, credentials_path)
    return refreshed.access_token


# ─── OpenAI Codex CLI (~/.codex/auth.json) ───────────────────────────


def load_codex_cli_credential(
    auth_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential:
    """Load the Codex CLI OAuth credential, raising typed errors."""
    path = _coerce_path(auth_path, CODEX_CLI_AUTH_PATH)
    data = _read_json_file(path, provider="openai-codex", hint=OPENAI_CODEX_LOGIN_HINT)
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise _not_configured(
            "openai-codex",
            f"Credentials file at {path} has no tokens section.",
            OPENAI_CODEX_LOGIN_HINT,
        )
    access = tokens.get("access_token")
    if not isinstance(access, str) or not access:
        raise _not_configured(
            "openai-codex",
            f"Credentials file at {path} has no access token.",
            OPENAI_CODEX_LOGIN_HINT,
        )
    refresh = tokens.get("refresh_token")
    account_id = tokens.get("account_id") or extract_chatgpt_account_id(access)
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        expires_at=jwt_expires_at_ms(access),
        account_id=account_id if isinstance(account_id, str) and account_id else None,
    )


def read_codex_cli_credential(
    auth_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential | None:
    """Optional-style loader: None when no usable credential exists."""
    try:
        return load_codex_cli_credential(auth_path)
    except NotConfiguredError:
        return None


def refresh_codex_cli_credential(refresh_token: str) -> LocalOAuthCredential:
    payload = _post_form(
        OPENAI_CODEX_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OPENAI_CODEX_CLIENT_ID,
        },
    )
    access = payload.get("access_token")
    refresh = payload.get("refresh_token") or refresh_token
    if not isinstance(access, str) or not isinstance(refresh, str):
        raise RuntimeError("Codex token refresh response is missing required fields")
    account_id = extract_chatgpt_account_id(access)
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh,
        expires_at=jwt_expires_at_ms(access),
        account_id=account_id,
    )


def write_codex_cli_credential(
    credential: LocalOAuthCredential,
    auth_path: str | os.PathLike[str] | None = None,
    *,
    id_token: str | None = None,
) -> None:
    path = _coerce_path(auth_path, CODEX_CLI_AUTH_PATH)
    data = _read_json_file_or_none(path) or {}
    tokens = data.get("tokens")
    current = tokens if isinstance(tokens, dict) else {}
    current["access_token"] = credential.access_token
    if credential.refresh_token:
        current["refresh_token"] = credential.refresh_token
    if credential.account_id:
        current["account_id"] = credential.account_id
    if id_token is not None:
        current["id_token"] = id_token
    data["tokens"] = current
    data.setdefault("auth_mode", "chatgpt")
    data["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_private_json(path, data)


def get_codex_cli_access_token(
    auth_path: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = True,
) -> LocalOAuthCredential:
    """Return a usable Codex CLI credential, refreshing it if expired.

    Raises NotConfiguredError when no credential exists, AuthError when the
    credential is expired and cannot be refreshed.
    """
    credential = load_codex_cli_credential(auth_path)
    if not credential.expired:
        return credential
    if not refresh or not credential.refresh_token:
        raise AuthError(
            "Codex CLI OAuth token is expired and no refresh token is available.",
            provider="openai-codex",
            credential_hint=OPENAI_CODEX_LOGIN_HINT,
        )
    try:
        refreshed = refresh_codex_cli_credential(credential.refresh_token)
    except Exception as exc:
        raise AuthError(
            "Codex CLI OAuth token is expired and the refresh attempt failed.",
            provider="openai-codex",
            credential_hint=OPENAI_CODEX_LOGIN_HINT,
        ) from exc
    original = _read_json_file_or_none(_coerce_path(auth_path, CODEX_CLI_AUTH_PATH)) or {}
    tokens = original.get("tokens") if isinstance(original.get("tokens"), dict) else {}
    id_token = tokens.get("id_token") if isinstance(tokens, dict) and isinstance(tokens.get("id_token"), str) else None
    write_codex_cli_credential(refreshed, auth_path, id_token=id_token)
    return refreshed
