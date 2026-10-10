"""MAP-18: a provider's "this key is not valid" is AuthError, whatever the
status; AUTH-1/AUTH-5 (amended 2026-10-10): a key put in ``credential=``
is refused without being repeated."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import lm15
from lm15 import AnthropicLM, AsyncAnthropicLM, AuthError, GeminiLM, InvalidRequestError, NotConfiguredError, XaiLM
from lm15.errors import AUTH_FAILED_FORMS, google_error_reasons, is_pinned_auth_failure

SENTINEL = "SECRET-SENTINEL-DO-NOT-PRINT"
CONTRACT = Path(__file__).resolve().parents[2] / "lm15-contract"

GEMINI_BAD_KEY = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID",
             "domain": "googleapis.com", "metadata": {"service": "generativelanguage.googleapis.com"}},
            {"@type": "type.googleapis.com/google.rpc.LocalizedMessage", "locale": "en-US",
             "message": "API key not valid. Please pass a valid API key."},
        ],
    }
}


def test_table_matches_the_contract() -> None:
    spec = CONTRACT / "spec" / "auth-failed.json"
    if not spec.is_file():
        pytest.skip("lm15-contract checkout not beside this repository")
    forms = json.loads(spec.read_text(encoding="utf-8"))["forms"]
    pinned = [{k: v for k, v in f.items() if k not in ("providers", "evidence")} for f in forms]
    assert list(AUTH_FAILED_FORMS) == pinned


def test_gemini_bad_key_is_auth() -> None:
    error = GeminiLM(api_key="k").normalize_error(400, json.dumps(GEMINI_BAD_KEY))
    assert type(error) is AuthError
    assert (error.status, error.provider_code, error.provider) == (400, "INVALID_ARGUMENT", "gemini")


def test_gemini_other_invalid_argument_stays_invalid_request() -> None:
    body = json.loads(json.dumps(GEMINI_BAD_KEY))
    body["error"]["details"][0]["reason"] = "FIELD_INVALID"
    body["error"]["message"] = "API key not valid. Please pass a valid API key."  # the text alone never decides
    error = GeminiLM(api_key="k").normalize_error(400, json.dumps(body))
    assert type(error) is InvalidRequestError


def test_xai_bad_key_is_auth() -> None:
    body = {"code": "invalid-argument",
            "error": "Incorrect API key provided. You can obtain an API key from https://console.x.ai."}
    error = XaiLM(api_key="k").normalize_error(400, json.dumps(body))
    assert type(error) is AuthError
    assert error.provider_code == "invalid-argument"


def test_xai_model_not_found_is_unchanged() -> None:
    body = {"code": "invalid-argument", "error": "Model not found: grok-nonexistent"}
    assert type(XaiLM(api_key="k").normalize_error(400, json.dumps(body))) is lm15.UnsupportedModelError


def test_reason_only_from_error_info() -> None:
    err = {"details": [{"@type": "type.googleapis.com/google.rpc.Help", "reason": "API_KEY_INVALID"}]}
    assert google_error_reasons(err) == ()
    assert not is_pinned_auth_failure("INVALID_ARGUMENT", "API key not valid.", google_error_reasons(err))


@pytest.mark.parametrize("make", [
    lambda: AnthropicLM(credential=SENTINEL),
    lambda: AsyncAnthropicLM(credential=SENTINEL),
    lambda: GeminiLM(credential=SENTINEL),
    lambda: lm15.LMRouter(lm15.RouterConfig(credentials={"vertex": SENTINEL})),
])
def test_key_given_as_credential_is_never_shown(make) -> None:
    with pytest.raises(NotConfiguredError) as raised:
        make()
    assert SENTINEL not in str(raised.value)
    assert SENTINEL not in repr(raised.value)
    assert "api_key" in str(raised.value)


def test_a_name_on_a_key_door_is_still_named() -> None:
    with pytest.raises(NotConfiguredError, match="credential='platform'"):
        AnthropicLM(credential="platform")
