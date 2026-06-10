from __future__ import annotations

import json

from lm15.errors import (
    AuthError,
    BillingError,
    CapabilityError,
    ConfigurationError,
    ContextLengthError,
    InvalidRequestError,
    NotConfiguredError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
    TransportError,
    UnsupportedFeatureError,
    UnsupportedModelError,
    canonical_error_code,
    error_class_for_code,
    map_http_error,
)
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM


def test_canonical_error_codes_cover_public_taxonomy() -> None:
    cases = {
        ContextLengthError: "context_length",
        UnsupportedModelError: "unsupported_model",
        AuthError: "auth",
        BillingError: "billing",
        RateLimitError: "rate_limit",
        InvalidRequestError: "invalid_request",
        TimeoutError: "timeout",
        ServerError: "server",
        UnsupportedFeatureError: "unsupported_feature",
        NotConfiguredError: "not_configured",
        TransportError: "transport",
        ProviderError: "provider",
    }

    for cls, code in cases.items():
        assert canonical_error_code(cls) == code
        assert error_class_for_code(code) is cls

    assert canonical_error_code(ConfigurationError) == "not_configured"
    assert canonical_error_code(CapabilityError) == "unsupported_feature"


def test_map_http_error_keeps_structured_metadata() -> None:
    err = map_http_error(
        429,
        "slow down",
        provider="openai",
        provider_code="rate_limit_exceeded",
        request_id="req_123",
    )

    assert isinstance(err, RateLimitError)
    assert err.code == "rate_limit"
    assert err.provider == "openai"
    assert err.provider_code == "rate_limit_exceeded"
    assert err.status == 429
    assert err.request_id == "req_123"


def test_auth_guidance_is_provider_specific_not_openai_global() -> None:
    lm = AnthropicLM(api_key="test")
    try:
        err = lm.normalize_error(
            401,
            json.dumps({"error": {"type": "authentication_error", "message": "invalid x-api-key"}}),
        )
    finally:
        lm.close()

    text = str(err)
    assert isinstance(err, AuthError)
    assert err.provider == "anthropic"
    assert err.provider_code == "authentication_error"
    assert "ANTHROPIC_API_KEY" in text
    assert "OPENAI_API_KEY" not in text
    assert "lm15.call" not in text
    assert "lm15.configure" not in text


def test_rate_limit_guidance_does_not_reference_missing_top_level_api() -> None:
    text = str(RateLimitError("rate limited"))

    assert "lm15.model" not in text
    # lm15 has no retries option anywhere; guidance must not imply one.
    assert "Enable retries" not in text
    assert "application layer" in text


def test_openai_model_not_found_maps_to_unsupported_model() -> None:
    lm = OpenAILM(api_key="test")
    try:
        err = lm.normalize_error(
            404,
            json.dumps(
                {
                    "error": {
                        "message": "The model `gpt-missing` does not exist",
                        "code": "model_not_found",
                        "type": "invalid_request_error",
                    }
                }
            ),
        )
    finally:
        lm.close()

    assert isinstance(err, UnsupportedModelError)
    assert isinstance(err, InvalidRequestError)
    assert err.code == "unsupported_model"
    assert err.provider == "openai"
    assert err.provider_code == "model_not_found"
    assert err.status == 404


def test_anthropic_model_not_found_maps_to_unsupported_model() -> None:
    lm = AnthropicLM(api_key="test")
    try:
        err = lm.normalize_error(
            404,
            json.dumps(
                {
                    "error": {
                        "type": "not_found_error",
                        "message": "model claude-missing not found",
                    },
                    "request_id": "req_ant",
                }
            ),
        )
    finally:
        lm.close()

    assert isinstance(err, UnsupportedModelError)
    assert err.provider == "anthropic"
    assert err.provider_code == "not_found_error"
    assert err.request_id == "req_ant"


def test_gemini_model_not_found_maps_to_unsupported_model() -> None:
    lm = GeminiLM(api_key="test")
    try:
        err = lm.normalize_error(
            404,
            json.dumps(
                {
                    "error": {
                        "status": "NOT_FOUND",
                        "message": "models/gemini-missing is not found for API version v1beta",
                    }
                }
            ),
        )
    finally:
        lm.close()

    assert isinstance(err, UnsupportedModelError)
    assert err.provider == "gemini"
    assert err.provider_code == "NOT_FOUND"
    assert err.status == 404
