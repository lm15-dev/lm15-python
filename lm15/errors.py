from __future__ import annotations


class ULMError(Exception):
    pass


class TransportError(ULMError):
    pass


class ProviderError(ULMError):
    pass


class AuthError(ProviderError):
    """Authentication failed — invalid, expired, or missing API key."""

    def __init__(self, message: str = "") -> None:
        guidance = (
            "\n\n"
            "  To fix, do one of:\n"
            "    1. Check that your API key is correct and not expired\n"
            "    2. Set it in your environment: export OPENAI_API_KEY=sk-...\n"
            "    3. Pass it directly: lm15.call(..., api_key='sk-...')\n"
            "    4. Add it to a .env file and call lm15.configure(env='.env')\n"
        )
        if guidance.strip() not in message:
            message = message.rstrip() + guidance
        super().__init__(message)


class RateLimitError(ProviderError):
    """Rate limited by the provider (HTTP 429)."""

    def __init__(self, message: str = "") -> None:
        guidance = (
            "\n\n"
            "  To fix:\n"
            "    - Wait a moment and retry\n"
            "    - Use retries= on model objects: lm15.model(..., retries=3)\n"
            "    - Reduce request rate or upgrade your API plan\n"
        )
        if guidance.strip() not in message:
            message = message.rstrip() + guidance
        super().__init__(message)


class BillingError(ProviderError):
    """402 — billing or payment issue."""


class TimeoutError(ProviderError):
    pass


class InvalidRequestError(ProviderError):
    pass


class ContextLengthError(InvalidRequestError):
    """The input exceeds the model's context window."""

    def __init__(self, message: str = "") -> None:
        guidance = (
            "\n\n"
            "  To fix:\n"
            "    - Reduce the prompt or system prompt length\n"
            "    - Clear conversation history: model.history.clear()\n"
            "    - Use a model with a larger context window\n"
            "    - Lower max_tokens to leave more room for input\n"
        )
        if guidance.strip() not in message:
            message = message.rstrip() + guidance
        super().__init__(message)


class ServerError(ProviderError):
    pass


class UnsupportedModelError(ProviderError):
    pass


class UnsupportedFeatureError(ProviderError):
    pass


class NotConfiguredError(ProviderError):
    """No API key found for a provider."""


def map_http_error(status: int, message: str) -> ProviderError:
    """Map an HTTP status + extracted message to a typed ProviderError.

    Adapters are responsible for extracting the human-readable message
    from the provider's error body in their ``normalize_error`` override.
    This function only maps status codes to error classes.
    """
    if status in (401, 403):
        return AuthError(message)
    if status == 402:
        return BillingError(message)
    if status in (408, 504):
        return TimeoutError(message)
    if status == 429:
        return RateLimitError(message)
    if status in (400, 404, 409, 413, 422):
        return InvalidRequestError(message)
    if 500 <= status <= 599:
        return ServerError(message)
    return ProviderError(message)


def canonical_error_code(error: type[ProviderError] | ProviderError) -> str:
    """Return canonical lm15 error code for an error class/instance.

    Canonical codes are provider-agnostic and stable across adapters.
    """
    cls = error if isinstance(error, type) else type(error)

    if issubclass(cls, ContextLengthError):
        return "context_length"
    if issubclass(cls, AuthError):
        return "auth"
    if issubclass(cls, BillingError):
        return "billing"
    if issubclass(cls, RateLimitError):
        return "rate_limit"
    if issubclass(cls, InvalidRequestError):
        return "invalid_request"
    if issubclass(cls, TimeoutError):
        return "timeout"
    if issubclass(cls, ServerError):
        return "server"
    return "provider"


def error_class_for_canonical_code(code: str) -> type[ProviderError]:
    """Return ProviderError subclass for canonical lm15 error code."""
    return {
        "auth": AuthError,
        "billing": BillingError,
        "rate_limit": RateLimitError,
        "invalid_request": InvalidRequestError,
        "context_length": ContextLengthError,
        "timeout": TimeoutError,
        "server": ServerError,
        "provider": ProviderError,
    }.get(code, ProviderError)
