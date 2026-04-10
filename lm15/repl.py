from __future__ import annotations

import atexit
import difflib
import os
import re
import sys
from typing import Callable

from .errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ULMError,
)


_DEBUG_REPL_ERRORS = bool(os.getenv("LM15_DEBUG"))
_ENABLED = False
_PREV_SYS_EXCEPTHOOK: Callable | None = None
_MODEL_ID_CACHE: tuple[str, ...] | None = None


def repl_debug(enabled: bool = True) -> None:
    global _DEBUG_REPL_ERRORS
    _DEBUG_REPL_ERRORS = enabled


def _is_interactive() -> bool:
    if hasattr(sys, "ps1") or bool(getattr(sys.flags, "interactive", 0)):
        return True
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _extract_model_name(message: str) -> str | None:
    patterns = (
        r"requested model '([^']+)' does not exist",
        r"requested model \"([^\"]+)\" does not exist",
        r"model '([^']+)' does not exist",
        r"model \"([^\"]+)\" does not exist",
    )
    for p in patterns:
        m = re.search(p, message, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _known_model_ids() -> tuple[str, ...]:
    from .capabilities import known_models

    models = known_models()
    if models:
        return models

    global _MODEL_ID_CACHE
    if _MODEL_ID_CACHE is not None:
        return _MODEL_ID_CACHE

    if os.getenv("LM15_REPL_FETCH_MODELS", "1") == "0":
        _MODEL_ID_CACHE = ()
        return _MODEL_ID_CACHE

    timeout = float(os.getenv("LM15_REPL_FETCH_MODELS_TIMEOUT", "1.0"))
    try:
        from .discovery import models

        specs = models(live=True, timeout=timeout)
        _MODEL_ID_CACHE = tuple(sorted({s.id for s in specs if s.id}))
    except Exception:
        _MODEL_ID_CACHE = ()
    return _MODEL_ID_CACHE


def _suggest_models(model_name: str) -> list[str]:
    models = _known_model_ids()
    if not models:
        return []
    return difflib.get_close_matches(model_name, models, n=3, cutoff=0.45)


def format_lm15_error(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc) or name

    lines = [f"LM15 {name}", msg]

    _has_guidance = "To fix" in msg or "To fix" in str(exc)

    if isinstance(exc, AuthError) and not _has_guidance:
        lines.append("")
        lines.append("  To fix, do one of:")
        lines.append("    1. Set the API key in your environment:")
        lines.append("       export OPENAI_API_KEY=sk-...  (or ANTHROPIC_API_KEY / GEMINI_API_KEY)")
        lines.append("    2. Add it to a .env file and call lm15.configure(env='.env')")
        lines.append("    3. Pass it directly: lm15.call(..., api_key='sk-...')")
    elif isinstance(exc, BillingError):
        lines.append("")
        lines.append("  Your provider returned a billing/quota error (HTTP 402).")
        lines.append("  Check your account at:")
        lines.append("    OpenAI:    platform.openai.com/account/billing")
        lines.append("    Anthropic: console.anthropic.com/settings/plans")
        lines.append("    Gemini:    aistudio.google.com")
    elif isinstance(exc, RateLimitError) and not _has_guidance:
        lines.append("")
        lines.append("  The provider rate-limited this request (HTTP 429).")
        lines.append("  To fix:")
        lines.append("    - Wait a moment and retry")
        lines.append("    - Use retries= on model objects: lm15.model(..., retries=3)")
        lines.append("    - Reduce request rate or upgrade your API plan")
    elif isinstance(exc, ContextLengthError) and not _has_guidance:
        lines.append("")
        lines.append("  The input exceeds this model's context window.")
        lines.append("  To fix:")
        lines.append("    - Reduce the prompt or system prompt length")
        lines.append("    - Clear conversation history: model.history.clear()")
        lines.append("    - Use a model with a larger context window")
        lines.append("    - Lower max_tokens to leave more room for input")
    elif isinstance(exc, TimeoutError):
        lines.append("")
        lines.append("  The request timed out before the provider responded.")
        lines.append("  To fix:")
        lines.append("    - Reduce prompt size (shorter prompts = faster responses)")
        lines.append("    - Use retries= on model objects: lm15.model(..., retries=2)")
        lines.append("    - Check your network connection")
    elif isinstance(exc, InvalidRequestError):
        model_name = _extract_model_name(msg)
        if model_name:
            suggestions = _suggest_models(model_name)
            if suggestions:
                lines.append("")
                lines.append("  Did you mean: " + ", ".join(suggestions))
                lines.append("  List available models: lm15.models()")
            else:
                lines.append("")
                lines.append("  Model not found. List available models with lm15.models()")
                lines.append("  For custom/fine-tuned models, specify the provider:")
                lines.append(f"    lm15.call('{model_name}', ..., provider='openai')")

    if isinstance(exc, ServerError):
        lines.append("")
        lines.append("  The provider returned a server error (HTTP 5xx).")
        lines.append("  This is usually transient. To handle automatically:")
        lines.append("    - Use retries= on model objects: lm15.model(..., retries=3)")
        lines.append("    - Or switch providers: lm15.call('gemini-2.5-flash', ...)")


    return "\n".join(lines)


def _sys_excepthook(exc_type, exc, tb):
    if _DEBUG_REPL_ERRORS or not isinstance(exc, ULMError):
        assert _PREV_SYS_EXCEPTHOOK is not None
        return _PREV_SYS_EXCEPTHOOK(exc_type, exc, tb)
    print(format_lm15_error(exc), file=sys.stderr)


def _install_sys_hook() -> None:
    global _PREV_SYS_EXCEPTHOOK
    if _PREV_SYS_EXCEPTHOOK is not None:
        return
    _PREV_SYS_EXCEPTHOOK = sys.excepthook
    sys.excepthook = _sys_excepthook


def _uninstall_sys_hook() -> None:
    global _PREV_SYS_EXCEPTHOOK
    if _PREV_SYS_EXCEPTHOOK is None:
        return
    sys.excepthook = _PREV_SYS_EXCEPTHOOK
    _PREV_SYS_EXCEPTHOOK = None


def _install_ipython_hook() -> None:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return

    ip = get_ipython()
    if ip is None:
        return

    def _custom(shell, etype, evalue, tb, tb_offset=None):
        if _DEBUG_REPL_ERRORS or not isinstance(evalue, ULMError):
            return shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        print(format_lm15_error(evalue), file=sys.stderr)
        return []

    try:
        ip.set_custom_exc((ULMError,), _custom)
    except Exception:
        return


def enable_repl_errors() -> None:
    global _ENABLED
    if _ENABLED:
        return
    if os.getenv("LM15_REPL_ERRORS", "1") == "0":
        return
    if not _is_interactive():
        return
    _install_sys_hook()
    _install_ipython_hook()
    _ENABLED = True
    atexit.register(_uninstall_sys_hook)
