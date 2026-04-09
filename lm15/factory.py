from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .capabilities import hydrate_with_specs
from .client import UniversalLM
from .features import ProviderManifest
from .model_catalog import fetch_models_dev
from .plugins import load_plugins
from .providers.anthropic import AnthropicAdapter
from .providers.gemini import GeminiAdapter
from .providers.openai import OpenAIAdapter
from .transports.base import TransportPolicy
from .transports.pycurl_transport import PyCurlTransport
from .transports.urllib_transport import UrlLibTransport


# Ordered list of core adapter classes.  Each must have a ClassVar `manifest`
# with populated `env_keys`.  build_default iterates this to discover keys.
_CORE_ADAPTERS: Sequence[type] = (OpenAIAdapter, AnthropicAdapter, GeminiAdapter)


def _build_env_key_map(adapters: Sequence[type] = _CORE_ADAPTERS) -> dict[str, str]:
    """Derive {ENV_VAR: provider_name} from adapter manifests."""
    out: dict[str, str] = {}
    for cls in adapters:
        m: ProviderManifest = cls.manifest  # type: ignore[attr-defined]
        for var in m.env_keys:
            out.setdefault(var, m.provider)
    return out


def _parse_env_file(
    path: str | Path,
    env_key_map: dict[str, str],
) -> dict[str, str]:
    """Extract provider API keys from a key-value file.

    Handles ``.env``, ``.bashrc``, ``.zshrc`` and similar formats:
    ``KEY=VALUE``, ``export KEY=VALUE``, ``export KEY="VALUE"``.
    Returns ``{provider: key}`` for recognised variable names only.
    """
    result: dict[str, str] = {}
    try:
        text = Path(path).expanduser().read_text()
    except (OSError, ValueError):
        return result

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip leading 'export ' (bash/zsh)
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        provider = env_key_map.get(key)
        if provider and value:
            result[provider] = value
    return result


def _push_env_file_to_environ(path: str | Path) -> None:
    """Set all KEY=VALUE pairs from *path* into ``os.environ``.

    Only sets variables that are not already present, so explicit
    environment variables still win.  This lets third-party plugins
    discover keys from the user-specified ``env=`` file.
    """
    try:
        text = Path(path).expanduser().read_text()
    except (OSError, ValueError):
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


def providers() -> dict[str, tuple[str, ...]]:
    """Return ``{provider_name: (ENV_VAR, ...)}`` for all core adapters.

    Useful for discovering valid ``api_key`` dict keys and which
    environment variables each provider reads.

    >>> import lm15
    >>> lm15.providers()
    {'openai': ('OPENAI_API_KEY',), 'anthropic': ('ANTHROPIC_API_KEY',), ...}
    """
    return {
        cls.manifest.provider: cls.manifest.env_keys  # type: ignore[attr-defined]
        for cls in _CORE_ADAPTERS
    }


def _resolve_api_keys(
    api_key: str | dict[str, str] | None,
    provider_hint: str | None,
    known_providers: Sequence[str],
) -> dict[str, str]:
    """Return a dict of {provider: key} from the api_key argument.

    - ``None``  → empty (fall back to env file / os.environ).
    - ``str``   → assigned to *provider_hint*, or broadcast to all
                  *known_providers* if no hint.
    - ``dict``  → used directly, keyed by provider name.
    """
    if api_key is None:
        return {}
    if isinstance(api_key, dict):
        return api_key
    if provider_hint:
        return {provider_hint: api_key}
    return {p: api_key for p in known_providers}


def build_default(
    use_pycurl: bool = True,
    policy: TransportPolicy | None = None,
    hydrate_models_dev: bool = False,
    discover_plugins: bool = True,
    api_key: str | dict[str, str] | None = None,
    provider_hint: str | None = None,
    env: str | Path | None = None,
) -> UniversalLM:
    policy = policy or TransportPolicy()
    transport = UrlLibTransport(policy=policy)
    if use_pycurl:
        try:
            import pycurl  # noqa: F401

            transport = PyCurlTransport(policy=policy)
        except Exception:
            transport = UrlLibTransport(policy=policy)

    env_key_map = _build_env_key_map()
    known_providers = list(dict.fromkeys(env_key_map.values()))  # unique, order-preserving

    # Priority: api_key > env file > os.environ
    explicit = _resolve_api_keys(api_key, provider_hint, known_providers)
    from_file: dict[str, str] = {}
    if env:
        from_file = _parse_env_file(env, env_key_map)
        # Also set any KEY=VALUE from the file into os.environ so that
        # third-party plugins (which read os.getenv in their own factories)
        # can discover keys from the user-specified file.
        _push_env_file_to_environ(env)

    client = UniversalLM()

    for cls in _CORE_ADAPTERS:
        manifest: ProviderManifest = cls.manifest  # type: ignore[attr-defined]
        p = manifest.provider
        # First key found wins: explicit kwarg > env file > os.environ
        key = explicit.get(p)
        if not key:
            key = from_file.get(p)
        if not key:
            for var in manifest.env_keys:
                key = os.getenv(var)
                if key:
                    break
        if key:
            client.register(cls(api_key=key, transport=transport))  # type: ignore[call-arg]

    if hydrate_models_dev:
        try:
            hydrate_with_specs(fetch_models_dev())
        except Exception:
            pass

    if discover_plugins:
        load_plugins(client)

    return client
