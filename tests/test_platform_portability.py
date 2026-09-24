"""lm15 imports, and degrades by name, on Pythons missing ssl, fcntl/msvcrt, or a home.

Each case runs in a fresh interpreter: what is being tested is module-scope
behaviour, which the importing process cannot undo. The stubs mirror what the
reduced targets really look like -- componentize-py's libpython has no ``_ssl``,
WASI reports ``posix`` without ``fcntl``, and ``Path.expanduser()`` raises
``RuntimeError`` wherever no home directory can be determined.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _run(source: str, *, env: dict[str, str] | None = None) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env=env,
        timeout=60,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _base_env(tmp_path: Path) -> dict[str, str]:
    """A minimal environment, per platform. Windows keeps SYSTEMROOT (sockets
    cannot initialize without it: WinError 10106) and puts the home folder in
    USERPROFILE."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "PYTHONPATH": str(_REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        env["PATH"] = os.path.join(os.environ["SYSTEMROOT"], "System32")
        env["USERPROFILE"] = str(tmp_path)
    return env


def test_imports_and_serves_plain_http_without_ssl(tmp_path: Path) -> None:
    out = _run(
        """
        import json, sys
        sys.modules["ssl"] = None  # ImportError on `import ssl`, as on a build without _ssl
        import lm15
        from lm15.transports._sync import StdlibTransport
        from lm15.transports._async import StdlibAsyncTransport
        tls_loaded = "lm15.transports._ssl" in sys.modules
        try:
            StdlibTransport()._tls_half()
        except Exception as exc:
            error = type(exc).__name__, str(exc)
        print(json.dumps({"tls_loaded": tls_loaded, "error": error}))
        """,
        env=_base_env(tmp_path),
    )
    assert out["tls_loaded"] is False, "importing the transports must not pull in ssl"
    name, message = out["error"]
    assert name == "ConnectError"
    assert "FetchTransport" in message


def test_imports_without_a_home_directory(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    del env["HOME"]
    env.pop("USERPROFILE", None)
    out = _run(
        """
        import json, pathlib

        def _no_home(self):
            raise RuntimeError("Could not determine home directory.")

        pathlib.Path.expanduser = _no_home  # what CPython raises with no HOME and no passwd entry
        from lm15 import auth
        from lm15._authlock import hold_file_lock
        try:
            with hold_file_lock(auth.CLAUDE_CODE_CREDENTIALS_PATH):
                pass
        except Exception as exc:
            error = type(exc).__name__, str(exc)
        print(json.dumps({
            "path": str(auth.CLAUDE_CODE_CREDENTIALS_PATH),
            "exists": auth.CLAUDE_CODE_CREDENTIALS_PATH.exists(),
            "error": error,
        }))
        """,
        env=env,
    )
    assert out["path"].startswith("~"), "constant stays a default, unexpanded"
    assert out["exists"] is False
    name, message = out["error"]
    assert name == "NotConfiguredError"
    assert "LM15_LOCK_DIR" in message


@pytest.mark.skipif(os.name == "nt", reason="Windows' own subprocess module imports msvcrt, so a Windows Python without it cannot start lm15 at all")
def test_lock_refuses_by_name_without_fcntl_or_msvcrt(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env["LM15_LOCK_DIR"] = str(tmp_path / "locks")
    out = _run(
        """
        import json, os, sys
        sys.modules["fcntl"] = None
        sys.modules["msvcrt"] = None
        from lm15 import auth, errors
        from lm15._authlock import hold_file_lock
        try:
            with hold_file_lock(os.environ["LM15_LOCK_DIR"] + "/../creds.json"):
                pass
        except errors.ConfigurationError as exc:
            error = type(exc).__name__, str(exc)
        try:
            auth.load_claude_code_credential(credentials_path=os.environ["LM15_LOCK_DIR"] + "/../absent.json")
        except errors.NotConfiguredError as exc:
            read = "not logged in"
        print(json.dumps({"error": error, "read": read}))
        """,
        env=env,
    )
    name, message = out["error"]
    assert name == "NotConfiguredError"
    assert "fcntl" in message and "msvcrt" in message
    assert out["read"] == "not logged in", "reading never needs the lock"


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is the POSIX primitive")
def test_real_platform_still_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lm15._authlock import hold_file_lock

    monkeypatch.setenv("LM15_LOCK_DIR", str(tmp_path / "locks"))
    target = tmp_path / "creds.json"
    with hold_file_lock(target, timeout_s=1.0):
        pass
