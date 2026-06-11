"""Scratch venv management for competitor benchmarks.

Each package under test gets its own throwaway venv under
/tmp/lm15-bench-venvs/, created with ``uv venv`` and populated with
``uv pip install``.  lm15 itself is installed from the locally built wheel
so the measurement matches what an end user would download.

Install failures are recorded (not raised) so the suite degrades
gracefully when a competitor cannot be installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VENV_ROOT = Path("/tmp/lm15-bench-venvs")
REPO = Path(__file__).resolve().parents[2]

# name -> (pip install target, fairest import statement)
PACKAGES: dict[str, tuple[str, str]] = {
    "lm15": ("<local-wheel>", "import lm15"),
    "openai": ("openai", "import openai"),
    "anthropic": ("anthropic", "import anthropic"),
    "google-genai": ("google-genai", "from google import genai"),
    "litellm": ("litellm", "import litellm"),
    "langchain-openai": ("langchain-openai", "import langchain_openai"),
}


@dataclass
class Venv:
    name: str
    path: Path
    import_stmt: str
    ok: bool
    error: str | None = None
    install_target: str = ""

    @property
    def python(self) -> Path:
        return self.path / "bin" / "python"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def local_wheel() -> Path:
    wheels = sorted((REPO / "dist").glob("lm15-*.whl"))
    if not wheels:
        raise RuntimeError("no lm15 wheel in dist/ — run `uv build` first")
    return wheels[-1]


def make_venv(name: str, fresh: bool = True) -> Venv:
    target, import_stmt = PACKAGES[name]
    path = VENV_ROOT / name
    if fresh and path.exists():
        shutil.rmtree(path)
    if not (path / "bin" / "python").exists():
        r = _run(["uv", "venv", "--python", sys.executable, str(path)])
        if r.returncode != 0:
            return Venv(name, path, import_stmt, ok=False, error=r.stderr[-500:])
    if target == "<local-wheel>":
        target = str(local_wheel())
    r = _run(["uv", "pip", "install", "--python", str(path / "bin" / "python"), target])
    if r.returncode != 0:
        return Venv(name, path, import_stmt, ok=False,
                    error=r.stderr[-500:], install_target=target)
    return Venv(name, path, import_stmt, ok=True, install_target=target)


def make_baseline() -> Venv:
    """Empty venv used as the size/RSS baseline."""
    path = VENV_ROOT / "_baseline"
    if path.exists():
        shutil.rmtree(path)
    r = _run(["uv", "venv", "--python", sys.executable, str(path)])
    ok = r.returncode == 0
    return Venv("_baseline", path, "pass", ok=ok,
                error=None if ok else r.stderr[-500:])


def site_packages(v: Venv) -> Path:
    hits = sorted(v.path.glob("lib/python*/site-packages"))
    if not hits:
        raise RuntimeError(f"no site-packages in {v.path}")
    return hits[0]


def installed_packages(v: Venv) -> list[str]:
    r = _run(["uv", "pip", "list", "--python", str(v.python), "--format", "json"])
    if r.returncode != 0:
        return []
    import json
    return [p["name"] for p in json.loads(r.stdout)]
