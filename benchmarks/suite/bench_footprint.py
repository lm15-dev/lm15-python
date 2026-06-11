"""Install footprint: on-disk size of site-packages and transitive
dependency count, per scratch venv.

Size = bytes under the venv's site-packages minus the baseline (empty)
venv's site-packages, i.e. exactly what the install added.
Dependency count = installed packages minus the package itself.
"""

from __future__ import annotations

from pathlib import Path

from venvs import Venv, installed_packages, site_packages


def _du_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def measure_footprint(v: Venv, baseline: Venv) -> dict:
    size = _du_bytes(site_packages(v)) - _du_bytes(site_packages(baseline))
    pkgs = installed_packages(v)
    deps = max(0, len(pkgs) - 1)  # exclude the package under test itself
    return {
        "install_bytes": size,
        "install_mib": round(size / (1024 * 1024), 2),
        "package_count": len(pkgs),
        "transitive_deps": deps,
        "packages": sorted(pkgs),
    }
