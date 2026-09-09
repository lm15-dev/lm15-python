"""Source version must not depend on an installed distribution's metadata."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("ambient_version", ["7.8.9", None])
def test_source_version_ignores_distribution_metadata(ambient_version):
    code = f"""
import importlib.metadata

def foreign_version(name):
    assert name == 'lm15'
    if {ambient_version!r} is None:
        raise importlib.metadata.PackageNotFoundError(name)
    return {ambient_version!r}

importlib.metadata.version = foreign_version
import lm15
from lm15 import vet
from lm15._version import __version__
assert lm15.__version__ == __version__ == '1.0.0a1'
assert vet.IMPL_VERSION == __version__
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)
