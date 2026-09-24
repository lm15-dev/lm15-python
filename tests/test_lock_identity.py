"""AUTH-4 lock identity regressions. Windows symlink tests need Developer Mode.

These exercise path computation only, not a locking backend. Windows old/new
SDK processes must not overlap during the lock-filename migration.
"""

import errno
import hashlib
import os
from pathlib import Path

import pytest

from lm15._authlock import _windows_identity_key, lock_path_for


def test_existing_posix_hash_is_unchanged(tmp_path):
    target = tmp_path / "credentials.json"
    target.write_text("{}", encoding="utf-8")
    canonical = os.path.realpath(target)
    if os.name == "nt":
        canonical = _windows_identity_key(canonical)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32] + ".lock"
    assert lock_path_for(target).name == expected


def test_missing_leaf_and_nested_directories_behind_alias(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    for suffix in ("credentials.json", "one/two/credentials.json", "one/./two/../credentials.json"):
        assert lock_path_for(f"{alias}/{suffix}") == lock_path_for(f"{real}/{suffix}")
    assert lock_path_for(f"{alias}/one/./two/../credentials.json") == lock_path_for(real / "one/credentials.json")
    assert not (real / "one").exists()
    before = lock_path_for(alias / "credentials.json")
    (real / "credentials.json").write_text("{}", encoding="utf-8")
    assert lock_path_for(alias / "credentials.json") == before


def test_dangling_leaf_links_and_link_aliases(tmp_path):
    (tmp_path / "real").mkdir()
    dangling = tmp_path / "dangling"
    dangling.symlink_to("real/missing/credentials.json")
    alias = tmp_path / "alias"
    alias.symlink_to("dangling")
    absolute = tmp_path / "absolute"
    absolute.symlink_to(tmp_path / "real/missing/credentials.json")
    direct = lock_path_for(tmp_path / "real/missing/credentials.json")
    assert lock_path_for(dangling) == direct
    assert lock_path_for(alias) == direct
    assert lock_path_for(absolute) == direct


def test_dotdot_is_applied_after_symlink_resolution(tmp_path):
    (tmp_path / "real/child").mkdir(parents=True)
    (tmp_path / "alias").symlink_to("real/child", target_is_directory=True)
    expected = lock_path_for(tmp_path / "real/credentials.json")
    assert lock_path_for(f"{tmp_path}/alias/../credentials.json") == expected
    assert lock_path_for(f"{tmp_path}/absent/../alias/./../credentials.json") == expected
    assert expected != lock_path_for(tmp_path / "credentials.json")
    (tmp_path / "dangling-dir").symlink_to("real/not-created/child", target_is_directory=True)
    assert lock_path_for(f"{tmp_path}/dangling-dir/../credentials.json") == lock_path_for(tmp_path / "real/not-created/credentials.json")


def test_relative_paths_preserve_symlink_dotdot(tmp_path, monkeypatch):
    (tmp_path / "real/child").mkdir(parents=True)
    (tmp_path / "alias").symlink_to("real/child", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    assert lock_path_for("alias/../credentials.json") == lock_path_for(tmp_path / "real/credentials.json")


@pytest.mark.skipif(os.name == "nt", reason="POSIX HOME expansion")
def test_home_expansion_preserves_symlink_dotdot(tmp_path, monkeypatch):
    (tmp_path / "real/child").mkdir(parents=True)
    (tmp_path / "alias").symlink_to("real/child", target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert lock_path_for("~/alias/../credentials.json") == lock_path_for(tmp_path / "real/credentials.json")


def test_loops_and_nondirectory_ancestors_fail_closed(tmp_path):
    (tmp_path / "loop-a").symlink_to("loop-b")
    (tmp_path / "loop-b").symlink_to("loop-a")
    with pytest.raises(OSError) as caught:
        lock_path_for(tmp_path / "loop-a")
    assert caught.value.errno == errno.ELOOP
    leaf = tmp_path / "file"
    leaf.write_text("{}", encoding="utf-8")
    for suffix in ("child", "../credentials.json", "./credentials.json"):
        with pytest.raises(NotADirectoryError):
            lock_path_for(f"{leaf}/{suffix}")


def test_permission_error_is_not_a_missing_path(tmp_path, monkeypatch):
    target = Path(os.path.realpath(tmp_path)) / "denied"
    original = os.lstat

    def denied(path, *args, **kwargs):
        if os.fspath(path) == str(target):
            raise PermissionError(errno.EACCES, "permission denied", str(target))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", denied)
    with pytest.raises(PermissionError):
        lock_path_for(target)


@pytest.mark.skipif(os.name == "nt", reason="POSIX raw symlink bytes")
def test_non_unicode_symlink_target_is_not_hashed_lossily(tmp_path):
    os.symlink(b"invalid-\xff", os.fsencode(tmp_path / "invalid"))
    with pytest.raises(UnicodeEncodeError):
        lock_path_for(tmp_path / "invalid")


@pytest.mark.parametrize(("raw", "expected"), [
    (r"C:\Users\MAX\Auth.JSON", r"c:\users\max\auth.json"),
    (r"\\?\C:\Users\MAX\Auth.JSON", r"c:\users\max\auth.json"),
    ("C:/Users/MAX/Auth.JSON", r"c:\users\max\auth.json"),
    (r"\\?\UNC\Server\Share\Auth.JSON", r"\\server\share\auth.json"),
    (r"\\SERVER\SHARE\Auth.JSON", r"\\server\share\auth.json"),
    ("\\\\?\\UNC\\Server\\Share\\", "\\\\server\\share\\"),
    (r"C:\ÉCOLE\İ\ΟΣ\Auth.JSON", "c:\\école\\i\u0307\\ος\\auth.json"),
])
def test_windows_key_vectors_on_every_platform(raw, expected):
    assert _windows_identity_key(raw) == expected


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystem identity")
def test_windows_prefix_case_and_missing_leaf(tmp_path):
    real = tmp_path / "MixedCase"
    real.mkdir()
    target = str(real / "Auth.JSON")
    expected = lock_path_for(target)
    assert lock_path_for(target.swapcase()) == expected
    assert lock_path_for("\\\\?\\" + target) == expected
    assert lock_path_for(target.replace("\\", "/")) == expected
    (real / "Auth.JSON").write_text("{}", encoding="utf-8")
    assert lock_path_for(target.swapcase()) == expected
    unicode_file = real / "ΟΣ.json"
    unicode_file.write_text("{}", encoding="utf-8")
    assert lock_path_for(real / "οσ.json") == lock_path_for(unicode_file)
    # Final sigma: whether the volume's upcase table folds "ς" to "Σ" varies
    # (the GitHub Windows runner's NTFS does not: "ος.json" is another, missing
    # file there). The safety property is never a DIFFERENT lock for what may be
    # the same file: the same lock, or a refusal (a missing non-ASCII name).
    try:
        other = lock_path_for(real / "ος.json")
    except ValueError:
        pass
    else:
        assert other == lock_path_for(unicode_file)


@pytest.mark.skipif(os.name != "nt", reason="Windows unsupported namespaces")
def test_windows_ambiguous_missing_names_fail_closed(tmp_path):
    for name in ("ΟΣ.json", "οσ.json", "trailing. ", "file:stream", "NUL"):
        with pytest.raises(ValueError):
            lock_path_for(tmp_path / name)
    for target in (r"C:relative.json", r"\\.\NUL", r"\\?\GLOBALROOT\Device\X"):
        with pytest.raises(ValueError):
            lock_path_for(target)
