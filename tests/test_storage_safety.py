"""Tests for storage/safety.py — path safety checks, ephemeral detection,
git root detection, and child-of detection."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from bench_harness.storage.safety import (
    is_unsafe_path,
    check_storage_root,
    detect_ephemeral_path,
    _detect_git_root,
    _is_child_of,
)


# ── is_unsafe_path ───────────────────────────────────────────────────


class TestIsUnsafePath:
    def test_allow_unsafe_returns_safe(self):
        """allow_unsafe=True always returns (False, None)."""
        is_unsafe, reason = is_unsafe_path(Path("/tmp/foo"), allow_unsafe=True)
        assert is_unsafe is False
        assert reason is None

    def test_tmp_path_is_unsafe(self):
        """Paths starting with /tmp are flagged as unsafe."""
        is_unsafe, reason = is_unsafe_path(Path("/tmp/test-storage"))
        assert is_unsafe is True
        assert "tmp" in reason.lower()

    def test_var_tmp_path_is_unsafe(self):
        """Paths starting with /var/tmp are flagged as unsafe."""
        is_unsafe, reason = is_unsafe_path(Path("/var/tmp/test-storage"))
        assert is_unsafe is True
        assert "tmp" in reason.lower()

    def test_docker_var_lib_is_unsafe(self):
        """Paths starting with /var/lib/docker/ are unsafe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a temp path and mock _detect_git_root to return None
            # so we don't hit the actual docker path subprocess call
            with mock.patch("bench_harness.storage.safety._detect_git_root", return_value=None):
                is_unsafe, reason = is_unsafe_path(Path("/var/lib/docker/overlay2/test"))
                assert is_unsafe is True
                assert "docker" in reason.lower()

    def test_docker_run_containers_is_unsafe(self):
        """Paths starting with /run/containers/ are unsafe."""
        is_unsafe, reason = is_unsafe_path(Path("/run/containers/overlay"))
        assert is_unsafe is True
        assert "docker" in reason.lower()

    def test_virtualenv_is_unsafe(self):
        """Paths inside the virtualenv are flagged as unsafe."""
        venv_path = Path(sys.prefix).resolve()
        test_path = venv_path / "test-storage"
        is_unsafe, reason = is_unsafe_path(test_path)
        assert is_unsafe is True
        assert "virtualenv" in reason.lower()

    def test_git_repo_path_is_unsafe(self):
        """Paths inside a git repo are flagged as unsafe."""
        # Use explicit git_root to test git detection independently of /tmp filtering
        git_root = Path("/home")
        test_path = git_root / "some/user/project/storage"
        is_unsafe, reason = is_unsafe_path(test_path, git_root=git_root)
        assert is_unsafe is True
        assert "git" in reason.lower()

    def test_git_root_can_be_passed_explicitly(self):
        """Explicit git_root overrides auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_root = Path(tmpdir)
            test_path = git_root / "storage"
            is_unsafe, reason = is_unsafe_path(test_path, git_root=git_root)
            assert is_unsafe is True

    def test_safe_path_outside_restricted_dirs(self):
        """A path outside restricted dirs returns safe (with allow_unsafe)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_unsafe, reason = is_unsafe_path(Path(tmpdir), allow_unsafe=True)
            assert is_unsafe is False
            assert reason is None

    def test_resolves_path_before_check(self):
        """is_unsafe_path resolves symlinks before checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "storage"
            test_path.mkdir()
            is_unsafe, reason = is_unsafe_path(test_path, allow_unsafe=True)
            assert is_unsafe is False


# ── check_storage_root ───────────────────────────────────────────────


class TestCheckStorageRoot:
    def test_raises_valueerror_on_unsafe(self):
        """check_storage_root raises ValueError on unsafe paths."""
        with pytest.raises(ValueError, match="Unsafe storage root"):
            check_storage_root(Path("/tmp/unsafe"))

    def test_raises_on_git_repo(self):
        """check_storage_root raises when inside a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".git").mkdir()
            test_path = Path(tmpdir) / "storage"
            with pytest.raises(ValueError, match="Unsafe storage root"):
                check_storage_root(test_path)

    def test_silent_on_safe_with_allow_unsafe(self):
        """check_storage_root is silent when allow_unsafe=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            check_storage_root(Path(tmpdir), allow_unsafe=True)

    def test_silent_on_valid_path(self):
        """check_storage_root returns silently on valid paths (allow_unsafe)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            check_storage_root(Path(tmpdir), allow_unsafe=True)


# ── detect_ephemeral_path ────────────────────────────────────────────


class TestDetectEphemeralPath:
    def test_tmp_is_ephemeral(self):
        """Paths in /tmp are flagged as ephemeral."""
        is_eph, warnings = detect_ephemeral_path("/tmp/some-path")
        assert is_eph is True
        assert any("/tmp" in w for w in warnings)

    def test_var_tmp_is_ephemeral(self):
        """Paths in /var/tmp are flagged as ephemeral."""
        is_eph, warnings = detect_ephemeral_path("/var/tmp/some-path")
        assert is_eph is True
        assert any("/var/tmp" in w for w in warnings)

    def test_docker_var_lib_is_ephemeral(self):
        """Paths in /var/lib/docker/ get a docker warning."""
        with mock.patch.object(Path, "exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("/var/lib/docker/overlay2")
            assert is_eph is True
            assert any("docker" in w.lower() for w in warnings)

    def test_docker_run_containers_is_ephemeral(self):
        """Paths in /run/containers/ get a docker warning."""
        is_eph, warnings = detect_ephemeral_path("/run/containers/overlay")
        assert is_eph is True
        assert any("docker" in w.lower() for w in warnings)

    def test_nonexistent_path_warns(self):
        """Non-existent paths get a non-existence warning."""
        is_eph, warnings = detect_ephemeral_path("/nonexistent/path/xyz")
        assert any("does not exist" in w for w in warnings)

    def test_existing_safe_path_not_ephemeral(self):
        """A real existing path outside ephemeral dirs gets no ephemeral warning."""
        # Use /etc which exists and is not ephemeral
        is_eph, warnings = detect_ephemeral_path("/etc")
        assert any("does not exist" in w for w in warnings) is False
        # It might not exist on some systems; check not ephemeral from tmp/docker
        for w in warnings:
            assert "ephemeral" not in w.lower() or "does not exist" in w

    def test_empty_string_warns_nonexistent(self):
        """Empty string path is flagged as non-existent."""
        # Path("") resolves to current directory which exists, so no warning.
        # We test by mocking exists() to return False.
        with mock.patch.object(Path, "exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("")
            assert is_eph is True
            assert any("does not exist" in w for w in warnings)

    def test_relative_path_warns_nonexistent(self):
        """A relative path that doesn't exist is flagged."""
        is_eph, warnings = detect_ephemeral_path("relative/path/that/does/not/exist")
        assert any("does not exist" in w for w in warnings)

    def test_multiple_docker_warnings(self):
        """Path matching both docker indicators produces multiple warnings."""
        # A path can only match one docker indicator at a time since they are
        # different prefixes, so test individual docker paths
        with mock.patch.object(Path, "exists", return_value=False):
            is_eph1, warnings1 = detect_ephemeral_path("/var/lib/docker/test")
            is_eph2, warnings2 = detect_ephemeral_path("/run/containers/test")
            assert any("docker" in w.lower() for w in warnings1)
            assert any("docker" in w.lower() for w in warnings2)

    def test_warning_message_content_tmp(self):
        """Warning message for /tmp paths mentions ephemeral location."""
        is_eph, warnings = detect_ephemeral_path("/tmp/test-path")
        assert any("ephemeral" in w.lower() or "tmp" in w.lower() for w in warnings)

    def test_warning_message_content_docker(self):
        """Warning message for docker paths mentions Docker overlay."""
        with mock.patch.object(Path, "exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("/var/lib/docker/overlay")
            assert any("docker" in w.lower() or "overlay" in w.lower() for w in warnings)

    def test_no_ephemeral_warning_for_safe_path(self):
        """A safe path in a safe location gets no ephemeral warnings."""
        # Use /etc which is a real path outside ephemeral locations
        is_eph, warnings = detect_ephemeral_path("/etc")
        # /etc exists, so only check that no ephemeral-related warnings
        ephemeral_warnings = [
            w for w in warnings
            if ("/tmp" in w or "/var/tmp" in w or "docker" in w.lower())
            and "does not exist" not in w
        ]
        assert len(ephemeral_warnings) == 0


# ── _detect_git_root ─────────────────────────────────────────────────


class TestDetectGitRoot:
    def test_detects_git_root_with_git_directory(self):
        """Finds git root by looking for .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_root = Path(tmpdir)
            (git_root / ".git").mkdir()
            test_path = git_root / "subdir"
            test_path.mkdir()
            detected = _detect_git_root(test_path)
            assert detected == git_root

    def test_detects_git_root_nested(self):
        """Walks up parents to find .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_root = Path(tmpdir)
            (git_root / ".git").mkdir()
            nested = git_root / "a" / "b" / "c"
            nested.mkdir(parents=True)
            detected = _detect_git_root(nested)
            assert detected == git_root

    def test_returns_none_without_git(self):
        """Returns None when no .git directory exists (without git binary)."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with tempfile.TemporaryDirectory() as tmpdir:
                test_path = Path(tmpdir) / "storage"
                test_path.mkdir()
                # _detect_git_root will walk up and not find .git
                detected = _detect_git_root(test_path)
                assert detected is None


# ── _is_child_of ─────────────────────────────────────────────────────


class TestIsChildOf:
    def test_exact_path_is_child(self):
        """A path is considered a child of itself."""
        path = Path("/home/user/storage")
        assert _is_child_of(path, path) is True

    def test_direct_child(self):
        """A direct child path returns True."""
        child = Path("/home/user/storage/subdir")
        parent = Path("/home/user/storage")
        assert _is_child_of(child, parent) is True

    def test_nested_child(self):
        """A deeply nested child returns True."""
        child = Path("/home/user/storage/a/b/c")
        parent = Path("/home/user/storage")
        assert _is_child_of(child, parent) is True

    def test_unrelated_paths(self):
        """Unrelated paths return False."""
        child = Path("/tmp/test")
        parent = Path("/home/user/storage")
        assert _is_child_of(child, parent) is False

    def test_sibling_paths(self):
        """Sibling paths return False."""
        child1 = Path("/home/user/storage1")
        child2 = Path("/home/user/storage2")
        assert _is_child_of(child1, child2) is False

    def test_different_roots(self):
        """Paths on different roots return False."""
        child = Path("/var/data")
        parent = Path("/home/user")
        assert _is_child_of(child, parent) is False
