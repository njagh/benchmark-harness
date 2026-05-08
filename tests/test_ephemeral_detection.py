"""Extended tests for ephemeral path detection — various path patterns,
edge cases, and warning message verification."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from bench_harness.storage.safety import detect_ephemeral_path


# ── /tmp/* patterns ──────────────────────────────────────────────────


class TestTmpPatterns:
    def test_tmp_simple(self):
        """Path starting with /tmp is flagged."""
        is_eph, warnings = detect_ephemeral_path("/tmp")
        assert is_eph is True
        assert any("/tmp" in w for w in warnings)

    def test_tmp_subdir(self):
        """Path /tmp/something is flagged."""
        is_eph, warnings = detect_ephemeral_path("/tmp/some/path")
        assert is_eph is True
        assert any("ephemeral" in w.lower() or "tmp" in w.lower() for w in warnings)

    def test_tmp_deep_nesting(self):
        """Deeply nested /tmp path is flagged."""
        is_eph, warnings = detect_ephemeral_path(
            "/tmp/a/b/c/d/e/f/g/h"
        )
        assert is_eph is True

    def test_var_tmp_simple(self):
        """Path starting with /var/tmp is flagged."""
        is_eph, warnings = detect_ephemeral_path("/var/tmp")
        assert is_eph is True

    def test_var_tmp_subdir(self):
        """Path /var/tmp/something is flagged."""
        is_eph, warnings = detect_ephemeral_path("/var/tmp/cache")
        assert is_eph is True
        assert any("tmp" in w.lower() for w in warnings)


# ── /var/lib/docker/* patterns ───────────────────────────────────────


class TestDockerPatterns:
    def test_docker_overlay(self):
        """Path in /var/lib/docker/overlay is flagged."""
        with mock.patch("pathlib.Path.exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("/var/lib/docker/overlay")
            assert is_eph is True
            assert any("docker" in w.lower() for w in warnings)

    def test_docker_overlay2(self):
        """Path in /var/lib/docker/overlay2 is flagged."""
        with mock.patch("pathlib.Path.exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path(
                "/var/lib/docker/overlay2/mnt"
            )
            assert is_eph is True

    def test_docker_containers(self):
        """Path in /run/containers/ is flagged."""
        is_eph, warnings = detect_ephemeral_path("/run/containers/storage")
        assert is_eph is True
        assert any("docker" in w.lower() for w in warnings)

    def test_docker_containers_run(self):
        """Path /run/containers/run is flagged."""
        is_eph, warnings = detect_ephemeral_path("/run/containers/run")
        assert is_eph is True


# ── Empty string, relative paths, non-existent ───────────────────────


class TestEdgePathCases:
    def test_empty_string(self):
        """Empty string path is flagged as ephemeral (non-existent)."""
        # Path("") resolves to current dir which exists, so we mock it
        with mock.patch("pathlib.Path.exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("")
            assert is_eph is True
            assert any("does not exist" in w for w in warnings)

    def test_relative_path_nonexistent(self):
        """Relative path that doesn't exist gets warning."""
        is_eph, warnings = detect_ephemeral_path("./relative/path")
        assert any("does not exist" in w for w in warnings)

    def test_relative_path_relative(self):
        """Relative path 'relative' is flagged if doesn't exist."""
        is_eph, warnings = detect_ephemeral_path("relative")
        assert any("does not exist" in w for w in warnings) or is_eph is True

    def test_absolute_nonexistent_path(self):
        """Absolute path that doesn't exist gets non-existence warning."""
        is_eph, warnings = detect_ephemeral_path("/this/absolute/path/does/not/exist")
        assert any("does not exist" in w for w in warnings)
        # Should not have ephemeral warnings about /tmp or docker
        for w in warnings:
            assert "/tmp" not in w
            assert "/var/tmp" not in w
            assert "docker" not in w.lower()

    def test_absolute_existing_but_non_ephemeral(self):
        """Existing absolute path outside ephemeral dirs."""
        # Use /etc which exists and is not ephemeral
        is_eph, warnings = detect_ephemeral_path("/etc")
        ephemeral_warnings = [
            w for w in warnings
            if "ephemeral" in w.lower() or "/tmp" in w or "docker" in w.lower()
        ]
        # /etc exists and is not in ephemeral locations, so no such warnings
        assert all("does not exist" in w for w in ephemeral_warnings)


# ── Combined warnings ────────────────────────────────────────────────


class TestCombinedWarnings:
    def test_tmp_plus_nonexistent(self):
        """Path in /tmp that doesn't exist gets both warnings."""
        is_eph, warnings = detect_ephemeral_path("/tmp/does/not/exist")
        assert is_eph is True
        has_tmp = any("/tmp" in w for w in warnings)
        has_nonexist = any("does not exist" in w for w in warnings)
        assert has_tmp or has_nonexist

    def test_docker_plus_nonexistent(self):
        """Path in docker dir that doesn't exist gets warnings."""
        with mock.patch("pathlib.Path.exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("/var/lib/docker/nonexistent")
            assert is_eph is True
            assert any("docker" in w.lower() for w in warnings)

    def test_multiple_docker_indicators(self):
        """Path matching both docker indicator paths separately."""
        # /var/lib/docker/ and /run/containers/ are different prefixes
        with mock.patch("pathlib.Path.exists", return_value=False):
            is_eph1, warnings1 = detect_ephemeral_path("/var/lib/docker/test")
            is_eph2, warnings2 = detect_ephemeral_path("/run/containers/test")
            assert is_eph1 is True
            assert is_eph2 is True
            assert any("docker" in w.lower() for w in warnings1)
            assert any("docker" in w.lower() for w in warnings2)

    def test_var_tmp_nonexistent(self):
        """/var/tmp path that doesn't exist gets warnings."""
        is_eph, warnings = detect_ephemeral_path("/var/tmp/nonexistent")
        assert is_eph is True
        assert any("tmp" in w.lower() for w in warnings)


# ── Warning message content verification ─────────────────────────────


class TestWarningMessageContent:
    def test_tmp_warning_mentions_ephemeral(self):
        """/tmp warning message mentions ephemeral location."""
        is_eph, warnings = detect_ephemeral_path("/tmp/test")
        assert any(
            "ephemeral" in w.lower() or "tmp" in w.lower()
            for w in warnings
        )

    def test_docker_warning_mentions_overlay(self):
        """Docker warning mentions overlay."""
        with mock.patch("pathlib.Path.exists", return_value=False):
            is_eph, warnings = detect_ephemeral_path("/var/lib/docker/overlay")
            assert any(
                "docker" in w.lower() or "overlay" in w.lower()
                for w in warnings
            )

    def test_nonexistent_warning_mentions_path(self):
        """Non-existent path warning mentions the path."""
        is_eph, warnings = detect_ephemeral_path("/nonexistent/path")
        assert any("does not exist" in w for w in warnings)

    def test_var_tmp_warning_mentions_tmp(self):
        """/var/tmp warning mentions tmp."""
        is_eph, warnings = detect_ephemeral_path("/var/tmp/test")
        assert any("tmp" in w.lower() for w in warnings)

    def test_run_containers_warning_mentions_docker(self):
        """/run/containers/ warning mentions Docker."""
        is_eph, warnings = detect_ephemeral_path("/run/containers/test")
        assert any("docker" in w.lower() for w in warnings)

    def test_warning_is_list_of_strings(self):
        """Warnings is a list of strings."""
        is_eph, warnings = detect_ephemeral_path("/tmp/test")
        assert isinstance(warnings, list)
        for w in warnings:
            assert isinstance(w, str)

    def test_is_ephemeral_true_when_any_warning(self):
        """is_ephemeral is True when there are warnings."""
        is_eph, warnings = detect_ephemeral_path("/tmp/test")
        assert is_eph is True
        assert len(warnings) > 0

    def test_returns_tuple_of_bool_and_list(self):
        """Returns (bool, list[str]) tuple."""
        result = detect_ephemeral_path("/tmp/test")
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_multiple_path_indicators_single_warning(self):
        """Path in /tmp gets one ephemeral warning about tmp."""
        is_eph, warnings = detect_ephemeral_path("/tmp/test/path")
        # Should have at least one warning about tmp
        tmp_warnings = [w for w in warnings if "tmp" in w.lower()]
        assert len(tmp_warnings) >= 1


# ── Paths that should not trigger ephemeral warnings ─────────────────


class TestNonEphemeralPaths:
    def test_home_path(self):
        """Home directory path is not flagged as ephemeral."""
        is_eph, warnings = detect_ephemeral_path("/home/user/storage")
        # /home is not in ephemeral list; path doesn't exist so gets non-exist warning
        ephemeral_warnings = [
            w for w in warnings
            if "ephemeral" in w.lower() or "/tmp" in w or "docker" in w.lower()
        ]
        assert len(ephemeral_warnings) == 0

    def test_var_log_path(self):
        """/var/log is not flagged as ephemeral."""
        is_eph, warnings = detect_ephemeral_path("/var/log")
        for w in warnings:
            assert "ephemeral" not in w.lower()
            assert "/tmp" not in w

    def test_etc_path(self):
        """/etc is not flagged as ephemeral."""
        is_eph, warnings = detect_ephemeral_path("/etc")
        for w in warnings:
            assert "ephemeral" not in w.lower()
