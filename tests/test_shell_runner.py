"""Tests for M22 — ShellRunner (zero-coverage module)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bench_harness.runners.shell_runner import ShellRunner


class TestShellRunner:
    def test_run_allows_without_allowlist(self, tmp_path):
        """No allowlist → all commands permitted."""
        runner = ShellRunner()
        result = runner.run("echo hello", str(tmp_path))
        assert result["passed"] is True
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_run_blocked_by_allowlist(self, tmp_path):
        """Command not in allowlist → blocked."""
        runner = ShellRunner(allowlist=["docker compose", "python"])
        result = runner.run("rm -rf /", str(tmp_path))
        assert result["passed"] is False
        assert result["exit_code"] == 1
        assert "not in allowlist" in result["stderr"]

    def test_allowlist_prefix_match(self, tmp_path):
        """Allowlist entry matches prefix of command."""
        runner = ShellRunner(allowlist=["docker compose"])
        result = runner.run("docker compose config -q", str(tmp_path))
        # docker compose may not be installed, but should pass allowlist
        assert "not in allowlist" not in result["stderr"]

    def test_allowlist_first_word_match(self, tmp_path):
        """Allowlist entry matches first word of command."""
        runner = ShellRunner(allowlist=["echo"])
        result = runner.run("echo test --verbose", str(tmp_path))
        assert result["passed"] is True
        assert "test" in result["stdout"]

    def test_allowlist_partial_not_matched(self, tmp_path):
        """Allowlist does not match partial words."""
        runner = ShellRunner(allowlist=["echo"])
        # "techo" is not "echo"
        result = runner.run("techo hello", str(tmp_path))
        assert "not in allowlist" in result["stderr"]

    def test_run_timeout(self):
        """Command exceeding timeout returns timeout error."""
        runner = ShellRunner()
        result = runner.run("sleep 10", "/tmp", timeout=1)
        assert result["passed"] is False
        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"]
        assert result["duration_ms"] > 0

    def test_run_failure_not_blocked(self, tmp_path):
        """Command in allowlist that fails → failure recorded."""
        runner = ShellRunner()
        result = runner.run("ls /nonexistent_dir_12345", str(tmp_path))
        assert result["passed"] is False
        assert result["exit_code"] != 0

    def test_run_duration_measured(self, tmp_path):
        """duration_ms field is populated."""
        runner = ShellRunner()
        result = runner.run("echo hello", str(tmp_path))
        assert result["duration_ms"] > 0

    def test_run_command_not_found(self, tmp_path):
        """Non-existent command handled gracefully."""
        runner = ShellRunner()
        result = runner.run("definitely_not_a_real_command_xyz123", str(tmp_path))
        # Shell returns non-zero exit code for not found
        assert result["passed"] is False
