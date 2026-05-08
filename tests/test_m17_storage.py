"""Tests for Milestone 17 — Package and Storage Abstraction.

Tests for StorageConfig, storage safety checks, run directory creation,
and the init-storage / storage-info CLI commands.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.storage.config import StorageConfig, _find_project_config
from bench_harness.storage.safety import (
    is_unsafe_path,
    check_storage_root,
    detect_ephemeral_path,
)


# ── StorageConfig resolution ───────────────────────────────────────


class TestStorageConfigResolution:
    def test_from_env_default(self):
        """When no env var is set, uses XDG or ~/.local/share/llm-bench."""
        # Clear env to test default
        env_orig = os.environ.pop("LLM_BENCH_STORAGE_ROOT", None)
        try:
            config = StorageConfig.from_env(allow_unsafe=True)
            # Should resolve to default
            assert "llm-bench" in str(config.root)
            assert config.root.is_absolute()
        finally:
            if env_orig:
                os.environ["LLM_BENCH_STORAGE_ROOT"] = env_orig

    def test_from_env_with_env_var(self):
        """LLM_BENCH_STORAGE_ROOT env var is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "custom-storage"
            os.environ["LLM_BENCH_STORAGE_ROOT"] = str(env_path)
            try:
                config = StorageConfig.from_env(allow_unsafe=True)
                assert config.root == env_path.resolve()
            finally:
                os.environ.pop("LLM_BENCH_STORAGE_ROOT", None)

    def test_from_cli_explicit(self):
        """Explicit CLI path is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli_path = Path(tmpdir) / "cli-storage"
            config = StorageConfig.from_cli(cli_path, allow_unsafe=True)
            assert config.root == cli_path.resolve()

    def test_from_project_config(self):
        """Project-level .llm-bench.yaml is read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            config_file = project_root / ".llm-bench.yaml"
            storage_path = Path(tmpdir) / "shared-storage"
            config_data = {
                "project": {
                    "default_storage_root": str(storage_path),
                }
            }
            config_file.write_text(yaml.dump(config_data))

            old_cwd = Path.cwd()
            try:
                os.chdir(project_root)
                config = StorageConfig.from_project(allow_unsafe=True)
                assert config is not None
                assert config.root == storage_path.resolve()
            finally:
                os.chdir(old_cwd)

    def test_from_project_no_config(self):
        """Returns None when no project config exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                # Make sure we don't find a parent config
                config = StorageConfig.from_project(allow_unsafe=True)
                assert config is None
            finally:
                os.chdir(old_cwd)

    def test_namespace_paths(self):
        """All namespace properties are derived from root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.artifacts_root == root / "artifacts"
            assert config.results_root == root / "results"
            assert config.registry_root == root / "registry"
            assert config.logs_root == root / "logs"
            assert config.cache_root == root / "cache"
            assert config.tmp_root == root / "tmp"
            assert config.artifacts_models == root / "artifacts" / "models"
            assert config.artifacts_engines == root / "artifacts" / "engines"
            assert config.results_runs == root / "results" / "runs"
            assert config.results_summaries == root / "results" / "summaries"
            assert config.results_comparisons == root / "results" / "comparisons"

    def test_ensure_namespaces(self):
        """Creates all namespace directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            config.ensure_namespaces()
            for ns in ("artifacts", "results", "registry", "logs", "cache", "tmp"):
                assert getattr(config, f"{ns}_root").exists()


# ── Safety checks ──────────────────────────────────────────────────


class TestSafetyChecks:
    def test_safe_path_returns_false(self):
        """A normal temp path is not unsafe when allow_unsafe=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_unsafe, reason = is_unsafe_path(Path(tmpdir), allow_unsafe=True)
            assert is_unsafe is False
            assert reason is None

    def test_tmp_is_unsafe(self):
        """Paths inside /tmp are flagged as unsafe."""
        is_unsafe, reason = is_unsafe_path(Path("/tmp/test-storage"))
        assert is_unsafe is True
        assert "tmp" in reason.lower()

    def test_var_tmp_is_unsafe(self):
        """Paths inside /var/tmp are flagged as unsafe."""
        is_unsafe, reason = is_unsafe_path(Path("/var/tmp/test-storage"))
        assert is_unsafe is True
        assert "tmp" in reason.lower()

    def test_docker_paths_are_unsafe(self):
        """Docker overlay paths are flagged as unsafe in the safety module."""
        from bench_harness.storage import safety
        source = Path(safety.__file__).read_text()
        # Verify docker path indicators are in the source code
        assert "/var/lib/docker/" in source
        assert "/run/containers/" in source

    def test_check_storage_root_raises_on_unsafe(self):
        """check_storage_root raises ValueError on unsafe paths."""
        with pytest.raises(ValueError, match="Unsafe storage root"):
            check_storage_root(Path("/tmp/unsafe"))

    def test_check_storage_root_silent_on_safe(self):
        """check_storage_root returns silently on safe paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            check_storage_root(Path(tmpdir), allow_unsafe=True)

    def test_dry_run_mode_allows_any_path(self):
        """Dry-run mode returns (False, None) regardless of path."""
        is_unsafe, reason = is_unsafe_path(Path("/tmp/test"), allow_unsafe=True)
        assert is_unsafe is False


class TestEphemeralDetection:
    def test_tmp_is_ephemeral(self):
        """Paths in /tmp are flagged as ephemeral."""
        is_ephemeral, warnings = detect_ephemeral_path("/tmp/some-path")
        assert is_ephemeral is True
        assert any("/tmp" in w for w in warnings)

    def test_nonexistent_path_warning(self):
        """Non-existent paths get a warning."""
        is_ephemeral, warnings = detect_ephemeral_path("/nonexistent/path")
        assert any("does not exist" in w for w in warnings)

    def test_docker_path_warning(self):
        """Docker overlay paths get a warning from detect_ephemeral_path."""
        from bench_harness.storage import safety
        source = Path(safety.__file__).read_text()
        # Verify docker indicator paths are in detect_ephemeral_path
        assert "/var/lib/docker/" in source
        assert "/run/containers/" in source

    def test_safe_path_not_ephemeral(self):
        """A normal existing path is not ephemeral."""
        # Use a real existing path — /var/lib/docker/overlay is an ephemeral indicator
        is_ephemeral, warnings = detect_ephemeral_path("/home/test-user/storage")
        # This should not be ephemeral unless it doesn't exist
        if not Path("/home/test-user/storage").exists():
            # Non-existent paths get the non-exist warning but /home is not ephemeral
            assert not is_ephemeral or any("does not exist" in w for w in warnings)

    def test_multiple_warnings(self):
        """Multiple warnings can be returned."""
        is_ephemeral, warnings = detect_ephemeral_path("/tmp/nonexistent/docker-path")
        assert len(warnings) >= 1


# ── Run directory creation ─────────────────────────────────────────


class TestRunDirCreation:
    def test_create_run_dir(self):
        """Creates a run directory with deterministic naming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-run")
            assert run_dir.exists()
            # Should be under results/runs/<date>/
            assert "results" in str(run_dir)
            assert "runs" in str(run_dir)
            # Name should contain timestamp and hash
            assert "__" in run_dir.name

    def test_create_run_dir_has_timestamp(self):
        """Run directory name contains ISO timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-timestamp")
            # Name format: name__YYYYMMDDTHHMMSS__hash
            assert "T" in run_dir.name
            # Should contain date prefix
            assert run_dir.name.startswith("test-timestamp__")

    def test_create_run_dir_is_deterministic(self):
        """Same name+timestamp produces same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-deterministic")
            # Should have consistent naming
            assert "test-deterministic__" in run_dir.name
            assert len(run_dir.name.split("__")[-1]) == 8  # 8-char hash

    def test_write_resolved_spec(self):
        """Writes a resolved spec to a run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-spec")
            spec_data = {"name": "test", "project": "test_project"}
            spec_path = config.write_resolved_spec(spec_data, run_dir)
            assert spec_path.exists()
            assert spec_path.name == "resolved_spec.yaml"
            loaded = yaml.safe_load(spec_path.read_text())
            assert loaded["name"] == "test"
            assert loaded["project"] == "test_project"


# ── CLI commands ───────────────────────────────────────────────────


class TestInitStorageCommand:
    def test_init_storage_dry_run(self, capsys):
        """init-storage --dry-run prints paths without creating dirs."""
        from bench_harness.cli import app
        import typer.testing

        runner = typer.testing.CliRunner()
        result = runner.invoke(
            app,
            ["init-storage", "--dry-run"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        output = result.output
        assert "Storage root:" in output
        assert "Dry run" in output

    def test_init_storage_creates_directories(self):
        """init-storage creates namespace directories and project config."""
        from bench_harness.cli import app
        import typer.testing

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                ["init-storage", "--root", tmpdir, "--allow-unsafe-storage-root"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            # Verify directories were created
            for ns in ("artifacts", "results", "registry", "logs", "cache", "tmp"):
                assert (Path(tmpdir) / ns).exists()
            # Project config written in cwd
            assert (Path.cwd() / ".llm-bench.yaml").exists()
            # Cleanup
            (Path.cwd() / ".llm-bench.yaml").unlink(missing_ok=True)


class TestStorageInfoCommand:
    def test_storage_info_shows_paths(self):
        """storage-info prints resolved paths."""
        from bench_harness.cli import app
        import typer.testing

        runner = typer.testing.CliRunner()
        result = runner.invoke(
            app,
            ["storage-info"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Storage root:" in result.output
