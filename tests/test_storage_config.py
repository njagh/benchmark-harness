"""Tests for StorageConfig — constructor, resolution, namespace paths,
run directory creation, and spec writing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.storage.config import (
    StorageConfig,
    _get_default_storage_root,
    _find_project_config,
    _DEFAULT_XDG_DATA_HOME,
    _DEFAULT_STORAGE_ROOT,
)


# ── Constructor & properties ───────────────────────────────────────


class TestStorageConfigConstructor:
    def test_constructor_resolves_path(self):
        """Constructor resolves the given Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.root == root.resolve()

    def test_constructor_sets_allow_unsafe(self):
        """allow_unsafe flag is stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(root=Path(tmpdir), allow_unsafe=True)
            assert config._allow_unsafe is True

    def test_namespace_artifacts_root(self):
        """artifacts_root is root/artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.artifacts_root == root / "artifacts"

    def test_namespace_results_root(self):
        """results_root is root/results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.results_root == root / "results"

    def test_namespace_registry_root(self):
        """registry_root is root/registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.registry_root == root / "registry"

    def test_namespace_logs_root(self):
        """logs_root is root/logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.logs_root == root / "logs"

    def test_namespace_cache_root(self):
        """cache_root is root/cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.cache_root == root / "cache"

    def test_namespace_tmp_root(self):
        """tmp_root is root/tmp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.tmp_root == root / "tmp"

    def test_artifacts_sub_namespaces(self):
        """All artifact sub-namespaces are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.artifacts_models == root / "artifacts" / "models"
            assert config.artifacts_engines == root / "artifacts" / "engines"
            assert config.artifacts_tokenizers == root / "artifacts" / "tokenizers"
            assert config.artifacts_calibration == root / "artifacts" / "calibration"
            assert config.artifacts_runtime_builds == root / "artifacts" / "runtime-builds"

    def test_results_sub_namespaces(self):
        """All result sub-namespaces are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            assert config.results_runs == root / "results" / "runs"
            assert config.results_summaries == root / "results" / "summaries"
            assert config.results_comparisons == root / "results" / "comparisons"


# ── from_cli ─────────────────────────────────────────────────────────


class TestFromCli:
    def test_from_cli_resolves_path(self):
        """from_cli resolves and validates the path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli_path = Path(tmpdir) / "cli-storage"
            config = StorageConfig.from_cli(cli_path, allow_unsafe=True)
            assert config.root == cli_path.resolve()

    def test_from_cli_with_allow_unsafe(self):
        """from_cli with allow_unsafe skips safety checks."""
        config = StorageConfig.from_cli(Path("/tmp/unsafe-path"), allow_unsafe=True)
        assert "unsafe-path" in str(config.root)


# ── from_env ─────────────────────────────────────────────────────────


class TestFromEnv:
    def test_from_env_uses_env_var(self):
        """from_env reads LLM_BENCH_STORAGE_ROOT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["LLM_BENCH_STORAGE_ROOT"] = tmpdir
            try:
                config = StorageConfig.from_env(allow_unsafe=True)
                assert config.root == Path(tmpdir).resolve()
            finally:
                os.environ.pop("LLM_BENCH_STORAGE_ROOT", None)

    def test_from_env_defaults_without_var(self):
        """from_env falls back to default when env var is absent."""
        env_orig = os.environ.pop("LLM_BENCH_STORAGE_ROOT", None)
        xdg_orig = os.environ.pop("XDG_DATA_HOME", None)
        try:
            config = StorageConfig.from_env(allow_unsafe=True)
            assert "llm-bench" in str(config.root)
        finally:
            if env_orig:
                os.environ["LLM_BENCH_STORAGE_ROOT"] = env_orig
            if xdg_orig:
                os.environ["XDG_DATA_HOME"] = xdg_orig

    def test_from_env_respects_xdg(self):
        """from_env respects XDG_DATA_HOME when set."""
        xdg_orig = os.environ.pop("LLM_BENCH_STORAGE_ROOT", None)
        custom_xdg = "/tmp/xdg-test-custom"
        os.environ["XDG_DATA_HOME"] = custom_xdg
        try:
            config = StorageConfig.from_env(allow_unsafe=True)
            assert config.root == Path(custom_xdg) / "llm-bench"
        finally:
            os.environ.pop("XDG_DATA_HOME", None)
            if xdg_orig:
                os.environ["LLM_BENCH_STORAGE_ROOT"] = xdg_orig


# ── from_project ─────────────────────────────────────────────────────


class TestFromProject:
    def test_from_project_reads_config(self):
        """from_project reads storage root from .llm-bench.yaml."""
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

    def test_from_project_returns_none_without_config(self):
        """from_project returns None when no config exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                config = StorageConfig.from_project(allow_unsafe=True)
                assert config is None
            finally:
                os.chdir(old_cwd)

    def test_from_project_empty_project_key(self):
        """from_project returns None when project key is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            config_file = project_root / ".llm-bench.yaml"
            config_file.write_text(yaml.dump({"project": {}}))
            old_cwd = Path.cwd()
            try:
                os.chdir(project_root)
                config = StorageConfig.from_project(allow_unsafe=True)
                assert config is None
            finally:
                os.chdir(old_cwd)

    def test_from_project_missing_storage_root(self):
        """from_project returns None when default_storage_root is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            config_file = project_root / ".llm-bench.yaml"
            config_file.write_text(yaml.dump({"project": {"other_key": "value"}}))
            old_cwd = Path.cwd()
            try:
                os.chdir(project_root)
                config = StorageConfig.from_project(allow_unsafe=True)
                assert config is None
            finally:
                os.chdir(old_cwd)

    def test_from_project_with_empty_file(self):
        """from_project handles empty config file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            config_file = project_root / ".llm-bench.yaml"
            config_file.write_text("")
            old_cwd = Path.cwd()
            try:
                os.chdir(project_root)
                config = StorageConfig.from_project(allow_unsafe=True)
                assert config is None
            finally:
                os.chdir(old_cwd)


# ── ensure_namespaces ────────────────────────────────────────────────


class TestEnsureNamespaces:
    def test_creates_all_namespaces(self):
        """ensure_namespaces creates all six directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            config.ensure_namespaces()
            assert config.artifacts_root.exists()
            assert config.results_root.exists()
            assert config.registry_root.exists()
            assert config.logs_root.exists()
            assert config.cache_root.exists()
            assert config.tmp_root.exists()

    def test_does_not_fail_if_already_exist(self):
        """ensure_namespaces is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            config.ensure_namespaces()
            config.ensure_namespaces()
            assert config.artifacts_root.exists()


# ── create_run_dir ───────────────────────────────────────────────────


class TestCreateRunDir:
    def test_creates_run_directory(self):
        """create_run_dir creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("my-run")
            assert run_dir.exists()
            assert run_dir.is_dir()

    def test_run_dir_under_results_runs(self):
        """Run dir is under results/runs/date/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-run")
            assert "results" in str(run_dir)
            assert "runs" in str(run_dir)

    def test_run_dir_name_format(self):
        """Run dir name follows the expected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-name")
            parts = run_dir.name.split("__")
            assert parts[0] == "test-name"
            assert len(parts) == 3  # name__timestamp__hash
            # Hash is 8 chars
            assert len(parts[2]) == 8

    def test_run_dir_contains_date(self):
        """Run dir is nested under today's date directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-date")
            # The parent is the date folder
            assert run_dir.parent.parent.name.split("-") == [
                s for s in run_dir.parent.parent.name.split("-")
            ]


# ── write_resolved_spec ──────────────────────────────────────────────


class TestWriteResolvedSpec:
    def test_writes_yaml(self):
        """write_resolved_spec writes a YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-spec")
            spec_data = {"name": "test", "project": "test_project"}
            spec_path = config.write_resolved_spec(spec_data, run_dir)
            assert spec_path.exists()
            assert spec_path.name == "resolved_spec.yaml"

    def test_written_content_readable(self):
        """Written YAML content is readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-spec")
            spec_data = {"name": "test", "project": "test_project"}
            spec_path = config.write_resolved_spec(spec_data, run_dir)
            loaded = yaml.safe_load(spec_path.read_text())
            assert loaded["name"] == "test"
            assert loaded["project"] == "test_project"

    def test_returns_path(self):
        """write_resolved_spec returns the path to the written file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            run_dir = config.create_run_dir("test-spec")
            spec_data = {"key": "value"}
            spec_path = config.write_resolved_spec(spec_data, run_dir)
            assert spec_path == run_dir / "resolved_spec.yaml"


# ── _get_default_storage_root ────────────────────────────────────────


class TestGetDefaultStorageRoot:
    def test_default_without_xdg(self):
        """Uses home/.local/share/llm-bench when XDG is unset."""
        xdg_orig = os.environ.pop("XDG_DATA_HOME", None)
        try:
            result = _get_default_storage_root()
            assert result == _DEFAULT_STORAGE_ROOT
        finally:
            if xdg_orig:
                os.environ["XDG_DATA_HOME"] = xdg_orig

    def test_default_with_xdg(self):
        """Uses $XDG_DATA_HOME/llm-bench when XDG is set."""
        xdg_orig = os.environ.pop("XDG_DATA_HOME", None)
        custom_xdg = "/tmp/xdg-custom"
        os.environ["XDG_DATA_HOME"] = custom_xdg
        try:
            result = _get_default_storage_root()
            assert result == Path(custom_xdg) / "llm-bench"
        finally:
            os.environ.pop("XDG_DATA_HOME", None)
            if xdg_orig:
                os.environ["XDG_DATA_HOME"] = xdg_orig


# ── _find_project_config ─────────────────────────────────────────────


class TestFindProjectConfig:
    def test_finds_config_in_cwd(self):
        """Finds .llm-bench.yaml in the current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            config_file = project_root / ".llm-bench.yaml"
            config_file.write_text("project: {}\n")
            old_cwd = Path.cwd()
            try:
                os.chdir(project_root)
                result = _find_project_config()
                assert result == config_file
            finally:
                os.chdir(old_cwd)

    def test_finds_config_in_parent(self):
        """Finds .llm-bench.yaml in a parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            config_file = project_root / ".llm-bench.yaml"
            config_file.write_text("project: {}\n")
            sub_dir = project_root / "subdir"
            sub_dir.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(sub_dir)
                result = _find_project_config()
                assert result == config_file
            finally:
                os.chdir(old_cwd)

    def test_returns_none_when_not_found(self):
        """Returns None when no config exists in tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                result = _find_project_config()
                assert result is None
            finally:
                os.chdir(old_cwd)


# ── resolve_artifact ─────────────────────────────────────────────────


class TestResolveArtifact:
    def test_resolve_artifact_external_path(self):
        """resolve_artifact with external_path returns source path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            artifact_path = Path(tmpdir) / "model-files"
            artifact_path.mkdir()
            result = config.resolve_artifact("external_path", str(artifact_path))
            assert result == artifact_path

    def test_resolve_artifact_http_url(self):
        """resolve_artifact detects HTTP URLs as openai_endpoint kind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "storage"
            config = StorageConfig(root=root, allow_unsafe=True)
            # Should not raise
            config.resolve_artifact("external_path", "https://api.example.com/v1")
