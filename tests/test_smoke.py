"""Smoke tests for Milestone 1 — Project Bootstrap."""

import tempfile
from pathlib import Path

import pytest

from bench_harness.config import (
    load_model_config,
    get_model,
    load_suite_config,
    get_suite,
    load_dataset_config,
    get_dataset,
    resolve_task_dir,
)
from bench_harness.tasks.loaders import load_task, load_tasks, filter_tasks
from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.runners.completion_runner import RunResult


@pytest.fixture
def project_root():
    """Return the project root directory."""
    # The test module is in tests/, so parent is project root
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir(project_root):
    """Return path to configs/."""
    return project_root / "configs"


@pytest.fixture
def smoke_task_dir(project_root):
    """Return path to tasks/smoke/."""
    return project_root / "tasks" / "smoke"


class TestModelConfig:
    def test_load_model_config(self, config_dir):
        """Config loads and contains expected models."""
        cfg = load_model_config()
        assert "models" in cfg
        assert "agent-code" in cfg["models"]
        assert "qwen-dense" in cfg["models"]
        assert "max-brain" in cfg["models"]

    def test_get_model(self, config_dir):
        """get_model returns correct config for alias."""
        cfg = load_model_config()
        model = get_model(cfg, "agent-code")
        assert model is not None
        assert model["backend"] == "vllm"
        assert model["quantization"] == "FP8"

    def test_get_model_not_found(self, config_dir):
        """get_model returns None for unknown alias."""
        cfg = load_model_config()
        assert get_model(cfg, "nonexistent") is None


class TestSuiteConfig:
    def test_load_suite_config(self, config_dir):
        cfg = load_suite_config()
        assert "suites" in cfg
        assert "smoke" in cfg["suites"]

    def test_get_suite(self, config_dir):
        cfg = load_suite_config()
        suite = get_suite(cfg, "smoke")
        assert suite is not None
        assert suite["default_runs"] == 1


class TestDatasetConfig:
    def test_load_dataset_config(self, config_dir):
        cfg = load_dataset_config()
        # datasets key may or may not be present
        assert isinstance(cfg, dict)

    def test_get_dataset(self, config_dir):
        cfg = load_dataset_config()
        if "datasets" in cfg:
            ds = get_dataset(cfg, "human_eval_v1")
            assert ds is not None


class TestTaskLoader:
    def test_load_smoke_tasks(self, smoke_task_dir):
        """Loads 5 tasks from smoke directory."""
        tasks = load_tasks(str(smoke_task_dir))
        assert len(tasks) >= 5

    def test_task_has_required_fields(self, smoke_task_dir):
        """Each loaded task has id, prompt, scoring."""
        tasks = load_tasks(str(smoke_task_dir))
        for task in tasks:
            assert "id" in task
            assert "prompt" in task
            assert "scoring" in task
            assert task["id"]

    def test_load_single_task(self, smoke_task_dir):
        """Load a single task file."""
        task_file = smoke_task_dir / "factual_trivial.yaml"
        task = load_task(str(task_file))
        assert task["id"] == "smoke.factual_001"
        assert "Paris" in task["expected"]["answer"]

    def test_filter_tasks_by_family(self, smoke_task_dir):
        """filter_tasks returns correct subset."""
        tasks = load_tasks(str(smoke_task_dir))
        coding = filter_tasks(tasks, "coding")
        assert all(t["family"] == "coding" for t in coding)
        assert len(coding) > 0

    def test_filter_tasks_no_match(self, smoke_task_dir):
        """filter_tasks returns empty for non-existent family."""
        tasks = load_tasks(str(smoke_task_dir))
        filtered = filter_tasks(tasks, "nonexistent")
        assert len(filtered) == 0

    def test_load_tasks_missing_keys_fails(self):
        """Task with missing required keys raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("id: bad_task\nprompt: hello\n")  # missing scoring
            f.flush()
            with pytest.raises(ValueError):
                load_task(f.name)


class TestSQLiteStore:
    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteStore(db_path)
        store.init()
        yield store
        Path(db_path).unlink(missing_ok=True)

    def test_sqlite_store_init(self, temp_db):
        """Database is created without error."""
        assert temp_db.db_path.exists()

    def test_sqlite_save_and_retrieve(self, temp_db):
        """Round-trip save and get of a run record."""
        result = RunResult(
            run_id="test-run-001",
            suite_id="smoke",
            task_id="smoke.factual_001",
            model_alias="agent-code",
            prompt="What is the capital of France?",
            raw_response="Paris",
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            ttft_ms=50.0,
            total_wall_ms=200.0,
            exit_status="success",
        )

        temp_db.save_run(result)
        runs = temp_db.get_runs(suite_id="smoke")
        assert len(runs) == 1
        assert runs[0]["task_id"] == "smoke.factual_001"
        assert runs[0]["model_alias"] == "agent-code"

    def test_sqlite_filter_by_model(self, temp_db):
        """get_runs filters correctly by model_alias."""
        r1 = RunResult(
            run_id="r1", suite_id="smoke", task_id="t1",
            model_alias="agent-code", exit_status="success",
        )
        r2 = RunResult(
            run_id="r2", suite_id="smoke", task_id="t1",
            model_alias="qwen-dense", exit_status="success",
        )
        temp_db.save_run(r1)
        temp_db.save_run(r2)

        agent_runs = temp_db.get_runs(model_alias="agent-code")
        assert len(agent_runs) == 1
        assert agent_runs[0]["run_id"] == "r1"

    def test_sqlite_save_runs_bulk(self, temp_db):
        """Bulk insert works."""
        results = [
            RunResult(run_id=f"bulk-{i}", suite_id="smoke", task_id="t1",
                       model_alias="agent-code", exit_status="success")
            for i in range(5)
        ]
        temp_db.save_runs(results)
        runs = temp_db.get_runs(suite_id="smoke")
        assert len(runs) == 5

    def test_sqlite_run_summary(self, temp_db):
        """get_run_summary produces correct stats."""
        temp_db.save_run(RunResult(run_id="r1", suite_id="smoke", task_id="t1",
                                    model_alias="agent-code", exit_status="success",
                                    ttft_ms=100.0, total_wall_ms=200.0,
                                    completion_tokens=50))
        temp_db.save_run(RunResult(run_id="r2", suite_id="smoke", task_id="t2",
                                    model_alias="agent-code", exit_status="error",
                                    error_message="timeout"))

        summary = temp_db.get_run_summary(suite_id="smoke")
        assert len(summary) == 1
        assert summary[0]["model_alias"] == "agent-code"
        assert summary[0]["tasks_run"] == 2
        assert summary[0]["passed"] == 1
        assert summary[0]["failed"] == 1


class TestResolveTaskDir:
    def test_resolve_task_dir(self, smoke_task_dir):
        """resolve_task_dir returns correct path."""
        result = resolve_task_dir("smoke")
        assert result.exists()
        assert result == smoke_task_dir

    def test_resolve_task_dir_not_found(self):
        """resolve_task_dir raises for unknown suite."""
        with pytest.raises(ValueError, match="not found"):
            resolve_task_dir("nonexistent_suite")
