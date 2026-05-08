"""Shared test fixtures for the benchmark harness test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_storage_root(tmp_path: Path) -> Path:
    """Temporary storage directory with runs/ and benchmark.db structure."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    db_path = runs_dir / "benchmark.db"
    db_path.write_text("")
    return tmp_path


@pytest.fixture
def fake_run_spec() -> dict[str, Any]:
    """A minimal RunSpec dict for testing (not pydantic, just a plain dict)."""
    return {
        "name": "test-model-smoke",
        "project": "test_project",
        "tags": ["test", "smoke"],
        "artifact": {
            "kind": "hf_checkpoint",
            "path": "/models/test-model",
            "model_id": "test-model",
        },
        "runtime": {
            "kind": "openai_compatible",
            "launch": "existing",
        },
        "workload": {
            "prompt_suite": "smoke",
            "max_tokens": 256,
            "temperature": 0.0,
            "num_runs": 1,
            "concurrency": 1,
        },
        "hardware": {
            "profile": "default",
        },
    }


@pytest.fixture
def mock_server() -> dict[str, Any]:
    """Mock LiteLLM server response structure for testing without real API calls."""
    return {
        "status": "success",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response for testing.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "response_time_ms": 150.0,
        "ttft_ms": 50.0,
    }


@pytest.fixture
def sample_task() -> dict[str, Any]:
    """A sample task dict for testing with all common fields."""
    return {
        "id": "test.coding.task_001",
        "family": "coding",
        "name": "Simple coding task",
        "scoring": {"primary": "exact_match"},
        "expected": {"type": "exact", "answer": "Hello, World!"},
        "prompt": "Write a hello world program.",
    }


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Temporary project directory with storage structure (runs/, tasks/)."""
    project_dir = tmp_path / "project"
    (project_dir / "runs").mkdir(parents=True)
    (project_dir / "tasks").mkdir(parents=True)
    (project_dir / "tasks" / "smoke").mkdir()
    (project_dir / "tasks" / "coding").mkdir()
    db_path = project_dir / "runs" / "benchmark.db"
    db_path.write_text("")
    return project_dir


@pytest.fixture
def sample_style_runs() -> list[dict[str, Any]]:
    """Sample run data with prompt_style for analysis testing."""
    return [
        {
            "run_id": "run-001",
            "task_id": "test.coding.task_001",
            "suite_id": "suite-a",
            "model_alias": "test-model",
            "prompt_style": "plain",
            "score_primary": 0.8,
            "score_secondary": None,
            "completion_tokens": 50,
            "total_wall_ms": 100,
            "ttft_ms": 20,
            "exit_status": "success",
        },
        {
            "run_id": "run-002",
            "task_id": "test.coding.task_002",
            "suite_id": "suite-a",
            "model_alias": "test-model",
            "prompt_style": "plain",
            "score_primary": 0.7,
            "score_secondary": None,
            "completion_tokens": 60,
            "total_wall_ms": 120,
            "ttft_ms": 25,
            "exit_status": "success",
        },
        {
            "run_id": "run-003",
            "task_id": "test.coding.task_001",
            "suite_id": "suite-a",
            "model_alias": "test-model",
            "prompt_style": "repl",
            "score_primary": 0.9,
            "score_secondary": None,
            "completion_tokens": 150,
            "total_wall_ms": 300,
            "ttft_ms": 40,
            "exit_status": "success",
        },
        {
            "run_id": "run-004",
            "task_id": "test.coding.task_002",
            "suite_id": "suite-a",
            "model_alias": "test-model",
            "prompt_style": "repl",
            "score_primary": 0.85,
            "score_secondary": None,
            "completion_tokens": 160,
            "total_wall_ms": 320,
            "ttft_ms": 45,
            "exit_status": "success",
        },
        {
            "run_id": "run-005",
            "task_id": "test.coding.task_001",
            "suite_id": "suite-a",
            "model_alias": "test-model",
            "prompt_style": "architect",
            "score_primary": 0.75,
            "score_secondary": None,
            "completion_tokens": 200,
            "total_wall_ms": 500,
            "ttft_ms": 60,
            "exit_status": "success",
        },
        {
            "run_id": "run-006",
            "task_id": "test.coding.task_003",
            "suite_id": "suite-a",
            "model_alias": "test-model",
            "prompt_style": "plain",
            "score_primary": 0.85,
            "score_secondary": None,
            "completion_tokens": 45,
            "total_wall_ms": 90,
            "ttft_ms": 15,
            "exit_status": "success",
        },
    ]


@pytest.fixture
def sample_patch_runs() -> list[dict[str, Any]]:
    """Sample run data with patch_only style and format_compliance failures."""
    return [
        {
            "run_id": "patch-001",
            "task_id": "test.patching.task_001",
            "suite_id": "suite-patch",
            "model_alias": "test-model",
            "prompt_style": "patch_only",
            "score_primary": 0.5,
            "score_secondary": json.dumps({"format_compliance": {"passed": False, "reason": "extra text"}}),
            "completion_tokens": 100,
            "total_wall_ms": 200,
            "exit_status": "success",
        },
        {
            "run_id": "patch-002",
            "task_id": "test.patching.task_002",
            "suite_id": "suite-patch",
            "model_alias": "test-model",
            "prompt_style": "patch_only",
            "score_primary": 0.6,
            "score_secondary": json.dumps({"format_compliance": {"passed": False, "reason": "no diff"}}),
            "completion_tokens": 120,
            "total_wall_ms": 210,
            "exit_status": "success",
        },
        {
            "run_id": "patch-003",
            "task_id": "test.patching.task_003",
            "suite_id": "suite-patch",
            "model_alias": "test-model",
            "prompt_style": "patch_only",
            "score_primary": 0.4,
            "score_secondary": json.dumps({"format_compliance": {"passed": True}}),
            "completion_tokens": 90,
            "total_wall_ms": 180,
            "exit_status": "success",
        },
        {
            "run_id": "patch-004",
            "task_id": "test.patching.task_004",
            "suite_id": "suite-patch",
            "model_alias": "test-model",
            "prompt_style": "patch_only",
            "score_primary": 0.55,
            "score_secondary": json.dumps({"format_compliance": {"passed": False, "reason": "wrong format"}}),
            "completion_tokens": 110,
            "total_wall_ms": 190,
            "exit_status": "success",
        },
    ]


@pytest.fixture
def sample_json_schema_runs() -> list[dict[str, Any]]:
    """Sample run data with json_schema style showing low scores but high tokens."""
    return [
        {
            "run_id": "json-001",
            "task_id": "test.json.task_001",
            "suite_id": "suite-json",
            "model_alias": "test-model",
            "prompt_style": "json_schema",
            "score_primary": 0.3,
            "score_secondary": None,
            "completion_tokens": 300,
            "total_wall_ms": 400,
            "exit_status": "success",
        },
        {
            "run_id": "json-002",
            "task_id": "test.json.task_002",
            "suite_id": "suite-json",
            "model_alias": "test-model",
            "prompt_style": "json_schema",
            "score_primary": 0.4,
            "score_secondary": None,
            "completion_tokens": 350,
            "total_wall_ms": 420,
            "exit_status": "success",
        },
    ]


@pytest.fixture
def sample_architect_runs() -> list[dict[str, Any]]:
    """Sample run data with architect style that is good but slow."""
    return [
        {
            "run_id": "arch-001",
            "task_id": "test.arch.task_001",
            "suite_id": "suite-arch",
            "model_alias": "test-model",
            "prompt_style": "architect",
            "score_primary": 0.9,
            "score_secondary": None,
            "completion_tokens": 250,
            "total_wall_ms": 1000,
            "exit_status": "success",
        },
        {
            "run_id": "arch-002",
            "task_id": "test.arch.task_002",
            "suite_id": "suite-arch",
            "model_alias": "test-model",
            "prompt_style": "architect",
            "score_primary": 0.88,
            "score_secondary": None,
            "completion_tokens": 260,
            "total_wall_ms": 1100,
            "exit_status": "success",
        },
        {
            "run_id": "arch-003",
            "task_id": "test.arch.task_003",
            "suite_id": "suite-arch",
            "model_alias": "test-model",
            "prompt_style": "plain",
            "score_primary": 0.7,
            "score_secondary": None,
            "completion_tokens": 50,
            "total_wall_ms": 100,
            "exit_status": "success",
        },
    ]


@pytest.fixture
def minimal_runs() -> list[dict[str, Any]]:
    """Minimal run data — one style, one task, one score."""
    return [
        {
            "run_id": "min-001",
            "task_id": "test.family.task_001",
            "suite_id": "suite-minimal",
            "model_alias": "test-model",
            "prompt_style": "plain",
            "score_primary": 0.8,
            "completion_tokens": 50,
            "total_wall_ms": 100,
        },
    ]


@pytest.fixture
def empty_runs() -> list[dict[str, Any]]:
    """Empty run list for edge case testing."""
    return []


@pytest.fixture
def runs_with_missing_fields() -> list[dict[str, Any]]:
    """Run data with missing optional fields to test robustness."""
    return [
        {
            "run_id": "partial-001",
            "task_id": "test.family.task_001",
            "suite_id": "suite-partial",
            "prompt_style": None,
            "score_primary": None,
        },
        {
            "run_id": "partial-002",
            "task_id": "test.family.task_002",
            "suite_id": "suite-partial",
            "model_alias": "test-model",
        },
        {
            "run_id": "partial-003",
            "task_id": "test.family.task_003",
            "prompt_style": "plain",
            "score_primary": 0.9,
        },
    ]


@pytest.fixture
def yaml_proposals_spec(tmp_path: Path) -> Path:
    """YAML spec file with custom template proposals."""
    spec_content = """
name: custom-proposals
baselines:
  - plain
  - repl
candidates:
  - name: custom-v1
    instructions: "Custom v1 prompt style"
    template: "You are a helpful assistant. {{ user_message }}"
    task_family: coding
    baseline: plain
  - name: custom-v2
    instructions: "Custom v2 prompt style"
    template: "Briefly answer: {{ user_message }}"
    task_family: docker_compose
    baseline: repl
"""
    spec_path = tmp_path / "proposals.yaml"
    spec_path.write_text(spec_content)
    return spec_path


@pytest.fixture
def invalid_yaml_proposals_spec(tmp_path: Path) -> Path:
    """YAML spec file missing required fields."""
    spec_content = """
name: invalid-proposals
baselines:
  - plain
candidates:
  - instructions: "Missing name and template"
"""
    spec_path = tmp_path / "invalid_proposals.yaml"
    spec_path.write_text(spec_content)
    return spec_path
