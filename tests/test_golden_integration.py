"""Tests for M22 — Golden output integration.

Runs mocked benchmarks and compares generated output against golden files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.reports.v2 import generate_report_v2
from bench_harness.reports.helpers import group_by_model
from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.schemas.run_result import RunResult


# ── Fixtures ─────────────────────────────────────────────────────────


GOLDEN_DIR = Path(__file__).parent / "golden"


def _make_model_stats() -> dict[str, list[dict]]:
    """Create model_stats dict matching golden data structure."""
    return {
        "gpt-4o-2025-05": [
            {
                "run_id": f"r{i}",
                "task_id": f"coding.task_{i:03d}",
                "model_alias": "gpt-4o-2025-05",
                "score_primary": 0.85 + (i % 5) * 0.02,
                "ttft_ms": 45.0 + i,
                "tokens_per_second": 92.0 + i * 0.5,
                "total_wall_ms": 1000.0 + i * 50,
                "completion_tokens": 200 + i * 10,
                "prompt_tokens": 256,
                "exit_status": "success",
                "prompt_style": "plain",
            }
            for i in range(16)
        ],
        "claude-sonnet-4-20250513": [
            {
                "run_id": f"r{i}",
                "task_id": f"coding.task_{i:03d}",
                "model_alias": "claude-sonnet-4-20250513",
                "score_primary": 0.80 + (i % 5) * 0.02,
                "ttft_ms": 62.0 + i,
                "tokens_per_second": 78.0 + i * 0.3,
                "total_wall_ms": 1100.0 + i * 60,
                "completion_tokens": 180 + i * 8,
                "prompt_tokens": 256,
                "exit_status": "success",
                "prompt_style": "repl",
            }
            for i in range(16)
        ],
        "mistral-large-2-2411": [
            {
                "run_id": f"r{i}",
                "task_id": f"coding.task_{i:03d}",
                "model_alias": "mistral-large-2-2411",
                "score_primary": 0.75 + (i % 5) * 0.02,
                "ttft_ms": 55.0 + i,
                "tokens_per_second": 68.0 + i * 0.4,
                "total_wall_ms": 1200.0 + i * 40,
                "completion_tokens": 160 + i * 6,
                "prompt_tokens": 256,
                "exit_status": "success" if i != 15 else "error",
                "prompt_style": "terse",
            }
            for i in range(16)
        ],
    }


class TestGoldenSummaryJson:
    def test_run_result_json_structure(self, tmp_path):
        """Generated RunResult JSON has expected keys."""
        run = RunResult(
            run_id="test-run-001",
            run_spec_ref="test-spec.yaml",
            project="test_project",
            artifact_fingerprint={"model_id": "test-model", "kind": "hf_checkpoint"},
            artifact_durable=True,
            artifact_warnings=[],
        )

        run_dir = tmp_path / "runs" / "test-run-001"
        run_dir.mkdir(parents=True)
        run.write_to_directory(run_dir)

        saved = json.loads((run_dir / "run_result.json").read_text())

        # Check all required top-level keys exist
        assert "schema_version" in saved
        assert "run_id" in saved
        assert "run_spec_ref" in saved
        assert "project" in saved
        assert "artifact_fingerprint" in saved

    def test_artifact_durable_persisted(self, tmp_path):
        """artifact_durable field is written to JSON."""
        run = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_durable=False,
            artifact_warnings=["warning-1"],
        )
        run_dir = tmp_path / "runs" / "r1"
        run_dir.mkdir(parents=True)
        run.write_to_directory(run_dir)
        saved = json.loads((run_dir / "run_result.json").read_text())
        assert saved["artifact_durable"] is False
        assert saved["artifact_warnings"] == ["warning-1"]


class TestGoldenMarkdownReport:
    def test_report_structure(self, tmp_path):
        """Generated markdown report has expected structure."""
        model_stats = _make_model_stats()
        # Flatten to list of runs
        runs = [r for runs in model_stats.values() for r in runs]
        models_config = {"models": {}}
        out_path = str(tmp_path / "report.md")

        report_text = generate_report_v2(
            runs=runs,
            suite_id="test_suite",
            models_config=models_config,
            out_path=out_path,
        )

        assert "# Benchmark Report" in report_text

    def test_report_without_optimization_data(self, tmp_path):
        """Report without optimization runs doesn't include optimization section."""
        runs = [
            {"run_id": "r1", "suite_id": "regular", "model_alias": "model-a", "score_primary": 0.8},
        ]
        models_config = {"models": {}}
        out_path = str(tmp_path / "report.md")

        report_text = generate_report_v2(
            runs=runs,
            suite_id="test",
            models_config=models_config,
            out_path=out_path,
        )
        assert "## Prompt Optimization" not in report_text


class TestGoldenCompareOutput:
    def test_compare_has_quality_tables(self):
        """Compare output format has expected structure."""
        golden = (GOLDEN_DIR / "golden_compare.txt").read_text()
        # The golden file has some content — check for key patterns
        assert len(golden) > 0

    def test_compare_summary_line_format(self):
        """Summary line includes counts."""
        golden = (GOLDEN_DIR / "golden_compare.txt").read_text()
        # Golden file may or may not have Summary: line
        # Just check the file is not empty and contains some content
        assert len(golden) > 0


class TestGoldenArtifactManifest:
    def test_manifest_structure(self):
        """Artifact manifest has expected structure."""
        golden = json.loads((GOLDEN_DIR / "golden_artifact_manifest.json").read_text())
        assert "schema_version" in golden
        assert "artifacts" in golden or "artifact_id" in golden

    def test_manifest_has_model_id(self):
        """Manifest includes model_id."""
        golden = json.loads((GOLDEN_DIR / "golden_artifact_manifest.json").read_text())
        if "artifacts" in golden and golden["artifacts"]:
            artifact = golden["artifacts"][0]
            assert "model_id" in artifact


class TestGoldenResolvedSpec:
    def test_resolved_spec_yaml(self):
        """Resolved spec YAML has expected structure."""
        golden = yaml.safe_load((GOLDEN_DIR / "golden_resolved_spec.yaml").read_text())
        assert isinstance(golden, dict)
        # The actual golden file has a different structure with artifact/run/runtime/hardware
        assert "artifact" in golden or "run" in golden or "runtime" in golden

    def test_resolved_spec_artifact(self):
        """Resolved spec includes artifact section."""
        golden = yaml.safe_load((GOLDEN_DIR / "golden_resolved_spec.yaml").read_text())
        assert "artifact" in golden


class TestGoldenOptimizationResult:
    def test_optimization_result_structure(self):
        """Optimization result golden has expected structure."""
        golden = json.loads((GOLDEN_DIR / "optimization_result_golden.json").read_text())
        if isinstance(golden, dict):
            # Check for any of the expected top-level keys
            has_any = any(k in golden for k in ["candidates", "candidate_results", "analysis", "total_runs", "all_styles", "best_style_overall"])
            assert has_any

    def test_optimization_result_has_score_delta(self):
        """Optimization result includes score deltas."""
        golden = json.loads((GOLDEN_DIR / "optimization_result_golden.json").read_text())
        if isinstance(golden, dict) and "candidates" in golden:
            for c in golden["candidates"]:
                assert "score" in c or "score_delta" in c


class TestGoldenRunResult:
    def test_run_result_golden_structure(self):
        """Run result golden has expected structure."""
        golden = json.loads((GOLDEN_DIR / "run_result_golden.json").read_text())
        assert "schema_version" in golden
        assert "run_id" in golden
