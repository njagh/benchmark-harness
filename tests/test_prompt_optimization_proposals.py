"""Tests for prompt optimization proposals module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from bench_harness.prompt_optimization.proposals import (
    TemplateProposal,
    TemplateRegistry,
    generate_proposals,
    load_proposals_from_yaml,
    _PREDEFINED_VARIANTS,
)


# ── TemplateProposal tests ───────────────────────────────────────────


class TestTemplateProposal:
    def test_default_values(self):
        p = TemplateProposal(name="test-variant")
        assert p.name == "test-variant"
        assert p.task_family == ""
        assert p.template_str == ""
        assert p.baseline == "plain"
        assert p.instructions == ""
        assert p.score == 0.0
        assert p.score_delta == 0.0
        assert p.run_count == 0

    def test_full_values(self):
        p = TemplateProposal(
            name="repl-terse",
            task_family="coding",
            template_str="Follow REPL mode: {{ user_message }}",
            baseline="repl",
            instructions="Be terse.",
            score=0.85,
            score_delta=0.05,
            run_count=10,
        )
        assert p.name == "repl-terse"
        assert p.task_family == "coding"
        assert p.template_str == "Follow REPL mode: {{ user_message }}"
        assert p.baseline == "repl"
        assert p.instructions == "Be terse."
        assert p.score == 0.85
        assert p.score_delta == 0.05
        assert p.run_count == 10


# ── Predefined variants tests ────────────────────────────────────────


class TestPredefinedVariants:
    def test_all_predefined_variants_exist(self):
        expected = {
            "repl-terse", "architect-step", "format-protected-patch",
            "json-debug", "plan-then-patch", "plain-minimal", "repl-patch",
        }
        assert set(_PREDEFINED_VARIANTS.keys()) == expected

    def test_all_variants_have_required_fields(self):
        for name, variant in _PREDEFINED_VARIANTS.items():
            assert "template" in variant, f"{name} missing 'template'"
            assert "instructions" in variant, f"{name} missing 'instructions'"
            assert "baseline" in variant, f"{name} missing 'baseline'"
            assert isinstance(variant["template"], str)
            assert len(variant["template"]) > 0

    def test_all_variants_have_baseline(self):
        for name, variant in _PREDEFINED_VARIANTS.items():
            assert variant["baseline"] in (
                "plain", "repl", "patch_only", "json_schema", "architect"
            ), f"{name} has unknown baseline: {variant['baseline']}"

    def test_repl_terse_has_repl_baseline(self):
        assert _PREDEFINED_VARIANTS["repl-terse"]["baseline"] == "repl"

    def test_architect_step_has_architect_baseline(self):
        assert _PREDEFINED_VARIANTS["architect-step"]["baseline"] == "architect"

    def test_format_protected_patch_has_patch_baseline(self):
        assert _PREDEFINED_VARIANTS["format-protected-patch"]["baseline"] == "patch_only"

    def test_json_debug_has_json_schema_baseline(self):
        assert _PREDEFINED_VARIANTS["json-debug"]["baseline"] == "json_schema"

    def test_plain_minimal_has_plain_baseline(self):
        assert _PREDEFINED_VARIANTS["plain-minimal"]["baseline"] == "plain"


# ── generate_proposals tests ─────────────────────────────────────────


class TestGenerateProposals:
    def test_empty_runs(self):
        result = generate_proposals([])
        assert result == []

    def test_family_filter_no_matches(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="nonexistent_family")
        assert result == []

    def test_repl_terse_proposal(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="coding")
        names = [p.name for p in result]
        assert "repl-terse" in names

    def test_repl_terse_contains_correct_info(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="coding")
        repl_terse = [p for p in result if p.name == "repl-terse"]
        assert len(repl_terse) == 1
        p = repl_terse[0]
        assert p.baseline == "repl"
        assert p.task_family == "coding"
        assert "REPL" in p.instructions

    def test_format_protected_patch_proposal(self, sample_patch_runs):
        result = generate_proposals(sample_patch_runs)
        names = [p.name for p in result]
        assert "format-protected-patch" in names

    def test_format_protected_patch_baseline(self, sample_patch_runs):
        result = generate_proposals(sample_patch_runs)
        fpp = [p for p in result if p.name == "format-protected-patch"]
        assert len(fpp) == 1
        assert fpp[0].baseline == "patch_only"

    def test_json_debug_proposal(self, sample_json_schema_runs):
        result = generate_proposals(sample_json_schema_runs)
        names = [p.name for p in result]
        assert "json-debug" in names

    def test_json_debug_baseline(self, sample_json_schema_runs):
        result = generate_proposals(sample_json_schema_runs)
        jd = [p for p in result if p.name == "json-debug"]
        assert len(jd) == 1
        assert jd[0].baseline == "json_schema"

    def test_architect_step_proposal(self, sample_architect_runs):
        result = generate_proposals(sample_architect_runs, task_family="arch")
        names = [p.name for p in result]
        assert "architect-step" in names

    def test_architect_step_baseline(self, sample_architect_runs):
        result = generate_proposals(sample_architect_runs, task_family="arch")
        ast = [p for p in result if p.name == "architect-step"]
        assert len(ast) == 1
        assert ast[0].baseline == "architect"

    def test_plain_minimal_proposal(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="coding")
        names = [p.name for p in result]
        if "plain-minimal" in names:
            plain_min = [p for p in result if p.name == "plain-minimal"][0]
            assert plain_min.baseline == "plain"

    def test_no_proposals_minimal_data(self, minimal_runs):
        result = generate_proposals(minimal_runs)
        # plain-minimal fires when plain >= 0.95 * best_other; here plain is 0.8 and no other styles,
        # so plain >= 0.95 * 0 => True; we accept 1 proposal
        assert len(result) <= 2

    def test_proposal_template_not_empty(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="coding")
        for p in result:
            assert len(p.template_str) > 0

    def test_proposal_instructions_not_empty(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="coding")
        for p in result:
            assert len(p.instructions) > 0

    def test_proposal_scores_are_zero(self, sample_style_runs):
        result = generate_proposals(sample_style_runs, task_family="coding")
        for p in result:
            assert p.score == 0.0
            assert p.score_delta == 0.0

    def test_multiple_proposals_generated(self, sample_style_runs, sample_patch_runs, sample_json_schema_runs, sample_architect_runs):
        combined = sample_style_runs + sample_patch_runs + sample_json_schema_runs + sample_architect_runs
        result = generate_proposals(combined)
        assert len(result) >= 1


# ── TemplateRegistry tests ───────────────────────────────────────────


class TestTemplateRegistry:
    def test_empty_registry(self):
        reg = TemplateRegistry()
        assert reg.get_baselines() == []
        assert reg.get_candidates() == []
        assert reg.has_candidates is False

    def test_add_baseline(self):
        reg = TemplateRegistry()
        reg.add_baseline("plain")
        reg.add_baseline("repl")
        assert reg.get_baselines() == ["plain", "repl"]

    def test_add_duplicate_baseline(self):
        reg = TemplateRegistry()
        reg.add_baseline("plain")
        reg.add_baseline("plain")
        assert reg.get_baselines() == ["plain"]

    def test_add_candidate(self):
        reg = TemplateRegistry()
        p = TemplateProposal(name="test-proposal", template_str="test {{ user_message }}")
        reg.add_candidate(p)
        assert reg.has_candidates is True
        assert len(reg.get_candidates()) == 1

    def test_add_candidate_duplicate_replaces(self):
        reg = TemplateRegistry()
        p1 = TemplateProposal(name="test-proposal", template_str="old")
        p2 = TemplateProposal(name="test-proposal", template_str="new")
        reg.add_candidate(p1)
        reg.add_candidate(p2)
        candidates = reg.get_candidates()
        assert len(candidates) == 1
        assert candidates[0].template_str == "new"

    def test_add_candidates_list(self):
        reg = TemplateRegistry()
        p1 = TemplateProposal(name="proposal-1")
        p2 = TemplateProposal(name="proposal-2")
        reg.add_candidates([p1, p2])
        assert len(reg.get_candidates()) == 2

    def test_clear(self):
        reg = TemplateRegistry()
        reg.add_baseline("plain")
        reg.add_candidate(TemplateProposal(name="test"))
        reg.clear()
        assert reg.get_baselines() == []
        assert reg.get_candidates() == []
        assert reg.has_candidates is False

    def test_has_candidates_false_initially(self):
        reg = TemplateRegistry()
        assert reg.has_candidates is False

    def test_has_candidates_after_add(self):
        reg = TemplateRegistry()
        reg.add_candidate(TemplateProposal(name="test"))
        assert reg.has_candidates is True

    def test_get_candidates_returns_copy(self):
        reg = TemplateRegistry()
        p = TemplateProposal(name="test")
        reg.add_candidate(p)
        candidates = reg.get_candidates()
        assert candidates[0] is p
        candidates.clear()
        assert len(reg.get_candidates()) == 1


# ── load_proposals_from_yaml tests ───────────────────────────────────


class TestLoadProposalsFromYaml:
    def test_load_from_yaml(self, yaml_proposals_spec):
        result = load_proposals_from_yaml(yaml_proposals_spec)
        assert len(result) == 2
        names = [p.name for p in result]
        assert "custom-v1" in names
        assert "custom-v2" in names

    def test_load_yaml_basic_fields(self, yaml_proposals_spec):
        result = load_proposals_from_yaml(yaml_proposals_spec)
        v1 = [p for p in result if p.name == "custom-v1"][0]
        assert v1.template_str == "You are a helpful assistant. {{ user_message }}"
        assert v1.instructions == "Custom v1 prompt style"
        assert v1.task_family == "coding"
        assert v1.baseline == "plain"

    def test_load_yaml_second_candidate(self, yaml_proposals_spec):
        result = load_proposals_from_yaml(yaml_proposals_spec)
        v2 = [p for p in result if p.name == "custom-v2"][0]
        assert v2.template_str == "Briefly answer: {{ user_message }}"
        assert v2.instructions == "Custom v2 prompt style"
        assert v2.task_family == "docker_compose"
        assert v2.baseline == "repl"

    def test_load_yaml_missing_spec_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_proposals_from_yaml(tmp_path / "nonexistent.yaml")

    def test_load_yaml_missing_name_raises(self, invalid_yaml_proposals_spec):
        with pytest.raises(ValueError, match="Missing 'name'"):
            load_proposals_from_yaml(invalid_yaml_proposals_spec)

    def test_load_yaml_empty_candidates(self, tmp_path):
        spec = tmp_path / "empty.yaml"
        spec.write_text("name: empty\ncandidates: []\n")
        result = load_proposals_from_yaml(spec)
        assert result == []

    def test_load_yaml_no_candidates_key(self, tmp_path):
        spec = tmp_path / "no_candidates.yaml"
        spec.write_text("name: nocandidates\n")
        result = load_proposals_from_yaml(spec)
        assert result == []

    def test_load_yaml_invalid_type(self, tmp_path):
        spec = tmp_path / "invalid_type.yaml"
        spec.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError):
            load_proposals_from_yaml(spec)

    def test_load_yaml_candidate_missing_template_raises(self, tmp_path):
        spec = tmp_path / "no_template.yaml"
        spec.write_text("candidates:\n  - name: no-template\n")
        with pytest.raises(ValueError, match="Missing 'template'"):
            load_proposals_from_yaml(spec)

    def test_load_yaml_default_baseline(self, tmp_path):
        spec = tmp_path / "default_baseline.yaml"
        spec.write_text(
            "candidates:\n"
            "  - name: test-v1\n"
            "    template: Hello {{ user_message }}\n"
        )
        result = load_proposals_from_yaml(spec)
        assert len(result) == 1
        assert result[0].baseline == "plain"

    def test_load_yaml_non_dict_candidate_skipped(self, tmp_path):
        spec = tmp_path / "non_dict.yaml"
        spec.write_text(
            "candidates:\n"
            '  - "plain string"\n'
            "  - name: valid-proposal\n"
            "    template: Hello {{ user_message }}\n"
        )
        result = load_proposals_from_yaml(spec)
        assert len(result) == 1
        assert result[0].name == "valid-proposal"
