# Milestone 24 — Scorer Test Suite Deep Dive

## Goal

Achieve >80% test coverage on the four lowest-coverage scorer modules:
`pairwise`, `llm_judge`, `config_valid`, and `command_safety`.
These scorers represent 606 missed lines — the single largest coverage gap
in the codebase — and are feature-complete, testable, and low-risk to cover.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- M17–M23: all infrastructure, tests, and CLI complete

## Acceptance Criteria (Definition of Done)

- `scorers/pairwise.py` coverage >= 80%
- `scorers/llm_judge.py` coverage >= 80%
- `scorers/config_valid.py` coverage >= 80%
- `scorers/command_safety.py` coverage >= 80%
- All existing tests continue to pass (no regressions)
- Total test count increases by >= 130 new tests

## Subtasks

### 24.1 PairwiseScorer Tests

File: `tests/test_scorers_pairwise.py` — minimum 30 tests

Covers:
- `PairwiseScorer.__init__` with OpenAI client, base_url+api_key, no-args
- `_load_rubric` with existing rubric, missing rubric (fallback path)
- `_extract_comparison_data` with JSON string, dict, empty strings, missing fields
- `_safe_parse_pairwise` — all winner values, missing winner, bad confidence,
  invalid `dimension_comparison`, non-dict input
- `_build_pairwise_score_result` with winner a/b/tie, scoring math
- `score()` — happy path (mocked client), error paths (client=None, API error, parse failure)
- `_score_pairwise` — async method with valid JSON, invalid JSON, empty response, error response

### 24.2 LLMJudgeScorer Tests

File: `tests/test_scorers_llm_judge.py` — minimum 40 tests

Covers:
- `load_rubric` with valid name, missing name (KeyError), custom path
- `LLMJudgeScorer.__init__` with all constructor params
- `_fallback_rubric` — returns correct structure
- `_safe_parse_dimension_scores` — dict, non-dict, missing dimensions, mixed score types
- `_compute_weighted_score` — multiple dimensions, varying weights, missing dim
- `_build_score_result` — with/without stddevs, with/without explanations, with error
- `_score_single_round` — mocked, `_score_with_consistency` — multiple rounds and stddev
- `score()` — single round, self-consistency mode (>1 rounds), all error paths

### 24.3 ConfigValidatorScorer Tests

File: `tests/test_scorers_config_valid.py` — minimum 25 tests

Covers:
- `validate_task` with/without config_type
- `_validate_yaml_config` — valid YAML, invalid YAML, missing required fields, pattern matching
- `_validate_json_config` — valid JSON, invalid JSON
- `_validate_systemd_config` — unit/service/install sections, missing ExecStart
- `_validate_ini_config` — sections, missing required fields, parse error
- `score()` — composite scoring: format_score * 0.5 + fields_score * 0.3 + pattern_score * 0.2

### 24.4 CommandSafetyScorer Tests

File: `tests/test_scorers_command_safety.py` — minimum 35 tests

Covers:
- `_extract_commands` from code blocks, inline commands, mixed text, empty input
- `_extract_single_command` with known_cmds, pipes, backtick commands, non-command text
- `_classify_command` for all 8 risk categories (broad_deletion, permission_escalation,
  secret_exposure, network_install, force_push, history_rewrite, destructive_docker,
  blind_file_overwrite)
- `_classify_command` with all safe patterns
- `_check_dry_run` override logic
- `CommandSafetyScorer.score()` — happy path (all safe), unsafe detected, zero commands (score=1.0), --dry-run override

## Files Created

- `tests/test_scorers_pairwise.py`
- `tests/test_scorers_llm_judge.py`
- `tests/test_scorers_config_valid.py`
- `tests/test_scorers_command_safety.py`

## Definition of Done Checklist

When all of the following are true:

- [ ] pairwise.py >= 80% coverage
- [ ] llm_judge.py >= 80% coverage
- [ ] config_valid.py >= 80% coverage
- [ ] command_safety.py >= 80% coverage
- [ ] All existing 1331+ tests pass
- [ ] >= 130 new tests added
- [ ] No code changes required (tests only)
- [ ] Commit message references M24 acceptance criteria
