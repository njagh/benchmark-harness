# Milestone 13 — Training-Data Export

## Goal

Turn benchmark results into future fine-tuning and preference datasets: SFT JSONL, DPO/ORPO preference pairs, regression test generation, and judge-labeled example export with manual review support and data hygiene filters.

## Phase

Phase D — Data flywheel (Milestone 1 of 4 in phase)

## Dependencies

- Milestone 1: SQLite storage with full run records
- Milestone 4: Basic scorers (for pass/fail determination)
- Milestone 5: Coding task runner (for code output extraction)
- Milestone 7: LLM judge integration (judge scores for labeling)
- Milestone 12: Report generator v2 (report data layer for queries)

---

## Subtasks

### 13.1 Implement SFT JSONL export

**File:** `src/bench_harness/storage/export.py` (new or extend)

**Class:** `SFTExporter`

**Methods:**
```python
class SFTExporter:
    """Exports successful benchmark examples to SFT training format."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)

    def export_openai_messages(
        self,
        output_path: str,
        min_score: float = 0.8,
        families: list[str] | None = None,
        models: list[str] | None = None,
        max_examples: int | None = None,
    ) -> int:
        """
        Export high-quality responses in OpenAI messages format.

        Output format (sft_openai_messages.jsonl):
        {
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ],
          "meta": {
            "task_id": "...",
            "model": "...",
            "score": 0.95,
            "family": "coding",
            "suite": "coding_smoke"
          }
        }

        Returns number of examples exported.
        """

    def _build_system_message(self, task: dict) -> str:
        """Construct system message from task prompt template or suite config."""

    def _build_user_message(self, task: dict, run: dict) -> str:
        """Construct user message from task prompt + context files."""

    def _build_assistant_message(self, run: dict) -> str:
        """Extract clean assistant response from raw_response or parsed_response."""
```

**Actions:**
- [ ] Implement OpenAI messages format export
- [ ] Filter by minimum score threshold
- [ ] Support family and model filtering
- [ ] Handle missing system prompts gracefully (use empty or default)
- [ ] Strip thinking tags or reasoning traces from assistant output (configurable)

### 13.2 Implement DPO/ORPO preference pair export

**File:** `src/bench_harness/storage/export.py` (extend)

**Class:** `PreferenceExporter`

**Methods:**
```python
class PreferenceExporter:
    """Exports pairwise comparison results as preference training data."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)

    def export_preference_pairs(
        self,
        output_path: str,
        min_score_delta: float = 0.1,
        families: list[str] | None = None,
    ) -> int:
        """
        Export chosen/rejected pairs from pairwise comparisons or score deltas.

        Output format (preference_chosen_rejected.jsonl):
        {
          "chosen": {
            "role": "assistant",
            "content": "..."
          },
          "rejected": {
            "role": "assistant",
            "content": "..."
          },
          "prompt": {
            "role": "user",
            "content": "..."
          },
          "meta": {
            "task_id": "...",
            "chosen_model": "max-brain",
            "rejected_model": "qwen-dense",
            "chosen_score": 0.92,
            "rejected_score": 0.65,
            "score_delta": 0.27,
            "family": "coding"
          }
        }

        Strategy 1: Use pairwise judge results (M7) directly.
        Strategy 2: For same task, different models — pair higher-scoring as chosen,
                   lower-scoring as rejected, when delta > min_score_delta.

        Returns number of pairs exported.
        """

    def _from_pairwise_judgments(self) -> list[dict]:
        """Extract preference pairs from judge pairwise comparison results."""

    def _from_score_comparison(self) -> list[dict]:
        """Derive preference pairs by comparing model scores on same task."""
```

**Actions:**
- [ ] Implement preference pair extraction from pairwise judge results
- [ ] Implement fallback score-based pair derivation
- [ ] Enforce minimum score delta to avoid near-equal pairs
- [ ] Deduplicate: one pair per task (pick highest-confidence comparison)

### 13.3 Implement regression task export

**File:** `src/bench_harness/storage/export.py` (extend)

**Class:** `RegressionExporter`

**Methods:**
```python
class RegressionExporter:
    """Exports failed examples as regression test tasks."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)

    def export_regression_tasks(
        self,
        output_path: str,
        max_score: float = 0.7,
        families: list[str] | None = None,
        models: list[str] | None = None,
    ) -> int:
        """
        Export failed or low-scoring tasks as regression test YAML.

        Output format (regression_tasks.yaml):
        # Auto-generated regression tasks from benchmark failures
        # Source: runs/2026-05-04-coding-smoke/benchmark.db
        # Generated: 2026-05-04T12:00:00Z

        tasks:
          - id: "regression.coding.patch_002_agent_code"
            source_run_id: "abc-123"
            source_task_id: "coding.patch_002"
            source_model: "agent-code"
            source_score: 0.45
            family: coding
            prompt: "..."
            context_files: [...]
            expected:
              type: rubric
              ...
            scoring:
              primary: unit_tests
            regression_note: "Model failed to apply correct patch"
            human_reviewed: false

        Returns number of regression tasks exported.
        """

    def _build_regression_task(self, run: dict, task: dict) -> dict:
        """Convert a failed run into a self-contained regression task."""
```

**Actions:**
- [ ] Implement regression task YAML generation
- [ ] Include source metadata (run_id, task_id, model, score)
- [ ] Include original prompt and expected output
- [ ] Add `human_reviewed: false` flag for manual curation
- [ ] Support filtering by family and model

### 13.4 Implement judge-labeled example export

**File:** `src/bench_harness/storage/export.py` (extend)

**Class:** `JudgeExport`

**Methods:**
```python
class JudgeExport:
    """Exports judge-scored examples for calibration and analysis."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)

    def export_judge_scores(
        self,
        output_path: str,
        families: list[str] | None = None,
    ) -> int:
        """
        Export all judge-labeled examples with rubric breakdowns.

        Output format (judge_scores.jsonl):
        {
          "task_id": "...",
          "model": "...",
          "run_id": "...",
          "judge_model": "...",
          "rubric_scores": {
            "correctness": 4,
            "completeness": 3,
            "specificity": 5,
            "safety": 4,
            "format_compliance": 5,
            "minimality": 3
          },
          "judge_reason": "...",
          "human_override": null,
          "human_reviewed": false
        }

        Returns number of examples exported.
        """
```

**Actions:**
- [ ] Implement judge score export with full rubric breakdown
- [ ] Include judge model identity and reason text
- [ ] Include human override and review status fields

### 13.5 Add manual review status fields

**File:** `src/bench_harness/storage/sqlite.py` (extend)

**New table:** `example_review`

```sql
CREATE TABLE example_review (
    run_id TEXT PRIMARY KEY,
    human_reviewed INTEGER NOT NULL DEFAULT 0,
    human_approved INTEGER,          -- 0 = reject, 1 = approve, NULL = not decided
    human_notes TEXT,
    review_date TEXT,
    reviewer TEXT,
    sft_approved INTEGER,            -- approved for SFT training
    preference_approved INTEGER,     -- approved for preference data
    regression_approved INTEGER,     -- approved as regression test
    created_at TEXT NOT NULL
);
```

**New methods:**
```python
def save_example_review(self, run_id: str, review: dict)
def get_reviewed_examples(self, approved: bool | None = None) -> list[dict]
def get_export_ready_examples(
    self,
    export_type: str,  # "sft" | "preference" | "regression"
    approved_only: bool = True,
) -> list[dict]
```

**CLI support in `src/bench_harness/cli.py`:**
```
bench-harness review
  --db       TEXT   Database path
  --run-id   TEXT   Specific run to review
  --approve  BOOLEAN  Mark as approved/rejected
  --notes    TEXT   Review notes
```

**Actions:**
- [ ] Add review table to schema
- [ ] Implement save/query methods
- [ ] Add CLI review command for manual curation
- [ ] Export filters respect review status

### 13.6 Add data hygiene filters

**File:** `src/bench_harness/storage/hygiene_filters.py` (new)

**Class:** `DataHygieneFilter`

**Methods:**
```python
class DataHygieneFilter:
    """Filters training data for quality and safety."""

    @staticmethod
    def filter_secrets(text: str) -> str:
        """
        Detect and redact potential secrets in exported data:
        - API keys (patterns: sk-, ghp_, xoxb-, etc.)
        - Passwords in command output
        - Private keys
        - Token strings
        """

    @staticmethod
    def filter_too_long(response: str, max_tokens: int = 8000) -> str | None:
        """Filter responses that are too long for training."""

    @staticmethod
    def filter_refusals(response: str) -> bool:
        """Detect and optionally filter model refusals ('I can't', 'I won't', etc.)."""

    @staticmethod
    def filter_duplicates(examples: list[dict], key_field: str = "task_id") -> list[dict]:
        """Remove duplicate examples, keeping highest-scoring version."""

    @staticmethod
    def filter_prompt_injection(response: str) -> bool:
        """Detect potential prompt injection patterns in model responses."""

    def apply_all(
        self,
        examples: list[dict],
        config: dict | None = None,
    ) -> list[dict]:
        """Apply all hygiene filters and return cleaned examples."""
```

**Filter config in `configs/hygiene.yaml` (new):**
```yaml
hygiene:
  max_response_tokens: 8000
  min_response_tokens: 20
  redact_secrets: true
  filter_refusals: false          # keep refusals — useful for training
  filter_duplicates: true
  dedup_key: task_id              # field used for deduplication
  secret_patterns:
    - "sk-[a-zA-Z0-9]{20,}"
    - "ghp_[a-zA-Z0-9]{36}"
    - "xoxb-[a-zA-Z0-9-]+"
    - "AKIA[0-9A-Z]{16}"
```

**Actions:**
- [ ] Implement all hygiene filter methods
- [ ] Create hygiene config file
- [ ] Wire filters into export pipeline
- [ ] Log filtered-out examples with reason codes

### 13.7 Implement main export CLI command

**File:** `src/bench_harness/cli.py` (extend)

**New CLI command:** `export`

```
bench-harness export
  --db            TEXT   Path to SQLite database           (required)
  --format        TEXT   Export format: sft | dpo | regression | judge | all
                      (default: all)
  --out           PATH   Output directory                   (default: export/YYYY-MM-DD)
  --min-score     FLOAT  Minimum score for SFT export       (default: 0.8)
  --max-score     FLOAT  Maximum score for regression export (default: 0.7)
  --families      TEXT   Task families to include            (default: all)
  --models        TEXT   Models to include                   (default: all)
  --approved-only BOOLEAN  Only export human-approved examples (default: false)
  --max-examples  INT    Maximum examples per format         (default: unlimited)
```

**Export workflow:**
1. Load database
2. Load hygiene config
3. For each requested format:
   - Query relevant examples
   - Apply hygiene filters
   - Apply review status filter
   - Write output file
4. Print summary: examples exported per format, examples filtered out

**Actions:**
- [ ] Implement export command with all arguments
- [ ] Wire all export classes into the command
- [ ] Apply hygiene filters before writing
- [ ] Print summary with counts and filter statistics

### 13.8 Implement export script

**File:** `scripts/export_training_data.py`

**Content:**
```python
#!/usr/bin/env python
"""Convenience script for full training data export."""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export benchmark data for training")
    parser.add_argument("--db", required=True, help="Path to benchmark SQLite database")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out or f"export/{Path(args.db).stem}"

    # Use CLI module
    from bench_harness.cli import app
    app(["export", "--db", args.db, "--out", out_dir, "--format", "all"])

if __name__ == "__main__":
    main()
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 13.9 Add analysis module for export statistics

**File:** `src/bench_harness/analysis/export_stats.py` (new)

**Functions:**
```python
def compute_export_statistics(db_path: str) -> dict:
    """
    Compute statistics about available training data:
    - Total high-quality examples (score > 0.8)
    - Total low-quality examples (score < 0.7)
    - Total pairwise comparison examples
    - Distribution by task family
    - Distribution by model
    - Estimated SFT dataset size
    - Estimated DPO dataset size
    """

def print_export_summary(db_path: str) -> None:
    """Print human-readable summary of exportable data."""
```

**Actions:**
- [ ] Implement statistics computation
- [ ] Add to CLI as `bench-harness export --stats` option

### 13.10 Add tests

**File:** `tests/test_export.py`

**Tests:**
- `test_sft_export_format` — exported lines are valid JSON with messages array
- `test_sft_export_score_filter` — only examples above min_score are exported
- `test_sft_export_family_filter` — only matching families are exported
- `test_preference_export_format` — exported lines have chosen/rejected fields
- `test_preference_export_delta_filter` — only pairs with sufficient delta are exported
- `test_preference_deduplication` — one pair per task
- `test_regression_export_format` — exported YAML is valid regression tasks
- `test_regression_export_score_filter` — only failing examples are exported
- `test_judge_export_format` — exported lines have rubric scores
- `test_hygiene_secret_redaction` — secrets are redacted from output
- `test_hygiene_filter_too_long` — oversized responses are filtered
- `test_hygiene_deduplication` — duplicates removed keeping highest score
- `test_review_table_round_trip` — save and retrieve review status
- `test_export_approved_only` — only approved examples pass filter

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixtures for synthetic databases with known scores
- [ ] Mock hygiene filter outputs for isolated testing

---

## Acceptance Criteria Checklist

- [ ] A failed benchmark task can become a regression test (regression_tasks.yaml)
- [ ] A high-quality response can become an SFT example (sft_openai_messages.jsonl)
- [ ] A model comparison can become preference data (preference_chosen_rejected.jsonl)
- [ ] Judge-labeled examples are exported with rubric breakdowns (judge_scores.jsonl)
- [ ] Manual review status can be recorded per example
- [ ] Data hygiene filters redact secrets and filter duplicates
- [ ] Export respects human approval status
- [ ] `bench-harness export --db <path> --format all` works end-to-end
- [ ] `pytest tests/test_export.py` passes

## Estimated Effort

3–4 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/storage/export.py` | To create |
| `src/bench_harness/storage/hygiene_filters.py` | To create |
| `src/bench_harness/storage/sqlite.py` | Extend |
| `src/bench_harness/analysis/export_stats.py` | To create |
| `src/bench_harness/cli.py` | Extend |
| `configs/hygiene.yaml` | To create |
| `scripts/export_training_data.py` | To create |
| `tests/test_export.py` | To create |
