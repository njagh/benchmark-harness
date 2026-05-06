# Benchmark Harness — Local LLM Quality Evaluation

Evaluates response quality, coding ability, instruction following, and performance tradeoffs across locally served models. Produces actionable recommendations for choosing the right model and serving configuration for real workflows.

## Current Status

**Phase:** Milestones M1–M12 complete. M13 (training-data export) next.

The `ROADMAP.md` defines 15 milestones across 4 phases. Detailed execution plans with library integration notes are in `milestones/`. Virtual environment and core dependencies are set up.

## Quick Start

```bash
source .venv/bin/activate
python -m bench_harness --help
```

For analysis/dashboard work:
```bash
pip install -e ".[analysis]"
```

For coding benchmark integration (bigcode-tools, evalplus):
```bash
pip install -e ".[coding]"
```

For long-context benchmarks (GPU monitoring, tokenizers):
```bash
pip install -e ".[long-context]"
```

## Milestones

| # | Milestone | Phase | Status |
|---|---|---|---|
| 1 | Project Bootstrap | A — Minimal harness | **Done** |
| 2 | Task Schema and Registry | B — Coding usefulness | **Done** |
| 3 | Timing and Token Metrics | A — Minimal harness | **Done** |
| 4 | Basic Scorers | B — Coding usefulness | **Done** |
| 5 | Coding Task Runner | B — Coding usefulness | **Done** |
| 6 | Local Coding-Agent Suite v1 | B — Coding usefulness | **Done** |
| 7 | LLM Judge Integration | D — Data flywheel | **Done** |
| 8 | Prompt Style Comparison | B — Coding usefulness | **Done** |
| 9 | Long-Context Benchmark Suite | C — Deep comparison | **Done** |
| 10 | Quantization Comparison Suite | C — Deep comparison | **Done** |
| 11 | Agent Safety and Command Discipline | C — Deep comparison | Not started |
| 12 | Report Generator v2 | C — Deep comparison | **Done** |
| 13 | Training-Data Export | D — Data flywheel | Not started |
| 14 | Dashboard / Analysis Notebook | D — Data flywheel | Not started |
| 15 | CI / Regression Mode | D — Data flywheel | Not started |

## Build Order

1. **Phase A** (M1, M3) — Run smoke tests against local models with timing
2. **Phase B** (M2, M4–M6, M8) — Real coding tasks, scorers, prompt comparison
3. **Phase C** (M9–M12) — Long context, quantization, safety, reports
4. **Phase D** (M7, M13–M15) — Judge scoring, training data export, CI

## Target Command

```bash
python -m bench_harness run \
  --suite smoke,coding_smoke \
  --models agent-code,qwen-dense,max-brain \
  --runs 3
```

## Leveraged Libraries

| Library | Used By | Purpose |
|---|---|---|
| `openai` | M1 | OpenAI-compatible client for local endpoints |
| `sqlite-utils` | M1 | SQLite storage, bulk insert, CSV export |
| `typer` / `rich` | M1 | CLI framework and terminal output |
| `pydantic` | M1 | Config and task schema validation |
| `jsonschema` | M4 | JSON schema validation scorer |
| `jinja2` | M2 | Prompt template rendering |
| `unidiff` | M5 | Unified diff parsing for patches |
| `bigcode-tools` | M5 (opt) | HumanEval/MBPP execution infrastructure |
| `evalplus` | M5 (opt) | Deobfuscated coding test suites |
| `lm-eval` | M6 (opt) | Public benchmark integration (MMLU, GPQA, BBH, etc.) |
| `lm-evaluation-harness` | M6 (opt) | Standard eval harness |
| `shellingham` / `shlex` | M11 | Shell command parsing |
| `pynvml` | M9 (opt) | GPU memory/utilization monitoring |
| `transformers` | M9 (opt) | Tokenizer for context packing |
| `duckdb` | M12 (opt) | SQL queries on SQLite for reports |
| `matplotlib` / `seaborn` | M12/M14 (opt) | Chart generation |
| `great-tables` | M12 (opt) | Polished report tables |

## Data Access

Per `STORAGE_PLAN.md`:

- External eval data lives in `/mnt/datasets-big/evals/` — pinned local copies with manifests
- HF cache at `/mnt/datasets-big/hf-cache/` — set `HF_HOME` etc. before any HF downloads
- No live streaming from HuggingFace during measured runs
- Dataset registry in `configs/datasets.yaml`

## Details

- `ROADMAP.md` — Full project specification and design decisions
- `STORAGE_PLAN.md` — Dataset layout, HF cache config, DGX Spark access patterns
- `milestones/` — Per-milestone execution plans with subtasks, file specs, library integration notes, and acceptance criteria
- `pyproject.toml` — Dependencies (core + optional groups)
