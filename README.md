# Benchmark Harness — Local LLM Quality Evaluation

Evaluates response quality, coding ability, instruction following, and performance tradeoffs across locally served models. Produces actionable recommendations for choosing the right model and serving configuration for real workflows.

## Current Status

**Phase:** Planning complete. Implementation not yet started.

The `ROADMAP.md` defines 15 milestones across 4 phases. Detailed execution plans for all milestones have been generated under `milestones/`.

## Milestones

| # | Milestone | Phase | Status |
|---|---|---|---|
| 1 | Project Bootstrap | A — Minimal harness | Not started |
| 2 | Task Schema and Registry | B — Coding usefulness | Not started |
| 3 | Timing and Token Metrics | A — Minimal harness | Not started |
| 4 | Basic Scorers | B — Coding usefulness | Not started |
| 5 | Coding Task Runner | B — Coding usefulness | Not started |
| 6 | Local Coding-Agent Suite v1 | B — Coding usefulness | Not started |
| 7 | LLM Judge Integration | D — Data flywheel | Not started |
| 8 | Prompt Style Comparison | B — Coding usefulness | Not started |
| 9 | Long-Context Benchmark Suite | C — Deep comparison | Not started |
| 10 | Quantization Comparison Suite | C — Deep comparison | Not started |
| 11 | Agent Safety and Command Discipline | C — Deep comparison | Not started |
| 12 | Report Generator v2 | C — Deep comparison | Not started |
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

## Details

- `ROADMAP.md` — Full project specification and design decisions
- `milestones/` — Per-milestone execution plans with subtasks, file specs, and acceptance criteria
