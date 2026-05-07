"""Analysis module for interactive notebook dashboards.

Provides query builders and data transformers that load benchmark results
from SQLite into pandas DataFrames for interactive analysis in Jupyter.

Usage in a notebook:
    from bench_harness.analysis import BenchDB
    db = BenchDB("runs/2026-05-06-coding_benchmark/benchmark.db")
    df = db.load_runs()
    df.plot.scatter(x="tokens_per_second", y="score_primary")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlite_utils import Database

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


__all__ = ["BenchDB", "runs_to_df", "runs_to_duck", "timing_summary_df",
           "quantization_comparison_df", "context_degradation_df",
           "score_variance_df", "style_comparison_df", "failures_df",
           "identify_mismatches"]


def _ensure_pandas():
    if pd is None:
        raise RuntimeError(
            "pandas not installed. Install with: pip install bench-harness[analysis]"
        )


def runs_to_df(db_path: str | Path) -> pd.DataFrame:
    """Load all runs from SQLite into a pandas DataFrame.

    Parses JSON text fields (score_secondary, judge_dimensions) back to dicts.
    """
    _ensure_pandas()
    db = Database(str(db_path), memory=False)
    rows = list(db["runs"].rows)
    json_fields = {"score_secondary", "judge_dimensions"}
    for row in rows:
        for field in json_fields:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    pass
    return pd.DataFrame(rows)


def runs_to_duck(db_path: str | Path) -> str:
    """Load runs into DuckDB for faster analysis.

    Returns the name of the registered table.
    """
    _ensure_pandas()
    if duckdb is None:
        raise RuntimeError(
            "duckdb not installed. Install with: pip install bench-harness[analysis]"
        )
    df = runs_to_df(db_path)
    con = duckdb.connect()
    con.register("runs", df)
    return "runs"


# ── DataFrame builders for common analyses ──────────────────────────────


def timing_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model timing aggregates: mean/min/max/p95 for TTFT, decode, wall, tps."""
    _ensure_pandas()
    numeric_cols = ["ttft_ms", "prefill_ms", "decode_ms", "total_wall_ms",
                    "tokens_per_second", "tokens_per_second_total",
                    "prompt_tokens", "completion_tokens", "total_tokens"]
    agg: dict[str, Any] = {"run_count": ("run_id", "count")}
    for col in numeric_cols:
        agg[col] = [f"{col}_mean", f"{col}_min", f"{col}_max", f"{col}_p95"]

    numeric_dfs = {c: df[c] for c in numeric_cols if c in df.columns}
    # Custom p95 aggregation
    p95_fn = lambda x: x.quantile(0.95)  # noqa: E731

    group = df.groupby("model_alias")
    out = pd.DataFrame({
        "model_alias": group.ngroups.keys() if hasattr(group.ngroups, "keys") else group.groups.keys(),
    })
    # Simpler approach: compute directly
    out = df.groupby("model_alias").agg({
        "run_id": "count",
        "ttft_ms": ["mean", "min", "max", p95_fn],
        "prefill_ms": "mean",
        "decode_ms": ["mean", "min", "max"],
        "total_wall_ms": ["mean", "min", "max"],
        "tokens_per_second": ["mean", "min", "max"],
        "tokens_per_second_total": ["mean"],
        "prompt_tokens": "mean",
        "completion_tokens": "mean",
    }).reset_index()
    out.columns = ["model_alias", "run_count",
                     "ttft_mean", "ttft_min", "ttft_max", "ttft_p95",
                     "prefill_mean",
                     "decode_mean", "decode_min", "decode_max",
                     "wall_mean", "wall_min", "wall_max",
                     "tps_mean", "tps_min", "tps_max",
                     "tps_total_mean",
                     "prompt_tokens_mean", "completion_tokens_mean"]
    out = out.sort_values("model_alias").reset_index(drop=True)
    return out


def quantization_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compare scores by quantization level per model.

    Returns columns: model_alias, quantization, avg_score, run_count,
                      score_std, score_min, score_max.
    """
    _ensure_pandas()
    scored = df[df["score_primary"].notna()].copy()
    scored["quantization"] = scored["quantization"].fillna("unquantized")
    group = scored.groupby(["model_alias", "quantization"])
    out = group["score_primary"].agg(["mean", "count", "std", "min", "max"]).reset_index()
    out.columns = ["model_alias", "quantization", "avg_score", "run_count",
                    "score_std", "score_min", "score_max"]
    out["score_std"] = out["score_std"].fillna(0.0)
    return out.sort_values(["model_alias", "quantization"]).reset_index(drop=True)


def context_degradation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Quality vs context length analysis.

    Returns: model_alias, context_bucket, avg_score, run_count, score_std, avg_tps.
    """
    _ensure_pandas()
    scored = df[df["score_primary"].notna()].copy()
    bucketed = scored.copy()
    if "prompt_tokens" in bucketed.columns:
        def _bucket(n):
            if n is None:
                return "unknown"
            if n <= 1024:
                return "1k"
            if n <= 4096:
                return "4k"
            if n <= 16384:
                return "16k"
            if n <= 65536:
                return "64k"
            return "256k+"
        bucketed["context_bucket"] = bucketed["prompt_tokens"].apply(_bucket)
    else:
        bucketed["context_bucket"] = bucketed.get("context_tokens", "unknown")

    group = bucketed.groupby(["model_alias", "context_bucket"])
    out = group.agg(
        avg_score=("score_primary", "mean"),
        run_count=("score_primary", "count"),
        score_std=("score_primary", "std"),
        avg_tps=("tokens_per_second", "mean"),
    ).reset_index()
    out["score_std"] = out["score_std"].fillna(0.0)
    return out.sort_values(["model_alias", "context_bucket"]).reset_index(drop=True)


def score_variance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Tasks with highest score variance across models (discriminating tasks).

    Returns: task_id, variance, score_range, best_model, best_score,
             worst_model, worst_score, num_models.
    """
    _ensure_pandas()
    scored = df[df["score_primary"].notna()].copy()
    task_groups = scored.groupby("task_id")
    results = []
    for task_id, group in task_groups:
        scores = group["score_primary"].values
        if len(scores) < 2:
            continue
        models = group["model_alias"].values
        variance = float(np.var(scores)) if (
            np := _import_numpy()
        ) else 0.0
        best_idx = int(np.argmax(scores)) if np else 0
        worst_idx = int(np.argmin(scores)) if np else 0
        results.append({
            "task_id": task_id,
            "variance": variance,
            "score_range": float(scores.max() - scores.min()) if np else 0.0,
            "best_model": str(models[best_idx]) if np else "",
            "best_score": float(scores[best_idx]) if np else 0.0,
            "worst_model": str(models[worst_idx]) if np else "",
            "worst_score": float(scores[worst_idx]) if np else 0.0,
            "num_models": len(models),
        })
    if results:
        return pd.DataFrame(results).sort_values("variance", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def _import_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def style_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compare scores by prompt style per task.

    Returns: task_id, prompt_style, avg_score, run_count, avg_wall_ms, avg_tokens.
    """
    _ensure_pandas()
    has_style = "prompt_style" in df.columns and df["prompt_style"].notna().any()
    if not has_style:
        return pd.DataFrame(columns=["task_id", "prompt_style", "avg_score",
                                      "run_count", "avg_wall_ms", "avg_tokens"])
    scored = df[df["score_primary"].notna()].copy()
    group = scored.groupby(["task_id", "prompt_style"])
    out = group.agg(
        avg_score=("score_primary", "mean"),
        run_count=("score_primary", "count"),
        avg_wall_ms=("total_wall_ms", "mean"),
        avg_tokens=("total_tokens", "mean"),
    ).reset_index()
    return out.sort_values(["task_id", "prompt_style"]).reset_index(drop=True)


def failures_df(df: pd.DataFrame) -> pd.DataFrame:
    """Failed/error runs grouped by error pattern.

    Returns: error_cluster, model_alias, task_id, run_id, error_message.
    """
    _ensure_pandas()
    errors = df[df["exit_status"] == "error"].copy()
    if errors.empty:
        return pd.DataFrame(columns=["error_cluster", "model_alias", "task_id",
                                      "run_id", "error_message"])
    def _cluster(msg):
        if pd.isna(msg) or not msg:
            return "unknown"
        return str(msg)[:80]
    errors = errors.copy()
    errors["error_cluster"] = errors["error_message"].apply(_cluster)
    return errors[["error_cluster", "model_alias", "task_id", "run_id", "error_message"]]


def identify_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Detect alias-vs-actual model mismatches from identity stamp fields.

    Returns: requested_alias, openai_models_id, vllm_served_model_name,
             vllm_container_name, hf_model_id, run_count, mismatch_detected.
    """
    _ensure_pandas()
    has_identity = any(c in df.columns for c in
                       ["openai_models_id", "vllm_served_model_name"])
    if not has_identity:
        return pd.DataFrame(columns=["requested_alias", "openai_models_id",
                                      "vllm_served_model_name", "run_count",
                                      "mismatch_detected"])
    stamp = df[["model_alias", "openai_models_id", "vllm_served_model_name",
                "vllm_container_name", "hf_model_id"]].dropna(
        subset=["openai_models_id"]
    ).copy()
    if stamp.empty:
        return pd.DataFrame(columns=["requested_alias", "openai_models_id",
                                      "vllm_served_model_name", "run_count",
                                      "mismatch_detected"])
    stamp["requested_alias"] = stamp["model_alias"]
    group = stamp.groupby(["requested_alias", "openai_models_id",
                           "vllm_served_model_name", "vllm_container_name",
                           "hf_model_id"]).size().reset_index(name="run_count")
    group["mismatch_detected"] = group["requested_alias"] != group["openai_models_id"]
    return group.sort_values(["mismatch_detected", "requested_alias"],
                             ascending=[False, True]).reset_index(drop=True)


# ── Quick helper for DuckDB queries ──────────────────────────────────────

def query_duck(query: str, db_path: str | Path | None = None) -> pd.DataFrame:
    """Run a DuckDB SQL query directly on the SQLite database.

    If db_path is provided, attaches it as a virtual SQLite database.
    Otherwise expects 'runs' to be registered already.
    """
    _ensure_pandas()
    if duckdb is None:
        raise RuntimeError("duckdb not installed")
    con = duckdb.connect()
    if db_path:
        con.execute(f"ATTACH '{db_path}' AS src (TYPE SQLite)")
        result = con.execute(query.replace("runs", "src.runs").replace("run_timings", "src.run_timings")).fetchdf()
    else:
        result = con.execute(query).fetchdf()
    con.close()
    return result
