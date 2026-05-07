"""CLI for the benchmark harness."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from typing import Any

from bench_harness.config import (
    load_model_config,
    get_model,
    get_quantization,
    load_suite_config,
    resolve_task_dir,
    load_judge_config,
)
from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.runners.completion_runner import CompletionRunner
from bench_harness.runners.style_sweep_runner import StyleSweepRunner
from bench_harness.runners.context_sweep_runner import ContextSizeSweepRunner
from bench_harness.tasks.loaders import load_tasks
from bench_harness.tasks.registry import TaskRegistry
from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.storage.artifacts import save_run_artifact
from bench_harness.reports.markdown import generate_report, print_summary
from bench_harness.lm_eval_adapter import (
    LMEvalAdapter,
    get_lm_eval_error,
    TASK_DEFS as LM_EVAL_TASKS,
)

app = typer.Typer(
    name="bench-harness",
    help="Local LLM quality benchmark harness",
    add_completion=False,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def _parse_models_response(resp: dict[str, Any], backend_url: str) -> dict[str, Any]:
    """Parse /v1/models response into identity stamp fields.

    Called before each task to record the actual model served by the backend.
    This makes 'qwen-dense served as agent-code' obvious instead of inferential.

    Args:
        resp: Raw response from client.fetch_models().
        backend_url: The backend URL used for this request.

    Returns:
        Dict with identity stamp fields to pass to the runner.
    """
    result = {
        "backend_url": backend_url,
        "openai_models_id": None,
        "vllm_served_model_name": None,
        "vllm_container_name": None,
        "hf_model_id": None,
        "server_start_time": None,
        "speculative_decoding_enabled": None,
    }
    error = resp.get("error")
    if error:
        logger.warning("/v1/models returned error: %s", error)
        return result

    data = resp.get("data", [])
    if not data:
        logger.warning("/v1/models returned empty data list")
        return result

    # vLLM exposes model info in each model object
    # Look for the first model entry and extract fields
    for model_entry in data:
        model_id = model_entry.get("id", "")
        if not model_id:
            continue

        # This is the actual model id returned by the server
        result["openai_models_id"] = model_id

        # vLLM specific fields
        if "served_model_name" in model_entry:
            result["vllm_served_model_name"] = model_entry.get("served_model_name")
        if "container_name" in model_entry:
            result["vllm_container_name"] = model_entry.get("container_name")
        if "hf_model_id" in model_entry:
            result["hf_model_id"] = model_entry.get("hf_model_id")
        if "server_start_time" in model_entry:
            result["server_start_time"] = model_entry.get("server_start_time")
        if "speculative_decoder_id" in model_entry:
            result["speculative_decoding_enabled"] = bool(
                model_entry.get("speculative_decoder_id")
            )
        # Also check for speculative decoding flag
        if result["speculative_decoding_enabled"] is None:
            spec = model_entry.get("speculative_decoder_id") or model_entry.get(
                "speculative_decoding_enabled"
            )
            result["speculative_decoding_enabled"] = bool(spec)

        break  # First model entry is the one we care about

    logger.info(
        "identity stamp: openai_models_id=%s served=%s url=%s",
        result["openai_models_id"],
        result["vllm_served_model_name"],
        backend_url,
    )
    return result


@app.command()
def run(
    suite: str = typer.Option("smoke", "--suite", help="Suite name(s), comma-separated"),
    models: str = typer.Option("agent-code", "--models", help="Model alias(es), comma-separated"),
    endpoint: str | None = typer.Option(None, "--endpoint", help="Base URL override"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Max output tokens"),
    runs: int = typer.Option(1, "--runs", help="Number of repetitions per task"),
    styles: str = typer.Option("plain", "--styles", help="Comma-separated list of prompt styles (default: plain)"),
    context_sizes: str = typer.Option("small", "--context-sizes", help="Comma-separated list of context sizes (small,medium,large,xlarge)"),
    out: str | None = typer.Option(None, "--out", help="Output directory"),
    timing_detail: bool = typer.Option(False, "--timing-detail", help="Show per-task timing in CLI output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print tasks without executing"),
    judge: bool = typer.Option(False, "--judge", help="Run LLM judge scorer on scored tasks"),
    report_v2: bool = typer.Option(False, "--report-v2", help="Use v2 modular report format"),
    sections: str = typer.Option("", "--sections", help="Comma-separated list of report sections to include (v2 only)"),
    prior_runs: str = typer.Option(None, "--prior-runs", help="Path to prior run results JSONL for regression detection"),
):
    """Run benchmark suite against one or more models."""
    suite_names = [s.strip() for s in suite.split(",")]
    model_aliases = [m.strip() for m in models.split(",")]

    if out is None:
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        out = f"runs/{date_str}-{suite_names[0]}"

    # Load configs
    try:
        model_config = load_model_config()
    except FileNotFoundError as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    suite_id = suite_names[0]

    # Resolve task directory
    try:
        task_dir = resolve_task_dir(suite_id)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Suite error:[/red] {e}")
        sys.exit(1)

    # Load tasks
    tasks = load_tasks(str(task_dir))
    if not tasks:
        console.print(f"[red]No tasks loaded from {task_dir}[/red]")
        sys.exit(1)

    # Dry run: print tasks and exit
    if dry_run:
        console.print(f"[cyan]Tasks in suite '{suite_id}':[/cyan]")
        for t in tasks:
            console.print(f"  - {t.get('id')}: {t.get('family', 'unknown')}")
        console.print(f"[green]Dry run complete. {len(tasks)} tasks.[/green]")
        return

    # Resolve model base URL
    first_model_alias = model_aliases[0]
    first_model_cfg = get_model(model_config, first_model_alias)
    if first_model_cfg is None:
        console.print(f"[red]Model '{first_model_alias}' not found in config[/red]")
        sys.exit(1)

    base_url = endpoint or first_model_cfg.get("base_url", "")
    if not base_url:
        console.print("[red]No base_url configured and no --endpoint provided[/red]")
        sys.exit(1)

    # Setup output
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Setup SQLite
    db_path = str(out_path / "benchmark.db")
    store = SQLiteStore(db_path)
    store.init()

    # Run
    style_list = [s.strip() for s in styles.split(",")]
    context_size_list = [s.strip() for s in context_sizes.split(",")]
    all_results = []

    for alias in model_aliases:
        model_cfg = get_model(model_config, alias)
        if model_cfg is None:
            console.print(f"[yellow]Skipping unknown model: {alias}[/yellow]")
            continue

        url = endpoint or model_cfg.get("base_url", base_url)
        model_name = model_cfg.get("model", alias)
        backend = model_cfg.get("backend", "")

        console.print(f"\n[cyan]=== Running model: {alias} ({model_name}) ===[/cyan]")

        client = OpenAICompatClient(base_url=url, model=model_name)
        runner = CompletionRunner(client)

        quant = get_quantization(model_config, alias)
        params = {
            "model_alias": alias,
            "model_backend": backend,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "quantization": quant,
        }

        # Fetch identity stamp once per model, not per task
        models_resp = asyncio.run(client.fetch_models())
        identity = _parse_models_response(models_resp, url)
        identity["requested_alias"] = alias
        identity["litellm_model_name"] = model_name

        for task in tasks:
            for run_num in range(1, runs + 1):
                console.print(
                    f"  [dim]{task.get('id')} (run {run_num}/{runs})[/dim]"
                )

                # Context sweep mode (overrides single-style path when sizes > 1)
                if len(context_size_list) > 1:
                    context_params = {
                        "model_alias": alias,
                        "model_backend": backend,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    # Inject identity stamp
                    for k, v in identity.items():
                        context_params[k] = v
                    sweep = ContextSizeSweepRunner(runner, sizes=context_size_list)
                    sweep_results = asyncio.run(
                        sweep.run_sweep([task], [alias], context_params, suite_id=suite_id)
                    )
                    for result in sweep_results:
                        all_results.append(result)
                        store.save_run(result)
                        save_run_artifact(result, str(out_path))

                        if result.exit_status == "error":
                            console.print(
                                f"  [red]  [context={result.context_tokens}] Error:[/red] {result.error_message}"
                            )
                        else:
                            if timing_detail:
                                tps_str = (
                                    f"{result.tokens_per_second:.1f}"
                                    if result.tokens_per_second > 0
                                    else "N/A"
                                )
                                console.print(
                                    f"  [green]  [context={result.context_tokens}] TTFT: {result.ttft_ms:.0f}ms[/green] "
                                    f"[green]Decode: {result.decode_ms:.0f}ms[/green] "
                                    f"[green]Wall: {result.total_wall_ms:.0f}ms[/green] "
                                    f"[dim]{result.prompt_tokens}p + {result.completion_tokens}c tokens[/dim] "
                                    f"[dim]{tps_str} tok/s[/dim]"
                                )
                            else:
                                console.print(
                                    f"  [green]  [context={result.context_tokens}] Wall: {result.total_wall_ms:.0f}ms[/green] "
                                    f"[dim]{result.total_tokens} tokens[/dim]"
                                )
                    continue

                if len(style_list) > 1:
                    # Multi-style sweep
                    style_params = {
                        "model_alias": alias,
                        "model_backend": backend,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "quantization": quant,
                    }
                    # Inject identity stamp
                    for k, v in identity.items():
                        style_params[k] = v
                    sweep = StyleSweepRunner(runner, styles=style_list)
                    sweep_results = asyncio.run(
                        sweep.run_sweep([task], [alias], style_params, suite_id=suite_id)
                    )
                    for result in sweep_results:
                        all_results.append(result)
                        store.save_run(result)
                        save_run_artifact(result, str(out_path))

                        if result.exit_status == "error":
                            console.print(
                                f"  [red]  [{result.prompt_style}] Error:[/red] {result.error_message}"
                            )
                        else:
                            if timing_detail:
                                tps_str = (
                                    f"{result.tokens_per_second:.1f}"
                                    if result.tokens_per_second > 0
                                    else "N/A"
                                )
                                console.print(
                                    f"  [green]  [{result.prompt_style}] TTFT: {result.ttft_ms:.0f}ms[/green] "
                                    f"[green]Decode: {result.decode_ms:.0f}ms[/green] "
                                    f"[green]Wall: {result.total_wall_ms:.0f}ms[/green] "
                                    f"[dim]{result.prompt_tokens}p + {result.completion_tokens}c tokens[/dim] "
                                    f"[dim]{tps_str} tok/s[/dim]"
                                )
                            else:
                                console.print(
                                    f"  [green]  [{result.prompt_style}] Wall: {result.total_wall_ms:.0f}ms[/green] "
                                    f"[dim]{result.total_tokens} tokens[/dim]"
                                )
                else:
                    # Single style (legacy path) — inject identity stamp into params
                    for k, v in identity.items():
                        params[k] = v
                    result = asyncio.run(runner.run(task, params, suite_id=suite_id))
                    all_results.append(result)

                    # Save to SQLite
                    store.save_run(result)

                    # Save JSONL artifact
                    save_run_artifact(result, str(out_path))

                    if result.exit_status == "error":
                        console.print(
                            f"  [red]  Error:[/red] {result.error_message}"
                        )
                    else:
                        if timing_detail:
                            tps_str = (
                                f"{result.tokens_per_second:.1f}"
                                if result.tokens_per_second > 0
                                else "N/A"
                            )
                            console.print(
                                f"  [green]  TTFT: {result.ttft_ms:.0f}ms[/green] "
                                f"[green]Decode: {result.decode_ms:.0f}ms[/green] "
                                f"[green]Wall: {result.total_wall_ms:.0f}ms[/green] "
                                f"[dim]{result.prompt_tokens}p + {result.completion_tokens}c tokens[/dim] "
                                f"[dim]{tps_str} tok/s[/dim]"
                            )
                        else:
                            console.print(
                                f"  [green]  Wall: {result.total_wall_ms:.0f}ms[/green] "
                                f"[dim]{result.total_tokens} tokens[/dim]"
                            )

    # Parse sections if specified
    section_list = None
    if sections:
        section_list = [s.strip() for s in sections.split(",") if s.strip()]

    # Load prior runs for regression detection
    prior_run_list: list[dict] | None = None
    if prior_runs:
        prior_path = Path(prior_runs)
        if prior_path.exists():
            import json
            with open(prior_path, "r") as f:
                prior_run_list = json.load(f)
            logger.info("Loaded %d prior runs for regression detection", len(prior_run_list))

    # Generate report
    runs_list = store.get_runs(suite_id=suite_id)
    report_md = generate_report(
        runs_list,
        suite_id,
        model_config,
        str(out_path / "report.md"),
        v2=report_v2,
        sections=section_list,
        prior_runs=prior_run_list,
    )

    # Print style comparison summary when multiple styles are used
    if len(style_list) > 1:
        console.print(f"\n[yellow]=== Style Comparison ({', '.join(style_list)}) ===[/yellow]")
        # Group results by task_id and style
        from collections import defaultdict
        task_style_results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for run in runs_list:
            tid = run.get("task_id", "unknown")
            ps = run.get("prompt_style", "plain")
            task_style_results[tid][ps].append(run)

        for tid in sorted(task_style_results.keys()):
            styles_for_task = task_style_results[tid]
            console.print(f"\n  [cyan]{tid}[/cyan]")
            for style in style_list:
                style_runs = styles_for_task.get(style, [])
                if style_runs:
                    avg_wall = sum(r.get("total_wall_ms", 0) for r in style_runs) / len(style_runs)
                    avg_tokens = sum(r.get("total_tokens", 0) for r in style_runs) / len(style_runs)
                    passed = sum(1 for r in style_runs if r.get("score_primary") and r["score_primary"] > 0)
                    console.print(
                        f"    [dim]{style}[/dim]: "
                        f"wall={avg_wall:.0f}ms "
                        f"tokens={avg_tokens:.0f} "
                        f"passed={passed}/{len(style_runs)}"
                    )

    # Print context sweep summary when multiple sizes are used
    if len(context_size_list) > 1:
        console.print(f"\n[yellow]=== Context Size Sweep ({', '.join(context_size_list)}) ===[/yellow]")
        from collections import defaultdict
        task_size_results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for run in runs_list:
            tid = run.get("task_id", "unknown")
            cs = run.get("context_tokens", "small")
            task_size_results[tid][cs].append(run)

        for tid in sorted(task_size_results.keys()):
            sizes_for_task = task_size_results[tid]
            console.print(f"\n  [cyan]{tid}[/cyan]")
            for size in context_size_list:
                size_runs = sizes_for_task.get(size, [])
                if size_runs:
                    avg_wall = sum(r.get("total_wall_ms", 0) for r in size_runs) / len(size_runs)
                    avg_tokens = sum(r.get("total_tokens", 0) for r in size_runs) / len(size_runs)
                    passed = sum(1 for r in size_runs if r.get("score_primary") and r["score_primary"] > 0)
                    console.print(
                        f"    [dim]{size}[/dim]: "
                        f"wall={avg_wall:.0f}ms "
                        f"tokens={avg_tokens:.0f} "
                        f"passed={passed}/{len(size_runs)}"
                    )

    # Print summary
    console.print(f"\n[yellow]=== Results ===[/yellow]")
    print_summary(runs_list, suite_id)

    console.print(f"\n[green]Report saved to: {out_path / 'report.md'}[/green]")
    console.print(f"[green]Database saved to: {out_path / 'benchmark.db'}[/green]")
    console.print(f"[green]Artifacts saved to: {out_path}/[/green]")

    return all_results


@app.command()
def list_tasks(
    family: str | None = typer.Option(None, "--family", help="Filter by task family, comma-separated"),
    source: str | None = typer.Option(None, "--source", help="Filter by task source"),
):
    """List all registered tasks, optionally filtered by family or source."""
    registry = TaskRegistry()

    # Auto-discover task directories from configs/
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    task_dirs = []

    # Check for known task directories
    root = Path(__file__).resolve().parent.parent.parent
    for td in ("tasks/smoke", "tasks/coding_smoke", "tasks/local_coding_agent_v1", "tasks/coding_benchmark",
               "tasks/agent_safety", "tasks/public_baseline", "tasks/quantization_test"):
        td_path = root / td
        if td_path.exists():
            task_dirs.append(str(td_path))

    if not task_dirs:
        console.print("[yellow]No task directories found.[/yellow]")
        return

    for td in task_dirs:
        registry.load_from_directory(td)

    if not registry:
        console.print("[yellow]No tasks registered.[/yellow]")
        return

    tasks = registry.list_all()

    if family:
        family_filter = [f.strip() for f in family.split(",")]
        tasks = [t for t in tasks if t.family in family_filter]
    if source:
        tasks = [t for t in tasks if t.source == source]

    if not tasks:
        console.print("[yellow]No tasks match the given filters.[/yellow]")
        return

    from rich.table import Table
    from rich.console import Console

    table = Table(title=f"Tasks (total: {len(tasks)})")
    table.add_column("ID", style="cyan")
    table.add_column("Version", style="dim")
    table.add_column("Family", style="green")
    table.add_column("Source", style="blue")
    table.add_column("Risk", style="yellow")
    table.add_column("Scorer", style="magenta")

    for task in sorted(tasks, key=lambda t: t.id):
        table.add_row(
            task.id,
            task.version,
            task.family,
            task.source,
            task.risk_level,
            task.scoring.primary,
        )

    console.print(table)


@app.command()
def show_task(
    task_id: str = typer.Argument(..., help="Task ID to display"),
):
    """Show the full details of a specific task by ID."""
    registry = TaskRegistry()

    root = Path(__file__).resolve().parent.parent.parent
    for td in ("tasks/smoke", "tasks/coding_smoke", "tasks/local_coding_agent_v1", "tasks/coding_benchmark",
               "tasks/agent_safety", "tasks/public_baseline", "tasks/quantization_test"):
        td_path = root / td
        if td_path.exists():
            registry.load_from_directory(str(td_path))

    task = registry.get(task_id)
    if task is None:
        console.print(f"[red]Task '{task_id}' not found.[/red]")
        console.print(f"[dim]Registered tasks: {[t.id for t in registry.list_all()]}[/dim]")
        return

    console.print(f"[cyan]Task: {task.id}[/cyan]")
    console.print(f"  Version:    {task.version}")
    console.print(f"  Family:     {task.family}")
    console.print(f"  Category:   {task.category or 'N/A'}")
    console.print(f"  Source:     {task.source}")
    console.print(f"  Risk:       {task.risk_level}")
    console.print(f"  Context:    {task.context_tokens}")
    console.print(f"  Scoring:    primary={task.scoring.primary}")
    if task.scoring.secondary:
        console.print(f"              secondary={', '.join(task.scoring.secondary)}")
    console.print()
    console.print(f"  [bold]Prompt:[/bold]")
    for line in (task.prompt or task.to_dict().get("prompt", "")).split("\n"):
        console.print(f"    {line}")
    console.print()
    console.print(f"  [bold]Expected:[/bold]")
    expected = task.expected
    console.print(f"    type: {expected.type}")
    if expected.answer:
        console.print(f"    answer: {expected.answer}")
    if expected.patterns:
        console.print(f"    patterns: {expected.patterns}")


judge_app = typer.Typer(
    name="judge-config",
    help="Manage judge configuration and rubrics",
    add_completion=False,
)


@judge_app.command()
def list_rubrics():
    """List available judge rubrics."""
    from bench_harness.config import load_rubric_config
    try:
        rubrics = load_rubric_config()
        if not rubrics:
            console.print("[yellow]No rubrics found.[/yellow]")
            return
        # Config format is {"rubrics": {"name": {...}, ...}}
        rubric_dict = rubrics.get("rubrics", rubrics) if isinstance(rubrics, dict) else {}
        if not rubric_dict:
            console.print("[yellow]No rubrics found.[/yellow]")
            return
        if isinstance(rubric_dict, dict):
            console.print(f"[cyan]Available rubrics:[/cyan]")
            for name in sorted(rubric_dict.keys()):
                rubric = rubric_dict[name]
                if isinstance(rubric, dict):
                    desc = rubric.get("description", "")
                    console.print(f"  - {name}: {desc}")
                else:
                    console.print(f"  - {name}")
        else:
            console.print(f"[dim](Single rubric configuration)[/dim]")
    except FileNotFoundError:
        console.print("[yellow]No judge_rubrics.yaml found. Add rubrics to configs/judge_rubrics.yaml[/yellow]")


@judge_app.command()
def show_rubric(
    rubric_name: str = typer.Argument(..., help="Rubric name to display"),
):
    """Show details of a specific rubric."""
    from bench_harness.config import load_rubric_config
    try:
        rubrics = load_rubric_config()
        # Config format is {"rubrics": {"name": {...}, ...}}
        rubric_dict = rubrics.get("rubrics", rubrics) if isinstance(rubrics, dict) else {}
        rubric = rubric_dict.get(rubric_name)
        if rubric is None:
            console.print(f"[red]Rubric '{rubric_name}' not found.[/red]")
            return
        if isinstance(rubric, dict):
            console.print(f"[cyan]Rubric: {rubric_name}[/cyan]")
            for key, value in rubric.items():
                console.print(f"  {key}: {value}")
        else:
            console.print(f"[cyan]Rubric: {rubric_name}[/cyan]")
            console.print(f"  {rubric}")
    except FileNotFoundError:
        console.print(f"[red]Rubric '{rubric_name}' not found.[/red]")


def main():
    """Entry point when called as `bench-harness`."""
    app()


export_app = typer.Typer(
    name="export",
    help="Export benchmark results as training datasets",
    add_completion=False,
)


@export_app.command(name="sft")
def export_sft_cmd(
    suite_id: str = typer.Option("smoke", "--suite", help="Suite ID to export from"),
    db_path: str = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(Path("exports"), "--out", help="Output directory"),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum score for SFT export"),
    no_system: bool = typer.Option(False, "--no-system", help="Exclude system messages"),
):
    """Export successful runs as SFT JSONL (OpenAI messages format)."""
    from bench_harness.export import export_sft as _export_sft

    db = db_path or _resolve_db_path(out, suite_id)
    result = _export_sft(
        db, suite_id, str(out), min_score=min_score,
        include_system_messages=not no_system,
    )
    console.print(f"[green]SFT export complete:[/green] {result}")


@export_app.command(name="preference")
def export_preference_cmd(
    suite_id: str = typer.Option("smoke", "--suite", help="Suite ID to export from"),
    db_path: str = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(Path("exports"), "--out", help="Output directory"),
    min_margin: float = typer.Option(0.1, "--min-margin", help="Min score margin for preference export"),
):
    """Export pairwise preferences as DPO JSONL."""
    from bench_harness.export import export_preference as _export_pref

    db = db_path or _resolve_db_path(out, suite_id)
    result = _export_pref(db, suite_id, str(out), min_score_margin=min_margin)
    console.print(f"[green]Preference export complete:[/green] {result}")


@export_app.command(name="regression")
def export_regression_cmd(
    suite_id: str = typer.Option("smoke", "--suite", help="Suite ID to export from"),
    db_path: str = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(Path("exports"), "--out", help="Output directory"),
    include_api_errors: bool = typer.Option(False, "--include-api-errors", help="Include API errors in regression"),
):
    """Export failed tasks as YAML regression suite."""
    from bench_harness.export import export_regression as _export_reg

    db = db_path or _resolve_db_path(out, suite_id)
    result = _export_reg(
        db, suite_id, str(out),
        exclude_api_errors=not include_api_errors,
    )
    console.print(f"[green]Regression export complete:[/green] {result}")


@export_app.command(name="judge")
def export_judge_cmd(
    suite_id: str = typer.Option("smoke", "--suite", help="Suite ID to export from"),
    db_path: str = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(Path("exports"), "--out", help="Output directory"),
    no_pairwise: bool = typer.Option(False, "--no-pairwise", help="Exclude pairwise comparisons"),
):
    """Export judge evaluations as JSONL."""
    from bench_harness.export import export_judge as _export_judge

    db = db_path or _resolve_db_path(out, suite_id)
    result = _export_judge(db, suite_id, str(out), include_pairwise=not no_pairwise)
    console.print(f"[green]Judge export complete:[/green] {result}")


@export_app.command(name="all")
def export_all_cmd(
    suite_id: str = typer.Option("smoke", "--suite", help="Suite ID to export from"),
    db_path: str = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(Path("exports"), "--out", help="Output directory"),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum score for SFT export"),
    min_margin: float = typer.Option(0.1, "--min-margin", help="Min score margin for preference export"),
    include_api_errors: bool = typer.Option(False, "--include-api-errors", help="Include API errors in regression"),
    no_pairwise: bool = typer.Option(False, "--no-pairwise", help="Exclude pairwise comparisons"),
    no_system: bool = typer.Option(False, "--no-system", help="Exclude system messages"),
):
    """Export all four formats: SFT, DPO, regression, and judge data."""
    from bench_harness.export import (
        export_sft as _export_sft,
        export_preference as _export_pref,
        export_regression as _export_reg,
        export_judge as _export_judge,
    )

    db = db_path or _resolve_db_path(out, suite_id)

    sft_path = _export_sft(
        db, suite_id, str(out), min_score=min_score,
        include_system_messages=not no_system,
    )
    console.print(f"[green]SFT export:[/green] {sft_path}")

    pref_path = _export_pref(
        db, suite_id, str(out), min_score_margin=min_margin,
    )
    console.print(f"[green]Preference export:[/green] {pref_path}")

    reg_path = _export_reg(
        db, suite_id, str(out),
        exclude_api_errors=not include_api_errors,
    )
    console.print(f"[green]Regression export:[/green] {reg_path}")

    judge_path = _export_judge(
        db, suite_id, str(out), include_pairwise=not no_pairwise,
    )
    console.print(f"[green]Judge export:[/green] {judge_path}")


def _resolve_db_path(out_dir: Path, suite_id: str) -> str:
    """Resolve default database path."""
    out = Path(out_dir)
    db = out / "benchmark.db"
    if db.exists():
        return str(db)
    return str(out / "benchmark.db")


# Register apps
app.add_typer(export_app, name="export")
app.add_typer(judge_app, name="judge-config")

analyze_app = typer.Typer(
    name="analyze",
    help="Interactive analysis tools for benchmark results",
    add_completion=False,
)


@analyze_app.command()
def notebook(
    db: Path = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(Path("notebooks"), "--out", help="Output notebook directory"),
):
    """Open or generate the analysis notebook for a benchmark database."""
    if db is None:
        out_path = Path(out)
        db_candidates = sorted(out_path.glob("*/benchmark.db"), reverse=True)
        if db_candidates:
            db = db_candidates[0]
        else:
            db = Path("runs/2026-05-06-coding_benchmark/benchmark.db")
            if not db.exists():
                console.print(f"[red]No benchmark.db found. Specify --db or ensure DB exists at {db}[/red]")
                return
    notebook_path = Path(__file__).resolve().parent.parent.parent / "notebooks" / "bench_dashboard.ipynb"
    if not notebook_path.exists():
        console.print(f"[red]Notebook not found at {notebook_path}[/red]")
        return
    # Copy and update the DB_PATH cell
    import json
    with open(notebook_path, "r") as f:
        nb = json.load(f)
    # Replace the DB_PATH default in the notebook
    for cell in nb.get("cells", []):
        if isinstance(cell.get("source"), list):
            src = "".join(cell["source"])
        else:
            src = cell["source"]
        if "DB_PATH" in src and "benchmark.db" in src:
            # Update the DB_PATH cell to point to the actual DB
            for i, line in enumerate(cell["source"]):
                if "DB_PATH" in line and isinstance(line, str):
                    cell["source"][i] = f'DB_PATH = "{db}"\n'
                    break
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = f"bench_{db.parent.name}.ipynb"
    out_path = out_dir / notebook_name
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    console.print(f"[green]Notebook saved:[/green] {out_path}")
    console.print(f"[dim]Open with:[/dim] jupyter notebook {out_path}")


compare_app = typer.Typer(
    name="compare",
    help="Compare two benchmark runs and detect regressions",
    add_completion=False,
)


@compare_app.command()
def compare_runs_cmd(
    baseline: str = typer.Argument(..., help="Path to baseline benchmark.db"),
    candidate: str = typer.Argument(..., help="Path to candidate benchmark.db"),
    score_threshold: float = typer.Option(0.05, "--score-threshold", help="Minimum absolute score change to flag (default 0.05)"),
    tps_threshold: float = typer.Option(0.1, "--tps-threshold", help="Relative TPS change threshold as fraction (default 0.1 = 10%)"),
):
    """Compare two benchmark runs and flag quality/performance regressions."""
    from bench_harness.compare import compare_runs, format_comparison_output

    if not Path(baseline).exists():
        console.print(f"[red]Baseline DB not found: {baseline}[/red]")
        sys.exit(1)
    if not Path(candidate).exists():
        console.print(f"[red]Candidate DB not found: {candidate}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Comparing benchmarks:[/cyan]")
    console.print(f"  Baseline:   {baseline}")
    console.print(f"  Candidate:  {candidate}")
    console.print(f"  Score threshold: {score_threshold}")
    console.print(f"  TPS threshold: {tps_threshold}")
    console.print()

    results = compare_runs(
        baseline_db=baseline,
        candidate_db=candidate,
        score_threshold=score_threshold,
        tps_threshold=tps_threshold,
    )

    format_comparison_output(results)


regression_app = typer.Typer(
    name="regression",
    help="Generate a quick regression test suite from benchmark history",
    add_completion=False,
)


@regression_app.command()
def regression_suite(
    db: str = typer.Option(..., "--db", help="Path to benchmark.db"),
    max_tasks: int = typer.Option(10, "--max-tasks", help="Maximum number of variance-based tasks to include (default: 10)"),
    out_dir: str = typer.Option(None, "--out-dir", help="Output directory for the regression suite (default: current directory)"),
):
    """Generate a YAML regression suite from benchmark run history.

    Selects tasks with the highest score variance across models and
    includes all tasks that have failed, producing a compact YAML file
    for fast regression testing before model/backend changes.
    """
    from bench_harness.regression import generate_regression_suite

    if not Path(db).exists():
        console.print(f"[red]Database not found: {db}[/red]")
        sys.exit(1)

    if out_dir is None:
        out_dir = "."

    result = generate_regression_suite(db, out_dir, max_tasks=max_tasks)
    console.print(f"[green]Regression suite generated:[/green] {result}")


# Register apps
app.add_typer(export_app, name="export")
app.add_typer(judge_app, name="judge-config")
app.add_typer(analyze_app, name="analyze")
app.add_typer(compare_app, name="compare")
app.add_typer(regression_app, name="regression")

@app.command("run-lm-eval")
def run_lm_eval(
    suite: str = typer.Option("public_baseline", "--suite", help="Suite name"),
    models: str = typer.Option("agent-code", "--models", help="Model alias(es), comma-separated"),
    tasks: str = typer.Option("mmlu_college_math", "--tasks", help="Comma-separated lm_eval task names"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Override max samples per task"),
    runs: int = typer.Option(1, "--runs", help="Number of repetitions per task"),
    endpoint: str | None = typer.Option(None, "--endpoint", help="Base URL override"),
    report_v2: bool = typer.Option(False, "--report-v2", help="Use v2 modular report format"),
):
    """Run public benchmark tasks (MMLU, GPQA, BBH, MATH) via lm-evaluation-harness."""
    from bench_harness.config import load_model_config, get_model

    model_aliases = [m.strip() for m in models.split(",")]
    task_names = [t.strip() for t in tasks.split(",")]

    # Validate task names
    for tn in task_names:
        if tn not in LM_EVAL_TASKS:
            console.print(f"[red]Unknown task: {tn}[/red]")
            available = ", ".join(sorted(LM_EVAL_TASKS.keys()))
            console.print(f"[dim]Available tasks: {available}[/dim]")
            sys.exit(1)

    # Load model config
    try:
        model_config = load_model_config()
    except FileNotFoundError as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    # Resolve endpoint
    first_alias = model_aliases[0]
    first_model = get_model(model_config, first_alias)
    if first_model is None:
        console.print(f"[red]Model '{first_alias}' not found in config[/red]")
        sys.exit(1)

    base_url = endpoint or first_model.get("base_url", "")
    if not base_url:
        console.print("[red]No base_url configured and no --endpoint provided[/red]")
        sys.exit(1)

    model_name = first_model.get("model", first_alias)

    # Setup output
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    out = f"runs/{date_str}-{suite}"
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Setup SQLite
    db_path = str(out_path / "benchmark.db")
    store = SQLiteStore(db_path)
    store.init()

    console.print(f"[cyan]=== Public Benchmark: {suite} ===[/cyan]")
    console.print(f"[dim]Model: {model_name} | Endpoint: {base_url}[/dim]")
    console.print(f"[dim]Tasks: {', '.join(task_names)} | Samples: {max_samples or 'default'} | Runs: {runs}[/dim]")
    console.print()

    all_results = []

    for alias in model_aliases:
        model_cfg = get_model(model_config, alias)
        if model_cfg is None:
            console.print(f"[yellow]Skipping unknown model: {alias}[/yellow]")
            continue

        url = endpoint or model_cfg.get("base_url", base_url)
        name = model_cfg.get("model", alias)
        backend = model_cfg.get("backend", "")

        console.print(f"\n[cyan]=== Model: {alias} ({name}) ===[/cyan]")

        # Check if lm_eval is available
        if not _check_lm_eval_available():
            console.print(f"  [yellow]lm-evaluation-harness not installed. {get_lm_eval_error()}[/yellow]")
            # Store error results in database
            import uuid as _uuid
            from bench_harness.runners.completion_runner import RunResult

            for tn in task_names:
                task_def = LM_EVAL_TASKS[tn]
                run_id = str(_uuid.uuid4())
                error_result = RunResult(
                    run_id=run_id,
                    suite_id=suite,
                    task_id=f"public.{tn}",
                    model_alias=alias,
                    model_backend=backend,
                    prompt="",
                    raw_response="",
                    exit_status="error",
                    error_message=get_lm_eval_error(),
                    score_primary=None,
                    total_wall_ms=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    ttft_ms=0,
                    prefill_ms=0,
                    decode_ms=0,
                    tokens_per_second=0,
                    tokens_per_second_total=0,
                    token_source="api",
                )
                store.save_run(error_result)
                all_results.append(__import__('dataclasses').asdict(error_result))
            continue

        adapter = LMEvalAdapter(
            base_url=url,
            model=name,
            max_samples=max_samples,
            runs=runs,
        )

        lm_results = adapter.run_suite(task_names=task_names, suite_id=suite)

        for lm_r in lm_results:
            import uuid as _uuid
            from bench_harness.runners.completion_runner import RunResult

            accuracy = lm_r.accuracy if lm_r.accuracy is not None else 0.0
            run_id = str(_uuid.uuid4())

            result = RunResult(
                run_id=run_id,
                suite_id=suite,
                task_id=lm_r.task_id,
                model_alias=alias,
                model_backend=backend,
                prompt="",
                raw_response=f"accuracy={lm_r.accuracy}, samples={lm_r.samples_run}",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                ttft_ms=0,
                prefill_ms=0,
                decode_ms=0,
                total_wall_ms=0,
                tokens_per_second=0,
                tokens_per_second_total=0,
                token_source="api",
                exit_status="success" if lm_r.error_count == 0 else "error",
                error_message=lm_r.error_message,
                score_primary=accuracy,
                score_secondary={
                    "samples_run": lm_r.samples_run,
                    "error_count": lm_r.error_count,
                },
                scorer_version="lm_eval",
                score_explanation=f"lm_eval task: {task_names[0] if task_names else lm_r.task_id}, shots={lm_r.shots}, family={lm_r.family}",
                created_at=lm_r.created_at if lm_r.created_at else datetime.datetime.now().isoformat(),
            )

            store.save_run(result)
            all_results.append(__import__('dataclasses').asdict(result))

            if lm_r.error_count > 0:
                console.print(
                    f"  [red]  {lm_r.task_id} — Error:[/red] {lm_r.error_message}"
                )
            else:
                acc_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                console.print(
                    f"  [green]  {lm_r.task_id} — accuracy: {acc_str} "
                    f"[dim]({lm_r.samples_run} samples)[/dim]"
                )

    # Generate report
    runs_list = store.get_runs(suite_id=suite)
    report_md = generate_report(
        runs_list,
        suite,
        model_config,
        str(out_path / "report.md"),
        v2=report_v2,
    )

    console.print(f"\n[green]Report saved to: {out_path / 'report.md'}[/green]")
    console.print(f"[green]Database saved to: {out_path / 'benchmark.db'}[/green]")

    return all_results


def _check_lm_eval_available() -> bool:
    """Return True if lm-evaluation-harness is installed."""
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()

