"""CLI for the benchmark harness."""

from __future__ import annotations

import asyncio
import json
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
from bench_harness.schemas import RunSpec, ModelArtifact
from bench_harness.storage.config import StorageConfig

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
    spec: str | None = typer.Argument(None, help="Path to run_spec.yaml or run_spec.json"),
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
    from bench_harness.schemas import RunSpec, build_run_spec_from_flags

    if spec is not None:
        spec_path = Path(spec)
        if not spec_path.exists():
            console.print(f"[red]Spec file not found: {spec_path}[/red]")
            sys.exit(1)
        if spec_path.suffix in (".yaml", ".yml"):
            run_spec = RunSpec.from_yaml(spec_path)
        else:
            run_spec = RunSpec.from_json(spec_path)
        suite_names = [run_spec.workload.prompt_suite]
        model_aliases = [run_spec.artifact.model_id or run_spec.artifact.path]
    else:
        run_spec = build_run_spec_from_flags(
            suite=suite,
            models=[m.strip() for m in models.split(",")],
            num_runs=runs,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=1,
            context_tokens=context_sizes.split(",")[0].strip(),
        )
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

    # Write resolved spec to result directory
    config = StorageConfig.from_env(allow_unsafe=True)
    run_dir = config.create_run_dir(run_spec.name)
    config.write_resolved_spec(run_spec, run_dir)
    logger.info("Wrote resolved spec to: %s", run_dir)

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


@app.command("init-storage")
def init_storage(
    root: str | None = typer.Option(None, "--root", help="Storage root directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be created without creating anything"),
    allow_unsafe: bool = typer.Option(False, "--allow-unsafe-storage-root", help="Allow storage inside git repo or other unsafe locations"),
):
    """Initialize storage directories for the benchmark harness."""
    from bench_harness.storage.config import StorageConfig

    if root is None:
        config = StorageConfig.from_env(allow_unsafe=allow_unsafe)
    else:
        config = StorageConfig.from_cli(Path(root), allow_unsafe=allow_unsafe)

    if dry_run:
        console.print(f"[cyan]Storage root:[/cyan] {config.root}")
        console.print()
        console.print("[dim]Directories that would be created:[/dim]")
        for ns_name in ("artifacts", "results", "registry", "logs", "cache", "tmp"):
            ns_path = getattr(config, f"{ns_name}_root")
            console.print(f"  - {ns_path}")
        # Also write project config
        project_config = Path.cwd() / ".llm-bench.yaml"
        console.print(f"  - {project_config} (project config)")
        console.print()
        console.print("[yellow]Dry run — no directories created.[/yellow]")
        return

    config.ensure_namespaces()

    # Write project config in cwd
    project_config = Path.cwd() / ".llm-bench.yaml"
    config_data = {
        "project": {
            "default_storage_root": str(config.root),
        }
    }
    project_config.write_text(_yaml_dump(config_data))
    console.print(f"[green]Storage initialized at:[/green] {config.root}")
    console.print(f"[green]Project config written to:[/green] {project_config}")

    console.print()
    console.print("[dim]Namespace paths:[/dim]")
    for ns_name in ("artifacts", "results", "registry", "logs", "cache", "tmp"):
        ns_path = getattr(config, f"{ns_name}_root")
        exists = " [green]✓[/green]" if ns_path.exists() else " [yellow]✗[/yellow]"
        console.print(f"  {ns_path}{exists}")


@app.command("storage-info")
def storage_info():
    """Print resolved storage configuration and validity status."""
    from bench_harness.storage.config import StorageConfig

    try:
        config = StorageConfig.from_env()
    except ValueError as e:
        console.print(f"[red]Storage config error:[/red] {e}")
        return

    console.print(f"[cyan]Storage root:[/cyan] {config.root}")
    console.print()

    # Check safety
    try:
        from bench_harness.storage.safety import check_storage_root
        check_storage_root(config.root)
        console.print("[green]✓ Storage root is valid[/green]")
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")

    console.print()
    console.print("[dim]Namespace paths:[/dim]")
    for ns_name in ("artifacts", "results", "registry", "logs", "cache", "tmp"):
        ns_path = getattr(config, f"{ns_name}_root")
        exists = " [green]exists[/green]" if ns_path.exists() else " [yellow]missing[/yellow]"
        console.print(f"  {ns_path}{exists}")


def _yaml_dump(data: dict) -> str:
    """Safely dump YAML without needing pyyaml as a hard import here."""
    import yaml  # noqa: F811 — re-import locally
    return yaml.safe_dump(data, default_flow_style=False)


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


# ── Dry-run utility ──────────────────────────────────────────────────


def _dry_run(spec: RunSpec, config: StorageConfig) -> None:
    """Print what a run would do without executing it."""
    run_dir = config.create_run_dir(spec.name)
    console.print(f"[bold]dry-run[/bold] Storage root: {config.root}")
    console.print(f"[bold]dry-run[/bold] Result directory: {run_dir}")
    console.print(f"[bold]dry-run[/bold] Artifact: {spec.artifact.kind} at {spec.artifact.path}")
    console.print(f"[bold]dry-run[/bold] Runtime: {spec.runtime.kind} (launch={spec.runtime.launch})")
    console.print(f"[bold]dry-run[/bold] Workload: {spec.workload.prompt_suite}, {spec.workload.num_runs} runs")
    console.print(f"[bold]dry-run[/bold] Would write results to: {run_dir}")


# ── New storage-aware commands ──────────────────────────────────────


@app.command("register-artifact")
def register_artifact(
    artifact_yaml: str = typer.Argument(..., help="Path to artifact YAML file"),
    storage_root: str | None = typer.Option(None, "--storage-root", help="Override storage root"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be registered without writing"),
    allow_unsafe_storage_root: bool = typer.Option(False, "--allow-unsafe-storage-root", help="Allow storage inside git repo or other unsafe locations"),
):
    """Register a model artifact in the benchmark registry."""
    from bench_harness.registry import ArtifactRegistry, manage_artifact
    from bench_harness.utils.hashing import compute_artifact_fingerprint, scan_artifact_path
    from bench_harness.storage.safety import detect_ephemeral_path
    from bench_harness.schemas.model_artifact import ArtifactMode

    config = StorageConfig.from_env()
    if storage_root:
        config = StorageConfig.from_cli(Path(storage_root), allow_unsafe=allow_unsafe_storage_root)
    else:
        config = StorageConfig.from_env(allow_unsafe=allow_unsafe_storage_root)

    console = Console()
    artifact = ModelArtifact.from_yaml(artifact_yaml)

    source_path = Path(artifact.source_path)

    if dry_run:
        console.print(f"[bold]dry-run[/bold] Would register artifact: {artifact.artifact_id}")
        console.print(f"[bold]dry-run[/bold] Kind: {artifact.kind}, Mode: {artifact.mode}")
        console.print(f"[bold]dry-run[/bold] Source: {artifact.source_path}")

        is_ephemeral, warnings = detect_ephemeral_path(artifact.source_path)
        if is_ephemeral:
            for w in warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

        if artifact.mode == ArtifactMode.managed_copy and source_path.exists():
            scan = scan_artifact_path(source_path)
            console.print(f"[bold]dry-run[/bold] Would copy to: {config.artifacts_models / artifact.artifact_id}")
            console.print(f"  Files: {scan['file_count']}, Size: {scan['total_size_bytes']:,} bytes")
        elif artifact.mode == ArtifactMode.managed_symlink and source_path.exists():
            console.print(f"[bold]dry-run[/bold] Would symlink to: {config.artifacts_models / artifact.artifact_id}")
        elif artifact.mode == ArtifactMode.external_path:
            console.print(f"[bold]dry-run[/bold] External path recorded (no copy)")

        console.print(f"[bold]dry-run[/bold] Registry: {config.registry_root / 'artifacts.jsonl'}")
        return

    # Handle artifact based on mode
    effective_path = manage_artifact(artifact, config)
    console.print(f"[green]Artifact managed:[/green] {effective_path}")

    # Compute fingerprint
    if source_path.exists():
        fingerprint = compute_artifact_fingerprint(artifact, source_path)
        artifact.artifact_fingerprint = fingerprint
    else:
        console.print(f"[yellow]Source path {source_path} does not exist, skipping fingerprint[/yellow]")

    # Register in the artifact registry
    registry = ArtifactRegistry(config)
    registry.register(artifact)

    # Write artifact manifest to registry location
    manifest_path = config.registry_root / "artifacts" / f"{artifact.artifact_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = artifact.model_dump(mode='python', default=str)
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2, default=str)

    console.print(f"[green]Registered artifact[/green]: {artifact.artifact_id}")
    console.print(f"Registry: {config.registry_root / 'artifacts.jsonl'}")
    console.print(f"Manifest: {manifest_path}")


@app.command("inspect-artifact")
def inspect_artifact(
    path: str = typer.Argument(..., help="Path to model artifact"),
    dry_run: bool = typer.Option(True, "--dry-run", help="Always implicit for inspect"),
):
    """Inspect a model artifact and print its metadata."""
    import os
    import hashlib
    from pathlib import Path

    from bench_harness.storage.safety import detect_ephemeral_path

    console = Console()
    p = Path(path)

    is_ephemeral, warnings = detect_ephemeral_path(path)
    if is_ephemeral:
        for w in warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")

    if not p.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)

    console.print(f"[bold]Path:[/bold] {p}")
    console.print(f"[bold]Is directory:[/bold] {p.is_dir()}")
    console.print(f"[bold]Ephemeral:[/bold] {'yes' if is_ephemeral else 'no'}")

    if p.is_dir():
        total_size = 0
        file_count = 0
        config_files = []
        for root, dirs, files in os.walk(p):
            for f in files:
                fp = Path(root) / f
                size = fp.stat().st_size
                total_size += size
                file_count += 1
                if f in ('config.json', 'model.safetensors.index.json', 'generation_config.json',
                         'tokenizer.json', 'tokenizer_config.json', 'config.yaml'):
                    config_files.append((str(fp.relative_to(p)), size))

        console.print(f"[bold]Total files:[/bold] {file_count}")
        console.print(f"[bold]Total size:[/bold] {total_size:,} bytes")
        if config_files:
            console.print(f"[bold]Config files:[/bold]")
            for cf, sz in config_files:
                console.print(f"  - {cf} ({sz:,} bytes)")
    else:
        console.print(f"[bold]Size:[/bold] {p.stat().st_size:,} bytes")


@app.command("list-runs")
def list_runs(
    project: str | None = typer.Option(None, "--project", help="Filter by project name"),
    storage_root: str | None = typer.Option(None, "--storage-root", help="Override storage root"),
):
    """List all benchmark runs in the storage."""
    config = StorageConfig.from_env()
    if storage_root:
        config = StorageConfig.from_cli(Path(storage_root))
    else:
        config = StorageConfig.from_env()

    from rich.table import Table

    console = Console()
    runs_dir = config.results_runs

    if not runs_dir.exists():
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID")
    table.add_column("Name")
    table.add_column("Project")
    table.add_column("Date")
    table.add_column("Runtime")

    run_count = 0
    for date_dir in sorted(runs_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            result_path = run_dir / "run_result.json"
            spec_path = run_dir / "resolved_spec.yaml"

            name = run_dir.name
            run_project = "unknown"
            runtime = "unknown"
            date_str = date_dir.name

            if spec_path.exists():
                import yaml
                with open(spec_path) as f:
                    spec_data = yaml.safe_load(f)
                run_project = spec_data.get("project", "unknown")
                runtime = spec_data.get("runtime", {}).get("kind", "unknown")

            if project and run_project != project:
                continue

            table.add_row(name, name.split("__")[0] if "__" in name else name, run_project, date_str, runtime)
            run_count += 1

    if run_count == 0:
        console.print("[yellow]No matching runs found.[/yellow]")
    else:
        console.print(f"\nTotal: {run_count} run(s)")


@app.command("summarize")
def summarize(
    project: str = typer.Argument(..., help="Project name to summarize"),
    storage_root: str | None = typer.Option(None, "--storage-root", help="Override storage root"),
    fmt: str = typer.Option("json", "--format", help="Output format: json or markdown"),
):
    """Summarize benchmark results for a project."""
    config = StorageConfig.from_env()
    if storage_root:
        config = StorageConfig.from_cli(Path(storage_root))
    else:
        config = StorageConfig.from_env()

    console = Console()
    runs_dir = config.results_runs

    summaries = []
    for date_dir in runs_dir.glob("*/**/summary.json"):
        if date_dir.exists():
            with open(date_dir) as f:
                data = json.load(f)
            spec_path = date_dir.parent / "resolved_spec.yaml"
            if spec_path.exists():
                import yaml
                with open(spec_path) as f2:
                    spec_data = yaml.safe_load(f2)
                if spec_data.get("project") == project:
                    summaries.append({
                        "run_dir": str(date_dir.parent),
                        "summary": data,
                    })

    if not summaries:
        console.print(f"[yellow]No results found for project: {project}[/yellow]")
        return

    if fmt == "markdown":
        console.print(f"[bold]Project: {project}[/bold]")
        console.print(f"Total runs: {len(summaries)}\n")
        for s in summaries:
            summary = s["summary"]
            console.print(f"  Run: {s['run_dir'].split('/')[-1]}")
            console.print(f"    Mean TTFT: {summary.get('mean_ttft_ms', 0):.0f}ms")
            console.print(f"    Mean decode TPS: {summary.get('mean_decode_tps', 0):.1f}")
            console.print(f"    Success rate: {summary.get('success_rate', 0):.0%}")
            console.print()
    else:
        output = {"project": project, "runs": summaries}
        console.print(json.dumps(output, indent=2, default=str))


@app.command("export-summary")
def export_summary(
    project: str = typer.Argument(..., help="Project name to export"),
    storage_root: str | None = typer.Option(None, "--storage-root", help="Override storage root"),
    fmt: str = typer.Option("markdown", "--format", help="Output format: markdown or json"),
):
    """Export project summary to a file."""
    config = StorageConfig.from_env()
    if storage_root:
        config = StorageConfig.from_cli(Path(storage_root))
    else:
        config = StorageConfig.from_env()

    console = Console()

    runs_dir = config.results_runs

    summaries = []
    for date_dir in runs_dir.glob("*/**/summary.json"):
        if date_dir.exists():
            with open(date_dir) as f:
                data = json.load(f)
            spec_path = date_dir.parent / "resolved_spec.yaml"
            if spec_path.exists():
                import yaml
                with open(spec_path) as f2:
                    spec_data = yaml.safe_load(f2)
                if spec_data.get("project") == project:
                    summaries.append({
                        "run_dir": str(date_dir.parent),
                        "summary": data,
                    })

    output_dir = config.results_summaries
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "markdown":
        output_path = output_dir / f"{project}_summary.md"
        with open(output_path, 'w') as f:
            f.write(f"# Project: {project}\n\n")
            f.write(f"Total runs: {len(summaries)}\n\n")
            for s in summaries:
                summary = s["summary"]
                f.write(f"## Run: {s['run_dir'].split('/')[-1]}\n\n")
                f.write(f"- Mean TTFT: {summary.get('mean_ttft_ms', 0):.0f}ms\n")
                f.write(f"- Mean decode TPS: {summary.get('mean_decode_tps', 0):.1f}\n")
                f.write(f"- Success rate: {summary.get('success_rate', 0):.0%}\n\n")
    else:
        output_path = output_dir / f"{project}_summary.json"
        with open(output_path, 'w') as f:
            json.dump({"project": project, "runs": summaries}, f, indent=2, default=str)

    console.print(f"[green]Exported[/green]: {output_path}")


@app.command("compare")
def compare(
    run_id_a: str = typer.Argument(..., help="First run ID (partial match on run directory name)"),
    run_id_b: str = typer.Argument(..., help="Second run ID (partial match on run directory name)"),
    storage_root: str | None = typer.Option(None, "--storage-root", help="Override storage root"),
    allow_unsafe: bool = typer.Option(False, "--allow-unsafe-storage-root", help="Allow storage inside git repo or other unsafe locations"),
):
    """Compare two benchmark runs by result directory."""
    from rich.table import Table

    config = StorageConfig.from_env()
    if storage_root:
        config = StorageConfig.from_cli(Path(storage_root), allow_unsafe=allow_unsafe)
    else:
        config = StorageConfig.from_env(allow_unsafe=allow_unsafe)

    console = Console()

    runs_dir = config.results_runs

    def find_run(run_id: str):
        for date_dir in runs_dir.glob("*/"):
            if not date_dir.is_dir():
                continue
            for run_dir in date_dir.glob("*/"):
                if run_id in run_dir.name:
                    return run_dir
        return None

    result_a = find_run(run_id_a)
    result_b = find_run(run_id_b)

    if not result_a or not result_b:
        console.print(f"[red]Could not find both run directories. Search path: {runs_dir}[/red]")
        if not result_a:
            console.print(f"  Missing: '{run_id_a}'")
        if not result_b:
            console.print(f"  Missing: '{run_id_b}'")
        raise typer.Exit(1)

    summary_a_path = result_a / "summary.json"
    summary_b_path = result_b / "summary.json"

    if not summary_a_path.exists() or not summary_b_path.exists():
        console.print("[red]Both runs must have a summary.json file[/red]")
        raise typer.Exit(1)

    with open(summary_a_path) as f:
        summary_a = json.load(f)
    with open(summary_b_path) as f:
        summary_b = json.load(f)

    # Also load specs for metadata
    spec_a = None
    spec_b = None
    spec_a_path = result_a / "resolved_spec.yaml"
    spec_b_path = result_b / "resolved_spec.yaml"
    if spec_a_path.exists():
        import yaml
        with open(spec_a_path) as f:
            spec_a = yaml.safe_load(f)
    if spec_b_path.exists():
        import yaml
        with open(spec_b_path) as f:
            spec_b = yaml.safe_load(f)

    console.print(f"[bold]Comparing:[/bold] {result_a.name} vs {result_b.name}\n")

    if spec_a or spec_b:
        for label, spec in [("Run A", spec_a), ("Run B", spec_b)]:
            if spec:
                console.print(f"[dim]{label}: project={spec.get('project', 'N/A')}, artifact={spec.get('artifact', {}).get('kind', 'N/A')}[/dim]")

    console.print()

    keys = ["mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms",
            "mean_decode_tps", "median_decode_tps", "p95_decode_tps",
            "success_rate"]
    key_labels = {
        "mean_ttft_ms": "Mean TTFT (ms)",
        "median_ttft_ms": "Median TTFT (ms)",
        "p95_ttft_ms": "P95 TTFT (ms)",
        "mean_decode_tps": "Mean Decode TPS",
        "median_decode_tps": "Median Decode TPS",
        "p95_decode_tps": "P95 Decode TPS",
        "success_rate": "Success Rate",
    }

    table = Table(title="Metrics Comparison")
    table.add_column("Metric", style="bold")
    table.add_column("Run A", justify="right")
    table.add_column("Run B", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Flag", justify="center")

    for key in keys:
        val_a = summary_a.get(key, 0)
        val_b = summary_b.get(key, 0)
        delta = val_b - val_a

        if key == "success_rate":
            fmt_a = f"{val_a:.0%}"
            fmt_b = f"{val_b:.0%}"
            delta_str = f"{delta:+.0%}"
        else:
            fmt_a = f"{val_a:.2f}"
            fmt_b = f"{val_b:.2f}"
            delta_str = f"{delta:+.2f}"

        # Flag: green for improvement, red for regression
        flag = ""
        if key == "success_rate":
            if delta > 0:
                flag = "[green]▲[/green]"
            elif delta < 0:
                flag = "[red]▼[/red]"
        elif "tps" in key:
            if delta > 0:
                flag = "[green]▲[/green]"
            elif delta < 0:
                flag = "[red]▼[/red]"
        elif "ttft" in key:
            if delta < 0:
                flag = "[green]▲[/green]"
            elif delta > 0:
                flag = "[red]▼[/red]"

        table.add_row(key_labels[key], fmt_a, fmt_b, delta_str, flag)

    console.print(table)


@app.command("compare-runs-v2")
def compare_v2(
    run_id_a: str = typer.Argument(..., help="First run identifier (partial match on run dir name)"),
    run_id_b: str = typer.Argument(..., help="Second run identifier (partial match on run dir name)"),
    storage_root: str | None = typer.Option(None, "--storage-root", help="Override storage root"),
):
    """Compare two benchmark runs by result directory."""
    config = StorageConfig.from_env()
    if storage_root:
        config = StorageConfig.from_cli(Path(storage_root))
    else:
        config = StorageConfig.from_env()

    console = Console()

    runs_dir = config.results_runs

    def find_run(run_id: str):
        for date_dir in runs_dir.glob("*/"):
            for run_dir in date_dir.glob("*/"):
                if run_id in run_dir.name:
                    return run_dir
        return None

    result_a = find_run(run_id_a)
    result_b = find_run(run_id_b)

    if not result_a or not result_b:
        console.print("[red]Could not find both run directories.[/red]")
        raise typer.Exit(1)

    with open(result_a / "summary.json") as f:
        summary_a = json.load(f)
    with open(result_b / "summary.json") as f:
        summary_b = json.load(f)

    console.print(f"[bold]Comparing:[/bold] {result_a.name} vs {result_b.name}\n")

    keys = ["mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms",
            "mean_decode_tps", "median_decode_tps", "p95_decode_tps",
            "success_rate"]
    for key in keys:
        val_a = summary_a.get(key, 0)
        val_b = summary_b.get(key, 0)
        delta = val_b - val_a
        sign = "+" if delta >= 0 else ""
        console.print(f"  {key}: {val_a:.2f} -> {val_b:.2f} ({sign}{delta:.2f})")


@app.command("bench-run")
def bench_run(
    spec: str = typer.Argument(None, help="Path to run_spec.yaml or run_spec.json"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be done without executing"),
    allow_unsafe: bool = typer.Option(False, "--allow-unsafe-storage-root", help="Allow storage inside git repo or other unsafe locations"),
):
    """Run a benchmark using the RunSpec + RuntimeRunner interface (M17-M19)."""
    from bench_harness.runners import get_runner, RUNNER_REGISTRY

    # Load spec
    if spec is None:
        console.print("[red]Spec file is required. Provide a path to run_spec.yaml or run_spec.json[/red]")
        sys.exit(1)

    spec_path = Path(spec)
    if not spec_path.exists():
        console.print(f"[red]Spec file not found: {spec_path}[/red]")
        sys.exit(1)

    if spec_path.suffix in (".yaml", ".yml"):
        run_spec = RunSpec.from_yaml(spec_path)
    else:
        run_spec = RunSpec.from_json(spec_path)

    console.print(f"[cyan]=== Benchmark Run ===[/cyan]")
    console.print(f"[dim]Name: {run_spec.name} | Project: {run_spec.project}[/dim]")
    console.print(f"[dim]Artifact: {run_spec.artifact.kind.value} at {run_spec.artifact.path}[/dim]")
    console.print(f"[dim]Runtime: {run_spec.runtime.kind.value} (launch={run_spec.runtime.launch.value})[/dim]")
    console.print(f"[dim]Workload: suite={run_spec.workload.prompt_suite}, runs={run_spec.workload.num_runs}[/dim]")

    # Resolve storage config
    storage_config = StorageConfig.from_env(allow_unsafe=allow_unsafe)

    if dry_run:
        _dry_run(run_spec, storage_config)
        return

    # Create run directory
    run_dir = storage_config.create_run_dir(run_spec.name)
    storage_config.write_resolved_spec(run_spec, run_dir)
    console.print(f"[dim]Run directory: {run_dir}[/dim]")

    # Resolve runner by kind
    runner_kind = run_spec.runtime.kind.value
    runner = get_runner(runner_kind, storage_config)

    console.print(f"[dim]Runner: {runner.kind}[/dim]")

    # Prepare
    prep = runner.prepare(run_spec)
    console.print("[dim]Preparing runtime...[/dim]")

    # Launch if managed
    handle = None
    if run_spec.runtime.launch.value == "managed_process":
        handle = runner.launch(run_spec, prep)
        if handle is not None:
            console.print(f"[dim]Launching process on {handle.host}:{handle.port}...[/dim]")
            ready = runner.wait_until_ready(run_spec, prep, timeout=120.0)
            if not ready:
                console.print("[red]Runtime failed to become ready within timeout[/red]")
                runner.shutdown(run_spec, prep, handle)
                sys.exit(1)
            console.print("[green]Runtime is ready[/green]")
    else:
        console.print("[dim]Using external/existing runtime[/dim]")

    try:
        # Run workload
        result_dir = run_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        console.print("[dim]Running workload...[/dim]")
        result = runner.run_workload(run_spec, prep, result_dir)

        console.print(f"[green]Workload complete[/green]")
        console.print(f"  Schema: {result.schema_version}")
        console.print(f"  Run ID: {result.run_id}")
        console.print(f"  Requests: {len(result.per_request)}")

        if result.summary:
            console.print(f"  Success rate: {result.summary.success_rate:.0%}")
            console.print(f"  Mean TTFT: {result.summary.mean_ttft_ms:.0f}ms")
            console.print(f"  Mean decode TPS: {result.summary.mean_decode_tps:.1f}")

        # Write results to run directory
        result.write_to_directory(run_dir)
        console.print(f"[green]Results written to {run_dir}[/green]")

        # Collect logs
        logs = runner.collect_logs(run_spec, prep, result_dir)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in logs.items():
            log_path = logs_dir / filename
            log_path.write_text(content)
            console.print(f"  [dim]Log: {log_path}[/dim]")

    finally:
        if handle is not None:
            runner.shutdown(run_spec, prep, handle)
            console.print("[dim]Shutdown complete[/dim]")


prompt_opt_app = typer.Typer(
    name="prompt-opt",
    help="Automatic prompt optimization — analyze styles, propose templates, run candidates",
    add_completion=False,
)


@prompt_opt_app.command("analyze")
def prompt_opt_analyze(
    suite: str = typer.Option("smoke", "--suite", help="Suite ID to analyze"),
    db: Path = typer.Option(None, "--db", help="Path to benchmark.db"),
    min_runs: int = typer.Option(1, "--min-runs", help="Minimum runs per style to consider valid"),
):
    """Analyze existing benchmark results to rank prompt styles per task family."""
    from bench_harness.prompt_optimization import PromptOptimizationRunner
    from rich.table import Table

    if db is None:
        # Auto-discover: look for benchmark.db in recent run directories
        runs_dir = Path("runs")
        candidates = sorted(runs_dir.glob("*/benchmark.db"), reverse=True)
        if not candidates:
            console.print("[red]No benchmark.db found. Specify --db or run a benchmark suite first.[/red]")
            return
        db = candidates[0]

    runner = PromptOptimizationRunner()
    analysis = runner.analyze(str(db), suite_id=suite, min_runs_per_style=min_runs)

    console.print(f"[cyan]Prompt Analysis — suite: {suite}[/cyan]")
    console.print(f"[dim]DB: {db} | Total style runs: {analysis.total_style_runs} | Styles: {', '.join(analysis.all_styles) if analysis.all_styles else 'none'}[/dim]")
    console.print()

    if not analysis.all_styles:
        console.print("[yellow]No style-tagged runs found in this suite.[/yellow]")
        return

    # Best style overall
    if analysis.best_style_overall:
        console.print(f"[green]Best style overall:[/green] [bold]{analysis.best_style_overall}[/bold]")
    console.print()

    # Family rankings
    if analysis.family_rankings:
        table = Table(title="Best Style Per Task Family")
        table.add_column("Family", style="cyan")
        table.add_column("Best Style", style="green")
        table.add_column("Avg Score", justify="right")
        table.add_column("Margin", justify="right")
        table.add_column("Tasks", justify="right")

        for family, rankings in sorted(analysis.family_rankings.items()):
            if rankings:
                best = rankings[0]
                tasks_run = sum(1 for r in rankings)
                table.add_row(
                    family,
                    best[0],
                    f"{best[1]:.3f}",
                    f"{best[2]:+.3f}",
                    str(tasks_run),
                )
        console.print(table)

        # Insufficient data warning
        if analysis.insufficient_data:
            console.print()
            console.print(
                f"[yellow]⚠ Insufficient data (<{min_runs} runs/style) for: "
                f"{', '.join(analysis.insufficient_data)}[/yellow]"
            )
    else:
        console.print("[yellow]No family data available (tasks may lack family metadata).[/yellow]")

    # Style variance
    if analysis.style_variances:
        console.print()
        console.print("[dim]Score Variance by Style (higher = more task-dependent):[/dim]")
        for style, var in sorted(analysis.style_variances.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(var * 20)
            console.print(f"  {style:20s} {var:.4f} {bar}")


@prompt_opt_app.command("propose")
def prompt_opt_propose(
    suite: str = typer.Option("smoke", "--suite", help="Suite ID to analyze"),
    db: Path = typer.Option(None, "--db", help="Path to benchmark.db"),
    family: str = typer.Option(None, "--family", help="Filter to a specific task family"),
):
    """Propose new candidate prompt templates based on failure patterns."""
    from bench_harness.storage.sqlite import BenchmarkDB
    from bench_harness.prompt_optimization import generate_proposals

    if db is None:
        runs_dir = Path("runs")
        candidates = sorted(runs_dir.glob("*/benchmark.db"), reverse=True)
        if not candidates:
            console.print("[red]No benchmark.db found. Specify --db or run a benchmark suite first.[/red]")
            return
        db = candidates[0]

    bench_db = BenchmarkDB(str(db))
    runs = bench_db.get_runs(suite_id=suite)

    if not runs:
        console.print(f"[yellow]No runs found for suite '{suite}'.[/yellow]")
        return

    proposals = generate_proposals(runs, task_family=family or "")

    if not proposals:
        console.print("[cyan]No pattern-based proposals generated.[/cyan]")
        console.print("[dim]This usually means: (1) insufficient data, (2) no failure patterns matched, or (3) styles are well-balanced.[/dim]")
        console.print("[dim]Use `bench-harness prompt-opt run` to manually test custom templates.[/dim]")
        return

    console.print(f"[cyan]Proposed {len(proposals)} candidate template(s):[/cyan]")
    console.print()

    for i, p in enumerate(proposals, 1):
        console.print(f"[bold][{i}] {p.name}[/bold] [dim](target: {p.baseline})[/dim]")
        console.print(f"  [yellow]Why:[/yellow] {p.instructions}")
        console.print(f"  [dim]Template:[/dim] {p.template_str[:120]}{'...' if len(p.template_str) > 120 else ''}")
        if p.task_family:
            console.print(f"  [dim]Family:[/dim] {p.task_family}")
        console.print()

    # Also list predefined variants that could be tried
    from bench_harness.prompt_optimization.proposals import _PREDEFINED_VARIANTS
    console.print("[dim]Other predefined variants available via custom YAML:[/dim]")
    for variant_name in sorted(_PREDEFINED_VARIANTS.keys()):
        if not any(p.name == variant_name for p in proposals):
            info = _PREDEFINED_VARIANTS[variant_name]
            console.print(f"  - [cyan]{variant_name}[/cyan]: {info['instructions'][:80]}")


@prompt_opt_app.command("run-proposed")
def prompt_opt_run_proposed(
    spec: Path = typer.Option(..., "--spec", help="Path to proposals YAML spec file"),
    models: str = typer.Option("agent-code", "--models", help="Model alias(es), comma-separated"),
    suite: str = typer.Option("smoke", "--suite", help="Suite ID"),
    db: Path = typer.Option(None, "--db", help="Path to benchmark.db (for task loading)"),
    out: Path = typer.Option(None, "--out", help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be tested without running"),
):
    """Run proposed candidate templates against tasks for evaluation."""
    from bench_harness.prompt_optimization import PromptOptimizationRunner, TemplateRegistry, load_proposals_from_yaml
    from bench_harness.config import load_model_config, get_model
    from bench_harness.models.openai_client import OpenAICompatClient
    from bench_harness.runners.completion_runner import CompletionRunner
    import asyncio

    # Load proposals from YAML spec
    proposals = load_proposals_from_yaml(spec)
    if not proposals:
        console.print("[yellow]No candidates found in spec file.[/yellow]")
        return

    model_aliases = [m.strip() for m in models.split(",")]
    model_config = load_model_config()

    # Resolve endpoint
    first_alias = model_aliases[0]
    first_model = get_model(model_config, first_alias)
    if first_model is None:
        console.print(f"[red]Model '{first_alias}' not found in config[/red]")
        return
    base_url = first_model.get("base_url", "")
    if not base_url:
        console.print("[red]No base_url configured[/red]")
        return
    model_name = first_model.get("model", first_alias)

    registry = TemplateRegistry()
    registry.add_baseline("plain")
    registry.add_candidates(proposals)

    # Determine DB path
    if db is None:
        runs_dir = Path("runs")
        candidates = sorted(runs_dir.glob("*/benchmark.db"), reverse=True)
        if not candidates:
            console.print("[red]No benchmark.db found. Specify --db.[/red]")
            return
        db = candidates[0]

    if out is None:
        out = Path(f"runs/prompt-opt-{proposals[0].name}-{suite}-{db.parent.name}")

    if dry_run:
        console.print(f"[cyan]Dry run — would test {len(proposals)} candidate(s)[/cyan]")
        console.print(f"  Models: {', '.join(model_aliases)}")
        console.print(f"  Suite: {suite}")
        console.print(f"  DB: {db}")
        console.print()
        for p in proposals:
            console.print(f"  [bold]{p.name}[/bold] [dim](baseline: {p.baseline})[/dim]")
            console.print(f"    [dim]{p.instructions[:80]}...[/dim]")
        return

    # Setup runner
    client = OpenAICompatClient(base_url=base_url, model=model_name)
    runner = CompletionRunner(client)

    opt_runner = PromptOptimizationRunner()
    opt_runner.base_runner = runner

    console.print(f"[cyan]=== Running Prompt Optimization ===[/cyan]")
    console.print(f"  Models: {', '.join(model_aliases)}")
    console.print(f"  Suite: {suite}")
    console.print(f"  DB: {db}")
    console.print(f"  Candidates: {len(proposals)}")
    console.print()

    results = opt_runner.run_candidates(
        registry=registry,
        db_path=str(db),
        model_aliases=model_aliases,
        suite_id=suite,
        output_dir=str(out),
    )

    if not results:
        console.print("[yellow]No results produced.[/yellow]")
        return

    # Print results
    from rich.table import Table
    table = Table(title="Candidate Results")
    table.add_column("Candidate", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Status", style="dim")

    for r in sorted(results, key=lambda x: x.get("score_delta", -999), reverse=True):
        delta_str = f"{r['score_delta']:+.3f}" if r.get("score_delta") is not None else "N/A"
        baseline_str = f"{r.get('baseline_score', 0):.3f}" if r.get('baseline_score') is not None else "-"
        table.add_row(
            r["name"],
            f"{r.get('score', 0):.3f}",
            baseline_str,
            delta_str,
            str(r.get('run_count', 0)),
            r.get('status', 'unknown'),
        )
    console.print(table)

    # Recommendations
    recommended = [r for r in results if r.get("score_delta", 0) > 0.05]
    if recommended:
        console.print()
        console.print("[green]Recommendations:[/green]")
        for r in recommended:
            console.print(
                f"  - `{r['name']}` scores [green]{r['score_delta']:+.3f}[/green] vs "
                f"baseline `{r['baseline']}`"
            )
    else:
        console.print()
        console.print("[yellow]No candidates exceeded the 0.05 improvement threshold.[/yellow]")

    # Generate full report
    analysis = opt_runner.analyze(str(db), suite_id=suite)
    report = opt_runner.generate_report(analysis, results)
    report_path = out / "optimization_report.md"
    report_path.write_text(report)
    console.print(f"\n[green]Report saved to:[/green] {report_path}")

    return results


@prompt_opt_app.command("run")
def prompt_opt_run(
    templates: str = typer.Option(..., "--templates", help="Comma-separated template names/styles to test"),
    base_styles: str = typer.Option("plain", "--base-styles", help="Comma-separated baseline styles for comparison"),
    models: str = typer.Option("agent-code", "--models", help="Model alias(es), comma-separated"),
    suite: str = typer.Option("smoke", "--suite", help="Suite ID"),
    db: Path = typer.Option(None, "--db", help="Path to benchmark.db"),
    out: Path = typer.Option(None, "--out", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be tested without running"),
):
    """Run a custom prompt optimization sweep with user-specified templates."""
    from bench_harness.prompt_optimization import PromptOptimizationRunner, TemplateRegistry
    from bench_harness.config import load_model_config, get_model
    from bench_harness.models.openai_client import OpenAICompatClient
    from bench_harness.runners.completion_runner import CompletionRunner

    template_list = [t.strip() for t in templates.split(",")]
    baseline_list = [s.strip() for s in base_styles.split(",")]
    model_aliases = [m.strip() for m in models.split(",")]

    model_config = load_model_config()
    first_alias = model_aliases[0]
    first_model = get_model(model_config, first_alias)
    if first_model is None:
        console.print(f"[red]Model '{first_alias}' not found in config[/red]")
        return
    base_url = first_model.get("base_url", "")
    if not base_url:
        console.print("[red]No base_url configured[/red]")
        return
    model_name = first_model.get("model", first_alias)

    if db is None:
        runs_dir = Path("runs")
        candidates = sorted(runs_dir.glob("*/benchmark.db"), reverse=True)
        if not candidates:
            console.print("[red]No benchmark.db found. Specify --db.[/red]")
            return
        db = candidates[0]

    if out is None:
        out = Path(f"runs/prompt-opt-custom-{suite}-{db.parent.name}")

    if dry_run:
        console.print(f"[cyan]Dry run — custom prompt sweep[/cyan]")
        console.print(f"  Templates to test: {', '.join(template_list)}")
        console.print(f"  Baselines: {', '.join(baseline_list)}")
        console.print(f"  Models: {', '.join(model_aliases)}")
        console.print(f"  Suite: {suite}")
        console.print(f"  DB: {db}")
        return

    client = OpenAICompatClient(base_url=base_url, model=model_name)
    runner = CompletionRunner(client)

    opt_runner = PromptOptimizationRunner()
    opt_runner.base_runner = runner

    # Build candidates from template names
    registry = TemplateRegistry()
    for style in baseline_list:
        registry.add_baseline(style)
    for template in template_list:
        if template not in baseline_list:
            from bench_harness.prompt_optimization.proposals import TemplateProposal
            registry.add_candidate(TemplateProposal(
                name=template,
                template_str=f"{{{{ user_message }}}}",  # Will use the style file directly
                baseline=baseline_list[0],
                instructions=f"Custom template: {template}",
            ))

    console.print(f"[cyan]=== Custom Prompt Optimization Sweep ===[/cyan]")
    console.print(f"  Testing: {', '.join(template_list)}")
    console.print(f"  Baselines: {', '.join(baseline_list)}")
    console.print(f"  Models: {', '.join(model_aliases)}")
    console.print()

    results = opt_runner.run_candidates(
        registry=registry,
        db_path=str(db),
        model_aliases=model_aliases,
        suite_id=suite,
        output_dir=str(out),
    )

    if not results:
        console.print("[yellow]No results produced.[/yellow]")
        return

    from rich.table import Table
    table = Table(title="Custom Sweep Results")
    table.add_column("Template", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Status", style="dim")

    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        table.add_row(
            r["name"],
            f"{r.get('score', 0):.3f}",
            str(r.get('run_count', 0)),
            r.get('status', 'unknown'),
        )
    console.print(table)

    # Save report
    analysis = opt_runner.analyze(str(db), suite_id=suite)
    report = opt_runner.generate_report(analysis, results)
    report_path = out / "optimization_report.md"
    report_path.write_text(report)
    console.print(f"\n[green]Report saved to:[/green] {report_path}")

    return results


# Register apps
app.add_typer(prompt_opt_app, name="prompt-opt")


# ── Entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    main()

