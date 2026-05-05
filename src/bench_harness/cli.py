"""CLI for the benchmark harness."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from bench_harness.config import (
    load_model_config,
    get_model,
    load_suite_config,
    resolve_task_dir,
)
from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.runners.completion_runner import CompletionRunner
from bench_harness.tasks.loaders import load_tasks
from bench_harness.tasks.registry import TaskRegistry
from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.storage.artifacts import save_run_artifact
from bench_harness.reports.markdown import generate_report, print_summary

app = typer.Typer(
    name="bench-harness",
    help="Local LLM quality benchmark harness",
    add_completion=False,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@app.command()
def run(
    suite: str = typer.Option("smoke", "--suite", help="Suite name(s), comma-separated"),
    models: str = typer.Option("agent-code", "--models", help="Model alias(es), comma-separated"),
    endpoint: str | None = typer.Option(None, "--endpoint", help="Base URL override"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Max output tokens"),
    runs: int = typer.Option(1, "--runs", help="Number of repetitions per task"),
    out: str | None = typer.Option(None, "--out", help="Output directory"),
    timing_detail: bool = typer.Option(False, "--timing-detail", help="Show per-task timing in CLI output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print tasks without executing"),
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

        params = {
            "model_alias": alias,
            "model_backend": backend,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for task in tasks:
            for run_num in range(1, runs + 1):
                console.print(
                    f"  [dim]{task.get('id')} (run {run_num}/{runs})[/dim]"
                )

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

    # Generate report
    runs_list = store.get_runs(suite_id=suite_id)
    report_md = generate_report(runs_list, suite_id, model_config, str(out_path / "report.md"))

    # Print summary
    console.print(f"\n[yellow]=== Results ===[/yellow]")
    print_summary(runs_list, suite_id)

    console.print(f"\n[green]Report saved to: {out_path / 'report.md'}[/green]")
    console.print(f"[green]Database saved to: {out_path / 'benchmark.db'}[/green]")
    console.print(f"[green]Artifacts saved to: {out_path}/[/green]")

    return all_results


@app.command()
def list_tasks(
    family: str | None = typer.Option(None, "--family", help="Filter by task family"),
    source: str | None = typer.Option(None, "--source", help="Filter by task source"),
):
    """List all registered tasks, optionally filtered by family or source."""
    registry = TaskRegistry()

    # Auto-discover task directories from configs/
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    task_dirs = []

    # Check for known task directories
    root = Path(__file__).resolve().parent.parent.parent
    for td in ("tasks/smoke", "tasks/coding_smoke"):
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
        tasks = [t for t in tasks if t.family == family]
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
    for td in ("tasks/smoke", "tasks/coding_smoke"):
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


def main():
    """Entry point when called as `bench-harness`."""
    app()


if __name__ == "__main__":
    main()

