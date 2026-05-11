"""Service for running benchmarks and tracking progress."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Optional

from bench_harness.config import load_model_config, get_model, get_quantization, resolve_task_dir, load_suite_config, get_suite
from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.runners.completion_runner import CompletionRunner
from bench_harness.runners.style_sweep_runner import StyleSweepRunner
from bench_harness.runners.context_sweep_runner import ContextSizeSweepRunner
from bench_harness.tasks.loaders import load_tasks
from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.storage.artifacts import save_run_artifact
from bench_harness.reports.markdown import generate_report
from bench_harness.server.services.config_service import build_run_spec

logger = logging.getLogger(__name__)


class RunState:
    """Tracks the state of a running benchmark."""
    def __init__(self, run_id: str, config_id: Optional[str], config_name: Optional[str],
                 model_alias: str, suite_id: str, total_tasks: int):
        self.run_id = run_id
        self.config_id = config_id
        self.config_name = config_name
        self.status = "running"
        self.model_alias = model_alias
        self.suite_id = suite_id
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.started_at = datetime.datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.results_path: Optional[str] = None
        self.error_message: Optional[str] = None
        self.current_task: Optional[str] = None
        self.current_style: Optional[str] = None
        self.events: list[dict] = []
        self._lock = threading.Lock()


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def launch_run(
    config_data: dict[str, Any],
    storage_root: Path,
    on_event: Optional[callable] = None,
) -> dict[str, str]:
    """Start a benchmark run in a background thread. Returns run_id immediately."""
    run_id = uuid.uuid4().hex[:12]
    config = config_data
    config_id = config.get("id", "")
    config_name = config.get("name", config_id)

    # Parse the config to determine model and suite
    workload = config.get("workload", {})
    artifact = config.get("artifact", {})
    runtime = config.get("runtime", {})

    model_alias = artifact.get("model_id", "")
    if not model_alias:
        # Try to get model alias from runtime model_name
        model_alias = runtime.get("model_name", "unknown-model")

    suite_id = workload.get("prompt_suite", "smoke")
    styles = config.get("advanced", {}).get("styles", [])
    context_sizes = config.get("advanced", {}).get("context_sizes", [])

    # Load model config to resolve actual model name
    model_name = model_alias
    try:
        model_cfg = load_model_config()
        m = get_model(model_cfg, model_alias)
        if m:
            model_name = m.get("model", model_alias)
    except Exception:
        pass

    # Estimate total tasks
    try:
        task_dir_path = resolve_task_dir(suite_id)
        tasks = load_tasks(str(task_dir_path))
        num_tasks = len(tasks)
    except Exception:
        tasks = []
        num_tasks = 0

    # Apply style/context multipliers
    if styles:
        num_tasks *= len(styles)
    if context_sizes and len(context_sizes) > 1:
        num_tasks *= len(context_sizes)

    total_runs = workload.get("num_runs", 1)
    if total_runs > 1:
        num_tasks *= total_runs

    state = RunState(run_id, config_id, config_name, model_alias, suite_id, num_tasks)
    _active_runs[run_id] = state

    # Broadcast start event
    _emit(state, {
        "type": "started",
        "run_id": run_id,
        "model_alias": model_alias,
        "suite_id": suite_id,
    })

    # Start background thread
    thread = threading.Thread(
        target=_run_worker,
        args=(run_id, config, storage_root),
        daemon=True,
    )
    thread.start()

    return {"run_id": run_id, "status": "queued"}


def get_run_state(run_id: str) -> Optional[dict[str, Any]]:
    """Get current state of a run."""
    state = _active_runs.get(run_id)
    if not state:
        return None
    return {
        "run_id": state.run_id,
        "config_id": state.config_id,
        "config_name": state.config_name,
        "status": state.status,
        "model_alias": state.model_alias,
        "suite_id": state.suite_id,
        "total_tasks": state.total_tasks,
        "completed_tasks": state.completed_tasks,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "results_path": state.results_path,
        "error_message": state.error_message,
        "current_task": state.current_task,
        "current_style": state.current_style,
        "events": state.events[-50:],  # Last 50 events
    }


def cancel_run(run_id: str) -> bool:
    """Cancel a running benchmark (best-effort)."""
    state = _active_runs.get(run_id)
    if state and state.status == "running":
        state.status = "cancelled"
        state.completed_at = _now_utc()
        state.error_message = "Cancelled by user"
        return True
    return False


def _run_worker(run_id: str, config: dict, storage_root: Path) -> None:
    """Worker that executes the benchmark in the background."""
    state = _active_runs.get(run_id)
    if not state:
        return

    try:
        _execute_run(state, config, storage_root)
    except Exception as e:
        logger.exception("Run %s failed", run_id)
        state.status = "error"
        state.error_message = str(e)
        state.completed_at = _now_utc()
        _emit(state, {"type": "error", "error": str(e)})


def _execute_run(state: RunState, config: dict, storage_root: Path) -> None:
    """Execute the actual benchmark."""
    state.status = "running"

    workload = config.get("workload", {})
    artifact = config.get("artifact", {})
    runtime = config.get("runtime", {})
    advanced = config.get("advanced", {})

    # Resolve model
    model_alias = artifact.get("model_id", "")
    if not model_alias:
        model_alias = runtime.get("model_name", "unknown-model")

    # Get base_url
    base_url = runtime.get("host", "") or artifact.get("path", "")
    if not base_url:
        try:
            model_cfg = load_model_config()
            m = get_model(model_cfg, model_alias)
            if m:
                base_url = m.get("base_url", "")
        except Exception:
            pass

    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:4000/v1")

    model_name = artifact.get("model_id", "") or runtime.get("model_name", model_alias)
    try:
        model_cfg = load_model_config()
        m = get_model(model_cfg, model_alias)
        if m:
            model_name = m.get("model", model_alias)
            base_url = m.get("base_url", base_url)
    except Exception:
        pass

    quant = get_quantization(model_cfg, model_alias) if 'model_cfg' in dir() else None
    try:
        quant = get_quantization(model_cfg, model_alias)
    except Exception:
        quant = None

    # Resolve suite and tasks
    suite_id = workload.get("prompt_suite", "smoke")
    try:
        task_dir_path = resolve_task_dir(suite_id)
        tasks = load_tasks(str(task_dir_path))
    except Exception as e:
        state.status = "error"
        state.error_message = f"Failed to load tasks: {e}"
        state.completed_at = _now_utc()
        return

    # Parameters
    temperature = workload.get("temperature", 0.0)
    max_tokens = workload.get("max_tokens", 256)
    num_runs = workload.get("num_runs", 1)
    style_list = advanced.get("styles", []) or []
    context_size_list = advanced.get("context_sizes", []) or []
    judge_enabled = advanced.get("judge", False)

    # Setup storage
    run_dir = storage_root / "results" / "runs" / datetime.datetime.now().strftime("%Y-%m-%d") / f"{state.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(run_dir / "benchmark.db")
    store = SQLiteStore(db_path)
    store.init()

    state.results_path = str(run_dir)
    state.completed_tasks = 0

    # Client and runner
    client = OpenAICompatClient(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        model=model_name,
        timeout=300.0,
        max_retries=3,
    )
    runner = CompletionRunner(client)

    # Identity stamp
    identity = {
        "backend_url": base_url,
        "openai_models_id": None,
        "vllm_served_model_name": None,
        "vllm_container_name": None,
        "hf_model_id": None,
        "server_start_time": None,
        "speculative_decoding_enabled": None,
        "requested_alias": model_alias,
        "litellm_model_name": model_name,
    }

    # Build params
    params = {
        "model_alias": model_alias,
        "model_backend": "",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "quantization": quant,
    }

    # Execute tasks
    all_results = []
    task_idx = 0

    for task in tasks:
        state.current_task = task.get("id", "unknown")

        styles_to_run = style_list if style_list else ["plain"]
        contexts_to_run = context_size_list if len(context_size_list) > 1 else [""]

        for style in styles_to_run:
            state.current_style = style
            for ctx_size in contexts_to_run:
                for run_num in range(1, num_runs + 1):
                    task_params = dict(params)
                    task_params.update(identity)

                    if style and style != "plain":
                        task_params["prompt_style"] = style

                    if ctx_size:
                        task_params["context_tokens"] = ctx_size

                    # Use StyleSweepRunner or ContextSizeSweepRunner if multi
                    if style_list and len(style_list) > 1:
                        sweep = StyleSweepRunner(runner, styles=style_list)
                        results = asyncio.run(
                            sweep.run_sweep([task], [model_alias], task_params, suite_id=suite_id)
                        )
                        results_to_save = results
                    elif context_size_list and len(context_size_list) > 1:
                        sweep = ContextSizeSweepRunner(runner, sizes=context_size_list)
                        results = asyncio.run(
                            sweep.run_sweep([task], [model_alias], task_params, suite_id=suite_id)
                        )
                        results_to_save = results
                    else:
                        result = asyncio.run(
                            runner.run(task, task_params, suite_id=suite_id)
                        )
                        results_to_save = [result]

                    for result in results_to_save:
                        store.save_run(result)
                        save_run_artifact(result, str(run_dir))
                        all_results.append(result)

                        task_idx += 1
                        state.completed_tasks += 1

                        _emit(state, {
                            "type": "task_completed",
                            "run_id": run_id,
                            "task_id": task.get("id", ""),
                            "model_alias": model_alias,
                            "score": result.score_primary,
                            "ttft_ms": result.ttft_ms,
                            "tokens_per_second": result.tokens_per_second,
                            "exit_status": result.exit_status,
                        })

        state.current_task = None
        state.current_style = None

    state.completed_tasks = task_idx

    # Generate report
    try:
        runs_list = store.get_runs(suite_id=suite_id)
        report_path = run_dir / "report.md"
        generate_report(
            runs_list, suite_id,
            model_cfg if 'model_cfg' in dir() else {},
            str(report_path),
            v2=True,
        )
    except Exception:
        pass

    state.status = "completed"
    state.completed_at = _now_utc()
    _emit(state, {
        "type": "completed",
        "run_id": run_id,
        "results_path": str(run_dir),
        "total_results": len(all_results),
    })


_active_runs: dict[str, RunState] = {}


def _emit(state: RunState, event: dict) -> None:
    """Add event to state and notify listeners."""
    with state._lock:
        state.events.append(event)


def list_completed_runs(storage_root: Path, limit: int = 50) -> list[dict[str, Any]]:
    """List completed runs from the results directory."""
    results_runs = storage_root / "results" / "runs"
    if not results_runs.exists():
        return []

    runs = []
    for date_dir in sorted(results_runs.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if not run_dir.is_dir():
                continue
            db_path = run_dir / "benchmark.db"
            if not db_path.exists():
                continue

            try:
                store = SQLiteStore(str(db_path))
                summary = store.get_run_summary()

                # Try to get resolved spec for name
                resolved = run_dir / "resolved_spec.yaml"
                run_name = run_dir.name
                if resolved.exists():
                    try:
                        spec = RunSpec.from_yaml(resolved)
                        run_name = spec.name
                    except Exception:
                        pass

                for s in summary:
                    runs.append({
                        "run_id": run_dir.name,
                        "results_path": str(run_dir),
                        "date_dir": date_dir.name,
                        "model_alias": s.get("model_alias", ""),
                        "tasks_run": s.get("tasks_run", 0),
                        "passed": s.get("passed", 0),
                        "failed": s.get("failed", 0),
                        "avg_ttft_ms": round(s.get("avg_ttft_ms", 0), 2) if s.get("avg_ttft_ms") else None,
                        "avg_wall_ms": round(s.get("avg_wall_ms", 0), 2) if s.get("avg_wall_ms") else None,
                        "suite_id": "",  # Would need to extract from spec
                        "started_at": date_dir.name,
                    })
            except Exception:
                pass

            if len(runs) >= limit:
                return runs

    return runs[:limit]


def get_run_results(db_path: str) -> list[dict[str, Any]]:
    """Get all run results from a database."""
    try:
        store = SQLiteStore(db_path)
        return store.get_runs()
    except Exception:
        return []


def get_run_summary(db_path: str) -> list[dict[str, Any]]:
    """Get summary for a run's database."""
    try:
        store = SQLiteStore(db_path)
        return store.get_run_summary()
    except Exception:
        return []


def get_timing_summary(db_path: str, model_alias: Optional[str] = None) -> list[dict[str, Any]]:
    """Get timing summary for a run's database."""
    try:
        store = SQLiteStore(db_path)
        return store.get_timing_summary(model_alias=model_alias)
    except Exception:
        return []
