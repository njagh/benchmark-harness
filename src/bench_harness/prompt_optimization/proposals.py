"""Template proposal system — generate candidate prompt templates from benchmark data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TemplateProposal:
    """A candidate prompt template proposal for optimization.

    Attributes:
        name: Unique identifier, e.g. "debugging-v2"
        task_family: Which task family this targets (empty = all families)
        template_str: Jinja2 template string
        baseline: Baseline style to compare against (default "plain")
        instructions: Why this variant was proposed
        score: Score after evaluation (set post-run)
        score_delta: Delta vs baseline (set post-run)
        run_count: Number of runs for this variant (set post-run)
    """
    name: str
    task_family: str = ""
    template_str: str = ""
    baseline: str = "plain"
    instructions: str = ""
    score: float = 0.0
    score_delta: float = 0.0
    run_count: int = 0


# Predefined template variations that combine or modify existing styles
_PREDEFINED_VARIANTS: dict[str, dict[str, str]] = {
    "repl-terse": {
        "template": (
            "Follow REPL mode: hypothesize a single test, run it, then interpret. "
            "Be brief in explanations — focus on results.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "REPL mode with terse output to reduce tokens while keeping debug effectiveness.",
        "baseline": "repl",
    },
    "architect-step": {
        "template": (
            "Think architecturally first: analyze the problem and outline the approach. "
            "Then execute step by step.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "Combines architect-style planning with step-by-step execution.",
        "baseline": "architect",
    },
    "format-protected-patch": {
        "template": (
            "Output a unified diff for the fix. Before the patch, add a one-line "
            "explanation of what was wrong and why the fix works. "
            "No other text.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "Patch-only output with explicit format guard — explanation before patch.",
        "baseline": "patch_only",
    },
    "json-debug": {
        "template": (
            "Output valid JSON with fields: hypothesis, test_command, result, explanation. "
            "Follow REPL reasoning but constrain to JSON.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "JSON output with debug fields (hypothesis, test, result) for structured debugging.",
        "baseline": "json_schema",
    },
    "plan-then-patch": {
        "template": (
            "First, briefly describe your plan (2-3 sentences). "
            "Then output only a unified diff.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "Plan-then-patch: brief explanation before the diff, balancing clarity and conciseness.",
        "baseline": "patch_only",
    },
    "plain-minimal": {
        "template": (
            "Respond with the minimum necessary text to solve this.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "Minimal plain prompt — strips all style overhead for the simplest possible request.",
        "baseline": "plain",
    },
    "repl-patch": {
        "template": (
            "Follow REPL mode: hypothesize, test, interpret — but when outputting code, "
            "produce only a unified diff.\n\n"
            "{{ user_message }}"
        ),
        "instructions": "REPL reasoning style with patch-only output — debug reasoning with clean diffs.",
        "baseline": "repl",
    },
}


def generate_proposals(
    runs: list[dict[str, Any]],
    task_family: str = "",
) -> list[TemplateProposal]:
    """Generate candidate template proposals based on failure patterns in runs.

    Analysis logic:
    - If patch_only has low format_compliance scores: suggest format-protected-patch
    - If repl scores high but is verbose (high completion_tokens): suggest repl-terse
    - If json_schema has high format but low correctness: suggest json-debug
    - If architect has high scores but long wall time: suggest architect-step
    - If plain consistently ties or beats styled prompts: suggest plain-minimal

    Args:
        runs: Run result dicts (with prompt_style, score_primary, score_secondary, etc.)
        task_family: If provided, filter runs to this family. Empty = all families.

    Returns:
        List of TemplateProposal objects.
    """
    # Filter by family if specified
    if task_family:
        filtered = []
        for run in runs:
            tid = run.get("task_id", "")
            parts = tid.split(".")
            if len(parts) >= 3 and parts[1] == task_family:
                filtered.append(run)
            elif len(parts) == 2 and parts[1] == task_family:
                filtered.append(run)
        runs = filtered

    if not runs:
        return []

    # Gather per-style metrics
    style_metrics: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        style = run.get("prompt_style")
        if style is None:
            continue
        if style not in style_metrics:
            style_metrics[style] = {"scores": [], "comp_tokens": [], "wall_ms": []}
        score = run.get("score_primary")
        if score is not None:
            style_metrics[style]["scores"].append(score)
        comp = run.get("completion_tokens")
        if comp:
            style_metrics[style]["comp_tokens"].append(comp)
        wall = run.get("total_wall_ms")
        if wall:
            style_metrics[style]["wall_ms"].append(wall)

    def _avg(style_key: str, metric_key: str) -> float:
        values = style_metrics.get(style_key, {}).get(metric_key, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _has_style(style: str) -> bool:
        return style in style_metrics and len(style_metrics[style]["scores"]) > 0

    proposals: list[TemplateProposal] = []

    # Pattern 1: repl is best by score but verbose -> suggest repl-terse
    if _has_style("repl") and _has_style("plain"):
        repl_score = _avg("repl", "scores")
        plain_score = _avg("plain", "scores")
        repl_tokens = _avg("repl", "comp_tokens")
        plain_tokens = _avg("plain", "comp_tokens")
        if repl_score >= plain_score and repl_tokens > plain_tokens * 2:
            info = _PREDEFINED_VARIANTS["repl-terse"]
            proposals.append(TemplateProposal(
                name="repl-terse",
                task_family=task_family,
                template_str=info["template"],
                baseline="repl",
                instructions=f"REPL scores {repl_score:.2f} vs plain {plain_score:.2f} but uses {repl_tokens:.0f} vs {plain_tokens:.0f} tokens. This variant keeps REPL reasoning with less verbosity.",
                score=0.0,
                score_delta=0.0,
            ))

    # Pattern 2: patch_only has format issues -> suggest format-protected-patch
    if _has_style("patch_only"):
        patch_scores = style_metrics["patch_only"]["scores"]
        # Check for format_compliance failures in secondary scores
        format_failures = 0
        total_checked = 0
        for run in runs:
            if run.get("prompt_style") == "patch_only" and run.get("score_secondary"):
                import json
                try:
                    secondary = json.loads(run["score_secondary"]) if isinstance(run["score_secondary"], str) else run["score_secondary"]
                    fc = secondary.get("format_compliance", {})
                    if isinstance(fc, dict):
                        total_checked += 1
                        if not fc.get("passed", True):
                            format_failures += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        if total_checked > 0 and format_failures / max(total_checked, 1) > 0.3:
            info = _PREDEFINED_VARIANTS["format-protected-patch"]
            proposals.append(TemplateProposal(
                name="format-protected-patch",
                task_family=task_family,
                template_str=info["template"],
                baseline="patch_only",
                instructions=f"patch_only has {format_failures}/{total_checked} format_compliance failures. This variant adds an explanation-before-patch guard.",
                score=0.0,
                score_delta=0.0,
            ))

    # Pattern 3: json_schema has high format but low correctness -> suggest json-debug
    if _has_style("json_schema"):
        json_scores = style_metrics["json_schema"]["scores"]
        json_tokens = style_metrics["json_schema"].get("comp_tokens", [])
        if json_scores:
            avg_json_score = sum(json_scores) / len(json_scores)
            avg_json_tokens = sum(json_tokens) / len(json_tokens) if json_tokens else 0
            if avg_json_score < 0.5 and avg_json_tokens > 100:
                info = _PREDEFINED_VARIANTS["json-debug"]
                proposals.append(TemplateProposal(
                    name="json-debug",
                    task_family=task_family,
                    template_str=info["template"],
                    baseline="json_schema",
                    instructions=f"json_schema scores {avg_json_score:.2f} (low) but produces {avg_json_tokens:.0f} tokens. json-debug adds structured debug fields for better reasoning.",
                    score=0.0,
                    score_delta=0.0,
                ))

    # Pattern 4: architect is good but slow -> suggest architect-step
    if _has_style("architect") and _has_style("plain"):
        arch_score = _avg("architect", "scores")
        plain_score = _avg("plain", "scores")
        arch_wall = _avg("architect", "wall_ms")
        plain_wall = _avg("plain", "wall_ms")
        if arch_score >= plain_score and arch_wall > plain_wall * 3:
            info = _PREDEFINED_VARIANTS["architect-step"]
            proposals.append(TemplateProposal(
                name="architect-step",
                task_family=task_family,
                template_str=info["template"],
                baseline="architect",
                instructions=f"Architect scores {arch_score:.2f} vs plain {plain_score:.2f} but takes {arch_wall:.0f}ms vs {plain_wall:.0f}ms. Step variant adds explicit planning before execution.",
                score=0.0,
                score_delta=0.0,
            ))

    # Pattern 5: plain ties or beats all -> suggest plain-minimal
    if _has_style("plain"):
        plain_score = _avg("plain", "scores")
        best_other = 0
        for style, scores in style_metrics.items():
            if style != "plain" and scores["scores"]:
                avg = sum(scores["scores"]) / len(scores["scores"])
                best_other = max(best_other, avg)
        if plain_score >= best_other * 0.95:
            info = _PREDEFINED_VARIANTS["plain-minimal"]
            proposals.append(TemplateProposal(
                name="plain-minimal",
                task_family=task_family,
                template_str=info["template"],
                baseline="plain",
                instructions=f"Plain ({plain_score:.2f}) ties or beats all styled variants (best other: {best_other:.2f}). Minimal variant strips all overhead.",
                score=0.0,
                score_delta=0.0,
            ))

    return proposals


def load_proposals_from_yaml(spec_path: str | Path) -> list[TemplateProposal]:
    """Load custom template proposals from a YAML spec file.

    Spec format:
        name: my-optimization
        baselines:
          - plain
        candidates:
          - name: custom-v1
            instructions: "My custom prompt style"
            template: "Your instructions here {{ user_message }}"
            task_family: docker_compose

    Args:
        spec_path: Path to the YAML spec file.

    Returns:
        List of TemplateProposal objects.

    Raises:
        FileNotFoundError: If the spec file doesn't exist.
        ValueError: If required fields are missing.
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for loading proposals from YAML. "
            "Install it with: pip install pyyaml"
        )

    content = spec_path.read_text()
    spec = yaml.safe_load(content)

    if not isinstance(spec, dict):
        raise ValueError(f"Spec file must be a YAML mapping, got {type(spec).__name__}")

    proposals: list[TemplateProposal] = []
    candidates = spec.get("candidates", [])

    if not candidates:
        return proposals

    for candidate in candidates:
        if not isinstance(candidate, dict):
            logger.warning("Skipping non-dict candidate: %s", candidate)
            continue

        name = candidate.get("name")
        template = candidate.get("template")

        if not name:
            raise ValueError(f"Missing 'name' in candidate: {candidate}")
        if not template:
            raise ValueError(f"Missing 'template' for candidate '{name}'")

        proposals.append(TemplateProposal(
            name=name,
            task_family=candidate.get("task_family", ""),
            template_str=template,
            baseline=candidate.get("baseline", "plain"),
            instructions=candidate.get("instructions", ""),
        ))

    return proposals


class TemplateRegistry:
    """Manages a collection of baseline and candidate templates for optimization.

    Usage:
        registry = TemplateRegistry()
        registry.add_baseline("plain")
        registry.add_candidate(TemplateProposal(name="repl-terse", ...))
        baselines = registry.get_baselines()     # ["plain"]
        candidates = registry.get_candidates()   # [TemplateProposal(...)]
    """

    def __init__(self) -> None:
        self._baselines: list[str] = []
        self._candidates: list[TemplateProposal] = []

    def add_baseline(self, name: str) -> None:
        """Add a baseline style name."""
        if name not in self._baselines:
            self._baselines.append(name)

    def add_candidate(self, proposal: TemplateProposal) -> None:
        """Add a candidate template proposal."""
        for c in self._candidates:
            if c.name == proposal.name:
                self._candidates.remove(c)
                break
        self._candidates.append(proposal)

    def add_candidates(self, proposals: list[TemplateProposal]) -> None:
        """Add multiple candidate proposals."""
        for p in proposals:
            self.add_candidate(p)

    def get_baselines(self) -> list[str]:
        """Return the list of baseline style names."""
        return list(self._baselines)

    def get_candidates(self) -> list[TemplateProposal]:
        """Return the list of candidate proposals."""
        return list(self._candidates)

    def clear(self) -> None:
        """Clear all baselines and candidates."""
        self._baselines.clear()
        self._candidates.clear()

    @property
    def has_candidates(self) -> bool:
        """Whether any candidate proposals have been added."""
        return len(self._candidates) > 0
