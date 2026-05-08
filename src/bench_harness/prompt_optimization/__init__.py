"""Automatic prompt optimization — analyze styles, propose templates, run candidates."""

from __future__ import annotations

from bench_harness.prompt_optimization.analysis import (
    PromptAnalysis,
    analyze_style_data,
    best_style_family,
    detect_insufficient_data,
)
from bench_harness.prompt_optimization.proposals import (
    TemplateProposal,
    TemplateRegistry,
    generate_proposals,
    load_proposals_from_yaml,
)
from bench_harness.prompt_optimization.runner import PromptOptimizationRunner

__all__ = [
    "PromptAnalysis",
    "PromptOptimizationRunner",
    "TemplateProposal",
    "TemplateRegistry",
    "analyze_style_data",
    "best_style_family",
    "detect_insufficient_data",
    "generate_proposals",
    "load_proposals_from_yaml",
]
