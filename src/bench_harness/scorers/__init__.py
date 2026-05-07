"""Scorer package — automatically registers all scorer implementations."""

from bench_harness.scorers.base import (
    BaseScorer,
    ScoreResult,
    register_scorer,
    get_scorer,
    list_scorers,
    score_all,
)

# Import all scorer modules to trigger @register_scorer decorators
from bench_harness.scorers import exact_match
from bench_harness.scorers import multiple_choice
from bench_harness.scorers import regex
from bench_harness.scorers import json_schema
from bench_harness.scorers import contains
from bench_harness.scorers import format_compliance
from bench_harness.scorers import unit_test
from bench_harness.scorers import config_valid
from bench_harness.scorers import llm_judge
from bench_harness.scorers import pairwise
from bench_harness.scorers import command_safety

__all__ = [
    "BaseScorer",
    "ScoreResult",
    "register_scorer",
    "get_scorer",
    "list_scorers",
    "score_all",
    "exact_match",
    "multiple_choice",
    "regex",
    "json_schema",
    "contains",
    "format_compliance",
    "unit_test",
    "config_valid",
    "llm_judge",
    "pairwise",
    "command_safety",
]
