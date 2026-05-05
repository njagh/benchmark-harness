"""Tests for Milestone 4 — Basic Scorers."""

import pytest
from bench_harness.scorers.base import (
    BaseScorer,
    ScoreResult,
    register_scorer,
    get_scorer,
    list_scorers,
    score_all,
)
from bench_harness.scorers import (
    exact_match,
    multiple_choice,
    regex,
    json_schema,
    contains,
    format_compliance,
)


@pytest.fixture
def sample_task():
    """A minimal task dict compatible with scorer expectations."""
    return {
        "id": "test.task_001", "family": "general",
        "scoring": {"primary": "exact_match"},
        "expected": {"type": "exact", "answer": "Paris"},
    }


# ── Base Scorer / Registry Tests ─────────────────────────────────────


class TestScorerRegistry:
    def test_list_scorers(self):
        """All 6 scorers are registered."""
        names = list_scorers()
        expected = {"exact_match", "multiple_choice", "regex", "json_schema", "contains", "format_compliance"}
        assert expected.issubset(set(names))

    def test_get_scorer(self):
        """get_scorer returns an instance of the correct type."""
        scorer = get_scorer("exact_match")
        assert isinstance(scorer, exact_match.ExactMatchScorer)

    def test_get_scorer_unknown(self):
        """get_scorer raises KeyError for unknown scorer."""
        with pytest.raises(KeyError):
            get_scorer("nonexistent")

    def test_register_decorator(self):
        """@register_scorer decorator registers the class."""
        class TestScorer(BaseScorer):
            name = "test_custom"
            version = "0.1"
            def score(self, task, raw_response):
                return ScoreResult(scorer_name=self.name, scorer_version=self.version, score=1.0, passed=True, details={})
        register_scorer(TestScorer)
        assert "test_custom" in list_scorers()
        got = get_scorer("test_custom")
        assert isinstance(got, TestScorer)

    def test_score_all(self, sample_task):
        """score_all runs multiple scorers and returns results."""
        task = {
            "id": "test.multi", "family": "general",
            "scoring": {"primary": "exact_match", "secondary": ["regex"]},
            "expected": {"type": "exact", "answer": "hello"},
        }
        results = score_all(task, "hello", ["exact_match", "regex"])
        assert "exact_match" in results
        assert "regex" in results
        assert results["exact_match"].score == 1.0

    def test_score_all_scorer_error(self, sample_task):
        """score_all handles scorer errors gracefully."""
        results = score_all(sample_task, "hello", ["exact_match", "nonexistent_scorer"])
        # exact_match ran but response "hello" != "Paris", so score is 0.0
        assert results["exact_match"].score == 0.0
        assert results["exact_match"].passed is False
        # The nonexistent scorer should return an error result
        assert "nonexistent_scorer" in results
        assert results["nonexistent_scorer"].score == -1.0

    def test_score_all_validation_skip(self, sample_task):
        """score_all skips scorers that don't validate."""
        results = score_all(sample_task, "hello", ["exact_match", "multiple_choice"])
        # multiple_choice won't validate for exact tasks
        if "multiple_choice" in results:
            assert not results["multiple_choice"].passed


# ── Exact Match Tests ────────────────────────────────────────────────


class TestExactMatchScorer:
    def test_exact_match_pass(self, sample_task):
        scorer = get_scorer("exact_match")
        result = scorer.score(sample_task, "Paris")
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_pass_stripped(self, sample_task):
        """Whitespace is stripped before comparison."""
        scorer = get_scorer("exact_match")
        result = scorer.score(sample_task, "  Paris  ")
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_fail(self, sample_task):
        """Different string fails."""
        scorer = get_scorer("exact_match")
        result = scorer.score(sample_task, "London")
        assert result.score == 0.0
        assert result.passed is False

    def test_exact_match_case_insensitive(self):
        """case_insensitive option works."""
        scorer = get_scorer("exact_match", case_insensitive=True)
        task = {
            "id": "test.task_001", "family": "general", 
            "scoring": {"primary": "exact_match"}, "expected": {"type": "exact", "answer": "Paris"}}
        result = scorer.score(task, "paris")
        assert result.passed is True

    def test_exact_match_missing_answer(self):
        """No expected answer returns score 0."""
        task = {
            "id": "test.task_001", "family": "general", 
            "scoring": {"primary": "exact_match"}, "expected": {"type": "exact"}}
        scorer = get_scorer("exact_match")
        result = scorer.score(task, "Paris")
        assert result.score == 0.0
        assert result.passed is False


# ── Multiple Choice Tests ───────────────────────────────────────────


class TestMultipleChoiceScorer:
    def test_multiple_choice_letter(self):
        """Simple letter answer is detected."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "multiple_choice"},
            "expected": {
                "type": "multiple_choice",
                "answer": "A",
                "choices": {"A": "Paris", "B": "London"},
            },
        }
        scorer = get_scorer("multiple_choice")
        result = scorer.score(task, "A")
        assert result.passed is True
        assert result.score == 1.0

    def test_multiple_choice_verbose(self):
        """Verbose answer format is detected."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "multiple_choice"},
            "expected": {
                "type": "multiple_choice",
                "answer": "B",
                "choices": {"A": "Paris", "B": "London"},
            },
        }
        scorer = get_scorer("multiple_choice")
        result = scorer.score(task, "The answer is B")
        assert result.passed is True

    def test_multiple_choice_parenthesized(self):
        """Parenthesized letter is detected."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "multiple_choice"},
            "expected": {
                "type": "multiple_choice",
                "answer": "C",
                "choices": {"A": "Paris", "B": "London", "C": "Berlin"},
            },
        }
        scorer = get_scorer("multiple_choice")
        result = scorer.score(task, "I think it's (C)")
        assert result.passed is True

    def test_multiple_choice_wrong(self):
        """Wrong choice fails."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "multiple_choice"},
            "expected": {
                "type": "multiple_choice",
                "answer": "A",
                "choices": {"A": "Paris", "B": "London"},
            },
        }
        scorer = get_scorer("multiple_choice")
        result = scorer.score(task, "B")
        assert result.passed is False
        assert result.score == 0.0

    def test_multiple_choice_no_extract(self):
        """Cannot extract letter returns 0 score."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "multiple_choice"},
            "expected": {
                "type": "multiple_choice",
                "answer": "A",
                "choices": {"A": "Paris"},
            },
        }
        scorer = get_scorer("multiple_choice")
        result = scorer.score(task, "I don't know, maybe Paris?")
        assert result.passed is False


# ── Regex Tests ─────────────────────────────────────────────────────


class TestRegexScorer:
    def test_regex_all_mode(self):
        """All patterns must match for full score."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "regex"},
            "expected": {
                "type": "all",
                "patterns": ["def hello", "return True"],
            },
        }
        scorer = get_scorer("regex")
        result = scorer.score(task, "def hello():\n    return True")
        assert result.passed is True
        assert result.score == 1.0

    def test_regex_partial_score(self):
        """Partial match gives partial score."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "regex"},
            "expected": {
                "type": "all",
                "patterns": ["def hello", "return True", "print('hi')"],
            },
        }
        scorer = get_scorer("regex")
        result = scorer.score(task, "def hello():\n    return True")
        assert result.passed is False
        assert result.score == 2.0 / 3.0

    def test_regex_any_mode(self):
        """At least one pattern matches in any mode."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "regex"},
            "expected": {
                "type": "any",
                "patterns": ["foo", "bar"],
            },
        }
        scorer = get_scorer("regex")
        result = scorer.score(task, "baz bar baz")
        assert result.passed is True
        assert result.score == 0.5

    def test_regex_none_mode(self):
        """None patterns should match in none mode."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "regex"},
            "expected": {
                "type": "none",
                "patterns": ["bad_word"],
            },
        }
        scorer = get_scorer("regex")
        result = scorer.score(task, "good text here")
        assert result.passed is True
        assert result.score == 1.0

    def test_regex_none_mode_violation(self):
        """Pattern found in none mode fails."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "regex"},
            "expected": {
                "type": "none",
                "patterns": ["bad_word"],
            },
        }
        scorer = get_scorer("regex")
        result = scorer.score(task, "this has bad_word in it")
        assert result.passed is False
        assert result.score == 0.0


# ── JSON Schema Tests ───────────────────────────────────────────────


class TestJsonSchemaScorer:
    def test_json_schema_valid(self):
        """Valid JSON passes schema validation."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "json_schema"},
            "expected": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "required": ["name", "language"],
                    "properties": {
                        "name": {"type": "string"},
                        "language": {"type": "string"},
                    },
                },
            },
        }
        scorer = get_scorer("json_schema")
        result = scorer.score(task, '{"name": "Python", "language": "programming"}')
        assert result.passed is True

    def test_json_schema_invalid(self):
        """Invalid JSON fails validation."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "json_schema"},
            "expected": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        scorer = get_scorer("json_schema")
        result = scorer.score(task, '{"name": 123}')
        assert result.passed is False

    def test_json_schema_extract_from_fences(self):
        """JSON inside markdown code fences is extracted."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "json_schema"},
            "expected": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
        scorer = get_scorer("json_schema")
        result = scorer.score(
            task,
            '```json\n{"x": 42}\n```',
        )
        assert result.passed is True

    def test_json_schema_malformed(self):
        """Completely malformed JSON returns 0."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "json_schema"},
            "expected": {
                "type": "json_schema",
                "json_schema": {"type": "object"},
            },
        }
        scorer = get_scorer("json_schema")
        result = scorer.score(task, "this is not json at all")
        assert result.passed is False


# ── Contains Scorer Tests ───────────────────────────────────────────


class TestContainsScorer:
    def test_contains_pass(self):
        """Required patterns found."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "contains"},
            "expected": {
                "type": "contains",
                "patterns": ["a + b", "def add"],
            },
        }
        scorer = get_scorer("contains")
        result = scorer.score(task, "The fix: def add(a, b): return a + b")
        assert result.passed is True
        assert result.score == 1.0

    def test_contains_absent_violation(self):
        """Forbidden string found results in penalty."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "contains"},
            "expected": {
                "type": "contains",
                "patterns": ["a + b"],
                "absent_patterns": ["a - b"],
            },
        }
        scorer = get_scorer("contains")
        result = scorer.score(task, "return a + b; also a - b here")
        assert result.passed is False

    def test_contains_partial(self):
        """Only some required patterns found."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "contains"},
            "expected": {
                "type": "contains",
                "patterns": ["def hello", "return True"],
            },
        }
        scorer = get_scorer("contains")
        result = scorer.score(task, "def hello(): pass")
        assert result.score == 0.5


# ── Format Compliance Tests ─────────────────────────────────────────


class TestFormatComplianceScorer:
    def test_format_numbered_list(self):
        """Numbered list is detected."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "format_compliance"},
            "expected": {
                "type": "format", "id": "test.task_001", "family": "format_following",
                "format_checks": [{"starts_with_numbered_list": True}],
            },
        }
        scorer = get_scorer("format_compliance")
        result = scorer.score(task, "1. First point\n2. Second point")
        assert result.passed is True

    def test_format_line_count(self):
        """Line count checks work."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "format_compliance"},
            "expected": {
                "type": "format", "id": "test.task_001", "family": "format_following",
                "format_checks": [
                    {"line_count_max": {"max": 5}},
                ],
            },
        }
        scorer = get_scorer("format_compliance")
        result = scorer.score(task, "a\nb\nc")
        assert result.passed is True

    def test_format_no_filler(self):
        """Conversational filler is detected."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "format_compliance"},
            "expected": {
                "type": "format", "id": "test.task_001", "family": "format_following",
                "format_checks": [
                    {"no_conversational_filler": True},
                ],
            },
        }
        scorer = get_scorer("format_compliance")
        result = scorer.score(task, "Sure! Here you go...")
        assert result.passed is False

    def test_format_code_block(self):
        """Code block is detected."""
        task = {
            "id": "test.task_001",
            "family": "general",
            "scoring": {"primary": "format_compliance"},
            "expected": {
                "type": "format", "id": "test.task_001", "family": "format_following",
                "format_checks": [{"is_code_block": True}],
            },
        }
        scorer = get_scorer("format_compliance")
        result = scorer.score(task, "Here's the code:\n```\nprint('hi')\n```")
        assert result.passed is True


# ── ScoreResult Tests ───────────────────────────────────────────────


class TestScoreResult:
    def test_score_result_fields(self):
        """ScoreResult has all expected fields."""
        result = ScoreResult(
            scorer_name="test",
            scorer_version="1.0",
            score=0.95,
            passed=True,
            details={"key": "value"},
            explanation="Good match",
        )
        assert result.scorer_name == "test"
        assert result.score == 0.95
        assert result.passed is True
        assert result.details == {"key": "value"}

    def test_score_result_serialization(self):
        """ScoreResult serializes to dict correctly."""
        result = ScoreResult(
            scorer_name="test",
            scorer_version="1.0",
            score=0.5,
            passed=False,
            details={"a": 1},
        )
        d = result.to_dict()
        assert d["scorer_name"] == "test"
        assert d["score"] == 0.5
        assert d["passed"] is False

    def test_score_result_deserialization(self):
        """ScoreResult deserializes from dict correctly."""
        d = {
            "scorer_name": "test",
            "scorer_version": "1.0",
            "score": 0.8,
            "passed": True,
            "details": {},
            "explanation": None,
            "duration_ms": None,
        }
        result = ScoreResult.from_dict(d)
        assert result.scorer_name == "test"
        assert result.score == 0.8
