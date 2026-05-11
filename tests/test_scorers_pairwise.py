"""Comprehensive tests for PairwiseScorer — pairwise comparison using a judge LLM."""

import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from bench_harness.scorers.pairwise import PairwiseScorer, PairwiseResult
from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.tasks.task_schema import Task


# ── Helpers ────────────────────────────────────────────────────────────────

def make_task(**overrides):
    """Build a minimal Task for testing."""
    d = {
        "id": "test.pairwise.task_001",
        "family": "general",
        "expected": {"type": "exact", "answer": "hello"},
        "scoring": {"primary": "pairwise"},
    }
    d.update(overrides)
    return Task.model_validate(d)


def make_task_with_responses(response_a="Response A text", response_b="Response B text",
                              model_a="model_a", model_b="model_b", **overrides):
    """Build a Task with input containing messages and override fields."""
    d = {
        "id": "test.pairwise.task_002",
        "family": "general",
        "expected": {"type": "exact", "answer": "hello"},
        "scoring": {"primary": "pairwise"},
        "response_a": response_a,
        "response_b": response_b,
        "model_a": model_a,
        "model_b": model_b,
        "prompt": "Test prompt",
    }
    d.update(overrides)
    return Task.model_validate(d)


VALID_PAIRWISE_JSON = json.dumps({
    "response_a": "Response A text",
    "response_b": "Response B text",
    "model_a": "model_a",
    "model_b": "model_b",
})

VALID_PAIRWISE_RESULT = {
    "winner": "a",
    "margin": "Significantly better",
    "confidence": 0.85,
    "reason": "Response A is more accurate",
    "dimension_comparison": {
        "accuracy": {"score_a": 4.0, "score_b": 2.0, "winner": "a"},
        "clarity": {"score_a": 5.0, "score_b": 3.0, "winner": "a"},
    },
}


# ── TestPairwiseScorerInit ────────────────────────────────────────────────


class TestPairwiseScorerInit:
    def test_init_with_client(self):
        """Create scorer with mock OpenAICompatClient."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        scorer = PairwiseScorer(client=mock_client)
        assert scorer.client is mock_client
        assert scorer.rubric_name == "default"

    def test_init_with_base_url(self):
        """Create scorer with base_url+api_key, verifies client is created."""
        with patch("bench_harness.scorers.pairwise.OpenAICompatClient") as MockClient:
            mock_instance = MagicMock(spec=OpenAICompatClient)
            MockClient.return_value = mock_instance
            scorer = PairwiseScorer(
                base_url="http://localhost:8000",
                api_key="test-key",
                model="test-model",
            )
            MockClient.assert_called_once_with(
                base_url="http://localhost:8000",
                api_key="test-key",
                model="test-model",
            )
            assert scorer.client is mock_instance
            assert scorer.rubric_name == "default"

    def test_init_no_args(self):
        """Create scorer with no args, verifies client is None."""
        scorer = PairwiseScorer()
        assert scorer.client is None
        assert scorer.rubric_name == "default"


# ── TestPairwiseScorerRubric ──────────────────────────────────────────────


class TestPairwiseScorerRubric:
    def test_load_rubric_fallback(self):
        """Rubric not found in YAML, returns fallback dict structure."""
        with patch("bench_harness.scorers.pairwise._find_rubrics_file") as mock_find:
            mock_find.side_effect = KeyError("rubric not found")
            scorer = PairwiseScorer()
            result = scorer._load_rubric()
            assert result["name"] == "default"
            assert "dimensions" in result

    def test_fallback_rubric_structure(self):
        """Verify fallback has correct keys and defaults."""
        scorer = PairwiseScorer()
        rubric = scorer._fallback_rubric()
        assert rubric["name"] == "default"
        assert rubric["description"] == "Default rubric for pairwise comparison"
        assert rubric["max_scale"] == 5
        dims = rubric["dimensions"]
        assert "overall_quality" in dims
        assert dims["overall_quality"]["weight"] == 1.0
        assert dims["overall_quality"]["scale"] == 5
        assert dims["overall_quality"]["description"] == "Overall quality comparison"


# ── TestPairwiseScorerDataExtraction ──────────────────────────────────────


class TestPairwiseScorerDataExtraction:
    def test_extract_from_json(self):
        """raw_response is JSON string with response_a, response_b, model_a, model_b."""
        scorer = PairwiseScorer()
        task = make_task_with_responses()
        resp_a, resp_b, mod_a, mod_b = scorer._extract_comparison_data(task, VALID_PAIRWISE_JSON)
        assert resp_a == "Response A text"
        assert resp_b == "Response B text"
        assert mod_a == "model_a"
        assert mod_b == "model_b"

    def test_extract_from_dict_fallback(self):
        """raw_response can't parse, falls back to task dict."""
        scorer = PairwiseScorer()
        # Use a plain dict (not Task model) so response_a/response_b are preserved
        task_dict = {
            "id": "test.pairwise.task_002",
            "family": "general",
            "response_a": "Task A",
            "response_b": "Task B",
            "model_a": "task_model_a",
            "model_b": "task_model_b",
        }
        raw = "not valid json at all"
        resp_a, resp_b, mod_a, mod_b = scorer._extract_comparison_data(task_dict, raw)
        assert resp_a == "Task A"
        assert resp_b == "Task B"
        assert mod_a == "task_model_a"
        assert mod_b == "task_model_b"

    def test_extract_missing_fields(self):
        """Empty/missing fields in JSON."""
        scorer = PairwiseScorer()
        empty_json = json.dumps({"other_key": "value"})
        task = make_task_with_responses()
        resp_a, resp_b, mod_a, mod_b = scorer._extract_comparison_data(task, empty_json)
        assert resp_a == ""
        assert resp_b == ""

    def test_extract_empty_strings(self):
        """Both responses empty in JSON."""
        scorer = PairwiseScorer()
        task = make_task_with_responses()
        empty_responses = json.dumps({
            "response_a": "",
            "response_b": "",
            "model_a": "",
            "model_b": "",
        })
        resp_a, resp_b, mod_a, mod_b = scorer._extract_comparison_data(task, empty_responses)
        assert resp_a == ""
        assert resp_b == ""


# ── TestPairwiseScorerParse ───────────────────────────────────────────────


class TestPairwiseScorerParse:
    def test_safe_parse_dict_valid(self):
        """Full dict with winner="a", all fields present."""
        scorer = PairwiseScorer()
        result = scorer._safe_parse_pairwise(VALID_PAIRWISE_RESULT)
        assert result["winner"] == "a"
        assert result["confidence"] == 0.85
        assert result["margin"] == "Significantly better"
        assert result["reason"] == "Response A is more accurate"
        assert "accuracy" in result["dimension_comparison"]
        assert result["dimension_comparison"]["accuracy"]["score_a"] == 4.0

    def test_safe_parse_non_dict(self):
        """Non-dict input returns tie with 0.0 confidence."""
        scorer = PairwiseScorer()
        result = scorer._safe_parse_pairwise("just a string")
        assert result["winner"] == "tie"
        assert result["confidence"] == 0.0
        assert result["reason"] == "Judge returned non-dict response"

    def test_safe_parse_bad_winner(self):
        """Winner not in ("a","b","tie") defaults to "tie"."""
        scorer = PairwiseScorer()
        result = scorer._safe_parse_pairwise({"winner": "invalid", "confidence": 0.9})
        assert result["winner"] == "tie"
        assert result["confidence"] == 0.9

    def test_safe_parse_bad_confidence(self):
        """Non-numeric confidence defaults to 0.0."""
        scorer = PairwiseScorer()
        result = scorer._safe_parse_pairwise({
            "winner": "a", "confidence": "not_a_number", "margin": "x", "reason": "y",
        })
        assert result["confidence"] == 0.0

    def test_safe_parse_missing_dims(self):
        """dimension_comparison missing defaults to empty dict."""
        scorer = PairwiseScorer()
        result = scorer._safe_parse_pairwise({"winner": "a", "confidence": 0.7, "margin": "m", "reason": "r"})
        assert result["dimension_comparison"] == {}

    def test_safe_parse_dim_non_dict(self):
        """Each dimension value that is not a dict defaults to 0.0."""
        scorer = PairwiseScorer()
        input_data = {
            "winner": "a", "confidence": 0.7, "margin": "m", "reason": "r",
            "dimension_comparison": {
                "dim1": "not a dict",
                "dim2": 42,
            },
        }
        result = scorer._safe_parse_pairwise(input_data)
        assert result["dimension_comparison"]["dim1"]["score_a"] == 0.0
        assert result["dimension_comparison"]["dim1"]["score_b"] == 0.0
        assert result["dimension_comparison"]["dim1"]["winner"] == "tie"
        assert result["dimension_comparison"]["dim2"]["score_a"] == 0.0


# ── TestPairwiseScorerBuildResult ─────────────────────────────────────────


class TestPairwiseScorerBuildResult:
    def test_build_a_wins(self):
        """winner="a" produces correct score and passed=True."""
        scorer = PairwiseScorer()
        pairwise = {"winner": "a", "confidence": 0.85, "reason": "A better", "margin": "clear", "dimension_comparison": {}}
        rubric = {"name": "test", "dimensions": {}}
        result = scorer._build_pairwise_score_result(pairwise, rubric, "tid", "ma", "mb", None, 0.0)
        assert result.score == 0.85
        assert result.passed is True
        assert result.scorer_name == "pairwise"
        assert "A wins" in result.explanation

    def test_build_b_wins(self):
        """winner="b" produces correct score and passed=True."""
        scorer = PairwiseScorer()
        pairwise = {"winner": "b", "confidence": 0.72, "reason": "B better", "margin": "close", "dimension_comparison": {}}
        rubric = {"name": "test", "dimensions": {}}
        result = scorer._build_pairwise_score_result(pairwise, rubric, "tid", "ma", "mb", None, 0.0)
        assert result.score == 0.72
        assert result.passed is True

    def test_build_tie(self):
        """winner="tie" produces score 0.5 and passed=False."""
        scorer = PairwiseScorer()
        pairwise = {"winner": "tie", "confidence": 0.5, "reason": "tied", "margin": "even", "dimension_comparison": {}}
        rubric = {"name": "test", "dimensions": {}}
        result = scorer._build_pairwise_score_result(pairwise, rubric, "tid", "ma", "mb", None, 0.0)
        assert result.score == 0.5
        assert result.passed is False

    def test_build_with_details(self):
        """Full details dict contains expected keys."""
        scorer = PairwiseScorer()
        pairwise = {
            "winner": "a", "confidence": 0.9, "reason": "A wins", "margin": "big",
            "dimension_comparison": {"quality": {"score_a": 5.0, "score_b": 3.0, "winner": "a"}},
        }
        rubric = {"name": "my_rubric", "dimensions": {"quality": {}}}
        result = scorer._build_pairwise_score_result(pairwise, rubric, "task-123", "mod_a", "mod_b", None, 0.0)
        details = result.details
        assert details["winner"] == "a"
        assert details["confidence"] == 0.9
        assert details["rubric_name"] == "my_rubric"
        assert details["task_id"] == "task-123"
        assert details["model_a"] == "mod_a"
        assert details["model_b"] == "mod_b"
        assert details["dimension_comparison"]["quality"]["score_a"] == 5.0


# ── TestPairwiseScorerScore ──────────────────────────────────────────────


class TestPairwiseScorerScore:
    def test_score_no_client(self):
        """client=None returns score=0.0, passed=False."""
        scorer = PairwiseScorer(client=None)
        task = make_task()
        result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.score == 0.0
        assert result.passed is False
        assert result.details == {}
        assert result.explanation == "No judge client configured"

    def test_score_missing_response(self):
        """Empty responses in raw_response returns score=0.0."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        scorer = PairwiseScorer(client=mock_client)
        task = make_task()
        empty_json = json.dumps({"response_a": "", "response_b": ""})
        result = scorer.score(task, empty_json)
        assert result.score == 0.0
        assert result.passed is False
        assert "Missing one or both responses" in result.explanation

    def test_score_parse_data_error(self):
        """_extract_comparison_data raises, caught gracefully."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        scorer = PairwiseScorer(client=mock_client)
        task = make_task()
        with patch.object(scorer, "_extract_comparison_data", side_effect=ValueError("parse fail")):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.score == 0.0
        assert result.passed is False
        assert "Failed to extract comparison data" in result.explanation

    def test_score_rubric_error(self):
        """_load_rubric raises, caught gracefully."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        scorer = PairwiseScorer(client=mock_client)
        task = make_task_with_responses()
        with patch.object(scorer, "_load_rubric", side_effect=KeyError("rubric missing")):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.score == 0.0
        assert result.passed is False
        assert "Failed to load rubric" in result.explanation

    def test_score_prompt_error(self):
        """_build_pairwise_prompt raises FileNotFoundError, caught gracefully."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        scorer = PairwiseScorer(client=mock_client)
        task = make_task_with_responses()
        with patch.object(scorer, "_build_pairwise_prompt", side_effect=FileNotFoundError("template missing")):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.score == 0.0
        assert result.passed is False
        assert "Failed to build pairwise prompt" in result.explanation

    def test_score_judge_error(self):
        """_score_pairwise raises, caught and returns tie."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        scorer = PairwiseScorer(client=mock_client)
        task = make_task_with_responses()
        rubric = {"name": "test", "dimensions": {}}

        with patch.object(scorer, "_load_rubric", return_value=rubric):
            with patch.object(scorer, "_build_pairwise_prompt", return_value="prompt text"):
                with patch.object(scorer, "_score_pairwise", side_effect=RuntimeError("api down")):
                    result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.passed is False
        assert result.score == 0.5
        assert result.details["winner"] == "tie"
        assert "Error during scoring" in result.details["margin"]


# ── TestPairwiseScorerAsync ──────────────────────────────────────────────


class TestPairwiseScorerAsync:
    def _make_scorer_with_mocked_client(self):
        """Create a PairwiseScorer with a fully mocked client."""
        mock_client = MagicMock(spec=OpenAICompatClient)
        mock_client.chat_complete = AsyncMock()
        return PairwiseScorer(client=mock_client), mock_client

    def test_async_valid_json(self):
        """Mocked client returns valid JSON with winner/dimensions."""
        scorer, mock_client = self._make_scorer_with_mocked_client()
        mock_client.chat_complete.return_value = {"content": json.dumps(VALID_PAIRWISE_RESULT)}

        task = make_task_with_responses()
        rubric = {"name": "test", "dimensions": {}}
        with patch.object(scorer, "_load_rubric", return_value=rubric):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.score > 0
        assert result.passed is True
        assert result.details["winner"] == "a"

    def test_async_invalid_json(self):
        """Client returns invalid JSON raises RuntimeError, caught gracefully."""
        scorer, mock_client = self._make_scorer_with_mocked_client()
        mock_client.chat_complete.return_value = {"content": "this is not json"}

        task = make_task_with_responses()
        rubric = {"name": "test", "dimensions": {}}
        with patch.object(scorer, "_load_rubric", return_value=rubric):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.passed is False
        assert result.details["winner"] == "tie"
        assert "invalid JSON" in result.details["reason"]

    def test_async_empty_response(self):
        """Client returns empty content raises RuntimeError, caught gracefully."""
        scorer, mock_client = self._make_scorer_with_mocked_client()
        mock_client.chat_complete.return_value = {"content": ""}

        task = make_task_with_responses()
        rubric = {"name": "test", "dimensions": {}}
        with patch.object(scorer, "_load_rubric", return_value=rubric):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.passed is False
        assert result.details["winner"] == "tie"
        assert "empty response" in result.details["reason"]

    def test_async_api_error(self):
        """Client returns error dict raises RuntimeError, caught gracefully."""
        scorer, mock_client = self._make_scorer_with_mocked_client()
        mock_client.chat_complete.return_value = {"error": "rate limited"}

        task = make_task_with_responses()
        rubric = {"name": "test", "dimensions": {}}
        with patch.object(scorer, "_load_rubric", return_value=rubric):
            result = scorer.score(task, VALID_PAIRWISE_JSON)
        assert result.passed is False
        assert result.details["winner"] == "tie"
        assert "rate limited" in result.details["reason"]

    def test_async_parse_safe(self):
        """_safe_parse_pairwise normalizes dimension_comparison for async results."""
        scorer, _ = self._make_scorer_with_mocked_client()
        raw = {
            "winner": "b",
            "margin": "B is better",
            "confidence": 0.6,
            "reason": "response B is clearer",
            "dimension_comparison": {
                "accuracy": {"score_a": 2.0, "score_b": 4.0, "winner": "b"},
                "format": "not_a_dict",
            },
        }
        parsed = scorer._safe_parse_pairwise(raw)
        assert parsed["winner"] == "b"
        assert parsed["confidence"] == 0.6
        assert parsed["dimension_comparison"]["accuracy"]["winner"] == "b"
        assert parsed["dimension_comparison"]["format"]["winner"] == "tie"
        assert parsed["dimension_comparison"]["format"]["score_a"] == 0.0
