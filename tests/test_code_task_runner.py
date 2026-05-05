"""Tests for CodeTaskRunner — function completion and patch generation."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from bench_harness.runners.code_task_runner import CodeTaskRunner


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    return CodeTaskRunner(preserve_on_success=False)


@pytest.fixture
def function_completion_task():
    """Return a basic task dict for function completion."""
    return {
        "id": "test.plus_simple",
        "code_type": "function_completion",
        "expected": {
            "entry_point": "plus",
            "test_code": """\
import generated_code

def test_plus_positive():
    assert generated_code.plus(2, 3) == 5

def test_plus_negative():
    assert generated_code.plus(-1, 1) == 0

def test_plus_zeros():
    assert generated_code.plus(0, 0) == 0

def test_plus_large():
    assert generated_code.plus(100, 200) == 300
""",
            "test_framework": "pytest",
        },
    }


@pytest.fixture
def patch_generation_task():
    """Return a basic task dict for patch generation."""
    return {
        "id": "test.patch_simple",
        "code_type": "patch_generation",
        "expected": {
            "entry_point": "add",
            "test_code": """\
from src.algorithms import add

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
""",
            "test_framework": "pytest",
            "test_files": ["src/algorithms.py"],
        },
    }


# ── Function Completion — Passing Tests ───────────────────────────────


class TestFunctionCompletionPassing:
    def test_all_tests_pass(self, runner, function_completion_task):
        """Generated code passes all tests — score_primary == 1.0."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        assert result["score_primary"] == 1.0
        assert result["tests_passed"] == 4
        assert result["tests_failed"] == 0
        assert result["tests_total"] == 4
        assert result["code_status"] in ("compiled", "valid_python")
        assert result["generated_code"] == code

    def test_single_test_pass(self, runner):
        """A single test that passes yields score 1.0."""
        task = {
            "id": "test.single",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "triple",
                "test_code": """\
import generated_code

def test_triple():
    assert generated_code.triple(4) == 12
""",
                "test_framework": "pytest",
            },
        }
        code = "def triple(n):\n    return n * 3\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 1.0
        assert result["tests_passed"] == 1
        assert result["tests_total"] == 1

    def test_multiple_tests_all_pass(self, runner):
        """Multiple independent tests all pass."""
        task = {
            "id": "test.multi",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "multiply",
                "test_code": """\
import generated_code

def test_multiply_one():
    assert generated_code.multiply(2, 3) == 6

def test_multiply_two():
    assert generated_code.multiply(0, 5) == 0

def test_multiply_three():
    assert generated_code.multiply(-2, 4) == -8
""",
                "test_framework": "pytest",
            },
        }
        code = "def multiply(a, b):\n    return a * b\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 1.0
        assert result["tests_passed"] == 3
        assert result["tests_total"] == 3


# ── Function Completion — Syntax Error ────────────────────────────────


class TestFunctionCompletionSyntaxError:
    def test_syntax_error_bad_indentation(self, runner):
        """Bad indentation in generated code is detected."""
        task = {
            "id": "test.bad_indent",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "foo",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        code = "def foo():\nreturn 1\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 0.0
        assert result["code_status"] == "syntax_error"

    def test_syntax_error_mismatched_parens(self, runner):
        """Mismatched parentheses caught as syntax error."""
        task = {
            "id": "test.mismatch",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "g",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        code = "def g(x):\n    return (x + 1\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 0.0
        assert result["code_status"] == "syntax_error"

    def test_syntax_error_missing_colon(self, runner):
        """Missing colon after function definition is detected."""
        task = {
            "id": "test.no_colon",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "h",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        code = "def h(x)\n    return x\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 0.0
        assert result["code_status"] == "syntax_error"

    def test_syntax_error_f_string_unclosed(self, runner):
        """Unclosed f-string is detected as syntax error."""
        task = {
            "id": "test.unclosed",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "fmt",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        code = 'def fmt(n):\n    return f"count is {n\n"'
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 0.0
        assert result["code_status"] == "syntax_error"

    def test_syntax_error_triggers_test_output_message(self, runner):
        """Syntax error result includes a descriptive message in test_output."""
        task = {
            "id": "test.output_msg",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "bad",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        code = "def bad(x):\n    return (\n"
        result = runner.run(task, code, params={})

        assert "SYNTAX ERROR" in result["test_output"]


# ── Function Completion — All Tests Fail ──────────────────────────────


class TestFunctionCompletionAllFail:
    def test_all_tests_fail(self, runner):
        """Wrong implementation — all tests fail, score 0.0."""
        task = {
            "id": "test.wrong_impl",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "plus",
                "test_code": """\
import generated_code

def test_plus_one():
    assert generated_code.plus(2, 3) == 5

def test_plus_two():
    assert generated_code.plus(-1, 1) == 0

def test_plus_three():
    assert generated_code.plus(10, 20) == 30
""",
                "test_framework": "pytest",
            },
        }
        # Always returns 0 — only one test passes (plus(-1, 1) == 0)
        code = "def plus(a, b):\n    return 0\n"
        result = runner.run(task, code, params={})

        # 1 out of 3 passes because -1 + 1 = 0 == 0
        assert result["tests_passed"] == 1
        assert result["tests_failed"] == 2

    def test_all_tests_fail_completely(self, runner):
        """Function returns None so all assertion-based tests fail."""
        task = {
            "id": "test.none_impl",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "double",
                "test_code": """\
import generated_code

def test_double_one():
    assert generated_code.double(4) == 8

def test_double_zero():
    assert generated_code.double(0) == 0

def test_double_neg():
    assert generated_code.double(-3) == -6
""",
                "test_framework": "pytest",
            },
        }
        code = "def double(n):\n    pass\n"
        result = runner.run(task, code, params={})

        assert result["tests_passed"] == 0
        assert result["tests_failed"] == 3
        assert result["tests_total"] == 3

    def test_wrong_function_name(self, runner):
        """Generated function has different name — test cannot import it."""
        task = {
            "id": "test.wrong_name",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "add",
                "test_code": """\
import generated_code

def test_add():
    assert generated_code.add(1, 2) == 3
""",
                "test_framework": "pytest",
            },
        }
        code = "def adder(a, b):\n    return a + b\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 0.0
        assert result["tests_failed"] == 1

    def test_returns_wrong_type(self, runner):
        """Function runs but returns wrong value."""
        task = {
            "id": "test.wrong_val",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "double",
                "test_code": """\
import generated_code

def test_double():
    assert generated_code.double(5) == 10

def test_double_zero():
    assert generated_code.double(0) == 0
""",
                "test_framework": "pytest",
            },
        }
        # Returns n+1 instead of 2*n
        code = "def double(n):\n    return n + 1\n"
        result = runner.run(task, code, params={})

        assert result["tests_failed"] == 2
        assert result["tests_passed"] == 0


# ── Function Completion — Partial Pass ───────────────────────────────


class TestFunctionCompletionPartial:
    def test_partial_pass_some_fail(self, runner):
        """Some tests pass, some fail — score reflects ratio."""
        task = {
            "id": "test.partial",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "clamp",
                "test_code": """\
import generated_code

def test_clamp_low():
    assert generated_code.clamp(-3, 0, 10) == 0

def test_clamp_high():
    assert generated_code.clamp(15, 0, 10) == 10

def test_clamp_mid():
    assert generated_code.clamp(5, 0, 10) == 5
""",
                "test_framework": "pytest",
            },
        }
        # Only handles upper bound via min(), not lower bound
        code = "def clamp(val, lo, hi):\n    return min(val, hi)\n"
        result = runner.run(task, code, params={})

        # clamp(15,0,10)=10 passes, clamp(5,0,10)=5 passes, clamp(-3,0,10)=-3 fails
        assert result["score_primary"] == 2 / 3
        assert result["tests_passed"] == 2
        assert result["tests_failed"] == 1
        assert result["tests_total"] == 3

    def test_partial_pass_half(self, runner):
        """Exactly half tests pass — score 0.5."""
        task = {
            "id": "test.half",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "sign",
                "test_code": """\
import generated_code

def test_sign_positive():
    assert generated_code.sign(5) == 1

def test_sign_negative():
    assert generated_code.sign(-3) == -1

def test_sign_zero():
    assert generated_code.sign(0) == 0

def test_sign_large():
    assert generated_code.sign(1000) == 1
""",
                "test_framework": "pytest",
            },
        }
        # Handles positive and zero but not negative
        code = "def sign(n):\n    if n > 0:\n        return 1\n    return 0\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 0.75  # 3/4 pass
        assert result["tests_passed"] == 3
        assert result["tests_failed"] == 1

    def test_partial_pass_one_of_three(self, runner):
        """One out of three tests pass — score ~0.333."""
        task = {
            "id": "test.one_third",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "abs_val",
                "test_code": """\
import generated_code

def test_abs_pos():
    assert generated_code.abs_val(5) == 5

def test_abs_neg():
    assert generated_code.abs_val(-3) == 3

def test_abs_neg_two():
    assert generated_code.abs_val(-7) == 7
""",
                "test_framework": "pytest",
            },
        }
        # Just returns the input unchanged — all negative cases fail
        code = "def abs_val(n):\n    return n\n"
        result = runner.run(task, code, params={})

        assert result["score_primary"] == 1 / 3
        assert result["tests_passed"] == 1
        assert result["tests_failed"] == 2
        assert result["tests_total"] == 3


# ── Patch Generation — Successful Apply ──────────────────────────────


class TestPatchGenerationPassing:
    def test_patch_creates_file_and_tests_pass(self, runner):
        """Valid unified diff that creates a new file, then tests pass."""
        patch = """\
--- /dev/null
+++ b/src/algorithms.py
@@ -0,0 +1,2 @@
+def add(a, b):
+    return a + b
"""
        task = {
            "id": "test.patch_create",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": """\
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from algorithms import add

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
""",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})

        assert result["score_primary"] == 1.0
        assert result["code_status"] == "patch_applied"
        assert result["tests_passed"] == 2
        assert result["tests_failed"] == 0
        assert result["tests_total"] == 2

    def test_patch_creates_file_and_tests_fail(self, runner):
        """Valid patch creates a file but with wrong implementation."""
        patch = """\
--- /dev/null
+++ b/src/algorithms.py
@@ -0,0 +1,2 @@
+def add(a, b):
+    return a - b
"""
        task = {
            "id": "test.patch_wrong_impl",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": """\
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from algorithms import add

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
""",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})

        assert result["code_status"] == "patch_applied"
        assert result["score_primary"] == 0.0
        assert result["tests_failed"] == 2

    def test_patch_empty_file_creation(self, runner):
        """Patch task creates files in the temp directory."""
        patch = """\
--- /dev/null
+++ b/src/algorithms.py
@@ -0,0 +1,2 @@
+def add(a, b):
+    return a + b
"""
        task = {
            "id": "test.patch_artifacts",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})
        run_dir = result["run_dir"]
        assert run_dir is not None

        run_path = Path(run_dir)
        assert (run_path / "patch.diff").exists()
        assert (run_path / "test_output.txt").exists()
        assert (run_path / "test_generated.py").exists()


# ── Patch Generation — Apply Failure ─────────────────────────────────


class TestPatchGenerationFail:
    def test_completely_broken_patch(self, runner):
        """Completely invalid patch content."""
        broken_patch = "this is not a patch at all\n@@@ @@\nrandom text\n"

        task = {
            "id": "test.broken_patch",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, broken_patch, params={})

        assert result["code_status"] == "patch_failed"
        assert result["score_primary"] == 0.0
        assert result["tests_passed"] == 0
        assert result["tests_total"] == 0

    def test_patch_with_no_context_lines(self, runner):
        """Patch that references files that don't exist in the empty repo."""
        # This patch tries to modify a pre-existing file that doesn't exist
        bad_patch = """\
--- a/src/algorithms.py
+++ b/src/algorithms.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a - b
"""
        task = {
            "id": "test.no_context_file",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, bad_patch, params={})

        assert result["code_status"] == "patch_failed"
        assert result["score_primary"] == 0.0

    def test_patch_malformed_header(self, runner):
        """Patch with malformed --- / +++ lines."""
        malformed = """\
@@ src/algorithms.py
+def add(a, b):
+    return a + b
"""
        task = {
            "id": "test.malformed",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, malformed, params={})

        assert result["code_status"] == "patch_failed"
        assert result["score_primary"] == 0.0


# ── Patch Generation — Unrelated File Changes ────────────────────────


class TestPatchGenerationUnrelated:
    def test_patch_modifies_unrelated_file(self, runner):
        """Patch modifies a file not in test_files — runner logs warning."""
        patch = """\
--- /dev/null
+++ b/src/extra.py
@@ -0,0 +1,2 @@
+# extra module
+def multiply(a, b): return a * b
"""
        task = {
            "id": "test.unrelated",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": """\
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from algorithms import add

def test_add():
    assert add(1, 2) == 3
""",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})

        # The patch applied successfully (it's valid git diff)
        assert result["code_status"] == "patch_applied"
        # But the test imports from algorithms which doesn't exist after this patch
        assert result["tests_failed"] == 1

    def test_patch_modifies_multiple_files_one_unrelated(self, runner):
        """Patch modifies expected file plus an unrelated extra file."""
        patch = """\
--- /dev/null
+++ b/src/algorithms.py
@@ -0,0 +1,2 @@
+def add(a, b):
+    return a + b
--- /dev/null
+++ b/src/logger.py
@@ -0,0 +1 @@
+import logging
"""
        task = {
            "id": "test.multi_unrelated",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": """\
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from algorithms import add

def test_add():
    assert add(1, 2) == 3
""",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})

        # Patch applies but modifies files outside test_files
        assert result["code_status"] == "patch_applied"
        # Tests still run against the patched algorithms.py which is correct
        assert result["tests_passed"] == 1


# ── Artifact File Creation ───────────────────────────────────────────


class TestArtifactFiles:
    def test_function_task_creates_generated_code(self, runner, function_completion_task):
        """Function completion creates generated_code.py in temp dir."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        run_dir = result["run_dir"]
        assert run_dir is not None
        assert Path(run_dir, "generated_code.py").exists()
        content = Path(run_dir, "generated_code.py").read_text()
        assert "def plus" in content

    def test_function_task_creates_test_output(self, runner, function_completion_task):
        """Function completion creates test_output.txt with test results."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        run_dir = result["run_dir"]
        output_path = Path(run_dir, "test_output.txt")
        assert output_path.exists()
        output_content = output_path.read_text()
        assert "test_plus" in output_content

    def test_function_task_creates_test_file(self, runner, function_completion_task):
        """Function completion creates test_generated.py."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        run_dir = result["run_dir"]
        assert Path(run_dir, "test_generated.py").exists()

    def test_patch_task_creates_patch_file(self, runner):
        """Patch generation creates patch.diff in temp dir."""
        patch = """\
--- /dev/null
+++ b/src/algorithms.py
@@ -0,0 +1,2 @@
+def add(a, b):
+    return a + b
"""
        task = {
            "id": "test.artifacts",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})

        run_dir = result["run_dir"]
        assert Path(run_dir, "patch.diff").exists()
        content = Path(run_dir, "patch.diff").read_text()
        assert content == patch

    def test_patch_task_creates_test_output(self, runner):
        """Patch generation creates test_output.txt with results."""
        patch = """\
--- /dev/null
+++ b/src/algorithms.py
@@ -0,0 +1,2 @@
+def add(a, b):
+    return a + b
"""
        task = {
            "id": "test.patch_artifacts",
            "code_type": "patch_generation",
            "expected": {
                "entry_point": "add",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
                "test_files": ["src/algorithms.py"],
            },
        }

        result = runner.run(task, patch, params={})

        output_path = Path(result["run_dir"], "test_output.txt")
        assert output_path.exists()


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_generated_code(self, runner):
        """Empty string is valid Python — runs without error."""
        task = {
            "id": "test.empty",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "foo",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        result = runner.run(task, "", params={})

        # Empty code is valid Python (no-op module), so it compiles
        assert result["code_status"] in ("compiled", "valid_python")

    def test_no_test_code_provided(self, runner):
        """Task with no test_code uses default test that imports module."""
        task = {
            "id": "test.no_test_code",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "greet",
                "test_framework": "pytest",
            },
        }
        code = "def greet(name):\n    return f'Hello, {name}'\n"
        result = runner.run(task, code, params={})

        # At least the import test should pass
        assert result["tests_total"] >= 1

    def test_result_has_expected_keys(self, runner, function_completion_task):
        """Run result contains all expected keys."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        expected_keys = {
            "score_primary",
            "score_secondary",
            "score_explanation",
            "scorer_version",
            "tests_passed",
            "tests_failed",
            "tests_total",
            "test_output",
            "exit_code",
            "generated_code",
            "code_status",
            "run_dir",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_valid_python_code_status(self, runner, function_completion_task):
        """Simple pure-function module gets valid_python status (not importable)."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        assert result["code_status"] == "valid_python"

    def test_whitespace_only_code(self, runner):
        """Whitespace-only code is valid Python."""
        task = {
            "id": "test.whitespace",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "x",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        result = runner.run(task, "   \n\n   ", params={})

        # Whitespace-only is valid Python
        assert result["code_status"] in ("compiled", "valid_python")

    def test_result_exit_code_zero_on_success(self, runner, function_completion_task):
        """Successful run has exit_code 0."""
        code = "def plus(a, b):\n    return a + b\n"
        result = runner.run(function_completion_task, code, params={})

        assert result["exit_code"] == 0

    def test_syntax_error_exit_code_one(self, runner):
        """Syntax error run has exit_code 1."""
        task = {
            "id": "test.exc",
            "code_type": "function_completion",
            "expected": {
                "entry_point": "f",
                "test_code": "def test_one(): pass",
                "test_framework": "pytest",
            },
        }
        result = runner.run(task, "def f(x)\n    return x\n", params={})

        assert result["exit_code"] == 1
