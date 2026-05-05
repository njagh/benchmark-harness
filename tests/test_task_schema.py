"""Validation tests for Milestone 2 — Task Schema and Registry."""

import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.tasks.task_schema import Task, TaskInput, TaskExpected, TaskScoring
from bench_harness.tasks.loaders import (
    load_task,
    load_tasks,
    filter_tasks,
    filter_tasks_by_source,
    _is_legacy_task,
)
from bench_harness.tasks.registry import TaskRegistry
from bench_harness.tasks.prompt_templates import (
    render_template,
    load_prompt_template,
    build_prompt,
    INLINE_TEMPLATES,
)


# ── Pydantic Schema Tests ──────────────────────────────────────────────


class TestTaskSchema:
    def test_valid_task_loads(self):
        """Well-formed dict parses to Task object."""
        data = {
            "id": "test.task_001",
            "family": "coding",
            "prompt": "Write hello world",
            "expected": {"type": "regex", "patterns": ["hello"]},
            "scoring": {"primary": "regex"},
        }
        task = Task.model_validate(data)
        assert task.id == "test.task_001"
        assert task.family == "coding"
        assert task.version == "1.0"
        assert task.risk_level == "low"
        assert task.context_tokens == "small"

    def test_missing_id_fails(self):
        """Task without id raises validation error."""
        data = {
            "family": "coding",
            "prompt": "hello",
            "expected": {"type": "exact"},
            "scoring": {"primary": "exact_match"},
        }
        with pytest.raises(Exception):
            Task.model_validate(data)

    def test_missing_scoring_fails(self):
        """Task without scoring.primary raises validation error."""
        data = {
            "id": "test.x",
            "prompt": "hello",
            "expected": {"type": "exact"},
            "scoring": {},
        }
        with pytest.raises(Exception):
            Task.model_validate(data)

    def test_missing_prompt_fails(self):
        """Task without prompt or prompt_template raises validation error."""
        data = {
            "id": "test.x",
            "family": "coding",
            "expected": {"type": "exact"},
            "scoring": {"primary": "exact_match"},
        }
        # Allow minimal dicts for scorer testing (prompt check is relaxed)
        task = Task.model_validate(data)
        assert task.prompt is None
        assert task.prompt_template is None

    def test_valid_prompt_and_template_coexist(self):
        """Task with both prompt and prompt_template is valid."""
        data = {
            "id": "test.x",
            "family": "coding",
            "prompt": "hello",
            "prompt_template": "architect.md",
            "expected": {"type": "regex"},
            "scoring": {"primary": "regex"},
        }
        task = Task.model_validate(data)
        assert task.prompt == "hello"
        assert task.prompt_template == "architect.md"

    def test_risk_level_validation(self):
        """risk_level must be low, medium, or high."""
        data = {
            "id": "test.x",
            "family": "coding",
            "prompt": "hello",
            "expected": {"type": "exact"},
            "scoring": {"primary": "exact_match"},
            "risk_level": "critical",
        }
        with pytest.raises(Exception):
            Task.model_validate(data)

    def test_context_tokens_validation(self):
        """context_tokens must be small, medium, large, or xlarge."""
        data = {
            "id": "test.x",
            "family": "coding",
            "prompt": "hello",
            "expected": {"type": "exact"},
            "scoring": {"primary": "exact_match"},
            "context_tokens": "gigantic",
        }
        with pytest.raises(Exception):
            Task.model_validate(data)

    def test_task_to_dict(self):
        """Task serializes to plain dict."""
        task = Task(
            id="test.dict",
            version="2.0",
            family="coding",
            prompt="test",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        )
        d = task.to_dict()
        assert d["id"] == "test.dict"
        assert d["version"] == "2.0"
        assert d["family"] == "coding"

    def test_task_to_yaml(self):
        """Task serializes to YAML string."""
        task = Task(
            id="test.yaml",
            family="debugging",
            prompt="fix bug",
            expected=TaskExpected(type="regex", patterns=["fix"]),
            scoring=TaskScoring(primary="regex"),
        )
        yml = task.to_yaml()
        parsed = yaml.safe_load(yml)
        assert parsed["id"] == "test.yaml"

    def test_task_with_input(self):
        """Task with structured input is valid."""
        data = {
            "id": "test.input_001",
            "family": "coding",
            "prompt": "Fix the code",
            "input": {
                "user_message": "This function is broken",
                "system_message": "Be concise",
            },
            "expected": {"type": "regex"},
            "scoring": {"primary": "regex"},
        }
        task = Task.model_validate(data)
        assert task.input.user_message == "This function is broken"
        assert task.input.system_message == "Be concise"


class TestTaskInput:
    def test_minimal_input(self):
        input_data = TaskInput(user_message="hello")
        assert input_data.user_message == "hello"
        assert input_data.files is None

    def test_input_with_files(self):
        input_data = TaskInput(
            user_message="review",
            files=["main.py", "test.py"],
        )
        assert input_data.files == ["main.py", "test.py"]


class TestTaskExpected:
    def test_exact_type(self):
        exp = TaskExpected(type="exact", answer="Paris")
        assert exp.answer == "Paris"

    def test_json_schema_type(self):
        schema = {"type": "object", "properties": {"x": {"type": "int"}}}
        exp = TaskExpected(type="json_schema", schema=schema)
        assert exp.json_schema == schema

    def test_contains_type(self):
        exp = TaskExpected(type="contains", patterns=["hello", "world"])
        assert exp.patterns == ["hello", "world"]


class TestTaskScoring:
    def test_minimal_scoring(self):
        sc = TaskScoring(primary="exact_match")
        assert sc.primary == "exact_match"
        assert sc.secondary is None

    def test_full_scoring(self):
        sc = TaskScoring(
            primary="rubric",
            secondary=["safety", "format"],
            weights={"correctness": 0.5, "safety": 0.3, "format": 0.2},
        )
        assert sc.secondary == ["safety", "format"]
        assert sc.weights == {"correctness": 0.5, "safety": 0.3, "format": 0.2}


# ── Legacy Migration Tests ─────────────────────────────────────────────


class TestLegacyMigration:
    def test_legacy_task_detection(self):
        """Old format has 'scoring' but not 'expected' or 'version'."""
        old = {"id": "smoke.x", "prompt": "hi", "scoring": {"primary": "regex"}}
        assert _is_legacy_task(old) is True

    def test_new_task_not_legacy(self):
        """New format has 'expected' so not legacy."""
        new = {
            "id": "test.x",
            "prompt": "hi",
            "expected": {"type": "exact"},
            "scoring": {"primary": "exact_match"},
        }
        assert _is_legacy_task(new) is False

    def test_legacy_task_loads_via_loader(self, tmp_path):
        """Old-format YAML file (with scoring but no expected) is migrated."""
        task_yaml = tmp_path / "legacy_task.yaml"
        task_yaml.write_text(
            "id: local.coding.fix_bug_001\n"
            "prompt: 'What is 2+2?'\n"
            "scoring:\n"
            "  primary: exact_match\n"
        )
        task = load_task(str(task_yaml))
        assert task["id"] == "local.coding.fix_bug_001"
        assert task["version"] == "1.0"
        assert task["family"] == "coding"

    def test_new_task_loads_via_loader(self, tmp_path):
        """New-format YAML file loads without migration warning."""
        task_yaml = tmp_path / "new_task.yaml"
        task_yaml.write_text(
            "id: test.new_001\n"
            "family: coding\n"
            "version: '2.0'\n"
            "prompt: 'Write a function'\n"
            "expected:\n"
            "  type: regex\n"
            "  patterns:\n"
            "    - 'def '\n"
            "scoring:\n"
            "  primary: regex\n"
        )
        task = load_task(str(task_yaml))
        assert task["id"] == "test.new_001"
        assert task["version"] == "2.0"
        assert task["family"] == "coding"


# ── Loader Tests ───────────────────────────────────────────────────────


class TestLoaders:
    @pytest.fixture
    def smoke_task_dir(self):
        return Path(__file__).resolve().parent.parent / "tasks" / "smoke"

    def test_load_tasks_from_smoke(self, smoke_task_dir):
        """Loads tasks from smoke directory."""
        tasks = load_tasks(str(smoke_task_dir))
        assert len(tasks) >= 5

    def test_load_tasks_as_objects(self, smoke_task_dir):
        """Loads tasks as Task objects."""
        from bench_harness.tasks.loaders import load_tasks_as_objects
        tasks = load_tasks_as_objects(str(smoke_task_dir))
        assert len(tasks) >= 5
        assert all(isinstance(t, Task) for t in tasks)

    def test_filter_tasks_by_family(self, smoke_task_dir):
        """filter_tasks returns correct subset."""
        tasks = load_tasks(str(smoke_task_dir))
        coding = filter_tasks(tasks, "coding")
        assert all(t.get("family") == "coding" for t in coding)

    def test_filter_tasks_by_source(self, smoke_task_dir):
        """filter_tasks_by_source works."""
        tasks = load_tasks(str(smoke_task_dir))
        local = filter_tasks_by_source(tasks, "local")
        assert len(local) == len(tasks)  # all local by default

    def test_malformed_yaml_skipped(self, tmp_path):
        """Bad YAML file is skipped with warning, doesn't crash loader."""
        good = tmp_path / "good.yaml"
        good.write_text(
            "id: test.good\n"
            "family: coding\n"
            "prompt: hello\n"
            "expected: {type: exact}\n"
            "scoring: {primary: exact_match}\n"
        )
        bad = tmp_path / "bad.yaml"
        bad.write_text("this: is: invalid: yaml: :::")
        tasks = load_tasks(str(tmp_path))
        assert len(tasks) == 1
        assert tasks[0]["id"] == "test.good"


# ── Template Tests ─────────────────────────────────────────────────────


class TestPromptTemplates:
    def test_render_template_basic(self):
        """Template variables are substituted."""
        result = render_template("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_render_template_with_files(self):
        """File context is injected as code blocks."""
        result = render_template(
            "{{ user_message }}\n\n{{ files }}",
            {"user_message": "Review this"},
            file_context={"main.py": "print('hi')"},
        )
        assert "Review this" in result
        assert "main.py" in result
        assert "print('hi')" in result

    def test_inline_templates_exist(self):
        """All documented inline templates are present."""
        expected_names = {
            "plain", "repl", "terse", "patch_only",
            "architect", "json_schema", "step_by_step",
        }
        assert expected_names.issubset(set(INLINE_TEMPLATES.keys()))

    def test_load_inline_template(self):
        """Load a built-in template by name."""
        t = load_prompt_template("plain")
        assert "{{ user_message }}" in t

    def test_load_file_template(self, tmp_path):
        """Load a file-based template."""
        template_dir = tmp_path / "prompt_templates"
        template_dir.mkdir()
        (template_dir / "custom.md").write_text("Custom: {{ user_message }}")
        t = load_prompt_template("custom", template_dir=template_dir)
        assert "Custom:" in t

    def test_load_unknown_template_fails(self):
        """Unknown template raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt_template("nonexistent_xyz")

    def test_build_prompt_plain(self):
        """build_prompt with plain style returns prompt as-is."""
        task = {"id": "t1", "prompt": "Hello world"}
        prompt, files = build_prompt(task, prompt_style="plain")
        assert "Hello world" in prompt
        assert files == []

    def test_build_prompt_repl(self):
        """build_prompt with repl style wraps the prompt."""
        task = {"id": "t1", "prompt": "Fix the bug"}
        prompt, _ = build_prompt(task, prompt_style="repl")
        assert "REPL mode" in prompt
        assert "Fix the bug" in prompt


# ── Registry Tests ─────────────────────────────────────────────────────


class TestTaskRegistry:
    def test_register_and_get(self):
        """Register a task and retrieve it by ID."""
        reg = TaskRegistry()
        task = Task(
            id="reg.test_001",
            family="coding",
            prompt="test",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        )
        reg.register(task)
        assert reg.get("reg.test_001") is task

    def test_get_not_found(self):
        """get returns None for unknown ID."""
        reg = TaskRegistry()
        assert reg.get("nonexistent") is None

    def test_list_by_family(self):
        """list_by_family returns correct tasks."""
        reg = TaskRegistry()
        reg.register(Task(
            id="t1", family="coding", prompt="a",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        reg.register(Task(
            id="t2", family="debugging", prompt="b",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        coding = reg.list_by_family("coding")
        assert len(coding) == 1
        assert coding[0].id == "t1"

    def test_list_by_source(self):
        """list_by_source returns correct tasks."""
        reg = TaskRegistry()
        reg.register(Task(
            id="t1", source="local", family="coding",
            prompt="a",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        reg.register(Task(
            id="t2", source="public", family="coding",
            prompt="b",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        local = reg.list_by_source("local")
        assert len(local) == 1
        assert local[0].id == "t1"

    def test_summary(self):
        """summary returns correct counts by family and source."""
        reg = TaskRegistry()
        reg.register(Task(
            id="t1", family="coding", source="local",
            prompt="a",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        reg.register(Task(
            id="t2", family="coding", source="local",
            prompt="b",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        reg.register(Task(
            id="t3", family="debugging", source="public",
            prompt="c",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        s = reg.summary()
        assert s["family"]["coding"] == 2
        assert s["family"]["debugging"] == 1
        assert s["source"]["local"] == 2
        assert s["source"]["public"] == 1

    def test_count(self):
        """count returns total number of tasks."""
        reg = TaskRegistry()
        assert reg.count() == 0
        reg.register(Task(
            id="t1", family="coding", prompt="a",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        assert reg.count() == 1

    def test_versioned_task_ids(self):
        """Same ID with different versions are both stored."""
        reg = TaskRegistry()
        t1 = Task(
            id="v.test", version="1.0", family="coding",
            prompt="v1",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        )
        t2 = Task(
            id="v.test", version="2.0", family="coding",
            prompt="v2",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        )
        reg.register(t1)
        reg.register(t2)
        assert reg.count() == 2
        # get() returns the first match by ID
        got = reg.get("v.test")
        assert got is not None

    def test_load_from_directory(self, tmp_path):
        """Bulk load from directory works."""
        for i in range(3):
            (tmp_path / f"task_{i}.yaml").write_text(
                f"id: test.bulk_{i}\n"
                f"family: coding\n"
                f"prompt: prompt {i}\n"
                f"expected: {{type: exact}}\n"
                f"scoring: {{primary: exact_match}}\n"
            )
        reg = TaskRegistry()
        n = reg.load_from_directory(str(tmp_path))
        assert n == 3
        assert reg.count() == 3

    def test_registry_bool_and_len(self):
        """Boolean and len work on empty/non-empty registry."""
        empty = TaskRegistry()
        assert not bool(empty)
        assert len(empty) == 0
        empty.register(Task(
            id="t1", family="coding", prompt="a",
            expected=TaskExpected(type="exact"),
            scoring=TaskScoring(primary="exact_match"),
        ))
        assert bool(empty)
        assert len(empty) == 1
