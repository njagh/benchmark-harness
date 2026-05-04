# Milestone 2 — Task Schema and Registry

## Goal

Make benchmark tasks structured, versioned, and easy to add without Python code changes.

## Phase

Phase B — Real coding usefulness (Milestone 1 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap, task loader exists)

---

## Subtasks

### 2.1 Define task YAML schema

**File:** `src/bench_harness/tasks/task_schema.py`

**Schema (as Pydantic model):**

```python
class TaskInput(BaseModel):
    user_message: str
    files: list[str] | None = None          # relative paths to context files
    system_message: str | None = None

class TaskExpected(BaseModel):
    type: str                                # exact, json_schema, contains, regex, rubric, unit_test, etc.
    answer: str | None = None
    schema: dict | None = None               # JSON schema dict
    patterns: list[str] | None = None        # regex patterns
    test_files: list[str] | None = None      # paths to test files for code tasks

class TaskScoring(BaseModel):
    primary: str                             # scorer name
    secondary: list[str] | None = None       # additional scorer names
    weights: dict[str, float] | None = None  # for composite scores

class Task(BaseModel):
    id: str                                  # stable unique ID, e.g. "local.docker_compose.fix_yaml_001"
    version: str = "1.0"
    family: str                              # coding, debugging, shell, format_following, etc.
    category: str | None = None              # subcategory
    source: str = "local"                    # local, public, synthetic
    prompt_template: str | None = None       # template file name, if using templating
    prompt: str | None = None                # direct prompt (alternative to template)
    input: TaskInput | None = None
    expected: TaskExpected
    scoring: TaskScoring
    risk_level: str = "low"                  # low, medium, high
    context_tokens: str = "small"            # small, medium, large, xlarge
    allowed_commands: list[str] | None = None
    metadata: dict | None = None             # freeform extra fields
```

**Actions:**
- [ ] Implement Pydantic models (or dataclasses with manual validation if avoiding Pydantic dependency)
- [ ] Add `model_validate()` based loader
- [ ] Document all fields with docstrings

### 2.2 Implement task loader with schema validation

**File:** `src/bench_harness/tasks/loaders.py` (update existing)

**Functions:**
- `load_task_yaml(path: str) -> Task` — loads and validates against schema
- `load_task_directory(dir_path: str) -> list[Task]` — loads all `.yaml` files
- `load_tasks_by_family(tasks: list[Task], family: str) -> list[Task]` — filter
- `load_tasks_by_category(tasks: list[Task], category: str) -> list[Task]` — filter

**Error handling:**
- On validation failure, print task ID + field errors, skip task (don't crash)
- Collect and report all load errors at end of directory scan

**Actions:**
- [ ] Update loader to use Task schema validation
- [ ] Add warning logging for skipped tasks
- [ ] Backward-compatible: accept old flat YAML format with migration warnings

### 2.3 Implement prompt template renderer

**File:** `src/bench_harness/tasks/prompt_templates.py`

**Template format:** Jinja2 or Python `.format()` style

**Function:** `render_template(template_str: str, context: dict) -> str`

**Built-in template variables:**
- `{{ user_message }}` — from task input
- `{{ files }}` — file contents, formatted as code blocks
- `{{ system_message }}` — system prompt override
- `{{ prompt_style }}` — current prompt style (plain, repl, terse, etc.)

**Prompt style wrappers (stored as templates):**

**File:** `configs/prompt_templates/`
```
plain.md          — "{{ user_message }}"
repl.md           — "Follow REPL mode: hypothesize, test, interpret. {{ user_message }}"
terse.md          — "Answer briefly. {{ user_message }}"
patch_only.md     — "Output only a unified diff. {{ user_message }}"
architect.md      — "Think architecturally first, then implement. {{ user_message }}"
json_schema.md    — "Output valid JSON matching this schema: {{ schema }}. {{ user_message }}"
step_by_step.md   — "Plan step by step, then execute. {{ user_message }}"
```

**Actions:**
- [ ] Implement template rendering function
- [ ] Create prompt template files in `configs/prompt_templates/`
- [ ] Support file context injection (read file, inject as fenced code block)

### 2.4 Implement task registry

**File:** `src/bench_harness/tasks/registry.py`

**Class:** `TaskRegistry`

**Methods:**
- `__init__()`
- `register(task: Task)` — add a task
- `load_from_directory(dir_path: str)` — bulk load
- `get(task_id: str) -> Task | None`
- `list_by_family(family: str) -> list[Task]`
- `list_by_source(source: str) -> list[Task]`
- `list_all() -> list[Task]`
- `count() -> int`
- `summary() -> dict` — returns `{family: count, source: count}`

**CLI integration:**
- Add `bench_harness list-tasks` command
- Add `bench_harness list-tasks --family coding` filter
- Add `bench_harness show-task <task_id>` command

**Actions:**
- [ ] Implement TaskRegistry class
- [ ] Wire to CLI with new subcommands
- [ ] Registry is a singleton accessible from runner

### 2.5 Add task versioning

**Concept:** Each task has a `version` field. The registry tracks versions.

**Implementation:**
- Task ID format: `{source}.{family}.{name}_{seq}`
- Version field defaults to `"1.0"`
- When loading, if same ID with different version exists, keep both with `{id}@{version}` key
- Report can group by task ID and show version used

**Actions:**
- [ ] Update Task model to include version
- [ ] Update registry to handle versioned lookups
- [ ] Add version to run records in SQLite

### 2.6 Add suite configuration

**File:** `configs/suites.yaml` (update)

```yaml
suites:
  smoke:
    description: "Verify harness and model endpoint work"
    task_dir: "tasks/smoke"
    families: null          # load all families in dir
    max_concurrency: 4
    default_runs: 1
    default_temperature: 0

  coding_smoke:
    description: "Fast coding quality regression"
    task_dir: "tasks/coding_smoke"
    families:
      - coding
      - debugging
    max_concurrency: 2
    default_runs: 1
    default_temperature: 0
```

**Actions:**
- [ ] Update suites.yaml with coding_smoke suite
- [ ] Update config loader to parse suite config fully
- [ ] Suite config drives task loading, concurrency, and defaults

### 2.7 Add validation tests

**File:** `tests/test_task_schema.py`

**Tests:**
- `test_valid_task_loads` — well-formed YAML parses to Task object
- `test_missing_id_fails` — task without `id` raises validation error
- `test_missing_scoring_fails` — task without `scoring.primary` raises
- `test_missing_prompt_fails` — task without `prompt` or `prompt_template` raises
- `test_valid_prompt_template` — template renders with context
- `test_registry_list_by_family` — filter works correctly
- `test_registry_summary` — counts are accurate
- `test_versioned_task_ids` — same ID different versions both load
- `test_malformed_yaml_skipped` — bad YAML file is skipped with warning
- `test_file_context_injection` — file contents injected into prompt

**Actions:**
- [ ] Implement all tests
- [ ] Use pytest fixtures for sample task YAMLs
- [ ] Ensure tests run against schema, not external services

---

## Acceptance Criteria Checklist

- [ ] New task can be added by dropping a YAML file into a task directory
- [ ] Invalid task configs fail fast with field-level error messages
- [ ] `bench_harness list-tasks` lists all registered tasks
- [ ] `bench_harness list-tasks --family coding` filters correctly
- [ ] `bench_harness show-task <id>` displays full task YAML
- [ ] Prompt templates render correctly with context variables
- [ ] Task versions are tracked and stored in run records
- [ ] All validation tests pass

## Estimated Effort

1.5–2 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/tasks/task_schema.py` | To create |
| `src/bench_harness/tasks/registry.py` | To create |
| `src/bench_harness/tasks/prompt_templates.py` | To create |
| `configs/prompt_templates/plain.md` | To create |
| `configs/prompt_templates/repl.md` | To create |
| `configs/prompt_templates/terse.md` | To create |
| `configs/prompt_templates/patch_only.md` | To create |
| `configs/prompt_templates/architect.md` | To create |
| `configs/prompt_templates/json_schema.md` | To create |
| `configs/prompt_templates/step_by_step.md` | To create |
| `tests/test_task_schema.py` | To create |
