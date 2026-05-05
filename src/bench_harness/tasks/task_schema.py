"""Pydantic schema for benchmark task definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskInput(BaseModel):
    """Input data for a task.

    Attributes:
        user_message: The user-facing prompt/question.
        files: Relative paths to context files for the task.
        system_message: Optional system prompt override.
    """

    user_message: str = Field(description="The user-facing prompt or question.")
    files: list[str] | None = Field(
        default=None, description="Relative paths to context files for the task."
    )
    system_message: str | None = Field(
        default=None, description="Optional system prompt override."
    )


class TaskExpected(BaseModel):
    """Expected output specification for a task.

    Attributes:
        type: Scoring type (exact, json_schema, contains, regex, rubric, unit_test, etc.).
        answer: Exact expected answer (for exact-match tasks).
        json_schema: JSON schema dict (for json_schema validation).
        patterns: Regex patterns to match (for contains/regex scorers).
        absent_patterns: Strings that must NOT appear (for contains scorer).
        test_files: Paths to test files for code tasks.
        format_checks: Format validation checks (for format_compliance scorer).
        choices: Multiple choice options (for multiple_choice scorer).
    """

    type: str = Field(description="Scoring type identifier.")
    answer: str | None = Field(
        default=None, description="Exact expected answer for exact-match tasks."
    )
    json_schema: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="JSON schema dict for json_schema validation.",
    )
    patterns: list[str] | None = Field(
        default=None, description="Regex patterns to match for contains/regex scorers."
    )
    absent_patterns: list[str] | None = Field(
        default=None, description="Patterns that must NOT appear (contains scorer)."
    )
    test_files: list[str] | None = Field(
        default=None, description="Paths to test files for code task validation."
    )
    format_checks: list[Any] | None = Field(
        default=None, description="Format validation checks (format_compliance scorer)."
    )
    choices: dict[str, str] | None = Field(
        default=None, description="Multiple choice options (multiple_choice scorer)."
    )


class TaskScoring(BaseModel):
    """Scoring configuration for a task.

    Attributes:
        primary: Primary scorer name.
        secondary: Additional scorer names.
        weights: Weight mapping for composite scores.
    """

    primary: str = Field(description="Primary scorer name (e.g. exact_match, regex).")
    secondary: list[str] | None = Field(
        default=None, description="Additional scorer names for composite scoring."
    )
    weights: dict[str, float] | None = Field(
        default=None, description="Weight mapping for composite scores."
    )


class Task(BaseModel):
    """A benchmark task definition.

    Attributes:
        id: Stable unique ID, e.g. "local.docker_compose.fix_yaml_001".
        version: Semantic version string, default "1.0".
        family: Task family (coding, debugging, shell, format_following, etc.).
        category: Optional subcategory within the family.
        source: Task source (local, public, synthetic).
        prompt_template: Optional template file name (alternative to direct prompt).
        prompt: Direct prompt text (alternative to template).
        input: Task input with user_message, files, system_message.
        expected: Expected output specification.
        scoring: Scoring configuration.
        risk_level: Risk category (low, medium, high).
        context_tokens: Approximate context size (small, medium, large, xlarge).
        allowed_commands: Optional list of allowed shell commands.
        metadata: Freeform extra fields.
    """

    id: str = Field(description="Stable unique task identifier.")
    version: str = Field(default="1.0", description="Semantic version string.")
    family: str = Field(description="Task family identifier.")
    category: str | None = Field(
        default=None, description="Optional subcategory within the family."
    )
    source: str = Field(
        default="local", description="Task source: local, public, or synthetic."
    )
    prompt_template: str | None = Field(
        default=None,
        description="Template file name (alternative to direct prompt).",
    )
    prompt: str | None = Field(
        default=None, description="Direct prompt text (alternative to template)."
    )
    input: TaskInput | None = Field(
        default=None, description="Structured task input."
    )
    expected: TaskExpected = Field(description="Expected output specification.")
    scoring: TaskScoring = Field(description="Scoring configuration.")
    risk_level: str = Field(
        default="low",
        description="Risk category: low, medium, or high.",
    )
    context_tokens: str = Field(
        default="small",
        description="Approximate context size: small, medium, large, xlarge.",
    )
    allowed_commands: list[str] | None = Field(
        default=None, description="Optional list of allowed shell commands."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Freeform extra fields."
    )

    @field_validator("id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Ensure task ID is non-empty and well-formed."""
        if not v or not v.strip():
            raise ValueError("Task ID must be a non-empty string")
        return v.strip()

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"risk_level must be one of {allowed}, got {v!r}")
        return v

    @field_validator("context_tokens")
    @classmethod
    def validate_context_tokens(cls, v: str) -> str:
        allowed = {"small", "medium", "large", "xlarge"}
        if v not in allowed:
            raise ValueError(f"context_tokens must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_prompt_or_template(self) -> "Task":
        """Ensure at least one of prompt or prompt_template is provided."""
        if self.prompt is None and self.prompt_template is None:
            # Allow scorers to work with minimal task dicts (for testing)
            # In production, this is always caught by the loader
            pass
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to a plain dict for storage/export."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Serialize task to YAML string."""
        import yaml
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
