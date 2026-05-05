"""Prompt template renderer using Jinja2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, BaseLoader

logger = logging.getLogger(__name__)

# Built-in template names and their inline content
INLINE_TEMPLATES: dict[str, str] = {
    "plain": "{{ user_message }}",
    "repl": "Follow REPL mode: hypothesize a single test, run it, then interpret the result. {{ user_message }}",
    "terse": "Answer briefly and directly. {{ user_message }}",
    "patch_only": "Output only a unified diff (no explanation). {{ user_message }}",
    "architect": "Think architecturally first: analyze the problem, outline the approach, then implement. {{ user_message }}",
    "json_schema": "Output valid JSON matching this schema: {{ schema }}. {{ user_message }}",
    "step_by_step": "Plan step by step, then execute. {{ user_message }}",
}

# Default template directory (relative to project root configs/)
_TEMPLATE_DIR_NAME = "prompt_templates"


def _find_template_dir() -> Path:
    """Find the configs/prompt_templates/ directory."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    dir_path = root / "configs" / _TEMPLATE_DIR_NAME
    if dir_path.exists():
        return dir_path
    cwd_dir = Path.cwd() / "configs" / _TEMPLATE_DIR_NAME
    if cwd_dir.exists():
        return cwd_dir
    return root / "configs" / _TEMPLATE_DIR_NAME  # return even if missing


def render_template(
    template_str: str,
    context: dict[str, Any],
    file_context: dict[str, str] | None = None,
) -> str:
    """Render a Jinja2 template string with the given context.

    Args:
        template_str: The Jinja2 template string.
        context: Dict of template variables.
        file_context: Optional mapping of filename -> file content for injection.

    Returns:
        Rendered template string with all variables substituted.
    """
    env = Environment(
        loader=BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(template_str)

    # Inject file contents as code blocks if file_context is provided
    if file_context:
        files_str_parts = []
        for filepath, content in file_context.items():
            files_str_parts.append(f"## File: {filepath}\n```\n{content}\n```")
        context["files"] = "\n\n".join(files_str_parts)

    return template.render(**context)


def load_prompt_template(name: str, template_dir: Path | None = None) -> str:
    """Load a named prompt template from the prompt_templates/ directory.

    Supports both inline built-in templates and file-based templates.

    Args:
        name: Template name (e.g. "plain", "repl", or a file name).
        template_dir: Optional override for the template directory path.

    Returns:
        Template string content.

    Raises:
        FileNotFoundError: If the template file doesn't exist and isn't built-in.
    """
    # Check inline templates first
    if name in INLINE_TEMPLATES:
        return INLINE_TEMPLATES[name]

    # Fall back to file-based templates
    if template_dir is None:
        template_dir = _find_template_dir()

    # Support both .md and plain extensions
    for ext in (".md", ""):
        template_path = template_dir / f"{name}{ext}"
        if template_path.exists():
            content = template_path.read_text()
            return content.strip()

    raise FileNotFoundError(
        f"Template '{name}' not found in {template_dir} "
        f"(not a built-in template and no file exists)"
    )


def build_prompt(
    task: dict[str, Any],
    prompt_style: str = "plain",
    file_context: dict[str, str] | None = None,
    base_dir: str | None = None,
) -> tuple[str, list[str]]:
    """Build a prompt for a task by applying a prompt style wrapper.

    Args:
        task: Task dict with 'prompt', 'expected', 'input', etc.
        prompt_style: Style name (plain, repl, terse, etc.).
        file_context: Optional file contents for injection.
        base_dir: Base directory for resolving file paths.

    Returns:
        Tuple of (prompt_text, list_of_file_paths_referenced).
    """
    task_id = task.get("id", "unknown")

    # Get the base prompt
    prompt_template = task.get("prompt_template")
    prompt_text = task.get("prompt") or ""

    # If prompt_template is specified, load it
    if prompt_template:
        try:
            template_dir = None
            if base_dir:
                template_dir = Path(base_dir) / "configs" / _TEMPLATE_DIR_NAME
            prompt_text = load_prompt_template(prompt_template, template_dir)
        except FileNotFoundError as e:
            logger.warning("Could not load template for %s: %s", task_id, e)
            prompt_text = task.get("prompt", "")

    # Get template variables from task input
    context: dict[str, Any] = {}
    task_input = task.get("input")
    if task_input:
        context["user_message"] = task_input.get("user_message", prompt_text)
        context["system_message"] = task_input.get("system_message")
        context["schema"] = task.get("expected", {}).get("schema", {})
    else:
        context["user_message"] = prompt_text
        context["system_message"] = None
        context["schema"] = task.get("expected", {}).get("schema", {})

    # Load the prompt style wrapper
    try:
        if base_dir:
            style_template_path = Path(base_dir) / "configs" / _TEMPLATE_DIR_NAME / f"{prompt_style}.md"
            if style_template_path.exists():
                style_template = style_template_path.read_text().strip()
            else:
                style_template = INLINE_TEMPLATES.get(prompt_style, "{{ user_message }}")
        else:
            style_template = load_prompt_template(prompt_style)
    except FileNotFoundError:
        style_template = "{{ user_message }}"

    # Inject file contents if file_context is provided
    referenced_files: list[str] = []
    if file_context:
        files_str_parts = []
        for filepath in sorted(file_context.keys()):
            referenced_files.append(filepath)
            content = file_context[filepath]
            files_str_parts.append(f"## File: {filepath}\n```\n{content}\n```")
        context["files"] = "\n\n".join(files_str_parts)
    elif task.get("input", {}).get("files"):
        # Try to read files from disk
        for fpath in task["input"]["files"]:
            full_path = Path(base_dir) / fpath if base_dir else Path(fpath)
            if full_path.exists():
                try:
                    file_context = file_context or {}
                    file_context[fpath] = full_path.read_text()
                    referenced_files.append(fpath)
                except OSError as e:
                    logger.warning("Could not read file %s: %s", fpath, e)

    # Render with the style wrapper
    rendered = render_template(style_template, context, file_context)

    return rendered, referenced_files
