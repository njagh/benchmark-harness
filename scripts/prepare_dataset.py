#!/usr/bin/env python3
"""Download and prepare HumanEval / MBPP datasets for benchmarking.

If the JSONL files already exist at the configured paths, they are used as-is.
If not, the script downloads them from HuggingFace and writes the local copies.

After download, this script can also regenerate tasks/coding_benchmark/*.yaml
from the JSONL source data so task files stay in sync with the canonical datasets.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    logger.error("pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)


def load_datasets_config(config_path: Path = Path("configs/datasets.yaml")) -> dict:
    """Load dataset registry from YAML config."""
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f).get("datasets", {})


def download_huggingface(name: str, split: str, output_path: Path, hf_filter=None) -> int:
    """Download a HuggingFace dataset and write as JSONL."""
    import jsonlines

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "datasets (huggingface_hub) is required. "
            "Install with: pip install datasets"
        )
        sys.exit(1)

    logger.info("Downloading %s (split=%s) ...", name, split)
    ds = load_dataset(name, split=split, streaming=True)
    count = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for example in ds:
            if hf_filter is not None and not hf_filter(example):
                continue
            writer.write(example)
            count += 1
    logger.info("Wrote %d examples to %s", count, output_path)
    return count


def prepare_human_eval(dataset_cfg: dict, tasks_dir: Path) -> dict:
    """Download or verify HumanEval JSONL and return the parsed data."""
    jsonl_path = Path(dataset_cfg["output_dir"]) / "tasks.jsonl"

    if jsonl_path.exists():
        count = sum(1 for _ in jsonl_path.open())
        logger.info("HumanEval JSONL already exists: %d examples", count)
    else:
        count = download_huggingface(
            dataset_cfg["hf_name"],
            dataset_cfg["hf_split"],
            jsonl_path,
            hf_filter=lambda x: x.get("canonical_solution"),
        )

    # Load the JSONL data
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_mbpp(dataset_cfg: dict, tasks_dir: Path) -> dict:
    """Download or verify MBPP JSONL and return the parsed data."""
    jsonl_path = Path(dataset_cfg["output_dir"]) / "tasks.jsonl"

    if jsonl_path.exists():
        count = sum(1 for _ in jsonl_path.open())
        logger.info("MBPP JSONL already exists: %d examples", count)
    else:
        count = download_huggingface(
            dataset_cfg["hf_name"],
            dataset_cfg["hf_split"],
            jsonl_path,
            hf_filter=lambda x: "test_example" in x,
        )

    # MBPP JSONL format: each line has text, test_setup, test_list, etc.
    # Some versions store test_list as a string of Python code
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_test_code(entry_point: str, examples: list) -> str:
    """Build a test_code string from HumanEval test examples."""
    lines = []
    for ex in examples:
        lines.append(ex)
    return "\n".join(lines)


def yaml_str(s: str) -> str:
    """Return a YAML block scalar for multi-line strings."""
    return "|-\n" + "\n".join("    " + line for line in s.split("\n"))


def generate_task_yaml(
    source: str,
    idx: int,
    problem: dict,
    test_code: str,
    entry_point: str,
    patterns: list[str],
) -> str:
    """Generate a YAML task file from HumanEval data."""
    canonical = problem.get("canonical_solution", "")
    prompt_text = problem.get("prompt", "")

    # Build the prompt with the canonical solution
    prompt = f"Complete the following Python function:\n\n```python\n{prompt_text}\n```\n\nOutput ONLY the completed function. No explanations."

    yaml_content = f"""id: benchmark.{source}_{idx:03d}
family: coding
category: function_completion
source: public
prompt: |
  {prompt.strip()}
scoring:
  primary: unit_test
  secondary: [format_compliance]
expected:
  type: unit_test
  test_code: |
    import unittest
    from {source}_{idx:03d} import {entry_point}
    class Test{entry_point.capitalize()}(unittest.TestCase):
{build_test_code(entry_point, [f"        def test_example(self):\n            {ex}" for ex in test_code.splitlines() if ex.strip()])}
  entry_point: {entry_point}
  test_framework: pytest
  patterns:
{chr(10).join(f"    - \"{p}\"" for p in patterns)}
  format_checks:
    - type: code_block
risk_level: low
code_type: function_completion
"""
    return yaml_content


def main():
    root = Path(__file__).resolve().parent.parent
    config = load_datasets_config(root / "configs" / "datasets.yaml")
    tasks_dir = root / "tasks" / "coding_benchmark"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    datasets = config.get("datasets", config)

    # --- HumanEval ---
    if "human_eval_v1" in datasets:
        logger.info("=== Preparing HumanEval ===")
        he_data = prepare_human_eval(datasets["human_eval_v1"], tasks_dir)

        for i, problem in enumerate(he_data[:16]):
            entry_point = problem.get("entry_point", "solution")
            test_code = problem.get("test", "")
            canonical = problem.get("canonical_solution", "")

            patterns = [f"def {entry_point}"]
            if canonical:
                patterns.append(canonical.split("\n")[1].strip() if len(canonical.split("\n")) > 1 else "")

            task_file = tasks_dir / f"human_eval_{i:03d}.yaml"
            prompt = f"Complete the following Python function:\n\n```python\n{problem.get('prompt', '')}\n```"

            # Build test assertions from the problem's test code
            test_assertions = [l.strip() for l in test_code.splitlines() if l.strip() and l.strip().startswith("assert")]

            yaml_lines = [
                f"id: benchmark.human_eval_{i:03d}",
                "family: coding",
                "category: function_completion",
                "source: public",
                "prompt: |",
                f"  {prompt.strip()}",
                "scoring:",
                "  primary: unit_test",
                "  secondary: [format_compliance]",
                "expected:",
                "  type: unit_test",
                "  test_code: |",
                f"    import unittest",
                f"    from human_eval_{i:03d} import {entry_point}",
                f"    class TestHumanEval{entry_point.capitalize()}(unittest.TestCase):",
            ]
            for assertion in test_assertions:
                yaml_lines.append(f"        {assertion}")
            yaml_lines.append(f"  entry_point: {entry_point}")
            yaml_lines.append("  test_framework: pytest")
            yaml_lines.append("  patterns:")
            yaml_lines.append(f"    - \"def {entry_point}\"")
            yaml_lines.append("  format_checks:")
            yaml_lines.append("    - type: code_block")
            yaml_lines.append("risk_level: low")
            yaml_lines.append("code_type: function_completion")

            task_file.write_text("\n".join(yaml_lines) + "\n")
            logger.info("  Created: %s (entry_point=%s, %d assertions)", task_file.name, entry_point, len(test_assertions))

        logger.info("Generated %d HumanEval tasks", len(he_data[:16]))

    # --- MBPP ---
    if "mbpp_v1" in datasets:
        logger.info("=== Preparing MBPP ===")
        mbpp_data = prepare_mbpp(datasets["mbpp_v1"], tasks_dir)

        for i in range(min(4, len(mbpp_data))):
            example = mbpp_data[i]
            text = example.get("text", "")
            # MBPP test examples are stored in different formats
            # Usually test_list is a string or list of test assertions
            test_examples = example.get("test_list", "")
            if isinstance(test_examples, str):
                test_assertions = [l.strip() for l in test_examples.split("\n") if l.strip().startswith("assert")]
            else:
                test_assertions = [l.strip() for l in str(test_examples).split("\n") if l.strip().startswith("assert")]

            # Extract the function name from the task text
            import re
            match = re.search(r'def\s+(\w+)\(', text)
            func_name = match.group(1) if match else "solution"

            task_file = tasks_dir / f"mbpp_{i:03d}.yaml"
            prompt = f"Write a Python function for the following task:\n\nTask: {text}"

            yaml_lines = [
                f"id: benchmark.mbpp_{i:03d}",
                "family: coding",
                "category: function_completion",
                "source: public",
                "prompt: |",
                f"  {prompt.strip()}",
                "scoring:",
                "  primary: unit_test",
                "  secondary: [format_compliance]",
                "expected:",
                "  type: unit_test",
                "  test_code: |",
                f"    import unittest",
                f"    from mbpp_{i:03d} import {func_name}",
                f"    class TestMbpp{func_name.capitalize()}(unittest.TestCase):",
            ]
            for assertion in test_assertions[:5]:  # limit to 5 assertions
                yaml_lines.append(f"        {assertion}")
            yaml_lines.append(f"  entry_point: {func_name}")
            yaml_lines.append("  test_framework: pytest")
            yaml_lines.append("  patterns:")
            yaml_lines.append(f"    - \"def {func_name}\"")
            yaml_lines.append("  format_checks:")
            yaml_lines.append("    - type: code_block")
            yaml_lines.append("risk_level: low")
            yaml_lines.append("code_type: function_completion")

            task_file.write_text("\n".join(yaml_lines) + "\n")
            logger.info("  Created: %s (func=%s, %d assertions)", task_file.name, func_name, len(test_assertions))

        logger.info("Generated %d MBPP tasks", min(4, len(mbpp_data)))

    logger.info("Done. Tasks in: %s", tasks_dir)


if __name__ == "__main__":
    main()
