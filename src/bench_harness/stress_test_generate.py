#!/usr/bin/env python3
"""Generate context stress test tasks with escalating context sizes.

Creates task YAML files in tasks/stress_test_context/ with progressively
larger blocks of distractor (filler) context embedded in the prompt.

Usage:
    python -m bench_harness.stress_test_generate [--output-dir tasks/stress_test_context]

Context sizes: 10K, 25K, 50K, 100K, 250K tokens (approximate)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Filler text generator
# ---------------------------------------------------------------------------

def _gen_paragraph() -> str:
    """Return one ~80-token paragraph of coherent English text."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
        "The five boxing wizards jump quickly. "
        "Sphinx of black quartz, judge my vow. "
        "Two driven jocks help fax my big quiz. "
        "Jackdaws love my big sphinx of quartz. "
        "We promptly judged antique ivory buckles for the next prize. "
        "Bright vixens jump; dozy fowl quack. "
        "Waltz, nymph, for quick jigs vex Bud."
    )


def _gen_block(n_paragraphs: int = 10) -> str:
    """Generate a block of n_paragraphs paragraphs (~800 tokens)."""
    return "\n\n".join(_gen_paragraph() for _ in range(n_paragraphs))


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ~ 1.3 words."""
    return len(text.split())


def generate_stress_tasks(
    output_dir: str | Path = "tasks/stress_test_context",
    block_size: int = 10,  # paragraphs per block
    num_blocks: dict[int, int] | None = None,
) -> list[Path]:
    """Generate context stress test YAML files.

    Args:
        output_dir: Directory to write task YAML files.
        block_size: Number of paragraphs per block (must be > 1).
        num_blocks: Maps context sizes (in K) to number of blocks needed.
            Defaults: 10K->2, 25K->5, 50K->10, 100K->20, 250K->50.

    Returns:
        List of generated YAML file paths.
    """
    if num_blocks is None:
        # Rough estimates: we need ~800 tokens per block
        k_to_blocks = {
            10: 13,
            25: 32,
            50: 64,
            100: 128,
            250: 320,
        }
    else:
        k_to_blocks = num_blocks

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_text = _gen_block(block_size)
    block_tokens = _estimate_tokens(block_text)

    files = []
    for k_size, n_blocks in sorted(k_to_blocks.items()):
        total_context_text = block_text * n_blocks
        context_tokens = _estimate_tokens(total_context_text)

        # Build the prompt with header, context, and instruction
        header = (
            "You are evaluating long-context comprehension of a model. "
            "Read the context below carefully. "
            "The relevant information you need is embedded in the distractor text.\n\n"
            f"## Context ({k_size}K tokens)\n---\n\n"
        )
        footer = (
            f"\n---\n\n**Question:** After reading the entire context above, "
            "count the total number of English letters (A-Z, a-z, ignoring spaces "
            "and punctuation) in the first paragraph of the context. "
            "Return only the count as an integer. "
            "First paragraph: '{_gen_paragraph()}'"
        )

        prompt = header + total_context_text + footer
        estimated_tokens = context_tokens

        task = {
            "id": f"stress.ctx_{k_size}k",
            "family": "stress_test_context",
            "prompt": prompt,
            "scoring": {
                "primary": "exact_match",
                "secondary": [],
            },
            "expected": {
                "type": "exact_match",
                "answer": str(len(_gen_paragraph().replace(" ", "").replace(".", ""))),
            },
            "context_tokens": f"{k_size}k",
            "estimated_prompt_tokens": estimated_tokens,
            "notes": (
                f"Context stress test at ~{k_size}K tokens of filler text. "
                "Tests whether the model can maintain context quality at large prefill sizes."
            ),
        }

        yaml_path = out_dir / f"stress_ctx_{k_size}k.yaml"
        yaml_path.write_text(_to_yaml(task))
        files.append(yaml_path)

        print(
            f"  {k_size:>4}K tokens: {n_blocks} blocks, "
            f"prompt ~{len(prompt)} chars ({estimated_tokens} est. tokens)"
        )

    return files


def _to_yaml(obj: dict) -> str:
    """Convert dict to a simple YAML string (no special dependencies)."""
    lines = []

    def _dump(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    _dump(v, indent + 1)
                elif isinstance(v, list):
                    if not v:
                        lines.append(f"{prefix}{k}: []")
                    elif all(isinstance(item, str) for item in v):
                        lines.append(f"{prefix}{k}:")
                        for item in v:
                            lines.append(f"{prefix}  - {item}")
                    else:
                        lines.append(f"{prefix}{k}: {json.dumps(v)}")
                elif isinstance(v, str):
                    # Check if multiline
                    if "\n" in v and len(v.split("\n")) > 2:
                        lines.append(f"{prefix}{k}: |")
                        for line in v.split("\n"):
                            lines.append(f"{prefix}  {line}")
                    else:
                        val = json.dumps(v)
                        lines.append(f"{prefix}{k}: {val}")
                elif v is None:
                    lines.append(f"{prefix}{k}: null")
                else:
                    lines.append(f"{prefix}{k}: {json.dumps(v)}")
        elif isinstance(obj, str):
            lines.append(obj)

    _dump(obj)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate context stress test tasks")
    parser.add_argument(
        "--output-dir", default="tasks/stress_test_context",
        help="Directory to write task YAML files",
    )
    parser.add_argument(
        "--blocks-per-250k", type=int, default=320,
        help="Number of 800-token blocks for 250K target (default 320)",
    )
    parser.add_argument(
        "--paragraphs-per-block", type=int, default=10,
        help="Paragraphs per block (default 10, ~800 tokens)",
    )
    args = parser.parse_args()

    print(f"Generating context stress test tasks to {args.output_dir}/\n")

    num_blocks = {
        10: 13,
        25: 32,
        50: 64,
        100: 128,
        250: args.blocks_per_250k,
    }

    files = generate_stress_tasks(
        output_dir=args.output_dir,
        block_size=args.paragraphs_per_block,
        num_blocks=num_blocks,
    )

    print(f"\nGenerated {len(files)} task files:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
