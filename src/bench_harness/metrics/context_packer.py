"""Context packing utilities for long-context benchmarking."""

from __future__ import annotations

import logging
from typing import Any

from bench_harness.tasks.prompt_templates import _estimate_tokens

logger = logging.getLogger(__name__)

# Token budget mapping
CONTEXT_BUDGETS: dict[str, int] = {
    "small": 1024,
    "medium": 4096,
    "large": 16384,
    "xlarge": 65536,
}

# Default distractor text — irrelevant filler
_DEFAULT_DISTRACTOR = """
This is irrelevant context that should not affect the model's ability to
answer the question. It contains no useful information for the task at hand.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia deserunt
mollit anim id est laborum.

Additional filler text to pad the context window without providing any
actionable information. The model should ignore this content.
"""


def _generate_distractor_text(index: int, target_token_count: int) -> str:
    """Generate a block of distractor content with approximately the target token count.

    Args:
        index: Distractor index (used for unique header).
        target_token_count: Approximate token count target.

    Returns:
        Distractor text string.
    """
    parts = [_DEFAULT_DISTRACTOR.strip()]
    # Pad with repeated filler if we need more tokens
    current = "\n".join(parts)
    current_tokens = _estimate_tokens(current)
    attempts = 0
    while current_tokens < target_token_count and attempts < 10:
        parts.append(_DEFAULT_DISTRACTOR.strip())
        current = "\n".join(parts)
        current_tokens = _estimate_tokens(current)
        attempts += 1

    header = f"## Distractor Context Block {index}\n---\n"
    return header + current


class ContextPacker:
    """Packs context files and filler content to approximate a target token budget.

    Usage:
        packer = ContextPacker()
        context = packer.pack(files, target_budget="large", max_budget=32768)
    """

    def pack(
        self,
        files: list[dict[str, str]],
        target_budget: str = "large",
        max_budget: int = 65536,
    ) -> str:
        """Pack context files to approximate the target budget.

        Args:
            files: List of dicts with 'name', 'content' keys.
            target_budget: Context bucket name ("small", "medium", "large", "xlarge").
            max_budget: Absolute max token budget regardless of bucket.

        Returns:
            String of concatenated file contents as markdown code blocks.
        """
        if not files:
            return ""

        budget = min(CONTEXT_BUDGETS.get(target_budget, max_budget), max_budget)
        accumulated: list[str] = []
        tokens_used = 0

        for i, f in enumerate(files):
            content = f.get("content", "")
            name = f.get("name", f"file_{i}")
            block = f"## File: {name}\n```\n{content}\n```"
            file_tokens = _estimate_tokens(block)

            if tokens_used + file_tokens <= budget:
                accumulated.append(block)
                tokens_used += file_tokens
            else:
                # Try truncating to fit budget
                if file_tokens > 0 and budget > tokens_used:
                    available = budget - tokens_used
                    # Approximate truncation: keep roughly proportional text
                    truncated = self._truncate_to_tokens(content, available - 50)
                    if truncated:
                        truncated_block = f"## File: {name}\n```\n{truncated}...\n```"
                        accumulated.append(truncated_block)
                        tokens_used += _estimate_tokens(truncated_block)
                break

        return "\n\n".join(accumulated)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens.

        Args:
            text: Source text.
            max_tokens: Approximate token limit.

        Returns:
            Truncated text.
        """
        if max_tokens <= 0:
            return ""
        words = text.split()
        # Rough estimate: 1 token ~ 1.3 words
        word_limit = int(max_tokens * 1.3)
        word_limit = min(word_limit, len(words))
        return " ".join(words[:word_limit])

    def add_distractors(
        self,
        files: list[dict[str, str]],
        num_distractors: int = 3,
        target_budget: str = "large",
    ) -> list[dict[str, str]]:
        """Add distractor content to increase context without adding relevant info.

        Args:
            files: Original context files.
            num_distractors: Number of irrelevant files to add.
            target_budget: Target context bucket.

        Returns:
            Extended list of files with distractors appended before relevant files.
        """
        if not files:
            return []

        budget = CONTEXT_BUDGETS.get(target_budget, 65536)
        # Estimate tokens used by relevant files
        relevant_tokens = sum(
            _estimate_tokens(f.get("content", "")) for f in files
        )
        remaining_budget = budget - relevant_tokens
        per_distractor = max(512, remaining_budget // max(num_distractors, 1))

        distractors: list[dict[str, str]] = []
        for i in range(num_distractors):
            content = _generate_distractor_text(i + 1, per_distractor)
            distractors.append({
                "name": f"distractor_{i + 1}.txt",
                "content": content,
            })

        return distractors + files

    def apply_relevant_fact_placement(
        self,
        content: str,
        target_position: str = "end",
        fact_text: str | None = None,
    ) -> str:
        """Place the fact that the model needs to retrieve at a specific position.

        Args:
            content: Full context text (files + distractors).
            target_position: "beginning", "middle", or "end".
            fact_text: The relevant fact to place. If None, extracts the last
                markdown file block as the fact.

        Returns:
            Content with the relevant fact moved to the target position.
        """
        if not content:
            return ""

        if fact_text is None:
            # Extract the last file block (the relevant fact)
            fact_text = self._extract_last_file_block(content)
            if not fact_text:
                return content
        else:
            # Find and remove the fact from its current position if it exists
            if fact_text in content:
                content = content.replace(fact_text, "", 1)

        lines = content.split("\n")
        fact_lines = fact_text.split("\n")

        if target_position == "beginning":
            return "\n".join(fact_lines) + "\n\n" + content
        elif target_position == "end":
            if content.strip():
                return content.rstrip() + "\n\n" + "\n".join(fact_lines)
            return "\n".join(fact_lines)
        elif target_position == "middle":
            split_idx = len(lines) // 2
            before = "\n".join(lines[:split_idx])
            after = "\n".join(lines[split_idx:])
            return before + "\n\n" + "\n".join(fact_lines) + "\n\n" + after
        else:
            return content + "\n\n" + "\n".join(fact_lines)

    def _extract_last_file_block(self, content: str) -> str:
        """Extract the last markdown code block from content.

        Args:
            content: Full context text.

        Returns:
            The last file block string, or empty string if none found.
        """
        blocks = []
        current_marker = None
        start_idx = None
        in_code_block = False
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if line.startswith("## File:"):
                if in_code_block and start_idx is not None:
                    block_text = "\n".join(lines[start_idx:i])
                    blocks.append(block_text)
                current_marker = line
                start_idx = i
                in_code_block = True
            elif in_code_block and line.startswith("```"):
                block_text = "\n".join(lines[start_idx:i + 1])
                blocks.append(block_text)
                in_code_block = False
                current_marker = None
                start_idx = None

        if in_code_block and start_idx is not None:
            block_text = "\n".join(lines[start_idx:])
            blocks.append(block_text)

        return blocks[-1] if blocks else ""
