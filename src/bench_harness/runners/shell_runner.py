"""Shell runner — validates generated shell/config artifacts by running commands."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)


class ShellRunner:
    """Validates generated shell/config artifacts by running validation commands.

    Usage:
        runner = ShellRunner(allowlist=["docker compose", "python -c"])
        result = runner.run("docker compose config -q", "/path/to/project")
    """

    def __init__(self, allowlist: list[str] | None = None):
        """Initialize with an allowlist of permitted command prefixes.

        Args:
            allowlist: Commands (or prefixes) that are permitted to run.
                If empty, all commands are allowed. If provided, a command
                is allowed if any allowlist entry is a prefix of the command
                or its first word.
        """
        self.allowlist = allowlist or []

    def run(self, command: str, working_dir: str, timeout: int = 60) -> dict[str, Any]:
        """Run a validation command in a temp directory and return results.

        Args:
            command: Shell command to execute.
            working_dir: Directory to run the command in.
            timeout: Maximum seconds to wait for command completion.

        Returns:
            Dict with keys: exit_code, stdout, stderr, passed, duration_ms.
            'passed' is True when exit_code == 0.
        """
        if not self._is_allowed(command):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Command not in allowlist: {command}",
                "passed": False,
                "duration_ms": 0,
            }

        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": result.returncode == 0,
                "duration_ms": duration_ms,
            }

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"Command timed out after {timeout}s"
            logger.warning("ShellRunner timeout: %s — %s", command, error_msg)
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": error_msg,
                "passed": False,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = str(e)
            logger.exception("ShellRunner failed: %s — %s", command, e)
            return {
                "exit_code": -2,
                "stdout": "",
                "stderr": error_msg,
                "passed": False,
                "duration_ms": duration_ms,
            }

    def _is_allowed(self, command: str) -> bool:
        """Check if command is permitted by the allowlist.

        A command is allowed if:
        - allowlist is empty (no restrictions)
        - any allowlist entry is a prefix of the full command string
        - any allowlist entry matches the first word of the command
        """
        if not self.allowlist:
            return True

        first_word = command.split()[0] if command.strip() else ""

        for allowed in self.allowlist:
            if command.startswith(allowed) or first_word == allowed:
                return True

        return False
