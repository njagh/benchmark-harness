"""Storage root safety checks for the benchmark harness.

Provides functions to detect unsafe storage roots (inside git repos, /tmp,
virtualenvs, Docker overlay paths, or on nearly-full filesystems) and
ephemeral artifact paths.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_MIN_FREE_SPACE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


def is_unsafe_path(
    path: Path,
    git_root: Path | None = None,
    allow_unsafe: bool = False,
) -> tuple[bool, str | None]:
    """Check whether *path* is an unsafe storage root.

    Returns ``(True, reason)`` if the path is unsafe and
    ``(False, None)`` if it is safe (or ``allow_unsafe`` is ``True``).

    Checks (in order):
    1. Inside the git repository that contains this code.
    2. Inside ``/tmp`` or ``/var/tmp``.
    3. Inside any Python virtualenv (``sys.prefix``).
    4. Inside a Docker overlay path.
    5. Less than 10 GB free disk space.
    """
    if allow_unsafe:
        return False, None

    p = path.resolve()

    # 1. /tmp or /var/tmp
    if str(p).startswith("/tmp") or str(p).startswith("/var/tmp"):
        return True, f"path {p} is inside /tmp or /var/tmp"

    # 2. Inside virtualenv
    venv_prefix = Path(sys.prefix).resolve()
    if _is_child_of(p, venv_prefix):
        return True, f"path {p} is inside the virtualenv at {sys.prefix}"

    # 3. Inside git repo
    if git_root is None:
        git_root = _detect_git_root(p)
    if git_root and _is_child_of(p, git_root):
        return True, f"path {p} is inside the git repo at {git_root}"

    # 4. Docker overlay
    docker_paths = [
        "/var/lib/docker/",
        "/run/containers/",
    ]
    for docker_path in docker_paths:
        if str(p).startswith(docker_path):
            return True, f"path {p} is inside Docker overlay at {docker_path}"

    # 5. Free disk space
    try:
        st = os.statvfs(str(p))
        free_bytes = st.f_bavail * st.f_frsize
        if free_bytes < _MIN_FREE_SPACE_BYTES:
            free_gb = free_bytes / (1024 ** 3)
            return True, (
                f"less than {_MIN_FREE_SPACE_BYTES // (1024**3)} GB free "
                f"on filesystem ({free_gb:.1f} GB free)"
            )
    except OSError:
        pass  # Can't check — be lenient

    return False, None


def check_storage_root(root: Path, allow_unsafe: bool = False) -> None:
    """Raise ``ValueError`` if *root* is unsafe.

    Returns silently if the root is valid.
    """
    is_unsafe, reason = is_unsafe_path(root, allow_unsafe=allow_unsafe)
    if is_unsafe:
        raise ValueError(f"Unsafe storage root: {reason}")


def detect_ephemeral_path(path: str) -> tuple[bool, list[str]]:
    """Detect whether *path* refers to an ephemeral location.

    Returns ``(is_ephemeral, warnings)`` — ``is_ephemeral`` is ``True`` if
    any ephemeral condition was detected and ``warnings`` is a list of
    human-readable warning strings.
    """
    warnings: list[str] = []
    p = Path(path)

    if str(p).startswith("/tmp") or str(p).startswith("/var/tmp"):
        warnings.append(f"path {path} is in an ephemeral location (/tmp or /var/tmp)")

    if not p.exists():
        warnings.append(f"path {path} does not exist")

    # Docker container paths
    docker_indicators = ["/var/lib/docker/", "/run/containers/"]
    for indicator in docker_indicators:
        if str(p).startswith(indicator):
            warnings.append(
                f"path {path} may be inside a Docker container overlay ({indicator})"
            )

    return len(warnings) > 0, warnings


def _detect_git_root(search_path: Path) -> Path | None:
    """Try to detect the top-level git repo containing *search_path*."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(search_path),
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Walk up parents looking for .git directory
    current = search_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def _is_child_of(child: Path, parent: Path) -> bool:
    """Return True if *child* is the same as or a descendant of *parent*."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
