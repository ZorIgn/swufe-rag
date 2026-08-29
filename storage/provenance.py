"""Measured source-control provenance for reproducible build and evaluation jobs."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from storage.release import sha256_file


def git_provenance(*, worktree: str | Path | None = None) -> dict[str, object]:
    """Measure HEAD and all tracked/untracked changes without trusting CLI labels."""

    cwd = Path(worktree) if worktree is not None else None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=10,
        )
        diff_result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=15,
        )
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "diff_sha256": None,
        }

    commit = commit_result.stdout.decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "diff_sha256": None,
        }
    status = status_result.stdout
    dirty = bool(status)
    fingerprint = hashlib.sha256()
    fingerprint.update(status)
    fingerprint.update(b"\0tracked-diff\0")
    fingerprint.update(diff_result.stdout)
    base = cwd or Path.cwd()
    for raw_path in sorted(path for path in untracked_result.stdout.split(b"\0") if path):
        fingerprint.update(b"\0untracked\0")
        fingerprint.update(raw_path)
        candidate = base / os.fsdecode(raw_path)
        if candidate.is_file() and not candidate.is_symlink():
            fingerprint.update(sha256_file(candidate).encode("ascii"))
    return {
        "available": True,
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": fingerprint.hexdigest() if dirty else None,
    }


__all__ = ["git_provenance"]
