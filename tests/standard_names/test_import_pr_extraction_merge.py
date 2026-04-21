"""Tests for PR extraction — classic merge commit.

Uses a temporary git repo with a merge-commit-style message.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _init_repo(d: Path) -> None:
    """Initialise a bare git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=str(d), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )
    (d / "README.md").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=str(d), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )


def _make_commit(d: Path, message: str, body: str = "") -> str:
    """Create a commit with the given subject and optional body."""
    (d / "file.txt").write_text(message)
    subprocess.run(["git", "add", "."], cwd=str(d), capture_output=True, check=True)
    full_msg = f"{message}\n\n{body}" if body else message
    subprocess.run(
        ["git", "commit", "-m", full_msg],
        cwd=str(d),
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(d),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class TestExtractPrInfoMerge:
    """Classic merge commit: ``Merge pull request #42 from owner/branch``."""

    def test_classic_merge_returns_pr_info(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Merge pull request #42 from user/feature-branch",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 42
        assert info.pr_url == "https://github.com/org/repo/pull/42"
        assert info.commit_sha == sha

    def test_classic_merge_high_pr_number(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Merge pull request #9999 from contributor/big-change",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 9999
