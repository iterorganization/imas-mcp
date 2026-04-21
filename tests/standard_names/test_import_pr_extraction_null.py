"""Tests for PR extraction — direct push with no PR reference."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _init_repo(d: Path) -> None:
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


class TestExtractPrInfoNull:
    """Direct push with no PR reference → returns None."""

    def test_plain_commit_returns_none(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(tmp_path, "chore: update README")

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is None

    def test_commit_with_hash_in_body_but_no_trailer(self, tmp_path: Path) -> None:
        """A stray ``#42`` in body text should NOT be extracted as a PR."""
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "chore: bump dependencies",
            body="This relates to issue #42 but is not a PR ref.",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        # The body mentions #42 but not in a trailer-style line
        # The regex requires it to start the line with a keyword
        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is None

    def test_walk_yields_empty_for_direct_push(self, tmp_path: Path) -> None:
        """walk_pr_commits with no PR commits returns empty list."""
        _init_repo(tmp_path)
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        _make_commit(tmp_path, "direct push 1")
        _make_commit(tmp_path, "direct push 2")

        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        prs = walk_pr_commits(
            tmp_path,
            from_sha=initial_sha,
            repo_url="https://github.com/org/repo",
        )
        assert prs == []
