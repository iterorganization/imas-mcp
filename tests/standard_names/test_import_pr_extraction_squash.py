"""Tests for PR extraction — squash merge ``(#123)`` pattern."""

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


class TestExtractPrInfoSquash:
    """Squash merge: ``Fix electron_temperature description (#123)``."""

    def test_squash_merge_extracts_pr(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Fix electron_temperature description (#123)",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 123
        assert info.pr_url == "https://github.com/org/repo/pull/123"

    def test_squash_merge_at_end_of_subject(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(tmp_path, "Update docs (#7)")

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 7

    def test_squash_merge_with_body_text(self, tmp_path: Path) -> None:
        """Body content shouldn't interfere with subject parsing."""
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Improve plasma_current kind (#55)",
            body="This PR improves the kind classification.\n\nSigned-off-by: dev",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 55
