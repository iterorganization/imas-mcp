"""Tests for PR extraction — rebase merge via git trailers."""

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


class TestExtractPrInfoRebase:
    """Rebase merge: PR number in commit body trailers."""

    def test_closes_trailer(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Update electron_temperature description",
            body="Closes: #88",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 88
        assert info.pr_url == "https://github.com/org/repo/pull/88"

    def test_pull_request_trailer(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Refactor kind derivation",
            body="Pull-Request: #201",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 201

    def test_fixes_trailer(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Fix documentation typo",
            body="Some detailed description.\n\nFixes: #15",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 15

    def test_pr_trailer_without_colon(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        sha = _make_commit(
            tmp_path,
            "Add new standard name",
            body="PR #33",
        )

        from imas_codex.standard_names.pr_extractor import extract_pr_info

        info = extract_pr_info(tmp_path, sha, repo_url="https://github.com/org/repo")
        assert info is not None
        assert info.pr_number == 33
