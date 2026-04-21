"""PR extraction integration tests — all merge styles + batch walk.

Phase 6e of plan 35: consolidated PR extraction with a single git
fixture repo containing merge, squash, rebase, and direct-push
commits, plus ``walk_pr_commits()`` batch verification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# ============================================================================
# Git helpers
# ============================================================================


def _init_repo(d: Path) -> None:
    """Initialise a git repo with an initial commit."""
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
    (d / "README.md").write_text("initial\n")
    subprocess.run(["git", "add", "."], cwd=str(d), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )


def _make_commit(d: Path, message: str, body: str = "") -> str:
    """Create a commit with the given subject and optional body."""
    # Use a unique file to avoid git-add idempotence
    import time

    fname = f"f_{int(time.time() * 1e6)}.txt"
    (d / fname).write_text(message)
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


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def multi_style_repo(tmp_path: Path) -> dict[str, str]:
    """Git repo with merge, squash, rebase, and direct-push commits.

    Returns dict mapping style name → commit SHA.
    """
    repo = tmp_path / "isnc"
    repo.mkdir()
    _init_repo(repo)

    shas: dict[str, str] = {}

    # Initial commit SHA for watermark
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    shas["initial"] = result.stdout.strip()

    # 1. Classic merge commit
    shas["merge"] = _make_commit(
        repo, "Merge pull request #42 from user/feature-branch"
    )

    # 2. Squash merge
    shas["squash"] = _make_commit(
        repo, "feat: add electron temperature validation (#17)"
    )

    # 3. Rebase merge with trailer
    shas["rebase"] = _make_commit(
        repo,
        "refactor: simplify gate logic",
        body="Pull-Request: #99\nReviewed-by: someone",
    )

    # 4. Direct push (no PR reference)
    shas["direct"] = _make_commit(repo, "chore: update README with no PR reference")

    # 5. Another merge with high PR number
    shas["merge_high"] = _make_commit(
        repo, "Merge pull request #1234 from org/big-feature"
    )

    shas["_repo_dir"] = str(repo)
    return shas


# ============================================================================
# 6e. Single-commit extraction tests
# ============================================================================


class TestExtractPrInfoIntegration:
    """Test extract_pr_info against real git commits with all merge styles."""

    def test_classic_merge(self, multi_style_repo: dict[str, str]) -> None:
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo_dir = Path(multi_style_repo["_repo_dir"])
        info = extract_pr_info(
            repo_dir, multi_style_repo["merge"], repo_url="https://github.com/org/repo"
        )
        assert info is not None
        assert info.pr_number == 42
        assert info.pr_url == "https://github.com/org/repo/pull/42"

    def test_squash_merge(self, multi_style_repo: dict[str, str]) -> None:
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo_dir = Path(multi_style_repo["_repo_dir"])
        info = extract_pr_info(
            repo_dir,
            multi_style_repo["squash"],
            repo_url="https://github.com/org/repo",
        )
        assert info is not None
        assert info.pr_number == 17
        assert info.pr_url == "https://github.com/org/repo/pull/17"

    def test_rebase_with_trailer(self, multi_style_repo: dict[str, str]) -> None:
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo_dir = Path(multi_style_repo["_repo_dir"])
        info = extract_pr_info(
            repo_dir,
            multi_style_repo["rebase"],
            repo_url="https://github.com/org/repo",
        )
        assert info is not None
        assert info.pr_number == 99
        assert info.pr_url == "https://github.com/org/repo/pull/99"

    def test_direct_push_returns_none(self, multi_style_repo: dict[str, str]) -> None:
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo_dir = Path(multi_style_repo["_repo_dir"])
        info = extract_pr_info(
            repo_dir,
            multi_style_repo["direct"],
            repo_url="https://github.com/org/repo",
        )
        assert info is None

    def test_high_pr_number(self, multi_style_repo: dict[str, str]) -> None:
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo_dir = Path(multi_style_repo["_repo_dir"])
        info = extract_pr_info(
            repo_dir,
            multi_style_repo["merge_high"],
            repo_url="https://github.com/org/repo",
        )
        assert info is not None
        assert info.pr_number == 1234

    def test_no_repo_url_uses_fallback_format(self, tmp_path: Path) -> None:
        """When repo_url is None and no remote, PR URL uses #PR-N format."""
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo = tmp_path / "no_remote"
        repo.mkdir()
        _init_repo(repo)
        sha = _make_commit(repo, "Merge pull request #5 from user/branch")

        info = extract_pr_info(repo, sha)  # no repo_url
        assert info is not None
        assert info.pr_number == 5
        # Without a remote, should use fallback format
        assert "#" in info.pr_url and "5" in info.pr_url

    def test_commit_sha_recorded_correctly(
        self, multi_style_repo: dict[str, str]
    ) -> None:
        """PRInfo.commit_sha stores the full SHA of the inspected commit."""
        from imas_codex.standard_names.pr_extractor import extract_pr_info

        repo_dir = Path(multi_style_repo["_repo_dir"])
        info = extract_pr_info(
            repo_dir, multi_style_repo["merge"], repo_url="https://github.com/org/repo"
        )
        assert info is not None
        assert len(info.commit_sha) == 40  # full SHA
        assert info.commit_sha == multi_style_repo["merge"]


# ============================================================================
# Batch walk tests
# ============================================================================


class TestWalkPrCommitsIntegration:
    """walk_pr_commits batch extraction from watermark to HEAD."""

    def test_walk_from_initial_finds_all_prs(
        self, multi_style_repo: dict[str, str]
    ) -> None:
        """Walking from initial commit should find PRs #42, #17, #99, #1234."""
        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        repo_dir = Path(multi_style_repo["_repo_dir"])
        prs = walk_pr_commits(
            repo_dir,
            multi_style_repo["initial"],
            repo_url="https://github.com/org/repo",
        )

        pr_numbers = [pr.pr_number for pr in prs]
        assert 42 in pr_numbers
        assert 17 in pr_numbers
        assert 99 in pr_numbers
        assert 1234 in pr_numbers
        # Direct push should NOT appear
        assert all(pr.pr_number != 0 for pr in prs)

    def test_walk_excludes_direct_push(self, multi_style_repo: dict[str, str]) -> None:
        """Direct push commit has no PR reference — excluded from walk results."""
        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        repo_dir = Path(multi_style_repo["_repo_dir"])
        prs = walk_pr_commits(
            repo_dir,
            multi_style_repo["initial"],
            repo_url="https://github.com/org/repo",
        )

        # 4 PR commits, 1 direct push
        assert len(prs) == 4

    def test_walk_from_partial_range(self, multi_style_repo: dict[str, str]) -> None:
        """Walking from after squash should find only rebase, direct-skip, merge_high."""
        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        repo_dir = Path(multi_style_repo["_repo_dir"])
        prs = walk_pr_commits(
            repo_dir,
            multi_style_repo["squash"],  # start after squash
            repo_url="https://github.com/org/repo",
        )

        pr_numbers = [pr.pr_number for pr in prs]
        assert 42 not in pr_numbers  # before the range
        assert 17 not in pr_numbers  # excluded (start point)
        assert 99 in pr_numbers
        assert 1234 in pr_numbers

    def test_walk_oldest_first(self, multi_style_repo: dict[str, str]) -> None:
        """Walk results should be in chronological order (oldest first)."""
        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        repo_dir = Path(multi_style_repo["_repo_dir"])
        prs = walk_pr_commits(
            repo_dir,
            multi_style_repo["initial"],
            repo_url="https://github.com/org/repo",
        )

        # PR order should be: 42, 17, 99, 1234
        pr_numbers = [pr.pr_number for pr in prs]
        assert pr_numbers == [42, 17, 99, 1234]

    def test_walk_from_none_includes_all_history(
        self, multi_style_repo: dict[str, str]
    ) -> None:
        """from_sha=None walks all history."""
        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        repo_dir = Path(multi_style_repo["_repo_dir"])
        prs = walk_pr_commits(
            repo_dir,
            None,  # no watermark
            repo_url="https://github.com/org/repo",
        )

        # Should include all PR commits
        pr_numbers = {pr.pr_number for pr in prs}
        assert {42, 17, 99, 1234} <= pr_numbers

    def test_walk_head_to_head_empty(self, multi_style_repo: dict[str, str]) -> None:
        """Walking from HEAD to HEAD produces empty results."""
        from imas_codex.standard_names.pr_extractor import walk_pr_commits

        repo_dir = Path(multi_style_repo["_repo_dir"])
        prs = walk_pr_commits(
            repo_dir,
            multi_style_repo["merge_high"],  # latest commit
            repo_url="https://github.com/org/repo",
        )

        assert len(prs) == 0
