"""Extract PR metadata from ISNC git commit history.

Covers three GitHub merge strategies:
  1. Classic merge commit: ``Merge pull request #N from ...``
  2. Squash merge: ``<title> (#N)``
  3. Rebase merge: ``Pull-Request: #N`` or ``Closes: #N`` trailers

Used by ``sn import`` to stamp ``catalog_pr_number`` and ``catalog_pr_url``
on imported StandardName nodes.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for PR number extraction
# ---------------------------------------------------------------------------

#: Classic merge commit: ``Merge pull request #123 from owner/branch``
_MERGE_RE = re.compile(r"Merge pull request #(\d+)\s+from\s+")

#: Squash merge: ``Some PR title (#123)``
_SQUASH_RE = re.compile(r"\(#(\d+)\)(?:\s|$)")

#: Rebase merge trailers in body: ``Closes: #123``, ``Pull-Request: #123``, etc.
_TRAILER_RE = re.compile(
    r"^(?:Closes|Fixes|Refs|Related|Pull-Request|PR)\s*:?\s*#(\d+)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PRInfo:
    """Extracted pull request metadata from a git commit."""

    pr_number: int
    pr_url: str
    commit_sha: str


# ---------------------------------------------------------------------------
# Repo URL helpers
# ---------------------------------------------------------------------------


def _derive_repo_url(repo_dir: Path) -> str | None:
    """Derive a GitHub HTTPS base URL from the repo's origin remote.

    Returns e.g. ``https://github.com/owner/repo`` or None.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        raw = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # SSH: git@github.com:owner/repo.git
    m = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", raw)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"

    # HTTPS: https://github.com/owner/repo.git
    m = re.match(r"https?://([^/]+)/(.+?)(?:\.git)?$", raw)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"

    return None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_pr_info(
    repo_dir: Path,
    commit_sha: str,
    *,
    repo_url: str | None = None,
) -> PRInfo | None:
    """Extract PR metadata from a single git commit.

    Tries three patterns in order:
    1. Classic merge commit subject.
    2. Squash merge ``(#N)`` in subject.
    3. Rebase merge trailers in body.

    Parameters
    ----------
    repo_dir:
        Path to the ISNC git checkout.
    commit_sha:
        Full or abbreviated commit SHA to inspect.
    repo_url:
        GitHub repo base URL (e.g. ``https://github.com/owner/repo``).
        If None, derived from ``remote.origin.url``.

    Returns
    -------
    PRInfo if a PR number was found, None for direct pushes.
    """
    # Read commit subject and body
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H%x00%s%x00%b", commit_sha],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(
                "git log failed for %s: %s", commit_sha, result.stderr.strip()
            )
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not read commit %s: %s", commit_sha, exc)
        return None

    parts = result.stdout.strip().split("\x00", 2)
    if len(parts) < 2:
        return None

    full_sha = parts[0]
    subject = parts[1]
    body = parts[2] if len(parts) > 2 else ""

    # Try pattern 1: classic merge commit
    m = _MERGE_RE.search(subject)
    if m:
        pr_num = int(m.group(1))
        return _build_pr_info(pr_num, full_sha, repo_dir, repo_url)

    # Try pattern 2: squash merge
    m = _SQUASH_RE.search(subject)
    if m:
        pr_num = int(m.group(1))
        return _build_pr_info(pr_num, full_sha, repo_dir, repo_url)

    # Try pattern 3: rebase merge trailers in body
    m = _TRAILER_RE.search(body)
    if m:
        pr_num = int(m.group(1))
        return _build_pr_info(pr_num, full_sha, repo_dir, repo_url)

    logger.debug("No PR reference found in commit %s", commit_sha[:12])
    return None


def _build_pr_info(
    pr_number: int,
    commit_sha: str,
    repo_dir: Path,
    repo_url: str | None,
) -> PRInfo:
    """Build PRInfo with derived or provided repo URL."""
    if repo_url is None:
        repo_url = _derive_repo_url(repo_dir)

    if repo_url:
        pr_url = f"{repo_url}/pull/{pr_number}"
    else:
        pr_url = f"#PR-{pr_number}"

    return PRInfo(pr_number=pr_number, pr_url=pr_url, commit_sha=commit_sha)


# ---------------------------------------------------------------------------
# Batch: walk first-parent commits between watermark and HEAD
# ---------------------------------------------------------------------------


def walk_pr_commits(
    repo_dir: Path,
    from_sha: str | None,
    to_ref: str = "HEAD",
    *,
    repo_url: str | None = None,
) -> list[PRInfo]:
    """Walk first-parent commits from ``from_sha..to_ref`` and extract PR info.

    Parameters
    ----------
    repo_dir:
        Path to the ISNC git checkout.
    from_sha:
        Exclusive start SHA (watermark). If None, walks all history to ``to_ref``.
    to_ref:
        Inclusive end ref (default HEAD).
    repo_url:
        GitHub repo base URL. If None, derived from remote.

    Returns
    -------
    List of PRInfo for commits that have PR references, oldest first.
    """
    if from_sha:
        rev_range = f"{from_sha}..{to_ref}"
    else:
        rev_range = to_ref

    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--first-parent",
                "--reverse",
                "--format=%H%x00%s%x00%b%x01",
                rev_range,
            ],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("git log walk failed: %s", result.stderr.strip())
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not walk commits: %s", exc)
        return []

    if repo_url is None:
        repo_url = _derive_repo_url(repo_dir)

    prs: list[PRInfo] = []
    # Split on record separator \x01
    for record in result.stdout.split("\x01"):
        record = record.strip()
        if not record:
            continue
        parts = record.split("\x00", 2)
        if len(parts) < 2:
            continue

        full_sha = parts[0].strip()
        subject = parts[1].strip()
        body = parts[2].strip() if len(parts) > 2 else ""

        pr_num = None

        # Pattern 1: classic merge
        m = _MERGE_RE.search(subject)
        if m:
            pr_num = int(m.group(1))

        # Pattern 2: squash merge
        if pr_num is None:
            m = _SQUASH_RE.search(subject)
            if m:
                pr_num = int(m.group(1))

        # Pattern 3: rebase trailers
        if pr_num is None:
            m = _TRAILER_RE.search(body)
            if m:
                pr_num = int(m.group(1))

        if pr_num is not None:
            if repo_url:
                pr_url = f"{repo_url}/pull/{pr_num}"
            else:
                pr_url = f"#PR-{pr_num}"
            prs.append(PRInfo(pr_number=pr_num, pr_url=pr_url, commit_sha=full_sha))

    return prs
