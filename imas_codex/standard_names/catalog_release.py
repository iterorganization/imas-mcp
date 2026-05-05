"""Release workflow for the ISNC (imas-standard-names-catalog).

Orchestrates the full release cycle: export → publish → tag → push.
The state machine follows the same two-state pattern as codex and ISN
releases (Stable ↔ RC mode).

State machine:
    Stable (v1.0.0) ──bump──→ RC (v1.1.0rc1) ──rc──→ (v1.1.0rc2) ──final──→ Stable (v1.1.0)
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Only match clean semver tags (ignore legacy suffixed tags like v0.3.0-rc1-w40-corpus)
_SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")

_RC_REMOTE = "origin"
_FINAL_REMOTE = "upstream"


# =============================================================================
# Report model
# =============================================================================


@dataclass
class ReleaseReport:
    """Result of a catalog release operation."""

    version: str = ""
    git_tag: str = ""
    remote: str = ""
    export_count: int = 0
    files_copied: int = 0
    commit_sha: str | None = None
    pushed: bool = False
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "git_tag": self.git_tag,
            "remote": self.remote,
            "export_count": self.export_count,
            "files_copied": self.files_copied,
            "commit_sha": self.commit_sha,
            "pushed": self.pushed,
            "dry_run": self.dry_run,
            "errors": self.errors,
        }


# =============================================================================
# Git helpers (operate on ISNC checkout)
# =============================================================================


def _run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command in the ISNC checkout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _format_tag(major: int, minor: int, patch: int, rc: int | None) -> str:
    """Format version components as a git tag (v1.0.0 or v1.0.0rc1)."""
    base = f"v{major}.{minor}.{patch}"
    return f"{base}rc{rc}" if rc else base


def _parse_version(tag: str) -> tuple[int, int, int, int | None]:
    """Parse a version tag into (major, minor, patch, rc_number|None).

    Handles: v1.0.0, v1.0.0rc1
    """
    match = _SEMVER_RE.match(tag)
    if not match:
        raise ValueError(f"Cannot parse version tag: {tag}")
    major, minor, patch = int(match[1]), int(match[2]), int(match[3])
    rc = int(match[4]) if match[4] else None
    return major, minor, patch, rc


def _tag_exists(tag: str, *, cwd: Path | None = None) -> bool:
    result = _run_git("tag", "-l", tag, cwd=cwd)
    return bool(result.stdout.strip())


def _commits_since_tag(tag: str, *, cwd: Path | None = None) -> int:
    result = _run_git("rev-list", f"{tag}..HEAD", "--count", cwd=cwd)
    return int(result.stdout.strip()) if result.returncode == 0 else 0


# =============================================================================
# State detection
# =============================================================================


def _get_semver_tags(cwd: Path | None = None) -> list[str]:
    """Get all clean semver tags, sorted by version (descending)."""
    result = _run_git("tag", "--sort=-v:refname", cwd=cwd)
    if result.returncode != 0:
        return []
    return [
        tag.strip()
        for tag in result.stdout.strip().splitlines()
        if _SEMVER_RE.match(tag.strip())
    ]


def detect_state(isnc_path: Path, *, fetch_remote: str | None = None) -> dict:
    """Detect current release state from ISNC git tags.

    Parameters
    ----------
    isnc_path:
        Path to the ISNC git checkout.
    fetch_remote:
        If provided, fetch tags from this remote before detecting state.

    Returns
    -------
    Dict with keys: state, tag, major, minor, patch, rc, commits_since.
    """
    if fetch_remote:
        _run_git("fetch", "--tags", fetch_remote, cwd=isnc_path)

    tags = _get_semver_tags(cwd=isnc_path)
    if not tags:
        return {
            "state": None,
            "tag": None,
            "major": 0,
            "minor": 0,
            "patch": 0,
            "rc": None,
            "commits_since": 0,
        }

    latest = tags[0]
    major, minor, patch, rc = _parse_version(latest)
    state = "rc" if rc is not None else "stable"
    commits = _commits_since_tag(latest, cwd=isnc_path)

    return {
        "state": state,
        "tag": latest,
        "major": major,
        "minor": minor,
        "patch": patch,
        "rc": rc,
        "commits_since": commits,
    }


def _get_latest_stable_tag(cwd: Path | None = None) -> str | None:
    """Get the most recent stable (non-RC) tag."""
    for tag in _get_semver_tags(cwd=cwd):
        _, _, _, rc = _parse_version(tag)
        if rc is None:
            return tag
    return None


def _apply_bump(major: int, minor: int, patch: int, bump: str) -> tuple[int, int, int]:
    if bump == "major":
        return major + 1, 0, 0
    if bump == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def compute_next_version(
    isnc_path: Path,
    bump: str | None,
    *,
    final: bool = False,
) -> tuple[str, str]:
    """Compute next version tag from current ISNC state.

    Returns (git_tag, version_string) e.g. ("v1.0.0rc1", "1.0.0rc1").

    Raises
    ------
    ValueError
        If on stable and no bump specified, or other invalid transitions.
    """
    info = detect_state(isnc_path)
    state = info["state"]
    major, minor, patch = info["major"], info["minor"], info["patch"]

    if state is None:
        # No tags at all — start fresh
        if bump:
            m, n, p = _apply_bump(0, 0, 0, bump)
        else:
            m, n, p = 1, 0, 0  # Default to v1.0.0
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc)
        return tag, tag.lstrip("v")

    if state == "stable":
        if not bump:
            raise ValueError(
                f"On stable release {info['tag']}. "
                "Specify --bump (major|minor|patch) to start a new release."
            )
        m, n, p = _apply_bump(major, minor, patch, bump)
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc)
        return tag, tag.lstrip("v")

    # RC mode
    if bump:
        # Abandon current RC, start new series from latest stable
        stable = _get_latest_stable_tag(cwd=isnc_path)
        if stable:
            s_maj, s_min, s_pat, _ = _parse_version(stable)
        else:
            s_maj, s_min, s_pat = major, minor, patch
        m, n, p = _apply_bump(s_maj, s_min, s_pat, bump)
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc)
        return tag, tag.lstrip("v")

    if final:
        # Finalize: v1.0.0rc2 → v1.0.0
        tag = _format_tag(major, minor, patch, None)
        return tag, tag.lstrip("v")

    # Increment RC: v1.0.0rc1 → v1.0.0rc2
    next_rc = info["rc"] + 1
    tag = _format_tag(major, minor, patch, next_rc)
    return tag, tag.lstrip("v")


# =============================================================================
# Pre-flight checks
# =============================================================================


def _check_on_main(isnc_path: Path) -> None:
    result = _run_git("branch", "--show-current", cwd=isnc_path)
    branch = result.stdout.strip()
    if branch != "main":
        raise ValueError(
            f"ISNC not on main branch (current: {branch}). "
            f"Switch first: cd {isnc_path} && git checkout main"
        )


def _check_clean_tree(isnc_path: Path, *, strict: bool = True) -> list[str]:
    """Check if ISNC working tree is clean.

    Returns list of warning strings (empty if clean).
    Raises ValueError if strict and dirty.
    """
    result = _run_git("status", "--porcelain", cwd=isnc_path)
    dirty_lines = [
        line
        for line in result.stdout.strip().splitlines()
        if line.strip() and ".sn-publish.lock" not in line
    ]
    if dirty_lines:
        if strict:
            raise ValueError(
                f"ISNC working tree has {len(dirty_lines)} uncommitted change(s). "
                "Commit changes first."
            )
        return [
            f"Working tree has {len(dirty_lines)} uncommitted change(s) "
            "(allowed for RC)"
        ]
    return []


def _check_synced(isnc_path: Path, remote: str, *, strict: bool = True) -> list[str]:
    """Check if ISNC is synced with the target remote.

    Returns list of warning strings.
    Raises ValueError if strict and out of sync.
    """
    _run_git("fetch", remote, "main", cwd=isnc_path)
    result = _run_git(
        "rev-list",
        "--left-right",
        "--count",
        f"main...{remote}/main",
        cwd=isnc_path,
    )
    if result.returncode != 0:
        return [f"Could not check sync with {remote}/main"]

    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return []

    ahead, behind = int(parts[0]), int(parts[1])
    warnings = []

    if behind > 0:
        msg = (
            f"ISNC is {behind} commits behind {remote}/main. "
            f"Pull first: cd {isnc_path} && git pull {remote} main"
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    if ahead > 0:
        msg = (
            f"ISNC is {ahead} commits ahead of {remote}/main. "
            f"Push first: cd {isnc_path} && git push {remote} main"
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    return warnings


# =============================================================================
# Release status display
# =============================================================================


def get_release_status(isnc_path: Path) -> dict[str, Any]:
    """Get ISNC release status for display.

    Returns dict with state info, available commands, ISN dep version, etc.
    """
    info = detect_state(isnc_path, fetch_remote="origin")

    # Get ISN dependency version from ISNC pyproject.toml
    isn_version = _get_isn_dep_version(isnc_path)

    # Get remotes
    remotes = {}
    for name in ("origin", "upstream"):
        result = _run_git("remote", "get-url", name, cwd=isnc_path)
        if result.returncode == 0:
            remotes[name] = result.stdout.strip()

    # Check GitHub Pages
    pages_enabled = _check_pages_status(isnc_path)

    return {
        **info,
        "isnc_path": str(isnc_path),
        "isn_version": isn_version,
        "remotes": remotes,
        "pages_enabled": pages_enabled,
    }


def _get_isn_dep_version(isnc_path: Path) -> str | None:
    """Extract ISN dependency version from ISNC pyproject.toml."""
    pyproject = isnc_path / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        content = pyproject.read_text(encoding="utf-8")
        # Look for the ISN git dependency tag
        match = re.search(r"imas-standard-names.*@(v[\d.]+(?:rc\d+)?)", content)
        if match:
            return match.group(1)
        # Fallback: look for version specifier
        match = re.search(r"imas-standard-names[>=<~!]*\s*([\d.]+(?:rc\d+)?)", content)
        return match.group(1) if match else None
    except Exception:
        return None


def _check_pages_status(isnc_path: Path) -> bool | None:
    """Check if gh-pages branch exists (proxy for GitHub Pages setup)."""
    result = _run_git("ls-remote", "--heads", "origin", "gh-pages", cwd=isnc_path)
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


# =============================================================================
# Main release function
# =============================================================================


def run_release(
    isnc_path: Path,
    message: str,
    *,
    staging_dir: Path | None = None,
    bump: str | None = None,
    final: bool = False,
    remote: str | None = None,
    dry_run: bool = False,
    skip_export: bool = False,
    export_kwargs: dict[str, Any] | None = None,
) -> ReleaseReport:
    """Run the full catalog release workflow.

    Steps:
    1. Pre-flight checks on ISNC checkout
    2. Auto-export (graph → staging) unless skip_export
    3. Copy staging → ISNC (publish)
    4. Compute next version tag
    5. Git commit in ISNC
    6. Create git tag
    7. Push commit + tag to remote

    Parameters
    ----------
    isnc_path:
        Path to the ISNC git checkout.
    message:
        Release message (used for git tag annotation and commit).
    staging_dir:
        Staging directory. If None, uses default from settings.
    bump:
        Version bump type (major, minor, patch). Required for first
        release or when on a stable tag.
    final:
        If True, finalize current RC to stable release.
    remote:
        Git remote to push to. Default: origin for RC, upstream for final.
    dry_run:
        Validate and report without making changes.
    skip_export:
        Skip the export step (use existing staging content).
    export_kwargs:
        Additional kwargs for run_export (e.g., min_score, domain).

    Returns
    -------
    ReleaseReport with version, tag, commit SHA, and any errors.
    """
    report = ReleaseReport(dry_run=dry_run)

    # ── Resolve paths ──────────────────────────────────────
    if staging_dir is None:
        from imas_codex.settings import get_sn_staging_dir

        staging_dir = get_sn_staging_dir()

    is_rc = not final
    effective_remote = remote or (_FINAL_REMOTE if final else _RC_REMOTE)
    report.remote = effective_remote

    # ── Pre-flight checks ──────────────────────────────────
    logger.info("Pre-flight checks on %s", isnc_path)

    try:
        _check_on_main(isnc_path)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    try:
        warnings = _check_clean_tree(isnc_path, strict=not is_rc)
        for w in warnings:
            logger.warning(w)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    try:
        warnings = _check_synced(isnc_path, effective_remote, strict=not dry_run)
        for w in warnings:
            logger.warning(w)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    # ── Compute version ────────────────────────────────────
    # Fetch tags from remote before computing version
    _run_git("fetch", "--tags", effective_remote, cwd=isnc_path)

    try:
        git_tag, version = compute_next_version(isnc_path, bump, final=final)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    report.git_tag = git_tag
    report.version = version

    if _tag_exists(git_tag, cwd=isnc_path):
        report.errors.append(f"Tag {git_tag} already exists")
        return report

    logger.info("Next version: %s (tag: %s)", version, git_tag)

    # ── Auto-export ────────────────────────────────────────
    if not skip_export:
        from imas_codex.standard_names.export import run_export

        staging_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "staging_dir": staging_dir,
            "force": True,  # Overwrite existing staging
            **(export_kwargs or {}),
        }

        logger.info("Exporting to %s", staging_dir)
        try:
            export_report = run_export(**kwargs)
            report.export_count = export_report.exported_count

            if not export_report.all_gates_passed:
                failed = [g.gate for g in export_report.gate_results if not g.passed]
                report.errors.append(
                    f"Export quality gates failed: {', '.join(failed)}. "
                    "Fix issues or pass --skip-export to bypass."
                )
                return report
        except Exception as exc:
            report.errors.append(f"Export failed: {exc}")
            return report
    else:
        # Validate existing staging
        catalog_yml = staging_dir / "catalog.yml"
        if not catalog_yml.is_file():
            report.errors.append(
                f"No catalog.yml found at {staging_dir}. "
                "Run 'sn export' first, or remove --skip-export."
            )
            return report

    # ── Publish (copy staging → ISNC) ─────────────────────
    from imas_codex.standard_names.publish import run_publish

    logger.info("Publishing to %s", isnc_path)
    pub_report = run_publish(
        staging_dir=str(staging_dir),
        isnc_path=str(isnc_path),
        push=False,  # We handle push ourselves (with tag)
        dry_run=dry_run,
    )

    if pub_report.errors:
        report.errors.extend(pub_report.errors)
        return report

    report.files_copied = pub_report.files_copied
    report.commit_sha = pub_report.commit_sha

    if dry_run:
        logger.info(
            "[dry-run] Would tag %s and push to %s",
            git_tag,
            effective_remote,
        )
        return report

    # If publish created no commit (no changes), still tag and push
    if report.commit_sha is None:
        logger.info("No changes to commit — tagging current HEAD")

    # ── Create tag ─────────────────────────────────────────
    tag_result = _run_git("tag", "-a", git_tag, "-m", message, cwd=isnc_path)
    if tag_result.returncode != 0:
        report.errors.append(f"Failed to create tag: {tag_result.stderr}")
        return report
    logger.info("Created tag %s", git_tag)

    # ── Push commit + tag ──────────────────────────────────
    # Push main branch first (if there's a new commit)
    if report.commit_sha:
        push_result = _run_git("push", effective_remote, "main", cwd=isnc_path)
        if push_result.returncode != 0:
            # Roll back the tag on push failure
            _run_git("tag", "-d", git_tag, cwd=isnc_path)
            report.errors.append(
                f"Failed to push to {effective_remote}: {push_result.stderr}"
            )
            return report

    # Push tag
    tag_push_result = _run_git("push", effective_remote, git_tag, cwd=isnc_path)
    if tag_push_result.returncode != 0:
        # Roll back the tag on push failure
        _run_git("tag", "-d", git_tag, cwd=isnc_path)
        report.errors.append(
            f"Failed to push tag to {effective_remote}: {tag_push_result.stderr}"
        )
        return report

    report.pushed = True
    logger.info("Pushed %s to %s", git_tag, effective_remote)

    return report
