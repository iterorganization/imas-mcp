"""Transport staging directory to an ISNC (imas-standard-names-catalog) checkout.

This module is the second half of the two-step export→publish flow.
It takes a staging directory produced by ``export.py`` and mirrors it
into an ISNC git checkout, creating a commit and optionally pushing.

**No gate logic** — that already ran during ``sn export``.

See plan 35 §Phase 3 (3c).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Report model
# =============================================================================


@dataclass
class PublishReport:
    """Result of a publish operation."""

    staging_dir: str = ""
    isnc_path: str = ""
    files_copied: int = 0
    commit_sha: str | None = None
    pushed: bool = False
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "staging_dir": self.staging_dir,
            "isnc_path": self.isnc_path,
            "files_copied": self.files_copied,
            "commit_sha": self.commit_sha,
            "pushed": self.pushed,
            "dry_run": self.dry_run,
            "errors": self.errors,
        }


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_staging_dir(staging_dir: Path) -> list[str]:
    """Validate that the staging directory is well-formed.

    Returns a list of error strings (empty if valid).
    """
    errors: list[str] = []

    if not staging_dir.is_dir():
        errors.append(f"Staging directory does not exist: {staging_dir}")
        return errors

    manifest = staging_dir / "catalog.yml"
    if not manifest.is_file():
        errors.append(f"Missing manifest: {manifest}")
    else:
        try:
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                errors.append("catalog.yml is not a YAML mapping")
            elif "catalog_name" not in data:
                errors.append("catalog.yml missing required field 'catalog_name'")
        except Exception as exc:
            errors.append(f"catalog.yml parse error: {exc}")

    sn_dir = staging_dir / "standard_names"
    if not sn_dir.is_dir():
        errors.append(f"Missing standard_names directory: {sn_dir}")
    else:
        yml_files = list(sn_dir.rglob("*.yml"))
        if not yml_files:
            errors.append("standard_names/ contains no .yml files")

    return errors


def _get_codex_commit_sha() -> str:
    """Get the current imas-codex git commit SHA (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# =============================================================================
# Main publish function
# =============================================================================


def run_publish(
    staging_dir: str | Path,
    isnc_path: str | Path,
    *,
    push: bool = False,
    dry_run: bool = False,
) -> PublishReport:
    """Transport a staging directory to an ISNC checkout.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory produced by ``sn export``.
    isnc_path:
        Path to a local clone of the imas-standard-names-catalog repo.
    push:
        If ``True``, push the commit to origin after creating it.
    dry_run:
        If ``True``, validate and report without modifying ISNC.

    Returns
    -------
    PublishReport with commit SHA, file counts, and any errors.
    """
    staging = Path(staging_dir)
    isnc = Path(isnc_path)
    report = PublishReport(
        staging_dir=str(staging),
        isnc_path=str(isnc),
        dry_run=dry_run,
    )

    # ── 1. Validate staging directory ───────────────────────────
    errors = _validate_staging_dir(staging)
    if errors:
        report.errors.extend(errors)
        logger.error("Staging validation failed: %s", errors)
        return report

    # ── 2. Validate ISNC path ──────────────────────────────────
    if not isnc.is_dir():
        report.errors.append(f"ISNC path does not exist: {isnc}")
        return report

    git_dir = isnc / ".git"
    if not git_dir.exists():
        report.errors.append(f"ISNC path is not a git repository: {isnc}")
        return report

    if dry_run:
        # Count files that would be copied
        yml_files = list((staging / "standard_names").rglob("*.yml"))
        report.files_copied = len(yml_files) + 1  # +1 for catalog.yml
        logger.info("[dry-run] Would copy %d files to %s", report.files_copied, isnc)
        return report

    # ── 3. Clear ISNC standard_names tree ──────────────────────
    isnc_sn_dir = isnc / "standard_names"
    if isnc_sn_dir.exists():
        shutil.rmtree(isnc_sn_dir)
        logger.debug("Cleared %s", isnc_sn_dir)

    # ── 4. Mirror staging → ISNC ───────────────────────────────
    staging_sn_dir = staging / "standard_names"
    shutil.copytree(staging_sn_dir, isnc_sn_dir)

    staging_manifest = staging / "catalog.yml"
    isnc_manifest = isnc / "catalog.yml"
    shutil.copy2(staging_manifest, isnc_manifest)

    yml_files = list(isnc_sn_dir.rglob("*.yml"))
    report.files_copied = len(yml_files) + 1  # +1 for catalog.yml
    logger.info("Copied %d files to %s", report.files_copied, isnc)

    # ── 5. Git commit ──────────────────────────────────────────
    codex_sha = _get_codex_commit_sha()
    commit_msg = f"chore(catalog): sync from imas-codex {codex_sha}"

    try:
        # Stage all changes
        subprocess.run(
            ["git", "add", "standard_names/", "catalog.yml"],
            cwd=isnc,
            check=True,
            capture_output=True,
            timeout=30,
        )

        # Check if there are changes to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=isnc,
            capture_output=True,
            timeout=10,
        )

        if status.returncode == 0:
            logger.info("No changes to commit in ISNC")
            return report

        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=isnc,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.info("Committed: %s", commit_msg)

        # Get commit SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=isnc,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        report.commit_sha = sha_result.stdout.strip()

    except subprocess.CalledProcessError as exc:
        report.errors.append(f"Git commit failed: {exc.stderr}")
        logger.error("Git commit failed: %s", exc.stderr)
        return report

    # ── 6. Optionally push ─────────────────────────────────────
    if push:
        try:
            subprocess.run(
                ["git", "push", "origin"],
                cwd=isnc,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            report.pushed = True
            logger.info("Pushed to origin")
        except subprocess.CalledProcessError as exc:
            report.errors.append(f"Git push failed: {exc.stderr}")
            logger.error("Git push failed: %s", exc.stderr)

    return report
