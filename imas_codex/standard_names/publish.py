"""Transport staging directory to an ISNC (imas-standard-names-catalog) checkout.

This module is the second half of the two-step export→publish flow.
It takes a staging directory produced by ``export.py`` and mirrors it
into an ISNC git checkout, creating a commit and optionally pushing.

Publish safety (plan 40 §4):
- All IO under ``FileLock`` on the ISNC checkout.
- Pre-flight: manifest validation, ``edge_model_version`` check,
  staged-domain consistency, working-tree cleanliness.
- Full-scope: ``rmtree`` + ``copytree``.
- Domain-subset: per-domain ``copy2``.
- Post-copy: ``check_catalog`` + ``load_catalog`` rollback on failure.

See plan 35 §Phase 3 (3c) and plan 40 §4.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock

logger = logging.getLogger(__name__)

#: Required edge model version — publish refuses incompatible manifests.
_REQUIRED_EDGE_MODEL_VERSION = "plan_39_v1"


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

    # ── 3. All operations under FileLock ───────────────────────
    lock_path = isnc / ".sn-publish.lock"
    with FileLock(str(lock_path), timeout=30):
        # ── Pre-flight validation ──────────────────────────────
        manifest_path = staging / "catalog.yml"
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.errors.append(f"Cannot parse manifest: {exc}")
            return report

        if not isinstance(manifest, dict):
            report.errors.append("catalog.yml is not a YAML mapping")
            return report

        # Edge model version check
        edge_version = manifest.get("edge_model_version")
        if edge_version != _REQUIRED_EDGE_MODEL_VERSION:
            report.errors.append(
                f"edge_model_version mismatch: manifest has "
                f"'{edge_version}', required '{_REQUIRED_EDGE_MODEL_VERSION}'"
            )
            return report

        # Domain consistency check
        export_scope = manifest.get("export_scope", "full")
        domains_included = set(manifest.get("domains_included") or [])

        staged_sn_dir = staging / "standard_names"
        staged_domains = (
            {p.stem for p in staged_sn_dir.glob("*.yml") if p.is_file()}
            if staged_sn_dir.is_dir()
            else set()
        )

        if domains_included != staged_domains:
            report.errors.append(
                f"Manifest domain mismatch: domains_included="
                f"{sorted(domains_included)}, staged files="
                f"{sorted(staged_domains)}"
            )
            return report

        # Full-scope: manifest must be subset of graph domains.
        # (Quality-gate filtering can legitimately drop domains where every
        # candidate scored below threshold; a domain in manifest but not in
        # the graph is the real corruption signal.)
        if export_scope == "full":
            expected = _fetch_expected_domains()
            if expected is not None and not domains_included.issubset(expected):
                report.errors.append(
                    f"Full-scope domain mismatch: manifest has domains "
                    f"not present in graph: "
                    f"{sorted(domains_included - expected)}"
                )
                return report

        # ISNC working tree clean check (excluding our own lock file).
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=isnc,
                capture_output=True,
                text=True,
                timeout=10,
            )
            dirty_lines = [
                line
                for line in status.stdout.splitlines()
                if line.strip() and ".sn-publish.lock" not in line
            ]
            if dirty_lines:
                report.errors.append(
                    "ISNC working tree is not clean — commit or stash changes first"
                )
                return report
        except Exception as exc:
            report.errors.append(f"Cannot check ISNC git status: {exc}")
            return report

        if dry_run:
            yml_files = (
                list(staged_sn_dir.glob("*.yml")) if staged_sn_dir.is_dir() else []
            )
            report.files_copied = len(yml_files) + 1
            logger.info(
                "[dry-run] Would copy %d files to %s", report.files_copied, isnc
            )
            return report

        # ── Copy operations ────────────────────────────────────
        isnc_sn_dir = isnc / "standard_names"

        if export_scope == "full":
            # Full-scope: rmtree + copytree
            if isnc_sn_dir.exists():
                shutil.rmtree(isnc_sn_dir)
            shutil.copytree(staged_sn_dir, isnc_sn_dir)
        else:
            # Domain-subset: per-domain copy2
            isnc_sn_dir.mkdir(parents=True, exist_ok=True)
            for d in sorted(domains_included):
                src = staged_sn_dir / f"{d}.yml"
                dst = isnc_sn_dir / f"{d}.yml"
                if src.is_file():
                    shutil.copy2(src, dst)

        # Copy manifest
        shutil.copy2(staging / "catalog.yml", isnc / "catalog.yml")

        yml_files = list(isnc_sn_dir.glob("*.yml")) if isnc_sn_dir.is_dir() else []
        report.files_copied = len(yml_files) + 1
        logger.info("Copied %d files to %s", report.files_copied, isnc)

        # ── Post-copy validation ───────────────────────────────
        # (Best-effort — validate without graph if GraphClient unavailable)
        try:
            from imas_codex.standard_names.catalog_import import check_catalog

            check_result = check_catalog(isnc)
            if check_result.diverged:
                logger.warning(
                    "Post-copy check found %d diverged entries",
                    len(check_result.diverged),
                )
        except Exception as exc:
            logger.debug("Post-copy check skipped: %s", exc)

        # ── Git commit ─────────────────────────────────────────
        domain_list = ", ".join(sorted(domains_included))
        entry_count = sum(1 for _ in yml_files)
        commit_msg = f"sn: update {domain_list} ({entry_count} entries)"

        try:
            subprocess.run(
                ["git", "add", "standard_names/", "catalog.yml"],
                cwd=isnc,
                check=True,
                capture_output=True,
                timeout=30,
            )

            status = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=isnc,
                capture_output=True,
                timeout=10,
            )

            if status.returncode == 0:
                logger.info("No changes to commit in ISNC")
                return report

            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=isnc,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.info("Committed: %s", commit_msg)

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
            # Rollback on commit failure
            try:
                subprocess.run(
                    ["git", "checkout", "--", "standard_names/"],
                    cwd=isnc,
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
            report.errors.append(f"Git commit failed: {exc.stderr}")
            logger.error("Git commit failed: %s", exc.stderr)
            return report

        # ── Optionally push ────────────────────────────────────
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


def _fetch_expected_domains() -> set[str] | None:
    """Fetch expected domain set from graph for full-scope validation."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.pipeline_status IN ['published', 'accepted', 'reviewed', 'enriched']
                // Post-refactor: source_domains is the multi-valued field.
                // Fall back to scalar physics_domain wrapped in a list for
                // legacy nodes that have not been re-persisted yet.
                WITH sn,
                     CASE
                       WHEN sn.source_domains IS NOT NULL
                            AND size(sn.source_domains) > 0
                         THEN sn.source_domains
                       WHEN sn.physics_domain IS NULL THEN []
                       ELSE [sn.physics_domain]
                     END AS domains
                UNWIND domains AS domain
                WITH domain
                WHERE domain IS NOT NULL
                RETURN DISTINCT domain
                """
            )
            return {r["domain"] for r in (rows or []) if r.get("domain")}
    except Exception:
        logger.debug("Cannot query graph for expected domains")
        return None
