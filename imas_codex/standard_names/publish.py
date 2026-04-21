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


# =============================================================================
# Legacy backward-compatible re-exports
# =============================================================================
# These functions are from the pre-Phase-3 publish module and are kept
# temporarily for backward compatibility with cli/sn.py and existing
# tests. They will be removed by the CLI integration task.

_HIGH_THRESHOLD = 0.8
_MEDIUM_THRESHOLD = 0.5


def confidence_tier(confidence: float) -> str:
    """Classify a confidence score into high / medium / low.

    .. deprecated:: Phase 3
        Legacy function. Use ``export.py`` gate C instead.
    """
    if confidence >= _HIGH_THRESHOLD:
        return "high"
    if confidence >= _MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def generate_yaml_entry(
    entry: Any,
) -> str:
    """Generate YAML content for a single standard name entry.

    .. deprecated:: Phase 3
        Use ``export.py::run_export`` instead.
    """
    doc: dict[str, Any] = {
        "name": entry.name,
        "kind": entry.kind,
    }
    if entry.unit is not None:
        doc["unit"] = entry.unit
    if entry.tags:
        doc["tags"] = entry.tags
    doc["status"] = entry.status
    if entry.description:
        doc["description"] = entry.description
    if entry.documentation:
        doc["documentation"] = entry.documentation
    if entry.links:
        doc["links"] = [{"name": link} for link in entry.links]
    if getattr(entry, "dd_paths", None):
        doc["dd_paths"] = entry.dd_paths
    if entry.constraints:
        doc["constraints"] = entry.constraints
    if entry.validity_domain:
        doc["validity_domain"] = entry.validity_domain
    if getattr(entry, "cocos_transformation_type", None):
        doc["cocos_transformation_type"] = entry.cocos_transformation_type
    if getattr(entry, "cocos", None) is not None:
        doc["cocos"] = entry.cocos
    doc["provenance"] = {
        "source": entry.provenance.source,
        "source_id": entry.provenance.source_id,
    }
    if entry.provenance.ids_name:
        doc["provenance"]["ids_name"] = entry.provenance.ids_name
    doc["provenance"]["confidence"] = entry.provenance.confidence
    doc["provenance"]["generated_by"] = entry.provenance.generated_by

    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False).rstrip("\n")


def generate_catalog_files(
    entries: list[Any],
    output_dir: Path,
) -> list[Path]:
    """Write YAML files to *output_dir*.

    .. deprecated:: Phase 3
        Use ``export.py::run_export`` instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for entry in entries:
        subdir = (
            getattr(entry, "physics_domain", None)
            or (entry.tags[0] if entry.tags else None)
            or "unscoped"
        )
        entry_dir = output_dir / subdir
        entry_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{entry.name}.yaml"
        filepath = entry_dir / filename
        content = generate_yaml_entry(entry)
        filepath.write_text(content + "\n", encoding="utf-8")
        written.append(filepath)

    return written


def batch_by_group(
    entries: list[Any],
    group_by: str = "ids",
) -> dict[str, list[Any]]:
    """Group entries into PR batches.

    .. deprecated:: Phase 3
    """
    groups: dict[str, list[Any]] = {}

    for entry in entries:
        if group_by == "ids":
            key = entry.provenance.ids_name or "unscoped"
        elif group_by == "domain":
            key = (
                getattr(entry, "physics_domain", None)
                or (entry.tags[0] if entry.tags else None)
                or "unscoped"
            )
        elif group_by == "confidence":
            key = confidence_tier(entry.provenance.confidence)
        else:
            key = "all"

        groups.setdefault(key, []).append(entry)

    return groups


def make_publish_batches(
    entries: list[Any],
    group_by: str = "ids",
) -> list[Any]:
    """Create publish batch objects from grouped entries.

    .. deprecated:: Phase 3
    """
    from imas_codex.standard_names.models import StandardNamePublishBatch

    groups = batch_by_group(entries, group_by)
    batches = []
    for key, group_entries in sorted(groups.items()):
        avg_conf = (
            sum(e.provenance.confidence for e in group_entries) / len(group_entries)
            if group_entries
            else 0.0
        )
        batches.append(
            StandardNamePublishBatch(
                group_key=key,
                entries=group_entries,
                confidence_tier=confidence_tier(avg_conf),
            )
        )
    return batches


def check_catalog_duplicates(
    entries: list[Any],
    catalog_dir: Path | None = None,
) -> tuple[list[Any], list[Any]]:
    """Check for duplicates against an existing catalog directory.

    .. deprecated:: Phase 3
    """
    existing_names: set[str] = set()

    if catalog_dir is not None:
        catalog_path = Path(catalog_dir)
        if catalog_path.is_dir():
            for yaml_file in catalog_path.rglob("*.yaml"):
                try:
                    with open(yaml_file, encoding="utf-8") as f:
                        doc = yaml.safe_load(f)
                    if isinstance(doc, dict) and "name" in doc:
                        existing_names.add(doc["name"])
                except Exception:
                    pass

    new: list[Any] = []
    duplicates: list[Any] = []
    seen: set[str] = set()

    for entry in entries:
        if entry.name in existing_names or entry.name in seen:
            duplicates.append(entry)
        else:
            new.append(entry)
            seen.add(entry.name)

    return new, duplicates


def graph_records_to_entries(
    records: list[dict[str, Any]],
) -> list[Any]:
    """Convert raw graph query dicts to publish entry objects.

    .. deprecated:: Phase 3
        Use ``export.py::run_export`` instead.
    """
    from imas_codex.standard_names.models import (
        StandardNameProvenance,
        StandardNamePublishEntry,
    )

    entries = []
    for rec in records:
        name = rec.get("name") or rec.get("id", "")
        if not name:
            continue

        source_types = rec.get("source_types") or []
        source = (
            source_types[0]
            if source_types
            else (rec.get("source") or rec.get("source_type") or "dd")
        )

        source_id = rec.get("source_path") or rec.get("source_id") or ""
        unit = rec.get("unit")
        ids_name = rec.get("ids_name")

        confidence = rec.get("confidence")
        if confidence is None:
            confidence = 1.0

        description = rec.get("description") or ""
        documentation = rec.get("documentation")
        kind = rec.get("kind") or "scalar"
        links_raw = rec.get("links") or []

        from imas_codex.standard_names.source_paths import strip_dd_prefix

        dd_paths_raw = [strip_dd_prefix(p) for p in rec.get("source_paths") or []]
        constraints_raw = rec.get("constraints") or []
        validity_domain = rec.get("validity_domain")
        cocos_transformation_type = rec.get("cocos_transformation_type")
        cocos = rec.get("cocos")

        tags: list[str] = list(rec.get("tags") or [])
        if not tags and ids_name:
            tags.append(ids_name)

        physics_domain = rec.get("physics_domain")

        provenance = StandardNameProvenance(
            source=str(source),
            source_id=str(source_id),
            ids_name=ids_name,
            confidence=float(confidence),
        )

        entries.append(
            StandardNamePublishEntry(
                name=name,
                kind=kind,
                unit=unit,
                tags=tags,
                status="drafted",
                physics_domain=physics_domain,
                description=description[:500] if description else "",
                documentation=documentation,
                links=links_raw if isinstance(links_raw, list) else [],
                dd_paths=dd_paths_raw if isinstance(dd_paths_raw, list) else [],
                constraints=constraints_raw
                if isinstance(constraints_raw, list)
                else [],
                validity_domain=validity_domain,
                cocos_transformation_type=cocos_transformation_type,
                cocos=int(cocos) if cocos is not None else None,
                provenance=provenance,
            )
        )

    return entries


def create_catalog_pr(
    batch: Any,
    catalog_repo: str,
    branch_name: str,
    yaml_files: list[Path],
    dry_run: bool = False,
) -> str | None:
    """Create a PR via ``gh`` CLI.

    .. deprecated:: Phase 3
        Use ``run_publish`` with ``push=True`` instead.
    """
    pr_title = (
        f"feat(sn): add {len(batch.entries)} standard name candidates "
        f"({batch.group_key})"
    )

    if dry_run:
        logger.info(
            "[dry-run] Would create PR on %s: %s",
            catalog_repo,
            pr_title,
        )
        return None

    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                catalog_repo,
                "--head",
                branch_name,
                "--title",
                pr_title,
                "--body",
                "Generated by imas-codex sn publish.",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
