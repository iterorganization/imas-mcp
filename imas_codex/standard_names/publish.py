"""Publish validated standard names to YAML catalog files.

Converts validated StandardName graph nodes into YAML files matching
the ``imas-standard-names-catalog`` format.  Supports batching by IDS,
domain, or confidence tier, and optional GitHub PR creation via ``gh``.

Usage (from CLI)::

    imas-codex sn publish --output-dir sn_catalog_output
    imas-codex sn publish --group-by ids --dry-run
    imas-codex sn publish --create-pr --catalog-repo org/repo
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from imas_codex.standard_names.models import (
    StandardNameProvenance,
    StandardNamePublishBatch,
    StandardNamePublishEntry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Confidence tier classification
# =============================================================================

_HIGH_THRESHOLD = 0.8
_MEDIUM_THRESHOLD = 0.5


def confidence_tier(confidence: float) -> str:
    """Classify a confidence score into high / medium / low."""
    if confidence >= _HIGH_THRESHOLD:
        return "high"
    if confidence >= _MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# =============================================================================
# YAML generation
# =============================================================================


def generate_yaml_entry(entry: StandardNamePublishEntry) -> str:
    """Generate YAML content for a single standard name entry.

    Returns a YAML string formatted to match the
    ``imas-standard-names-catalog`` convention.  All rich fields are
    included; empty/None optional fields are omitted.
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
    if entry.dd_paths:
        doc["dd_paths"] = entry.dd_paths
    if entry.constraints:
        doc["constraints"] = entry.constraints
    if entry.validity_domain:
        doc["validity_domain"] = entry.validity_domain
    if entry.cocos_transformation_type:
        doc["cocos_transformation_type"] = entry.cocos_transformation_type
    if entry.cocos is not None:
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
    entries: list[StandardNamePublishEntry],
    output_dir: Path,
) -> list[Path]:
    """Write YAML files to *output_dir*, grouped by primary tag into subdirectories.

    File names are ``{tag}/{name}.yaml`` (e.g. ``equilibrium/electron_temperature.yaml``).
    Entries without tags go into ``unscoped/``.
    Returns list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for entry in entries:
        # Group by physics_domain when available, fall back to first tag
        subdir = (
            entry.physics_domain
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
        logger.debug("Wrote %s", filepath)

    logger.info("Generated %d YAML catalog files in %s", len(written), output_dir)
    return written


# =============================================================================
# Batching
# =============================================================================


def batch_by_group(
    entries: list[StandardNamePublishEntry],
    group_by: str = "ids",
) -> dict[str, list[StandardNamePublishEntry]]:
    """Group entries into PR batches.

    Parameters
    ----------
    entries:
        Publish entries to group.
    group_by:
        Grouping strategy — ``"ids"`` groups by IDS name from provenance,
        ``"domain"`` groups by first tag, ``"confidence"`` groups by
        confidence tier.

    Returns
    -------
    dict mapping group key → entries in that group.
    """
    groups: dict[str, list[StandardNamePublishEntry]] = {}

    for entry in entries:
        if group_by == "ids":
            key = entry.provenance.ids_name or "unscoped"
        elif group_by == "domain":
            key = (
                entry.physics_domain
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
    entries: list[StandardNamePublishEntry],
    group_by: str = "ids",
) -> list[StandardNamePublishBatch]:
    """Create :class:`StandardNamePublishBatch` objects from grouped entries."""
    groups = batch_by_group(entries, group_by)
    batches: list[StandardNamePublishBatch] = []
    for key, group_entries in sorted(groups.items()):
        # Determine overall confidence tier for the batch
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


# =============================================================================
# Duplicate checking
# =============================================================================


def check_catalog_duplicates(
    entries: list[StandardNamePublishEntry],
    catalog_dir: Path | None = None,
) -> tuple[list[StandardNamePublishEntry], list[StandardNamePublishEntry]]:
    """Check for duplicates against an existing catalog directory.

    Scans ``catalog_dir`` recursively for ``.yaml`` files (including
    subdirectories created by tag-based grouping) and reads the ``name``
    field from each.  Also detects duplicates within *entries* itself.

    Returns ``(new_entries, duplicate_entries)``.
    """
    existing_names: set[str] = set()

    if catalog_dir is not None:
        catalog_path = Path(catalog_dir)
        if catalog_path.is_dir():
            # Scan both top-level and subdirectory YAML files
            for yaml_file in catalog_path.rglob("*.yaml"):
                try:
                    with open(yaml_file, encoding="utf-8") as f:
                        doc = yaml.safe_load(f)
                    if isinstance(doc, dict) and "name" in doc:
                        existing_names.add(doc["name"])
                except Exception:
                    logger.debug("Could not parse %s", yaml_file)

    new: list[StandardNamePublishEntry] = []
    duplicates: list[StandardNamePublishEntry] = []
    seen: set[str] = set()

    for entry in entries:
        if entry.name in existing_names or entry.name in seen:
            duplicates.append(entry)
        else:
            new.append(entry)
            seen.add(entry.name)

    if duplicates:
        logger.info(
            "Found %d duplicates (%d existing catalog, %d within batch)",
            len(duplicates),
            sum(1 for d in duplicates if d.name in existing_names),
            sum(1 for d in duplicates if d.name in seen),
        )

    return new, duplicates


# =============================================================================
# Graph → StandardNamePublishEntry conversion
# =============================================================================


def graph_records_to_entries(
    records: list[dict[str, Any]],
) -> list[StandardNamePublishEntry]:
    """Convert raw graph query dicts to :class:`StandardNamePublishEntry` objects.

    Handles both schema-canonical properties (``source_types``, ``source_path``,
    ``unit``) and legacy write properties (``source_type``,
    ``source_id``, ``units``).  Carries through all rich fields:
    documentation, links, dd_paths, constraints, validity_domain, kind.
    """
    entries: list[StandardNamePublishEntry] = []
    for rec in records:
        name = rec.get("name") or rec.get("id", "")
        if not name:
            continue

        # Source type (read from graph as 'source_types' list or legacy 'source')
        source_types = rec.get("source_types") or []
        source = (
            source_types[0]
            if source_types
            else (rec.get("source") or rec.get("source_type") or "dd")
        )

        # Resolve source ID (schema: source_path, legacy: source_id)
        source_id = rec.get("source_path") or rec.get("source_id") or ""

        unit = rec.get("unit")

        # IDS name from graph traversal
        ids_name = rec.get("ids_name")

        # Confidence: from node or default 1.0 for grammar-validated names
        confidence = rec.get("confidence")
        if confidence is None:
            confidence = 1.0

        description = rec.get("description") or ""

        # Rich fields
        documentation = rec.get("documentation")
        kind = rec.get("kind") or "scalar"
        links_raw = rec.get("links") or []
        dd_paths_raw = rec.get("source_paths") or []
        constraints_raw = rec.get("constraints") or []
        validity_domain = rec.get("validity_domain")
        cocos_transformation_type = rec.get("cocos_transformation_type")
        cocos = rec.get("cocos")

        # Build tags from available context
        tags: list[str] = list(rec.get("tags") or [])
        if not tags and ids_name:
            tags.append(ids_name)

        # Physics domain
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


# =============================================================================
# PR generation (stub for gh CLI)
# =============================================================================


def create_catalog_pr(
    batch: StandardNamePublishBatch,
    catalog_repo: str,
    branch_name: str,
    yaml_files: list[Path],
    dry_run: bool = False,
) -> str | None:
    """Create a PR via ``gh`` CLI.  Returns PR URL or ``None`` in dry-run.

    Parameters
    ----------
    batch:
        The publish batch (used for PR title/body).
    catalog_repo:
        GitHub repo slug (e.g. ``"iterorganization/imas-standard-names-catalog"``).
    branch_name:
        Branch to create for the PR.
    yaml_files:
        YAML files to include in the PR.
    dry_run:
        If ``True``, print what would happen without creating the PR.
    """
    # Build summary table for PR body
    lines = [
        f"## Standard Name Candidates — {batch.group_key}",
        "",
        f"Confidence tier: **{batch.confidence_tier}**",
        f"Entries: **{len(batch.entries)}**",
        "",
        "| Name | Unit | Source | Confidence |",
        "|------|------|--------|------------|",
    ]
    for entry in batch.entries:
        unit = entry.unit or "—"
        src = entry.provenance.source_id or entry.provenance.source
        conf = f"{entry.provenance.confidence:.2f}"
        lines.append(f"| `{entry.name}` | {unit} | {src} | {conf} |")
    lines.append("")
    lines.append("Generated by `imas-codex sn publish`.")

    pr_title = (
        f"feat(sn): add {len(batch.entries)} standard name candidates "
        f"({batch.group_key})"
    )
    pr_body = "\n".join(lines)

    if dry_run:
        logger.info(
            "[dry-run] Would create PR on %s:\n  branch: %s\n  title: %s\n  files: %d",
            catalog_repo,
            branch_name,
            pr_title,
            len(yaml_files),
        )
        return None

    # Attempt to create PR via gh CLI
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
                pr_body,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pr_url = result.stdout.strip()
        logger.info("Created PR: %s", pr_url)
        return pr_url
    except FileNotFoundError:
        logger.error("gh CLI not found — install GitHub CLI to create PRs")
        return None
    except subprocess.CalledProcessError as e:
        logger.error("PR creation failed: %s", e.stderr)
        return None
