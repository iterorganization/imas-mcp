"""Catalog feedback import — read reviewed YAML entries and write to graph.

Phase 4 rewrite: diff-based origin tracking, forbidden-key rejection,
domain-from-path, grammar reuse, lock/watermark concurrency control,
and unit/COCOS validation.

Catalog entries are authoritative: their fields overwrite graph fields.
Graph-only fields (embedding, model, generated_at) are preserved.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imas_codex.standard_names.protection import PROTECTED_FIELDS

logger = logging.getLogger(__name__)

_FORBIDDEN_YAML_KEYS = frozenset({"source_paths", "dd_paths"})

#: Computed fields — re-derived from graph edges on export; silently
#: ignored on import (never written to node properties, never trigger
#: ``_protected_fields_differ``).  See plan 40 §1.
COMPUTED_FIELDS: frozenset[str] = frozenset({"arguments", "error_variants"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImportReport:
    """Summary of a catalog import operation (Phase 4)."""

    imported: int = 0
    updated: int = 0
    created: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    catalog_commit_sha: str | None = None
    pr_numbers: list[int] = field(default_factory=list)
    dry_run: bool = False
    watermark_advanced: bool = False


@dataclass
class CheckResult:
    """Summary of a catalog-vs-graph sync check."""

    only_in_catalog: list[str] = field(default_factory=list)
    only_in_graph: list[str] = field(default_factory=list)
    diverged: list[dict[str, Any]] = field(default_factory=list)
    in_sync: int = 0
    catalog_commit_sha: str | None = None
    graph_commit_sha: str | None = None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _resolve_catalog_sha(catalog_dir: Path) -> str | None:
    """Resolve the git HEAD SHA of the catalog directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(catalog_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            logger.debug("Catalog commit SHA: %s", sha)
            return sha
        logger.debug("git rev-parse failed: %s", result.stderr.strip())
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("Could not resolve catalog SHA: %s", exc)
        return None


def _is_git_repo(path: Path) -> bool:
    """Check whether *path* is inside a git work-tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Domain-from-path derivation
# ---------------------------------------------------------------------------

#: Pattern: standard_names/<domain>.yml or standard_names/<domain>.yaml
#: (domain = file basename without extension; per-domain layout from plan 40)
_DOMAIN_PATH_RE = re.compile(r"standard_names/([^/]+)\.ya?ml$")


def _derive_domain_from_path(yaml_path: Path) -> str | None:
    """Derive physics_domain from the file path convention.

    Expects ``<root>/standard_names/<domain>.yml`` (per-domain layout).
    Returns the domain string or None if the path doesn't match.
    Rejects names containing ``/``.
    """
    path_str = str(yaml_path).replace("\\", "/")
    m = _DOMAIN_PATH_RE.search(path_str)
    if m:
        domain = m.group(1)
        if "/" in domain:
            return None
        return domain
    return None


# ---------------------------------------------------------------------------
# Grammar decomposition (re-uses graph_ops helper)
# ---------------------------------------------------------------------------


def _grammar_decomposition(name: str) -> dict[str, str | None]:
    """Parse name via ISN grammar, returning grammar_* fields."""
    from imas_codex.standard_names.graph_ops import _parse_grammar_vnext

    return _parse_grammar_vnext(name)


# ---------------------------------------------------------------------------
# Entry conversion
# ---------------------------------------------------------------------------

# Graph-only fields that import must never overwrite (omitting = preserving)
_GRAPH_ONLY_PRESERVE = frozenset(
    {
        "embedding",
        "embedded_at",
        "model",
        "generated_at",
        "confidence",
        "source_types",
        "source_paths",
        "pipeline_status",
        "reviewer_score_name",
        "reviewer_score_docs",
        "reviewer_scores_name",
        "reviewer_scores_docs",
        "reviewer_comments_name",
        "reviewer_comments_docs",
        "reviewed_name_at",
        "reviewed_docs_at",
        "review_tier",
        "review_input_hash",
        "vocab_gap_detail",
        "validation_issues",
        "validation_layer_summary",
        "validation_status",
        "link_status",
    }
)


def _entry_to_graph_dict(
    entry: Any,
    *,
    physics_domain: list[str],
) -> dict[str, Any]:
    """Convert a validated ISN entry to a graph-write dict.

    Does NOT include graph-only fields (source_paths, etc.).
    Grammar fields are derived from the entry name.
    """
    grammar = _grammar_decomposition(entry.name)

    links = [str(lnk) for lnk in entry.links] if entry.links else []
    constraints = list(entry.constraints) if entry.constraints else []

    result: dict[str, Any] = {
        "id": entry.name,
        "description": entry.description or None,
        "documentation": entry.documentation or None,
        "kind": str(entry.kind) if hasattr(entry, "kind") and entry.kind else None,
        "unit": str(entry.unit) if hasattr(entry, "unit") and entry.unit else None,
        "links": links or None,
        "validity_domain": entry.validity_domain or None,
        "constraints": constraints or None,
        "physics_domain": physics_domain,
        "status": str(entry.status) if entry.status else "draft",
        "deprecates": str(entry.deprecates) if entry.deprecates else None,
        "superseded_by": str(entry.superseded_by) if entry.superseded_by else None,
        "cocos_transformation_type": (
            str(entry.cocos_transformation_type)
            if entry.cocos_transformation_type
            else None
        ),
    }

    # Merge grammar fields
    result.update(grammar)

    return result


# ---------------------------------------------------------------------------
# Graph read: current state for diff
# ---------------------------------------------------------------------------


def _fetch_graph_state(
    gc: Any,
    name_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch current StandardName properties for diff comparison."""
    if not name_ids:
        return {}

    props = sorted(PROTECTED_FIELDS | {"origin"})
    return_clause = ", ".join(f"sn.{p} AS {p}" for p in props)

    rows = gc.query(
        f"""
        UNWIND $ids AS id
        OPTIONAL MATCH (sn:StandardName {{id: id}})
        RETURN sn.id AS id, {return_clause}
        """,
        ids=name_ids,
    )

    result: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if row.get("id"):
            result[row["id"]] = dict(row)
    return result


def _protected_fields_differ(
    graph_state: dict[str, Any],
    new_entry: dict[str, Any],
) -> bool:
    """Check whether any PROTECTED_FIELDS differ between graph and new entry."""
    for field_name in PROTECTED_FIELDS:
        old_val = graph_state.get(field_name)
        new_val = new_entry.get(field_name)
        # Normalise None-like empties
        if not old_val:
            old_val = None
        if not new_val:
            new_val = None
        if old_val != new_val:
            return True
    return False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_unit_against_graph(
    gc: Any,
    entries: list[dict[str, Any]],
) -> list[str]:
    """Check for unit mismatches between catalog entries and graph."""
    unit_batch = [{"id": e["id"], "unit": e["unit"]} for e in entries if e.get("unit")]
    if not unit_batch:
        return []

    rows = gc.query(
        """
        UNWIND $batch AS b
        MATCH (sn:StandardName {id: b.id})
        WHERE sn.unit IS NOT NULL AND b.unit IS NOT NULL
          AND sn.unit <> b.unit
        RETURN sn.id AS name,
               sn.unit AS existing_unit,
               b.unit AS incoming_unit
        """,
        batch=unit_batch,
    )

    errors = []
    for row in rows or []:
        errors.append(
            f"Unit mismatch for {row['name']}: "
            f"graph has '{row['existing_unit']}', "
            f"catalog has '{row['incoming_unit']}'. "
            f"Use --accept-unit-override to force."
        )
    return errors


def _validate_cocos_against_graph(
    gc: Any,
    entries: list[dict[str, Any]],
) -> list[str]:
    """Check for COCOS transformation type mismatches."""
    cocos_batch = [
        {"id": e["id"], "cocos": e["cocos_transformation_type"]}
        for e in entries
        if e.get("cocos_transformation_type")
    ]
    if not cocos_batch:
        return []

    rows = gc.query(
        """
        UNWIND $batch AS b
        MATCH (sn:StandardName {id: b.id})
        WHERE sn.cocos_transformation_type IS NOT NULL
          AND b.cocos IS NOT NULL
          AND sn.cocos_transformation_type <> b.cocos
        RETURN sn.id AS name,
               sn.cocos_transformation_type AS existing,
               b.cocos AS incoming
        """,
        batch=cocos_batch,
    )

    errors = []
    for row in rows or []:
        errors.append(
            f"COCOS mismatch for {row['name']}: "
            f"graph has '{row['existing']}', "
            f"catalog has '{row['incoming']}'. "
            f"Use --accept-cocos-override to force."
        )
    return errors


# ---------------------------------------------------------------------------
# Graph write: import entries
# ---------------------------------------------------------------------------


def _write_import_entries(
    gc: Any,
    entries: list[dict[str, Any]],
    *,
    catalog_commit_sha: str | None = None,
    pr_number: int | None = None,
    pr_url: str | None = None,
) -> int:
    """Write catalog entries to graph with catalog-authoritative semantics.

    Catalog-owned fields are SET directly (overwrite).
    Graph-only fields (embedding, model, generated_at, etc.) are preserved
    via omission (not in the SET clause at all).
    Returns the number of nodes written.
    """
    if not entries:
        return 0

    # Add provenance metadata to each entry
    for e in entries:
        e["_catalog_commit_sha"] = catalog_commit_sha
        e["_pr_number"] = pr_number
        e["_pr_url"] = pr_url
        e["_origin"] = e.pop("_origin", "catalog_edit")

    # Main MERGE — catalog-owned fields overwrite
    gc.query(
        """
        UNWIND $batch AS b
        MERGE (sn:StandardName {id: b.id})
        SET sn.description = b.description,
            sn.documentation = b.documentation,
            sn.kind = b.kind,
            sn.unit = b.unit,
            sn.links = b.links,
            sn.validity_domain = b.validity_domain,
            sn.constraints = b.constraints,
            sn.physics_domain = b.physics_domain,
            sn.status = b.status,
            sn.deprecates = b.deprecates,
            sn.superseded_by = b.superseded_by,
            sn.cocos_transformation_type = coalesce(b.cocos_transformation_type, sn.cocos_transformation_type),
            sn.pipeline_status = 'accepted',
            sn.origin = b._origin,
            sn.imported_at = datetime(),
            sn.catalog_commit_sha = b._catalog_commit_sha,
            sn.import_pr_number = b._pr_number,
            sn.import_pr_url = b._pr_url,
            sn.source_types = coalesce(sn.source_types, ['catalog']),
            sn.created_at = coalesce(sn.created_at, datetime())
        """,
        batch=entries,
    )

    # Grammar fields (separate SET to keep queries readable)
    grammar_batch = [
        {k: v for k, v in e.items() if k == "id" or k.startswith("grammar_")}
        for e in entries
    ]
    gc.query(
        """
        UNWIND $batch AS b
        MATCH (sn:StandardName {id: b.id})
        SET sn.grammar_component = b.grammar_component,
            sn.grammar_coordinate = b.grammar_coordinate,
            sn.grammar_subject = b.grammar_subject,
            sn.grammar_physical_base = b.grammar_physical_base,
            sn.grammar_geometric_base = b.grammar_geometric_base,
            sn.grammar_process = b.grammar_process,
            sn.grammar_transformation = b.grammar_transformation,
            sn.grammar_object = b.grammar_object,
            sn.grammar_geometry = b.grammar_geometry,
            sn.grammar_position = b.grammar_position,
            sn.grammar_device = b.grammar_device,
            sn.grammar_secondary_base = b.grammar_secondary_base,
            sn.grammar_binary_operator = b.grammar_binary_operator
        """,
        batch=grammar_batch,
    )

    # Create HAS_UNIT relationships
    units_batch = [{"id": e["id"], "unit": e["unit"]} for e in entries if e.get("unit")]
    if units_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            MERGE (u:Unit {id: b.unit})
            SET u.symbol = coalesce(u.symbol, b.unit)
            MERGE (sn)-[:HAS_UNIT]->(u)
            """,
            batch=units_batch,
        )

    # Emit structural edges: HAS_ARGUMENT, HAS_ERROR, HAS_PREDECESSOR,
    # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN.
    # Tail pass — all nodes in the batch exist before edges are written.
    # 'deprecates' → HAS_PREDECESSOR, 'superseded_by' → HAS_SUCCESSOR.
    from imas_codex.standard_names.graph_ops import _write_standard_name_edges

    _write_standard_name_edges(gc, entries)

    written = len(entries)
    logger.info("Imported %d catalog entries to graph", written)
    return written


# Keep legacy name as alias
_write_catalog_entries = _write_import_entries


# ---------------------------------------------------------------------------
# Main entry point: run_import
# ---------------------------------------------------------------------------


def run_import(
    catalog_dir: Path,
    dry_run: bool = False,
    accept_unit_override: bool = False,
    accept_cocos_override: bool = False,
) -> ImportReport:
    """Import YAML catalog entries into graph as accepted StandardName nodes.

    Phase 4 implementation with:
    - Forbidden-key rejection (source_paths, dd_paths)
    - Domain-from-path derivation
    - Grammar decomposition via graph_ops helper
    - Diff-based origin tracking
    - Lock and watermark concurrency control
    - Unit and COCOS validation

    Parameters
    ----------
    catalog_dir:
        Path to ISN catalog repository root (containing ``standard_names/`` subtree).
    dry_run:
        If True, parse and validate but do not write to graph.

    Returns
    -------
    ImportReport with counts and entry details.
    """
    import yaml
    from imas_standard_names.models import StandardNameEntry
    from pydantic import TypeAdapter

    ta = TypeAdapter(StandardNameEntry)
    report = ImportReport(dry_run=dry_run)

    isnc_path = catalog_dir

    # Resolve HEAD SHA
    catalog_sha = _resolve_catalog_sha(isnc_path)
    report.catalog_commit_sha = catalog_sha

    has_git = _is_git_repo(isnc_path)

    # ── Phase 1: Parse and validate all YAML files ──────────────────────

    sn_dir = isnc_path / "standard_names"
    if not sn_dir.is_dir():
        # Fall back to searching from root (flat layout)
        sn_dir = isnc_path

    yaml_files = sorted(
        p for p in sn_dir.rglob("*") if p.suffix in (".yml", ".yaml") and p.is_file()
    )

    if not yaml_files:
        logger.info("No YAML files found in %s", sn_dir)
        return report

    logger.info("Found %d YAML files in %s", len(yaml_files), sn_dir)

    prepared: list[dict[str, Any]] = []

    for yaml_path in yaml_files:
        relative = (
            yaml_path.relative_to(isnc_path)
            if isnc_path in yaml_path.parents
            else yaml_path
        )
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict):
                # Legacy per-file layout — reject with migration error
                report.errors.append(
                    f"{relative}: top-level YAML dict detected. "
                    f"Per-file layout is no longer supported; migrate to "
                    f"per-domain list layout (one YAML sequence per "
                    f"physics domain). See plan 40."
                )
                continue

            if not isinstance(data, list):
                report.errors.append(f"{relative}: expected a YAML list of entries")
                continue

            # Derive domain from file path (per-domain layout)
            path_domain = _derive_domain_from_path(yaml_path)
            if path_domain is None:
                report.errors.append(
                    f"{relative}: cannot derive physics_domain from file path. "
                    f"Expected standard_names/<domain>.yml layout."
                )
                continue

            for entry_data in data:
                if not isinstance(entry_data, dict):
                    report.errors.append(
                        f"{relative}: entry is not a mapping: {entry_data!r}"
                    )
                    continue

                # Silently ignore computed fields (plan 40 §1)
                for cf in COMPUTED_FIELDS:
                    if cf in entry_data:
                        logger.info(
                            "Ignoring computed field=%s name=%s",
                            cf,
                            entry_data.get("name", "?"),
                        )
                        entry_data = {
                            k: v
                            for k, v in entry_data.items()
                            if k not in COMPUTED_FIELDS
                        }
                        break

                # Reject forbidden keys (source_paths, dd_paths)
                forbidden_found = _FORBIDDEN_YAML_KEYS & set(entry_data.keys())
                if forbidden_found:
                    report.errors.append(
                        f"{relative}/{entry_data.get('name', '?')}: "
                        f"contains forbidden key(s) "
                        f"{sorted(forbidden_found)}. Provenance is graph-only; "
                        f"use 'sn run' for provenance management."
                    )
                    continue

                # Validate against ISN model
                entry = ta.validate_python(entry_data)

                # Parse physics_domain — must be a list (no bare-string fallback)
                raw_pd = entry_data.get("physics_domain")
                if raw_pd is not None and not isinstance(raw_pd, list):
                    report.errors.append(
                        f"{relative}/{entry_data.get('name', '?')}: "
                        f"physics_domain must be a list, got {type(raw_pd).__name__}. "
                        f"Catalog will be rebuilt fresh with list form."
                    )
                    continue
                physics_domain_list = raw_pd if raw_pd else [path_domain]

                # Grammar decomposition — hard fail on parse error
                grammar = _grammar_decomposition(entry.name)

                # Convert to graph dict
                graph_dict = _entry_to_graph_dict(
                    entry, physics_domain=physics_domain_list
                )
                graph_dict.update(grammar)
                prepared.append(graph_dict)
                report.entries.append(graph_dict)

        except Exception as exc:
            report.errors.append(f"{relative}: {exc}")
            logger.debug("Failed to parse %s: %s", yaml_path, exc)
            continue

    if report.errors:
        logger.warning("Encountered %d error(s) during parsing", len(report.errors))

    if not prepared:
        logger.info("No entries to import after validation")
        return report

    if dry_run:
        report.imported = len(prepared)
        report.skipped = 0
        logger.info("Dry run: would import %d entries", len(prepared))
        return report

    # ── Phase 2: Lock, diff, validate, write ────────────────────────────

    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.import_sync import (
        acquire_import_lock,
        advance_watermark,
        release_import_lock,
    )

    with GraphClient() as gc:
        # Acquire lock
        if not acquire_import_lock(gc):
            report.errors.append(
                "Could not acquire import lock — another import may be running."
            )
            return report

        try:
            # 4e: Diff-based origin tracking
            name_ids = [e["id"] for e in prepared]
            graph_state = _fetch_graph_state(gc, name_ids)

            for entry_dict in prepared:
                eid = entry_dict["id"]
                existing = graph_state.get(eid)

                if existing is None:
                    # New entry — mark as catalog_edit origin
                    entry_dict["_origin"] = "catalog_edit"
                    report.created += 1
                elif _protected_fields_differ(existing, entry_dict):
                    # Edited entry — flip origin
                    entry_dict["_origin"] = "catalog_edit"
                    report.updated += 1
                else:
                    # No protected-field changes — preserve current origin
                    entry_dict["_origin"] = existing.get("origin") or "pipeline"
                    report.skipped += 1

            # Count only created + updated as imported
            to_write = [
                e
                for e in prepared
                if e.get("_origin") == "catalog_edit"
                or graph_state.get(e["id"]) is None
            ]
            # Actually write ALL entries (even no-ops refresh timestamps)
            to_write = prepared

            # 4f: Unit validation
            if not accept_unit_override:
                unit_errors = _validate_unit_against_graph(gc, to_write)
                if unit_errors:
                    report.errors.extend(unit_errors)

            # 4g: COCOS validation
            if not accept_cocos_override:
                cocos_errors = _validate_cocos_against_graph(gc, to_write)
                if cocos_errors:
                    report.errors.extend(cocos_errors)

            # Write entries
            written = _write_import_entries(
                gc,
                to_write,
                catalog_commit_sha=catalog_sha,
            )
            report.imported = written

            # 4h: Advance watermark
            if catalog_sha and has_git:
                try:
                    advance_watermark(gc, catalog_sha)
                    report.watermark_advanced = True
                except Exception as exc:
                    report.errors.append(f"Failed to advance watermark: {exc}")
                    logger.warning("Watermark advance failed: %s", exc)

        finally:
            release_import_lock(gc)

    return report


# ---------------------------------------------------------------------------
# Check mode (catalog-vs-graph comparison)
# ---------------------------------------------------------------------------

_CHECK_FIELDS = (
    "description",
    "documentation",
    "kind",
    "unit",
    "validity_domain",
    "constraints",
    "physics_domain",
)

# Fields that are graph-only extensions (not in the ISN catalog model)
# but may appear in YAML files. Strip before model validation.
_GRAPH_ONLY_FIELDS = {"dd_paths", "physics_domain", "cocos_transformation_type"}


def check_catalog(
    catalog_dir: Path,
) -> CheckResult:
    """Compare catalog entries against graph without importing.

    Returns a :class:`CheckResult` describing which entries are only in
    the catalog, only in the graph, or present in both but with differing
    field values.

    Parameters
    ----------
    catalog_dir:
        Path to directory containing YAML catalog entries.

    Returns
    -------
    CheckResult with sync status details.
    """
    import yaml
    from imas_standard_names.models import StandardNameEntry
    from pydantic import TypeAdapter

    from imas_codex.graph.client import GraphClient

    ta = TypeAdapter(StandardNameEntry)
    catalog_sha = _resolve_catalog_sha(catalog_dir)
    result = CheckResult(catalog_commit_sha=catalog_sha)

    # Parse catalog entries
    yaml_files = sorted(
        p
        for p in catalog_dir.rglob("*")
        if p.suffix in (".yml", ".yaml") and p.is_file()
    )

    catalog_entries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for yaml_path in yaml_files:
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict):
                # Legacy per-file layout — skip with warning
                continue

            if not isinstance(data, list):
                continue

            path_domain = _derive_domain_from_path(yaml_path) or "unscoped"

            for entry_data in data:
                if not isinstance(entry_data, dict):
                    continue

                # Warn about computed-field edits (§1 curator warning)
                for cf in COMPUTED_FIELDS:
                    if cf in entry_data:
                        warnings.append(
                            f"{cf} is computed from HAS_ARGUMENT / HAS_ERROR "
                            f"graph edges and will be overwritten on next "
                            f"export — edit has no effect.  See plan 40 / "
                            f"COMPUTED_FIELDS."
                        )
                        logger.warning(
                            "%s is computed from HAS_ARGUMENT / HAS_ERROR "
                            "graph edges and will be overwritten on next "
                            "export — edit has no effect.  See plan 40 / "
                            "COMPUTED_FIELDS.",
                            cf,
                        )

                # Strip computed + graph-only fields before ISN model validation
                model_data = {
                    k: v
                    for k, v in entry_data.items()
                    if k not in _GRAPH_ONLY_FIELDS and k not in COMPUTED_FIELDS
                }
                entry = ta.validate_python(model_data)

                raw_pd = entry_data.get("physics_domain")
                pd_list = raw_pd if isinstance(raw_pd, list) else [path_domain]
                graph_dict = _entry_to_graph_dict(entry, physics_domain=pd_list)
                catalog_entries[graph_dict["id"]] = graph_dict

        except Exception:
            continue

    if not catalog_entries:
        return result

    # Fetch graph entries
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.pipeline_status = 'accepted'
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   sn.unit AS unit,
                   sn.source_paths AS source_paths,
                   sn.validity_domain AS validity_domain,
                   sn.constraints AS constraints,
                   sn.physics_domain AS physics_domain,
                   sn.catalog_commit_sha AS catalog_commit_sha
            """
        )

    graph_entries: dict[str, dict[str, Any]] = {}
    graph_sha: str | None = None
    for row in rows:
        graph_entries[row["id"]] = dict(row)
        if row.get("catalog_commit_sha") and not graph_sha:
            graph_sha = row["catalog_commit_sha"]

    result.graph_commit_sha = graph_sha

    # Compare
    catalog_names = set(catalog_entries.keys())
    graph_names = set(graph_entries.keys())

    result.only_in_catalog = sorted(catalog_names - graph_names)
    result.only_in_graph = sorted(graph_names - catalog_names)

    for name in sorted(catalog_names & graph_names):
        cat = catalog_entries[name]
        graph = graph_entries[name]

        diffs: dict[str, Any] = {}
        for fld in _CHECK_FIELDS:
            cat_val = _normalize_field(cat.get(fld))
            graph_val = _normalize_field(graph.get(fld))
            if cat_val != graph_val:
                diffs[fld] = {"catalog": cat_val, "graph": graph_val}

        if diffs:
            result.diverged.append({"name": name, "fields": diffs})
        else:
            result.in_sync += 1

    return result


def _normalize_field(val: Any) -> Any:
    """Normalize a field value for comparison."""
    if val is None:
        return None
    if isinstance(val, list):
        return tuple(sorted(str(v) for v in val)) if val else None
    if isinstance(val, str):
        return val.strip() if val.strip() else None
    return val
