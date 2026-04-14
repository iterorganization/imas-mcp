"""Catalog feedback import — read reviewed YAML entries and write to graph.

Implements the publish → review → import feedback loop for standard names.
Catalog entries are authoritative: their fields overwrite graph fields.
Graph-only fields (embedding, model, generated_at) are preserved.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Summary of a catalog import operation."""

    imported: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    catalog_commit_sha: str | None = None


@dataclass
class CheckResult:
    """Summary of a catalog-vs-graph sync check."""

    only_in_catalog: list[str] = field(default_factory=list)
    only_in_graph: list[str] = field(default_factory=list)
    diverged: list[dict[str, Any]] = field(default_factory=list)
    in_sync: int = 0
    catalog_commit_sha: str | None = None
    graph_commit_sha: str | None = None


def _resolve_catalog_sha(catalog_dir: Path) -> str | None:
    """Resolve the git HEAD SHA of the catalog directory.

    Returns the 40-character commit SHA, or None if the directory
    is not inside a git repository.
    """
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


def _parse_grammar_fields(name: str) -> dict[str, str | None]:
    """Derive grammar fields from a standard name string.

    Returns a dict with keys: physical_base, subject, component,
    coordinate, position, process. Values are strings or None.
    """
    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(name)
        return {
            "physical_base": str(parsed.physical_base)
            if parsed.physical_base
            else None,
            "subject": str(parsed.subject.value) if parsed.subject else None,
            "component": str(parsed.component.value) if parsed.component else None,
            "coordinate": str(parsed.coordinate.value) if parsed.coordinate else None,
            "position": str(parsed.position.value) if parsed.position else None,
            "process": str(parsed.process.value) if parsed.process else None,
        }
    except Exception:
        logger.debug("Grammar parse failed for name: %r", name)
        return {
            "physical_base": None,
            "subject": None,
            "component": None,
            "coordinate": None,
            "position": None,
            "process": None,
        }


def _catalog_entry_to_dict(entry: Any) -> dict[str, Any]:
    """Convert a validated catalog entry to a graph-write dict.

    Maps catalog field names to graph schema field names and derives
    grammar fields from the standard name.
    """
    # Derive grammar fields from the name
    grammar = _parse_grammar_fields(entry.name)

    # Convert tags/links to plain strings (catalog may use typed objects)
    tags = [str(t) for t in entry.tags] if entry.tags else None
    links = [str(lnk) for lnk in entry.links] if entry.links else None
    ids_paths = list(entry.ids_paths) if entry.ids_paths else None
    constraints = list(entry.constraints) if entry.constraints else None

    # Determine source_type from presence of ids_paths
    source_type = "dd" if ids_paths else "manual"

    return {
        "id": entry.name,
        "description": entry.description or None,
        "documentation": entry.documentation or None,
        "kind": str(entry.kind) if entry.kind else None,
        "unit": str(entry.unit) if entry.unit else None,
        "tags": tags or None,
        "links": links or None,
        "imas_paths": ids_paths or None,
        "validity_domain": entry.validity_domain or None,
        "constraints": constraints or None,
        "physics_domain": entry.physics_domain or None,
        "review_status": "accepted",
        "source_type": source_type,
        # Grammar fields
        "physical_base": grammar["physical_base"],
        "subject": grammar["subject"],
        "component": grammar["component"],
        "coordinate": grammar["coordinate"],
        "position": grammar["position"],
        "process": grammar["process"],
    }


def _write_catalog_entries(
    entries: list[dict[str, Any]],
    catalog_commit_sha: str | None = None,
) -> int:
    """Write catalog entries to graph with catalog-authoritative semantics.

    Catalog-owned fields are SET directly (overwrite).
    Graph-only fields (embedding, model, generated_at, etc.) are preserved
    via coalesce. Returns the number of nodes written.
    """
    if not entries:
        return 0

    from imas_codex.graph.client import GraphClient

    # Inject catalog_commit_sha into each entry for Cypher parameter access
    for e in entries:
        e["catalog_commit_sha"] = catalog_commit_sha

    with GraphClient() as gc:
        # MERGE StandardName nodes — catalog fields overwrite, graph-only preserved
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            SET sn.description = b.description,
                sn.documentation = b.documentation,
                sn.kind = b.kind,
                sn.unit = b.unit,
                sn.tags = b.tags,
                sn.links = b.links,
                sn.imas_paths = b.imas_paths,
                sn.validity_domain = b.validity_domain,
                sn.constraints = b.constraints,
                sn.physics_domain = b.physics_domain,
                sn.cocos_transformation_type = coalesce(b.cocos_transformation_type, sn.cocos_transformation_type),
                sn.review_status = 'accepted',
                sn.imported_at = datetime(),
                sn.catalog_commit_sha = b.catalog_commit_sha,
                sn.physical_base = b.physical_base,
                sn.subject = b.subject,
                sn.component = b.component,
                sn.coordinate = b.coordinate,
                sn.position = b.position,
                sn.process = b.process,
                sn.source_type = coalesce(b.source_type, sn.source_type),
                sn.created_at = coalesce(sn.created_at, datetime()),
                sn.embedding = coalesce(sn.embedding, null),
                sn.embedded_at = coalesce(sn.embedded_at, null),
                sn.model = coalesce(sn.model, null),
                sn.generated_at = coalesce(sn.generated_at, null),
                sn.confidence = coalesce(sn.confidence, null),
                sn.dd_version = coalesce(sn.dd_version, null)
            """,
            batch=entries,
        )

        # Create HAS_UNIT relationships: StandardName → Unit
        units_batch = [
            {"id": e["id"], "unit": e["unit"]} for e in entries if e.get("unit")
        ]
        if units_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MERGE (u:Unit {id: b.unit})
                MERGE (sn)-[:HAS_UNIT]->(u)
                """,
                batch=units_batch,
            )

        # Create HAS_STANDARD_NAME relationships from ids_paths
        dd_batch = []
        for e in entries:
            if e.get("imas_paths"):
                for path in e["imas_paths"]:
                    dd_batch.append({"id": e["id"], "source_id": path})

        if dd_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=dd_batch,
            )

    written = len(entries)
    logger.info("Imported %d catalog entries to graph", written)
    return written


def import_catalog(
    catalog_dir: Path,
    dry_run: bool = False,
    tag_filter: list[str] | None = None,
) -> ImportResult:
    """Import YAML catalog entries into graph as accepted StandardName nodes.

    Reads all ``*.yml`` and ``*.yaml`` files from *catalog_dir* (recursive),
    validates each entry against the ``imas-standard-names`` catalog model,
    derives grammar fields via name parsing, and MERGEs into the graph.

    Catalog fields are authoritative and overwrite graph values.
    Graph-only fields (embedding, model, generated_at) are preserved.
    Imported entries receive ``review_status='accepted'``.

    Parameters
    ----------
    catalog_dir:
        Path to directory containing YAML catalog entries.
    dry_run:
        If True, parse and validate but do not write to graph.
    tag_filter:
        If provided, only import entries whose tags overlap with this list.

    Returns
    -------
    ImportResult with counts and entry details.
    """
    import yaml
    from imas_standard_names.models import StandardNameEntry
    from pydantic import TypeAdapter

    ta = TypeAdapter(StandardNameEntry)
    # Resolve catalog commit SHA for version tracking
    catalog_sha = _resolve_catalog_sha(catalog_dir)
    if catalog_sha:
        logger.info("Catalog commit SHA: %s", catalog_sha)

    result = ImportResult(catalog_commit_sha=catalog_sha)

    # Collect YAML files
    yaml_files = sorted(
        p
        for p in catalog_dir.rglob("*")
        if p.suffix in (".yml", ".yaml") and p.is_file()
    )

    if not yaml_files:
        logger.info("No YAML files found in %s", catalog_dir)
        return result

    logger.info("Found %d YAML files in %s", len(yaml_files), catalog_dir)

    # Parse and validate entries
    prepared: list[dict[str, Any]] = []

    for yaml_path in yaml_files:
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                result.errors.append(f"{yaml_path.name}: not a YAML mapping")
                continue

            entry = ta.validate_python(data)
        except Exception as exc:
            result.errors.append(f"{yaml_path.name}: {exc}")
            logger.debug("Failed to parse %s: %s", yaml_path, exc)
            continue

        # Apply tag filter if specified
        if tag_filter:
            entry_tags = {str(t) for t in entry.tags} if entry.tags else set()
            if not entry_tags.intersection(tag_filter):
                result.skipped += 1
                continue

        # Convert to graph dict
        graph_dict = _catalog_entry_to_dict(entry, extra=data)
        prepared.append(graph_dict)
        result.entries.append(graph_dict)

    if not prepared:
        logger.info("No entries to import after filtering")
        return result

    # Write to graph (unless dry run)
    if dry_run:
        result.imported = len(prepared)
        logger.info("Dry run: would import %d entries", len(prepared))
    else:
        written = _write_catalog_entries(prepared, catalog_commit_sha=catalog_sha)
        result.imported = written
        logger.info("Imported %d entries to graph", written)

    return result


# -- Fields compared during check mode (catalog-owned, excluding grammar fields) --
_CHECK_FIELDS = (
    "description",
    "documentation",
    "kind",
    "unit",
    "tags",
    "imas_paths",
    "validity_domain",
    "constraints",
    "physics_domain",
)


def check_catalog(
    catalog_dir: Path,
    tag_filter: list[str] | None = None,
) -> CheckResult:
    """Compare catalog entries against graph without importing.

    Returns a :class:`CheckResult` describing which entries are only in
    the catalog, only in the graph, or present in both but with differing
    field values.

    Parameters
    ----------
    catalog_dir:
        Path to directory containing YAML catalog entries.
    tag_filter:
        If provided, only check entries whose tags overlap with this list.

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
    for yaml_path in yaml_files:
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            entry = ta.validate_python(data)
        except Exception:
            continue

        # Apply tag filter
        if tag_filter:
            entry_tags = {str(t) for t in entry.tags} if entry.tags else set()
            if not entry_tags.intersection(tag_filter):
                continue

        graph_dict = _catalog_entry_to_dict(entry, extra=data)
        catalog_entries[graph_dict["id"]] = graph_dict

    if not catalog_entries:
        return result

    # Fetch graph entries
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.review_status = 'accepted'
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   sn.unit AS unit,
                   sn.tags AS tags,
                   sn.imas_paths AS imas_paths,
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
    """Normalize a field value for comparison.

    Converts lists to sorted tuples, None-like values to None,
    and strings to stripped strings.
    """
    if val is None:
        return None
    if isinstance(val, list):
        return tuple(sorted(str(v) for v in val)) if val else None
    if isinstance(val, str):
        return val.strip() if val.strip() else None
    return val
