"""DD source: extract standard name candidates from IMAS Data Dictionary paths.

Surfaces rich graph context for each path: authoritative unit from HAS_UNIT,
LLM-enriched descriptions, cluster siblings, coordinates, parent structure,
and keywords. Passes paths through enrichment layer for classification,
primary cluster selection, and global (cluster × unit) batching.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
from imas_codex.standard_names.sources.base import ExtractionBatch
from imas_codex.standard_names.unit_overrides import resolve_unit

logger = logging.getLogger(__name__)


def _is_unparseable_dd_unit(unit: str) -> bool:
    """Return True if *unit* is a DD-corrupted string that cannot be parsed.

    Catches three failure modes that escape the explicit rule-based override
    engine in ``unit_overrides.yaml``:

    1. Unit strings containing whitespace (e.g. ``"kg m"``).
    2. Unit strings with non-numeric exponents (e.g. ``"m^dimension"``).
    3. Unit tokens not in the SI/pint grammar (e.g. ``"as_parent_level_2"``).

    Uses :func:`imas_codex.units.normalize_unit_symbol` — the same pint-based
    parser backing the rest of the pipeline — so no logic is re-implemented.
    """
    if not unit or not unit.strip():
        return True
    unit = unit.strip()
    # Dimensionless sentinels are valid
    if unit in ("1", "dimensionless", "-", "mixed", "none"):
        return False
    # C1: whitespace in unit string
    if re.search(r"\s", unit):
        return True
    # C2: non-numeric exponents (e.g. m^dimension)
    if re.search(r"\^[a-zA-Z]", unit):
        return True
    # C3: attempt pint parse — catches everything else
    from imas_codex.units import normalize_unit_symbol

    return normalize_unit_symbol(unit) is None


def _apply_unit_overrides(
    results: list[dict],
    *,
    source_type: str = "dd",
    write_skipped: bool = True,
) -> list[dict]:
    """Apply DD unit overrides/skips in-place to an extraction result set.

    For each row: resolves ``(path, unit)`` through the override engine.
    - Override match: updates ``row['unit']`` to the corrected value and
      records provenance on ``row['_unit_override']``.
    - Skip match: removes the row from the returned list and queues a
      ``StandardNameSource`` skip record.

    After rule-based resolution, a catch-all check validates the effective
    unit via :func:`_is_unparseable_dd_unit`. Units that fail are skipped
    with ``skip_reason='dd_unit_unresolvable'`` — preventing downstream
    Pydantic validation errors from quarantining the resulting StandardName.

    When ``write_skipped`` is True and there is at least one skipped row,
    the skip records are persisted via ``graph_ops.write_skipped_sources``.

    Returns the filtered list of kept rows (skipped ones removed).
    """
    kept: list[dict] = []
    skip_records: list[dict] = []

    for row in results:
        path = row.get("path") or ""
        unit = row.get("unit")
        effective_unit, meta = resolve_unit(path, unit)

        if meta and meta.get("rule") == "skip":
            skip_records.append(
                {
                    "source_type": source_type,
                    "source_id": path,
                    "skip_reason": meta["skip_reason"],
                    "skip_reason_detail": meta["skip_reason_detail"],
                    "description": row.get("description") or "",
                }
            )
            continue

        if meta and meta.get("rule") == "override":
            row["unit"] = effective_unit
            row["_unit_override"] = meta

        # Catch-all: if the (possibly overridden) unit is still unparseable,
        # skip it upstream so it never reaches LLM composition or Pydantic
        # validation.  This guards against DD-corrupted unit strings like
        # 'as_parent_level_2', whitespace-containing units, and non-numeric
        # exponents that are not covered by explicit rules.
        final_unit = (
            row.get("unit") if meta and meta.get("rule") == "override" else unit
        )
        if final_unit and _is_unparseable_dd_unit(final_unit):
            skip_records.append(
                {
                    "source_type": source_type,
                    "source_id": path,
                    "skip_reason": "dd_unit_unresolvable",
                    "skip_reason_detail": (
                        f"Unit '{final_unit}' is not a valid SI expression "
                        f"(failed pint/pre-check parse)"
                    ),
                    "description": row.get("description") or "",
                }
            )
            continue

        kept.append(row)

    if write_skipped and skip_records:
        try:
            from imas_codex.standard_names.graph_ops import write_skipped_sources

            written = write_skipped_sources(skip_records)
            logger.info(
                "Recorded %d skipped DD sources (unit override config)", written
            )
        except Exception as exc:  # pragma: no cover
            # Graph failures shouldn't block extraction itself.
            logger.warning(
                "Failed to write skipped DD sources to graph: %s (%d records)",
                exc,
                len(skip_records),
            )

    return kept


# Path segments indicating "configurable meaning" blocks where the concrete
# physical quantity depends on runtime identifier selection (e.g., a generic
# slot inside a /process/ array-of-structures that could hold radiation,
# transport, or source terms depending on the identifier). These are poor
# Standard Name candidates because the semantic label is set by sibling
# identifier.index, not by the DD path itself. Skipped as 'configurable_meaning'.
_CONFIGURABLE_PATH_SEGMENTS = ("/process/",)


def _apply_skip_by_design(
    results: list[dict],
    *,
    source_type: str = "dd",
    write_skipped: bool = True,
) -> list[dict]:
    """Filter out paths that are semantically indeterminate by design.

    Counterpart to :func:`_apply_unit_overrides` but for path-structure
    based skips (not unit-config based). Writes StandardNameSource rows
    with ``status='skipped'`` and ``skip_reason='configurable_meaning'``
    so the audit layer can distinguish "we chose not to name" from
    "we tried and failed".
    """
    kept: list[dict] = []
    skip_records: list[dict] = []

    for row in results:
        path = row.get("path") or ""
        if any(seg in path for seg in _CONFIGURABLE_PATH_SEGMENTS):
            skip_records.append(
                {
                    "source_type": source_type,
                    "source_id": path,
                    "skip_reason": "configurable_meaning",
                    "skip_reason_detail": (
                        "Path inside a /process/ array-of-structures; "
                        "concrete quantity is determined by sibling "
                        "identifier.index at runtime, not by the DD path."
                    ),
                    "description": row.get("description") or "",
                }
            )
            continue
        kept.append(row)

    if write_skipped and skip_records:
        try:
            from imas_codex.standard_names.graph_ops import write_skipped_sources

            written = write_skipped_sources(skip_records)
            logger.info(
                "Recorded %d skip-by-design DD sources (configurable_meaning)",
                written,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to write skip-by-design DD sources to graph: %s (%d records)",
                exc,
                len(skip_records),
            )

    return kept


def _apply_extract_deny(
    results: list[dict],
    *,
    source_type: str = "dd",
    write_skipped: bool = True,
) -> list[dict]:
    """Filter out paths matched by the extract-deny YAML config.

    Counterpart to :func:`_apply_skip_by_design` but driven by the
    ``config/extract_deny.yaml`` rule set. Catches paths that pass the
    ``SN_SOURCE_CATEGORIES`` gate (correct ``node_category``) but are
    poor standard-name candidates: boolean constraint selectors,
    engineering coil geometry, control-system parameters, etc.

    Writes ``StandardNameSource`` rows with ``status='skipped'`` so the
    audit layer can account for denied paths.

    Returns the filtered list of kept rows (denied ones removed).
    """
    from imas_codex.standard_names.extract_deny import match_deny_rule

    kept: list[dict] = []
    skip_records: list[dict] = []

    for row in results:
        path = row.get("path") or ""
        # Build node_attrs for attribute-predicate rules
        node_attrs = {
            "data_type": row.get("data_type"),
            "units": row.get("unit") or row.get("unit_from_rel"),
            "documentation": row.get("documentation"),
        }
        rule = match_deny_rule(path, node_attrs=node_attrs)
        if rule is not None:
            skip_records.append(
                {
                    "source_type": source_type,
                    "source_id": path,
                    "skip_reason": rule.skip_reason,
                    "skip_reason_detail": rule.reason,
                    "description": row.get("description") or "",
                    "status": rule.status,
                }
            )
            continue
        kept.append(row)

    if skip_records:
        logger.info(
            "Extract deny gate filtered %d paths (%d kept)",
            len(skip_records),
            len(kept),
        )

    if write_skipped and skip_records:
        try:
            from imas_codex.standard_names.graph_ops import write_skipped_sources

            written = write_skipped_sources(skip_records)
            logger.info("Recorded %d extract-deny DD sources", written)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to write extract-deny DD sources to graph: %s (%d records)",
                exc,
                len(skip_records),
            )

    return kept


# Enriched extraction query — single Cypher surfacing all context.
# LIMIT is applied on DISTINCT (n, ids) pairs first, then clusters/coords
# are joined.  This guarantees $limit unique paths regardless of how many
# cluster memberships each path has.
_ENRICHED_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE {where_clause}
WITH DISTINCT n, ids
ORDER BY ids.id, n.id
LIMIT $limit
WITH n, ids
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(coord:IMASNode)
OPTIONAL MATCH (coord)-[:HAS_UNIT]->(cu:Unit)
OPTIONAL MATCH (n)-[:HAS_ERROR]->(err:IMASNode)
WITH n, ids, u, c, parent, coord, cu, collect(DISTINCT err.id) AS error_node_ids
RETURN n.id AS path,
       n.description AS description,
       n.documentation AS documentation,
       n.unit AS unit,
       u.id AS unit_from_rel,
       n.data_type AS data_type,
       n.node_type AS node_type,
       n.physics_domain AS physics_domain,
       n.keywords AS keywords,
       n.node_category AS node_category,
       n.ndim AS ndim,
       n.lifecycle_status AS lifecycle_status,
       ids.id AS ids_name,
       c.label AS cluster_label,
       c.id AS cluster_id,
       c.description AS cluster_description,
       c.scope AS cluster_scope,
       parent.id AS parent_path,
       parent.description AS parent_description,
       parent.data_type AS parent_type,
       coord.id AS coord_path,
       coord.description AS coord_description,
       cu.id AS coord_unit,
       n.cocos_transformation_type AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
       error_node_ids
ORDER BY ids.id, n.id
"""

# Cluster siblings query — paths sharing the same cluster
_SIBLINGS_QUERY = """
MATCH (c:IMASSemanticCluster)<-[:IN_CLUSTER]-(sibling:IMASNode)
WHERE c.id IN $cluster_ids AND NOT (sibling.id IN $exclude_paths)
WITH c.id AS cluster_id, sibling
OPTIONAL MATCH (sibling)-[:HAS_UNIT]->(su:Unit)
RETURN cluster_id, sibling.id AS sibling_path,
       su.id AS sibling_unit, sibling.description AS sibling_desc
ORDER BY cluster_id, sibling_path
"""

# Breakdown diagnostic queries — grouped counts and random samples for non-dynamic
# node_type buckets (i.e., newly-admitted paths after node_type filter removal).
# Both queries enforce the leaf invariant (data_type NOT IN STRUCTURE/STRUCT_ARRAY)
# so breakdown totals reflect exactly what extraction admits.
_BREAKDOWN_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
RETURN n.node_type AS node_type, n.node_category AS node_category,
       n.data_type AS data_type, count(*) AS cnt
ORDER BY node_type, node_category, data_type
"""

_BREAKDOWN_SAMPLES_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
  AND (n.node_type IS NULL OR NOT n.node_type IN ['dynamic'])
RETURN n.node_type AS node_type, n.node_category AS node_category, n.id AS path
ORDER BY rand()
LIMIT $sample_limit
"""

# Count candidates that have at least one HAS_ERROR sibling (B9 prep).
_BREAKDOWN_ERRORS_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
  AND EXISTS { (n)-[:HAS_ERROR]->(:IMASNode) }
RETURN count(n) AS has_errors_count
"""


def report_extract_breakdown(facility_ids: list[str] | None = None) -> dict:
    """Return a diagnostic breakdown of extractable DD paths by node_type and
    node_category.

    Queries the graph for all nodes matching the extraction filter
    (``node_category IN SN_SOURCE_CATEGORIES`` AND leaf invariant
    ``data_type NOT IN STRUCTURE/STRUCT_ARRAY``) and returns counts grouped by
    ``(node_type, node_category, data_type)`` plus up to 5 random sample paths
    per non-dynamic bucket and the count of candidates with HAS_ERROR siblings.

    Args:
        facility_ids: Reserved for future filtering; not currently used.
            DD nodes are not facility-scoped.

    Returns:
        Dict with keys:

        - **total** (*int*): Total qualifying nodes.
        - **by_node_type** (*dict*): Count per ``node_type`` value (``None`` key
          for nodes where ``node_type`` is not set).
        - **by_category** (*dict*): Count per ``node_category`` value.
        - **by_data_type** (*dict*): Count per ``data_type`` value — proves no
          STRUCTURE/STRUCT_ARRAY containers are admitted.
        - **has_errors_count** (*int*): Number of candidates with at least one
          HAS_ERROR sibling (B9 prep — error-sibling minting).
        - **samples** (*dict*): Up to 5 random path IDs per
          ``"{node_type}/{node_category}"`` bucket for non-dynamic node_types.
    """
    from imas_codex.graph.client import GraphClient

    params = {"sn_categories": list(SN_SOURCE_CATEGORIES)}

    with GraphClient() as gc:
        count_rows = list(gc.query(_BREAKDOWN_QUERY, **params))
        sample_rows = list(
            gc.query(_BREAKDOWN_SAMPLES_QUERY, sample_limit=50, **params)
        )
        errors_rows = list(gc.query(_BREAKDOWN_ERRORS_QUERY, **params))

    total = 0
    by_node_type: dict = {}
    by_category: dict = {}
    by_data_type: dict = {}

    for row in count_rows:
        nt = row["node_type"]
        nc = row["node_category"]
        dt = row["data_type"]
        cnt = row["cnt"]
        total += cnt
        by_node_type[nt] = by_node_type.get(nt, 0) + cnt
        by_category[nc] = by_category.get(nc, 0) + cnt
        by_data_type[dt] = by_data_type.get(dt, 0) + cnt

    has_errors_count = errors_rows[0]["has_errors_count"] if errors_rows else 0

    # Collect up to 5 sample paths per (node_type, node_category) bucket
    # for newly-admitted (non-dynamic) node_types.
    samples: dict[str, list[str]] = {}
    for row in sample_rows:
        nt = row["node_type"] or "none"
        nc = row["node_category"]
        key = f"{nt}/{nc}"
        bucket = samples.setdefault(key, [])
        if len(bucket) < 5:
            bucket.append(row["path"])

    return {
        "total": total,
        "by_node_type": by_node_type,
        "by_category": by_category,
        "by_data_type": by_data_type,
        "has_errors_count": has_errors_count,
        "samples": samples,
    }


def extract_dd_candidates(
    *,
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
    existing_names: set[str] | None = None,
    on_status: Callable[[str], None] | None = None,
    from_model: str | None = None,
    force: bool = False,
    name_only: bool = False,
    name_only_batch_size: int = 50,
    max_batch_size: int = 25,
    max_tokens: int | None = None,
) -> list[ExtractionBatch]:
    """Extract candidate quantities from IMAS DD paths with enriched context.

    Queries IMASNode paths from the graph with full context: authoritative unit,
    cluster siblings, coordinates, parent structure, and LLM-enriched descriptions.
    Classifies paths (quantity vs metadata vs skip), selects primary cluster per
    path, and groups **globally** by (cluster × unit) for unit-safe batching.

    Args:
        ids_filter: Restrict to specific IDS (e.g., "equilibrium")
        domain_filter: Restrict to physics domain
        limit: Max paths to extract
        existing_names: Known standard names for dedup awareness
        on_status: Optional callback ``(text: str) -> None`` for progress updates
        from_model: Only return paths whose existing StandardName was generated
            by a model containing this substring
        force: When False, exclude paths that already have a non-stale/non-failed
            StandardNameSource node (skip already-processed).
        name_only: When True, use the Workstream 2a coarser grouping
            (``physics_domain × unit``) via :func:`group_for_name_only`
            instead of :func:`group_by_concept_and_unit`. Batches are
            tagged ``mode="names"`` so the compose worker picks the
            lean user prompt.
        name_only_batch_size: Maximum items per batch when ``name_only``
            is True.
        max_batch_size: Maximum items per batch in full compose mode.
        max_tokens: When set, apply a pre-flight token check that
            binary-splits any batch exceeding this estimated token count.

    Returns:
        List of ExtractionBatch objects grouped by (primary_cluster, unit)
    """
    from imas_codex.graph.client import GraphClient

    # Emit a diagnostic breakdown of eligible DD paths (grouped by
    # node_type × node_category) before any graph query or LLM spend.
    if logger.isEnabledFor(logging.INFO):
        try:
            _bd = report_extract_breakdown()
            logger.info(
                "DD extract breakdown — total=%d | node_types=%s | categories=%s"
                " | data_types=%s | has_errors=%d",
                _bd["total"],
                {str(k): v for k, v in _bd["by_node_type"].items()},
                _bd["by_category"],
                _bd["by_data_type"],
                _bd["has_errors_count"],
            )
            if _bd["samples"]:
                for bucket, paths in sorted(_bd["samples"].items()):
                    logger.info("  sample[%s]: %s", bucket, paths[:3])
        except Exception as exc:
            logger.debug("report_extract_breakdown skipped: %s", exc)

    def _status(text: str) -> None:
        if on_status:
            on_status(text)

    if existing_names is None:
        existing_names = set()

    _status("querying graph…")

    with GraphClient() as gc:
        # Resolve DD version and COCOS convention (single source of truth)
        dv_row = next(
            iter(
                gc.query("""
                MATCH (dv:DDVersion {is_current: true})
                OPTIONAL MATCH (dv)-[:HAS_COCOS]->(c:COCOS)
                RETURN dv.id AS dd_version, dv.cocos AS cocos_version,
                       properties(c) AS cocos_params
            """)
            ),
            None,
        )
        dd_version = dv_row["dd_version"] if dv_row else None
        cocos_version = dv_row["cocos_version"] if dv_row else None
        cocos_params = dv_row["cocos_params"] if dv_row else None

        params: dict = {"limit": limit, "sn_categories": list(SN_SOURCE_CATEGORIES)}
        where_parts = [
            # node_category is the authoritative namability taxonomy; node_type
            # (dynamic/static/constant) is orthogonal and must NOT gate extraction.
            "n.node_category IN $sn_categories",
            "n.description IS NOT NULL",
            "n.description <> ''",
            # Leaf invariant: exclude container nodes (STRUCTURE/STRUCT_ARRAY).
            # A STRUCTURE wraps sub-fields; it has no physics value itself.
            # This replaces the old node_type=['dynamic','constant'] gate — it
            # correctly admits static/geometry and coordinate leaves while
            # blocking wrappers like summary/global_quantities/q_95{value,source}.
            "NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])",
            # S1 equivalent at query level: core_instant_changes is a whole-IDS
            # dedup policy (duplicates core_profiles with "change in X" prefix).
            # Push into Cypher so LIMIT/ORDER BY pagination doesn't fill a batch
            # entirely with paths the classifier will reject.
            "ids.id <> 'core_instant_changes'",
        ]
        if ids_filter:
            where_parts.append("ids.id = $ids_filter")
            params["ids_filter"] = ids_filter
        if domain_filter:
            where_parts.append("n.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter
        if from_model:
            where_parts.append(
                "EXISTS { MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName) "
                "WHERE sn.model CONTAINS $from_model }"
            )
            params["from_model"] = from_model
        if not force:
            where_parts.append(
                "NOT EXISTS { MATCH (sns:StandardNameSource {source_id: n.id, source_type: 'dd'}) "
                "WHERE NOT (sns.status IN ['stale', 'failed', 'extracted']) }"
            )

        where_clause = " AND ".join(where_parts)
        query = _ENRICHED_QUERY.format(where_clause=where_clause)

        results = list(gc.query(query, **params))

        if not results:
            logger.info("No DD paths found matching filters")
            return []

        # Materialise has_errors / error_node_ids from the pattern comprehension.
        # error_node_ids is a Cypher list (may be empty list, not None).
        for row in results:
            error_ids = row.get("error_node_ids") or []
            row["error_node_ids"] = list(error_ids)
            row["has_errors"] = bool(error_ids)

        _status(f"found {len(results)} paths, resolving units…")

        # Resolve authoritative unit: prefer HAS_UNIT relationship, fall back to node property
        for row in results:
            row["unit"] = row.get("unit_from_rel") or row.get("unit") or None

        # Apply DD unit override/skip config — fixes upstream defects and
        # records unresolvable paths as skipped StandardNameSource records.
        results = _apply_unit_overrides(results, source_type="dd")
        if not results:
            logger.info("No DD paths remain after unit override filtering")
            return []

        # Skip paths that are semantically indeterminate by design
        # (e.g., /process/ slots where the concrete quantity is selected
        # at runtime by a sibling identifier.index).
        results = _apply_skip_by_design(results, source_type="dd")
        if not results:
            logger.info("No DD paths remain after skip-by-design filtering")
            return []

        # Extract-phase deny gate (W19A): skip paths that pass the
        # SN_SOURCE_CATEGORIES gate but should not receive standard names.
        # Examples: boolean constraint selectors (use_exact_*), engineering
        # coil geometry, control-system force matrices.
        results = _apply_extract_deny(results, source_type="dd")
        if not results:
            logger.info("No DD paths remain after extract deny filtering")
            return []

        # Collect cluster IDs for sibling lookup
        cluster_ids = {r["cluster_id"] for r in results if r.get("cluster_id")}
        siblings_by_cluster: dict[str, list[dict]] = {}
        if cluster_ids:
            _status(f"fetching siblings for {len(cluster_ids)} clusters…")
            path_set = {r["path"] for r in results}
            sib_results = list(
                gc.query(
                    _SIBLINGS_QUERY,
                    cluster_ids=sorted(cluster_ids),
                    exclude_paths=sorted(path_set),
                )
            )
            for sr in sib_results:
                cid = sr["cluster_id"]
                siblings_by_cluster.setdefault(cid, []).append(
                    {
                        "path": sr["sibling_path"],
                        "unit": sr.get("sibling_unit"),
                        "description": sr.get("sibling_desc", ""),
                    }
                )

    # Attach siblings to each result
    for row in results:
        cid = row.get("cluster_id")
        if cid and cid in siblings_by_cluster:
            row["cluster_siblings"] = siblings_by_cluster[cid][:10]
        else:
            row["cluster_siblings"] = []

    # --- Enrichment layer: classify, deduplicate, select primary cluster ----
    _status(f"classifying {len(results)} paths…")
    from imas_codex.standard_names.enrichment import (
        enrich_paths,
        group_by_concept_and_unit,
        group_for_name_only,
    )

    enriched = enrich_paths(results)

    if not enriched:
        logger.info("No quantity paths after classification")
        return []

    # Group GLOBALLY by (primary_cluster × unit) — same concept across IDSs
    # gets batched together for coherent naming.  In ``name_only`` mode
    # use the coarser (physics_domain × unit) grouping for higher
    # throughput during bootstrap (see Workstream 2a).
    if name_only:
        _status(
            f"grouping {len(enriched)} quantities by "
            f"(physics_domain × unit) [name_only]…"
        )
        batches = group_for_name_only(
            enriched,
            batch_size=name_only_batch_size,
            existing_names=existing_names,
            max_tokens=max_tokens,
        )
    else:
        _status(f"grouping {len(enriched)} quantities into batches…")
        batches = group_by_concept_and_unit(
            enriched,
            max_batch_size=max_batch_size,
            existing_names=existing_names,
            max_tokens=max_tokens,
        )

    logger.info(
        "Extracted %d batches from %d DD paths (%d clusters, force=%s)",
        len(batches),
        len(results),
        len(cluster_ids),
        force,
    )

    # Propagate COCOS metadata to batches
    for batch in batches:
        batch.dd_version = dd_version
        batch.cocos_version = cocos_version
        batch.cocos_params = cocos_params

    return batches


# Targeted extraction query — single path with full context
_TARGETED_PATH_QUERY = """
MATCH (n:IMASNode {id: $path})-[:IN_IDS]->(ids:IDS)
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(coord:IMASNode)
OPTIONAL MATCH (coord)-[:HAS_UNIT]->(cu:Unit)
RETURN n.id AS path,
       n.description AS description,
       n.documentation AS documentation,
       n.unit AS unit,
       u.id AS unit_from_rel,
       n.data_type AS data_type,
       n.physics_domain AS physics_domain,
       n.keywords AS keywords,
       n.node_category AS node_category,
       n.ndim AS ndim,
       n.lifecycle_status AS lifecycle_status,
       n.cocos_transformation_type AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
       ids.id AS ids_name,
       c.label AS cluster_label,
       c.id AS cluster_id,
       c.description AS cluster_description,
       parent.id AS parent_path,
       parent.description AS parent_description,
       parent.data_type AS parent_type,
       coord.id AS coord_path,
       coord.description AS coord_description,
       cu.id AS coord_unit
"""


def extract_specific_paths(
    paths: list[str],
    *,
    existing_names: set[str] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> list[ExtractionBatch]:
    """Extract specific DD paths with full context — bypasses classifier.

    Used for targeted debugging via ``--paths`` CLI flag. Each path becomes
    its own batch (serial processing) with rich context including the
    previous StandardName metadata and all DD paths linked to that name.

    Args:
        paths: Explicit list of DD path IDs to process
        existing_names: Known standard names for dedup awareness in compose
        on_status: Optional progress callback

    Returns:
        List of ExtractionBatch objects — one per path for maximum context
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.enrichment import (
        build_batch_context,
        select_primary_cluster,
    )

    if existing_names is None:
        existing_names = set()

    def _status(text: str) -> None:
        if on_status:
            on_status(text)

    _status(f"fetching {len(paths)} targeted paths…")

    results: list[dict] = []
    with GraphClient() as gc:
        # Resolve extraction DD version and COCOS convention
        dv_row = next(
            iter(
                gc.query("""
                MATCH (dv:DDVersion {is_current: true})
                OPTIONAL MATCH (dv)-[:HAS_COCOS]->(c:COCOS)
                RETURN dv.id AS dd_version, dv.cocos AS cocos_version,
                       properties(c) AS cocos_params
            """)
            ),
            None,
        )
        dd_version = dv_row["dd_version"] if dv_row else None
        cocos_version = dv_row["cocos_version"] if dv_row else None
        cocos_params = dv_row["cocos_params"] if dv_row else None

        for path in paths:
            rows = list(gc.query(_TARGETED_PATH_QUERY, path=path))
            if rows:
                results.extend(rows)
            else:
                logger.warning("Path not found in graph: %s", path)

    if not results:
        logger.info("No targeted paths found in graph")
        return []

    # Pre-resolve authoritative unit (prefer HAS_UNIT relationship), then
    # apply DD unit override/skip config before further processing.
    for row in results:
        row["unit"] = row.get("unit_from_rel") or row.get("unit") or None

    results = _apply_unit_overrides(results, source_type="dd")
    if not results:
        logger.info("No targeted DD paths remain after unit override filtering")
        return []

    results = _apply_skip_by_design(results, source_type="dd")
    if not results:
        logger.info("No targeted DD paths remain after skip-by-design filtering")
        return []

    # Deduplicate rows (multi-cluster → multiple rows per path)
    path_base: dict[str, dict] = {}
    path_clusters: dict[str, list[dict]] = {}

    for row in results:
        p = row.get("path", "")
        # Unit already resolved + override-applied above.
        row["cluster_siblings"] = []

        if p not in path_base:
            path_base[p] = dict(row)
            path_clusters[p] = []

        cid = row.get("cluster_id")
        if cid:
            existing_cids = {c["cluster_id"] for c in path_clusters[p]}
            if cid not in existing_cids:
                path_clusters[p].append(
                    {
                        "cluster_id": cid,
                        "cluster_label": row.get("cluster_label") or "",
                        "cluster_description": row.get("cluster_description") or "",
                    }
                )

    # Select primary cluster + attach enrichment
    enriched: list[dict] = []
    for path, base_row in path_base.items():
        clusters = path_clusters.get(path, [])
        primary = select_primary_cluster(clusters)
        base_row["primary_cluster_id"] = primary["cluster_id"] if primary else None
        base_row["primary_cluster_label"] = (
            primary["cluster_label"] if primary else None
        )
        base_row["primary_cluster_description"] = (
            primary["cluster_description"] if primary else None
        )
        base_row["all_clusters"] = clusters
        enriched.append(base_row)

    _status(f"creating {len(enriched)} single-path batches…")

    # One batch per path for maximum context depth
    batches: list[ExtractionBatch] = []
    for item in enriched:
        ids_name = item.get("ids_name", "unknown")
        context = build_batch_context([item], ids_name)
        batches.append(
            ExtractionBatch(
                source="dd",
                group_key=ids_name,
                items=[item],
                context=context,
                existing_names=existing_names,
            )
        )

    logger.info(
        "Extracted %d single-path batches from %d targeted paths",
        len(batches),
        len(enriched),
    )

    # Propagate COCOS metadata to batches
    for batch in batches:
        batch.dd_version = dd_version
        batch.cocos_version = cocos_version
        batch.cocos_params = cocos_params

    return batches
