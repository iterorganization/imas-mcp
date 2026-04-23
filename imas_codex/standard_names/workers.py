"""Async workers for the standard-name build pipeline.

Five-phase generate pipeline:

    EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST

- **extract**: queries graph for DD paths or facility signals, builds batches
- **compose**: LLM-generates standard names from extraction batches
- **validate**: validates names against grammar via round-trip + fields check
- **consolidate**: cross-batch dedup, conflict detection, coverage accounting
- **persist**: writes consolidated names to graph with provenance

Workers follow the ``dd_workers.py`` pattern: each is an async function
with signature ``async def worker(state, **_kwargs)`` that updates stats,
marks phases done, and respects ``state.should_stop()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import cache as _cache
from typing import TYPE_CHECKING, Any

from imas_codex.standard_names.source_paths import (
    encode_source_path,
    strip_dd_prefix,
)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetLease
    from imas_codex.standard_names.sources.base import ExtractionBatch
    from imas_codex.standard_names.state import StandardNameBuildState

logger = logging.getLogger(__name__)

_GRAMMAR_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
    "geometric_base",
    "object",
)


# =============================================================================
# EXTRACT phase
# =============================================================================


async def extract_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Extract candidate quantities from graph entities into batches.

    For DD source: queries IMASNode paths, groups by cluster/IDS/prefix.
    Skips sources already linked via HAS_STANDARD_NAME unless --force.
    Stores ExtractionBatch objects in ``state.extracted``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_extract_worker")
    wlog.info("Starting extraction (source=%s)", state.source)

    def _on_status(text: str) -> None:
        state.extract_stats.status_text = text

    def _run() -> list:
        from imas_codex.standard_names.graph_ops import (
            get_existing_standard_names,
            get_named_source_ids,
        )
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        _on_status("loading existing names…")
        existing = get_existing_standard_names()

        # Regen mode: when --min-score F is passed, re-queue sources whose
        # linked StandardName has reviewer_score < min_score. The subsequent
        # feedback-injection block (always-on) attaches reviewer_comments /
        # tier / scores to each item so compose regenerates with critique
        # in-prompt.
        if state.is_regen_mode() and state.source == "dd":
            from imas_codex.standard_names.graph_ops import (
                fetch_low_score_sources,
            )
            from imas_codex.standard_names.sources.dd import extract_specific_paths

            _on_status(f"loading sources below reviewer_score={state.min_score}…")
            regen_sources = fetch_low_score_sources(
                min_score=state.min_score,
                domain=state.domain_filter,
                ids=state.ids_filter,
                limit=state.limit,
                source_type="dd",
            )
            if not regen_sources:
                wlog.info(
                    "Regen mode (--min-score %s): no low-score sources found "
                    "(domain=%s, ids=%s, limit=%s)",
                    state.min_score,
                    state.domain_filter,
                    state.ids_filter,
                    state.limit,
                )
                return []
            wlog.info(
                "Regen mode: %d sources below reviewer_score=%s "
                "(domain=%s, ids=%s, limit=%s)",
                len(regen_sources),
                state.min_score,
                state.domain_filter,
                state.ids_filter,
                state.limit,
            )
            paths = [r["source_id"] for r in regen_sources]
            batches = extract_specific_paths(
                paths=paths,
                existing_names=existing,
                on_status=_on_status,
            )
            return batches

        # Source-level skip for resumability (not in targeted mode)
        named_ids: set[str] = set()
        if not state.force and not state.paths_list:
            named_ids = get_named_source_ids()
            if named_ids:
                wlog.info("Skipping %d already-named sources", len(named_ids))

        if state.source == "dd":
            if state.paths_list:
                # Targeted mode: bypass graph query + classifier
                from imas_codex.standard_names.sources.dd import (
                    extract_specific_paths,
                )

                batches = extract_specific_paths(
                    paths=state.paths_list,
                    existing_names=existing,
                    on_status=_on_status,
                )
            else:
                from imas_codex.standard_names.batching import (
                    get_generate_batch_config,
                )

                batch_cfg = get_generate_batch_config()
                batches = extract_dd_candidates(
                    ids_filter=state.ids_filter,
                    domain_filter=state.domain_filter,
                    limit=state.limit or 500,
                    existing_names=existing,
                    on_status=_on_status,
                    from_model=state.from_model,
                    force=state.force,
                    name_only=state.name_only,
                    name_only_batch_size=state.name_only_batch_size,
                    max_batch_size=batch_cfg["batch_size"],
                    max_tokens=batch_cfg["max_tokens"],
                )
        else:
            wlog.error("Unknown source: %s", state.source)
            return []

        # Filter out already-named sources from batches
        if named_ids and not state.force:
            for batch in batches:
                batch.items = [
                    item
                    for item in batch.items
                    if item.get("path", item.get("signal_id")) not in named_ids
                ]
            # Remove empty batches
            batches = [b for b in batches if b.items]

        return batches

    batches = await asyncio.to_thread(_run)

    # Inject previous name context for --force regeneration
    if state.force:
        # --paths mode gets rich metadata (full docs, links, linked DD paths)
        use_rich = bool(state.paths_list)

        def _get_mapping():
            from imas_codex.standard_names.graph_ops import get_source_name_mapping

            return get_source_name_mapping(rich=use_rich)

        source_names = await asyncio.to_thread(_get_mapping)
        injected = 0
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path and path in source_names:
                    item["previous_name"] = source_names[path]
                    injected += 1
        if injected:
            wlog.info("Injected previous_name context for %d items", injected)

    # Inject prior reviewer feedback for targeted regeneration (always on).
    # The compose prompt's {% if item.review_feedback %} block surfaces the
    # previous reviewer critique so the LLM can directly address it in the
    # new name. No-op when no prior feedback exists.
    def _get_feedback():
        from imas_codex.standard_names.graph_ops import (
            fetch_review_feedback_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_review_feedback_for_sources(ids)

    feedback_map = await asyncio.to_thread(_get_feedback)
    fb_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in feedback_map:
                item["review_feedback"] = feedback_map[path]
                fb_injected += 1
    if fb_injected:
        wlog.info(
            "Injected review_feedback for %d items",
            fb_injected,
        )

    # Write StandardNameSource nodes for crash-resilient tracking
    if not state.dry_run and batches:
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        sources = []
        source_type = "dd" if state.source == "dd" else "signals"
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if not path:
                    continue
                sources.append(
                    {
                        "id": f"{source_type}:{path}",
                        "source_type": source_type,
                        "source_id": path,
                        "dd_path": path if source_type == "dd" else None,
                        "batch_key": batch.group_key,
                        "status": "extracted",
                        "description": item.get("description")
                        or item.get("documentation")
                        or "",
                    }
                )

        if sources:
            written = await asyncio.to_thread(
                merge_standard_name_sources, sources, force=state.force
            )
            wlog.info("Wrote %d StandardNameSource nodes to graph", written)

    total_items = sum(len(b.items) for b in batches)
    state.extracted = batches
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d batches, %d items",
        len(batches),
        total_items,
    )
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items

    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()
    state.extract_stats.stream_queue.add(
        [
            {
                "primary_text": "extract",
                "description": f"{total_items} paths in {len(batches)} batches",
            }
        ]
    )


# =============================================================================
# COMPOSE phase
# =============================================================================


def _search_nearby_names(query: str, k: int = 5) -> list[dict]:
    """Search for existing standard names near *query* for collision avoidance.

    Wraps :func:`imas_codex.standard_names.search.search_similar_names` with graceful
    fallback — never raises, returns ``[]`` if graph or embeddings are
    unavailable.
    """
    try:
        from imas_codex.standard_names.search import search_similar_names

        return search_similar_names(query, k=k)
    except Exception:
        return []


@_cache
def _get_secondary_tags() -> frozenset[str]:
    """Return the set of valid secondary tags from ISN grammar context."""
    try:
        from imas_standard_names.grammar.context import get_grammar_context

        ctx = get_grammar_context()
        td = ctx.get("tag_descriptions", {})
        return frozenset(td.get("secondary", {}).keys())
    except Exception:
        return frozenset()


def _normalize_links(links: list[str]) -> list[str]:
    """Normalize links to ``name:`` prefix, filtering out ``dd:`` links."""
    result = []
    for link in links:
        # Filter out any dd: links that slipped through
        if link.startswith("dd:"):
            continue
        # Drop raw DD paths (contain slash but no known prefix)
        if "/" in link and not link.startswith(("http://", "https://", "name:")):
            continue
        if link.startswith(("http://", "https://", "name:")):
            result.append(link)
        else:
            result.append(f"name:{link}")
    return result


def _filter_secondary_tags(tags: list[str]) -> list[str]:
    """Keep only valid secondary tags, stripping any primary tags."""
    secondary = _get_secondary_tags()
    if not secondary:
        return tags  # no vocabulary loaded, pass through
    return [t for t in tags if t in secondary]


# =============================================================================
# DD context enrichment — fetch rich graph data before composing
# =============================================================================

_DD_CONTEXT_QUERY = """
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(cs:IMASCoordinateSpec)
OPTIONAL MATCH (n)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (parent)-[:HAS_CHILD]->(sibling:IMASNode)
WHERE sibling.id <> $path
  AND NOT (sibling.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
WITH n, cs, ident, parent,
     collect(DISTINCT {
         path: sibling.id,
         description: sibling.description,
         data_type: sibling.data_type
     })[0..8] AS sibling_fields
RETURN n.coordinate1_same_as AS coordinate1,
       n.coordinate2_same_as AS coordinate2,
       n.coordinate3_same_as AS coordinate3,
       n.timebasepath AS timebase,
       n.cocos_transformation_type AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
       n.lifecycle_status AS lifecycle_status,
       cs.id AS coordinate_spec_id,
       cs.coordinate_description AS coordinate_spec_description,
       ident.name AS identifier_schema_name,
       ident.documentation AS identifier_schema_doc,
       ident.options AS identifier_options,
       parent.id AS parent_path,
       parent.description AS parent_description,
       sibling_fields
"""

_CROSS_IDS_QUERY = """
MATCH (n:IMASNode {id: $path})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN ['global', 'domain']
WITH c ORDER BY c.scope ASC LIMIT 3
MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
WHERE member.id <> $path
  AND NOT (member.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
RETURN c.label AS cluster_label,
       c.description AS cluster_description,
       c.scope AS cluster_scope,
       collect(DISTINCT member.id)[0..5] AS member_paths
"""

_VERSION_HISTORY_QUERY = """
MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n:IMASNode {id: $path})
WHERE vc.change_type IN [
    'path_added', 'cocos_transformation_type', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
RETURN vc.id AS change_id, vc.change_type AS change_type
"""

_ERROR_FIELDS_QUERY = """
MATCH (d:IMASNode {id: $path})-[:HAS_ERROR]->(e:IMASNode)
RETURN e.id AS error_path
"""

_IDS_CONTEXT_QUERY = """
MATCH (ids:IDS {id: $ids_name})
OPTIONAL MATCH (child:IMASNode)-[:IN_IDS]->(ids)
WHERE child.id STARTS WITH $ids_prefix
  AND child.data_type IN ['STRUCTURE', 'STRUCT_ARRAY', 'FLT_1D']
  AND size([x IN split(child.id, '/') WHERE true]) = 2
WITH ids,
     collect(DISTINCT {
         name: child.id, description: child.description, data_type: child.data_type
     })[0..10] AS top_sections
RETURN ids.description AS ids_description,
       ids.documentation AS ids_documentation,
       top_sections
"""


def _enrich_ids_context(ids_name: str) -> dict | None:
    """Fetch IDS-level context for batch header.

    Returns dict with ids_description, ids_documentation, top_sections,
    or None if IDS not found.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                _IDS_CONTEXT_QUERY,
                ids_name=ids_name,
                ids_prefix=f"{ids_name}/",
            )
        )
        if not rows:
            return None
        row = rows[0]
        sections = row.get("top_sections") or []
        valid = [s for s in sections if s.get("name")]
        return {
            "ids_description": row.get("ids_description") or "",
            "ids_documentation": row.get("ids_documentation") or "",
            "top_sections": valid,
        }


def _hybrid_search_neighbours(
    gc: Any,
    path: str,
    description: str | None = None,
    physics_domain: str | None = None,
    max_results: int = 15,
    search_k: int = 10,
) -> list[dict]:
    """Run parallel hybrid DD searches and pre-resolve HAS_STANDARD_NAME.

    Issues two hybrid queries per source path — one by description
    (physics-concept neighbours) and one by path text (structural cousins).
    Results are deduplicated, capped at *max_results*, and enriched with
    any already-minted standard name via a single batch Cypher query.

    Returns a list of dicts with keys: ``tag``, ``path``, ``ids``,
    ``unit``, ``physics_domain``, ``doc_short``, ``cocos_label``.
    """
    from imas_codex.graph.dd_search import hybrid_dd_search

    all_hits: dict[str, Any] = {}  # path → SearchHit (dedup by path)

    # Query 1: description-based (physics concept)
    desc_query = (description or "")[:200].strip()
    if desc_query:
        try:
            hits = hybrid_dd_search(
                gc,
                desc_query,
                node_category="quantity",
                physics_domain=physics_domain,
                k=search_k,
            )
            for h in hits:
                if h.path != path:
                    all_hits[h.path] = h
        except Exception:
            logger.debug(
                "Hybrid search (description) failed for %s", path, exc_info=True
            )

    # Query 2: path-text based (structural cousins)
    try:
        hits = hybrid_dd_search(
            gc,
            path,
            node_category="quantity",
            k=search_k,
        )
        for h in hits:
            if h.path != path and h.path not in all_hits:
                all_hits[h.path] = h
    except Exception:
        logger.debug("Hybrid search (path) failed for %s", path, exc_info=True)

    if not all_hits:
        return []

    # Cap to max_results (keep highest scored)
    sorted_hits = sorted(all_hits.values(), key=lambda h: h.score, reverse=True)[
        :max_results
    ]

    # Pre-resolve HAS_STANDARD_NAME in one batch query
    hit_paths = [h.path for h in sorted_hits]
    sn_map: dict[str, str | None] = {}
    try:
        rows = gc.query(
            """
            UNWIND $paths AS pid
            MATCH (n:IMASNode {id: pid})
            OPTIONAL MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN n.id AS path, sn.id AS sn_id
            """,
            paths=hit_paths,
        )
        for r in rows or []:
            sn_map[r["path"]] = r.get("sn_id")
    except Exception:
        logger.debug("HAS_STANDARD_NAME pre-resolution failed", exc_info=True)

    # Build compact dicts for prompt injection
    neighbours: list[dict] = []
    for h in sorted_hits:
        sn_id = sn_map.get(h.path)
        tag = f"name:{sn_id}" if sn_id else f"dd:{h.path}"
        doc = (h.documentation or h.description or "")[:120]
        neighbours.append(
            {
                "tag": tag,
                "path": h.path,
                "ids": h.ids_name,
                "unit": h.units or "",
                "physics_domain": h.physics_domain or "",
                "doc_short": doc,
                "cocos_label": h.cocos_transformation_type or "",
            }
        )

    return neighbours


# Cap for graph-relationship neighbour injection (per path).
_RELATED_MAX_RESULTS = 5


# Compose retry: on grammar/validation failure, retry with expanded context.
# Values resolved from settings accessors; module-level constants kept for
# backwards compatibility with any direct importers.
def _retry_attempts() -> int:
    from imas_codex.settings import get_sn_retry_attempts

    return get_sn_retry_attempts()


def _retry_k_expansion() -> int:
    from imas_codex.settings import get_sn_retry_k_expansion

    return get_sn_retry_k_expansion()


def _related_path_neighbours(
    gc: Any,
    path: str,
    *,
    max_results: int = _RELATED_MAX_RESULTS,
) -> list[dict]:
    """Fetch explicit graph-relationship neighbours for a DD path.

    Calls :func:`related_dd_search` to discover paths related via
    cluster membership, shared coordinates, matching units, identifier
    schemas, or COCOS transformation type.  Returns a compact list of
    dicts suitable for Jinja template injection.
    """
    from imas_codex.graph.dd_search import related_dd_search

    try:
        result = related_dd_search(
            gc,
            path,
            relationship_types="all",
            max_results=max_results,
        )
    except Exception:
        logger.debug("related_dd_search failed for %s", path, exc_info=True)
        return []

    if not result.hits:
        return []

    neighbours: list[dict] = []
    for hit in result.hits:
        neighbours.append(
            {
                "path": hit.path,
                "ids": hit.ids,
                "relationship_type": hit.relationship_type,
                "via": hit.via,
            }
        )

    return neighbours


def _enrich_batch_items(items: list[dict]) -> None:
    """Enrich batch items with rich DD context from the graph.

    Fetches coordinate specs, COCOS info, identifier schemas, sibling
    fields, cross-IDS cluster siblings, hybrid-search neighbours, and
    version history for each item. Modifies items in-place.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        for item in items:
            path = item.get("path")
            if not path:
                continue

            rows = list(gc.query(_DD_CONTEXT_QUERY, path=path))
            if not rows:
                continue

            row = rows[0]

            # Coordinate context
            coords = []
            for key in ("coordinate1", "coordinate2", "coordinate3"):
                val = row.get(key)
                if val:
                    coords.append(val)
            if coords:
                item["coordinate_paths"] = coords

            timebase = row.get("timebase")
            if timebase:
                item["timebase"] = timebase

            # COCOS
            cocos_label = row.get("cocos_label")
            cocos_expr = row.get("cocos_expression")
            if cocos_label:
                item["cocos_label"] = cocos_label
            if cocos_expr:
                item["cocos_expression"] = cocos_expr

            # Lifecycle
            lifecycle = row.get("lifecycle_status")
            if lifecycle and lifecycle != "active":
                item["lifecycle_status"] = lifecycle

            # Identifier schema
            ident_name = row.get("identifier_schema_name")
            if ident_name:
                item["identifier_schema"] = ident_name
                ident_doc = row.get("identifier_schema_doc")
                if ident_doc:
                    item["identifier_schema_doc"] = ident_doc

                # Parse identifier enum values from JSON-encoded options
                raw_options = row.get("identifier_options")
                if raw_options:
                    try:
                        parsed = (
                            json.loads(raw_options)
                            if isinstance(raw_options, str)
                            else raw_options
                        )
                        if isinstance(parsed, list):
                            item["identifier_values"] = [
                                {
                                    "name": opt.get("name", ""),
                                    "index": opt.get("index", 0),
                                    "description": opt.get("description", ""),
                                }
                                for opt in parsed[:20]
                                if opt.get("name")
                            ]
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Sibling fields (same parent, different leaf paths)
            siblings = row.get("sibling_fields") or []
            if siblings and isinstance(siblings, list):
                valid = [s for s in siblings if s.get("path")]
                if valid:
                    item["sibling_fields"] = valid

            # Cross-IDS cluster siblings
            cross_rows = list(gc.query(_CROSS_IDS_QUERY, path=path))
            if cross_rows:
                clusters = []
                cross_ids_paths = []
                for cr in cross_rows:
                    label = cr.get("cluster_label")
                    members = cr.get("member_paths") or []
                    if label and members:
                        clusters.append(
                            {
                                "label": label,
                                "description": cr.get("cluster_description") or "",
                                "scope": cr.get("cluster_scope") or "",
                                "members": members,
                            }
                        )
                        cross_ids_paths.extend(members)
                if clusters:
                    item["clusters"] = clusters
                if cross_ids_paths:
                    # Deduplicate
                    seen = set()
                    unique = []
                    for p in cross_ids_paths:
                        if p not in seen:
                            seen.add(p)
                            unique.append(p)
                    item["cross_ids_paths"] = unique[:8]

            # Version history (COCOS/sign changes are most important)
            version_rows = list(gc.query(_VERSION_HISTORY_QUERY, path=path))
            if version_rows:
                valid_changes = []
                for vr in version_rows:
                    change_id = vr.get("change_id") or ""
                    change_type = vr.get("change_type") or ""
                    # Version is encoded in the ID: path:change_type:version
                    parts = change_id.rsplit(":", 1)
                    version = parts[-1] if len(parts) >= 2 else ""
                    if version and change_type:
                        valid_changes.append(
                            {"version": version, "change_type": change_type}
                        )
                if valid_changes:
                    item["version_history"] = valid_changes

            # Hybrid-search neighbours (physics-concept + structural)
            # Parallel injection: both description-based and path-based
            # queries run, results deduplicated and pre-resolved for SN.
            hybrid = _hybrid_search_neighbours(
                gc,
                path,
                description=item.get("description"),
                physics_domain=item.get("physics_domain"),
            )
            if hybrid:
                item["hybrid_neighbours"] = hybrid

            # Graph-relationship neighbours (cluster, coordinate, unit,
            # identifier, COCOS — explicit graph edges, not vector search).
            related = _related_path_neighbours(gc, path)
            if related:
                item["related_neighbours"] = related

            # Error companion fields (uncertainty: _error_upper/lower/index)
            error_rows = list(gc.query(_ERROR_FIELDS_QUERY, path=path))
            if error_rows:
                error_fields = [
                    ef["error_path"] for ef in error_rows if ef.get("error_path")
                ]
                if error_fields:
                    item["error_fields"] = error_fields


def _is_attachment_consistent(source_id: str, sn_name: str) -> tuple[bool, str]:
    """Reject attachments where the DD path tense disagrees with the SN tense.

    E.g. ``change_in_electron_density`` may not be attached to
    ``core_profiles/.../density`` (a base quantity, not a change). Symmetric:
    a base-quantity SN may not absorb an ``instant_changes`` path.
    """
    change_prefixes = (
        "change_in_",
        "tendency_of_",
        "rate_of_",
        "rate_of_change_of_",
        "time_derivative_of_",
    )
    change_path_tokens = ("instant_changes", "/change/", "_delta", "tendency_")
    sn_is_change = any(sn_name.startswith(p) for p in change_prefixes)
    path_is_change = any(t in source_id for t in change_path_tokens)
    if sn_is_change and not path_is_change:
        return False, (
            f"tense mismatch: SN '{sn_name}' is a change/rate but path "
            f"'{source_id}' is a base quantity"
        )
    if path_is_change and not sn_is_change:
        return False, (
            f"tense mismatch: path '{source_id}' is a change/rate but SN "
            f"'{sn_name}' is a base quantity"
        )
    return True, ""


def _process_attachments(
    attachments: list, state: StandardNameBuildState, wlog: logging.LoggerAdapter
) -> None:
    """Attach DD paths to existing standard names without regeneration.

    Creates HAS_STANDARD_NAME relationships in the graph for paths that the
    compose LLM identified as mapping to existing names. Rejects attachments
    that fail deterministic consistency checks (e.g. tense mismatch).
    """
    from imas_codex.graph.client import GraphClient

    rejected: list[tuple[str, str, str]] = []
    accepted: list = []
    for a in attachments:
        ok, reason = _is_attachment_consistent(a.source_id, a.standard_name)
        if ok:
            accepted.append(a)
        else:
            rejected.append((a.source_id, a.standard_name, reason))

    for src, sn, why in rejected:
        wlog.warning("Rejected attachment %s → %s: %s", src, sn, why)

    if not accepted:
        if rejected:
            state.stats["attachments_rejected"] = state.stats.get(
                "attachments_rejected", 0
            ) + len(rejected)
        return

    batch = [
        {"source_id": a.source_id, "standard_name": a.standard_name} for a in accepted
    ]

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.standard_name})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                WITH sn, 'dd:' + b.source_id AS uri
                WHERE NOT uri IN coalesce(sn.source_paths, [])
                SET sn.source_paths = coalesce(sn.source_paths, []) + uri
                """,
                batch=batch,
            )

        for a in accepted:
            wlog.info("Attached %s → %s (%s)", a.source_id, a.standard_name, a.reason)

        prev = state.stats.get("attachments", 0)
        state.stats["attachments"] = prev + len(accepted)
        if rejected:
            state.stats["attachments_rejected"] = state.stats.get(
                "attachments_rejected", 0
            ) + len(rejected)
    except Exception:
        wlog.warning("Failed to process attachments", exc_info=True)


# ---------------------------------------------------------------------------
# StandardNameSource status updaters (Phase 5: incremental tracking)
# ---------------------------------------------------------------------------


def _update_sources_after_compose(
    candidates: list[dict], source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'composed' after successful batch composition."""
    from imas_codex.graph.client import GraphClient

    source_type = "dd" if source == "dd" else "signals"
    batch = []
    for c in candidates:
        source_id = c.get("source_id")
        sn_id = c.get("id")
        if source_id and sn_id:
            batch.append(
                {
                    "sns_id": f"{source_type}:{source_id}",
                    "sn_id": sn_id,
                }
            )

    if not batch:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $batch AS b
                MATCH (sns:StandardNameSource {id: b.sns_id})
                MATCH (sn:StandardName {id: b.sn_id})
                SET sns.status = 'composed',
                    sns.composed_at = datetime(),
                    sns.produced_sn_id = sn.id
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                RETURN count(sns) AS linked
                """,
                batch=batch,
            )
            linked = result[0]["linked"] if result else 0
        if linked < len(batch):
            wlog.warning(
                "Compose-linking gap: %d/%d sources had no matching "
                "StandardName (edge not written, source still 'extracted')",
                len(batch) - linked,
                len(batch),
            )
        wlog.debug("Updated %d StandardNameSource nodes to composed", linked)
    except Exception:
        wlog.warning("Failed to update StandardNameSource status", exc_info=True)


def _update_sources_after_attach(
    attachments: list, source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'attached' status."""
    from imas_codex.graph.client import GraphClient

    source_type = "dd" if source == "dd" else "signals"
    batch = []
    for a in attachments:
        if a.source_id:
            batch.append(
                {
                    "sns_id": f"{source_type}:{a.source_id}",
                    "sn_id": a.standard_name,
                }
            )

    if not batch:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $batch AS b
                MATCH (sns:StandardNameSource {id: b.sns_id})
                MATCH (sn:StandardName {id: b.sn_id})
                SET sns.status = 'attached',
                    sns.composed_at = datetime(),
                    sns.produced_sn_id = sn.id
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                RETURN count(sns) AS linked
                """,
                batch=batch,
            )
            linked = result[0]["linked"] if result else 0
        if linked < len(batch):
            wlog.warning(
                "Attach-linking gap: %d/%d sources had no matching "
                "StandardName (edge not written, source still 'extracted')",
                len(batch) - linked,
                len(batch),
            )
        wlog.debug("Updated %d StandardNameSource nodes to attached", linked)
    except Exception:
        wlog.warning(
            "Failed to update StandardNameSource attachment status", exc_info=True
        )


def _update_sources_after_vocab_gap(
    vocab_gaps: list[dict], source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'vocab_gap' status.

    Gaps reported on open/pseudo segments (e.g. ``physical_base``,
    ``grammar_ambiguity``) are ignored — they are not real vocabulary gaps
    and must not retire the source from future composition attempts.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.segments import is_open_segment

    source_type = "dd" if source == "dd" else "signals"
    source_ids = []
    skipped_open = 0
    for vg in vocab_gaps:
        if is_open_segment(vg.get("segment")):
            skipped_open += 1
            continue
        sid = vg.get("source_id")
        if sid:
            source_ids.append(f"{source_type}:{sid}")

    if skipped_open:
        wlog.debug(
            "Skipped vocab_gap status update for %d open-segment gaps",
            skipped_open,
        )

    if not source_ids:
        return

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS sns_id
                MATCH (sns:StandardNameSource {id: sns_id})
                SET sns.status = 'vocab_gap'
                """,
                ids=source_ids,
            )
        wlog.debug("Updated %d StandardNameSource nodes to vocab_gap", len(source_ids))
    except Exception:
        wlog.warning(
            "Failed to update StandardNameSource vocab_gap status", exc_info=True
        )


# Opus model for L7 borderline revision pass
_L7_REVISION_MODEL = "openrouter/anthropic/claude-opus-4.6"
_L7_MIN_REMAINING_BUDGET = 0.50  # Skip L7 if remaining budget < this


async def _grammar_retry(
    original_name: str,
    parse_error: str,
    model: str,
    acall_fn,
) -> str | None:
    """L6: Single grammar-failure re-prompt.

    Asks the LLM to revise a name that failed grammar round-trip,
    providing the parse error and a grammar cheat-sheet fragment.

    Returns the revised name string, or None on failure.
    """
    from pydantic import BaseModel, Field

    class GrammarRetryResponse(BaseModel):
        revised_name: str = Field(
            description="The revised standard name that passes grammar parsing"
        )
        explanation: str = Field(description="Brief explanation of the fix")

    retry_prompt = (
        f"The standard name `{original_name}` failed grammar parsing with error:\n"
        f"  {parse_error}\n\n"
        "Revise ONLY the name to pass the grammar round-trip. Rules:\n"
        "- Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]\n"
        "- physical_base is open vocabulary (snake_case physics terms)\n"
        "- All other segments must use valid grammar tokens\n"
        "- No abbreviations, no provenance verbs, no unit suffixes\n"
        "- Return the MINIMAL fix — keep the name as close to the original as possible.\n"
    )

    try:
        result, _cost, _tokens = await acall_fn(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            response_model=GrammarRetryResponse,
            service="standard-names",
        )
        return result.revised_name if result else None
    except Exception:
        return None


async def _opus_revise_candidate(
    candidate: dict,
    domain_vocabulary: str,
    reviewer_themes: list[str],
    acall_fn,
) -> str | None:
    """L7: Revision pass for low-confidence candidates using Opus model.

    Returns revised name string, or None on failure.
    """
    from pydantic import BaseModel, Field

    class OpusRevisionResponse(BaseModel):
        revised_name: str = Field(description="Improved standard name")
        confidence: float = Field(ge=0, le=1, description="Confidence in the revision")
        explanation: str = Field(description="Why this revision is better")

    name = candidate.get("id", "")
    reason = candidate.get("reason", "")
    description = candidate.get("description", "")
    original_confidence = candidate.get("confidence", 0.0)

    prompt_parts = [
        f"A standard name was generated with LOW confidence ({original_confidence:.2f}):",
        f"  Name: `{name}`",
        f"  Description: {description}",
        f"  Reason: {reason}",
        "",
        "Revise the name to be more precise, following ISN grammar rules.",
        "Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]",
        "",
    ]

    if domain_vocabulary:
        prompt_parts.append("Domain vocabulary (prefer these terms):")
        # Include first 10 lines of vocabulary
        for line in domain_vocabulary.split("\n")[:10]:
            prompt_parts.append(f"  {line}")
        prompt_parts.append("")

    if reviewer_themes:
        prompt_parts.append("Reviewer feedback themes to address:")
        for theme in reviewer_themes[:5]:
            prompt_parts.append(f"  - {theme}")
        prompt_parts.append("")

    prompt_parts.append(
        "Return a revised name with improved confidence. Only revise if you can do CLEARLY better."
    )

    try:
        result, _cost, _tokens = await acall_fn(
            model=_L7_REVISION_MODEL,
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            response_model=OpusRevisionResponse,
            service="standard-names",
        )
        if result and result.confidence > original_confidence:
            return result.revised_name
        return None
    except Exception:
        return None


async def compose_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """LLM-generate standard names from extracted batches.

    Uses acall_llm_structured() with system/user prompt split for
    prompt caching.  Runs batches concurrently with semaphore.
    Results stored in ``state.composed``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_compose_worker")

    total_items = sum(len(b.items) for b in state.extracted)

    if state.dry_run:
        wlog.info("Dry run — skipping composition for %d items", total_items)
        state.compose_stats.total = total_items
        state.compose_stats.processed = total_items
        state.stats["compose_skipped"] = True
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    if not state.extracted:
        wlog.info("No batches to compose — skipping")
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    model = state.compose_model or get_model("sn-run")
    context = build_compose_context()

    # Enrich batch items with rich DD context (coordinate specs, COCOS, siblings,
    # cross-IDS paths, version history)
    def _enrich_all_batches():
        for batch in state.extracted:
            _enrich_batch_items(batch.items)

    await asyncio.to_thread(_enrich_all_batches)
    wlog.info("Enriched batch items with DD context")

    # Propagate COCOS metadata to state (for downstream phases)
    if state.extracted:
        first_batch = state.extracted[0]
        state.dd_version = first_batch.dd_version
        state.cocos_version = first_batch.cocos_version
        state.cocos_params = first_batch.cocos_params

    # Pre-fetch IDS-level context for each unique IDS across batches
    ids_context_cache: dict[str, dict | None] = {}

    def _prefetch_ids_context():
        for batch in state.extracted:
            # Derive IDS names from item paths (first segment)
            ids_names = {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
            for ids_name in ids_names:
                if ids_name not in ids_context_cache:
                    ids_context_cache[ids_name] = _enrich_ids_context(ids_name)

    await asyncio.to_thread(_prefetch_ids_context)
    wlog.info("Fetched IDS context for %d IDS(s)", len(ids_context_cache))

    # Render system prompt once (cached via prompt caching)
    # Inject COCOS context for system prompt (all batches share one DD version)
    if state.extracted:
        first_batch = state.extracted[0]
        if first_batch.cocos_version:
            context["cocos_version"] = first_batch.cocos_version
            context["dd_version"] = first_batch.dd_version
            if first_batch.cocos_params:
                context["cocos_sigma_bp"] = first_batch.cocos_params.get("sigma_bp")
                context["cocos_e_bp"] = first_batch.cocos_params.get("e_bp")
                context["cocos_sigma_r_phi_z"] = first_batch.cocos_params.get(
                    "sigma_r_phi_z"
                )
                context["cocos_sigma_rho_theta_phi"] = first_batch.cocos_params.get(
                    "sigma_rho_theta_phi"
                )

    # --- L1: Domain-vocabulary pre-seeding ---
    # Inject validated domain vocabulary into system prompt context
    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab = ""
    if state.domain_filter:
        domain_vocab = await asyncio.to_thread(
            build_domain_vocabulary_preseed, state.domain_filter
        )
        if domain_vocab:
            wlog.info(
                "L1: Injected domain vocabulary preseed for %s", state.domain_filter
            )
    context["domain_vocabulary"] = domain_vocab

    # --- L4: Reviewer-theme extraction ---
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    if state.domain_filter:
        reviewer_themes = await asyncio.to_thread(
            extract_reviewer_themes, state.domain_filter
        )
        if reviewer_themes:
            wlog.info(
                "L4: Extracted %d reviewer themes for %s",
                len(reviewer_themes),
                state.domain_filter,
            )
    context["reviewer_themes"] = reviewer_themes

    # --- K3: Scored-example injection ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    # Derive physics_domains from domain_filter and batch items
    batch_domains: list[str] = []
    if state.domain_filter:
        batch_domains = [state.domain_filter]
    else:
        # Collect unique domains from all batch items
        _domains = {
            item.get("physics_domain")
            for batch in state.extracted
            for item in batch.items
            if item.get("physics_domain")
        }
        batch_domains = sorted(_domains)

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(gc, physics_domains=batch_domains, axis="name")

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    if compose_scored_examples:
        wlog.info(
            "K3: Loaded %d scored examples for compose (domains=%s)",
            len(compose_scored_examples),
            batch_domains or "all",
        )
    context["compose_scored_examples"] = compose_scored_examples

    system_prompt = render_prompt("sn/compose_system", context)

    wlog.info(
        "Composing standard names for %d items in %d batches (model=%s)",
        total_items,
        len(state.extracted),
        model,
    )
    state.compose_stats.total = total_items

    sem = asyncio.Semaphore(5)

    async def _compose_batch(batch: ExtractionBatch) -> list[dict]:
        async with sem:
            if state.should_stop():
                return []

            # Budget gate — reserve before doing any LLM work
            lease = None
            if state.budget_manager:
                max_retries = _retry_attempts()
                estimated = len(batch.items) * 0.01 * (max_retries + 1) * 1.3
                lease = state.budget_manager.reserve(estimated)
                if lease is None:
                    wlog.info(
                        "Budget exhausted — skipping compose batch %s",
                        batch.group_key,
                    )
                    return []

            try:
                return await _compose_batch_body(batch, lease)
            finally:
                if lease:
                    lease.release_unused()

    async def _compose_batch_body(
        batch: ExtractionBatch, lease: BudgetLease | None
    ) -> list[dict]:
        # Search for nearby existing names to help avoid duplicates
        nearby = _search_nearby_names(batch.context or batch.group_key)

        # IDS-level context — collect for each IDS present in batch
        ids_names = sorted(
            {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
        )
        ids_contexts = []
        for iname in ids_names:
            info = ids_context_cache.get(iname)
            if info:
                ids_contexts.append({"ids_name": iname, **info})

        # Pre-render COCOS guidance for items
        if batch.cocos_params:
            from imas_codex.standard_names.context import render_cocos_guidance

            for item in batch.items:
                cocos_label = item.get("cocos_label")
                if cocos_label:
                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, batch.cocos_params
                    )

        # --- Rate-quantity detection ---
        # When DD documentation indicates a rate/time-derivative, inject a
        # hard constraint so the LLM uses tendency_of_/change_in_ prefix
        # and writes a consistent description (prevents instant_change_*
        # names and name/description verb drift).
        import re as _re

        _RATE_DOC_PATTERNS = _re.compile(
            r"\b(instantaneous change|signed change|rate of change"
            r"|time derivative|per unit time|instant change|d/dt"
            r"|tendency of|time-rate)\b",
            _re.IGNORECASE,
        )
        for item in batch.items:
            haystack = " ".join(
                str(item.get(k, "") or "") for k in ("description", "documentation")
            )
            if _RATE_DOC_PATTERNS.search(haystack):
                item["rate_hint"] = True

        # --- L2: Reference SN few-shot retrieval ---
        # Synthesize query from first 3 path descriptions
        reference_exemplars: list[dict] = []
        try:
            from imas_codex.standard_names.search import (
                search_similar_sns_with_full_docs,
            )

            desc_snippets = [
                item.get("description", "")
                for item in batch.items[:3]
                if item.get("description")
            ]
            if desc_snippets:
                synth_query = "; ".join(desc_snippets)
                # Exclude names already in this batch's candidate IDs
                batch_ids = [
                    item.get("path", "").replace("/", "_") for item in batch.items
                ]
                reference_exemplars = await asyncio.to_thread(
                    search_similar_sns_with_full_docs,
                    synth_query,
                    k=5,
                    exclude_ids=batch_ids,
                )
        except Exception:
            wlog.debug("L2: Reference exemplar search failed", exc_info=True)

        user_context = {
            "items": batch.items,
            "ids_name": batch.group_key,
            "ids_contexts": ids_contexts,
            "existing_names": sorted(batch.existing_names)[:200],
            "cluster_context": batch.context,
            "nearby_existing_names": nearby,
            "reference_exemplars": reference_exemplars,
            "cocos_version": batch.cocos_version,
            "dd_version": batch.dd_version,
        }
        # Name-only batches (Workstream 2a) render a leaner user prompt
        # that trades per-item cluster siblings / COCOS blocks / sibling
        # fields for a "identify natural sub-groups, then name" directive.
        # System prompt and per-candidate L6/L7 logic are unchanged so
        # prompt caching and grammar safety stay intact.
        prompt_template = (
            "sn/compose_dd_names" if batch.mode == "names" else "sn/compose_dd"
        )
        user_prompt = render_prompt(prompt_template, {**context, **user_context})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # --- Delta H: bounded retry loop for failed compositions ---
        _total_compose_cost = 0.0
        _max_retries = _retry_attempts()
        for _compose_attempt in range(_max_retries + 1):
            result, cost, tokens = await acall_llm_structured(
                model=model,
                messages=messages,
                response_model=StandardNameComposeBatch,
                service="standard-names",
            )
            _total_compose_cost += cost

            # Charge actual LLM cost to budget lease
            if lease:
                lease.charge(cost)

            # Quick grammar round-trip check on all candidates
            _grammar_failures: list[str] = []
            try:
                from imas_standard_names.grammar import parse_standard_name

                for c in result.candidates:
                    try:
                        parse_standard_name(c.standard_name)
                    except Exception:
                        _grammar_failures.append(c.standard_name)
            except ImportError:
                pass  # ISN not installed — skip check

            if not _grammar_failures or _compose_attempt >= _max_retries:
                break

            # Re-enrich items with expanded hybrid search for retry
            wlog.info(
                "Composition retry %d/%d: %d grammar failures (%s) "
                "— re-composing with expanded DD context",
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
            )

            def _re_enrich_expanded():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as gc:
                    for item in batch.items:
                        path = item.get("path")
                        if not path:
                            continue
                        hybrid = _hybrid_search_neighbours(
                            gc,
                            path,
                            description=item.get("description"),
                            physics_domain=item.get("physics_domain"),
                            search_k=_retry_k_expansion(),
                        )
                        if hybrid:
                            item["hybrid_neighbours"] = hybrid

            await asyncio.to_thread(_re_enrich_expanded)

            _retry_reason = (
                f"Previous attempt failed: grammar round-trip failed for "
                f"{', '.join(_grammar_failures[:3])}. Consider expanded "
                f"neighbour context and produce a different name."
            )
            retry_render_ctx = {
                **context,
                **user_context,
                "retry_reason": _retry_reason,
            }
            user_prompt = render_prompt(prompt_template, retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        state.compose_stats.cost += _total_compose_cost
        state.compose_stats.processed += len(batch.items)
        state.compose_stats.record_batch(len(batch.items))

        candidates = []
        for c in result.candidates:
            # Find the matching source item to get authoritative fields
            source_item = next(
                (item for item in batch.items if item.get("path") == c.source_id),
                None,
            )
            # Inject unit from DD (authoritative, not LLM output)
            raw_unit = source_item.get("unit") if source_item else None
            # Normalize: '-', 'mixed', and None/empty are invalid in ISN
            if raw_unit in ("-", "mixed", None, ""):
                unit = "1"
            else:
                unit = raw_unit

            # Inject physics_domain from DD (authoritative, like unit)
            # Normalize empty string to None so coalesce works correctly
            raw_domain = source_item.get("physics_domain") if source_item else None
            physics_domain = raw_domain if raw_domain else None

            # Inject COCOS metadata from DD (authoritative, like unit)
            cocos_type = source_item.get("cocos_label") if source_item else None

            # Normalize name via grammar round-trip BEFORE persist
            # to avoid duplicate nodes if validate would rename
            name_id = c.standard_name
            grammar_failed = False
            try:
                from imas_standard_names.grammar import (
                    compose_standard_name,
                    parse_standard_name,
                )

                parsed = parse_standard_name(name_id)
                normalized = compose_standard_name(parsed)
                if normalized != name_id:
                    wlog.debug(
                        "Pre-persist normalization: %r → %r", name_id, normalized
                    )
                    name_id = normalized
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug("Grammar parse failed for %r — attempting L6 retry", name_id)

                # --- L6: Grammar-failure re-prompt (single retry) ---
                state.grammar_retries += 1
                try:
                    retry_name = await _grammar_retry(
                        name_id, str(gram_exc), model, acall_llm_structured
                    )
                    if retry_name and retry_name != name_id:
                        # Verify the retry result actually parses
                        parsed = parse_standard_name(retry_name)
                        normalized = compose_standard_name(parsed)
                        name_id = normalized
                        grammar_failed = False
                        state.grammar_retries_succeeded += 1
                        wlog.info(
                            "L6: Grammar retry succeeded: %r → %r",
                            c.standard_name,
                            name_id,
                        )
                except Exception:
                    wlog.debug("L6: Grammar retry also failed for %r", name_id)

            candidates.append(
                {
                    "id": name_id,
                    "source_types": ["dd"] if state.source == "dd" else ["signals"],
                    "source_id": c.source_id,
                    "kind": c.kind,
                    "source_paths": [
                        encode_source_path(
                            "dd" if state.source == "dd" else "signals", p
                        )
                        for p in (c.dd_paths or [])
                    ],
                    "fields": c.grammar_fields,
                    "confidence": c.confidence,
                    "reason": c.reason,
                    "unit": unit,
                    "physics_domain": physics_domain,
                    "cocos_transformation_type": cocos_type,
                    "cocos": batch.cocos_version,
                    "dd_version": batch.dd_version,
                    # L6: track grammar retry exhaustion
                    **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
                }
            )

            # --- B9: Mint error siblings deterministically ---
            # If the parent has HAS_ERROR edges, mint uncertainty
            # modifier siblings without LLM calls.
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                from imas_codex.standard_names.error_siblings import (
                    mint_error_siblings,
                )

                siblings = mint_error_siblings(
                    name_id,
                    error_node_ids=source_item["error_node_ids"],
                    unit=unit,
                    physics_domain=physics_domain,
                    cocos_type=cocos_type,
                    cocos_version=batch.cocos_version,
                    dd_version=batch.dd_version,
                )
                if siblings:
                    candidates.extend(siblings)
                    state.error_siblings_minted = getattr(
                        state, "error_siblings_minted", 0
                    ) + len(siblings)
                    wlog.debug(
                        "B9: Minted %d error siblings for parent %r",
                        len(siblings),
                        name_id,
                    )

        # Collect vocab gaps and persist immediately
        if result.vocab_gaps:
            gap_dicts = []
            for vg in result.vocab_gaps:
                gap_dict = {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "needed_token": vg.needed_token,
                    "reason": vg.reason,
                }
                state.stats.setdefault("vocab_gaps", []).append(gap_dict)
                gap_dicts.append(gap_dict)

            # Persist to graph immediately so gaps survive cost-limit interruption
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            source_type = "dd" if state.source == "dd" else "signals"
            await asyncio.to_thread(write_vocab_gaps, gap_dicts, source_type)
            wlog.debug("Persisted %d vocab gaps to graph", len(gap_dicts))

            if not state.dry_run:
                _update_sources_after_vocab_gap(gap_dicts, state.source, wlog)

        # Process attachments — paths that map to existing names
        if result.attachments:
            _process_attachments(result.attachments, state, wlog)
            if not state.dry_run:
                _update_sources_after_attach(result.attachments, state.source, wlog)

        # --- GRAPH-STATE-MACHINE: persist immediately per batch ---
        # This ensures completed batches survive cost-limit cancellation.
        if candidates:
            from imas_codex.standard_names.graph_ops import persist_composed_batch

            written = await asyncio.to_thread(
                persist_composed_batch,
                candidates,
                compose_model=model,
                dd_version=batch.dd_version,
                cocos_version=batch.cocos_version,
            )
            wlog.debug("Persisted %d names from batch %s", written, batch.group_key)

        # Update StandardNameSource nodes to composed status
        if candidates and not state.dry_run:
            _update_sources_after_compose(candidates, state.source, wlog)

        wlog.info(
            "Batch %s: %d composed, %d attached, %d skipped (cost=$%.4f)",
            batch.group_key,
            len(result.candidates),
            len(result.attachments),
            len(result.skipped),
            cost,
        )
        # Stream batch completion to progress display
        attach_label = (
            f"+{len(result.attachments)} attached  " if result.attachments else ""
        )
        state.compose_stats.stream_queue.add(
            [
                {
                    "primary_text": batch.group_key,
                    "description": (
                        f"{len(result.candidates)} names  {attach_label}${cost:.3f}"
                    ),
                }
            ]
        )
        return candidates

    tasks = [_compose_batch(batch) for batch in state.extracted]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    composed: list[dict] = []
    errors = 0
    for r in results:
        if isinstance(r, list):
            composed.extend(r)
        elif isinstance(r, Exception):
            errors += 1
            wlog.warning("Batch failed: %s", r)

    state.composed = composed
    state.compose_stats.errors = errors

    # --- L7: Borderline-confidence two-pass (Opus revision) ---
    remaining_budget = (state.cost_limit or float("inf")) - state.total_cost
    low_confidence = [
        c
        for c in composed
        if c.get("confidence", 1.0) < 0.7 and not c.get("_grammar_retry_exhausted")
    ][:5]  # Cap at 5 per batch to respect cost budget

    if low_confidence and remaining_budget >= _L7_MIN_REMAINING_BUDGET:
        wlog.info(
            "L7: Attempting Opus revision for %d low-confidence candidates",
            len(low_confidence),
        )
        for cand in low_confidence:
            if state.should_stop():
                break
            state.opus_revisions_attempted += 1
            try:
                revised = await _opus_revise_candidate(
                    cand,
                    domain_vocab,
                    reviewer_themes,
                    acall_llm_structured,
                )
                if revised and revised != cand.get("id"):
                    # Verify revised name parses
                    from imas_standard_names.grammar import (
                        compose_standard_name as _compose_sn,
                        parse_standard_name as _parse_sn,
                    )

                    parsed = _parse_sn(revised)
                    normalized = _compose_sn(parsed)
                    # Accept only if self-reported improvement
                    wlog.info(
                        "L7: Opus revision accepted: %r → %r",
                        cand["id"],
                        normalized,
                    )
                    cand["id"] = normalized
                    state.opus_revisions_accepted += 1
            except Exception:
                wlog.debug(
                    "L7: Opus revision failed for %r", cand.get("id"), exc_info=True
                )
    elif low_confidence and remaining_budget < _L7_MIN_REMAINING_BUDGET:
        wlog.info(
            "L7: Skipped — remaining budget $%.2f < $%.2f threshold",
            remaining_budget,
            _L7_MIN_REMAINING_BUDGET,
        )

    attached = state.stats.get("attachments", 0)
    wlog.info(
        "Composition complete: %d composed, %d attached, %d errors (cost=$%.4f)",
        len(composed),
        attached,
        errors,
        state.compose_stats.cost,
    )

    # --- Batch-size telemetry (Workstream 2a) ---
    # Report the distribution of items per batch and the name-only mode
    # indicator so rotation summaries can compare name-only vs default
    # throughput without bespoke log scraping.
    if state.extracted:
        sizes = [len(b.items) for b in state.extracted]
        total_items_in_batches = sum(sizes)
        name_only_batches = sum(
            1 for b in state.extracted if getattr(b, "mode", "default") == "names"
        )
        singleton_count = sum(1 for s in sizes if s == 1)
        wlog.info(
            "Batch telemetry: %d batches (%d name_only), total_items=%d, "
            "mean=%.2f, min=%d, max=%d, singletons=%d (%.1f%%), cost_per_batch=$%.4f",
            len(sizes),
            name_only_batches,
            total_items_in_batches,
            total_items_in_batches / len(sizes) if sizes else 0.0,
            min(sizes) if sizes else 0,
            max(sizes) if sizes else 0,
            singleton_count,
            100.0 * singleton_count / len(sizes) if sizes else 0.0,
            state.compose_stats.cost / len(sizes) if sizes else 0.0,
        )
        state.stats["compose_batches"] = len(sizes)
        state.stats["compose_batches_name_only"] = name_only_batches
        state.stats["compose_batch_mean_size"] = (
            total_items_in_batches / len(sizes) if sizes else 0.0
        )
        state.stats["compose_batch_singleton_pct"] = (
            100.0 * singleton_count / len(sizes) if sizes else 0.0
        )

    state.stats["compose_count"] = len(composed)
    state.stats["compose_errors"] = errors
    state.stats["compose_cost"] = state.compose_stats.cost
    state.stats["compose_model"] = model
    state.stats["grammar_retries"] = state.grammar_retries
    state.stats["grammar_retries_succeeded"] = state.grammar_retries_succeeded
    state.stats["opus_revisions_attempted"] = state.opus_revisions_attempted
    state.stats["opus_revisions_accepted"] = state.opus_revisions_accepted

    state.compose_stats.freeze_rate()
    state.compose_phase.mark_done()


# =============================================================================
# VALIDATE phase
# =============================================================================


def _validate_via_isn(
    entry: dict,
) -> tuple[list[str], dict]:
    """Construct ISN Pydantic model and collect ALL validation issues.

    Returns:
        (issues: list[str], layer_summary: dict)

    Compose is always name-only (ADR-1): validation uses ISN's name-only
    model. This function is purely an annotator — it never rejects entries.
    Parseability is checked upstream by the grammar round-trip in
    validate_worker. This function attaches quality annotations.
    """
    from pydantic import ValidationError

    issues: list[str] = []
    summary = {
        "pydantic": {"passed": True, "error_count": 0},
        "semantic": {"issue_count": 0, "skipped": False},
        "description": {"issue_count": 0},
    }

    # Map codex dict keys to ISN model fields
    from imas_codex.standard_names.kind_derivation import to_isn_kind

    # Name-only ISN variants forbid dd_paths and physics_domain — those
    # fields only exist on the full (non-name-only) model. Compose is
    # always name-only (ADR-1), so we omit them here. DD-path provenance
    # is retained on the codex graph node via `source_paths`.
    isn_dict: dict[str, Any] = {
        "name": entry.get("id", ""),
        "kind": to_isn_kind(entry.get("kind", "scalar")),
    }
    # ISN metadata kind forbids unit field entirely
    if isn_dict["kind"] != "metadata":
        unit = entry.get("unit") or "1"  # ISN requires '1' for dimensionless
        isn_dict["unit"] = unit

    # Layer 1: Pydantic model construction (fires 18 validators)
    model = None
    try:
        from imas_standard_names.models import create_standard_name_entry

        model = create_standard_name_entry(isn_dict, name_only=True)
    except ValidationError as e:
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = len(e.errors())
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            issues.append(f"[pydantic:{field}] {err['msg']}")
    except Exception as e:
        # Non-validation errors (import issues, etc.) — don't crash
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = 1
        issues.append(f"[pydantic:unknown] {e}")

    # Layer 2: Semantic checks (only if model constructed)
    if model is not None:
        try:
            from imas_standard_names.validation.semantic import run_semantic_checks

            sem_issues = run_semantic_checks({isn_dict["name"]: model})
            summary["semantic"]["issue_count"] = len(sem_issues)
            issues.extend(f"[semantic] {i}" for i in sem_issues)
        except Exception as e:
            summary["semantic"]["skipped"] = True
            issues.append(f"[semantic] check failed: {e}")
    else:
        summary["semantic"]["skipped"] = True

    # Layer 3: Description quality
    try:
        from imas_standard_names.validation.description import validate_description

        desc_issues = validate_description(isn_dict)
        summary["description"]["issue_count"] = len(desc_issues)
        issues.extend(f"[description] {i['message']}" for i in desc_issues)
    except Exception as e:
        issues.append(f"[description] check failed: {e}")

    return issues, summary


def _is_quarantined(issues: list[str], layer_summary: dict) -> bool:
    """Determine whether validation issues are critical (quarantine the name).

    Critical failures that make a name unusable for publication:
    - Grammar round-trip failure (``parse_error:`` prefix)
    - Pydantic validation failure (layer 1 did not pass)
    - Empty or missing description (no ``id`` or empty string)
    - Invalid kind value
    - L3 critical audit failures (latex_def_check, synonym_check, multi_subject_check)
    - L6 grammar retry exhausted

    Non-critical issues (semantic warnings, description quality hints,
    non-critical audits) do NOT trigger quarantine — they are advisory.
    """
    # Grammar round-trip failures are always critical
    if any(i.startswith("parse_error:") for i in issues):
        return True

    # Grammar ambiguity is also critical — the name can't be reliably parsed
    if any("grammar:ambiguity" in i for i in issues):
        return True

    # Pydantic validation failure (model construction failed)
    pydantic = layer_summary.get("pydantic", {})
    if not pydantic.get("passed", True):
        return True

    # L6: Grammar retry exhausted
    if any("audit:grammar_retry_exhausted" in i for i in issues):
        return True

    # L3: Critical audit failures
    from imas_codex.standard_names.audits import has_critical_audit_failure

    if has_critical_audit_failure(issues):
        return True

    return False


async def validate_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Validate composed names via ISN grammar checks (claim loop).

    Graph-primary: claims unvalidated StandardName nodes, runs ISN
    three-layer validation + grammar round-trip, marks results on graph.
    Follows the claim/mark/release pattern from discovery workers.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_validate_worker")

    # Initialize finalize progress (3 steps: validate, consolidate, persist)
    state.finalize_stats.total = 3
    state.finalize_stats.status_text = "validating…"

    if state.dry_run:
        wlog.info("Dry run — skipping validation")
        count = sum(len(b.items) for b in state.extracted) if state.extracted else 0
        state.validate_stats.total = count
        state.validate_stats.processed = count
        state.stats["validate_skipped"] = True
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    from imas_standard_names.grammar import (
        StandardName,
        compose_standard_name,
        parse_standard_name,
    )

    from imas_codex.standard_names.graph_ops import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    _BATCH_SIZE = 50

    total_valid = 0
    total_invalid = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 50

    wlog.info("Starting validation claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_validation, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more unvalidated names — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for validation (token=%s)", len(items), token[:8])

        # Process the claimed batch
        try:
            results: list[dict[str, Any]] = []
            batch_invalid = 0

            for entry in items:
                name = entry.get("id", "")
                try:
                    # Grammar round-trip validates parsability
                    # (normalization already done at compose time)
                    parsed = parse_standard_name(name)
                    compose_standard_name(parsed)

                    # Fields consistency check
                    fields_dict = {}
                    for fk in _GRAMMAR_FIELDS:
                        val = entry.get(fk)
                        if val:
                            fields_dict[fk] = val
                    if fields_dict:
                        try:
                            sn_fields = _convert_fields_to_grammar(fields_dict)
                            if sn_fields:
                                sn = StandardName(**sn_fields)
                                compose_standard_name(sn)
                        except Exception:
                            pass

                    # ISN three-layer validation (annotate, never reject)
                    issues, layer_summary = _validate_via_isn(entry)

                    # --- L3: Post-gen audits ---
                    try:
                        from imas_codex.standard_names.audits import run_audits

                        source_path = None
                        source_paths = entry.get("source_paths") or []
                        if source_paths:
                            # Use first source path for provenance check
                            source_path = strip_dd_prefix(source_paths[0])

                        audit_issues = run_audits(
                            candidate=entry,
                            existing_sns_in_domain=None,  # Synonym check needs embeddings — skip in basic mode
                            source_path=source_path,
                            source_cocos_type=entry.get("cocos_transformation_type"),
                        )
                        if audit_issues:
                            issues.extend(audit_issues)
                        state.audits_run += 1
                        if audit_issues:
                            state.audits_failed += 1
                    except Exception:
                        wlog.debug("L3: Audit failed for %r", name, exc_info=True)

                    # L6: grammar retry exhausted flag from compose
                    if entry.get("_grammar_retry_exhausted"):
                        issues.append("audit:grammar_retry_exhausted")

                    results.append(
                        {
                            "id": name,
                            "validation_issues": issues,
                            "validation_layer_summary": json.dumps(layer_summary),
                            "validation_status": (
                                "quarantined"
                                if _is_quarantined(issues, layer_summary)
                                else "valid"
                            ),
                        }
                    )
                except Exception as exc:
                    exc_msg = str(exc).lower()
                    issues: list[str] = []

                    # Classify specific grammar ambiguities
                    if "component" in exc_msg and "coordinate" in exc_msg:
                        issues.append(
                            f"grammar:ambiguity:component_coordinate_overlap: {name}"
                        )
                    elif "ambig" in exc_msg:
                        issues.append(f"grammar:ambiguity:unclassified: {name}")
                    else:
                        issues.append(
                            f"parse_error: grammar round-trip failed for {name}"
                        )

                    wlog.debug(
                        "Validation error for %r: %s — tagging with %s",
                        name,
                        exc_msg[:80],
                        issues[0].split(":")[0],
                    )
                    results.append(
                        {
                            "id": name,
                            "validation_issues": issues,
                            "validation_layer_summary": json.dumps({}),
                            "validation_status": "quarantined",
                        }
                    )
                    batch_invalid += 1

            # Mark results on graph (token-verified)
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            total_valid += marked
            total_invalid += batch_invalid
            state.validate_stats.processed += marked

            wlog.info(
                "Validated batch: %d marked, %d errors",
                marked,
                batch_invalid,
            )
            state.validate_stats.record_batch(marked)

        except Exception:
            wlog.warning("Validation batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_validation_claims, token)

    state.stats["validate_valid"] = total_valid
    state.stats["validate_invalid"] = total_invalid
    state.stats["audits_run"] = state.audits_run
    state.stats["audits_failed"] = state.audits_failed
    state.validate_stats.errors = total_invalid
    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()
    state.finalize_stats.processed = 1
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "validate",
                "description": f"{total_valid} valid  {total_invalid} invalid",
            }
        ]
    )


def _convert_fields_to_grammar(fields: dict) -> dict:
    """Convert string field values to grammar enum instances."""
    from imas_standard_names.grammar import (
        BinaryOperator,
        Component,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
        Transformation,
    )

    enum_map = {
        "subject": Subject,
        "component": Component,
        "coordinate": Component,
        "position": Position,
        "process": Process,
        "transformation": Transformation,
        "geometric_base": GeometricBase,
        "object": Object,
        "binary_operator": BinaryOperator,
    }

    sn_fields: dict = {}
    for k, v in fields.items():
        if k == "physical_base":
            sn_fields[k] = v
        elif k in enum_map:
            sn_fields[k] = enum_map[k](v)
    return sn_fields


# =============================================================================
# CONSOLIDATE phase
# =============================================================================


async def consolidate_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Cross-batch consolidation: dedup, conflict detection, coverage accounting.

    Graph-primary: queries all validated StandardNames from graph, runs
    consolidation analysis, marks approved names with ``consolidated_at``.
    Read-only query (no claims needed — single-pass batch analysis).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_consolidate_worker")

    state.finalize_stats.status_text = "consolidating…"

    if state.dry_run:
        wlog.info("Dry run — skipping consolidation")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    from imas_codex.standard_names.consolidation import consolidate_candidates
    from imas_codex.standard_names.graph_ops import (
        get_validated_names,
        mark_names_consolidated,
    )

    # Always read from graph — this is the primary data source
    validated = await asyncio.to_thread(
        get_validated_names,
        ids_filter=getattr(state, "ids_filter", None),
    )

    if not validated:
        wlog.info("No validated names to consolidate — skipping")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    wlog.info("Consolidating %d validated candidates from graph", len(validated))
    state.consolidate_stats.total = len(validated)

    result = await asyncio.to_thread(consolidate_candidates, validated)

    # Mark approved names with consolidated_at on graph
    approved_ids = [e["id"] for e in result.approved if e.get("id")]
    if approved_ids:
        marked = await asyncio.to_thread(mark_names_consolidated, approved_ids)
        wlog.info("Marked %d names as consolidated", marked)

    # Log results
    wlog.info(
        "Consolidation: %d approved, %d conflicts, %d coverage gaps, %d reused",
        len(result.approved),
        len(result.conflicts),
        len(result.coverage_gaps),
        len(result.reused),
    )

    # Record stats
    state.stats["consolidation"] = result.stats
    if result.conflicts:
        for conflict in result.conflicts:
            wlog.warning(
                "Conflict: %s (%s) — %s",
                conflict.standard_name,
                conflict.conflict_type,
                conflict.details,
            )
    if result.coverage_gaps:
        wlog.info("Coverage gaps: %d unmapped source paths", len(result.coverage_gaps))

    state.consolidate_stats.processed = len(validated)
    state.consolidate_stats.freeze_rate()
    state.consolidate_phase.mark_done()
    state.finalize_stats.processed = 2
    conflicts_count = len(result.conflicts) if result.conflicts else 0
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "consolidate",
                "description": (
                    f"{len(result.approved)} names  {conflicts_count} conflicts"
                ),
            }
        ]
    )


# =============================================================================
# PERSIST phase
# =============================================================================


async def persist_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Compute embeddings for consolidated StandardNames (claim loop).

    Graph-primary: claims unembedded StandardNames, computes embeddings,
    writes results back to graph. Names are already persisted by compose —
    this worker handles the embedding enrichment pass.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_persist_worker")

    state.finalize_stats.status_text = "embedding…"

    if state.dry_run:
        wlog.info("Dry run — skipping embedding")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    from imas_codex.standard_names.graph_ops import (
        claim_names_for_embedding,
        mark_names_embedded,
        release_embedding_claims,
    )

    total_embedded = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 100

    wlog.info("Starting embedding claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_embedding, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more names needing embedding — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for embedding (token=%s)", len(items), token[:8])

        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            enriched = await asyncio.to_thread(embed_descriptions_batch, items)
            embed_batch = [
                {"id": e["id"], "embedding": e["embedding"]}
                for e in enriched
                if e.get("embedding")
            ]

            marked = await asyncio.to_thread(mark_names_embedded, token, embed_batch)
            total_embedded += marked

            wlog.info("Embedded batch: %d names", marked)
            state.persist_stats.processed += marked
            state.persist_stats.record_batch(marked)

        except Exception:
            wlog.warning("Embedding batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_embedding_claims, token)

    # Post-success cleanup: detach stale HAS_STANDARD_NAME for targeted paths
    # Only runs when --force/--paths regenerated names for specific paths
    if state.force and total_embedded > 0 and state.extracted:
        new_name_ids: set[str] = set()
        source_paths: set[str] = set()

        # Collect from graph — names we just embedded
        from imas_codex.graph.client import GraphClient

        def _get_recent_names():
            with GraphClient() as gc:
                results = gc.query(
                    """
                    MATCH (sn:StandardName)
                    WHERE sn.embedded_at IS NOT NULL
                      AND sn.pipeline_status IN ['named', 'drafted']
                    RETURN sn.id AS id, sn.source_paths AS source_paths
                    """
                )
                for r in results:
                    new_name_ids.add(r["id"])
                    for p in r["source_paths"] or []:
                        source_paths.add(p)

        await asyncio.to_thread(_get_recent_names)

        if source_paths and new_name_ids:

            def _cleanup_stale():
                # Strip dd: prefix for IMASNode.id lookup
                bare_paths = [strip_dd_prefix(p) for p in source_paths]
                with GraphClient() as gc:
                    result = list(
                        gc.query(
                            """
                            UNWIND $paths AS path
                            MATCH (n:IMASNode {id: path})-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                            WHERE NOT (sn.id IN $keep_names)
                              AND sn.pipeline_status IN ['named', 'drafted', null]
                            DELETE r
                            RETURN count(r) AS detached
                            """,
                            paths=bare_paths,
                            keep_names=list(new_name_ids),
                        )
                    )
                    return result[0]["detached"] if result else 0

            detached = await asyncio.to_thread(_cleanup_stale)
            if detached:
                wlog.info("Cleaned %d stale HAS_STANDARD_NAME relationships", detached)
                state.stats["stale_detached"] = detached

    state.stats["persist_embedded"] = total_embedded
    wlog.info("Persist complete: %d embedded", total_embedded)
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.finalize_stats.processed = 3
    state.finalize_stats.status_text = "done"
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "persist",
                "description": f"{total_embedded} embedded",
            }
        ]
    )
    state.finalize_stats.freeze_rate()
