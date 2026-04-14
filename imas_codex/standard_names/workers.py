"""Async workers for the standard-name build pipeline.

Six-phase pipeline (review optional):

    EXTRACT → COMPOSE → [REVIEW] → VALIDATE → CONSOLIDATE → PERSIST

- **extract**: queries graph for DD paths or facility signals, builds batches
- **compose**: LLM-generates standard names from extraction batches
- **review**: cross-model review of composed candidates (optional)
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
from datetime import UTC, datetime
from functools import cache as _cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_codex.standard_names.sources.base import ExtractionBatch
    from imas_codex.standard_names.state import SNBuildState

logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACT phase
# =============================================================================


async def extract_worker(state: SNBuildState, **_kwargs) -> None:
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
                batches = extract_dd_candidates(
                    ids_filter=state.ids_filter,
                    domain_filter=state.domain_filter,
                    limit=state.limit or 500,
                    existing_names=existing,
                    on_status=_on_status,
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
    """Ensure all internal links have ``name:`` or ``dd:`` prefix."""
    result = []
    for link in links:
        if link.startswith(("http://", "https://", "name:", "dd:")):
            result.append(link)
        elif "/" in link:
            # Looks like a DD path (contains slash) — add dd: prefix
            result.append(f"dd:{link}")
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
       n.cocos_label_transformation AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
       n.lifecycle_status AS lifecycle_status,
       cs.id AS coordinate_spec_id,
       cs.coordinate_description AS coordinate_spec_description,
       ident.name AS identifier_schema_name,
       ident.documentation AS identifier_schema_doc,
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
    'path_added', 'cocos_label_transformation', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
RETURN vc.id AS change_id, vc.change_type AS change_type
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


def _enrich_batch_items(items: list[dict]) -> None:
    """Enrich batch items with rich DD context from the graph.

    Fetches coordinate specs, COCOS info, identifier schemas, sibling
    fields, cross-IDS cluster siblings, and version history for each
    item. Modifies items in-place.
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


async def compose_worker(state: SNBuildState, **_kwargs) -> None:
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
    from imas_codex.standard_names.models import SNComposeBatch

    model = state.compose_model or get_model("reasoning")
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

            user_context = {
                "items": batch.items,
                "ids_name": batch.group_key,
                "ids_contexts": ids_contexts,
                "existing_names": sorted(batch.existing_names)[:200],
                "cluster_context": batch.context,
                "nearby_existing_names": nearby,
                "cocos_version": batch.cocos_version,
                "dd_version": batch.dd_version,
            }
            user_prompt = render_prompt("sn/compose_dd", {**context, **user_context})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            result, cost, tokens = await acall_llm_structured(
                model=model,
                messages=messages,
                response_model=SNComposeBatch,
            )

            state.compose_stats.cost += cost
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
                physics_domain = (
                    source_item.get("physics_domain") if source_item else None
                )

                # Inject COCOS metadata from DD (authoritative, like unit)
                cocos_type = source_item.get("cocos_label") if source_item else None

                # Post-process links: ensure name: prefix for internal refs
                links = _normalize_links(c.links)

                # Post-process tags: strip primary tags, keep only secondary
                tags = _filter_secondary_tags(c.tags)

                candidates.append(
                    {
                        "id": c.standard_name,
                        "source_type": "dd" if state.source == "dd" else "signal",
                        "source_id": c.source_id,
                        "description": c.description,
                        "documentation": c.documentation,
                        "kind": c.kind,
                        "tags": tags,
                        "links": links,
                        "imas_paths": c.ids_paths,  # graph schema key
                        "fields": c.grammar_fields,
                        "confidence": c.confidence,
                        "reason": c.reason,
                        "validity_domain": c.validity_domain,
                        "constraints": c.constraints,
                        "unit": unit,
                        "physics_domain": physics_domain,
                        "cocos_transformation_type": cocos_type,
                        "dd_version": batch.dd_version,
                    }
                )

            # Collect vocab gaps for later reporting
            if result.vocab_gaps:
                for vg in result.vocab_gaps:
                    state.stats.setdefault("vocab_gaps", []).append(
                        {
                            "source_id": vg.source_id,
                            "segment": vg.segment,
                            "needed_token": vg.needed_token,
                            "reason": vg.reason,
                        }
                    )

            wlog.info(
                "Batch %s: %d composed, %d skipped (cost=$%.4f)",
                batch.group_key,
                len(result.candidates),
                len(result.skipped),
                cost,
            )
            # Stream batch completion to progress display
            state.compose_stats.stream_queue.add(
                [
                    {
                        "primary_text": batch.group_key,
                        "description": f"{len(result.candidates)} names  ${cost:.3f}",
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

    wlog.info(
        "Composition complete: %d composed, %d errors (cost=$%.4f)",
        len(composed),
        errors,
        state.compose_stats.cost,
    )
    state.stats["compose_count"] = len(composed)
    state.stats["compose_errors"] = errors

    state.compose_stats.freeze_rate()
    state.compose_phase.mark_done()


# =============================================================================
# REVIEW phase
# =============================================================================

# Default batch size for review LLM calls
_REVIEW_BATCH_SIZE = 10


def _get_grammar_enums() -> dict[str, list[str]]:
    """Return grammar enum values for prompt context."""
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

    return {
        "subjects": [e.value for e in Subject],
        "components": [e.value for e in Component],
        "coordinates": [e.value for e in Component],  # same enum
        "positions": [e.value for e in Position],
        "processes": [e.value for e in Process],
        "transformations": [e.value for e in Transformation],
        "geometric_bases": [e.value for e in GeometricBase],
        "objects": [e.value for e in Object],
        "binary_operators": [e.value for e in BinaryOperator],
    }


def _get_compose_context_for_review() -> dict[str, Any]:
    """Return compose context keys needed by shared prompt includes.

    The review prompt uses shared includes (_grammar_reference.md,
    _scoring_rubric.md) that require compose context variables like
    canonical_pattern, segment_order, vocabulary_sections, etc.
    """
    from imas_codex.standard_names.context import build_compose_context

    return build_compose_context()


async def review_worker(state: SNBuildState, **_kwargs) -> None:
    """Cross-model review of composed standard name candidates.

    Uses a different LLM model family to review candidates produced by
    the compose phase.  Applies accept/reject/revise verdicts.

    Results stored in ``state.reviewed``; validate reads from reviewed
    (falling back to composed when review is skipped).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_worker")

    if state.dry_run:
        wlog.info("Dry run — skipping review for %d candidates", len(state.composed))
        state.reviewed = list(state.composed)
        state.review_stats.total = len(state.composed)
        state.review_stats.processed = len(state.composed)
        state.stats["review_skipped"] = True
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    if not state.composed:
        wlog.info("No composed candidates to review — skipping")
        state.reviewed = []
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    wlog.info("Reviewing %d composed candidates", len(state.composed))
    state.review_stats.total = len(state.composed)

    from imas_codex.settings import get_model

    review_model = state.review_model or get_model("language")
    wlog.info("Review model: %s", review_model)

    grammar_enums = _get_grammar_enums()
    compose_ctx = _get_compose_context_for_review()
    existing_names = _get_existing_names_for_review()

    # Load calibration entries for consistent scoring
    calibration_entries = _load_calibration_entries()

    # Build batch context map from extracted batches (if available)
    batch_context_map: dict[str, str] = {}
    if state.extracted:
        for eb in state.extracted:
            for item in eb.items:
                path = item.get("path", "")
                if path:
                    batch_context_map[path] = eb.context

    scored: list[dict] = []
    duplicates = 0
    revised = 0
    errors = 0
    total_cost = 0.0
    total_tokens = 0

    # Build dedup set: all existing names, minus names for sources being
    # regenerated (--force mode) so they can be replaced.
    names_in_run: set[str] = set(existing_names)
    if state.force:
        # The composed candidates may re-generate the same names that already
        # exist in the graph. Remove those from the dedup set so they pass.
        composed_names = {c.get("id", "") for c in state.composed if c.get("id")}
        names_in_run -= composed_names

    candidates = list(state.composed)
    for batch_start in range(0, len(candidates), _REVIEW_BATCH_SIZE):
        if state.should_stop():
            wlog.info("Stop requested at batch starting %d", batch_start)
            break

        batch = candidates[batch_start : batch_start + _REVIEW_BATCH_SIZE]

        # Find batch context for this batch's candidates
        batch_ctx = ""
        for entry in batch:
            sid = entry.get("source_id", "")
            if sid in batch_context_map:
                batch_ctx = batch_context_map[sid]
                break

        try:
            batch_result = await _review_batch(
                batch,
                review_model,
                grammar_enums,
                names_in_run,
                wlog,
                calibration_entries=calibration_entries,
                batch_context=batch_ctx,
                compose_ctx=compose_ctx,
            )
            batch_scored, _, batch_revised, cost, tokens = batch_result
            total_cost += cost
            total_tokens += tokens

            for entry in batch_scored:
                name = entry.get("id", "")
                if name and name not in names_in_run:
                    scored.append(entry)
                    names_in_run.add(name)
                elif name in names_in_run:
                    wlog.debug("Duplicate name after review: %r — skipping", name)
                    duplicates += 1
                else:
                    scored.append(entry)

            revised += batch_revised

        except Exception:
            wlog.debug("Review batch failed at offset %d", batch_start, exc_info=True)
            errors += len(batch)
            scored.extend(batch)

        state.review_stats.processed = min(batch_start + len(batch), len(candidates))
        state.review_stats.record_batch(len(batch))
        # Stream review batch progress
        batch_num = batch_start // _REVIEW_BATCH_SIZE + 1
        low_score_count = sum(
            1 for s in scored if (s.get("reviewer_score") or 1.0) < 0.5
        )
        state.review_stats.stream_queue.add(
            [
                {
                    "primary_text": f"batch {batch_num}",
                    "description": (
                        f"{len(scored)} scored"
                        + (f"  ⚠{low_score_count} low" if low_score_count else "")
                        + f"  ${total_cost:.3f}"
                    ),
                }
            ]
        )

    # Store reviewer model and timestamp on all scored entries
    reviewed_at = datetime.now(UTC).isoformat()
    for entry in scored:
        entry["reviewer_model"] = review_model
        entry["reviewed_at"] = reviewed_at

    state.review_stats.errors = errors
    state.review_stats.cost = total_cost
    state.reviewed = scored

    wlog.info(
        "Review complete: %d scored, %d duplicates, %d revised, %d errors "
        "(cost: $%.4f, tokens: %d)",
        len(scored),
        duplicates,
        revised,
        errors,
        total_cost,
        total_tokens,
    )
    state.stats["review_scored"] = len(scored)
    state.stats["review_duplicates"] = duplicates
    state.stats["review_revised"] = revised
    state.stats["review_errors"] = errors
    state.stats["review_cost"] = total_cost

    state.review_stats.freeze_rate()
    state.review_phase.mark_done()


def _get_existing_names_for_review() -> set[str]:
    """Fetch existing standard names from graph for duplicate checking."""
    try:
        from imas_codex.standard_names.graph_ops import get_existing_standard_names

        return get_existing_standard_names()
    except Exception:
        return set()


def _load_calibration_entries() -> list[dict]:
    """Load calibration entries from benchmark_calibration.yaml."""
    from pathlib import Path

    import yaml

    cal_path = Path(__file__).parent / "benchmark_calibration.yaml"
    if not cal_path.exists():
        return []
    try:
        with open(cal_path) as f:
            data = yaml.safe_load(f)
        return data.get("entries", [])
    except Exception:
        return []


async def _review_batch(
    batch: list[dict],
    model: str,
    grammar_enums: dict[str, list[str]],
    existing_names: set[str],
    wlog: logging.LoggerAdapter,
    calibration_entries: list[dict] | None = None,
    batch_context: str | None = None,
    compose_ctx: dict[str, Any] | None = None,
) -> tuple[list[dict], int, int, float, int]:
    """Review a single batch of candidates via LLM with unified scoring."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import SNQualityReviewBatch, SNReviewVerdict

    cal = calibration_entries or []

    # Search for nearby existing names for collision detection
    nearby = _search_nearby_names(batch_context or "", k=5)

    # Enrich items with validation issues for reviewer context
    items_with_issues = []
    for item in batch:
        item_data = dict(item)
        item_data["validation_issues"] = item.get("validation_issues", [])
        items_with_issues.append(item_data)

    # Merge compose context (for shared includes) with review-specific keys
    base_ctx = dict(compose_ctx) if compose_ctx else {}

    # Build context for unified prompt
    context = {
        **base_ctx,
        "items": items_with_issues,
        "existing_names": sorted(existing_names)[:200],
        "calibration_entries": cal,
        "batch_context": batch_context or "",
        "nearby_existing_names": nearby,
        **grammar_enums,
    }

    # System prompt: rubric + calibration (cached across batches)
    system_context = {
        **base_ctx,
        "items": [],  # empty in system message
        "existing_names": [],
        "calibration_entries": cal,
        "batch_context": "",
        **grammar_enums,
    }
    system_prompt = render_prompt("sn/review", system_context)

    # User prompt: actual candidates + DD context
    user_prompt = render_prompt("sn/review", context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result, cost, tokens = await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=SNQualityReviewBatch,
    )

    entry_map: dict[str, dict] = {}
    for entry in batch:
        sid = entry.get("source_id") or entry.get("id") or ""
        entry_map[sid] = entry

    scored: list[dict] = []
    revised_count = 0

    for review in result.reviews:
        original = entry_map.get(review.source_id)
        if original is None:
            wlog.debug("Review returned unknown source_id: %s", review.source_id)
            continue

        # Store review scores on the entry for downstream persistence
        original["reviewer_score"] = review.scores.score
        original["reviewer_scores"] = json.dumps(review.scores.model_dump())
        original["reviewer_comments"] = review.reasoning
        original["review_tier"] = review.scores.tier

        if review.verdict == SNReviewVerdict.revise and review.revised_name:
            revised_entry = dict(original)
            revised_entry["id"] = review.revised_name
            if review.revised_fields:
                for key, value in review.revised_fields.items():
                    if key in revised_entry:
                        revised_entry[key] = value
            scored.append(revised_entry)
            revised_count += 1
            wlog.debug(
                "Revised %r → %r (score %.2f/%.1f): %s",
                review.standard_name,
                review.revised_name,
                review.scores.score,
                1.0,
                review.reasoning,
            )
        else:
            # All names are scored and persisted — no rejection
            scored.append(original)
            if review.verdict == SNReviewVerdict.reject:
                wlog.debug(
                    "Low score %r (%.2f/%.1f, %s): %s",
                    review.standard_name,
                    review.scores.score,
                    1.0,
                    review.scores.tier,
                    review.reasoning[:120],
                )

    # Entries not in the review result pass through (scored by default)
    reviewed_ids = {r.source_id for r in result.reviews}
    for entry in batch:
        sid = entry.get("source_id") or entry.get("id") or ""
        if sid not in reviewed_ids:
            scored.append(entry)

    return scored, 0, revised_count, cost, tokens


# =============================================================================
# VALIDATE phase
# =============================================================================


def _validate_via_isn(entry: dict) -> tuple[list[str], dict]:
    """Construct ISN Pydantic model and collect ALL validation issues.

    Returns:
        (issues: list[str], layer_summary: dict)

    This function is purely an annotator — it never rejects entries.
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
    isn_dict: dict[str, Any] = {
        "name": entry.get("id", ""),
        "kind": entry.get("kind", "scalar"),
        "description": entry.get("description", ""),
        "documentation": entry.get("documentation", ""),
        "tags": entry.get("tags", []),
        "links": entry.get("links", []),
        "physics_domain": entry.get("physics_domain", ""),
        "ids_paths": entry.get("imas_paths", []),
    }
    # ISN metadata kind forbids unit field entirely
    if isn_dict["kind"] != "metadata":
        unit = entry.get("unit") or "1"  # ISN requires '1' for dimensionless
        isn_dict["unit"] = unit

    # Layer 1: Pydantic model construction (fires 18 validators)
    model = None
    try:
        from imas_standard_names.models import create_standard_name_entry

        model = create_standard_name_entry(isn_dict)
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


async def validate_worker(state: SNBuildState, **_kwargs) -> None:
    """Validate composed names against grammar via round-trip + fields check.

    Reads from ``state.reviewed`` if review ran, else ``state.composed``.
    Reports distinct metrics: validate_valid, validate_invalid,
    validate_fields_consistent, validate_fields_inconsistent.
    Results stored in ``state.validated``.
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

    # Read from reviewed if review ran, else composed
    input_candidates = state.reviewed if state.reviewed is not None else state.composed

    if not input_candidates:
        wlog.info("No composed names to validate — skipping")
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    from imas_standard_names.grammar import (
        StandardName,
        compose_standard_name,
        parse_standard_name,
    )

    # Load tag vocabulary for soft validation
    try:
        from typing import get_args

        from imas_standard_names.grammar.tag_types import PrimaryTag, SecondaryTag

        valid_primary_tags = set(get_args(PrimaryTag))
        valid_secondary_tags = set(get_args(SecondaryTag))
        valid_tags = valid_primary_tags | valid_secondary_tags
    except Exception:
        valid_tags = set()

    # Collect existing names for link validation
    existing_names: set[str] = set()
    for entry in input_candidates:
        existing_names.add(entry.get("id", ""))

    wlog.info("Validating %d composed names", len(input_candidates))
    state.validate_stats.total = len(input_candidates)

    valid: list[dict] = []
    invalid_count = 0
    fields_consistent = 0
    fields_inconsistent = 0

    # Soft validation counters
    desc_present = 0
    desc_too_long = 0
    doc_present = 0
    doc_too_short = 0
    unit_valid = 0
    kind_valid = 0
    tags_valid = 0
    links_valid = 0

    _VALID_KINDS = {"scalar", "vector", "metadata"}

    for i, entry in enumerate(input_candidates):
        name = entry.get("id", "")
        try:
            parsed = parse_standard_name(name)
            normalized = compose_standard_name(parsed)
            if normalized != name:
                wlog.debug("Normalization: %r → %r", name, normalized)
                entry["id"] = normalized

            # Fields consistency check
            fields = entry.get("fields", {})
            if fields:
                try:
                    sn_fields = _convert_fields_to_grammar(fields)
                    if sn_fields:
                        sn = StandardName(**sn_fields)
                        from_fields = compose_standard_name(sn)
                        if from_fields == normalized:
                            fields_consistent += 1
                            entry["fields_consistent"] = True
                        else:
                            fields_inconsistent += 1
                            entry["fields_consistent"] = False
                            wlog.debug(
                                "Fields inconsistent: %r vs %r",
                                from_fields,
                                normalized,
                            )
                except Exception:
                    fields_inconsistent += 1
                    entry["fields_consistent"] = False

            # --- Soft validation checks (metrics only, never reject) ---

            # 1. Description present + length check
            desc = entry.get("description", "")
            if desc:
                desc_present += 1
                if len(desc) > 120:
                    desc_too_long += 1
                    wlog.debug("Description >120 chars for %r: %d", name, len(desc))
            else:
                wlog.debug("Missing description for %r", name)

            # 2. Documentation present + minimum length
            doc = entry.get("documentation", "")
            if doc:
                doc_present += 1
                if len(doc) < 200:
                    doc_too_short += 1
                    wlog.debug("Documentation <200 chars for %r: %d", name, len(doc))
            else:
                wlog.debug("Missing documentation for %r", name)

            # 3. Unit validity — simple pattern check
            unit = entry.get("unit")
            if unit and isinstance(unit, str) and len(unit) < 50:
                unit_valid += 1

            # 4. Kind validity
            kind = entry.get("kind", "")
            if kind in _VALID_KINDS:
                kind_valid += 1
            elif kind:
                wlog.debug("Invalid kind %r for %r", kind, name)

            # 5. Tags from vocabulary
            entry_tags = entry.get("tags") or []
            if entry_tags and valid_tags:
                if all(t in valid_tags for t in entry_tags):
                    tags_valid += 1
                else:
                    bad_tags = [t for t in entry_tags if t not in valid_tags]
                    wlog.debug("Unknown tags for %r: %s", name, bad_tags)
            elif entry_tags:
                tags_valid += 1  # no vocabulary loaded, accept any

            # 6. Links reference existing names
            entry_links = entry.get("links") or []
            if entry_links:
                if all(lnk in existing_names for lnk in entry_links):
                    links_valid += 1
                else:
                    unknown = [lnk for lnk in entry_links if lnk not in existing_names]
                    wlog.debug("Unknown links for %r: %s", name, unknown)

            # 7. ids_paths look like IMAS paths (contain '/')
            imas_paths = entry.get("imas_paths") or []
            for p in imas_paths:
                if "/" not in p:
                    wlog.debug("Suspicious ids_path for %r: %r", name, p)

            # --- ISN three-layer validation (annotate, never reject) ---
            try:
                issues, layer_summary = _validate_via_isn(entry)
                entry["validation_issues"] = issues
                entry["validation_layer_summary"] = json.dumps(layer_summary)
                if issues:
                    wlog.debug("ISN validation: %d issues for %r", len(issues), name)
            except Exception:
                wlog.debug("ISN validation failed for %r — skipping annotation", name)

            valid.append(entry)
        except Exception:
            wlog.debug("Validation failed for name: %r", name)
            invalid_count += 1

        state.validate_stats.processed = i + 1

    if valid:
        state.validate_stats.record_batch(len(valid))
    state.validate_stats.errors = invalid_count
    state.validated = valid

    wlog.info(
        "Validation complete: %d valid, %d invalid, "
        "%d fields consistent, %d fields inconsistent",
        len(valid),
        invalid_count,
        fields_consistent,
        fields_inconsistent,
    )
    wlog.info(
        "Soft checks: desc=%d/%d (>120: %d), doc=%d/%d (<200: %d), "
        "unit=%d, kind=%d, tags=%d, links=%d",
        desc_present,
        len(valid),
        desc_too_long,
        doc_present,
        len(valid),
        doc_too_short,
        unit_valid,
        kind_valid,
        tags_valid,
        links_valid,
    )
    state.stats["validate_valid"] = len(valid)
    state.stats["validate_invalid"] = invalid_count
    state.stats["validate_fields_consistent"] = fields_consistent
    state.stats["validate_fields_inconsistent"] = fields_inconsistent
    # Soft validation metrics
    state.stats["validate_desc_present"] = desc_present
    state.stats["validate_desc_too_long"] = desc_too_long
    state.stats["validate_doc_present"] = doc_present
    state.stats["validate_doc_too_short"] = doc_too_short
    state.stats["validate_unit_valid"] = unit_valid
    state.stats["validate_kind_valid"] = kind_valid
    state.stats["validate_tags_valid"] = tags_valid
    state.stats["validate_links_valid"] = links_valid

    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()
    state.finalize_stats.processed = 1


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


async def consolidate_worker(state: SNBuildState, **_kwargs) -> None:
    """Cross-batch consolidation: dedup, conflict detection, coverage accounting.

    Runs after VALIDATE, before PERSIST.  Reads from ``state.validated``,
    writes to ``state.consolidated``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_consolidate_worker")

    state.finalize_stats.status_text = "consolidating…"

    if state.dry_run:
        wlog.info("Dry run — skipping consolidation")
        state.consolidated = list(state.validated)
        state.consolidate_stats.total = len(state.validated)
        state.consolidate_stats.processed = len(state.validated)
        state.stats["consolidation_skipped"] = True
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    if not state.validated:
        wlog.info("No validated names to consolidate — skipping")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    from imas_codex.standard_names.consolidation import consolidate_candidates

    wlog.info("Consolidating %d validated candidates", len(state.validated))
    state.consolidate_stats.total = len(state.validated)

    # Collect all source paths for coverage accounting
    source_paths = None
    if state.extracted:
        source_paths = set()
        for batch in state.extracted:
            for item in batch.items:
                if item.get("path"):
                    source_paths.add(item["path"])

    result = await asyncio.to_thread(
        consolidate_candidates,
        state.validated,
        source_paths=source_paths,
    )

    state.consolidated = result.approved

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

    state.consolidate_stats.processed = len(state.validated)
    state.consolidate_stats.freeze_rate()
    state.consolidate_phase.mark_done()
    state.finalize_stats.processed = 2


# =============================================================================
# PERSIST phase
# =============================================================================


async def persist_worker(state: SNBuildState, **_kwargs) -> None:
    """Write consolidated standard names to graph with provenance.

    Creates StandardName nodes and HAS_STANDARD_NAME relationships.
    Enriches entries with model name, generation timestamp, review status.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_persist_worker")

    state.finalize_stats.status_text = "persisting…"

    if state.dry_run:
        wlog.info(
            "Dry run — skipping persist for %d consolidated names",
            len(state.consolidated),
        )
        state.persist_stats.total = len(state.consolidated)
        state.persist_stats.processed = len(state.consolidated)
        state.stats["persist_skipped"] = True
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    if not state.consolidated:
        wlog.info("No consolidated names to persist — skipping")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    from imas_codex.settings import get_model
    from imas_codex.standard_names.graph_ops import write_standard_names

    compose_model = state.compose_model or get_model("reasoning")
    now = datetime.now(UTC).isoformat()

    # Enrich with provenance — record the actual compose model used
    for entry in state.consolidated:
        entry.setdefault("model", compose_model)
        entry.setdefault("review_status", "drafted")
        entry.setdefault("generated_at", now)
        # confidence comes from LLM output — never default to 1.0

        # Extract grammar fields into top-level properties for graph
        fields = entry.get("fields", {})
        for field_name in (
            "physical_base",
            "subject",
            "component",
            "coordinate",
            "position",
            "process",
            "geometric_base",
            "object",
        ):
            if field_name in fields and field_name not in entry:
                entry[field_name] = fields[field_name]

    wlog.info("Persisting %d consolidated standard names", len(state.consolidated))
    state.persist_stats.total = len(state.consolidated)

    written = await asyncio.to_thread(write_standard_names, state.consolidated)

    # Embed descriptions for vector search
    if written > 0:
        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            embed_items = [
                {"id": e["id"], "description": e.get("description", "")}
                for e in state.consolidated
                if e.get("description")
            ]
            if embed_items:
                enriched = await asyncio.to_thread(
                    embed_descriptions_batch, embed_items
                )
                # Write embeddings back to graph
                from imas_codex.graph.client import GraphClient

                def _write_embeddings():
                    with GraphClient() as gc:
                        gc.query(
                            """
                            UNWIND $batch AS b
                            MATCH (sn:StandardName {id: b.id})
                            SET sn.embedding = b.embedding,
                                sn.embedded_at = datetime()
                            """,
                            batch=[
                                {"id": e["id"], "embedding": e["embedding"]}
                                for e in enriched
                                if e.get("embedding")
                            ],
                        )

                await asyncio.to_thread(_write_embeddings)
                wlog.info("Embedded %d StandardName descriptions", len(embed_items))
        except Exception:
            wlog.warning(
                "Embedding generation failed — names persisted without embeddings",
                exc_info=True,
            )

    state.persist_stats.processed = written
    state.persist_stats.record_batch(written)
    state.stats["persist_written"] = written

    # Post-success cleanup: detach stale HAS_STANDARD_NAME for targeted paths
    # Only runs when --force/--paths regenerated names for specific paths
    if state.force and written > 0:
        new_name_ids = {e["id"] for e in state.consolidated if e.get("id")}
        source_paths = set()
        for e in state.consolidated:
            for sp in e.get("source_ids", []):
                source_paths.add(sp)

        if source_paths and new_name_ids:

            def _cleanup_stale():
                with GraphClient() as gc:
                    result = list(
                        gc.query(
                            """
                            UNWIND $paths AS path
                            MATCH (n:IMASNode {id: path})-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                            WHERE NOT (sn.id IN $keep_names)
                              AND sn.review_status IN ['drafted', null]
                            DELETE r
                            RETURN count(r) AS detached
                            """,
                            paths=list(source_paths),
                            keep_names=list(new_name_ids),
                        )
                    )
                    return result[0]["detached"] if result else 0

            detached = await asyncio.to_thread(_cleanup_stale)
            if detached:
                wlog.info("Cleaned %d stale HAS_STANDARD_NAME relationships", detached)
                state.stats["stale_detached"] = detached

    wlog.info("Persist complete: %d written", written)
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.finalize_stats.processed = 3
    state.finalize_stats.status_text = "done"
    state.finalize_stats.freeze_rate()
