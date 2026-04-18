"""Async workers for the standard-name enrich pipeline.

Five-phase enrich pipeline:

    EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST

- **extract**: queries graph for ``review_status='named'`` StandardNames,
  batches them for downstream processing.  Uses claim-token pattern to
  prevent parallel workers from double-processing.
- **contextualise**: gathers DD path descriptions, vector-similar
  neighbours, and domain siblings to build per-item context bundles.
- **document**: (stub) LLM call with enrich system/user prompts to generate
  descriptions and documentation.  C.3 will implement.
- **validate**: (stub) spelling, link integrity checks.  C.4 will implement.
- **persist**: (stub) writes enriched data + REFERENCES rels to graph.
  C.4 will implement.

Workers follow the same async signature and error-handling convention
as ``workers.py`` workers: ``async def worker(state, **_kwargs)``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.claims import retry_on_deadlock

if TYPE_CHECKING:
    from imas_codex.standard_names.enrich_state import StandardNameEnrichState

logger = logging.getLogger(__name__)

# Default batch size for grouping SNs into enrichment batches.
_ENRICH_BATCH_SIZE = 10

# Claim timeout for enrichment claims (ISO 8601 duration string).
_CLAIM_TIMEOUT = "PT300S"  # 5 minutes


# =============================================================================
# Graph helpers (extract phase)
# =============================================================================


@retry_on_deadlock()
def claim_names_for_enrichment(
    *,
    limit: int = 50,
    domain: str | None = None,
    ids: str | None = None,
    force: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim ``review_status='named'`` StandardNames for enrichment.

    Uses the two-step claim-token pattern (SET + verify) to prevent
    parallel workers from double-processing.  Skips nodes already at
    ``enriched`` or later status unless *force* is True.

    Returns ``(token, items)`` where *token* must be passed to
    downstream stages for release or mark-done.
    """
    from imas_codex.graph.client import GraphClient

    token = str(uuid.uuid4())

    # Build WHERE clauses
    where_parts = [
        "sn.review_status = 'named'",
        "(sn.enrich_claimed_at IS NULL"
        "  OR sn.enrich_claimed_at < datetime() - duration($timeout))",
    ]
    params: dict[str, Any] = {"limit": limit, "token": token, "timeout": _CLAIM_TIMEOUT}

    if not force:
        # Skip already-enriched nodes
        where_parts.append("sn.enriched_at IS NULL")

    if domain:
        where_parts.append("sn.physics_domain = $domain")
        params["domain"] = domain

    if ids:
        # Filter by IDS via source paths or HAS_STANDARD_NAME relationships
        where_parts.append(
            "EXISTS { MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn)"
            " MATCH (src)-[:IN_IDS]->(i:IDS {id: $ids}) }"
        )
        params["ids"] = ids

    where_clause = " AND ".join(where_parts)

    with GraphClient() as gc:
        # Step 1: Claim with random ordering and unique token
        gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {where_clause}
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.enrich_claimed_at = datetime(),
                sn.enrich_claim_token = $token
            """,
            **params,
        )

        # Step 2: Verify — only nodes with our token
        results = gc.query(
            """
            MATCH (sn:StandardName {enrich_claim_token: $token})
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   sn.tags AS tags,
                   sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.physical_base AS physical_base,
                   sn.subject AS subject,
                   sn.component AS component,
                   sn.coordinate AS coordinate,
                   sn.position AS position,
                   sn.process AS process,
                   sn.physics_domain AS physics_domain,
                   sn.confidence AS confidence,
                   sn.model AS model
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def release_enrichment_claims(token: str) -> int:
    """Release enrichment claims on error. Token-verified."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {enrich_claim_token: $token})
            SET sn.enrich_claimed_at = null,
                sn.enrich_claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


# =============================================================================
# EXTRACT phase
# =============================================================================


async def enrich_extract_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """Extract ``review_status='named'`` StandardNames into enrichment batches.

    Queries the graph for named SNs (filtered by domain/ids/limit),
    claims them with a token, and groups into batches for downstream
    processing.  Skips already-enriched nodes unless ``force=True``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_extract_worker")
    wlog.info(
        "Starting enrichment extraction (domain=%s, ids=%s, limit=%s)",
        state.domain,
        state.ids,
        state.limit,
    )

    claim_limit = state.limit or 500

    def _claim() -> tuple[str, list[dict[str, Any]]]:
        return claim_names_for_enrichment(
            limit=claim_limit,
            domain=state.domain,
            ids=state.ids,
            force=state.force,
        )

    if state.dry_run:
        # In dry-run mode, query without claiming
        token, items = await asyncio.to_thread(_claim)
        # Release immediately — no mutations in dry_run
        if items:
            await asyncio.to_thread(release_enrichment_claims, token)

        # Build batches from claimed items
        batches = _build_batches(items)
        state.batches = batches
        total_items = sum(len(b["items"]) for b in batches)
        state.extract_stats.total = total_items
        state.extract_stats.processed = total_items
        state.stats["extract_count"] = total_items
        state.stats["extract_batches"] = len(batches)
        wlog.info(
            "Dry run — extracted %d items in %d batches (claims released)",
            total_items,
            len(batches),
        )
        state.extract_stats.freeze_rate()
        state.extract_phase.mark_done()
        return

    # Live mode: claim and batch
    token, items = await asyncio.to_thread(_claim)

    if not items:
        wlog.info("No named StandardNames found for enrichment")
        state.extract_stats.freeze_rate()
        state.extract_phase.mark_done()
        return

    batches = _build_batches(items, token=token)
    state.batches = batches
    total_items = sum(len(b["items"]) for b in batches)
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d items in %d batches",
        total_items,
        len(batches),
    )
    state.stats["extract_count"] = total_items
    state.stats["extract_batches"] = len(batches)

    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()


def _build_batches(
    items: list[dict[str, Any]],
    batch_size: int = _ENRICH_BATCH_SIZE,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Split flat item list into enrichment batches.

    Each batch is a dict with ``items`` (list of SN dicts) and
    ``claim_token`` for downstream release/mark.
    """
    if not items:
        return []

    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batches.append(
            {
                "items": chunk,
                "claim_token": token,
                "batch_index": len(batches),
            }
        )
    return batches


# =============================================================================
# CONTEXTUALISE phase (C.2)
# =============================================================================

# Maximum characters for truncated description/documentation strings.
_DESC_TRUNCATE = 200

# How many vector-similar neighbours to fetch per SN.
_NEARBY_K = 6

# How many domain siblings to fetch per SN.
_SIBLINGS_LIMIT = 8

# Over-fetch factor for vector search (to compensate for status filtering).
_VECTOR_OVERFETCH = 3

# Grammar segment keys expected on each SN item from the generate phase.
_GRAMMAR_KEYS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
)


def _truncate(text: str | None, max_len: int = _DESC_TRUNCATE) -> str | None:
    """Truncate *text* to *max_len* chars, adding ellipsis if trimmed."""
    if not text:
        return text
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _build_grammar(item: dict[str, Any]) -> dict[str, str]:
    """Extract grammar decomposition dict from SN item qualifier fields."""
    return {k: item[k] for k in _GRAMMAR_KEYS if item.get(k)}


def _fetch_dd_paths_batch(
    gc: Any,
    sn_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch DD path details and COCOS type for a batch of SN ids.

    Returns a mapping ``{sn_id: {"dd_paths": [...], "cocos": str|None}}``.
    Each ``dd_paths`` entry is ``{path, ids, description, documentation, unit}``.
    """
    if not sn_ids:
        return {}

    rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.id IN $ids
        OPTIONAL MATCH (n:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
        RETURN sn.id AS sn_id,
               sn.cocos_transformation_type AS cocos,
               n.id AS path,
               n.ids AS ids,
               n.description AS description,
               n.documentation AS documentation,
               n.unit AS unit
        """,
        ids=sn_ids,
    )

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row["sn_id"]
        if sid not in result:
            result[sid] = {"dd_paths": [], "cocos": row.get("cocos")}
        # Only add path entry if an IMASNode was matched
        if row.get("path"):
            result[sid]["dd_paths"].append(
                {
                    "path": row["path"],
                    "ids": row.get("ids"),
                    "description": _truncate(row.get("description")),
                    "documentation": _truncate(row.get("documentation")),
                    "unit": row.get("unit"),
                }
            )
    return result


def _fetch_nearby_sns(
    gc: Any,
    items: list[dict[str, Any]],
    k: int = _NEARBY_K,
) -> dict[str, list[dict[str, str | None]]]:
    """Fetch k vector-similar StandardNames for each item via the vector index.

    Iterates items but reuses the same *gc* session. Only returns neighbours
    with ``review_status IN ['enriched', 'published', 'accepted']``.

    Returns ``{sn_id: [{name, description}, ...]}``.
    """
    result: dict[str, list[dict[str, str | None]]] = {}
    fetch_k = k * _VECTOR_OVERFETCH

    for item in items:
        sid = item["id"]
        try:
            rows = gc.query(
                """
                MATCH (target:StandardName {id: $target_id})
                WHERE target.embedding IS NOT NULL
                CALL db.index.vector.queryNodes(
                    'standard_name_desc_embedding', $fetch_k, target.embedding
                ) YIELD node, score
                WHERE node.id <> $target_id
                  AND node.review_status IN ['enriched', 'published', 'accepted']
                RETURN node.id AS name,
                       node.description AS description
                LIMIT $k
                """,
                target_id=sid,
                fetch_k=fetch_k,
                k=k,
            )
            result[sid] = [
                {"name": r["name"], "description": _truncate(r.get("description"))}
                for r in rows
            ]
        except Exception:
            # No embedding or index unavailable — empty list is fine
            logger.debug("Vector search failed for %s — skipping nearby", sid)
            result[sid] = []

    return result


def _fetch_domain_siblings(
    gc: Any,
    items: list[dict[str, Any]],
    limit: int = _SIBLINGS_LIMIT,
) -> dict[str, list[dict[str, str | None]]]:
    """Fetch domain siblings for a batch of items.

    Groups items by ``physics_domain`` to minimise query count.
    Falls back to IDS-based siblings when ``physics_domain`` is unset.

    Returns ``{sn_id: [{name, description}, ...]}``.
    """
    result: dict[str, list[dict[str, str | None]]] = {}

    # Group items by domain
    domain_groups: dict[str, list[str]] = {}
    ids_fallback: list[dict[str, Any]] = []

    for item in items:
        sid = item["id"]
        domain = item.get("physics_domain")
        if domain:
            domain_groups.setdefault(domain, []).append(sid)
        else:
            ids_fallback.append(item)

    # One query per unique domain
    for domain, sn_ids in domain_groups.items():
        try:
            rows = gc.query(
                """
                MATCH (sibling:StandardName)
                WHERE sibling.physics_domain = $domain
                  AND NOT (sibling.id IN $exclude_ids)
                  AND sibling.review_status IN [
                      'named', 'enriched', 'reviewable',
                      'published', 'accepted'
                  ]
                RETURN sibling.id AS name,
                       sibling.description AS description
                LIMIT $limit
                """,
                domain=domain,
                exclude_ids=sn_ids,
                limit=limit,
            )
            siblings = [
                {"name": r["name"], "description": _truncate(r.get("description"))}
                for r in rows
            ]
        except Exception:
            logger.warning("Sibling query failed for domain=%s", domain)
            siblings = []

        for sid in sn_ids:
            result[sid] = siblings

    # Fallback: IDS-based siblings for items without physics_domain
    for item in ids_fallback:
        sid = item["id"]
        source_paths = item.get("source_paths") or []
        # Extract IDS name from first source path (e.g. "equilibrium/time_slice/...")
        ids_name = source_paths[0].split("/")[0] if source_paths else None
        if not ids_name:
            result[sid] = []
            continue

        try:
            rows = gc.query(
                """
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sibling:StandardName)
                WHERE src.ids = $ids_name
                  AND sibling.id <> $exclude_id
                  AND sibling.review_status IN [
                      'named', 'enriched', 'reviewable',
                      'published', 'accepted'
                  ]
                RETURN DISTINCT sibling.id AS name,
                       sibling.description AS description
                LIMIT $limit
                """,
                ids_name=ids_name,
                exclude_id=sid,
                limit=limit,
            )
            result[sid] = [
                {"name": r["name"], "description": _truncate(r.get("description"))}
                for r in rows
            ]
        except Exception:
            logger.warning("IDS sibling query failed for %s (ids=%s)", sid, ids_name)
            result[sid] = []

    return result


async def enrich_contextualise_worker(
    state: StandardNameEnrichState, **_kwargs
) -> None:
    """Gather DD docs, nearby SNs, and domain siblings for enrichment context.

    For each batch of StandardName items, enriches them with:

    - **dd_paths**: linked IMASNode descriptions, documentation, and units.
    - **nearby**: k nearest-neighbour SNs by vector similarity.
    - **siblings**: SNs in the same physics domain or IDS.
    - **grammar**: decomposition dict from qualifier fields.
    - **cocos**: COCOS transformation type from the SN node.

    Context is written directly onto each item dict so that downstream
    workers (DOCUMENT) can template against it.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_contextualise_worker")

    if not state.batches:
        wlog.info("No batches to contextualise — skipping")
        state.contextualise_stats.freeze_rate()
        state.contextualise_phase.mark_done()
        return

    total_items = sum(len(b["items"]) for b in state.batches)
    state.contextualise_stats.total = total_items

    wlog.info(
        "Contextualising %d items across %d batches",
        total_items,
        len(state.batches),
    )

    processed = 0
    errors = 0

    for batch in state.batches:
        if state.stop_requested:
            wlog.info("Stop requested — aborting contextualise")
            break

        items = batch["items"]
        if not items:
            continue

        sn_ids = [it["id"] for it in items]

        # --- Fetch all context in threadpool (graph I/O) ---
        def _fetch_context(
            _sn_ids: list[str] = sn_ids,
            _items: list[dict[str, Any]] = items,
        ):
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                dd_data = _fetch_dd_paths_batch(gc, _sn_ids)
                nearby_data = _fetch_nearby_sns(gc, _items)
                sibling_data = _fetch_domain_siblings(gc, _items)
            return dd_data, nearby_data, sibling_data

        try:
            dd_data, nearby_data, sibling_data = await asyncio.to_thread(_fetch_context)
        except Exception:
            wlog.warning(
                "Graph error fetching context for batch %d — skipping",
                batch.get("batch_index", 0),
                exc_info=True,
            )
            errors += len(items)
            continue

        # --- Merge context onto each item ---
        for item in items:
            if state.stop_requested:
                wlog.info("Stop requested — aborting contextualise mid-batch")
                break

            sid = item["id"]
            try:
                dd_info = dd_data.get(sid, {})
                item["dd_paths"] = dd_info.get("dd_paths", [])
                item["cocos"] = dd_info.get("cocos")
                item["nearby"] = nearby_data.get(sid, [])
                item["siblings"] = sibling_data.get(sid, [])
                item["grammar"] = _build_grammar(item)

                # Preserve existing description/documentation as "current"
                # so the LLM can improve upon prior enrichment attempts
                item["current"] = {
                    "description": item.get("description"),
                    "documentation": item.get("documentation"),
                    "tags": item.get("tags"),
                    "links": item.get("links"),
                }

                processed += 1
            except Exception:
                wlog.warning(
                    "Error merging context for %s — skipping", sid, exc_info=True
                )
                errors += 1

    state.contextualise_stats.processed = processed
    state.contextualise_stats.errors = errors
    state.stats["contextualise_processed"] = processed
    state.stats["contextualise_errors"] = errors

    wlog.info(
        "Contextualise complete: %d processed, %d errors",
        processed,
        errors,
    )

    state.contextualise_stats.freeze_rate()
    state.contextualise_phase.mark_done()


# =============================================================================
# DOCUMENT phase (stub — C.3)
# =============================================================================


async def enrich_document_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """LLM call to generate descriptions and documentation for enrichment.

    .. note:: Stub — C.3 will implement.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_document_worker")

    if not state.batches:
        wlog.info("No batches to document — skipping")
        state.document_stats.freeze_rate()
        state.document_phase.mark_done()
        return

    wlog.info(
        "TODO: C.3 document worker not implemented — passing %d batches through",
        len(state.batches),
    )

    state.document_stats.total = sum(len(b["items"]) for b in state.batches)
    state.document_stats.processed = state.document_stats.total
    state.document_stats.freeze_rate()
    state.document_phase.mark_done()


# =============================================================================
# VALIDATE phase (stub — C.4)
# =============================================================================


async def enrich_validate_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """Validate enriched names: spelling, link integrity, description quality.

    .. note:: Stub — C.4 will implement.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_validate_worker")

    if not state.batches:
        wlog.info("No batches to validate — skipping")
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    wlog.info(
        "TODO: C.4 validate worker not implemented — passing %d batches through",
        len(state.batches),
    )

    state.validate_stats.total = sum(len(b["items"]) for b in state.batches)
    state.validate_stats.processed = state.validate_stats.total
    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()


# =============================================================================
# PERSIST phase (stub — C.4)
# =============================================================================


async def enrich_persist_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """Write enriched data and REFERENCES relationships to graph.

    .. note:: Stub — C.4 will implement.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_persist_worker")

    if state.dry_run:
        wlog.info("Dry run — skipping persist")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    if not state.batches:
        wlog.info("No batches to persist — skipping")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    wlog.info(
        "TODO: C.4 persist worker not implemented — passing %d batches through",
        len(state.batches),
    )

    # Release enrichment claims since we're not actually persisting
    for batch in state.batches:
        token = batch.get("claim_token")
        if token:
            try:
                await asyncio.to_thread(release_enrichment_claims, token)
            except Exception as e:
                wlog.warning("Failed to release enrichment claims: %s", e)

    state.persist_stats.total = sum(len(b["items"]) for b in state.batches)
    state.persist_stats.processed = state.persist_stats.total
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
