"""Async workers for the standard-name enrich pipeline.

Five-phase enrich pipeline:

    EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST

- **extract**: queries graph for ``review_status='named'`` StandardNames,
  batches them for downstream processing.  Uses claim-token pattern to
  prevent parallel workers from double-processing.
- **contextualise**: (stub) gathers DD docs, nearby SNs, domain siblings.
  C.2 will implement.
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
# CONTEXTUALISE phase (stub — C.2)
# =============================================================================


async def enrich_contextualise_worker(
    state: StandardNameEnrichState, **_kwargs
) -> None:
    """Gather DD docs, nearby SNs, and domain siblings for enrichment context.

    .. note:: Stub — C.2 will implement.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_contextualise_worker")

    if not state.batches:
        wlog.info("No batches to contextualise — skipping")
        state.contextualise_stats.freeze_rate()
        state.contextualise_phase.mark_done()
        return

    wlog.info(
        "TODO: C.2 contextualise worker not implemented — passing %d batches through",
        len(state.batches),
    )

    # Pass batches through unchanged
    state.contextualise_stats.total = sum(len(b["items"]) for b in state.batches)
    state.contextualise_stats.processed = state.contextualise_stats.total
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
