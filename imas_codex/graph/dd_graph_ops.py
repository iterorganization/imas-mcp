"""Graph operations for the DD build pipeline state machine.

Claim/complete/has_pending functions following the standard discovery
pattern from ``imas_codex.discovery.base.claims``.

IMASNode lifecycle::

    built → enriched → embedded

DDVersion lifecycle::

    extracted → built

Workers claim batches via ``claimed_at`` + ``claim_token`` two-step
verify, process them, then advance the status and clear the claim.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Collection

from imas_codex.core.node_categories import EMBEDDABLE_CATEGORIES, ENRICHABLE_CATEGORIES
from imas_codex.discovery.base.claims import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    retry_on_deadlock,
)
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = DEFAULT_CLAIM_TIMEOUT_SECONDS


# =============================================================================
# IMASNode — enrichment claims (built → enriched)
# =============================================================================


@retry_on_deadlock()
def claim_paths_for_enrichment(
    limit: int = 50,
    *,
    ids_filter: set[str] | None = None,
) -> list[dict]:
    """Claim built IMASNodes for LLM enrichment.

    Returns list of dicts with path metadata needed by the enrichment
    pipeline.  Uses the standard claim_token two-step pattern.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"

    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {
        "status": "built",
        "cutoff": cutoff,
        "limit": limit,
        "token": token,
    }
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        # Step 1: Claim with ORDER BY depth then rand()
        gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              AND (p.claimed_at IS NULL
                   OR p.claimed_at < datetime() - duration($cutoff))
              {ids_clause}
            WITH p
            ORDER BY size(split(p.id, '/')) ASC, rand()
            LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            categories=list(ENRICHABLE_CATEGORIES),
            **params,
        )

        # Step 2: Read back only paths we won
        result = gc.query(
            """
            MATCH (p:IMASNode {claim_token: $token})
            RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
                   p.data_type AS data_type, p.ids AS ids,
                   p.cocos_label_transformation AS cocos_label_transformation,
                   p.enrichment_hash AS enrichment_hash
            """,
            token=token,
        )
        claimed = list(result)
        logger.debug(
            "claim_paths_for_enrichment: requested %d, won %d",
            limit,
            len(claimed),
        )
        return claimed


def mark_paths_enriched(updates: list[dict]) -> int:
    """Mark enriched paths: set status=enriched, clear claimed_at.

    Each update dict must have at minimum ``id``.  Additional fields
    (description, keywords, enrichment_hash, etc.) are set on the node.
    """
    if not updates:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS item
            MATCH (p:IMASNode {id: item.id})
            SET p.status = 'enriched',
                p.description = coalesce(item.description, p.description),
                p.keywords = coalesce(item.keywords, p.keywords),
                p.enrichment_hash = coalesce(item.enrichment_hash, p.enrichment_hash),
                p.enrichment_model = coalesce(item.enrichment_model, p.enrichment_model),
                p.enrichment_source = coalesce(item.enrichment_source, p.enrichment_source),
                p.physics_domain = coalesce(item.physics_domain, p.physics_domain),
                p.enriched_at = datetime(),
                p.claimed_at = null,
                p.claim_token = null
            RETURN count(p) AS updated
            """,
            updates=updates,
        )
        count = result[0]["updated"] if result else 0
        return count


def release_enrichment_claims(path_ids: list[str]) -> None:
    """Release enrichment claims on error (clear claimed_at)."""
    if not path_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            WHERE p.claimed_at IS NOT NULL
            SET p.claimed_at = null, p.claim_token = null
            """,
            ids=path_ids,
        )


def has_pending_enrichment(*, ids_filter: set[str] | None = None) -> bool:
    """Check if there are built IMASNodes awaiting enrichment.

    Counts ALL built nodes — both unclaimed and actively claimed.
    This prevents premature phase completion when workers still have
    claimed batches in flight.
    """
    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {"status": "built"}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              {ids_clause}
            RETURN count(p) AS pending
            """,
            categories=list(ENRICHABLE_CATEGORIES),
            **params,
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# IMASNode — embedding claims (enriched → embedded)
# =============================================================================


@retry_on_deadlock()
def claim_paths_for_embedding(limit: int = 500) -> list[dict]:
    """Claim enriched IMASNodes for embedding generation.

    Returns list of dicts with path metadata needed by the embedding
    pipeline.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"

    with GraphClient() as gc:
        # Step 1: Claim enriched data nodes for embedding.
        # Prefer LLM-enriched nodes over template-enriched ones, but include
        # template-enriched nodes that have no embedding (e.g. after
        # --reset-to enriched) to avoid an infinite spin where the worker
        # finds zero claimable paths.
        gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              AND (p.enrichment_source <> 'template' OR p.embedding IS NULL)
              AND (p.claimed_at IS NULL
                   OR p.claimed_at < datetime() - duration($cutoff))
            WITH p
            ORDER BY rand()
            LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            status="enriched",
            categories=list(EMBEDDABLE_CATEGORIES),
            cutoff=cutoff,
            limit=limit,
            token=token,
        )

        # Step 2: Read back
        result = gc.query(
            """
            MATCH (p:IMASNode {claim_token: $token})
            RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
                   p.data_type AS data_type, p.ids AS ids, p.unit AS unit,
                   p.description AS description, p.keywords AS keywords,
                   p.cocos_label_transformation AS cocos_label_transformation,
                   p.physics_domain AS physics_domain,
                   p.node_type AS node_type, p.ndim AS ndim,
                   p.embedding_hash AS embedding_hash
            """,
            token=token,
        )
        claimed = list(result)
        logger.debug(
            "claim_paths_for_embedding: requested %d, won %d",
            limit,
            len(claimed),
        )
        return claimed


def mark_paths_embedded(path_ids: list[str]) -> int:
    """Mark embedded paths: set status=embedded, clear claimed_at."""
    if not path_ids:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            SET p.status = 'embedded',
                p.claimed_at = null,
                p.claim_token = null
            RETURN count(p) AS updated
            """,
            ids=path_ids,
        )
        return result[0]["updated"] if result else 0


def release_embedding_claims(path_ids: list[str]) -> None:
    """Release embedding claims on error."""
    if not path_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            WHERE p.claimed_at IS NOT NULL
            SET p.claimed_at = null, p.claim_token = null
            """,
            ids=path_ids,
        )


def count_imas_nodes_by_status(
    *, node_categories: Collection[str] | None = None
) -> dict[str, int]:
    """Count IMASNode nodes grouped by status.

    Args:
        node_categories: Optional collection of node_category values to include.

    Returns dict mapping status → count, plus a 'total' key.
    """
    filter_cats = list(node_categories) if node_categories is not None else None
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.status IS NOT NULL
              AND ($filter_categories IS NULL OR p.node_category IN $filter_categories)
            RETURN p.status AS status, count(p) AS cnt
            """,
            filter_categories=filter_cats,
        )
        counts: dict[str, int] = {}
        total = 0
        for row in result:
            counts[row["status"]] = row["cnt"]
            total += row["cnt"]
        counts["total"] = total
        return counts


def has_pending_embedding() -> bool:
    """Check if there are enriched IMASNodes awaiting embedding.

    Counts ALL enriched nodes — both unclaimed and actively claimed.
    This prevents premature phase completion when workers still have
    claimed batches in flight (e.g. 4 embed workers each claim 500,
    the 4th goes idle with 1500 still being processed by the other 3).
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
            RETURN count(p) AS pending
            """,
            status="enriched",
            categories=list(EMBEDDABLE_CATEGORIES),
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# Orphan recovery
# =============================================================================


def reset_stale_imas_claims(*, timeout_seconds: int = CLAIM_TIMEOUT_SECONDS) -> int:
    """Release stale claims on IMASNode nodes."""
    cutoff = f"PT{timeout_seconds}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.claimed_at IS NOT NULL
              AND (p.claimed_at < datetime() - duration($cutoff)
                   OR p.claimed_at > datetime())
            SET p.claimed_at = null, p.claim_token = null
            RETURN count(p) AS reset_count
            """,
            cutoff=cutoff,
        )
        count = result[0]["reset_count"] if result else 0
        if count:
            logger.info("Released %d orphaned IMASNode claims", count)
        return count


# =============================================================================
# Status reset
# =============================================================================

# Fields to clear when resetting to each target status
_RESET_CLEAR_FIELDS: dict[str, list[str]] = {
    "built": [
        "description",
        "keywords",
        "enrichment_hash",
        "enrichment_model",
        "enrichment_source",
        "enriched_at",
        "physics_domain",
        "embedding",
        "embedding_hash",
        "embedded_at",
    ],
    "enriched": [
        "embedding",
        "embedding_hash",
        "embedded_at",
    ],
}

# Statuses eligible for reset to each target
_RESET_SOURCE_STATUSES: dict[str, list[str]] = {
    "built": ["enriched", "embedded"],
    "enriched": ["embedded"],
}


def _reset_to_extracted(*, ids_filter: set[str] | None = None) -> int:
    """Delete all IMASNode nodes and clear DDVersion.build_hash.

    This forces a full rebuild from DD XML on the next run.
    """
    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        # Delete IMASNode nodes (DETACH removes relationships too)
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE true {ids_clause}
            DETACH DELETE p
            RETURN count(p) AS deleted_count
            """,
            **params,
        )
        count = result[0]["deleted_count"] if result else 0

        # Clear DDVersion.build_hash to force rebuild
        gc.query("MATCH (d:DDVersion) SET d.build_hash = null")

    logger.info(
        "Deleted %d IMASNode nodes and cleared DDVersion.build_hash",
        count,
    )
    return count


def reset_imas_nodes(
    target_status: str,
    *,
    ids_filter: set[str] | None = None,
) -> int:
    """Reset IMASNode nodes to a target status for reprocessing.

    Args:
        target_status: Target status (``extracted``, ``built``, or ``enriched``).
        ids_filter: Optional set of IDS names to limit the reset.

    Returns:
        Number of nodes affected.
    """
    if target_status == "extracted":
        return _reset_to_extracted(ids_filter=ids_filter)

    if target_status not in _RESET_CLEAR_FIELDS:
        raise ValueError(
            f"Invalid target_status '{target_status}'. "
            f"Must be one of: extracted, {', '.join(_RESET_CLEAR_FIELDS)}"
        )

    clear_fields = _RESET_CLEAR_FIELDS[target_status]
    source_statuses = _RESET_SOURCE_STATUSES[target_status]

    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {"source_statuses": source_statuses, "target": target_status}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    set_parts = [
        "p.status = $target",
        "p.claimed_at = null",
        "p.claim_token = null",
    ]
    for fld in clear_fields:
        set_parts.append(f"p.{fld} = null")
    set_clause = ", ".join(set_parts)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status IN $source_statuses
            {ids_clause}
            SET {set_clause}
            RETURN count(p) AS reset_count
            """,
            **params,
        )
        count = result[0]["reset_count"] if result else 0

    logger.info(
        "Reset %d IMASNode nodes to '%s'",
        count,
        target_status,
    )
    return count
