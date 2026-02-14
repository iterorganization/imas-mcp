"""Common claim coordination for parallel discovery engines.

All discovery modules (paths, wiki, signals, files) use the same claim pattern:

1. Atomically SET claimed_at = datetime() on unclaimed/stale nodes
2. Process the claimed nodes
3. On completion: update status (claimed_at cleared implicitly or explicitly)
4. On error: release claim via SET claimed_at = null

Stale claims (older than timeout) are automatically recovered by other workers,
making the system safe for parallel execution across CLI instances.

Usage::

    from imas_codex.discovery.base.claims import (
        DEFAULT_CLAIM_TIMEOUT_SECONDS,
        release_claim,
        release_claims_batch,
        reset_stale_claims,
    )

    # Reset stale claims at startup
    reset_stale_claims("SourceFile", facility, timeout_seconds=300)

    # Release on error
    release_claim("SourceFile", node_id)
    release_claims_batch("FacilityPath", [id1, id2])
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes


def reset_stale_claims(
    label: str,
    facility: str,
    *,
    timeout_seconds: int = DEFAULT_CLAIM_TIMEOUT_SECONDS,
    facility_field: str = "facility_id",
    claimed_field: str = "claimed_at",
    silent: bool = False,
) -> int:
    """Release claims older than timeout_seconds for a node type.

    Uses timeout-based recovery so multiple CLI instances can run
    concurrently without wiping each other's active claims.

    Args:
        label: Node label (e.g., ``"SourceFile"``, ``"FacilityPath"``)
        facility: Facility ID
        timeout_seconds: Age threshold for orphaned claims
        facility_field: Property containing facility ID
        claimed_field: Property name for the claim timestamp
        silent: Suppress logging

    Returns:
        Number of claims released
    """
    from imas_codex.graph import GraphClient

    cutoff = f"PT{timeout_seconds}S"
    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (n:{label} {{{facility_field}: $facility}})
            WHERE n.{claimed_field} IS NOT NULL
              AND n.{claimed_field} < datetime() - duration($cutoff)
            SET n.{claimed_field} = null
            RETURN count(n) AS reset_count
            """,
            facility=facility,
            cutoff=cutoff,
        )
        count = result[0]["reset_count"] if result else 0

    if count and not silent:
        logger.info(
            "Released %d orphaned %s claims older than %ds for %s",
            count,
            label,
            timeout_seconds,
            facility,
        )

    return count


def release_claim(label: str, node_id: str) -> None:
    """Release claim on a single node by clearing ``claimed_at``.

    Args:
        label: Node label (e.g., ``"SourceFile"``)
        node_id: Node ID to release
    """
    from imas_codex.graph import GraphClient

    try:
        with GraphClient() as gc:
            gc.query(
                f"""
                MATCH (n:{label} {{id: $id}})
                SET n.claimed_at = null
                """,
                id=node_id,
            )
    except Exception as e:
        logger.warning("Failed to release %s claim for %s: %s", label, node_id, e)


def release_claims_batch(label: str, node_ids: list[str]) -> int:
    """Release claims on multiple nodes by clearing ``claimed_at``.

    Args:
        label: Node label (e.g., ``"SourceFile"``)
        node_ids: Node IDs to release

    Returns:
        Number of claims released
    """
    from imas_codex.graph import GraphClient

    if not node_ids:
        return 0

    try:
        with GraphClient() as gc:
            result = gc.query(
                f"""
                UNWIND $ids AS nid
                MATCH (n:{label} {{id: nid}})
                WHERE n.claimed_at IS NOT NULL
                SET n.claimed_at = null
                RETURN count(n) AS released
                """,
                ids=node_ids,
            )
            return result[0]["released"] if result else 0
    except Exception as e:
        logger.warning("Failed to release %s claims: %s", label, e)
        return 0
