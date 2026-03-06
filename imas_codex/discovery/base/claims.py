"""Common claim coordination for parallel discovery engines.

All discovery modules (paths, wiki, signals, files) use the same claim pattern:

1. Atomically SET claimed_at = datetime() on unclaimed/stale nodes
2. Process the claimed nodes
3. On completion: update status (claimed_at cleared implicitly or explicitly)
4. On error: release claim via SET claimed_at = null

Stale claims (older than timeout) are automatically recovered by other workers,
making the system safe for parallel execution across CLI instances.

Anti-deadlock pattern for claim functions:
- ORDER BY rand() to avoid deterministic lock ordering collisions
- claim_token (UUID) two-step verify to handle race conditions
- @retry_on_deadlock decorator for transient Neo4j deadlock errors

Usage::

    from imas_codex.discovery.base.claims import (
        DEFAULT_CLAIM_TIMEOUT_SECONDS,
        release_claim,
        release_claims_batch,
        reset_stale_claims,
        retry_on_deadlock,
    )

    @retry_on_deadlock()
    def claim_items(facility, limit=10):
        token = str(uuid.uuid4())
        with GraphClient() as gc:
            gc.query("... ORDER BY rand() LIMIT $limit SET n.claim_token = $token ...", ...)
            return list(gc.query("MATCH (n {claim_token: $token}) RETURN ...", ...))
"""

from __future__ import annotations

import functools
import logging
import random
import time

from neo4j.exceptions import TransientError

logger = logging.getLogger(__name__)

DEFAULT_CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes

# Retry configuration for Neo4j transient errors (deadlocks)
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.1  # seconds
RETRY_MAX_DELAY = 2.0  # seconds


def retry_on_deadlock(
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
):
    """Decorator to retry functions on Neo4j transient errors (e.g., deadlocks).

    Uses exponential backoff with jitter to reduce contention.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.5)
                        sleep_time = delay + jitter
                        logger.debug(
                            "%s: transient error (attempt %d/%d), "
                            "retrying in %.2fs: %s",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            sleep_time,
                            e,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.warning(
                            "%s: transient error after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            e,
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


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
        label: Node label (e.g., ``"CodeFile"``, ``"FacilityPath"``)
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
              AND (n.{claimed_field} < datetime() - duration($cutoff)
                   OR n.{claimed_field} > datetime())
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
        label: Node label (e.g., ``"CodeFile"``)
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
        label: Node label (e.g., ``"CodeFile"``)
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
