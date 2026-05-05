"""Stage-aware orphan recovery for the SN refine pipeline.

Periodically reverts StandardName nodes stuck in transient stages (e.g.
'refining') that have stale claimed_at timestamps. Mirrors the discovery
CLI orphan-sweep pattern but applies stage-aware predecessor reverts.

Predecessor-stage mapping
-------------------------
``name_stage='refining'``  →  revert to ``'reviewed'``
``docs_stage='refining'``  →  revert to ``'reviewed'``

For defense in depth, also clear stale ``claim_token``/``claimed_at``
on non-refining StandardName and StandardNameSource nodes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep queries — (label, cypher) pairs. Each RETURN alias is ``n``.
# ---------------------------------------------------------------------------

_SWEEP_QUERIES: Final[list[tuple[str, str]]] = [
    (
        "name_refining",
        """
        MATCH (sn:StandardName)
        WHERE sn.name_stage = 'refining'
          AND sn.claimed_at IS NOT NULL
          AND sn.claimed_at < datetime() - duration({seconds: $timeout_s})
        SET sn.name_stage = 'reviewed',
            sn.claim_token = null,
            sn.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        "docs_refining",
        """
        MATCH (sn:StandardName)
        WHERE sn.docs_stage = 'refining'
          AND sn.claimed_at IS NOT NULL
          AND sn.claimed_at < datetime() - duration({seconds: $timeout_s})
        SET sn.docs_stage  = 'reviewed',
            sn.claim_token = null,
            sn.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        "stale_token_sn",
        """
        MATCH (sn:StandardName)
        WHERE sn.claim_token IS NOT NULL
          AND sn.claimed_at IS NOT NULL
          AND sn.claimed_at < datetime() - duration({seconds: $timeout_s})
          AND NOT sn.name_stage = 'refining'
          AND NOT sn.docs_stage = 'refining'
        SET sn.claim_token = null,
            sn.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        "stale_token_source",
        """
        MATCH (s:StandardNameSource)
        WHERE s.claim_token IS NOT NULL
          AND s.claimed_at IS NOT NULL
          AND s.claimed_at < datetime() - duration({seconds: $timeout_s})
        SET s.claim_token = null,
            s.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _orphan_sweep_tick(*, timeout_s: int) -> dict[str, int]:
    """Run one full orphan-sweep pass (synchronous).

    Executes all four sweep queries in separate transactions and returns
    a mapping of sweep label → number of nodes reverted/cleared.

    Args:
        timeout_s: Age threshold in seconds.  Nodes whose ``claimed_at``
            is older than ``timeout_s`` seconds are considered orphaned.

    Returns:
        ``{"name_refining": n, "docs_refining": n, "stale_token_sn": n,
        "stale_token_source": n}``
    """
    counts: dict[str, int] = {}
    with GraphClient() as gc:
        for label, cypher in _SWEEP_QUERIES:
            rows = gc.query(cypher, timeout_s=timeout_s)
            counts[label] = rows[0]["n"] if rows else 0
    return counts


async def run_orphan_sweep_loop(
    *,
    interval_s: int,
    timeout_s: int,
    stop_event: asyncio.Event,
) -> None:
    """Background coroutine that periodically reverts orphaned claims.

    Loops every *interval_s* seconds, delegating the actual DB work to
    :func:`_orphan_sweep_tick` via ``asyncio.to_thread``.  Exits cleanly
    when *stop_event* is set (checked both before each tick and during the
    sleep phase).

    Args:
        interval_s: How often to run a sweep pass, in seconds.
        timeout_s: Age threshold passed to :func:`_orphan_sweep_tick`.
        stop_event: Cooperative shutdown signal; shared with worker pools.
    """
    logger.info(
        "Orphan sweep loop started (interval=%ds, timeout=%ds)",
        interval_s,
        timeout_s,
    )

    while not stop_event.is_set():
        try:
            counts = await asyncio.to_thread(
                _orphan_sweep_tick,
                timeout_s=timeout_s,
            )
            total = sum(counts.values())
            if total:
                logger.warning(
                    "Orphan sweep reverted %d claims: %s",
                    total,
                    counts,
                )
            else:
                logger.debug("Orphan sweep: no stuck claims found")
        except Exception:  # noqa: BLE001
            logger.exception("Orphan sweep tick failed; continuing")

        # Seed parent component sources for newly tagged parents
        try:
            from imas_codex.standard_names.graph_ops import seed_parent_sources

            parent_count = await asyncio.to_thread(seed_parent_sources)
            if parent_count:
                logger.info("Seeded %d parent component sources", parent_count)
        except Exception:  # noqa: BLE001
            logger.exception("Parent source seeding failed; continuing")

        # Sleep for interval_s, but wake early if stop_event fires.
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=interval_s,
            )
        except TimeoutError:
            pass  # Normal path — interval elapsed, loop again.

    logger.info("Orphan sweep loop stopped")
