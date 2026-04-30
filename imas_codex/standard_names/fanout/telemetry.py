"""Fanout telemetry node writer (plan 39 §8.2).

The :class:`Fanout` graph node is a **runtime telemetry** node — *not*
a LinkML-schema-managed entity.  It is in the same spirit as the
existing ``LLMCost`` node: written directly via Cypher with no Pydantic
model, no auto-generation, and no entry in
``agents/schema-reference.md``.  The plan declares this exemption
explicitly (§8.2) so a future schema-compliance check does not flag it
as drift.

The :data:`Fanout.id` (a uuid4) is also stamped onto each
:class:`LLMCostEvent.batch_id` emitted during the run, enabling the
join::

    MATCH (f:Fanout {id: $run_id})
    MATCH (c:LLMCost {batch_id: f.id})
    RETURN f, collect(c) AS charges
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from .schemas import FanoutOutcome

logger = logging.getLogger(__name__)


_CYPHER_CREATE_FANOUT = """
CREATE (f:Fanout {
    id: $id,
    sn_id: $sn_id,
    site: $site,
    outcome: $outcome,
    plan_size: $plan_size,
    hits_count: $hits_count,
    evidence_tokens: $evidence_tokens,
    arm: $arm,
    escalate: $escalate,
    created_at: datetime($created_at)
})
RETURN f.id AS id
"""


def write_fanout_node(
    gc: Any,
    *,
    run_id: str,
    sn_id: str,
    site: str,
    outcome: FanoutOutcome,
    plan_size: int,
    hits_count: int,
    evidence_tokens: int,
    arm: str = "on",
    escalate: bool = False,
    created_at: datetime | None = None,
) -> str | None:
    """Persist a single :class:`Fanout` runtime-telemetry node.

    Returns the written node id on success, or ``None`` if the write
    silently failed (Neo4j unavailable, transient error).  Telemetry
    must never break the parent refine cycle, so failures are logged
    at INFO level only.

    Skipped entirely when ``gc`` is ``None`` — used by the
    ``feature_disabled`` true-no-op path (plan 39 §7.2).
    """
    if gc is None:
        return None

    if created_at is None:
        created_at = datetime.now(UTC)
    created_iso = (
        created_at.isoformat() if not isinstance(created_at, str) else created_at
    )

    params = {
        "id": run_id,
        "sn_id": sn_id,
        "site": site,
        "outcome": outcome,
        "plan_size": int(plan_size),
        "hits_count": int(hits_count),
        "evidence_tokens": int(evidence_tokens),
        "arm": arm,
        "escalate": bool(escalate),
        "created_at": created_iso,
    }
    try:
        rows = gc.query(_CYPHER_CREATE_FANOUT, **params) or []
        if rows:
            return str(rows[0].get("id") or run_id)
        return run_id
    except Exception as e:
        logger.info(
            "Fanout telemetry write failed (run_id=%s): %s", run_id, e, exc_info=True
        )
        return None


__all__ = ["write_fanout_node"]
