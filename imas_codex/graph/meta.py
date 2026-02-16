"""Graph identity via GraphMeta node.

Every graph instance contains exactly one ``(:GraphMeta {id: "meta"})`` node
that records what data is in the graph: name, facilities, facilities_hash.

This enables:
- **Identity check**: ``graph status`` shows graph name + facilities
- **Ingestion gating**: before writing TCV data, verify ``"tcv"`` ∈ facilities
- **Graph switching**: multiple ``neo4j-{name}/`` data dirs coexist on disk

The GraphMeta node is created via ``graph init`` and checked on every write.
DD version is NOT stored here — it has its own ``(:DDVersion)`` nodes.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def compute_facilities_hash(facilities: list[str]) -> str:
    """Compute a deterministic hash of the facilities list.

    Sorted, joined, SHA-256, first 12 hex chars.  This lets agents and
    CI detect when the facility set has changed without comparing lists.

    Args:
        facilities: Facility IDs (order-insensitive).

    Returns:
        12-char hex digest (e.g. ``"a3f8c1d09e2b"``).
    """
    canonical = ",".join(sorted(facilities))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def init_graph_meta(
    client: Any,
    name: str,
    facilities: list[str],
) -> dict[str, Any]:
    """Create the GraphMeta singleton node (idempotent).

    Called by ``imas-codex graph init --name <name> --facility <fac> ...``

    Uses ``MERGE`` so it is safe to call repeatedly.  Updates name,
    facilities, and facilities_hash on every call.

    Args:
        client: A :class:`GraphClient` instance (or anything with ``.query()``).
        name: Graph name (e.g. ``"codex"``, ``"dev"``).
        facilities: List of facility IDs this graph contains.

    Returns:
        Dict of the created/updated GraphMeta properties.
    """
    now = datetime.now(UTC).isoformat()
    fhash = compute_facilities_hash(facilities)
    client.query(
        """
        MERGE (m:GraphMeta {id: "meta"})
        ON CREATE SET m.name = $name,
                      m.facilities = $facilities,
                      m.facilities_hash = $fhash,
                      m.created_at = $now,
                      m.updated_at = $now
        ON MATCH SET  m.name = $name,
                      m.facilities = $facilities,
                      m.facilities_hash = $fhash,
                      m.updated_at = $now
        """,
        name=name,
        facilities=facilities,
        fhash=fhash,
        now=now,
    )
    logger.info("GraphMeta: name=%s, facilities=%s, hash=%s", name, facilities, fhash)
    return {"name": name, "facilities": facilities, "facilities_hash": fhash}


# Backward-compat alias
ensure_graph_meta = init_graph_meta


def get_graph_meta(client: Any) -> dict[str, Any] | None:
    """Read the GraphMeta node from the active graph.

    Returns:
        Dict with ``name``, ``facilities``, ``facilities_hash``, etc.
        or ``None`` if no meta node exists yet (run ``graph init`` first).
    """
    result = client.query(
        "MATCH (m:GraphMeta {id: 'meta'}) "
        "RETURN m.name AS name, m.facilities AS facilities, "
        "       m.facilities_hash AS facilities_hash, "
        "       m.created_at AS created_at, m.updated_at AS updated_at"
    )
    if result:
        return dict(result[0])
    return None


def gate_ingestion(client: Any, facility_id: str) -> None:
    """Guard writes: raise if *facility_id* is not in GraphMeta.facilities.

    Prevents accidentally ingesting data into a graph that doesn't include
    the target facility.

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility about to be written (e.g. ``"tcv"``).

    Raises:
        ValueError: If the graph exists but *facility_id* is not in its
            ``facilities`` list.
    """
    meta = get_graph_meta(client)
    if meta is None:
        return  # No meta yet — allow first write
    facilities = meta.get("facilities") or []
    if facility_id not in facilities:
        raise ValueError(
            f"Graph '{meta.get('name', '?')}' does not include facility "
            f"'{facility_id}'. Allowed: {facilities}. "
            f"Add it first: imas-codex graph facility add {facility_id}"
        )


def add_facility_to_meta(client: Any, facility_id: str) -> None:
    """Append a facility to the GraphMeta.facilities list (idempotent).

    Recomputes ``facilities_hash`` after adding.

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to add.
    """
    meta = get_graph_meta(client)
    if meta is None:
        logger.warning("No GraphMeta node — run 'graph init' first")
        return

    facilities = list(meta.get("facilities") or [])
    if facility_id in facilities:
        logger.info("Facility '%s' already in GraphMeta", facility_id)
        return

    facilities.append(facility_id)
    fhash = compute_facilities_hash(facilities)
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = $facilities,
            m.facilities_hash = $fhash,
            m.updated_at = $now
        """,
        facilities=facilities,
        fhash=fhash,
        now=now,
    )
    logger.info("Added facility '%s' to GraphMeta (hash=%s)", facility_id, fhash)


def remove_facility_from_meta(client: Any, facility_id: str) -> None:
    """Remove a facility from the GraphMeta.facilities list.

    Recomputes ``facilities_hash`` after removing.

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to remove.
    """
    meta = get_graph_meta(client)
    if meta is None:
        logger.warning("No GraphMeta node — run 'graph init' first")
        return

    facilities = [f for f in (meta.get("facilities") or []) if f != facility_id]
    fhash = compute_facilities_hash(facilities)
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = $facilities,
            m.facilities_hash = $fhash,
            m.updated_at = $now
        """,
        facilities=facilities,
        fhash=fhash,
        now=now,
    )
    logger.info("Removed facility '%s' from GraphMeta (hash=%s)", facility_id, fhash)


__all__ = [
    "add_facility_to_meta",
    "compute_facilities_hash",
    "ensure_graph_meta",
    "gate_ingestion",
    "get_graph_meta",
    "init_graph_meta",
    "remove_facility_from_meta",
]
