"""Graph identity via GraphMeta node.

Every graph instance contains exactly one ``(:GraphMeta {id: "meta"})`` node
that records what data is in the graph: name, facilities, hash.

This enables:
- **Identity check**: ``graph status`` shows graph name + facilities
- **Ingestion gating**: before writing TCV data, verify ``"tcv"`` ∈ facilities
- **Graph switching**: hash-named dirs in ``.neo4j/`` with symlink selection

The GraphMeta node is created via ``graph init`` and checked on every write.
DD version is NOT stored here — it has its own ``(:DDVersion)`` nodes.

The ``hash`` property combines graph name + sorted facilities via SHA-256
(see :func:`imas_codex.graph.dirs.compute_graph_hash`).  The ``"imas"``
pseudo-facility is included when the Data Dictionary is present.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from imas_codex.graph.dirs import compute_graph_hash

logger = logging.getLogger(__name__)


def init_graph_meta(
    client: Any,
    name: str,
    facilities: list[str],
) -> dict[str, Any]:
    """Create the GraphMeta singleton node (idempotent).

    Called by ``imas-codex graph init --name <name> --facility <fac> ...``

    Uses ``MERGE`` so it is safe to call repeatedly.  Updates name,
    facilities, and hash on every call.

    Args:
        client: A :class:`GraphClient` instance (or anything with ``.query()``).
        name: Graph name (e.g. ``"codex"``, ``"dev"``).
        facilities: List of facility IDs this graph contains.
            Include ``"imas"`` when the Data Dictionary is present.

    Returns:
        Dict of the created/updated GraphMeta properties.
    """
    now = datetime.now(UTC).isoformat()
    graph_hash = compute_graph_hash(name, facilities)
    client.query(
        """
        MERGE (m:GraphMeta {id: "meta"})
        ON CREATE SET m.name = $name,
                      m.facilities = $facilities,
                      m.hash = $hash,
                      m.created_at = $now,
                      m.updated_at = $now
        ON MATCH SET  m.name = $name,
                      m.facilities = $facilities,
                      m.hash = $hash,
                      m.updated_at = $now
        """,
        name=name,
        facilities=facilities,
        hash=graph_hash,
        now=now,
    )
    logger.info(
        "GraphMeta: name=%s, facilities=%s, hash=%s", name, facilities, graph_hash
    )
    return {"name": name, "facilities": facilities, "hash": graph_hash}


def get_graph_meta(client: Any) -> dict[str, Any] | None:
    """Read the GraphMeta node from the active graph.

    Returns:
        Dict with ``name``, ``facilities``, ``hash``, etc.
        or ``None`` if no meta node exists yet (run ``graph init`` first).
    """
    result = client.query(
        "MATCH (m:GraphMeta {id: 'meta'}) "
        "RETURN m.name AS name, m.facilities AS facilities, "
        "       m.hash AS hash, "
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

    Recomputes ``hash`` after adding.

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
    name = meta.get("name", "")
    graph_hash = compute_graph_hash(name, facilities)
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = $facilities,
            m.hash = $hash,
            m.updated_at = $now
        """,
        facilities=facilities,
        hash=graph_hash,
        now=now,
    )
    logger.info("Added facility '%s' to GraphMeta (hash=%s)", facility_id, graph_hash)


def remove_facility_from_meta(client: Any, facility_id: str) -> None:
    """Remove a facility from the GraphMeta.facilities list.

    Recomputes ``hash`` after removing.

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to remove.
    """
    meta = get_graph_meta(client)
    if meta is None:
        logger.warning("No GraphMeta node — run 'graph init' first")
        return

    facilities = [f for f in (meta.get("facilities") or []) if f != facility_id]
    name = meta.get("name", "")
    graph_hash = compute_graph_hash(name, facilities)
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = $facilities,
            m.hash = $hash,
            m.updated_at = $now
        """,
        facilities=facilities,
        hash=graph_hash,
        now=now,
    )
    logger.info(
        "Removed facility '%s' from GraphMeta (hash=%s)", facility_id, graph_hash
    )


__all__ = [
    "add_facility_to_meta",
    "gate_ingestion",
    "get_graph_meta",
    "init_graph_meta",
    "remove_facility_from_meta",
]
