"""Graph identity via GraphMeta node.

Every graph instance contains exactly one ``(:GraphMeta {id: "meta"})`` node
that records what data is in the graph: name and facilities.

This enables:
- **Identity check**: ``graph status`` shows graph name + facilities
- **Ingestion gating**: before writing TCV data, verify ``"tcv"`` ∈ facilities

The GraphMeta node is created via ``graph init`` and checked on every write.
DD version is NOT stored here — it has its own ``(:DDVersion)`` nodes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def init_graph_meta(
    client: Any,
    name: str,
    facilities: list[str],
    *,
    imas: bool = True,
) -> dict[str, Any]:
    """Create the GraphMeta singleton node (idempotent).

    Uses ``MERGE`` so it is safe to call repeatedly.  Updates name,
    facilities, and imas on every call.

    Args:
        client: A :class:`GraphClient` instance (or anything with ``.query()``).
        name: Graph name (e.g. ``"codex"``).
        facilities: List of facility IDs this graph contains.
        imas: Whether this graph includes IMAS DD data (default True).

    Returns:
        Dict of the created/updated GraphMeta properties.
    """
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MERGE (m:GraphMeta {id: "meta"})
        ON CREATE SET m.name = $name,
                      m.facilities = $facilities,
                      m.imas = $imas,
                      m.created_at = $now,
                      m.updated_at = $now
        ON MATCH SET  m.name = $name,
                      m.facilities = $facilities,
                      m.imas = $imas,
                      m.updated_at = $now
        """,
        name=name,
        facilities=facilities,
        imas=imas,
        now=now,
    )
    logger.info("GraphMeta: name=%s, facilities=%s, imas=%s", name, facilities, imas)
    return {"name": name, "facilities": facilities, "imas": imas}


def get_graph_meta(client: Any) -> dict[str, Any] | None:
    """Read the GraphMeta node from the active graph.

    Returns:
        Dict with ``name``, ``facilities``, etc.
        or ``None`` if no meta node exists yet (run ``graph init`` first).
    """
    result = client.query(
        "MATCH (m:GraphMeta {id: 'meta'}) "
        "RETURN m.name AS name, m.facilities AS facilities, "
        "       m.imas AS imas, "
        "       m.created_at AS created_at, m.updated_at AS updated_at"
    )
    if result:
        return dict(result[0])
    return None


def gate_ingestion(client: Any, facility_id: str) -> None:
    """Guard writes: raise if *facility_id* is not in GraphMeta.facilities.

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
    """Append a facility to the GraphMeta.facilities list and ensure the
    Facility node exists in the graph (idempotent).

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to add.
    """
    meta = get_graph_meta(client)
    if meta is None:
        logger.warning("No GraphMeta node — run 'graph init' first")
        return

    facilities = list(meta.get("facilities") or [])
    if facility_id not in facilities:
        facilities.append(facility_id)
        now = datetime.now(UTC).isoformat()
        client.query(
            """
            MATCH (m:GraphMeta {id: "meta"})
            SET m.facilities = $facilities,
                m.updated_at = $now
            """,
            facilities=facilities,
            now=now,
        )
        logger.info("Added facility '%s' to GraphMeta", facility_id)
    else:
        logger.info("Facility '%s' already in GraphMeta", facility_id)

    # Ensure the Facility node exists in the graph
    client.query(
        """
        MERGE (f:Facility {id: $id})
        ON CREATE SET f.name = $id, f.created_at = datetime()
        """,
        id=facility_id,
    )
    logger.info("Ensured Facility node '%s' exists", facility_id)


def remove_facility_from_meta(client: Any, facility_id: str) -> None:
    """Remove a facility from the GraphMeta.facilities list and delete the
    Facility node from the graph.

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to remove.
    """
    meta = get_graph_meta(client)
    if meta is None:
        logger.warning("No GraphMeta node — run 'graph init' first")
        return

    facilities = [f for f in (meta.get("facilities") or []) if f != facility_id]
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = $facilities,
            m.updated_at = $now
        """,
        facilities=facilities,
        now=now,
    )
    logger.info("Removed facility '%s' from GraphMeta", facility_id)

    # DETACH DELETE the Facility node and clean up orphans
    result = client.query(
        "MATCH (f:Facility {id: $id}) DETACH DELETE f RETURN count(f) AS deleted",
        id=facility_id,
    )
    deleted = result[0]["deleted"] if result else 0
    if deleted:
        logger.info("Deleted Facility node '%s' (detached)", facility_id)

    # Remove nodes that were exclusively linked to this facility
    orphan_result = client.query(
        "MATCH (n) "
        "WHERE n.facility_id = $fid "
        "AND NOT (n)--() "
        "DELETE n "
        "RETURN count(n) AS deleted",
        fid=facility_id,
    )
    orphans = orphan_result[0]["deleted"] if orphan_result else 0
    if orphans:
        logger.info(
            "Cleaned up %d orphan nodes for facility '%s'", orphans, facility_id
        )


def check_pull_compatibility(
    meta: dict[str, Any],
    *,
    imas_only: bool = False,
    no_imas: bool = False,
    facilities: list[str] | None = None,
) -> list[str]:
    """Check whether a pull operation is compatible with the active graph.

    Compares the pull flags (which determine the GHCR package) against
    the active graph's GraphMeta to detect mismatches.

    Args:
        meta: GraphMeta dict from :func:`get_graph_meta`.
        imas_only: Whether pulling the IMAS-only package.
        no_imas: Whether pulling the no-imas variant.
        facilities: Facility filter for the pull.

    Returns:
        List of warning strings. Empty if compatible.
    """
    warnings = []
    graph_name = meta.get("name", "?")
    graph_facilities = set(meta.get("facilities") or [])
    graph_imas = meta.get("imas", True)  # default True for pre-existing graphs

    if imas_only:
        # Pulling DD-only into a graph that has facilities
        if graph_facilities:
            warnings.append(
                f"Pulling IMAS-only package into graph '{graph_name}' "
                f"which has facilities {sorted(graph_facilities)}. "
                f"This will replace all data including facility data."
            )
    elif facilities:
        pull_set = set(facilities)
        # Pulling facility package into wrong graph
        if not pull_set.issubset(graph_facilities):
            missing = pull_set - graph_facilities
            warnings.append(
                f"Pulling facility package for {sorted(pull_set)} but "
                f"graph '{graph_name}' does not include "
                f"{sorted(missing)}. Allowed: {sorted(graph_facilities)}."
            )
    else:
        # Pulling full graph
        if no_imas and graph_imas:
            warnings.append(
                f"Pulling no-imas variant into graph '{graph_name}' "
                f"which has imas=true. DD data will be lost."
            )
        elif not no_imas and not graph_imas:
            warnings.append(
                f"Pulling package with IMAS DD into graph '{graph_name}' "
                f"which has imas=false."
            )

    return warnings


__all__ = [
    "add_facility_to_meta",
    "check_pull_compatibility",
    "gate_ingestion",
    "get_graph_meta",
    "init_graph_meta",
    "remove_facility_from_meta",
]
