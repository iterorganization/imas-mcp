"""Graph identity via GraphMeta node.

Every graph instance contains exactly one ``(:GraphMeta {id: "meta"})`` node
that records what data is in the graph: name, facilities, DD version.

This enables:
- **Identity check**: ``graph db status`` shows graph name + facilities
- **Ingestion gating**: before writing TCV data, verify ``"tcv"`` ∈ facilities
- **Drift detection**: warn if config name doesn't match stored meta.name

The GraphMeta node is created on first write and checked on every connection.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def ensure_graph_meta(
    client: Any,
    name: str,
    facilities: list[str],
    dd_version: str | None = None,
) -> None:
    """Create or update the GraphMeta singleton node.

    Uses ``MERGE`` so it is safe to call repeatedly (idempotent).

    Args:
        client: A :class:`GraphClient` instance (or anything with ``.query()``).
        name: Graph name (e.g. ``"codex"``, ``"tcv"``).
        facilities: List of facility IDs this graph contains.
        dd_version: Data-dictionary version the graph was built against.
    """
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MERGE (m:GraphMeta {id: "meta"})
        ON CREATE SET m.name = $name,
                      m.facilities = $facilities,
                      m.dd_version = $dd_version,
                      m.created_at = $now,
                      m.updated_at = $now
        ON MATCH SET  m.name = $name,
                      m.facilities = $facilities,
                      m.dd_version = coalesce($dd_version, m.dd_version),
                      m.updated_at = $now
        """,
        name=name,
        facilities=facilities,
        dd_version=dd_version,
        now=now,
    )
    logger.debug("GraphMeta: name=%s, facilities=%s", name, facilities)


def get_graph_meta(client: Any) -> dict[str, Any] | None:
    """Read the GraphMeta node from the active graph.

    Returns:
        Dict with ``name``, ``facilities``, ``dd_version``, etc. or ``None``
        if no meta node exists yet (first-use scenario).
    """
    result = client.query(
        "MATCH (m:GraphMeta {id: 'meta'}) "
        "RETURN m.name AS name, m.facilities AS facilities, "
        "       m.dd_version AS dd_version, "
        "       m.created_at AS created_at, m.updated_at AS updated_at"
    )
    if result:
        return dict(result[0])
    return None


def check_graph_identity(client: Any, expected_name: str) -> str | None:
    """Return a warning if the graph name doesn't match, ``None`` if OK.

    A mismatch suggests the wrong graph is loaded (e.g. you expected
    ``"codex"`` but the data was swapped to ``"tcv"``).
    """
    meta = get_graph_meta(client)
    if meta is None:
        return None  # No meta yet — first use, not an error
    actual = meta.get("name")
    if actual and actual != expected_name:
        return f"Graph identity mismatch: expected '{expected_name}', found '{actual}'"
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
            f"Add it first: imas-codex graph meta add-facility {facility_id}"
        )


def add_facility_to_meta(client: Any, facility_id: str) -> None:
    """Append a facility to the GraphMeta.facilities list (idempotent).

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to add.
    """
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = CASE
            WHEN NOT $fid IN m.facilities
            THEN m.facilities + $fid
            ELSE m.facilities
        END,
        m.updated_at = $now
        """,
        fid=facility_id,
        now=now,
    )
    logger.info("Added facility '%s' to GraphMeta", facility_id)


def remove_facility_from_meta(client: Any, facility_id: str) -> None:
    """Remove a facility from the GraphMeta.facilities list.

    Args:
        client: A :class:`GraphClient` instance.
        facility_id: Facility ID to remove.
    """
    now = datetime.now(UTC).isoformat()
    client.query(
        """
        MATCH (m:GraphMeta {id: "meta"})
        SET m.facilities = [f IN m.facilities WHERE f <> $fid],
            m.updated_at = $now
        """,
        fid=facility_id,
        now=now,
    )
    logger.info("Removed facility '%s' from GraphMeta", facility_id)


__all__ = [
    "add_facility_to_meta",
    "check_graph_identity",
    "ensure_graph_meta",
    "gate_ingestion",
    "get_graph_meta",
    "remove_facility_from_meta",
]
