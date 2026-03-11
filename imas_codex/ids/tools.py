"""Tool functions for the IMAS mapping pipeline.

Plain functions (not MCP tools) that the pipeline orchestrator calls
to gather context for the LLM at each step. Each wraps existing graph
queries and utilities with a simple interface.
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IDS structure exploration
# ---------------------------------------------------------------------------


def fetch_imas_subtree(
    ids_name: str,
    path: str | None = None,
    *,
    gc: GraphClient | None = None,
    leaf_only: bool = False,
    max_paths: int | None = None,
) -> list[dict[str, Any]]:
    """Return IDS tree structure under *path* (or IDS root).

    Each row contains ``id``, ``name``, ``data_type``, ``node_type``,
    ``documentation``, ``units``.
    """
    if gc is None:
        gc = GraphClient()

    prefix = f"{ids_name}/{path}" if path else ids_name
    leaf_filter = (
        "AND NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']" if leaf_only else ""
    )
    limit_clause = f"LIMIT {max_paths}" if max_paths else ""

    # When no slash in prefix, match by ids field
    if "/" not in prefix:
        cypher = f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name
            {leaf_filter}
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            RETURN p.id AS id, p.name AS name, p.data_type AS data_type,
                   p.node_type AS node_type, p.documentation AS documentation,
                   u.symbol AS units
            ORDER BY p.id
            {limit_clause}
        """
        return gc.query(cypher, ids_name=ids_name)

    cypher = f"""
        MATCH (p:IMASNode)
        WHERE p.id STARTS WITH $prefix
        {leaf_filter}
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        RETURN p.id AS id, p.name AS name, p.data_type AS data_type,
               p.node_type AS node_type, p.documentation AS documentation,
               u.symbol AS units
        ORDER BY p.id
        {limit_clause}
    """
    return gc.query(cypher, prefix=prefix + "/")


def fetch_imas_fields(
    ids_name: str,
    paths: list[str],
    *,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Return detailed field info for specific IMAS paths.

    Includes documentation, data_type, units, coordinates, ndim,
    physics_domain, and cluster labels.
    """
    if gc is None:
        gc = GraphClient()

    # Qualify paths with ids prefix if missing
    qualified = [
        p if p.startswith(f"{ids_name}/") else f"{ids_name}/{p}" for p in paths
    ]

    cypher = """
        UNWIND $paths AS pid
        MATCH (p:IMASNode {id: pid})
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (p)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
        RETURN p.id AS id, p.name AS name, p.ids AS ids,
               p.documentation AS documentation, p.data_type AS data_type,
               p.node_type AS node_type, p.ndim AS ndim,
               p.physics_domain AS physics_domain,
               u.symbol AS units,
               collect(DISTINCT c.label) AS cluster_labels,
               collect(DISTINCT coord.id) AS coordinates
    """
    return gc.query(cypher, paths=qualified)


def search_imas_semantic(
    query: str,
    ids_name: str | None = None,
    *,
    gc: GraphClient | None = None,
    k: int = 20,
) -> list[dict[str, Any]]:
    """Semantic search for IMAS paths using vector index.

    Returns paths ranked by embedding similarity, optionally filtered
    to a specific IDS.
    """
    if gc is None:
        gc = GraphClient()

    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder

    encoder = Encoder(EncoderConfig())
    result = encoder.embed_texts([query])[0]
    embedding = result.tolist() if hasattr(result, "tolist") else list(result)

    filter_clause = ""
    params: dict[str, Any] = {
        "embedding": embedding,
        "k": min(k * 5, 500),
        "limit": k,
    }
    if ids_name:
        filter_clause = "AND path.ids = $ids_name"
        params["ids_name"] = ids_name

    cypher = f"""
        CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
        YIELD node AS path, score
        WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
        {filter_clause}
        OPTIONAL MATCH (path)-[:HAS_UNIT]->(u:Unit)
        RETURN path.id AS id, path.documentation AS documentation,
               path.data_type AS data_type, path.node_type AS node_type,
               u.symbol AS units, score
        ORDER BY score DESC
        LIMIT $limit
    """
    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# COCOS + Units
# ---------------------------------------------------------------------------


def get_sign_flip_paths(ids_name: str) -> list[str]:
    """Return IMAS paths requiring COCOS sign flip for *ids_name*."""
    from imas_codex.cocos.transforms import get_sign_flip_paths as _get

    return _get(ids_name)


def analyze_units(
    signal_unit: str | None,
    imas_unit: str | None,
) -> dict[str, Any]:
    """Analyse unit compatibility between a signal and an IMAS field.

    Returns a dict with keys:
        compatible (bool), conversion_factor (float|None),
        signal_unit, imas_unit, error (str|None).
    """
    result: dict[str, Any] = {
        "signal_unit": signal_unit,
        "imas_unit": imas_unit,
        "compatible": False,
        "conversion_factor": None,
        "error": None,
    }
    if not signal_unit or not imas_unit:
        result["compatible"] = signal_unit == imas_unit  # both None → compatible
        return result

    try:
        from imas_codex.units import unit_registry

        q_sig = unit_registry.Quantity(1.0, signal_unit)
        q_imas = unit_registry.Quantity(1.0, imas_unit)
        if q_sig.dimensionality == q_imas.dimensionality:
            result["compatible"] = True
            result["conversion_factor"] = q_sig.to(imas_unit).magnitude
        else:
            result["error"] = (
                f"Incompatible dimensions: {q_sig.dimensionality} vs "
                f"{q_imas.dimensionality}"
            )
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def check_imas_paths(
    paths: list[str],
    *,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Validate that IMAS paths exist in the graph.

    Returns a list of dicts with ``path``, ``exists``, ``data_type``,
    ``units``, and optionally ``suggestion`` if renamed.
    """
    if gc is None:
        gc = GraphClient()

    results: list[dict[str, Any]] = []
    for path in paths:
        row = gc.query(
            """
            MATCH (p:IMASNode {id: $path})
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            RETURN p.id AS id, p.data_type AS data_type, u.symbol AS units
            """,
            path=path,
        )
        if row:
            results.append(
                {
                    "path": row[0]["id"],
                    "exists": True,
                    "data_type": row[0]["data_type"],
                    "units": row[0]["units"],
                }
            )
        else:
            # Check for rename
            renamed = gc.query(
                """
                MATCH (old:IMASNode {id: $path})-[:RENAMED_TO]->(new:IMASNode)
                RETURN new.id AS new_path
                """,
                path=path,
            )
            entry: dict[str, Any] = {"path": path, "exists": False}
            if renamed:
                entry["suggestion"] = renamed[0]["new_path"]
            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Signal groups + existing mappings
# ---------------------------------------------------------------------------


def query_signal_sources(
    facility: str,
    ids_name: str | None = None,
    *,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Return enriched signal groups for a facility, optionally filtered by IDS.

    Each row contains group metadata plus member signals and any
    existing MAPS_TO_IMAS connections.
    """
    if gc is None:
        gc = GraphClient()

    ids_filter = ""
    params: dict[str, Any] = {"facility": facility}
    if ids_name:
        ids_filter = (
            "AND EXISTS { "
            "  MATCH (sg)-[:MAPS_TO_IMAS]->(ip:IMASNode) "
            "  WHERE ip.ids = $ids_name "
            "}"
        )
        params["ids_name"] = ids_name

    cypher = f"""
        MATCH (sg:SignalSource)
        WHERE sg.facility_id = $facility
        {ids_filter}
        OPTIONAL MATCH (m)-[:MEMBER_OF]->(sg)
        WITH sg, count(m) AS member_count,
             collect(DISTINCT m.id)[..5] AS sample_members
        OPTIONAL MATCH (sg)-[r:MAPS_TO_IMAS]->(ip:IMASNode)
        RETURN sg.id AS id, sg.group_key AS group_key,
               sg.description AS description,
               sg.keywords AS keywords,
               sg.physics_domain AS physics_domain,
               sg.status AS status,
               member_count,
               sample_members,
               collect(DISTINCT {{
                   target_id: ip.id,
                   transform: r.transform_expression,
                   source_units: r.source_units,
                   target_units: r.target_units
               }}) AS imas_mappings
        ORDER BY sg.group_key
    """
    return gc.query(cypher, **params)


def search_existing_mappings(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Return existing mapping state for a facility+IDS pair.

    Returns a dict with ``mapping`` (IMASMapping node info or None),
    ``sections`` (POPULATES connections), and ``bindings``
    (MAPS_TO_IMAS relationships).
    """
    if gc is None:
        gc = GraphClient()

    mapping_id = f"{facility}:{ids_name}"

    # IMASMapping node
    mapping_rows = gc.query(
        """
        MATCH (m:IMASMapping {id: $id})
        RETURN m.id AS id, m.facility_id AS facility_id,
               m.ids_name AS ids_name, m.dd_version AS dd_version,
               m.status AS status, m.provider AS provider
        """,
        id=mapping_id,
    )

    mapping = mapping_rows[0] if mapping_rows else None

    # Sections via POPULATES
    sections: list[dict[str, Any]] = []
    if mapping:
        sections = gc.query(
            """
            MATCH (m:IMASMapping {id: $id})-[r:POPULATES]->(ip:IMASNode)
            RETURN ip.id AS imas_path, ip.data_type AS data_type,
                   r.config AS config
            """,
            id=mapping_id,
        )

    # Field-level mappings via signal groups
    bindings: list[dict[str, Any]] = []
    if mapping:
        bindings = gc.query(
            """
            MATCH (m:IMASMapping {id: $id})-[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
            MATCH (sg)-[r:MAPS_TO_IMAS]->(ip:IMASNode)
            RETURN sg.id AS source_id,
                   ip.id AS target_id, r.transform_expression AS transform_expression,
                   r.source_units AS source_units, r.target_units AS target_units,
                   r.source_property AS source_property
            """,
            id=mapping_id,
        )

    return {
        "mapping": mapping,
        "sections": sections,
        "bindings": bindings,
    }
