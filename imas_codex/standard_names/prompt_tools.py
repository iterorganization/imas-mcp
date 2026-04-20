"""Tool definitions for the plan-32 Phase 2 prompt A/B/C harness.

Variant C of the extraction prompt uses *tool calling*: rather than
front-loading cluster siblings, reference exemplars and version history
in a long system prompt, the LLM fetches them on demand. This module
exposes:

- LiteLLM-compatible tool schemas (JSON-schema function specs ready to
  pass as ``tools=`` to ``completion()``)
- Thin Python implementations that execute the tool calls against the
  live Neo4j graph, for use by the harness runner.

The intent is research-only; if variant C wins the A/B/C bake-off the
implementations will be adopted into the production compose worker.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─── LiteLLM tool schemas ────────────────────────────────────────────────

FETCH_CLUSTER_SIBLINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_cluster_siblings",
        "description": (
            "Return Standard Names already assigned to paths in the same "
            "IMAS semantic cluster. Use this when you are unsure whether "
            "a similar name has been established for related DD paths."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "string",
                    "description": "Exact cluster id from the current path's metadata.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum siblings to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["cluster_id"],
        },
    },
}

FETCH_REFERENCE_EXEMPLAR_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_reference_exemplar",
        "description": (
            "Return a published exemplar Standard Name that matches a "
            "physics concept phrased in natural language. Use to confirm "
            "controlled-vocabulary choices (e.g. 'electron temperature' → "
            "electron_temperature)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "Natural-language description of the quantity.",
                },
            },
            "required": ["concept"],
        },
    },
}

FETCH_VERSION_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_version_history",
        "description": (
            "Return DD version change history for one path (renames, unit "
            "changes, COCOS transforms). Use when the path description "
            "references a deprecated or renamed field."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Exact IMAS DD path.",
                },
            },
            "required": ["path"],
        },
    },
}


TOOLS: list[dict] = [
    FETCH_CLUSTER_SIBLINGS_TOOL,
    FETCH_REFERENCE_EXEMPLAR_TOOL,
    FETCH_VERSION_HISTORY_TOOL,
]


# ─── Tool implementations (executed by the harness) ─────────────────────


def fetch_cluster_siblings(cluster_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return up to ``limit`` already-named siblings in a cluster."""
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (c:IMASSemanticCluster {id: $cluster_id})<-[:IN_CLUSTER]-(n:IMASNode)
        MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName)
        WHERE sn.name IS NOT NULL
        RETURN n.id AS path,
               sn.name AS standard_name,
               sn.review_status AS review_status
        LIMIT $limit
    """
    with GraphClient() as gc:
        return list(gc.query(cypher, cluster_id=cluster_id, limit=limit))


def fetch_reference_exemplar(concept: str) -> list[dict[str, Any]]:
    """Return up to 3 published Standard Names matching ``concept`` via vector search."""
    from imas_codex.graph.client import GraphClient

    try:
        from imas_codex.graph.embeddings import embed_query
    except ImportError:  # pragma: no cover — embeddings optional during tests
        return []

    try:
        vec = embed_query(concept)
    except Exception as exc:  # pragma: no cover
        logger.warning("fetch_reference_exemplar: embed_query failed: %s", exc)
        return []

    cypher = """
        CALL db.index.vector.queryNodes('standardname_vec', 5, $vec)
        YIELD node, score
        WHERE node.review_status IN ['published', 'accepted']
        RETURN node.name AS standard_name,
               node.description AS description,
               score
        ORDER BY score DESC
        LIMIT 3
    """
    with GraphClient() as gc:
        return list(gc.query(cypher, vec=vec))


def fetch_version_history(path: str) -> list[dict[str, Any]]:
    """Return DD change events for one path (rename, unit, COCOS)."""
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (ch:IMASNodeChange {node_id: $path})
        RETURN ch.change_type AS change_type,
               ch.from_version AS from_version,
               ch.to_version AS to_version,
               ch.detail AS detail
        ORDER BY ch.to_version DESC
    """
    with GraphClient() as gc:
        return list(gc.query(cypher, path=path))


# ─── Dispatcher used by the harness runner ──────────────────────────────


def dispatch_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    """Invoke one of the registered tool functions by name.

    Raises ``ValueError`` if the tool name is unknown.
    """
    if name == "fetch_cluster_siblings":
        return fetch_cluster_siblings(
            cluster_id=arguments["cluster_id"],
            limit=arguments.get("limit", 10),
        )
    if name == "fetch_reference_exemplar":
        return fetch_reference_exemplar(concept=arguments["concept"])
    if name == "fetch_version_history":
        return fetch_version_history(path=arguments["path"])
    raise ValueError(f"Unknown tool: {name}")
