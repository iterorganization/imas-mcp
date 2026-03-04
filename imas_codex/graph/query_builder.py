"""Schema-aware graph query builder.

Provides ``graph_search()`` — a deterministic query builder that translates
structured parameters into parameterised Cypher. No LLM in the loop; fast,
predictable, no hallucination risk.

Uses auto-generated ``NODE_LABEL_PROPS`` and ``VECTOR_INDEXES`` from
``schema_context_data.py`` to validate labels, property names, and resolve
vector indexes at call time.
"""

from __future__ import annotations

from typing import Any

from imas_codex.graph.schema_context_data import NODE_LABEL_PROPS, VECTOR_INDEXES

# Build reverse lookup: label -> index name
_LABEL_TO_INDEX: dict[str, str] = {
    meta[0]: name for name, meta in VECTOR_INDEXES.items()
}


def graph_search(
    label: str,
    *,
    where: dict[str, Any] | None = None,
    semantic: str | None = None,
    traverse: list[str] | None = None,
    return_props: list[str] | None = None,
    limit: int = 25,
    order_by: str | None = None,
    gc: Any = None,
    embed_fn: Any = None,
) -> list[dict]:
    """Flexible graph query builder with schema validation.

    Uses auto-generated schema data to validate labels, properties,
    and automatically resolve vector indexes.

    Args:
        label: Node label to search (validated against schema).
        where: Property filters ``{prop: value}`` (validated against schema).
        semantic: Text for vector similarity (auto-resolves index).
        traverse: Relationship paths to follow, e.g.
            ``["DATA_ACCESS>DataAccess"]`` produces
            ``MATCH (n)-[:DATA_ACCESS]->(da:DataAccess)``.
        return_props: Properties to project (default: key props from schema).
        limit: Maximum results.
        order_by: Property to order by.
        gc: GraphClient instance.
        embed_fn: Embedding function ``(text) -> list[float]``.

    Returns:
        List of result dicts.

    Raises:
        ValueError: On unknown label, invalid property, or missing vector index.
    """
    # --- validate label --------------------------------------------------
    if label not in NODE_LABEL_PROPS:
        raise ValueError(
            f"Unknown label '{label}'. "
            f"Valid labels: {', '.join(sorted(NODE_LABEL_PROPS))}"
        )

    valid_props = NODE_LABEL_PROPS[label]
    params: dict[str, Any] = {}

    # --- validate where properties ---------------------------------------
    if where:
        for prop in where:
            if prop not in valid_props:
                raise ValueError(
                    f"Unknown property '{prop}' on {label}. "
                    f"Valid: {', '.join(sorted(valid_props))}"
                )

    # --- semantic search -------------------------------------------------
    if semantic:
        index_name = _LABEL_TO_INDEX.get(label)
        if not index_name:
            raise ValueError(
                f"No vector index for '{label}'. "
                f"Labels with indexes: {', '.join(sorted(_LABEL_TO_INDEX))}"
            )
        if embed_fn is None:
            raise ValueError("embed_fn required for semantic search")

        embedding = embed_fn(semantic)
        params["embedding"] = embedding
        params["k"] = limit

        # Vector search as the base
        lines = [
            f'CALL db.index.vector.queryNodes("{index_name}", $k, $embedding)',
            "YIELD node AS n, score",
        ]
    else:
        lines = [f"MATCH (n:{label})"]

    # --- where clause ----------------------------------------------------
    if where:
        conditions = []
        for i, (prop, value) in enumerate(where.items()):
            pname = f"w_{i}"
            conditions.append(f"n.{prop} = ${pname}")
            params[pname] = value
        lines.append("WHERE " + " AND ".join(conditions))

    # --- traversals ------------------------------------------------------
    if traverse:
        for i, spec in enumerate(traverse):
            rel_type, target_label = _parse_traverse(spec)
            alias = f"t{i}"
            lines.append(f"MATCH (n)-[:{rel_type}]->({alias}:{target_label})")

    # --- return clause ---------------------------------------------------
    props = return_props or _default_return_props(label)
    return_parts = [f"n.{p} AS {p}" for p in props]

    # Add traversal properties
    if traverse:
        for i, spec in enumerate(traverse):
            _, target_label = _parse_traverse(spec)
            alias = f"t{i}"
            target_props = NODE_LABEL_PROPS.get(target_label, {})
            # Include key props from traversed nodes
            for tp in _default_return_props(target_label):
                if tp in target_props:
                    return_parts.append(f"{alias}.{tp} AS {target_label}_{tp}")

    if semantic:
        return_parts.append("score")

    lines.append("RETURN " + ", ".join(return_parts))

    # --- order by --------------------------------------------------------
    if order_by:
        if order_by not in valid_props:
            raise ValueError(f"Cannot order by '{order_by}': not a property of {label}")
        lines.append(f"ORDER BY n.{order_by}")
    elif semantic:
        lines.append("ORDER BY score DESC")

    # --- limit -----------------------------------------------------------
    lines.append(f"LIMIT {int(limit)}")

    cypher = "\n".join(lines)
    return gc.query(cypher, **params)


def _parse_traverse(spec: str) -> tuple[str, str]:
    """Parse a traverse spec like ``"DATA_ACCESS>DataAccess"``.

    Returns:
        (relationship_type, target_label)
    """
    if ">" not in spec:
        raise ValueError(
            f"Invalid traverse spec '{spec}'. Expected format: 'REL_TYPE>TargetLabel'"
        )
    rel_type, target_label = spec.split(">", 1)
    rel_type = rel_type.strip()
    target_label = target_label.strip()

    if target_label not in NODE_LABEL_PROPS:
        raise ValueError(
            f"Unknown traverse target '{target_label}'. "
            f"Valid labels: {', '.join(sorted(NODE_LABEL_PROPS))}"
        )
    return rel_type, target_label


def _default_return_props(label: str) -> list[str]:
    """Pick default return properties for a label — id + key descriptive fields."""
    props = NODE_LABEL_PROPS.get(label, {})
    defaults = ["id"]
    for candidate in ("name", "path", "description", "status", "facility_id"):
        if candidate in props:
            defaults.append(candidate)
    return defaults
