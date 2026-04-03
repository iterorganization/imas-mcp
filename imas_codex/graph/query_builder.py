"""Schema-aware graph query builder.

Provides ``graph_search()`` — a deterministic query builder that translates
structured parameters into parameterised Cypher. No LLM in the loop; fast,
predictable, no hallucination risk.

Uses auto-generated ``NODE_LABEL_PROPS`` and ``VECTOR_INDEXES`` from
``schema_context_data.py`` to validate labels, property names, and resolve
vector indexes at call time.

Filter operators in ``where`` dict
-----------------------------------

Plain keys use equality (``n.prop = $val``).  Append a double-underscore
suffix for richer predicates:

========== ============================== ============================
Suffix     Cypher                          Example value
========== ============================== ============================
(none)     ``n.prop = $val``               ``"value"``
__contains ``n.prop CONTAINS $val``        ``"fishbone"``
__starts_with ``n.prop STARTS WITH $val``  ``"\\\\RESULTS"``
__ends_with ``n.prop ENDS WITH $val``      ``".py"``
__in       ``n.prop IN $val``              ``["discovered", "ingested"]``
__gt       ``n.prop > $val``               ``0.7``
__gte      ``n.prop >= $val``              ``0.7``
__lt       ``n.prop < $val``               ``0.3``
__lte      ``n.prop <= $val``              ``0.3``
__ne       ``n.prop <> $val``              ``"failed"``
========== ============================== ============================
"""

from __future__ import annotations

from typing import Any

from imas_codex.graph.schema_context_data import NODE_LABEL_PROPS, VECTOR_INDEXES
from imas_codex.graph.vector_search import build_vector_search

# Build reverse lookup: label -> index name
_LABEL_TO_INDEX: dict[str, str] = {
    meta[0]: name for name, meta in VECTOR_INDEXES.items()
}

# Supported filter operators: suffix -> Cypher operator template
_FILTER_OPS: dict[str, str] = {
    "contains": "CONTAINS",
    "starts_with": "STARTS WITH",
    "ends_with": "ENDS WITH",
    "in": "IN",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "ne": "<>",
}


def _parse_filter_key(key: str) -> tuple[str, str]:
    """Parse ``prop__op`` into ``(prop, op)`` or ``(prop, "eq")``."""
    for suffix in _FILTER_OPS:
        tag = f"__{suffix}"
        if key.endswith(tag):
            return key[: -len(tag)], suffix
    return key, "eq"


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
        for key in where:
            prop, _op = _parse_filter_key(key)
            if prop not in valid_props:
                raise ValueError(
                    f"Unknown property '{prop}' on {label}. "
                    f"Valid: {', '.join(sorted(valid_props))}"
                )

    # --- pre-compute where conditions (needed before semantic block) --------
    where_conditions: list[str] = []
    if where:
        for i, (key, value) in enumerate(where.items()):
            prop, op = _parse_filter_key(key)
            pname = f"w_{i}"
            if op == "eq":
                where_conditions.append(f"n.{prop} = ${pname}")
            elif op == "in":
                where_conditions.append(f"n.{prop} IN ${pname}")
            elif op in ("contains", "starts_with", "ends_with"):
                cypher_op = _FILTER_OPS[op]
                where_conditions.append(f"n.{prop} {cypher_op} ${pname}")
            else:
                cypher_op = _FILTER_OPS[op]
                where_conditions.append(f"n.{prop} {cypher_op} ${pname}")
            params[pname] = value

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

        _label_val, emb_prop = VECTOR_INDEXES[index_name]
        search_block = build_vector_search(
            index_name,
            label,
            embedding_property=emb_prop,
            where_clauses=where_conditions or None,
        )
        # Vector search as the base
        lines = [search_block]
    else:
        lines = [f"MATCH (n:{label})"]
        if where_conditions:
            lines.append("WHERE " + " AND ".join(where_conditions))

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
