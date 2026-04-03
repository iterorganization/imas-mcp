"""Neo4j 2026.01 SEARCH clause builder for vector similarity queries.

Generates SEARCH syntax with in-index pre-filtering, replacing legacy
db.index.vector.queryNodes() procedure calls. Property-based WHERE
clauses are pushed inside the SEARCH block for index-level evaluation.
Relationship-based filters must be applied outside as post-filters.
"""

from __future__ import annotations


def build_vector_search(
    index: str,
    label: str,
    embedding_property: str = "embedding",
    *,
    where_clauses: list[str] | None = None,
    k: str = "$k",
    node_alias: str = "n",
    score_alias: str = "score",
    embedding_param: str = "$embedding",
) -> str:
    """Build a SEARCH clause for vector similarity queries.

    Generates Neo4j 2026.01 SEARCH syntax with in-index pre-filtering.
    Property-based WHERE clauses are pushed inside the SEARCH block
    for index-level evaluation. Relationship-based filters must be
    applied outside as post-filters.

    Args:
        index: Vector index name (e.g. 'imas_node_embedding').
        label: Node label (e.g. 'IMASNode').
        embedding_property: Property storing the embedding vector.
        where_clauses: Property-based filter expressions to apply
            inside the SEARCH block (pre-filtering). Do NOT include
            relationship traversals here — those must go outside.
        k: Cypher expression for limit (default: "$k" parameter).
            Can be a literal like "20" or a parameter like "$limit".
        node_alias: Variable name for the matched node.
        score_alias: Variable name for the similarity score.
        embedding_param: Cypher expression for the embedding vector
            (default: "$embedding"). Can also be a variable name.

    Returns:
        A CALL () { SEARCH ... } block as a string, ready to be
        embedded in a larger Cypher query.

    Example:
        >>> build_vector_search(
        ...     "facility_signal_desc_embedding",
        ...     "FacilitySignal",
        ...     where_clauses=["n.facility_id = $facility", "n.physics_domain = $domain"],
        ...     k="$k",
        ... )
        'CALL () {\\n  SEARCH n:FacilitySignal\\n  ...'
    """
    parts = [
        "CALL () {",
        f"  SEARCH {node_alias}:{label}",
        f"  USING VECTOR INDEX {index}",
    ]

    if where_clauses:
        where_str = " AND ".join(where_clauses)
        parts.append(f"  WHERE {where_str}")

    parts.extend(
        [
            f"  WITH {node_alias}, vector.similarity.cosine({node_alias}.{embedding_property}, {embedding_param}) AS {score_alias}",
            f"  ORDER BY {score_alias} DESC",
            f"  LIMIT {k}",
            "}",
        ]
    )

    return "\n".join(parts)
