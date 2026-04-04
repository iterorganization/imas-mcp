"""Neo4j 2026.01 SEARCH clause builder for vector similarity queries.

Generates Cypher 25 MATCH + SEARCH syntax, replacing legacy
db.index.vector.queryNodes() procedure calls.  All property and
relationship filters are applied as post-filters after the ANN
candidate selection.

In-index pre-filtering (WHERE inside SEARCH) requires properties
to be registered as additional vector index properties.  Without
that configuration, all filtering is post-filtering.
"""

from __future__ import annotations


def build_vector_search(
    index: str,
    label: str,
    *,
    where_clauses: list[str] | None = None,
    k: str = "$k",
    node_alias: str = "n",
    score_alias: str = "score",
    embedding_param: str = "$embedding",
) -> str:
    """Build a Cypher 25 MATCH + SEARCH clause for vector similarity.

    Generates the correct Neo4j 2026.01 SEARCH syntax::

        CYPHER 25
        MATCH (n:Label)
        SEARCH n IN (
          VECTOR INDEX index_name
          FOR $embedding
          LIMIT $k
        ) SCORE AS score
        WHERE n.prop = $val

    Args:
        index: Vector index name (e.g. 'imas_node_embedding').
        label: Node label (e.g. 'IMASNode').
        where_clauses: Filter expressions applied as post-filters
            after the ANN candidate selection.  Both property filters
            (``n.facility_id = $f``) and relationship pattern predicates
            (``NOT (n)-[:REL]->(:Other)``) are valid here.
        k: Cypher expression for the ANN candidate limit
            (default: "$k").  Can be a literal like "20".
        node_alias: Variable name for the matched node.
        score_alias: Variable name for the similarity score.
        embedding_param: Cypher expression for the query embedding
            (default: "$embedding").

    Returns:
        A complete query prefix starting with ``CYPHER 25``.
        Append OPTIONAL MATCH, WITH, and RETURN clauses as needed.
    """
    parts = [
        "CYPHER 25",
        f"MATCH ({node_alias}:{label})",
        f"SEARCH {node_alias} IN (",
        f"  VECTOR INDEX {index}",
        f"  FOR {embedding_param}",
        f"  LIMIT {k}",
        f") SCORE AS {score_alias}",
    ]

    if where_clauses:
        where_str = " AND ".join(where_clauses)
        parts.append(f"WHERE {where_str}")

    return "\n".join(parts)
