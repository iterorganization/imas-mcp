"""Centralized NodeCategory constants for pipeline participation.

All consumers that filter on ``node_category`` must import from this
module rather than hard-coding category strings.
"""

# Base set of categories that behave like physics quantities
# (enriched, embedded, searchable, SN-extractable).
QUANTITY_CATEGORIES: frozenset[str] = frozenset({"quantity", "geometry"})

# Nodes whose descriptions are embedded into the vector space.
EMBEDDABLE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES

# Nodes surfaced by search / MCP tools.
SEARCHABLE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES | {"coordinate"}

# Nodes eligible for standard-name extraction.
SN_SOURCE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES

# Nodes that enter the LLM enrichment pipeline
# (description generation, keyword extraction).
ENRICHABLE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES | {"coordinate"}

__all__ = [
    "EMBEDDABLE_CATEGORIES",
    "ENRICHABLE_CATEGORIES",
    "QUANTITY_CATEGORIES",
    "SEARCHABLE_CATEGORIES",
    "SN_SOURCE_CATEGORIES",
]
