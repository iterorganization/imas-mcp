"""Centralized NodeCategory constants for pipeline participation.

All consumers that filter on ``node_category`` must import from this
module rather than hard-coding category strings.
"""

# Base set of categories that behave like physics quantities
# (enriched, embedded, searchable, SN-extractable).
QUANTITY_CATEGORIES: frozenset[str] = frozenset({"quantity", "geometry"})
IDENTIFIER_CATEGORIES: frozenset[str] = frozenset({"identifier"})

# Nodes whose descriptions are embedded into the vector space.
EMBEDDABLE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES | IDENTIFIER_CATEGORIES

# Nodes that participate in semantic clustering (physics quantities only).
CLUSTERABLE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES

# Nodes surfaced by search / MCP tools.
SEARCHABLE_CATEGORIES: frozenset[str] = (
    QUANTITY_CATEGORIES | {"coordinate"} | IDENTIFIER_CATEGORIES
)

# Nodes eligible for standard-name extraction.
# Includes 'coordinate' because normalized flux coordinates, grid axes, and
# similar scalar leaves are first-class namable quantities — not mere bookkeeping.
# The leaf invariant (data_type NOT IN STRUCTURE/STRUCT_ARRAY) gates containers.
SN_SOURCE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES | {"coordinate"}

# Nodes that enter the LLM enrichment pipeline
# (description generation, keyword extraction).
ENRICHABLE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES | {"coordinate"}

__all__ = [
    "CLUSTERABLE_CATEGORIES",
    "EMBEDDABLE_CATEGORIES",
    "ENRICHABLE_CATEGORIES",
    "IDENTIFIER_CATEGORIES",
    "QUANTITY_CATEGORIES",
    "SEARCHABLE_CATEGORIES",
    "SN_SOURCE_CATEGORIES",
]
