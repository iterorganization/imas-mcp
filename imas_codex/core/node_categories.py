"""Centralized NodeCategory constants for pipeline participation.

All consumers that filter on ``node_category`` must import from this
module rather than hard-coding category strings.
"""

# Nodes whose descriptions are embedded into the vector space.
# Only physics quantities belong in the vector index.
EMBEDDABLE_CATEGORIES: frozenset[str] = frozenset({"quantity"})

# Nodes surfaced by search / MCP tools.
SEARCHABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "coordinate"})

# Nodes eligible for standard-name extraction.
SN_SOURCE_CATEGORIES: frozenset[str] = frozenset({"quantity"})

# Nodes that enter the LLM enrichment pipeline
# (description generation, keyword extraction).
ENRICHABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "coordinate"})

__all__ = [
    "EMBEDDABLE_CATEGORIES",
    "ENRICHABLE_CATEGORIES",
    "SEARCHABLE_CATEGORIES",
    "SN_SOURCE_CATEGORIES",
]
