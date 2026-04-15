"""Centralized NodeCategory constants for pipeline participation.

All consumers that filter on ``node_category`` must import from this
module rather than hard-coding category strings.  During the migration
from the old 3-value enum (data/error/metadata) to the expanded 6-value
enum, the transitional ``data`` value is included so that queries work
against both pre- and post-migration graph states.

After migration confirms zero ``data`` nodes remain, remove ``"data"``
from every set below and delete the ``data`` enum value from
``imas_codex/schemas/imas_dd.yaml``.
"""

# Nodes whose descriptions are embedded into the vector space.
# Only physics quantities belong in the vector index.
EMBEDDABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "data"})

# Nodes surfaced by search / MCP tools.
SEARCHABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "coordinate", "data"})

# Nodes eligible for standard-name extraction.
SN_SOURCE_CATEGORIES: frozenset[str] = frozenset({"quantity", "data"})

# Nodes that enter the LLM enrichment pipeline
# (description generation, keyword extraction).
ENRICHABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "coordinate", "data"})

__all__ = [
    "EMBEDDABLE_CATEGORIES",
    "ENRICHABLE_CATEGORIES",
    "SEARCHABLE_CATEGORIES",
    "SN_SOURCE_CATEGORIES",
]
