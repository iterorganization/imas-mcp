"""Naming-scope classifier for DD paths.

Classifies IMAS Data Dictionary paths into three categories for standard
name generation:

- **quantity**: Independent physics concept → gets a StandardName
- **metadata**: Storage artifact (data, time, validity) → modifier of
  parent concept
- **skip**: Not a nameable concept (container, identifier, string field, etc.)

The classifier is purely rule-based, deterministic, and operates on a single
dict of enriched DD node attributes (as returned by the enriched extraction
query in ``imas_codex.standard_names.sources.dd``).
"""

from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Leaf data types that can carry physics content.
PHYSICS_LEAF_TYPES: frozenset[str] = frozenset(
    {
        "FLT_0D",
        "FLT_1D",
        "FLT_2D",
        "FLT_3D",
        "FLT_4D",
        "FLT_5D",
        "FLT_6D",
        "INT_0D",
        "INT_1D",
        "INT_2D",
        "CPX_0D",
        "CPX_1D",
        "CPX_2D",
    }
)

#: Structure types that may own physics concepts.
STRUCTURE_TYPES: frozenset[str] = frozenset({"STRUCTURE", "STRUCT_ARRAY"})

#: Suffixes in path segments that indicate error companion fields.
ERROR_SUFFIXES: tuple[str, ...] = ("_error_upper", "_error_lower", "_error_index")

#: Regex for descriptions that indicate an index, flag, or counter —
#: i.e. INT_0D fields that are *not* physics quantities.
_INDEX_FLAG_RE: re.Pattern[str] = re.compile(
    r"\b(?:index|flag|identifier|number[\s_]of|count[\s_]of)\b",
    re.IGNORECASE,
)

# Type alias for the three-way classification.
Scope = Literal["quantity", "metadata", "skip"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_path(node: dict) -> Scope:
    """Classify a DD path for standard name generation.

    Args:
        node: Dict with keys from the enriched DD query:

            - **path** (*str*): Full DD path
            - **description** (*str*): Node description
            - **data_type** (*str*): e.g. FLT_0D, FLT_1D, STRUCTURE, …
            - **unit** (*str | None*): Authoritative unit from HAS_UNIT
            - **node_category** (*str | None*): Node category from DD
            - **parent_path** (*str | None*): Parent node path
            - **parent_type** (*str | None*): Parent data type
            - **cluster_label** (*str | None*): Semantic cluster label

    Returns:
        ``"quantity"`` – independent physics concept that gets a StandardName.
        ``"metadata"`` – storage artifact; modifier of parent concept.
        ``"skip"`` – not a nameable concept.
    """
    path: str = node.get("path", "")
    data_type: str = node.get("data_type", "")
    unit: str | None = node.get("unit") or None  # normalise "" → None
    parent_type: str | None = node.get("parent_type") or None
    description: str = node.get("description", "") or ""

    # Derive the last path segment once.
    last_segment = path.rsplit("/", 1)[-1] if path else ""

    # ------------------------------------------------------------------
    # Rule 1: /data under STRUCTURE parent with unit → metadata
    # ------------------------------------------------------------------
    if last_segment == "data" and parent_type in STRUCTURE_TYPES and unit is not None:
        return "metadata"

    # ------------------------------------------------------------------
    # Rule 2: /time leaf → metadata (coordinate, not concept)
    # ------------------------------------------------------------------
    if last_segment == "time":
        return "metadata"

    # ------------------------------------------------------------------
    # Rule 3: /validity and /validity_timed → metadata (quality flag)
    # ------------------------------------------------------------------
    if last_segment in ("validity", "validity_timed"):
        return "metadata"

    # ------------------------------------------------------------------
    # Rule 4: Error fields → skip (defensive — normally pre-filtered)
    # ------------------------------------------------------------------
    if _is_error_field(path):
        return "skip"

    # ------------------------------------------------------------------
    # Rule 5: STR_0D (string fields) → skip
    # ------------------------------------------------------------------
    if data_type == "STR_0D":
        return "skip"

    # ------------------------------------------------------------------
    # Rules 6–7: INT_0D without unit — disambiguate index/flag vs physics
    # ------------------------------------------------------------------
    if data_type == "INT_0D" and unit is None:
        if _INDEX_FLAG_RE.search(description):
            return "skip"  # Rule 6
        return "quantity"  # Rule 7 — genuinely dimensionless integer

    # ------------------------------------------------------------------
    # Rules 8–9: Physics leaf types
    # ------------------------------------------------------------------
    if data_type in PHYSICS_LEAF_TYPES:
        return "quantity"  # with or without unit

    # ------------------------------------------------------------------
    # Rule 10: STRUCTURE/STRUCT_ARRAY with unit → quantity
    # ------------------------------------------------------------------
    if data_type in STRUCTURE_TYPES and unit is not None:
        return "quantity"

    # ------------------------------------------------------------------
    # Rule 11: Everything else → skip
    # ------------------------------------------------------------------
    return "skip"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_error_field(path: str) -> bool:
    """Return True if *path* contains an error-field suffix."""
    return any(suffix in path for suffix in ERROR_SUFFIXES)
