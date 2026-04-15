"""Shared node classification logic — single source of truth.

Both the DD build pipeline (``build_dd._classify_node``) and the
one-off migration CLI (``dd migrate-categories``) delegate to this
module so that rules cannot drift.

Two-pass architecture:

* **Pass 1** (build-time) uses only XML-derived attributes that are
  available before any relationships exist in the graph.
* **Pass 2** (post-build / migration) uses graph relationships
  (``HAS_IDENTIFIER_SCHEMA``, ``HAS_COORDINATE``) and children
  evidence to refine the initial classification.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# Data types that represent physics leaf values
PHYSICS_LEAF_TYPES: frozenset[str] = frozenset(
    {
        "FLT_0D",
        "FLT_1D",
        "FLT_2D",
        "FLT_3D",
        "FLT_4D",
        "FLT_5D",
        "FLT_6D",
        "CPX_0D",
        "CPX_1D",
        "CPX_2D",
        "CPX_3D",
        "CPX_4D",
        "CPX_5D",
    }
)

# Structure data types
STRUCTURE_TYPES: frozenset[str] = frozenset({"STRUCTURE", "STRUCT_ARRAY"})

# Integer data types
INTEGER_TYPES: frozenset[str] = frozenset(
    {
        "INT_0D",
        "INT_1D",
        "INT_2D",
        "INT_3D",
    }
)

# String data types — always metadata
STRING_TYPES: frozenset[str] = frozenset({"STR_0D", "STR_1D"})

# Structural keyword fragments in path segments.  When an INT node's
# last path segment contains one of these, it is a storage index rather
# than a physics quantity.
STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "index",
        "grid_index",
        "type",
        "grid_type",
        "process",
        "flag",
        "identifier",
        "basis",
        "rank",
        "dimension",
        "count",
        "number",
        "size",
        "shape",
        "switch",
        "selector",
        "interpolation",
        "position",
    }
)

# Segments whose full name is a known coordinate.  Used ONLY for unitless
# FLT nodes (Pass 1, rule 10) where neither relational HAS_COORDINATE
# nor lexical /time detection fired.
COORDINATE_SEGMENTS: frozenset[str] = frozenset(
    {
        "rho_tor_norm",
        "rho_pol_norm",
        "psi_norm",
        "psi",
        "phi",
        "theta",
        "r",
        "z",
        "rho_tor",
        "zeta",
        "chi",
        "s",
        "rho",
    }
)

# Units that indicate a "no-unit" / dimensionless quantity.
_NO_UNIT: frozenset[str] = frozenset({"", "-", "mixed"})

# Metadata subtree segments (depth ≥ 2)
_METADATA_SUBTREES: frozenset[str] = frozenset({"ids_properties", "code"})

# Generic metadata leaf names (depth ≥ 3)
_METADATA_LEAVES: frozenset[str] = frozenset(
    {"description", "name", "comment", "source", "provider"}
)


# ──────────────────────────────────────────────────────────────────
# Pass 1 — build-time classification (XML signals only)
# ──────────────────────────────────────────────────────────────────


def classify_node_pass1(
    path: str,
    name: str,
    *,
    data_type: str | None = None,
    unit: str | None = None,
    parent_data_type: str | None = None,
) -> str:
    """Classify an IMAS node using build-time XML attributes.

    Parameters
    ----------
    path:
        Full IMAS path (e.g. ``"equilibrium/time_slice/profiles_1d/psi"``).
    name:
        Leaf segment of the path.
    data_type:
        DD data type string (e.g. ``"FLT_1D"``, ``"STRUCTURE"``).
    unit:
        Physical unit string from the DD (``None``, ``""``, ``"-"``, ``"eV"``…).
    parent_data_type:
        Data type of the parent node, if available.

    Returns
    -------
    str
        One of ``"error"``, ``"metadata"``, ``"coordinate"``, ``"structural"``,
        ``"quantity"``.
    """
    parts = path.split("/")
    last_seg = parts[-1] if parts else name
    dt = (data_type or "").upper()
    unit_str = unit if unit is not None else ""

    # Rule 1: Error suffix
    if (
        name.endswith("_error_upper")
        or name.endswith("_error_lower")
        or name.endswith("_error_index")
    ):
        return "error"

    # Rule 2: Metadata subtree (ids_properties/*, code/*)
    if any(seg in _METADATA_SUBTREES for seg in parts[1:]):
        return "metadata"

    # Rule 3: Generic metadata leaf at depth ≥ 3
    if len(parts) >= 3:
        if last_seg in _METADATA_LEAVES:
            return "metadata"
        if (
            len(parts) >= 2
            and parts[-2] == "identifier"
            and last_seg
            in (
                "description",
                "name",
            )
        ):
            return "metadata"

    # Rule 4: String types → metadata
    if dt in STRING_TYPES:
        return "metadata"

    # Rule 5: /time leaf with FLT type → coordinate
    if last_seg == "time" and dt in PHYSICS_LEAF_TYPES:
        return "coordinate"

    # Rule 6: validity / validity_timed → structural
    if last_seg in ("validity", "validity_timed"):
        return "structural"

    # Rule 7: /data leaf under STRUCTURE parent → structural
    if last_seg == "data" and (parent_data_type or "").upper() in STRUCTURE_TYPES:
        return "structural"

    # Rule 8: INT with structural keyword → structural
    if dt in INTEGER_TYPES:
        last_lower = last_seg.lower()
        for kw in STRUCTURAL_KEYWORDS:
            if kw in last_lower:
                return "structural"

    # Rule 9: Physics leaf type with physical unit → quantity
    if dt in PHYSICS_LEAF_TYPES and unit_str and unit_str not in _NO_UNIT:
        return "quantity"

    # Rule 10: Physics leaf type without unit
    if dt in PHYSICS_LEAF_TYPES and (not unit_str or unit_str in _NO_UNIT):
        if last_seg in COORDINATE_SEGMENTS:
            return "coordinate"
        # Dimensionless physics quantity (e.g. elongation, beta)
        return "quantity"

    # Rule 11: STRUCTURE / STRUCT_ARRAY with unit → provisional quantity
    if dt in STRUCTURE_TYPES and unit_str and unit_str not in _NO_UNIT:
        return "quantity"

    # Rule 12: STRUCTURE / STRUCT_ARRAY without unit → structural
    if dt in STRUCTURE_TYPES:
        return "structural"

    # Rule 13: INT_0D without unit, no structural keywords → quantity
    if dt == "INT_0D" and (not unit_str or unit_str in _NO_UNIT):
        return "quantity"

    # Rule 14: INT with physical unit → quantity
    if dt in INTEGER_TYPES and unit_str and unit_str not in _NO_UNIT:
        return "quantity"

    # Rule 15: Remaining INT → structural
    if dt in INTEGER_TYPES:
        return "structural"

    # Fallback: structural
    return "structural"


# ──────────────────────────────────────────────────────────────────
# Pass 2 — post-build relational classification
# ──────────────────────────────────────────────────────────────────


def classify_node_pass2(
    current_category: str,
    *,
    has_identifier_schema: bool = False,
    is_coordinate_target: bool = False,
    children_categories: list[str] | None = None,
    data_type: str | None = None,
    unit: str | None = None,
) -> str | None:
    """Refine classification using graph relationships.

    Called after all nodes and relationships exist.  Returns a new
    category if the relationship evidence overrides Pass 1, otherwise
    ``None`` (keep Pass 1 result).

    Parameters
    ----------
    current_category:
        Category assigned by Pass 1.
    has_identifier_schema:
        ``True`` if the node has a ``HAS_IDENTIFIER_SCHEMA`` relationship.
    is_coordinate_target:
        ``True`` if the node is the target of a ``HAS_COORDINATE``
        relationship from another node.
    children_categories:
        Categories of direct child nodes (for STRUCTURE validation).
    data_type:
        DD data type string.
    unit:
        Physical unit string.
    """
    dt = (data_type or "").upper()

    # R1: Identifier schema → identifier (overrides anything)
    if has_identifier_schema:
        return "identifier"

    # R2: Coordinate target → coordinate (overrides quantity/structural)
    if is_coordinate_target and current_category != "error":
        return "coordinate"

    # R3: Validate STRUCTURE+unit as quantity
    if (
        current_category == "quantity"
        and dt in STRUCTURE_TYPES
        and unit
        and unit not in _NO_UNIT
    ):
        children = children_categories or []
        has_data_child = any(
            c in ("quantity", "coordinate", "structural") for c in children
        )
        if not has_data_child:
            # No children evidence — demote to structural
            return "structural"

    return None


__all__ = [
    "COORDINATE_SEGMENTS",
    "INTEGER_TYPES",
    "PHYSICS_LEAF_TYPES",
    "STRING_TYPES",
    "STRUCTURAL_KEYWORDS",
    "STRUCTURE_TYPES",
    "classify_node_pass1",
    "classify_node_pass2",
]
