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

import re

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
# Fit-artifact constants
# ──────────────────────────────────────────────────────────────────

#: Leaf segments that are fitting-process diagnostics — chi², residual,
#: covariance, fitting_weight, fit_type.
FIT_DIAGNOSTIC_SEGMENTS: frozenset[str] = frozenset(
    {"chi_squared", "residual", "covariance", "fitting_weight", "fit_type"}
)

#: Regex for fit-diagnostic segments (including compound names like
#: ``pressure_chi_squared``).
_FIT_DIAGNOSTIC_RE: re.Pattern[str] = re.compile(
    r"(?:^|_)(?:chi_squared|residual|covariance|fitting_weight|fit_type)(?:_|$)",
)

#: Children of ``*_fit`` containers that are per-fit provenance rather
#: than independent physics concepts.
FIT_CHILD_SEGMENTS: frozenset[str] = frozenset(
    {"reconstructed", "measured", "weight", "time_measurement", "rho_tor_norm"}
)

#: Regex matching ``*_fit`` parent segment.
_FIT_PARENT_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_]*_fit$")

# ──────────────────────────────────────────────────────────────────
# Representation constants
# ──────────────────────────────────────────────────────────────────

#: Path fragments indicating GGD / basis-function representation storage.
_REPRESENTATION_PATH_MARKERS: tuple[str, ...] = (
    "grids_ggd/",
    "/ggd_fast/",
    "/grid_subset/",
)

#: Regex for representation leaf/parent segments — spline coefficients,
#: Fourier modes, finite-element interpolation, GGD grid infrastructure.
_REPRESENTATION_SEGMENT_RE: re.Pattern[str] = re.compile(
    r"(?:^|_)(?:coefficients?|ggd|finite_element|interpolation|basis|spline|"
    r"fourier_modes?|harmonics_coefficients|grid_object|grid_subset|"
    r"jacobian|metric)(?:_|$)",
)

# Path segments indicating machine / diagnostic hardware geometry.
GEOMETRY_PATH_PATTERNS: frozenset[str] = frozenset(
    {
        "geometry",
        "outline",
        "aperture",
        "line_of_sight",
        "first_point",
        "second_point",
        "position",
        "detector",
        "mirror",
        "waveguide",
        "launcher",
        "antenna",
        "annular_grid",
        "first_wall",
        "limiter",
        "divertor",
        "vessel",
        "pf_active",
        "pf_passive",
    }
)

# Path segments that indicate plasma physics outputs, NOT hardware
# geometry, even when spatial units are present.
GEOMETRY_EXCLUSION_PATTERNS: frozenset[str] = frozenset(
    {
        "boundary",
        "separatrix",
        "grid",
        "ggd",
        "flux_surface",
        "constraint",
        "magnetic_axis",
        "pedestal",
        "itb",
        "etb",
        "x_point",
        "strike_point",
    }
)

# Units that are purely spatial (lengths, angles, areas, volumes).
# Inverse-spatial units (m^-1, m^-2, m^-3) are NOT included — those
# are densities, not geometry.
SPATIAL_UNITS: frozenset[str] = frozenset({"m", "rad", "m^2", "m^3", "deg"})


# ──────────────────────────────────────────────────────────────────
# Geometry detection helper
# ──────────────────────────────────────────────────────────────────


def _is_geometry_path(path: str, unit: str | None) -> bool:
    """Check if a path represents machine/diagnostic hardware geometry.

    A node is geometry when BOTH conditions hold:

    1. **Spatial unit** (m, rad, m²,  m³, deg) — excludes inverse-spatial
       units like m⁻¹, m⁻², m⁻³ (densities).
    2. **Geometry ancestor** — at least one path segment matches
       :data:`GEOMETRY_PATH_PATTERNS` AND no segment matches
       :data:`GEOMETRY_EXCLUSION_PATTERNS`.

    Note: Inclusion matching uses **exact segment match**, while exclusion
    uses **substring** match so that composite names like
    ``boundary_separatrix`` are caught.
    """
    if not unit or unit not in SPATIAL_UNITS:
        return False
    segments = path.split("/")
    has_geometry_ancestor = any(seg in GEOMETRY_PATH_PATTERNS for seg in segments)
    has_exclusion = any(
        excl in seg for seg in segments for excl in GEOMETRY_EXCLUSION_PATTERNS
    )
    return has_geometry_ancestor and not has_exclusion


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
        ``"geometry"``, ``"quantity"``.
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

    # ── Fit-artifact rules (before quantity/geometry fallthrough) ──

    # Rule F1: Fit-diagnostic segments → fit_artifact
    if last_seg in FIT_DIAGNOSTIC_SEGMENTS or _FIT_DIAGNOSTIC_RE.search(last_seg):
        return "fit_artifact"

    # Rule F2: Known fit-child segment under *_fit parent → fit_artifact
    # Uses path-only heuristic (parent segment ends in _fit).
    if last_seg in FIT_CHILD_SEGMENTS or _FIT_PARENT_RE.search(
        parts[-2] if len(parts) >= 2 else ""
    ):
        parent_seg = parts[-2] if len(parts) >= 2 else ""
        if _FIT_PARENT_RE.search(parent_seg) and (
            last_seg in FIT_CHILD_SEGMENTS or last_seg.startswith("time_measurement")
        ):
            return "fit_artifact"

    # ── Representation rules (before quantity/geometry fallthrough) ──

    # Rule R1: GGD / grid_subset subtrees → representation
    if any(marker in path for marker in _REPRESENTATION_PATH_MARKERS):
        return "representation"

    # Rule R2: Representation-related leaf or parent segment
    if _REPRESENTATION_SEGMENT_RE.search(last_seg):
        return "representation"
    if len(parts) >= 2 and _REPRESENTATION_SEGMENT_RE.search(parts[-2]):
        return "representation"

    # Rule 9a: Physics leaf type + spatial unit + geometry ancestor → geometry
    if dt in PHYSICS_LEAF_TYPES and unit_str and unit_str not in _NO_UNIT:
        if _is_geometry_path(path, unit_str):
            return "geometry"
        # Rule 9b: Physics leaf type + physics unit → quantity
        return "quantity"

    # Rule 10: Physics leaf type without unit
    if dt in PHYSICS_LEAF_TYPES and (not unit_str or unit_str in _NO_UNIT):
        if last_seg in COORDINATE_SEGMENTS:
            return "coordinate"
        # Dimensionless physics quantity (e.g. elongation, beta)
        return "quantity"

    # Rule 11a: STRUCTURE + unit + geometry ancestor → geometry
    if dt in STRUCTURE_TYPES and unit_str and unit_str not in _NO_UNIT:
        if _is_geometry_path(path, unit_str):
            return "geometry"
        # Rule 11b: STRUCTURE + unit → provisional quantity
        return "quantity"

    # Rule 12: STRUCTURE / STRUCT_ARRAY without unit → structural
    if dt in STRUCTURE_TYPES:
        return "structural"

    # Rule 13: INT_0D without unit, no structural keywords → quantity
    if dt == "INT_0D" and (not unit_str or unit_str in _NO_UNIT):
        return "quantity"

    # Rule 14a: INT + spatial unit + geometry ancestor → geometry
    if dt in INTEGER_TYPES and unit_str and unit_str not in _NO_UNIT:
        if _is_geometry_path(path, unit_str):
            return "geometry"
        # Rule 14b: INT + physics unit → quantity
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
    name: str | None = None,
    parent_name: str | None = None,
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
    name:
        Leaf segment of the path (e.g. ``"psi"``).  When provided,
        enables the R2 name+unit guard that prevents physics data
        with meaningful units from being reclassified as coordinates.
    parent_name:
        Name (last segment) of the parent node, if available.
        Used by R4 to detect ``*_fit`` parents for fit-child promotion.
    """
    dt = (data_type or "").upper()

    # R1: Identifier schema → identifier (overrides anything)
    if has_identifier_schema:
        return "identifier"

    # R2: Coordinate target → coordinate (with name+unit guard)
    # Canonical coordinate names (r, z, psi, …) always become coordinate.
    # Physics data nodes with a meaningful unit keep their Pass 1 category
    # even when they appear as coordinate targets.
    if is_coordinate_target and current_category != "error":
        if name is not None:
            if name in COORDINATE_SEGMENTS:
                return "coordinate"
            if unit and unit not in _NO_UNIT:
                return None  # Keep current — physics data with meaningful unit
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

    # R4: Fit-child promotion — quantity leaf under *_fit parent → fit_artifact
    # Pass 1 catches most fit children via path-pattern, but some leaves
    # (e.g. those with physics units) fall through to quantity.  When the
    # parent's name ends in ``_fit``, they are provenance artifacts.
    if (
        current_category == "quantity"
        and parent_name
        and _FIT_PARENT_RE.search(parent_name)
    ):
        return "fit_artifact"

    return None


__all__ = [
    "COORDINATE_SEGMENTS",
    "FIT_CHILD_SEGMENTS",
    "FIT_DIAGNOSTIC_SEGMENTS",
    "GEOMETRY_EXCLUSION_PATTERNS",
    "GEOMETRY_PATH_PATTERNS",
    "INTEGER_TYPES",
    "PHYSICS_LEAF_TYPES",
    "SPATIAL_UNITS",
    "STRING_TYPES",
    "STRUCTURAL_KEYWORDS",
    "STRUCTURE_TYPES",
    "classify_node_pass1",
    "classify_node_pass2",
]
