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
        # W37: structural config flags / enumerations
        "closed",
        "spacing",
        "transformation",
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
# Hardware / engineering metadata constants
# ──────────────────────────────────────────────────────────────────

#: Non-leaf path segments identifying hardware-configuration subtrees.
#: All children of these containers describe instrument internals
#: (ADC settings, detector pixel layout, detector image geometry)
#: rather than measured or computed physics quantities.
_HARDWARE_SUBTREE_SEGMENTS: frozenset[str] = frozenset(
    {"adc", "detector_layout", "detector_image"}
)

#: Top-level IDS names that are pure scratch / user-defined storage.
#: Every node under these IDSs is provenance / scratch metadata, not
#: physics — e.g. ``temporary/constant_float5d/value`` is a generic
#: typed-storage container with arbitrary user content.
_SCRATCH_IDS: frozenset[str] = frozenset({"temporary"})

#: Path-segment markers for legacy temporary-storage subtrees that may
#: appear in older DD versions (``temporary_storage_*`` containers,
#: e.g. inside the ``summary`` IDS).  Any ancestor segment matching this
#: prefix routes the node to ``metadata``.
_TEMPORARY_STORAGE_PREFIX: str = "temporary_storage"

#: Integer identifier leaf names that appear inside ``data_entry_identifier``
#: structs (IMAS path parent segment ``data_entry``).  These fields pin a
#: reproducible dataset in the IMAS back-end (run number, shot number, pulse
#: number) — they are **not** physics quantities and must not enter the
#: StandardName generation pipeline.
#:
#: String siblings (``machine``, ``user``, ``pulse_type``) are already caught
#: by Rule 4 (STRING_TYPES → metadata), so only integer slots are listed here.
_DATA_ENTRY_ID_LEAVES: frozenset[str] = frozenset(
    {"run", "shot", "pulse", "occurrence", "run_index", "version"}
)

# ──────────────────────────────────────────────────────────────────
# INT_0D physics allowlist & boolean flag constants
# ──────────────────────────────────────────────────────────────────

#: INT_0D leaf names that represent genuine physics integers and must
#: remain classified as ``quantity`` under the revised Rule 13.  New
#: physics integers can be added one-line at a time.
_PHYSICS_INT_LEAVES: frozenset[str] = frozenset(
    {
        # Nuclear/atomic composition
        "z_n",  # Nuclear charge
        "atoms_n",  # Atoms in molecule
        # Mode numbers
        "n_tor",  # Toroidal mode number
        "n_phi",  # Toroidal mode number (alternative)
        "m_pol",  # Poloidal mode number
        "toroidal_mode",  # MHD perturbation mode index
        # Hardware counts
        "turns",  # Coil turns (including sign → physics)
        "poloidal_turns",  # Flux tube turns
        "nuclei_n",  # Target nuclei count
        "modules_n",  # Blanket module count
        "modules_pol_n",  # Poloidal module count
        "coils_n",  # Coil count
        "pixels_n_horizontal",  # Detector pixel count
        "pixels_n_vertical",  # Detector pixel count
        "voxels_n",  # Voxel count
        "fluids_n",  # Number of fluids in model
        # Convergence
        "iterations_n",  # Convergence iteration count
        # Composition
        "fraction",  # Species fraction (sometimes INT)
    }
)

#: INT_0D leaves that encode boolean flags or configuration enumerations.
#: These are treated as ``structural`` — they control model behaviour
#: rather than represent physics quantities.
_BOOLEAN_FLAG_LEAVES: frozenset[str] = frozenset(
    {
        "is_periodic",
        "is_neutral",
        "reciprocating",
        "adiabatic_electrons",
        "varying_n_tor",
        "distribution_assumption",
        "execution_mode",
        "collisions_finite_larmor_radius",
    }
)

#: INT_0D leaf names that are IDS-reference identifiers (occurrence, pulse,
#: shot) when found under identifier-like parent segments.
_IDS_REF_LEAVES: frozenset[str] = frozenset({"occurrence", "pulse", "shot", "run"})

#: Parent segments that signal an IDS cross-reference struct.
_IDS_REF_PARENTS: frozenset[str] = frozenset({"equilibrium_id", "parent_entry"})

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

#: Regex matching transport_solver_numerics boundary-condition subtree.
#: Covers both top-level ``boundary_conditions_*`` (plural, ion/electrons/...)
#: AND nested ``solver_1d/.../boundary_condition`` (singular) — both are
#: solver-internal numerical configuration, not physics quantities.
_TRANSPORT_SOLVER_BC_RE: re.Pattern[str] = re.compile(
    r"^transport_solver_numerics/(?:[^/]*boundary_conditions[^/]*/"
    r"|solver_1d(?:/[^/]+)*?/boundary_condition(?:/|$))"
)

#: Regex matching transport_solver_numerics solver_1d coefficient leaves.
#: Covers e.g. ``transport_solver_numerics/solver_1d/*/equation/*/coefficient``.
_TRANSPORT_SOLVER_COEFF_RE: re.Pattern[str] = re.compile(
    r"^transport_solver_numerics/solver_1d(?:/[^/]+)*/coefficient[^/]*(?:/.*)?$"
)

#: Regex matching transport_solver_numerics convergence subtree.
#: All paths under ``transport_solver_numerics/convergence/`` are solver
#: convergence diagnostics (residuals, iteration counts, delta_relative,
#: time_step) — numerical artefacts, not physics quantities.
_TRANSPORT_SOLVER_CONVERGENCE_RE: re.Pattern[str] = re.compile(
    r"^transport_solver_numerics/convergence(?:/|$)"
)

# ──────────────────────────────────────────────────────────────────
# Representation constants
# ──────────────────────────────────────────────────────────────────

#: Path fragments indicating GGD / basis-function representation storage.
_REPRESENTATION_PATH_MARKERS: tuple[str, ...] = (
    "grids_ggd/",
    "/ggd_fast/",
    "/grid_subset/",
)

#: Regex matching ``pulse_schedule/.../reference`` or ``.../reference_waveform``
#: subtrees.  These store the commanded waveform representation of an
#: underlying physics quantity (plasma current, densities, heating power, …)
#: — not an independent physics concept.
_PULSE_SCHEDULE_REFERENCE_RE: re.Pattern[str] = re.compile(
    r"^pulse_schedule/.+/reference(?:_waveform)?(?:/.+)?$"
)

#: Regex matching vector-container ``/diamagnetic`` leaves.  These DD paths
#: (e.g. ``core_profiles/profiles_1d/velocity/diamagnetic``) are mis-labelings
#: of drift-decomposition axes — see DD-01 (plan 31 §8, WS-E).  Until the DD
#: is corrected, route to ``representation`` so they are excluded from SN
#: extraction.
_DIAMAGNETIC_AXIS_RE: re.Pattern[str] = re.compile(
    r"^(?:.*)/(?:velocity|e_field|a_field|j_tot|b_field)/diamagnetic(?:/.+)?$"
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
    ids_root = parts[0] if parts else ""

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

    # Rule 2a: Scratch IDS (`temporary` and similar) — every node is
    # user-defined typed-storage scratch, never a physics quantity.
    if ids_root in _SCRATCH_IDS:
        return "metadata"

    # Rule 2b: Legacy temporary-storage subtree (any ancestor segment
    # starting with ``temporary_storage``).  Catches DD3-era paths like
    # ``summary/.../temporary_storage_integer_value``.
    if any(seg.startswith(_TEMPORARY_STORAGE_PREFIX) for seg in parts):
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

    # Rule 6 + R13a: validity / validity_timed / *_validity → structural
    if (
        last_seg == "validity"
        or last_seg == "validity_timed"
        or last_seg.endswith("_validity")
    ):
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

    # Rule F3: transport_solver_numerics/boundary_conditions_*/… → fit_artifact
    # Solver boundary-condition configuration nodes (value, rho_tor_norm,
    # identifier, …) are numerical-solver internals, not physics concepts.
    if _TRANSPORT_SOLVER_BC_RE.match(path):
        return "fit_artifact"

    # Rule F4: transport_solver_numerics/solver_1d/*/coefficient* → fit_artifact
    # Finite-volume / finite-difference PDE coefficients are solver internals.
    if _TRANSPORT_SOLVER_COEFF_RE.match(path):
        return "fit_artifact"

    # Rule F5: transport_solver_numerics/convergence/* → fit_artifact
    # Solver convergence diagnostics — residuals (delta_relative), iteration
    # counts (iterations_n), per-equation containers — are numerical artefacts
    # of the iterative solver, not independent physics quantities.
    if _TRANSPORT_SOLVER_CONVERGENCE_RE.match(path):
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

    # Rule R3: pulse_schedule/.../reference(_waveform) → representation
    # Commanded waveform representations of underlying physics quantities.
    if _PULSE_SCHEDULE_REFERENCE_RE.match(path):
        return "representation"

    # Rule R4: vector-container /diamagnetic axis → representation
    # DD mis-labeling (tracked upstream as DD-01 / plan 31 §8, WS-E);
    # exclude from SN extraction until DD is corrected.
    if _DIAMAGNETIC_AXIS_RE.match(path):
        return "representation"

    # ── Hardware / engineering metadata rules ──

    # Rule M1: Hardware-configuration subtree → metadata
    # Children of ADC, detector_layout, detector_image containers describe
    # instrument internals, not measured/computed physics quantities.
    if any(seg in _HARDWARE_SUBTREE_SEGMENTS for seg in parts[:-1]):
        return "metadata"

    # Rule M2: Engineering operational limits → metadata
    # Names ending in _limit_max / _limit_min define machine-protection
    # envelopes (max current, max temperature, …), not physics measurements.
    if last_seg.endswith("_limit_max") or last_seg.endswith("_limit_min"):
        return "metadata"

    # Rule M3: Instrument-specification suffixes → metadata
    # E.g. bandwidth_3db describes the probe's frequency response, not
    # a plasma physics quantity.
    if last_seg.endswith("_3db"):
        return "metadata"

    # Rule M4: data_entry_identifier integer slots → metadata
    # INT leaves (run, shot, pulse, …) that are direct children of a
    # ``data_entry`` parent segment are database-access coordinates used to
    # pin a reproducible dataset.  They are provenance metadata, NOT physics
    # quantities, and must not enter the StandardName generation pipeline.
    # String siblings (machine, user, pulse_type) are already caught by
    # Rule 4 (STRING_TYPES → metadata).
    if (
        len(parts) >= 2
        and parts[-2] == "data_entry"
        and last_seg in _DATA_ENTRY_ID_LEAVES
    ):
        return "metadata"

    # Rule M5: Epoch timestamp subtree → metadata
    # Epoch timestamps (seconds, nanoseconds since Unix epoch) under
    # ``pulse_time_*_epoch/`` are dataset-level temporal coordinates,
    # not physics quantities.
    if last_seg in ("seconds", "nanoseconds") and any(
        "epoch" in seg for seg in parts[:-1]
    ):
        return "metadata"

    # Rule M6: IDS-reference occurrence/pulse/shot → metadata
    # Outside ``data_entry/``, these fields still serve as dataset
    # cross-reference metadata (e.g. ``equilibrium_id/occurrence``).
    if last_seg in _IDS_REF_LEAVES and len(parts) >= 2:
        # Direct parent is an IDS-reference struct
        if parts[-2] in _IDS_REF_PARENTS:
            return "metadata"
        # Top-level children of dataset_description / summary
        if len(parts) == 2 and parts[0] in ("dataset_description", "summary"):
            return "metadata"

    # Rule M7: ``summary/<topic>/occurrence`` flag containers → metadata.
    # The ``summary`` IDS uses ``occurrence`` STRUCTURE nodes (with quirky
    # ``Hz`` units inherited from a time-reference template) to indicate
    # whether a feature was active during the pulse — pure flags
    # ("Flag set to 1 if …, 0 otherwise"), not physics quantities.
    if len(parts) == 3 and parts[0] == "summary" and last_seg == "occurrence":
        return "metadata"

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

    # Rule R13d: INT_0D boolean flags → structural
    # Known boolean flags and configuration enumerations are not physics
    # quantities — they control model behaviour.
    if dt == "INT_0D" and last_seg in _BOOLEAN_FLAG_LEAVES:
        return "structural"

    # Rule R13e: INT_0D /value containers → structural
    # Generic typed containers (e.g. summary/*/value, temporary/*/value)
    # where the physics meaning lives in the parent structure.
    if dt == "INT_0D" and last_seg == "value":
        return "structural"

    # Rule 13 (revised): INT_0D without unit — physics allowlist
    # Flipped default: unrecognised INT_0D → structural (conservative).
    # False negatives (missed physics) are recoverable by extending
    # _PHYSICS_INT_LEAVES; false positives (SN pollution) are not.
    if dt == "INT_0D" and (not unit_str or unit_str in _NO_UNIT):
        # 13a: Allowlisted physics integers
        if last_seg in _PHYSICS_INT_LEAVES:
            return "quantity"
        # 13b: Default to structural
        return "structural"

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
    "_BOOLEAN_FLAG_LEAVES",
    "_DATA_ENTRY_ID_LEAVES",
    "_PHYSICS_INT_LEAVES",
    "_SCRATCH_IDS",
    "_TEMPORARY_STORAGE_PREFIX",
    "classify_node_pass1",
    "classify_node_pass2",
]
