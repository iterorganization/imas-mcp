"""Gold-standard benchmark queries for IMAS DD search quality.

Each query has expected paths that a correct search should return in its
top results.  Queries span 6 categories to test different search modes.
Paths are DD-version-independent (omit version-specific suffixes).

The benchmark set is deliberately small (30 queries) to enable fast
iteration during development.  It is stratified across categories to
catch regressions in different search capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkQuery:
    """A single benchmark query with gold-standard expected paths."""

    query_text: str
    expected_paths: list[str]
    category: str
    notes: str = ""


# ── Category 1: Exact concept match ──────────────────────────────────────────
# Queries where the user asks for a well-known physics quantity by name.
# The expected path should appear at rank 1 for a good search engine.

EXACT_CONCEPT_QUERIES = [
    BenchmarkQuery(
        query_text="electron temperature",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "core_transport/model/profiles_1d/electrons/energy/d",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="plasma current",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/ip",
            "summary/global_quantities/ip/value",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="electron density",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
            "core_profiles/profiles_1d/electrons/density_thermal",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="safety factor profile",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/q",
            "core_profiles/profiles_1d/q",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="toroidal magnetic field on axis",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
            "summary/global_quantities/b0/value",
        ],
        category="exact_concept",
    ),
]

# ── Category 2: Disambiguating queries ───────────────────────────────────────
# Queries that target a specific IDS context, not just the physics concept.

DISAMBIGUATING_QUERIES = [
    BenchmarkQuery(
        query_text="electron temperature from ECE diagnostic",
        expected_paths=[
            "ece/channel/t_e",
        ],
        category="disambiguating",
        notes="Should prefer ece IDS over core_profiles",
    ),
    BenchmarkQuery(
        query_text="electron density from interferometry",
        expected_paths=[
            "interferometer/channel/n_e_line",
            "interferometer/channel/n_e_line_average",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="ion temperature from charge exchange",
        expected_paths=[
            "charge_exchange/channel/ion/temperature",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="magnetic flux in equilibrium profiles",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="radiated power from bolometry",
        expected_paths=[
            "bolometer/channel/power",
        ],
        category="disambiguating",
    ),
]

# ── Category 3: Structural / path queries ────────────────────────────────────
# Queries using exact or partial path notation.  These should use path
# lookup / text matching, not vector search.

STRUCTURAL_QUERIES = [
    BenchmarkQuery(
        query_text="equilibrium/time_slice/profiles_1d/psi",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="core_profiles/profiles_1d/electrons/temperature",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="magnetics/flux_loop",
        expected_paths=[
            "magnetics/flux_loop",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="wall/description_2d",
        expected_paths=[
            "wall/description_2d",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="pf_active/coil/current",
        expected_paths=[
            "pf_active/coil/current",
        ],
        category="structural",
    ),
]

# ── Category 4: Abbreviation / synonym queries ──────────────────────────────
# Queries using common physics abbreviations that the search engine should
# resolve to the correct IMAS paths.

ABBREVIATION_QUERIES = [
    BenchmarkQuery(
        query_text="Ip",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/ip",
            "summary/global_quantities/ip/value",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="Te profile",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="ne",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="q profile",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/q",
            "core_profiles/profiles_1d/q",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="Zeff",
        expected_paths=[
            "core_profiles/profiles_1d/zeff",
            "summary/global_quantities/zeff/value",
        ],
        category="abbreviation",
    ),
]

# ── Category 5: Accessor-oriented queries ────────────────────────────────────
# Queries about child/accessor properties of parent concept nodes.
# After Phase 7 (accessor routing), these should surface the parent node.

ACCESSOR_QUERIES = [
    BenchmarkQuery(
        query_text="radius of X-point",
        expected_paths=[
            "equilibrium/time_slice/boundary/x_point",
        ],
        category="accessor",
        notes="r child of x_point; parent should be surfaced",
    ),
    BenchmarkQuery(
        query_text="vertical position of magnetic axis",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis",
        ],
        category="accessor",
        notes="z child of magnetic_axis",
    ),
    BenchmarkQuery(
        query_text="toroidal angle of strike point",
        expected_paths=[
            "equilibrium/time_slice/boundary/strike_point",
        ],
        category="accessor",
        notes="phi child of strike_point",
    ),
    BenchmarkQuery(
        query_text="time base for electron temperature",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "core_profiles/profiles_1d/time",
        ],
        category="accessor",
        notes="time child — should surface parent or profiles_1d",
    ),
    BenchmarkQuery(
        query_text="measured value of loop voltage",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/v_loop",
            "summary/global_quantities/v_loop/value",
        ],
        category="accessor",
        notes="measured/value children of v_loop",
    ),
]

# ── Category 6: Cross-domain queries ─────────────────────────────────────────
# Queries that span multiple IDSs or physics domains.

CROSS_DOMAIN_QUERIES = [
    BenchmarkQuery(
        query_text="bootstrap current",
        expected_paths=[
            "core_transport/model/profiles_1d/j_bootstrap",
            "core_profiles/profiles_1d/j_bootstrap",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="plasma boundary shape",
        expected_paths=[
            "equilibrium/time_slice/boundary/outline/r",
            "equilibrium/time_slice/boundary/outline",
            "equilibrium/time_slice/boundary",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="neutral beam injection power",
        expected_paths=[
            "nbi/unit/power_launched",
            "summary/global_quantities/power_nbi/value",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="poloidal beta",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_pol",
            "summary/global_quantities/beta_pol/value",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="separatrix last closed flux surface",
        expected_paths=[
            "equilibrium/time_slice/boundary/psi",
            "equilibrium/time_slice/boundary/psi_norm",
            "equilibrium/time_slice/boundary_separatrix",
        ],
        category="cross_domain",
    ),
]

# ── Assembled query sets ─────────────────────────────────────────────────────

ALL_QUERIES: list[BenchmarkQuery] = (
    EXACT_CONCEPT_QUERIES
    + DISAMBIGUATING_QUERIES
    + STRUCTURAL_QUERIES
    + ABBREVIATION_QUERIES
    + ACCESSOR_QUERIES
    + CROSS_DOMAIN_QUERIES
)

# Queries suitable for vector search benchmarks (exclude structural paths)
SEMANTIC_QUERIES: list[BenchmarkQuery] = (
    EXACT_CONCEPT_QUERIES
    + DISAMBIGUATING_QUERIES
    + ABBREVIATION_QUERIES
    + ACCESSOR_QUERIES
    + CROSS_DOMAIN_QUERIES
)

# Queries suitable for text/BM25 benchmarks (all queries)
TEXT_QUERIES: list[BenchmarkQuery] = ALL_QUERIES

# Queries that are exact path lookups
PATH_QUERIES: list[BenchmarkQuery] = STRUCTURAL_QUERIES

CATEGORY_NAMES = [
    "exact_concept",
    "disambiguating",
    "structural",
    "abbreviation",
    "accessor",
    "cross_domain",
]
