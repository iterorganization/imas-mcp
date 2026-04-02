"""Gold-standard benchmark queries for IMAS DD search quality.

Each query has expected paths that a correct search should return in its
top results.  Queries span 6 categories to test different search modes.
Paths are validated against DD 4.1.0 (current version) — all expected
paths are non-deprecated and exist in the graph.

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
            "plasma_profiles/ggd/electrons/temperature",
            "plasma_profiles/profiles_1d/electrons/temperature",
            "summary/local/itb/t_e",
            "summary/line_average/t_e",
        ],
        category="exact_concept",
        notes="multiple IDS paths are valid; all represent electron temperature",
    ),
    BenchmarkQuery(
        query_text="plasma current",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/ip",
            "summary/global_quantities/ip",
        ],
        category="exact_concept",
        notes="summary/global_quantities/ip/value is template-enriched; parent is the concept",
    ),
    BenchmarkQuery(
        query_text="electron density",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
            "core_profiles/profiles_1d/electrons/density_thermal",
            "edge_profiles/profiles_1d/electrons/density",
            "summary/line_average/n_e",
            "summary/local/magnetic_axis/n_e",
        ],
        category="exact_concept",
        notes="multiple IDS contain electron density; all are valid",
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
            "equilibrium/time_slice/global_quantities/magnetic_axis",
            "equilibrium/vacuum_toroidal_field",
            "summary/global_quantities/b0",
        ],
        category="exact_concept",
        notes="b0/value is template-enriched; parent b0 is the concept",
    ),
]

# ── Category 2: Disambiguating queries ───────────────────────────────────────
# Queries that target a specific IDS context, not just the physics concept.

DISAMBIGUATING_QUERIES = [
    BenchmarkQuery(
        query_text="electron temperature from ECE diagnostic",
        expected_paths=[
            "ece/channel/t_radiation",
            "ece/t_radiation_central",
        ],
        category="disambiguating",
        notes="t_e deprecated in 4.0.0; t_radiation is the current path",
    ),
    BenchmarkQuery(
        query_text="electron density from interferometry",
        expected_paths=[
            "interferometer/channel/n_e_line",
            "interferometer/channel/n_e_line_average",
            "core_profiles/profiles_1d/electrons/density",
        ],
        category="disambiguating",
        notes="interferometer paths canonical; core_profiles density acceptable",
    ),
    BenchmarkQuery(
        query_text="ion temperature from charge exchange",
        expected_paths=[
            "charge_exchange/channel/ion/t_i",
            "charge_exchange/channel/t_i_average",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="magnetic flux in equilibrium profiles",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_1d",
        ],
        category="disambiguating",
        notes="psi is canonical; profiles_1d is parent",
    ),
    BenchmarkQuery(
        query_text="radiated power from bolometry",
        expected_paths=[
            "bolometer/power_radiated_total",
            "bolometer/power_radiated_inside_lcfs",
        ],
        category="disambiguating",
        notes="channel/power deprecated in 4.1.0; power_radiated_total is current",
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
            "summary/global_quantities/ip",
            "magnetics/ip",
        ],
        category="abbreviation",
        notes="ip/value is template-enriched; parent ip is the concept",
    ),
    BenchmarkQuery(
        query_text="Te profile",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "core_profiles/profiles_1d/electrons/temperature_fit",
            "core_profiles/profiles_1d/t_i_average",
            "edge_profiles/profiles_1d/electrons/temperature",
            "summary/line_average/t_e",
        ],
        category="abbreviation",
        notes="various temperature-related paths are acceptable",
    ),
    BenchmarkQuery(
        query_text="ne",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
            "edge_profiles/profiles_1d/electrons/density",
            "summary/line_average/n_e",
            "summary/local/magnetic_axis/n_e",
        ],
        category="abbreviation",
        notes="various density paths are acceptable",
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
            "summary/line_average/zeff",
            "edge_profiles/ggd/zeff",
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
            "equilibrium/time_slice/constraints/x_point",
            "pulse_schedule/position_control/x_point",
            "equilibrium/time_slice/boundary/x_point",
            "summary/boundary/x_point_main",
        ],
        category="accessor",
        notes="x_point parents or summary, not accessor children",
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
            "equilibrium/time_slice/constraints/strike_point",
            "pulse_schedule/position_control/strike_point",
            "summary/boundary/strike_point_outer_z",
            "summary/boundary/strike_point_inner_r",
        ],
        category="accessor",
        notes="summary boundary strike points are valid alternatives",
    ),
    BenchmarkQuery(
        query_text="time base for electron temperature",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "core_profiles/profiles_1d/time",
            "summary/local/itb/t_e",
        ],
        category="accessor",
        notes="time child — should surface parent, summary t_e also valid",
    ),
    BenchmarkQuery(
        query_text="measured value of loop voltage",
        expected_paths=[
            "summary/global_quantities/v_loop",
            "core_profiles/global_quantities/v_loop",
        ],
        category="accessor",
        notes="v_loop/value is template-enriched; parent v_loop is the concept",
    ),
]

# ── Category 6: Cross-domain queries ─────────────────────────────────────────
# Queries that span multiple IDSs or physics domains.

CROSS_DOMAIN_QUERIES = [
    BenchmarkQuery(
        query_text="bootstrap current",
        expected_paths=[
            "core_profiles/profiles_1d/j_bootstrap",
            "edge_profiles/profiles_1d/j_bootstrap",
            "core_profiles/global_quantities/current_bootstrap",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="plasma boundary shape",
        expected_paths=[
            "equilibrium/time_slice/boundary/outline",
            "equilibrium/time_slice/boundary",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="neutral beam injection power",
        expected_paths=[
            "nbi/unit/power_launched",
            "summary/heating_current_drive/nbi/power_launched",
            "summary/heating_current_drive/power_launched_nbi",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="poloidal beta",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_pol",
            "summary/global_quantities/beta_pol",
        ],
        category="cross_domain",
        notes="beta_pol/value is template-enriched; parent beta_pol is the concept",
    ),
    BenchmarkQuery(
        query_text="separatrix last closed flux surface",
        expected_paths=[
            "equilibrium/time_slice/boundary",
            "equilibrium/time_slice/boundary/psi",
            "equilibrium/time_slice/boundary/psi_norm",
            "summary/local/separatrix",
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


# ── Standalone MRR computation functions ─────────────────────────────────────


def compute_mrr(
    ranked_results: list[str],
    expected: list[str],
    allow_prefix: bool = False,
) -> float:
    """Compute reciprocal rank for a single query.

    Parameters
    ----------
    ranked_results:
        List of path IDs in ranked order (best first).
    expected:
        List of acceptable correct paths.
    allow_prefix:
        If True, a result is considered correct when it starts with an
        expected path followed by ``/``, or vice-versa.  Useful for
        accessor paths like ``ip`` → ``ip/data``.

    Returns
    -------
    1/rank of the first correct result, or 0.0 if none found.
    """
    for rank, result in enumerate(ranked_results, 1):
        for exp in expected:
            if result == exp:
                return 1.0 / rank
            if allow_prefix and (
                result.startswith(exp + "/") or exp.startswith(result + "/")
            ):
                return 1.0 / rank
    return 0.0


def compute_category_mrr(
    results: dict[str, list[str]],
    queries: list[BenchmarkQuery],
) -> float:
    """Compute average MRR across a list of benchmark queries.

    Parameters
    ----------
    results:
        Mapping of ``query_text`` → list of ranked result path IDs.
    queries:
        Benchmark queries to evaluate.

    Returns
    -------
    Average reciprocal rank (0.0–1.0).
    """
    if not queries:
        return 0.0

    mrrs = [
        compute_mrr(results.get(q.query_text, []), q.expected_paths) for q in queries
    ]
    return sum(mrrs) / len(mrrs)
