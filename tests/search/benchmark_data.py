"""Gold-standard benchmark queries for IMAS DD search quality.

Each query has expected paths that a correct search should return in its
top results.  Queries span 7 categories to test different search modes.
Paths are validated against DD 4.1.0 (current version) — all expected
paths are non-deprecated and exist in the graph.

The benchmark set contains ~200 queries stratified across categories to
catch regressions in different search capabilities:
  - EXACT_CONCEPT_QUERIES:    ~40 queries
  - DISAMBIGUATING_QUERIES:   ~20 queries
  - STRUCTURAL_QUERIES:       ~20 queries
  - ABBREVIATION_QUERIES:     ~40 queries
  - ACCESSOR_QUERIES:         ~20 queries
  - CROSS_DOMAIN_QUERIES:     ~40 queries
  - EDGE_CASE_QUERIES:        ~20 queries
"""

from __future__ import annotations

from dataclasses import dataclass


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
    BenchmarkQuery(
        query_text="toroidal magnetic field",
        expected_paths=[
            "equilibrium/vacuum_toroidal_field/b0",
            "summary/global_quantities/b0",
            "magnetics/b_field_tor_probe",
            "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
        ],
        category="exact_concept",
        notes="b0 and b_field_tor are canonical toroidal field paths",
    ),
    BenchmarkQuery(
        query_text="bootstrap current density",
        expected_paths=[
            "core_profiles/profiles_1d/j_bootstrap",
            "edge_profiles/profiles_1d/j_bootstrap",
        ],
        category="exact_concept",
        notes="j_bootstrap is the bootstrap current density profile",
    ),
    BenchmarkQuery(
        query_text="resistivity",
        expected_paths=[
            "core_profiles/profiles_1d/conductivity_parallel",
            "core_transport/model/profiles_1d/conductivity_parallel",
            "wall/description_2d/vessel/unit/annular/resistivity",
            "pf_passive/loop/resistivity",
        ],
        category="exact_concept",
        notes="conductivity_parallel is the inverse; wall/pf resistivity also valid",
    ),
    BenchmarkQuery(
        query_text="safety factor",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/q",
            "core_profiles/profiles_1d/q",
            "summary/local/magnetic_axis/q",
        ],
        category="exact_concept",
        notes="q profile in equilibrium and core_profiles; scalar at axis",
    ),
    BenchmarkQuery(
        query_text="elongation",
        expected_paths=[
            "equilibrium/time_slice/boundary/elongation",
            "summary/boundary/elongation",
            "pulse_schedule/position_control/elongation",
        ],
        category="exact_concept",
        notes="elongation in boundary, summary, and pulse schedule",
    ),
    BenchmarkQuery(
        query_text="ion temperature",
        expected_paths=[
            "core_profiles/profiles_1d/ion/temperature",
            "edge_profiles/profiles_1d/ion/temperature",
            "summary/local/itb/t_i",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="toroidal rotation",
        expected_paths=[
            "core_profiles/profiles_1d/ion/rotation/toroidal",
            "core_profiles/profiles_1d/rotation_frequency_tor_sonic",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="effective charge",
        expected_paths=[
            "core_profiles/profiles_1d/zeff",
            "summary/local/itb/zeff",
            "summary/line_average/zeff",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="poloidal beta",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_pol",
            "summary/global_quantities/beta_pol",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="toroidal beta",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_tor",
            "summary/global_quantities/beta_tor",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="magnetic axis position",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/r",
            "equilibrium/time_slice/global_quantities/magnetic_axis/z",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="internal inductance",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/li",
            "summary/global_quantities/li",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="loop voltage",
        expected_paths=[
            "summary/global_quantities/v_loop",
            "equilibrium/time_slice/global_quantities/v_loop",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="diamagnetic flux",
        expected_paths=[
            "magnetics/flux_loop/flux/data",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="radiated power",
        expected_paths=[
            "summary/global_quantities/power_radiated",
            "bolometer/channel/radiated_power",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="ohmic power",
        expected_paths=[
            "summary/global_quantities/power_ohm",
            "core_sources/source/profiles_1d/electrons/power_inside",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="stored energy",
        expected_paths=[
            "summary/global_quantities/energy_diamagnetic",
            "equilibrium/time_slice/global_quantities/w_mhd",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="line integrated density",
        expected_paths=[
            "interferometer/channel/n_e_line/data",
            "summary/line_average/n_e",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="neutral beam power",
        expected_paths=[
            "nbi/unit/power_launched/data",
            "summary/heating_current_drive/power_nbi",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="ECRH power",
        expected_paths=[
            "ec_launchers/beam/power_launched/data",
            "summary/heating_current_drive/power_ec",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="ICRH power",
        expected_paths=[
            "ic_antennas/antenna/power_launched/data",
            "summary/heating_current_drive/power_ic",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="plasma volume",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/volume",
            "summary/global_quantities/volume",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="greenwald density",
        expected_paths=[
            "summary/global_quantities/n_e_greenwald",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="confinement time",
        expected_paths=[
            "summary/global_quantities/tau_energy",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="q95",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/q_95",
            "summary/global_quantities/q_95",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="magnetic shear",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/magnetic_shear",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="pressure profile",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/pressure",
            "core_profiles/profiles_1d/pressure_thermal",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="current density profile",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/j_tor",
            "equilibrium/time_slice/profiles_1d/j_parallel",
            "core_profiles/profiles_1d/j_total",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="pedestal temperature",
        expected_paths=[
            "summary/local/pedestal/t_e",
            "summary/local/pedestal/t_i",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="pedestal density",
        expected_paths=[
            "summary/local/pedestal/n_e",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="triangularity",
        expected_paths=[
            "equilibrium/time_slice/boundary/triangularity_upper",
            "equilibrium/time_slice/boundary/triangularity_lower",
            "summary/boundary/triangularity",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="poloidal flux",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_2d/psi",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="normalized beta",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_normal",
            "summary/global_quantities/beta_normal",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="electron density profile",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
            "edge_profiles/profiles_1d/electrons/density",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="thermal conductivity",
        expected_paths=[
            "core_transport/model/profiles_1d/electrons/energy/conductivity",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="particle diffusivity",
        expected_paths=[
            "core_transport/model/profiles_1d/electrons/particles/d",
        ],
        category="exact_concept",
    ),
    BenchmarkQuery(
        query_text="minor radius",
        expected_paths=[
            "equilibrium/time_slice/boundary/minor_radius",
            "summary/boundary/minor_radius",
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
    BenchmarkQuery(
        query_text="NBI beam energy",
        expected_paths=[
            "nbi/unit/energy/data",
        ],
        category="disambiguating",
        notes="NBI beam particle energy, not plasma stored energy",
    ),
    BenchmarkQuery(
        query_text="Thomson scattering electron temperature",
        expected_paths=[
            "thomson_scattering/channel/t_e/data",
        ],
        category="disambiguating",
        notes="Te specifically from Thomson scattering diagnostic",
    ),
    BenchmarkQuery(
        query_text="Thomson scattering electron density",
        expected_paths=[
            "thomson_scattering/channel/n_e/data",
        ],
        category="disambiguating",
        notes="ne specifically from Thomson scattering diagnostic",
    ),
    BenchmarkQuery(
        query_text="magnetic probe field",
        expected_paths=[
            "magnetics/b_field_pol_probe/field/data",
        ],
        category="disambiguating",
        notes="B-field from poloidal field probes",
    ),
    BenchmarkQuery(
        query_text="MSE pitch angle",
        expected_paths=[
            "mse/channel/polarisation_angle/data",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="soft X-ray brightness",
        expected_paths=[
            "soft_x_rays/channel/brightness/data",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="CXRS ion temperature",
        expected_paths=[
            "charge_exchange/channel/ion/temperature",
        ],
        category="disambiguating",
        notes="Ti from charge exchange recombination spectroscopy",
    ),
    BenchmarkQuery(
        query_text="equilibrium boundary shape",
        expected_paths=[
            "equilibrium/time_slice/boundary/outline/r",
            "equilibrium/time_slice/boundary/outline/z",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="separatrix position",
        expected_paths=[
            "equilibrium/time_slice/boundary_separatrix/outline/r",
            "equilibrium/time_slice/boundary_separatrix/outline/z",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="X-point position",
        expected_paths=[
            "equilibrium/time_slice/boundary/x_point/r",
            "equilibrium/time_slice/boundary/x_point/z",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="PF coil current",
        expected_paths=[
            "pf_active/coil/current/data",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="gas injection rate",
        expected_paths=[
            "gas_injection/pipe/flow_rate/data",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="core electron source",
        expected_paths=[
            "core_sources/source/profiles_1d/electrons/particles",
            "core_sources/source/profiles_1d/electrons/energy",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="edge transport coefficients",
        expected_paths=[
            "edge_transport/model/ggd/electrons/energy/d",
            "edge_transport/model/ggd/electrons/particles/d",
        ],
        category="disambiguating",
    ),
    BenchmarkQuery(
        query_text="interferometer phase",
        expected_paths=[
            "interferometer/channel/phase/data",
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
    BenchmarkQuery(
        query_text="equilibrium/time_slice/global_quantities",
        expected_paths=[
            "equilibrium/time_slice/global_quantities",
        ],
        category="structural",
        notes="exact subtree path",
    ),
    BenchmarkQuery(
        query_text="core_profiles/profiles_1d/ion",
        expected_paths=[
            "core_profiles/profiles_1d/ion",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="magnetics/b_field_pol_probe",
        expected_paths=[
            "magnetics/b_field_pol_probe",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="summary/global_quantities",
        expected_paths=[
            "summary/global_quantities",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="nbi/unit",
        expected_paths=[
            "nbi/unit",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="core_transport/model/profiles_1d",
        expected_paths=[
            "core_transport/model/profiles_1d",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="pf_active/coil",
        expected_paths=[
            "pf_active/coil",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="ec_launchers/beam",
        expected_paths=[
            "ec_launchers/beam",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="core_sources/source",
        expected_paths=[
            "core_sources/source",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="profiles_1d/electrons/temperature",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "edge_profiles/profiles_1d/electrons/temperature",
        ],
        category="structural",
        notes="partial path matching across multiple IDS",
    ),
    BenchmarkQuery(
        query_text="equilibrium/time_slice/profiles_1d",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="equilibrium/time_slice/profiles_2d",
        expected_paths=[
            "equilibrium/time_slice/profiles_2d",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="equilibrium/time_slice/boundary",
        expected_paths=[
            "equilibrium/time_slice/boundary",
        ],
        category="structural",
    ),
    BenchmarkQuery(
        query_text="summary/local/pedestal",
        expected_paths=[
            "summary/local/pedestal",
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
    BenchmarkQuery(
        query_text="b0",
        expected_paths=[
            "equilibrium/vacuum_toroidal_field/b0",
            "summary/global_quantities/b0",
            "core_profiles/vacuum_toroidal_field/b0",
        ],
        category="abbreviation",
        notes="b0 is the vacuum toroidal field on axis",
    ),
    BenchmarkQuery(
        query_text="bt",
        expected_paths=[
            "equilibrium/vacuum_toroidal_field/b0",
            "magnetics/b_field_tor_probe",
            "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
        ],
        category="abbreviation",
        notes="bt commonly refers to the toroidal magnetic field",
    ),
    BenchmarkQuery(
        query_text="jt",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/j_tor",
            "equilibrium/time_slice/profiles_2d/j_tor",
        ],
        category="abbreviation",
        notes="jt is the toroidal current density",
    ),
    BenchmarkQuery(
        query_text="zeff",
        expected_paths=[
            "core_profiles/profiles_1d/zeff",
            "summary/line_average/zeff",
            "edge_profiles/ggd/zeff",
        ],
        category="abbreviation",
        notes="lowercase variant; same as Zeff",
    ),
    BenchmarkQuery(
        query_text="psi",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/boundary/psi",
            "core_profiles/profiles_1d/grid/psi",
        ],
        category="abbreviation",
        notes="psi is the poloidal magnetic flux",
    ),
    BenchmarkQuery(
        query_text="B0",
        expected_paths=[
            "equilibrium/vacuum_toroidal_field/b0",
            "summary/global_quantities/b0",
            "core_profiles/vacuum_toroidal_field/b0",
        ],
        category="abbreviation",
        notes="Uppercase variant of b0 — tests case-insensitive matching for vacuum toroidal field",
    ),
    BenchmarkQuery(
        query_text="nbi power",
        expected_paths=[
            "nbi/unit/power_launched",
            "summary/heating_current_drive/nbi/power_launched",
            "summary/heating_current_drive/power_launched_nbi",
        ],
        category="abbreviation",
        notes="Neutral beam injection power — tests abbreviation expansion for NBI",
    ),
    BenchmarkQuery(
        query_text="kappa",
        expected_paths=[
            "equilibrium/time_slice/boundary/elongation",
            "summary/boundary/elongation",
            "pulse_schedule/position_control/elongation",
        ],
        category="abbreviation",
        notes="Elongation via Greek letter kappa — tests abbreviation expansion",
    ),
    BenchmarkQuery(
        query_text="li",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/li",
            "summary/global_quantities/li",
        ],
        category="abbreviation",
        notes="internal inductance",
    ),
    BenchmarkQuery(
        query_text="beta_n",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_normal",
            "summary/global_quantities/beta_normal",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="beta_pol",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/beta_pol",
            "summary/global_quantities/beta_pol",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="Vloop",
        expected_paths=[
            "summary/global_quantities/v_loop",
            "equilibrium/time_slice/global_quantities/v_loop",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="Prad",
        expected_paths=[
            "summary/global_quantities/power_radiated",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="Wmhd",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/w_mhd",
            "summary/global_quantities/energy_diamagnetic",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="j_tor",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/j_tor",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="j_parallel",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/j_parallel",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="rho_tor",
        expected_paths=[
            "core_profiles/profiles_1d/grid/rho_tor",
            "equilibrium/time_slice/profiles_1d/rho_tor",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="tau_e",
        expected_paths=[
            "summary/global_quantities/tau_energy",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="n_e_line",
        expected_paths=[
            "interferometer/channel/n_e_line/data",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="R0",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/r",
        ],
        category="abbreviation",
        notes="major radius / magnetic axis R",
    ),
    BenchmarkQuery(
        query_text="Bt",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
            "summary/global_quantities/b0",
            "tf/b_field_tor_vacuum_r/data",
        ],
        category="abbreviation",
        notes="toroidal magnetic field",
    ),
    BenchmarkQuery(
        query_text="Bp",
        expected_paths=[
            "magnetics/b_field_pol_probe/field/data",
        ],
        category="abbreviation",
        notes="poloidal magnetic field from probes",
    ),
    BenchmarkQuery(
        query_text="Ti",
        expected_paths=[
            "core_profiles/profiles_1d/ion/temperature",
            "charge_exchange/channel/ion/temperature",
            "summary/local/itb/t_i",
        ],
        category="abbreviation",
        notes="ion temperature",
    ),
    BenchmarkQuery(
        query_text="vtor",
        expected_paths=[
            "core_profiles/profiles_1d/ion/rotation/toroidal",
            "core_profiles/profiles_1d/rotation_frequency_tor_sonic",
        ],
        category="abbreviation",
        notes="toroidal rotation velocity",
    ),
    BenchmarkQuery(
        query_text="Pnbi",
        expected_paths=[
            "nbi/unit/power_launched/data",
            "summary/heating_current_drive/power_nbi",
        ],
        category="abbreviation",
        notes="NBI power",
    ),
    BenchmarkQuery(
        query_text="Pec",
        expected_paths=[
            "ec_launchers/beam/power_launched/data",
            "summary/heating_current_drive/power_ec",
        ],
        category="abbreviation",
        notes="ECRH power",
    ),
    BenchmarkQuery(
        query_text="Pic",
        expected_paths=[
            "ic_antennas/antenna/power_launched/data",
            "summary/heating_current_drive/power_ic",
        ],
        category="abbreviation",
        notes="ICRH power",
    ),
    BenchmarkQuery(
        query_text="Pohm",
        expected_paths=[
            "summary/global_quantities/power_ohm",
        ],
        category="abbreviation",
        notes="ohmic power",
    ),
    BenchmarkQuery(
        query_text="q_min",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/q_min/value",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="q_axis",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/q_axis",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="n_greenwald",
        expected_paths=[
            "summary/global_quantities/n_e_greenwald",
        ],
        category="abbreviation",
    ),
    BenchmarkQuery(
        query_text="delta_upper",
        expected_paths=[
            "equilibrium/time_slice/boundary/triangularity_upper",
        ],
        category="abbreviation",
        notes="upper triangularity",
    ),
    BenchmarkQuery(
        query_text="delta_lower",
        expected_paths=[
            "equilibrium/time_slice/boundary/triangularity_lower",
        ],
        category="abbreviation",
        notes="lower triangularity",
    ),
    BenchmarkQuery(
        query_text="Rgeo",
        expected_paths=[
            "equilibrium/time_slice/boundary/geometric_axis/r",
            "summary/boundary/geometric_axis_r",
        ],
        category="abbreviation",
        notes="geometric major radius",
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
    BenchmarkQuery(
        query_text="major radius of geometric axis",
        expected_paths=[
            "equilibrium/time_slice/boundary/geometric_axis/r",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="value of safety factor at 95% flux",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/q_95",
            "summary/global_quantities/q_95",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="toroidal field at magnetic axis",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="R coordinate of boundary outline",
        expected_paths=[
            "equilibrium/time_slice/boundary/outline/r",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="Z coordinate of magnetic axis",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/z",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="time of equilibrium reconstruction",
        expected_paths=[
            "equilibrium/time_slice/time",
            "equilibrium/time",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="power launched by NBI unit",
        expected_paths=[
            "nbi/unit/power_launched/data",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="data from interferometer channel",
        expected_paths=[
            "interferometer/channel/n_e_line/data",
            "interferometer/channel/n_e_line_average/data",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="electron temperature from core profiles",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="ion density in edge profiles",
        expected_paths=[
            "edge_profiles/profiles_1d/ion/density",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="vacuum toroidal field reference value",
        expected_paths=[
            "tf/b_field_tor_vacuum_r/data",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="coil current in PF active system",
        expected_paths=[
            "pf_active/coil/current/data",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="wall position in 2D description",
        expected_paths=[
            "wall/description_2d/limiter/unit/outline/r",
            "wall/description_2d/limiter/unit/outline/z",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="channel brightness from soft X-rays",
        expected_paths=[
            "soft_x_rays/channel/brightness/data",
        ],
        category="accessor",
    ),
    BenchmarkQuery(
        query_text="polarisation angle from MSE diagnostic",
        expected_paths=[
            "mse/channel/polarisation_angle/data",
        ],
        category="accessor",
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
    BenchmarkQuery(
        query_text="radiated power",
        expected_paths=[
            "bolometer/power_radiated_total",
            "bolometer/power_radiated_inside_lcfs",
            "summary/global_quantities/power_radiated",
            "wall/global_quantities/power_radiated",
        ],
        category="cross_domain",
        notes="radiated power from bolometry, summary, and wall IDS",
    ),
    BenchmarkQuery(
        query_text="line integrated density",
        expected_paths=[
            "interferometer/channel/n_e_line",
            "interferometer/channel/n_e_line_average",
        ],
        category="cross_domain",
        notes="line-integrated electron density from interferometry",
    ),
    BenchmarkQuery(
        query_text="major radius",
        expected_paths=[
            "equilibrium/time_slice/boundary/geometric_axis/r",
            "equilibrium/vacuum_toroidal_field/r0",
            "summary/boundary/geometric_axis_r",
        ],
        category="cross_domain",
        notes="geometric axis R and vacuum field reference R0; allow_prefix useful",
    ),
    BenchmarkQuery(
        query_text="magnetic axis",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis",
            "summary/local/magnetic_axis",
        ],
        category="cross_domain",
        notes="magnetic axis position in equilibrium and summary",
    ),
    BenchmarkQuery(
        query_text="divertor heat flux",
        expected_paths=[
            "divertors/divertor/target/power_flux_peak",
            "summary/local/divertor_target/power_flux_peak",
            "divertors/divertor/target/heat_flux_steady_limit_max",
        ],
        category="cross_domain",
        notes="heat flux on divertor targets from divertors and summary IDS",
    ),
    BenchmarkQuery(
        query_text="MHD stability",
        expected_paths=[
            "mhd_linear/time_slice/toroidal_mode/n_tor",
            "mhd_linear/time_slice/toroidal_mode/growth_rate",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="sawtooth crash",
        expected_paths=[
            "summary/local/magnetic_axis/t_e",
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="energy confinement scaling",
        expected_paths=[
            "summary/global_quantities/tau_energy",
            "summary/global_quantities/energy_diamagnetic",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="H-mode transition",
        expected_paths=[
            "summary/local/pedestal/t_e",
            "summary/local/pedestal/n_e",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="plasma shape control",
        expected_paths=[
            "pulse_schedule/position_control/elongation",
            "pulse_schedule/position_control/triangularity_lower",
            "equilibrium/time_slice/boundary/elongation",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="current profile reconstruction",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/j_tor",
            "equilibrium/time_slice/profiles_1d/j_parallel",
            "core_profiles/profiles_1d/j_total",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="pellet injection",
        expected_paths=[
            "pellets/launcher/frequency",
            "pellets/launcher/pellet/velocity_initial",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="runaway electrons",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density_fast",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="impurity density",
        expected_paths=[
            "core_profiles/profiles_1d/ion/density",
            "edge_profiles/profiles_1d/ion/density",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="radial electric field",
        expected_paths=[
            "core_profiles/profiles_1d/e_field/radial",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="ECCD driven current",
        expected_paths=[
            "core_sources/source/profiles_1d/j_parallel",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="flux surface geometry",
        expected_paths=[
            "equilibrium/time_slice/profiles_2d/r",
            "equilibrium/time_slice/profiles_2d/z",
            "equilibrium/time_slice/profiles_2d/psi",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="plasma composition",
        expected_paths=[
            "core_profiles/profiles_1d/ion/element/z_n",
            "core_profiles/profiles_1d/ion/element/a",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="magnetic equilibrium reconstruction",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_2d/psi",
            "equilibrium/time_slice/global_quantities/ip",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="resistive wall mode",
        expected_paths=[
            "mhd_linear/time_slice/toroidal_mode/growth_rate",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="disruption prediction",
        expected_paths=[
            "summary/global_quantities/ip",
            "summary/global_quantities/li",
            "summary/global_quantities/beta_normal",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="particle transport",
        expected_paths=[
            "core_transport/model/profiles_1d/electrons/particles/d",
            "core_transport/model/profiles_1d/electrons/particles/v",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="heat transport coefficients",
        expected_paths=[
            "core_transport/model/profiles_1d/electrons/energy/conductivity",
            "core_transport/model/profiles_1d/ion/energy/conductivity",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="neoclassical transport",
        expected_paths=[
            "core_transport/model/profiles_1d/electrons/energy/conductivity",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="plasma fueling",
        expected_paths=[
            "gas_injection/pipe/flow_rate/data",
            "pellets/launcher/frequency",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="toroidal field coil",
        expected_paths=[
            "tf/coil/conductor/elements/current",
            "tf/b_field_tor_vacuum_r/data",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="vacuum vessel",
        expected_paths=[
            "wall/description_2d/vessel/unit/outline/r",
            "wall/description_2d/vessel/unit/outline/z",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="spectroscopy emission",
        expected_paths=[
            "spectrometer_visible/channel/intensity/data",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="Shafranov shift",
        expected_paths=[
            "equilibrium/time_slice/global_quantities/magnetic_axis/r",
            "equilibrium/time_slice/boundary/geometric_axis/r",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="fast ion distribution",
        expected_paths=[
            "distributions/distribution/profiles_1d/density",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="magnetic island",
        expected_paths=[
            "mhd_linear/time_slice/toroidal_mode/n_tor",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="plasma rotation measurement",
        expected_paths=[
            "core_profiles/profiles_1d/ion/rotation/toroidal",
            "charge_exchange/channel/ion/velocity_tor",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="radiation profile",
        expected_paths=[
            "bolometer/channel/radiated_power",
            "core_sources/source/profiles_1d/electrons/power_inside",
        ],
        category="cross_domain",
    ),
    BenchmarkQuery(
        query_text="edge localized mode",
        expected_paths=[
            "summary/local/pedestal/t_e",
            "summary/local/pedestal/n_e",
        ],
        category="cross_domain",
        notes="ELMs affect pedestal parameters",
    ),
]

# ── Category 7: Edge-case queries ────────────────────────────────────────────
# Queries with misspellings, path-like syntax, boolean notation, or unusual
# casing.  These test robustness of query preprocessing.

EDGE_CASE_QUERIES = [
    BenchmarkQuery(
        query_text="temperture",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "edge_profiles/profiles_1d/electrons/temperature",
        ],
        category="edge_case",
        notes="misspelling of 'temperature' — fuzzy matching should recover",
    ),
    BenchmarkQuery(
        query_text="rho_tor_norm",
        expected_paths=[
            "core_profiles/profiles_1d/grid/rho_tor_norm",
            "equilibrium/time_slice/profiles_1d/rho_tor_norm",
        ],
        category="edge_case",
        notes="path-like query — should match the normalized toroidal flux coordinate",
    ),
    BenchmarkQuery(
        query_text="core_profiles/profiles_1d",
        expected_paths=[
            "core_profiles/profiles_1d",
        ],
        category="edge_case",
        notes="partial path lookup — should find the profiles_1d subtree",
    ),
    BenchmarkQuery(
        query_text="Te AND ne",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "core_profiles/profiles_1d/electrons/density",
            "summary/line_average/t_e",
            "summary/line_average/n_e",
        ],
        category="edge_case",
        notes="boolean-style query — should find electron temperature and density",
    ),
    BenchmarkQuery(
        query_text="ELONGATION",
        expected_paths=[
            "equilibrium/time_slice/boundary/elongation",
            "summary/boundary/elongation",
            "pulse_schedule/position_control/elongation",
        ],
        category="edge_case",
        notes="uppercase query — case-insensitive matching should find elongation",
    ),
    BenchmarkQuery(
        query_text="elctron densty",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
            "edge_profiles/profiles_1d/electrons/density",
        ],
        category="edge_case",
        notes="severe misspelling — fuzzy should recover",
    ),
    BenchmarkQuery(
        query_text="safty factor",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/q",
        ],
        category="edge_case",
        notes="misspelling of 'safety factor'",
    ),
    BenchmarkQuery(
        query_text="equilbrium",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/global_quantities/ip",
        ],
        category="edge_case",
        notes="misspelling of 'equilibrium'",
    ),
    BenchmarkQuery(
        query_text="magntics",
        expected_paths=[
            "magnetics/b_field_pol_probe/field/data",
            "magnetics/flux_loop/flux/data",
        ],
        category="edge_case",
        notes="misspelling of 'magnetics'",
    ),
    BenchmarkQuery(
        query_text="profiles_1d/q",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/q",
        ],
        category="edge_case",
        notes="partial path without IDS prefix",
    ),
    BenchmarkQuery(
        query_text="summary/global_quantities/ip",
        expected_paths=[
            "summary/global_quantities/ip",
        ],
        category="edge_case",
        notes="full path as query — exact match",
    ),
    BenchmarkQuery(
        query_text="TE",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "summary/line_average/t_e",
        ],
        category="edge_case",
        notes="uppercase abbreviation",
    ),
    BenchmarkQuery(
        query_text="ne AND Ti",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
            "core_profiles/profiles_1d/ion/temperature",
        ],
        category="edge_case",
        notes="boolean-style multi-quantity query",
    ),
    BenchmarkQuery(
        query_text="  electron   temperature  ",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="edge_case",
        notes="extra whitespace — should be trimmed",
    ),
    BenchmarkQuery(
        query_text="core_profiles.profiles_1d.electrons.temperature",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="edge_case",
        notes="dot-separated path notation (Python-style)",
    ),
    BenchmarkQuery(
        query_text="what is the electron temperature?",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        category="edge_case",
        notes="natural language question format",
    ),
    BenchmarkQuery(
        query_text="find me the safety factor profile",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/q",
        ],
        category="edge_case",
        notes="conversational query with filler words",
    ),
    BenchmarkQuery(
        query_text="B_0",
        expected_paths=[
            "summary/global_quantities/b0",
            "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
        ],
        category="edge_case",
        notes="underscore in abbreviation",
    ),
    BenchmarkQuery(
        query_text="n_e(rho)",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/density",
        ],
        category="edge_case",
        notes="mathematical notation in query",
    ),
    BenchmarkQuery(
        query_text="T_e [eV]",
        expected_paths=[
            "core_profiles/profiles_1d/electrons/temperature",
            "summary/line_average/t_e",
        ],
        category="edge_case",
        notes="query with unit annotation",
    ),
    BenchmarkQuery(
        query_text="magnetic flux poloidal",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/grid/psi",
            "core_transport/model/profiles_1d/grid_flux/psi",
        ],
        category="disambiguating",
        notes="Poloidal magnetic flux should return psi profiles, not momentum transport flux",
    ),
    BenchmarkQuery(
        query_text="TF coil current",
        expected_paths=[
            "tf/coil/current",
            "tf/coil/current/data",
        ],
        category="structural",
        notes="TF coil current should return the structure and its leaf children",
    ),
    BenchmarkQuery(
        query_text="ion cyclotron heating power",
        expected_paths=[
            "ic_antennas/antenna/power_launched",
            "ic_antennas/antenna",
        ],
        category="exact_concept",
        notes="Ion cyclotron heating should surface ic_antennas IDS paths",
    ),
    BenchmarkQuery(
        query_text="poloidal flux profile",
        expected_paths=[
            "equilibrium/time_slice/profiles_1d/psi",
        ],
        category="exact_concept",
        notes="Poloidal flux profile is a common way to ask for psi radial profile",
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
    + EDGE_CASE_QUERIES
)

# Queries suitable for vector search benchmarks (exclude structural paths
# and edge cases that rely on text matching)
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
    "edge_case",
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
