"""Reference dataset of known-good standard names for benchmarking.

Each entry maps a DD source path to its expected standard name and
the grammar fields used to compose it.  Every name in this set must
pass a round-trip through ``parse_standard_name`` → ``compose_standard_name``.

The dataset covers a representative range of grammar features:
simple physical bases, subject-qualified quantities, component-qualified
vector quantities, positional variants, compound physical bases, and
geometric quantities.
"""

from __future__ import annotations

from imas_standard_names.grammar import (
    Component,
    GeometricBase,
    Object,
    Position,
    StandardName,
    Subject,
    compose_standard_name,
)

# ---------------------------------------------------------------------------
# Helper: build a reference entry from grammar fields
# ---------------------------------------------------------------------------


def _ref(fields: dict) -> dict:
    """Build a reference entry dict with name string and fields.

    Composes the standard name from the fields at import time so any
    grammar error is caught immediately.
    """
    sn = StandardName(**fields)
    name = compose_standard_name(sn)
    # Store string-valued fields for JSON serialization
    str_fields = {}
    for k, v in fields.items():
        if hasattr(v, "value"):
            str_fields[k] = v.value
        else:
            str_fields[k] = v
    return {"name": name, "fields": str_fields}


# ---------------------------------------------------------------------------
# Reference dataset
# ---------------------------------------------------------------------------

REFERENCE_NAMES: dict[str, dict] = {
    # --- Simple physical bases ---
    "equilibrium/time_slice/profiles_1d/safety_factor": _ref(
        {"physical_base": "safety_factor"}
    ),
    "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor": _ref(
        {"physical_base": "magnetic_field", "component": Component.TOROIDAL}
    ),
    "equilibrium/time_slice/profiles_1d/elongation": _ref(
        {"physical_base": "elongation"}
    ),
    "equilibrium/time_slice/profiles_1d/triangularity_upper": _ref(
        {"physical_base": "triangularity"}
    ),
    "equilibrium/time_slice/profiles_1d/magnetic_shear": _ref(
        {"physical_base": "magnetic_shear"}
    ),
    "equilibrium/time_slice/global_quantities/beta_pol": _ref(
        {"physical_base": "beta"}
    ),
    # --- Subject-qualified quantities ---
    "core_profiles/profiles_1d/electrons/temperature": _ref(
        {"physical_base": "temperature", "subject": Subject.ELECTRON}
    ),
    "core_profiles/profiles_1d/ion/temperature": _ref(
        {"physical_base": "temperature", "subject": Subject.ION}
    ),
    "core_profiles/profiles_1d/electrons/density": _ref(
        {"physical_base": "density", "subject": Subject.ELECTRON}
    ),
    "core_profiles/profiles_1d/ion/density": _ref(
        {"physical_base": "density", "subject": Subject.ION}
    ),
    "core_profiles/profiles_1d/electrons/pressure": _ref(
        {"physical_base": "pressure", "subject": Subject.ELECTRON}
    ),
    "core_profiles/profiles_1d/ion/pressure": _ref(
        {"physical_base": "pressure", "subject": Subject.ION}
    ),
    # --- Component-qualified vector quantities ---
    "equilibrium/time_slice/profiles_1d/j_tor": _ref(
        {"physical_base": "current_density", "component": Component.TOROIDAL}
    ),
    "equilibrium/time_slice/profiles_1d/j_parallel": _ref(
        {"physical_base": "current_density", "component": Component.PARALLEL}
    ),
    "magnetics/b_field_pol_probe/field/data": _ref(
        {"physical_base": "magnetic_field", "component": Component.POLOIDAL}
    ),
    "magnetics/b_field_tor_probe/field/data": _ref(
        {"physical_base": "magnetic_field", "component": Component.TOROIDAL}
    ),
    "core_profiles/profiles_1d/rotation_frequency_tor_sonic": _ref(
        {"physical_base": "rotation_frequency", "component": Component.TOROIDAL}
    ),
    # --- Position-qualified quantities ---
    "core_profiles/profiles_1d/electrons/temperature_fit/boundary_condition/value": _ref(
        {
            "physical_base": "temperature",
            "subject": Subject.ELECTRON,
            "position": Position.PLASMA_BOUNDARY,
        }
    ),
    "equilibrium/time_slice/global_quantities/magnetic_axis/r": _ref(
        {
            "geometric_base": GeometricBase.POSITION,
            "object": Object.ROGOWSKI_COIL,
            "position": Position.MAGNETIC_AXIS,
        }
    ),
    # --- Compound physical bases (generic terms qualified via compounding) ---
    "equilibrium/time_slice/global_quantities/ip": _ref(
        {"physical_base": "plasma_current"}
    ),
    "equilibrium/time_slice/global_quantities/psi_axis": _ref(
        {"physical_base": "poloidal_magnetic_flux"}
    ),
    "equilibrium/time_slice/global_quantities/psi_boundary": _ref(
        {
            "physical_base": "poloidal_magnetic_flux",
            "position": Position.PLASMA_BOUNDARY,
        }
    ),
    "equilibrium/time_slice/global_quantities/energy_mhd": _ref(
        {"physical_base": "stored_energy"}
    ),
    "summary/global_quantities/v_loop/value": _ref({"physical_base": "loop_voltage"}),
    "summary/global_quantities/li/value": _ref(
        {"physical_base": "internal_inductance"}
    ),
    "equilibrium/time_slice/global_quantities/resistivity": _ref(
        {"physical_base": "resistivity"}
    ),
    "equilibrium/time_slice/global_quantities/magnetic_axis/b_tor": _ref(
        {"physical_base": "toroidal_magnetic_field"}
    ),
    # --- Geometric bases ---
    "equilibrium/time_slice/global_quantities/minor_radius": _ref(
        {"physical_base": "minor_radius"}
    ),
    "equilibrium/time_slice/global_quantities/major_radius": _ref(
        {"physical_base": "major_radius"}
    ),
    "equilibrium/time_slice/global_quantities/aspect_ratio": _ref(
        {"physical_base": "aspect_ratio"}
    ),
}
"""Map of DD source_path → {name: str, fields: dict}.

Each entry is a known-good standard name that passes grammar round-trip.
"""
