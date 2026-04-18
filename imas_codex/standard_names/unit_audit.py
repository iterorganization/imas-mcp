"""Dimensional-sanity audit for standard name units (D5/P0.2).

Cross-checks the unit field against structural tokens in the standard name.
Returns tagged issue strings when a name's unit is physically implausible
given its semantic role (e.g. a ``_phase`` name must be ``rad`` or ``1``).

All rules are deterministic — no LLM inference.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Unit-equivalence helpers
# ---------------------------------------------------------------------------

# Tesla-equivalent unit strings (SI base or common derived forms).
_TESLA_EQUIVALENTS = frozenset(
    {
        "T",
        "kg.A^-1.s^-2",
        "kg.s^-2.A^-1",
        "kg/(A.s^2)",
        "V.s.m^-2",
        "Wb.m^-2",
    }
)

# Acceptable per-frequency / per-mode spectral denominators.
_SPECTRAL_FACTORS = re.compile(
    r"Hz\^-1|s(?:\^1)?(?![a-zA-Z])|rad\^-1|mode|"
    r"per.mode|per.Hz|per.frequency|per.toroidal|per.poloidal"
)


def _unit_contains_factor(unit: str, factor: str) -> bool:
    """Check if *unit* contains *factor* as a sub-expression."""
    return factor in unit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_unit_sanity(name: str, unit: str) -> list[str]:
    """Return issue tags for dimensional mismatches between *name* and *unit*.

    Each returned string has the format ``"unit_mismatch:<rule_id>"``.

    Rules implemented (from D5 §7.2):

    1. ``_phase`` suffix → unit ∈ {``rad``, ``1``, ``dimensionless``}.
    2. ``wave_magnetic_field`` → unit must be T-equivalent.
    3. ``wave_vector`` → unit must contain ``m^-1``.
    4. ``_spectrum`` suffix → unit must contain a per-frequency or per-mode
       factor (``Hz^-1``, ``s``, ``rad^-1``, or mode-related token).
    5. ``_per_magnetic_field_strength`` or ``_over_magnetic_field_strength``
       → unit must include ``T^-1``.
    """
    issues: list[str] = []
    unit_stripped = (unit or "").strip()
    name_lower = name.lower()

    # Rule 1: phase → rad or 1
    if name_lower.endswith("_phase") or "_phase_" in name_lower:
        acceptable_phase = {"rad", "1", "dimensionless", ""}
        if unit_stripped not in acceptable_phase:
            issues.append("unit_mismatch:phase_must_be_rad")

    # Rule 2: wave magnetic field → T-equivalent
    if "wave_magnetic_field" in name_lower:
        if unit_stripped not in _TESLA_EQUIVALENTS:
            issues.append("unit_mismatch:wave_b_field_must_be_tesla")

    # Rule 3: wave vector → m^-1
    if "wave_vector" in name_lower:
        if not _unit_contains_factor(unit_stripped, "m^-1"):
            issues.append("unit_mismatch:wave_vector_must_be_per_metre")

    # Rule 4: spectrum → per-frequency or per-mode factor
    if name_lower.endswith("_spectrum"):
        # Allow genuinely dimensionless or empty (integer-mode style)
        if unit_stripped not in {"1", "dimensionless", ""}:
            if not _SPECTRAL_FACTORS.search(unit_stripped):
                issues.append("unit_mismatch:spectrum_missing_spectral_denominator")

    # Rule 5: per/over magnetic field strength → must include T^-1
    if (
        "_per_magnetic_field_strength" in name_lower
        or "_over_magnetic_field_strength" in name_lower
    ):
        if not _unit_contains_factor(unit_stripped, "T^-1"):
            issues.append("unit_mismatch:per_b_must_include_inverse_tesla")

    return issues
