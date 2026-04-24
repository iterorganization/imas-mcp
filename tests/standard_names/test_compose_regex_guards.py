"""Pure-string regex validators for HARD PRE-EMIT CHECKS (Phase 5d).

These tests validate that the compose-prompt anti-pattern rules can be
detected statically via regex, without requiring LLM calls.  Each test
corresponds to one of the ten HARD PRE-EMIT CHECKS in compose_system.md.
"""

from __future__ import annotations

import re

import pytest

# =====================================================================
# Guard regex functions — one per HARD PRE-EMIT CHECK
# =====================================================================


def has_adjacent_duplicate_tokens(name: str) -> bool:
    """Check #1: adjacent duplicate tokens (e.g. magnetic_magnetic)."""
    tokens = name.split("_")
    return any(a == b for a, b in zip(tokens, tokens[1:], strict=False))


_ENTITY_LOCUS_TOKENS = frozenset(
    {
        "separatrix",
        "magnetic_axis",
        "plasma_boundary",
        "x_point",
        "pedestal",
        "limiter",
        "last_closed_flux_surface",
        "o_point",
        "strike_point",
    }
)


def has_entity_locus_at_violation(name: str) -> bool:
    """Check #2: entity locus with _at_ instead of _of_."""
    for entity in _ENTITY_LOCUS_TOKENS:
        if f"_at_{entity}" in name:
            return True
    return False


_HARDWARE_TOKENS = frozenset(
    {
        "probe",
        "sensor",
        "antenna",
        "channel",
        "injector",
        "aperture",
        "coil",
        "mirror",
        "launcher",
    }
)


def has_hardware_base_or_prefix(name: str) -> bool:
    """Check #3: hardware token as base or prefix (not after _of_)."""
    tokens = name.split("_")
    for i, tok in enumerate(tokens):
        if tok in _HARDWARE_TOKENS:
            # Allowed only if preceded by "of" (i.e. tokens[i-1] == "of")
            if i > 0 and tokens[i - 1] == "of":
                continue
            # Allowed if this is part of a compound like "neutral_beam_injector"
            # after _of_, check that somewhere before there's an "of"
            before = "_".join(tokens[:i])
            if "_of_" in f"_{before}_":
                continue
            return True
    return False


_PROVENANCE_PREFIXES = (
    "initial_",
    "launched_",
    "post_crash_",
    "prefill_",
    "reconstructed_",
)


def has_provenance_prefix(name: str) -> bool:
    """Check #4: provenance prefix."""
    return any(name.startswith(p) for p in _PROVENANCE_PREFIXES)


_ABBREVIATION_RE = re.compile(
    r"(?:^|_)(?:\d+\w*|\w*\d+\w*)(?:_|$)"  # alphanumeric tokens
    r"|(?:^|_)(?:mse|sol|nbi|lh|ic|ec|ntm|exb|norm|perp|par|temp|pos|max|min|sep)(?:_|$)",
    re.IGNORECASE,
)


def has_abbreviation_or_alphanumeric(name: str) -> bool:
    """Check #6: abbreviations, acronyms, or alphanumerics."""
    return bool(_ABBREVIATION_RE.search(name))


_FORBIDDEN_COMPOUND_SUBJECTS = (
    "hydrogen_ion",
    "deuterium_tritium_ion",
)


def has_forbidden_compound_subject(name: str) -> bool:
    """Check #7: forbidden compound subjects."""
    return any(cs in name for cs in _FORBIDDEN_COMPOUND_SUBJECTS)


_UK_SPELLINGS = (
    "analyse",
    "analysed",
    "fibre",
    "ionisation",
    "ionised",
    "normalised",
    "normalise",
    "centre",
    "behaviour",
    "colour",
    "metre",
    "flavour",
    "modelled",
    "labelled",
    "travelled",
    "fuelling",
    "channelling",
    "signalling",
    "polarised",
    "magnetised",
    "organised",
)


def has_uk_spelling(name: str) -> bool:
    """Check #8: British spelling variants."""
    tokens = name.split("_")
    return any(tok in _UK_SPELLINGS for tok in tokens)


def exceeds_length_or_nesting(name: str) -> bool:
    """Check #9: >70 chars or >2 _of_ segments."""
    if len(name) > 70:
        return True
    of_count = name.count("_of_")
    return of_count > 2


_STRUCTURAL_LEAKAGE_TOKENS = (
    "obtained_from",
    "stored_in",
    "derived_from",
    "referenced_by",
    "defined_in",
    "used_for",
)


def has_structural_leakage(name: str) -> bool:
    """Check #10: structural/provenance leakage tokens."""
    return any(tok in name for tok in _STRUCTURAL_LEAKAGE_TOKENS)


# =====================================================================
# REJECT list patterns
# =====================================================================

_REJECT_TOKENS = (
    "equilibrium_reconstruction_",
    "bandwidth_3db",
    "turn_count",
    "nuclear_charge_number",
    "azimuth_angle",
)

_REJECT_PATTERNS = (re.compile(r"distance_between_\w+_and_\w+"),)


def matches_reject_list(name: str) -> bool:
    """Check REJECT list from compose_system.md."""
    for tok in _REJECT_TOKENS:
        if tok in name:
            return True
    return any(pat.search(name) for pat in _REJECT_PATTERNS)


# =====================================================================
# Test classes — one per HARD PRE-EMIT CHECK
# =====================================================================


class TestAdjacentDuplicateTokens:
    """HARD CHECK #1: no adjacent duplicate tokens."""

    @pytest.mark.parametrize(
        "name",
        [
            "magnetic_magnetic_field",
            "beam_beam_power",
            "ion_ion_collision_frequency",
            "electron_electron_scattering",
        ],
    )
    def test_detects_duplicates(self, name: str) -> None:
        assert has_adjacent_duplicate_tokens(name)

    @pytest.mark.parametrize(
        "name",
        [
            "magnetic_field",
            "electron_temperature",
            "ion_cyclotron_resonance_frequency",
        ],
    )
    def test_passes_valid_names(self, name: str) -> None:
        assert not has_adjacent_duplicate_tokens(name)


class TestEntityLocusAt:
    """HARD CHECK #2: entity locus requires _of_, never _at_."""

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature_at_magnetic_axis",
            "poloidal_magnetic_flux_at_separatrix",
            "pressure_at_plasma_boundary",
            "safety_factor_at_x_point",
            "density_at_pedestal",
        ],
    )
    def test_detects_at_violation(self, name: str) -> None:
        assert has_entity_locus_at_violation(name)

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature_of_magnetic_axis",
            "poloidal_magnetic_flux_of_separatrix",
            "pressure_of_plasma_boundary",
        ],
    )
    def test_passes_of_form(self, name: str) -> None:
        assert not has_entity_locus_at_violation(name)


class TestHardwareBaseOrPrefix:
    """HARD CHECK #3: hardware tokens only after _of_."""

    @pytest.mark.parametrize(
        "name",
        [
            "probe_voltage",
            "sensor_electron_density",
            "antenna_power",
            "channel_brightness",
            "coil_current",
        ],
    )
    def test_detects_hardware_prefix(self, name: str) -> None:
        assert has_hardware_base_or_prefix(name)

    @pytest.mark.parametrize(
        "name",
        [
            "rotation_angle_of_electron_cyclotron_launcher_mirror",
            "voltage_of_flux_loop_probe",
            "electron_temperature",
        ],
    )
    def test_passes_hardware_after_of(self, name: str) -> None:
        assert not has_hardware_base_or_prefix(name)


class TestProvenancePrefix:
    """HARD CHECK #4: no provenance prefixes."""

    @pytest.mark.parametrize(
        "name",
        [
            "initial_electron_temperature",
            "launched_power",
            "post_crash_density",
            "prefill_pressure",
            "reconstructed_equilibrium",
        ],
    )
    def test_detects_provenance(self, name: str) -> None:
        assert has_provenance_prefix(name)

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature",
            "plasma_current",
        ],
    )
    def test_passes_clean_names(self, name: str) -> None:
        assert not has_provenance_prefix(name)


class TestAbbreviationsAlphanumerics:
    """HARD CHECK #6: no abbreviations, acronyms, or alphanumerics."""

    @pytest.mark.parametrize(
        "name",
        [
            "bandwidth_3db",
            "power_20_80",
            "mse_angle",
            "sol_density",
            "ec_power",
            "norm_flux",
        ],
    )
    def test_detects_abbreviations(self, name: str) -> None:
        assert has_abbreviation_or_alphanumeric(name)

    @pytest.mark.parametrize(
        "name",
        [
            "electron_cyclotron_power",
            "normalized_flux",
            "perpendicular_velocity",
            "electron_temperature",
        ],
    )
    def test_passes_spelled_out(self, name: str) -> None:
        assert not has_abbreviation_or_alphanumeric(name)


class TestForbiddenCompoundSubject:
    """HARD CHECK #7: exactly one subject; specific compounds forbidden."""

    @pytest.mark.parametrize(
        "name",
        [
            "hydrogen_ion_temperature",
            "deuterium_tritium_ion_density",
        ],
    )
    def test_detects_forbidden_compounds(self, name: str) -> None:
        assert has_forbidden_compound_subject(name)

    @pytest.mark.parametrize(
        "name",
        [
            "hydrogen_temperature",
            "deuterium_tritium_fusion_power",
            "ion_temperature",
        ],
    )
    def test_passes_valid_subjects(self, name: str) -> None:
        assert not has_forbidden_compound_subject(name)


class TestUKSpelling:
    """HARD CHECK #8: US spelling only."""

    @pytest.mark.parametrize(
        "name",
        [
            "normalised_flux",
            "ionisation_rate",
            "fibre_optic_signal",
            "analysed_spectrum",
            "centre_of_mass",
        ],
    )
    def test_detects_uk_spelling(self, name: str) -> None:
        assert has_uk_spelling(name)

    @pytest.mark.parametrize(
        "name",
        [
            "normalized_flux",
            "ionization_rate",
            "fiber_optic_signal",
            "analyzed_spectrum",
            "center_of_mass",
        ],
    )
    def test_passes_us_spelling(self, name: str) -> None:
        assert not has_uk_spelling(name)


class TestLengthAndNesting:
    """HARD CHECK #9: max 70 chars, max 2 _of_ segments."""

    def test_too_long(self) -> None:
        name = "a" * 71
        assert exceeds_length_or_nesting(name)

    def test_exactly_70_ok(self) -> None:
        name = "a" * 70
        assert not exceeds_length_or_nesting(name)

    def test_three_of_segments(self) -> None:
        name = "gradient_of_pressure_of_plasma_boundary_of_separatrix"
        assert exceeds_length_or_nesting(name)

    def test_two_of_segments_ok(self) -> None:
        name = "gradient_of_pressure_of_plasma_boundary"
        assert not exceeds_length_or_nesting(name)


class TestStructuralLeakage:
    """HARD CHECK #10: no structural leakage tokens."""

    @pytest.mark.parametrize(
        "name",
        [
            "temperature_obtained_from_fit",
            "density_stored_in_database",
            "pressure_derived_from_equilibrium",
            "current_referenced_by_shot",
            "field_defined_in_grid",
        ],
    )
    def test_detects_leakage(self, name: str) -> None:
        assert has_structural_leakage(name)

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature",
            "plasma_current",
            "poloidal_magnetic_flux",
        ],
    )
    def test_passes_clean_names(self, name: str) -> None:
        assert not has_structural_leakage(name)


class TestRejectList:
    """REJECT list patterns from compose_system.md (expanded)."""

    @pytest.mark.parametrize(
        "name",
        [
            "equilibrium_reconstruction_pressure",
            "bandwidth_3db",
            "turn_count",
            "nuclear_charge_number",
            "azimuth_angle",
            "distance_between_inner_and_outer_separatrices",
            "distance_between_magnetic_axis_and_geometric_axis",
        ],
    )
    def test_detects_rejected_patterns(self, name: str) -> None:
        assert matches_reject_list(name)

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature",
            "poloidal_magnetic_flux",
            "toroidal_angle",
            "atomic_number",
        ],
    )
    def test_passes_allowed_names(self, name: str) -> None:
        assert not matches_reject_list(name)
