"""Tests for DD unit override engine.

See ``imas_codex/standard_names/unit_overrides.py`` and
``imas_codex/standard_names/config/unit_overrides.yaml``.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.unit_overrides import (
    _glob_match,
    resolve_unit,
)


class TestGlobMatch:
    """Path-pattern glob matcher."""

    @pytest.mark.parametrize(
        "pattern,path,expected",
        [
            # "**" matches zero or more segments
            ("**", "a", True),
            ("**", "a/b/c", True),
            # "**/foo" matches any prefix
            ("**/element/multiplicity", "element/multiplicity", True),
            ("**/element/multiplicity", "a/b/element/multiplicity", True),
            ("**/element/multiplicity", "element/other", False),
            # "*" matches single segment only
            ("pulse_schedule/*/reference", "pulse_schedule/ec/reference", True),
            (
                "pulse_schedule/*/reference",
                "pulse_schedule/ec/beam/reference",
                False,
            ),
            # "**" between literals absorbs multiple segments
            (
                "pulse_schedule/**/reference",
                "pulse_schedule/ec/beam/power_launched/reference",
                True,
            ),
            ("pulse_schedule/**/reference", "pulse_schedule/reference", True),
            ("pulse_schedule/**/reference", "pulse_schedule/ec/reference", True),
            # Trailing-segment anchoring: trailing slash variants don't leak
            (
                "pulse_schedule/**/reference",
                "pulse_schedule/ec/reference/data",
                False,
            ),
            (
                "pulse_schedule/**/reference/data",
                "pulse_schedule/ec/beam/reference/data",
                True,
            ),
            # Literal segments
            ("equilibrium/time_slice/psi", "equilibrium/time_slice/psi", True),
            ("equilibrium/time_slice/psi", "equilibrium/time_slice/phi", False),
        ],
    )
    def test_glob_match(self, pattern: str, path: str, expected: bool) -> None:
        assert _glob_match(pattern, path) is expected


class TestResolveUnitOverrides:
    """Class 1–3: unit replacements (candidate flows through normally)."""

    def test_elementary_charge_unit_on_multiplicity(self) -> None:
        unit, meta = resolve_unit(
            "core_profiles/profiles_1d/ion/element/multiplicity",
            "Elementary Charge Unit",
        )
        assert unit == "1"
        assert meta is not None
        assert meta["rule"] == "override"
        assert meta["original_unit"] == "Elementary Charge Unit"

    def test_elementary_charge_unit_on_ionisation_potential(self) -> None:
        unit, meta = resolve_unit(
            "core_profiles/profiles_1d/ion/state/ionisation_potential",
            "Elementary Charge Unit",
        )
        assert unit == "eV"
        assert meta["rule"] == "override"

    def test_elementary_charge_unit_on_binding_energy(self) -> None:
        unit, meta = resolve_unit(
            "atomic_data/process/binding_energy", "Elementary Charge Unit"
        )
        assert unit == "eV"

    def test_atomic_mass_unit_on_element_a(self) -> None:
        unit, meta = resolve_unit("gas_injection/species/element/a", "Atomic Mass Unit")
        assert unit == "u"
        assert meta["rule"] == "override"

    def test_atomic_mass_unit_on_atomic_mass(self) -> None:
        unit, meta = resolve_unit(
            "spectrometer_mass/channel/atomic_mass", "Atomic Mass Unit"
        )
        assert unit == "u"

    @pytest.mark.parametrize(
        "leaf",
        [
            "z_n",
            "charge_number",
            "z_ion",
            "z_average",
            "z_square_average",
            "z_min",
            "z_max",
            "vibrational_level",
        ],
    )
    def test_e_on_dimensionless_charge_ratios(self, leaf: str) -> None:
        unit, meta = resolve_unit(f"core_profiles/profiles_1d/ion/{leaf}", "e")
        assert unit == "1", f"{leaf} with unit='e' should override to '1'"
        assert meta["rule"] == "override"

    def test_e_on_ionisation_potential_becomes_ev(self) -> None:
        unit, meta = resolve_unit(
            "edge_profiles/ggd/ion/state/ionisation_potential", "e"
        )
        assert unit == "eV"


class TestResolveUnitSkips:
    """Class 4–5: skip records (removed from composition pipeline)."""

    def test_unresolved_jinja_m_dimension(self) -> None:
        unit, meta = resolve_unit(
            "equilibrium/time_slice/ggd/grid/space/objects_per_dimension/"
            "object/measure",
            "m^dimension",
        )
        assert unit is None
        assert meta is not None
        assert meta["rule"] == "skip"
        assert meta["skip_reason"] == "dd_unit_unresolvable"
        assert "m^dimension" in meta["skip_reason_detail"]

    def test_pulse_schedule_reference_sentinel(self) -> None:
        unit, meta = resolve_unit(
            "pulse_schedule/pf_active/coil/resistance_additional/reference", "1"
        )
        assert unit is None
        assert meta["rule"] == "skip"
        assert meta["skip_reason"] == "dd_unit_context_dependent"

    def test_pulse_schedule_reference_data_sentinel(self) -> None:
        unit, meta = resolve_unit(
            "pulse_schedule/ec/beam/power_launched/reference/data", "1"
        )
        assert unit is None
        assert meta["rule"] == "skip"
        assert meta["skip_reason"] == "dd_unit_context_dependent"


class TestResolveUnitPassThrough:
    """Paths with no matching rule should pass through unchanged."""

    def test_valid_unit_unchanged(self) -> None:
        unit, meta = resolve_unit(
            "core_profiles/profiles_1d/electrons/temperature", "eV"
        )
        assert unit == "eV"
        assert meta is None

    def test_none_unit_returns_none(self) -> None:
        unit, meta = resolve_unit("some/path", None)
        assert unit is None
        assert meta is None

    def test_non_matching_dimensionless(self) -> None:
        """unit='1' outside the pulse_schedule/*/reference pattern is kept."""
        unit, meta = resolve_unit("core_profiles/profiles_1d/grid/rho_tor_norm", "1")
        assert unit == "1"
        assert meta is None

    def test_non_matching_e_on_non_listed_path(self) -> None:
        """A path with unit='e' that doesn't match any z_* pattern passes
        through. The override config is intentionally path-scoped: not
        every 'e' unit in DD is a defect."""
        unit, meta = resolve_unit("some/exotic/path/that/is/not/in/config", "e")
        assert unit == "e"
        assert meta is None
