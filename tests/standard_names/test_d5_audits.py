"""Tests for D5/P0 blocking audits: empty-doc, unit-sanity, kind-derivation.

Covers:
- P0.1: empty description/documentation → quarantine.
- P0.2: dimensional-sanity unit audit (check_unit_sanity).
- P0.3: auto-derive kind from name (derive_kind).
- Integration: wired into enrich_validate_worker.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Helpers
# =============================================================================


def _make_item(name: str, **overrides: Any) -> dict[str, Any]:
    """Build a mock SN item as returned by document worker."""
    base = {
        "id": name,
        "description": f"Description of {name}",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "tags": ["time-dependent"],
        "links": None,
        "source_paths": [f"equilibrium/time_slice/profiles_1d/{name}"],
        "physical_base": "temperature",
        "subject": "electron",
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "physics_domain": "equilibrium",
        "model": "test-model",
        "enriched_description": f"The {name.replace('_', ' ')} in plasma.",
        "enriched_documentation": f"Detailed docs for {name}.",
        "enriched_links": [],
        "enriched_tags": ["spatial-profile"],
    }
    base.update(overrides)
    return base


def _make_batch(items: list[dict], batch_index: int = 0) -> dict[str, Any]:
    return {
        "items": items,
        "claim_token": "test-token",
        "batch_index": batch_index,
    }


def _make_state(batches: list[dict]) -> MagicMock:
    state = MagicMock()
    state.batches = batches
    state.stop_requested = False
    state.validate_stats = MagicMock()
    state.validate_stats.total = 0
    state.validate_stats.processed = 0
    state.validate_stats.errors = 0
    state.validate_phase = MagicMock()
    state.stats = {}
    return state


# =============================================================================
# P0.1: Empty-doc validator
# =============================================================================


class TestEmptyDocValidator:
    """Empty description or documentation → quarantine."""

    def test_empty_description_quarantines(self):
        from imas_codex.standard_names.enrich_workers import _check_empty_documentation

        item = _make_item("electron_temperature", enriched_description="")
        issues = _check_empty_documentation(item)
        assert len(issues) == 1
        assert "empty_documentation" in issues[0]
        assert "description" in issues[0]

    def test_empty_documentation_quarantines(self):
        from imas_codex.standard_names.enrich_workers import _check_empty_documentation

        item = _make_item("electron_temperature", enriched_documentation="")
        issues = _check_empty_documentation(item)
        assert len(issues) == 1
        assert "empty_documentation" in issues[0]
        assert "documentation" in issues[0]

    def test_both_empty_quarantines(self):
        from imas_codex.standard_names.enrich_workers import _check_empty_documentation

        item = _make_item(
            "electron_temperature",
            enriched_description="",
            enriched_documentation="",
        )
        issues = _check_empty_documentation(item)
        assert len(issues) == 2

    def test_whitespace_only_quarantines(self):
        from imas_codex.standard_names.enrich_workers import _check_empty_documentation

        item = _make_item(
            "electron_temperature",
            enriched_description="   \n\t  ",
            enriched_documentation="  ",
        )
        issues = _check_empty_documentation(item)
        assert len(issues) == 2

    def test_none_documentation_quarantines(self):
        from imas_codex.standard_names.enrich_workers import _check_empty_documentation

        item = _make_item(
            "electron_temperature",
            enriched_documentation=None,
        )
        issues = _check_empty_documentation(item)
        assert len(issues) == 1

    def test_valid_docs_pass(self):
        from imas_codex.standard_names.enrich_workers import _check_empty_documentation

        item = _make_item(
            "electron_temperature",
            enriched_description="The electron temperature.",
            enriched_documentation="Detailed documentation here.",
        )
        issues = _check_empty_documentation(item)
        assert issues == []

    @pytest.mark.asyncio
    async def test_empty_doc_quarantines_in_worker(self):
        """Full worker test: empty docs → quarantined."""
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        item = _make_item(
            "safety_factor_of_magnetic_axis",
            enriched_description="",
            enriched_documentation="",
        )
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers._check_links_batch",
            return_value={"safety_factor_of_magnetic_axis": []},
        ):
            await enrich_validate_worker(state)

        assert item["validation_status"] == "quarantined"
        assert any("empty_documentation" in i for i in item["validation_issues"])


# =============================================================================
# P0.2: Dimensional sanity — unit_audit.check_unit_sanity
# =============================================================================


class TestUnitSanity:
    """Dimensional-sanity checks on name vs unit."""

    def test_phase_rad_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert check_unit_sanity("electric_field_phase", "rad") == []

    def test_phase_one_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert check_unit_sanity("electric_field_phase", "1") == []

    def test_phase_wrong_unit_fails(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        issues = check_unit_sanity(
            "right_hand_circularly_polarized_electric_field_phase", "m^-1.V"
        )
        assert "unit_mismatch:phase_must_be_rad" in issues

    def test_wave_b_field_tesla_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert check_unit_sanity("binormal_component_of_wave_magnetic_field", "T") == []

    def test_wave_b_field_wrong_unit_fails(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        issues = check_unit_sanity(
            "binormal_component_of_wave_magnetic_field", "m^-1.V"
        )
        assert "unit_mismatch:wave_b_field_must_be_tesla" in issues

    def test_wave_vector_per_metre_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert check_unit_sanity("perpendicular_component_of_wave_vector", "m^-1") == []

    def test_wave_vector_wrong_unit_fails(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        issues = check_unit_sanity("perpendicular_component_of_wave_vector", "V.m")
        assert "unit_mismatch:wave_vector_must_be_per_metre" in issues

    def test_spectrum_per_hz_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert (
            check_unit_sanity(
                "lower_hybrid_antenna_wave_power_density_spectrum", "W.Hz^-1"
            )
            == []
        )

    def test_spectrum_dimensionless_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        # Integer-mode style: unit is "1"
        assert check_unit_sanity("power_density_spectrum", "1") == []

    def test_spectrum_bare_W_fails(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        issues = check_unit_sanity(
            "lower_hybrid_antenna_wave_power_density_spectrum", "W"
        )
        assert "unit_mismatch:spectrum_missing_spectral_denominator" in issues

    def test_per_b_with_inverse_tesla_passes(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert (
            check_unit_sanity(
                "radial_component_of_ion_velocity_per_magnetic_field_strength",
                "m.s^-1.T^-1",
            )
            == []
        )

    def test_per_b_without_inverse_tesla_fails(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        issues = check_unit_sanity(
            "radial_component_of_ion_velocity_per_magnetic_field_strength",
            "m.s^-1",
        )
        assert "unit_mismatch:per_b_must_include_inverse_tesla" in issues

    def test_over_b_without_inverse_tesla_fails(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        issues = check_unit_sanity(
            "radial_component_of_ion_velocity_over_magnetic_field_strength",
            "m.s^-1",
        )
        assert "unit_mismatch:per_b_must_include_inverse_tesla" in issues

    def test_normal_name_no_issues(self):
        from imas_codex.standard_names.unit_audit import check_unit_sanity

        assert check_unit_sanity("electron_temperature", "eV") == []

    @pytest.mark.asyncio
    async def test_unit_mismatch_quarantines_in_worker(self):
        """Full worker test: unit mismatch → quarantined."""
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        item = _make_item(
            "right_hand_circularly_polarized_electric_field_phase",
            unit="m^-1.V",
            enriched_description="Phase of the RH polarized E-field.",
            enriched_documentation="Detailed documentation.",
        )
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers._check_links_batch",
            return_value={"right_hand_circularly_polarized_electric_field_phase": []},
        ):
            await enrich_validate_worker(state)

        assert item["validation_status"] == "quarantined"
        assert any("unit_mismatch" in i for i in item["validation_issues"])


# =============================================================================
# P0.3: Auto-derive kind from name
# =============================================================================


class TestKindDerivation:
    """derive_kind deterministically assigns kind from name tokens."""

    def test_component_of_returns_vector(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert derive_kind("binormal_component_of_wave_electric_field") == "vector"

    def test_tensor_returns_tensor(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert derive_kind("contravariant_metric_tensor") == "tensor"
        assert derive_kind("reynolds_stress_tensor_real_part") == "tensor"

    def test_eigenfunction_returns_eigenfunction(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert (
            derive_kind("perturbed_pressure_eigenfunction_real_part") == "eigenfunction"
        )

    def test_spectrum_returns_spectrum(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert (
            derive_kind("ion_cyclotron_heating_antenna_surface_current_spectrum")
            == "spectrum"
        )

    def test_real_part_returns_complex(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert derive_kind("perturbed_velocity_real_part") == "complex"

    def test_imaginary_part_returns_complex(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert derive_kind("perturbed_velocity_imaginary_part") == "complex"

    def test_plain_scalar_returns_scalar(self):
        from imas_codex.standard_names.kind_derivation import derive_kind

        assert derive_kind("electron_temperature") == "scalar"
        assert derive_kind("plasma_current") == "scalar"
        assert derive_kind("safety_factor") == "scalar"

    def test_priority_eigenfunction_over_complex(self):
        """Eigenfunction check has priority over complex."""
        from imas_codex.standard_names.kind_derivation import derive_kind

        # _eigenfunction appears before _real_part check
        assert (
            derive_kind("perturbed_pressure_eigenfunction_real_part") == "eigenfunction"
        )

    @pytest.mark.asyncio
    async def test_kind_overridden_in_worker(self):
        """Full worker test: kind is overridden from scalar → vector."""
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        item = _make_item(
            "binormal_component_of_wave_electric_field",
            kind="scalar",
            unit="V.m^-1",
            enriched_description="Binormal component of wave E-field.",
            enriched_documentation="Detailed documentation of component.",
        )
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers._check_links_batch",
            return_value={"binormal_component_of_wave_electric_field": []},
        ):
            await enrich_validate_worker(state)

        assert item["kind"] == "vector"

    @pytest.mark.asyncio
    async def test_kind_spectrum_set_in_worker(self):
        """Full worker test: spectrum name gets kind=spectrum."""
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        item = _make_item(
            "power_density_spectrum",
            kind="scalar",
            unit="W.Hz^-1",
            enriched_description="Power spectral density.",
            enriched_documentation="Detailed documentation of spectrum.",
        )
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers._check_links_batch",
            return_value={"power_density_spectrum": []},
        ):
            await enrich_validate_worker(state)

        assert item["kind"] == "spectrum"


# =============================================================================
# Integration: all P0 checks together
# =============================================================================


class TestP0Integration:
    """Integration tests ensuring all P0 checks work together."""

    @pytest.mark.asyncio
    async def test_clean_item_still_passes(self):
        """A normal item with good docs passes all new checks."""
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        item = _make_item(
            "electron_temperature",
            enriched_description="Temperature of the electrons in plasma.",
            enriched_documentation="Thermal energy per electron.",
        )
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers._check_links_batch",
            return_value={"electron_temperature": []},
        ):
            await enrich_validate_worker(state)

        assert item["validation_status"] == "valid"

    @pytest.mark.asyncio
    async def test_multiple_failures_accumulated(self):
        """An item with both empty docs and bad unit accumulates issues."""
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        item = _make_item(
            "right_hand_circularly_polarized_electric_field_phase",
            unit="m^-1.V",
            enriched_description="",
            enriched_documentation="",
        )
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers._check_links_batch",
            return_value={"right_hand_circularly_polarized_electric_field_phase": []},
        ):
            await enrich_validate_worker(state)

        assert item["validation_status"] == "quarantined"
        issue_text = " ".join(item["validation_issues"])
        assert "empty_documentation" in issue_text
        assert "unit_mismatch" in issue_text
