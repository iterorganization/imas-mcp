"""Tests for standard name ID normalization."""

import pytest

from imas_codex.standard_names.graph_ops import normalize_name_id


class TestNormalizeNameId:
    def test_exb_expansion(self):
        assert (
            normalize_name_id(
                "poloidal_component_of_neutral_particle_ExB_drift_velocity"
            )
            == "poloidal_component_of_neutral_particle_e_cross_b_drift_velocity"
        )

    def test_bxgradb_expansion(self):
        assert (
            normalize_name_id("BxGradB_drift_velocity")
            == "b_cross_grad_b_drift_velocity"
        )

    def test_general_lowercase(self):
        assert normalize_name_id("Electron_Temperature") == "electron_temperature"

    def test_already_lowercase(self):
        assert normalize_name_id("electron_temperature") == "electron_temperature"

    def test_mixed_case_with_abbreviation(self):
        assert normalize_name_id("ExB_Drift_Velocity") == "e_cross_b_drift_velocity"

    def test_empty_string(self):
        assert normalize_name_id("") == ""
