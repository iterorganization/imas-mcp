"""Tests for IDS graph operations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.ids.graph_ops import (
    MAGNETICS_BPOL_MAPPINGS,
    MAGNETICS_FLUX_LOOP_MAPPINGS,
    PF_ACTIVE_ASSEMBLY_CONFIG,
    PF_ACTIVE_CIRCUIT_MAPPINGS,
    PF_ACTIVE_COIL_MAPPINGS,
    PF_PASSIVE_LOOP_MAPPINGS,
    WALL_ASSEMBLY_CONFIG,
    WALL_LIMITER_MAPPINGS,
    FieldMapping,
    Recipe,
    _index_from_path,
    create_mappings,
    create_recipe,
    load_mappings,
    load_recipe,
    seed_ids_mappings,
)


class TestIndexFromPath:
    def test_device_xml_path(self):
        assert _index_from_path("jet:device_xml:p68613:pfcoils:5") == 5

    def test_jec2020_path(self):
        assert _index_from_path("jet:jec2020:pf_coil:12") == 12


class TestLoadRecipe:
    def test_returns_none_when_not_found(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = load_recipe("jet", "nonexistent", gc)
        assert result is None

    def test_returns_recipe_with_mappings(self):
        gc = MagicMock()
        # First query: recipe lookup
        gc.query.side_effect = [
            [
                {
                    "id": "jet:pf_active",
                    "facility_id": "jet",
                    "ids_name": "pf_active",
                    "dd_version": "4.1.1",
                    "provider": "imas-codex",
                    "assembly_config": '{"coil": {}}',
                }
            ],
            # Second query: mappings
            [
                {
                    "id": "jet:PF:r",
                    "source_property": "r",
                    "target_imas_path": "pf_active/coil/element/geometry/rectangle/r",
                    "transform_code": "value",
                    "units_in": "m",
                    "units_out": "m",
                    "cocos_source": None,
                    "cocos_target": None,
                    "driver": "device_xml",
                    "cocos_label": None,
                }
            ],
        ]
        result = load_recipe("jet", "pf_active", gc)
        assert result is not None
        assert isinstance(result, Recipe)
        assert result.ids_name == "pf_active"
        assert result.dd_version == "4.1.1"
        assert len(result.mappings) == 1
        assert result.mappings[0].source_property == "r"


class TestLoadMappings:
    def test_empty_result(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = load_mappings("jet:pf_active", gc)
        assert result == []

    def test_parses_mapping_fields(self):
        gc = MagicMock()
        gc.query.return_value = [
            {
                "id": "jet:PF:r",
                "source_property": "r",
                "target_imas_path": "pf_active/coil/element/geometry/rectangle/r",
                "transform_code": "value * 1.0",
                "units_in": "m",
                "units_out": "m",
                "cocos_source": None,
                "cocos_target": None,
                "driver": "device_xml",
                "cocos_label": None,
            }
        ]
        result = load_mappings("jet:pf_active", gc)
        assert len(result) == 1
        m = result[0]
        assert isinstance(m, FieldMapping)
        assert m.source_property == "r"
        assert m.transform_code == "value * 1.0"
        assert m.units_in == "m"

    def test_defaults_for_missing_fields(self):
        gc = MagicMock()
        gc.query.return_value = [
            {
                "id": "m1",
                "source_property": None,
                "target_imas_path": "pf_active/coil/name",
                "transform_code": None,
                "units_in": None,
                "units_out": None,
                "cocos_source": None,
                "cocos_target": None,
                "driver": None,
                "cocos_label": None,
            }
        ]
        result = load_mappings("jet:pf_active", gc)
        m = result[0]
        assert m.source_property == "value"
        assert m.transform_code == "value"
        assert m.driver == "device_xml"
        assert m.cocos_source is None
        assert m.cocos_target is None

    def test_cocos_warning_for_sensitive_path(self, caplog):
        """Warn when IMAS path is COCOS-sensitive but mapping has no cocos_source."""
        import logging

        gc = MagicMock()
        gc.query.return_value = [
            {
                "id": "jet:CI:current",
                "source_property": "current",
                "target_imas_path": "pf_active/circuit/current",
                "transform_code": "value",
                "units_in": "A",
                "units_out": "A",
                "cocos_source": None,
                "cocos_target": None,
                "driver": "device_xml",
                "cocos_label": "ip_like",
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = load_mappings("jet:pf_active", gc)
        assert len(result) == 1
        assert "COCOS-sensitive" in caplog.text
        assert "ip_like" in caplog.text

    def test_no_cocos_warning_for_non_sensitive_path(self, caplog):
        """No warning for paths without cocos_label_transformation."""
        import logging

        gc = MagicMock()
        gc.query.return_value = [
            {
                "id": "jet:PF:r",
                "source_property": "r",
                "target_imas_path": "pf_active/coil/element/geometry/rectangle/r",
                "transform_code": "value",
                "units_in": "m",
                "units_out": "m",
                "cocos_source": None,
                "cocos_target": None,
                "driver": "device_xml",
                "cocos_label": None,
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = load_mappings("jet:pf_active", gc)
        assert len(result) == 1
        assert "COCOS-sensitive" not in caplog.text


class TestMappingSpecs:
    """Validate the canonical mapping definitions."""

    def test_pf_active_coil_mappings_complete(self):
        # 6 mappings: r, z, dr, dz, turnsperelement, description
        assert len(PF_ACTIVE_COIL_MAPPINGS) == 6
        source_props = {m[0] for m in PF_ACTIVE_COIL_MAPPINGS}
        assert {"r", "z", "dr", "dz", "turnsperelement", "description"} == source_props

    def test_pf_active_circuit_mappings(self):
        assert len(PF_ACTIVE_CIRCUIT_MAPPINGS) == 1
        assert PF_ACTIVE_CIRCUIT_MAPPINGS[0][0] == "description"

    def test_magnetics_bpol_mappings(self):
        assert len(MAGNETICS_BPOL_MAPPINGS) == 4
        source_props = {m[0] for m in MAGNETICS_BPOL_MAPPINGS}
        assert {"r", "z", "angle", "description"} == source_props
        # angle should have deg→rad conversion
        angle_mapping = next(m for m in MAGNETICS_BPOL_MAPPINGS if m[0] == "angle")
        assert angle_mapping[2] == "math.radians(value)"
        assert angle_mapping[3] == "deg"
        assert angle_mapping[4] == "rad"

    def test_magnetics_flux_loop_mappings(self):
        assert len(MAGNETICS_FLUX_LOOP_MAPPINGS) == 4
        source_props = {m[0] for m in MAGNETICS_FLUX_LOOP_MAPPINGS}
        assert {"r", "z", "dphi", "description"} == source_props

    def test_pf_passive_loop_mappings(self):
        assert len(PF_PASSIVE_LOOP_MAPPINGS) == 6
        source_props = {m[0] for m in PF_PASSIVE_LOOP_MAPPINGS}
        assert {"r", "z", "dr", "dz", "resistance", "description"} == source_props

    def test_wall_limiter_mappings(self):
        assert len(WALL_LIMITER_MAPPINGS) == 3
        source_props = {m[0] for m in WALL_LIMITER_MAPPINGS}
        assert {"r_contour", "z_contour", "description"} == source_props
        # Contour fields should map to outline path
        r_mapping = next(m for m in WALL_LIMITER_MAPPINGS if m[0] == "r_contour")
        assert "outline/r" in r_mapping[1]

    def test_all_target_paths_use_slash_separators(self):
        """All target paths should use / (IMAS convention)."""
        all_specs = (
            PF_ACTIVE_COIL_MAPPINGS
            + PF_ACTIVE_CIRCUIT_MAPPINGS
            + MAGNETICS_BPOL_MAPPINGS
            + MAGNETICS_FLUX_LOOP_MAPPINGS
            + PF_PASSIVE_LOOP_MAPPINGS
            + WALL_LIMITER_MAPPINGS
        )
        for spec in all_specs:
            target_path = spec[1]
            assert "/" in target_path, f"{target_path} missing / separator"
            assert "." not in target_path, f"{target_path} uses . instead of /"


class TestAssemblyConfigs:
    """Validate the assembly config templates."""

    def test_pf_active_has_coil_and_circuit(self):
        assert "coil" in PF_ACTIVE_ASSEMBLY_CONFIG
        assert "circuit" in PF_ACTIVE_ASSEMBLY_CONFIG
        assert "static" in PF_ACTIVE_ASSEMBLY_CONFIG
        assert PF_ACTIVE_ASSEMBLY_CONFIG["coil"]["source"]["system"] == "PF"
        assert PF_ACTIVE_ASSEMBLY_CONFIG["circuit"]["source"]["system"] == "CI"

    def test_wall_has_nested_array_structure(self):
        assert "description_2d" in WALL_ASSEMBLY_CONFIG
        d2d = WALL_ASSEMBLY_CONFIG["description_2d"]
        assert d2d["structure"] == "nested_array"
        assert d2d["nested_path"] == "limiter.unit"
        assert d2d["parent_size"] == 1
        assert d2d["source"]["select_via"] == "USES_LIMITER"


class TestCreateMappings:
    def test_creates_mappings_with_correct_ids(self):
        gc = MagicMock()
        gc.query.return_value = []
        specs: list[tuple[str, str, str, str | None, str | None]] = [
            ("r", "test/section/r", "value", "m", "m"),
            ("z", "test/section/z", "value", "m", "m"),
        ]
        result = create_mappings("jet", "test", "section", "PF", specs, gc)
        assert len(result) == 2
        assert result[0] == "jet:PF:r→test/section/r"
        assert result[1] == "jet:PF:z→test/section/z"
        gc.query.assert_called_once()


class TestCreateRecipe:
    def test_creates_recipe_with_mappings(self):
        gc = MagicMock()
        gc.query.return_value = []
        config = {"coil": {"source": {"system": "PF"}}}
        result = create_recipe("jet", "pf_active", "4.1.1", config, ["m1", "m2"], gc)
        assert result == "jet:pf_active"
        assert gc.query.call_count == 2  # create recipe + link mappings


class TestSeedIdsMappings:
    def test_unsupported_ids_raises(self):
        gc = MagicMock()
        with pytest.raises(ValueError, match="No mapping definitions for IDS"):
            seed_ids_mappings("jet", "unsupported_ids", "4.1.1", gc)

    def test_pf_active_creates_all_sections(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = seed_ids_mappings("jet", "pf_active", "4.1.1", gc)
        assert result == "jet:pf_active"
        # Should have called query for: coil mappings + circuit mappings +
        # recipe creation + mappings link
        assert gc.query.call_count == 4

    def test_magnetics_creates_all_sections(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = seed_ids_mappings("jet", "magnetics", "4.1.1", gc)
        assert result == "jet:magnetics"

    def test_pf_passive_creates_all_sections(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = seed_ids_mappings("jet", "pf_passive", "4.1.1", gc)
        assert result == "jet:pf_passive"

    def test_wall_creates_all_sections(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = seed_ids_mappings("jet", "wall", "4.1.1", gc)
        assert result == "jet:wall"
        # description_2d mappings + recipe creation + mappings link
        assert gc.query.call_count == 3
