"""Tests for IDS graph operations."""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.ids.graph_ops import (
    FieldMapping,
    Recipe,
    _index_from_path,
    load_mappings,
    load_recipe,
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
                    "driver": "device_xml",
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
                "driver": "device_xml",
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
                "driver": None,
            }
        ]
        result = load_mappings("jet:pf_active", gc)
        m = result[0]
        assert m.source_property == "value"
        assert m.transform_code == "value"
        assert m.driver == "device_xml"
