"""Tests for IDS graph operations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.ids.graph_ops import (
    Mapping,
    SignalMapping,
    _index_from_path,
    create_imas_mapping,
    create_signal_source,
    load_mapping,
    load_sections,
    load_signal_mappings,
)


class TestIndexFromPath:
    def test_device_xml_path(self):
        assert _index_from_path("jet:device_xml:p68613:pfcoils:5") == 5

    def test_jec2020_path(self):
        assert _index_from_path("jet:jec2020:pf_coil:12") == 12


class TestLoadMapping:
    def test_returns_none_when_not_found(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = load_mapping("jet", "nonexistent", gc)
        assert result is None

    def test_returns_mapping_with_bindings(self):
        gc = MagicMock()
        # First query: mapping lookup
        # Second query: load_sections
        # Third query: load_signal_mappings
        gc.query.side_effect = [
            [
                {
                    "id": "jet:pf_active",
                    "facility_id": "jet",
                    "ids_name": "pf_active",
                    "dd_version": "4.1.1",
                    "provider": "imas-codex",
                    "static_config": "{}",
                }
            ],
            # Sections (POPULATES)
            [
                {
                    "root_path": "pf_active/coil",
                    "structure": "array_per_node",
                    "init_arrays": "{}",
                    "elements_config": '{"geometry_type": 2}',
                    "nested_path": None,
                    "parent_size": None,
                    "source_system": "PF",
                    "source_data_source": "device_xml",
                    "source_epoch_field": "introduced_version",
                    "source_select_via": None,
                    "enrichment": "[]",
                }
            ],
            # Field mappings (MAPS_TO_IMAS)
            [
                {
                    "source_id": "jet:ids:pf_active:PF",
                    "source_property": "r",
                    "target_id": "pf_active/coil/element/geometry/rectangle/r",
                    "transform_expression": "value",
                    "source_units": "m",
                    "target_units": "m",
                    "cocos_source": None,
                    "cocos_target": None,
                    "driver": "device_xml",
                    "cocos_label": None,
                }
            ],
        ]
        result = load_mapping("jet", "pf_active", gc)
        assert result is not None
        assert isinstance(result, Mapping)
        assert result.ids_name == "pf_active"
        assert result.dd_version == "4.1.1"
        assert len(result.bindings) == 1
        assert result.bindings[0].source_property == "r"
        assert len(result.sections) == 1


class TestLoadSignalMappings:
    def test_empty_result(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = load_signal_mappings("jet:pf_active", gc)
        assert result == []

    def test_parses_mapping_fields(self):
        gc = MagicMock()
        gc.query.return_value = [
            {
                "source_id": "jet:ids:pf_active:PF",
                "source_property": "r",
                "target_id": "pf_active/coil/element/geometry/rectangle/r",
                "transform_expression": "value * 1.0",
                "source_units": "m",
                "target_units": "m",
                "cocos_source": None,
                "cocos_target": None,
                "driver": "device_xml",
                "cocos_label": None,
            }
        ]
        result = load_signal_mappings("jet:pf_active", gc)
        assert len(result) == 1
        m = result[0]
        assert isinstance(m, SignalMapping)
        assert m.source_property == "r"
        assert m.transform_expression == "value * 1.0"
        assert m.source_units == "m"

    def test_defaults_for_missing_fields(self):
        gc = MagicMock()
        gc.query.return_value = [
            {
                "source_id": "jet:ids:pf_active:PF",
                "source_property": None,
                "target_id": "pf_active/coil/name",
                "transform_expression": None,
                "source_units": None,
                "target_units": None,
                "cocos_source": None,
                "cocos_target": None,
                "driver": None,
                "cocos_label": None,
            }
        ]
        result = load_signal_mappings("jet:pf_active", gc)
        m = result[0]
        assert m.source_property == "value"
        assert m.transform_expression == "value"
        assert m.driver == "device_xml"
        assert m.cocos_source is None
        assert m.cocos_target is None

    def test_cocos_warning_for_sensitive_path(self, caplog):
        """Warn when IMAS path is COCOS-sensitive but mapping has no cocos_source."""
        import logging

        gc = MagicMock()
        gc.query.return_value = [
            {
                "source_id": "jet:ids:pf_active:CI",
                "source_property": "current",
                "target_id": "pf_active/circuit/current",
                "transform_expression": "value",
                "source_units": "A",
                "target_units": "A",
                "cocos_source": None,
                "cocos_target": None,
                "driver": "device_xml",
                "cocos_label": "ip_like",
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = load_signal_mappings("jet:pf_active", gc)
        assert len(result) == 1
        assert "COCOS-sensitive" in caplog.text
        assert "ip_like" in caplog.text

    def test_no_cocos_warning_for_non_sensitive_path(self, caplog):
        """No warning for paths without cocos_transformation_type."""
        import logging

        gc = MagicMock()
        gc.query.return_value = [
            {
                "source_id": "jet:ids:pf_active:PF",
                "source_property": "r",
                "target_id": "pf_active/coil/element/geometry/rectangle/r",
                "transform_expression": "value",
                "source_units": "m",
                "target_units": "m",
                "cocos_source": None,
                "cocos_target": None,
                "driver": "device_xml",
                "cocos_label": None,
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = load_signal_mappings("jet:pf_active", gc)
        assert len(result) == 1
        assert "COCOS-sensitive" not in caplog.text


class TestCreateSignalSource:
    def test_creates_group_with_maps_to_imas(self):
        gc = MagicMock()
        gc.query.return_value = []
        specs: list[tuple[str, str, str, str | None, str | None]] = [
            ("r", "test/section/r", "value", "m", "m"),
            ("z", "test/section/z", "value", "m", "m"),
        ]
        result = create_signal_source("jet", "test", "section", "PF", specs, gc)
        assert result == "jet:ids:test:PF"
        # Called twice: create SignalSource node + create MAPS_TO_IMAS rels
        assert gc.query.call_count == 2


class TestCreateIMASMapping:
    def test_creates_mapping_with_signal_sources(self):
        gc = MagicMock()
        gc.query.return_value = []
        config = {"coil": {"source": {"system": "PF"}}}
        result = create_imas_mapping(
            "jet", "pf_active", "4.1.1", config, ["sg1", "sg2"], gc
        )
        assert result == "jet:pf_active"
        # create mapping + link signal sources + create POPULATES
        assert gc.query.call_count == 3
