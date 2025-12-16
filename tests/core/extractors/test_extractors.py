"""Tests for metadata, validation, coordinate, and physics extractors."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.core.extractors.base import ExtractorContext
from imas_codex.core.extractors.coordinate_extractor import CoordinateExtractor
from imas_codex.core.extractors.metadata_extractor import MetadataExtractor
from imas_codex.core.extractors.physics_extractor import LifecycleExtractor
from imas_codex.core.extractors.semantic_extractor import (
    PathExtractor,
    SemanticExtractor,
)
from imas_codex.core.extractors.validation_extractor import ValidationExtractor


@pytest.fixture
def mock_dd_accessor():
    """Create a mock data dictionary accessor."""
    accessor = MagicMock()
    return accessor


@pytest.fixture
def sample_ids_xml():
    """Create a sample IDS XML structure."""
    xml_str = """
    <ids name="equilibrium" documentation="Equilibrium data">
        <time_slice documentation="Time slice data">
            <profiles_1d documentation="1D profiles">
                <temperature
                    name="temperature"
                    path="equilibrium/time_slice/profiles_1d/temperature"
                    documentation="Temperature profile"
                    units="eV"
                    data_type="FLT_1D"
                    coordinate1="rho_tor_norm"
                    coordinate2="time"
                    timebase="time"
                    type="dynamic"
                    lifecycle_status="active"
                    lifecycle_version="4.0.0"
                    introduced_after_version="3.0.0"
                />
                <density
                    name="density"
                    path="equilibrium/time_slice/profiles_1d/density"
                    documentation="Density profile"
                    units="m^-3"
                    data_type="FLT_1D"
                    min="0"
                    max="1e25"
                />
                <dimensional
                    name="volume"
                    path="equilibrium/time_slice/profiles_1d/volume"
                    documentation="Volume"
                    units="m^dimension"
                    coordinate1="rho_tor_norm"
                />
            </profiles_1d>
        </time_slice>
        <coordinate name="coord1" documentation="Coordinate 1">
            <identifier name="r" documentation="Radial"/>
        </coordinate>
    </ids>
    """
    return ET.fromstring(xml_str)


@pytest.fixture
def extractor_context(mock_dd_accessor, sample_ids_xml):
    """Create an ExtractorContext for testing."""
    parent_map = {child: parent for parent in sample_ids_xml.iter() for child in parent}
    return ExtractorContext(
        dd_accessor=mock_dd_accessor,
        root=sample_ids_xml,
        ids_elem=sample_ids_xml,
        ids_name="equilibrium",
        parent_map=parent_map,
        excluded_patterns=set(),
        include_ggd=True,
        include_error_fields=True,
    )


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_extract_basic_metadata(self, extractor_context, sample_ids_xml):
        """Test extraction of basic metadata fields."""
        extractor = MetadataExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert "units" in result
        assert result["units"] == "eV"
        assert "data_type" in result
        assert result["data_type"] == "FLT_1D"

    def test_extract_coordinates(self, extractor_context, sample_ids_xml):
        """Test extraction of coordinate information."""
        extractor = MetadataExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert "coordinates" in result
        assert "rho_tor_norm" in result["coordinates"]
        assert "time" in result["coordinates"]
        assert result["coordinate1"] == "rho_tor_norm"
        assert result["coordinate2"] == "time"

    def test_extract_lifecycle_fields(self, extractor_context, sample_ids_xml):
        """Test extraction of lifecycle metadata."""
        extractor = MetadataExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert result.get("lifecycle_status") == "active"
        assert result.get("lifecycle_version") == "4.0.0"
        assert result.get("introduced_after_version") == "3.0.0"

    def test_extract_timebase_and_type(self, extractor_context, sample_ids_xml):
        """Test extraction of timebase and type."""
        extractor = MetadataExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert result.get("timebase") == "time"
        assert result.get("type") == "dynamic"

    def test_resolve_dimensional_units(self, extractor_context, sample_ids_xml):
        """Test resolution of dimensional units."""
        extractor = MetadataExtractor(extractor_context)

        vol_elem = sample_ids_xml.find(".//dimensional")
        result = extractor.extract(vol_elem)

        assert "units" in result

    def test_resolve_coordinate_system_units(self, extractor_context):
        """Test resolution of coordinate system unit references."""
        extractor = MetadataExtractor(extractor_context)

        resolved = extractor._resolve_coordinate_system_units(
            "units given by coordinate_system(:)/coordinate(r)/units"
        )
        assert resolved == "m"

        resolved = extractor._resolve_coordinate_system_units(
            "units given by coordinate_system(:)/coordinate(phi)/units"
        )
        assert resolved == "rad"

        resolved = extractor._resolve_coordinate_system_units(
            "units given by coordinate_system(:)/coordinate(:)/units"
        )
        assert resolved == ""

    def test_resolve_process_units(self, extractor_context, sample_ids_xml):
        """Test resolution of process-based unit references."""
        extractor = MetadataExtractor(extractor_context)
        temp_elem = sample_ids_xml.find(".//temperature")

        resolved = extractor._resolve_process_units(
            "units given by process(:)/results_units", temp_elem
        )
        assert resolved == ""

    def test_clean_metadata(self, extractor_context):
        """Test metadata cleaning."""
        extractor = MetadataExtractor(extractor_context)

        metadata = {
            "documentation": "Test doc",
            "units": "",
            "coordinates": [],
            "data_type": "FLT_1D",
            "extra_field": None,
        }

        cleaned = extractor._clean_metadata(metadata)
        assert "documentation" in cleaned
        assert "data_type" in cleaned

    def test_fallback_documentation(self, extractor_context):
        """Test fallback to direct documentation when hierarchy unavailable."""
        extractor = MetadataExtractor(extractor_context)

        elem = ET.Element("field")
        elem.set("documentation", "Direct documentation text")

        with patch(
            "imas_codex.core.xml_utils.DocumentationBuilder.collect_documentation_hierarchy",
            return_value=[],
        ):
            result = extractor.extract(elem)
            assert result.get("documentation") == "Direct documentation text"


class TestValidationExtractor:
    """Tests for ValidationExtractor."""

    def test_extract_validation_rules(self, extractor_context, sample_ids_xml):
        """Test extraction of validation rules."""
        extractor = ValidationExtractor(extractor_context)

        density_elem = sample_ids_xml.find(".//density")
        result = extractor.extract(density_elem)

        assert "validation_rules" in result
        rules = result["validation_rules"]
        assert rules.get("min_value") == "0"
        assert rules.get("max_value") == "1e25"
        assert rules.get("data_type") == "FLT_1D"

    def test_units_required_flag(self, extractor_context, sample_ids_xml):
        """Test units_required flag in validation rules."""
        extractor = ValidationExtractor(extractor_context)

        density_elem = sample_ids_xml.find(".//density")
        result = extractor.extract(density_elem)

        assert result["validation_rules"]["units_required"] is True

    def test_no_units_required(self, extractor_context):
        """Test when units are empty or dimensionless."""
        extractor = ValidationExtractor(extractor_context)

        elem = ET.Element("field")
        elem.set("units", "1")
        elem.set("data_type", "INT")

        result = extractor.extract(elem)

        assert result["validation_rules"]["units_required"] is False


class TestCoordinateExtractor:
    """Tests for CoordinateExtractor."""

    def test_extract_returns_empty_for_element(self, extractor_context, sample_ids_xml):
        """Test that extract returns empty dict for individual elements."""
        extractor = CoordinateExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert result == {}

    def test_extract_coordinate_systems(self, extractor_context, sample_ids_xml):
        """Test extraction of coordinate systems from IDS."""
        extractor = CoordinateExtractor(extractor_context)

        result = extractor.extract_coordinate_systems(sample_ids_xml)

        assert "coord1" in result
        assert result["coord1"]["name"] == "coord1"

    def test_extract_coordinate_identifiers(self, extractor_context, sample_ids_xml):
        """Test extraction of identifiers within coordinate systems."""
        extractor = CoordinateExtractor(extractor_context)

        result = extractor.extract_coordinate_systems(sample_ids_xml)

        assert "identifiers" in result["coord1"]
        identifiers = result["coord1"]["identifiers"]
        assert len(identifiers) > 0
        assert identifiers[0]["name"] == "r"


class TestLifecycleExtractor:
    """Tests for LifecycleExtractor (physics_extractor.py)."""

    def test_extract_lifecycle_data(self, extractor_context, sample_ids_xml):
        """Test extraction of lifecycle information."""
        extractor = LifecycleExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert result.get("lifecycle") == "active"
        assert result.get("lifecycle_version") == "4.0.0"

    def test_extract_no_lifecycle(self, extractor_context):
        """Test extraction when no lifecycle info present."""
        extractor = LifecycleExtractor(extractor_context)

        elem = ET.Element("field")
        result = extractor.extract(elem)

        assert result == {}


class TestPathExtractor:
    """Tests for PathExtractor."""

    def test_extract_path(self, extractor_context, sample_ids_xml):
        """Test path extraction from element."""
        extractor = PathExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert "path" in result
        assert "equilibrium" in result["path"]

    def test_build_element_path(self, extractor_context, sample_ids_xml):
        """Test building full element path."""
        extractor = PathExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        path = extractor._build_element_path(temp_elem)

        assert path.startswith("equilibrium/")


class TestSemanticExtractor:
    """Tests for SemanticExtractor."""

    def test_extract_returns_empty(self, extractor_context, sample_ids_xml):
        """Test that extract returns empty dict for individual elements."""
        extractor = SemanticExtractor(extractor_context)

        temp_elem = sample_ids_xml.find(".//temperature")
        result = extractor.extract(temp_elem)

        assert result == {}

    def test_extract_semantic_groups(self, extractor_context):
        """Test semantic grouping of paths."""
        extractor = SemanticExtractor(extractor_context)

        paths = {
            "equilibrium/temp1": {"units": "eV", "coordinates": []},
            "equilibrium/temp2": {"units": "eV", "coordinates": []},
            "equilibrium/density": {"units": "m^-3", "coordinates": []},
        }

        groups = extractor.extract_semantic_groups(paths)

        assert len(groups) >= 1

    def test_determine_semantic_group_by_units(self, extractor_context):
        """Test semantic grouping by units."""
        extractor = SemanticExtractor(extractor_context)

        group = extractor._determine_semantic_group(
            "equilibrium/temperature", {"units": "eV", "coordinates": []}
        )

        assert group == "units_eV"

    def test_determine_semantic_group_by_coordinates(self, extractor_context):
        """Test semantic grouping by coordinates."""
        extractor = SemanticExtractor(extractor_context)

        group = extractor._determine_semantic_group(
            "equilibrium/temperature",
            {"units": "", "coordinates": ["rho_tor_norm", "time"]},
        )

        assert group == "coordinates_rho_tor_norm_time"

    def test_determine_semantic_group_by_structure(self, extractor_context):
        """Test semantic grouping by path structure."""
        extractor = SemanticExtractor(extractor_context)

        group = extractor._determine_semantic_group(
            "equilibrium/time_slice/profiles_1d/temperature",
            {"units": "", "coordinates": []},
        )

        assert "structure_time_slice" in group

    def test_filter_single_item_groups(self, extractor_context):
        """Test that single-item groups are filtered out."""
        extractor = SemanticExtractor(extractor_context)

        paths = {
            "equilibrium/temp1": {"units": "eV", "coordinates": []},
            "equilibrium/density": {"units": "m^-3", "coordinates": []},
        }

        groups = extractor.extract_semantic_groups(paths)

        for group_paths in groups.values():
            assert len(group_paths) > 1
