"""Tests for extractor base classes and utilities."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.core.extractors.base import (
    BaseExtractor,
    ComposableExtractor,
    ExtractorContext,
)


class MockExtractor(BaseExtractor):
    """Mock extractor for testing base class functionality."""

    def extract(self, elem: ET.Element) -> dict:
        return {"extracted": elem.get("name", "unknown")}


class FailingExtractor(BaseExtractor):
    """Mock extractor that raises an exception."""

    def extract(self, elem: ET.Element) -> dict:
        raise ValueError("Intentional failure")


@pytest.fixture
def mock_dd_accessor():
    """Create a mock data dictionary accessor."""
    accessor = MagicMock()
    accessor.get_schema = MagicMock(return_value=None)
    return accessor


@pytest.fixture
def sample_xml():
    """Create sample XML for testing."""
    xml_str = """
    <ids name="test_ids">
        <field name="temperature" path="test_ids/temperature" units="eV" documentation="Temperature field">
            <subfield name="value" path="test_ids/temperature/value" data_type="FLT_1D"/>
        </field>
        <field name="density" path="test_ids/density" units="m^-3"/>
        <field name="ggd_field" path="test_ids/ggd/field"/>
        <field name="error_field" path="test_ids/error_upper"/>
        <field name="ids_properties" path="ids_properties/something"/>
    </ids>
    """
    return ET.fromstring(xml_str)


@pytest.fixture
def extractor_context(mock_dd_accessor, sample_xml):
    """Create an ExtractorContext for testing."""
    root = sample_xml
    return ExtractorContext(
        dd_accessor=mock_dd_accessor,
        root=root,
        ids_elem=root,
        ids_name="test_ids",
        parent_map={},
        excluded_patterns={"excluded_pattern"},
        include_ggd=False,
        include_error_fields=False,
    )


class TestExtractorContext:
    """Tests for ExtractorContext dataclass."""

    def test_context_initialization(self, extractor_context):
        """Test that context initializes correctly."""
        assert extractor_context.ids_name == "test_ids"
        assert extractor_context.include_ggd is False
        assert extractor_context.include_error_fields is False

    def test_parent_map_built(self, extractor_context):
        """Test that parent map is built in post_init."""
        assert isinstance(extractor_context.parent_map, dict)


class TestBaseExtractor:
    """Tests for BaseExtractor filtering logic."""

    def test_filter_element_with_excluded_pattern(self, extractor_context):
        """Test filtering elements with excluded patterns."""
        extractor = MockExtractor(extractor_context)

        elem = ET.Element("field")
        elem.set("path", "test/excluded_pattern/field")
        elem.set("name", "test_field")

        assert extractor.should_filter_element_or_path(elem=elem) is True

    def test_filter_ggd_when_excluded(self, extractor_context):
        """Test GGD elements are filtered when include_ggd is False."""
        extractor = MockExtractor(extractor_context)

        elem = ET.Element("field")
        elem.set("path", "test/ggd/field")
        elem.set("name", "ggd_value")

        assert extractor.should_filter_element_or_path(elem=elem) is True

    def test_allow_ggd_when_included(self, mock_dd_accessor, sample_xml):
        """Test GGD elements are allowed when include_ggd is True."""
        context = ExtractorContext(
            dd_accessor=mock_dd_accessor,
            root=sample_xml,
            ids_elem=sample_xml,
            ids_name="test_ids",
            parent_map={},
            excluded_patterns=set(),
            include_ggd=True,
            include_error_fields=False,
        )
        extractor = MockExtractor(context)

        elem = ET.Element("field")
        elem.set("path", "test/some/field")
        elem.set("name", "normal_field")

        assert extractor.should_filter_element_or_path(elem=elem) is False

    def test_filter_error_fields_when_excluded(self, extractor_context):
        """Test error fields are filtered when include_error_fields is False."""
        extractor = MockExtractor(extractor_context)

        elem = ET.Element("field")
        elem.set("path", "test/field")
        elem.set("name", "temperature_error_upper")

        assert extractor.should_filter_element_or_path(elem=elem) is True

    def test_is_error_field_patterns(self, extractor_context):
        """Test error field pattern detection."""
        extractor = MockExtractor(extractor_context)

        assert extractor._is_error_field("_error_upper") is True
        assert extractor._is_error_field("temp_error_lower") is True
        assert extractor._is_error_field("error_index") is True
        assert extractor._is_error_field("normal_field") is False
        assert extractor._is_error_field(None) is False

    def test_filter_coordinate_noise(self, extractor_context):
        """Test coordinate-specific noise filtering."""
        extractor = MockExtractor(extractor_context)

        assert (
            extractor._filter_coordinate_noise("ids_properties/coord", "coord") is True
        )
        assert extractor._filter_coordinate_noise(None, "coord") is True
        assert extractor._filter_coordinate_noise("normal/path", "coord") is False

    def test_filter_relationship_noise(self, extractor_context):
        """Test relationship-specific noise filtering."""
        extractor = MockExtractor(extractor_context)

        generic_names = ["name", "description", "type", "value", "index"]
        for name in generic_names:
            assert extractor._filter_relationship_noise("some/path", name) is True

        assert extractor._filter_relationship_noise("some/path", "temperature") is False
        assert extractor._filter_relationship_noise("some/path", None) is True

    def test_filter_element_noise(self, extractor_context):
        """Test basic element noise filtering."""
        extractor = MockExtractor(extractor_context)

        assert extractor._filter_element_noise("any/path", "any_name") is False

    def test_filter_with_no_identifying_info(self, extractor_context):
        """Test filtering when element has no path or name."""
        extractor = MockExtractor(extractor_context)

        elem = ET.Element("field")
        assert extractor.should_filter_element_or_path(elem=elem) is True

    def test_filter_by_path_string(self, extractor_context):
        """Test filtering using path string directly."""
        extractor = MockExtractor(extractor_context)

        assert (
            extractor.should_filter_element_or_path(
                path="test/excluded_pattern/value", elem_name="value"
            )
            is True
        )
        assert (
            extractor.should_filter_element_or_path(
                path="test/normal/value", elem_name="value"
            )
            is False
        )

    def test_filter_type_coordinate(self, extractor_context):
        """Test coordinate filter type."""
        extractor = MockExtractor(extractor_context)

        assert (
            extractor.should_filter_element_or_path(
                path="ids_properties/coord",
                elem_name="coord",
                filter_type="coordinate",
            )
            is True
        )

    def test_filter_type_relationship(self, extractor_context):
        """Test relationship filter type."""
        extractor = MockExtractor(extractor_context)

        assert (
            extractor.should_filter_element_or_path(
                path="test/path", elem_name="name", filter_type="relationship"
            )
            is True
        )


class TestComposableExtractor:
    """Tests for ComposableExtractor composition."""

    def test_extract_all_combines_results(self, extractor_context):
        """Test that extract_all combines results from multiple extractors."""
        extractor1 = MockExtractor(extractor_context)
        extractor2 = MockExtractor(extractor_context)

        composer = ComposableExtractor([extractor1, extractor2])

        elem = ET.Element("field")
        elem.set("name", "test_field")

        result = composer.extract_all(elem)

        assert "extracted" in result
        assert result["extracted"] == "test_field"

    def test_extract_all_handles_failures(self, extractor_context, capsys):
        """Test that extract_all continues after extractor failure."""
        good_extractor = MockExtractor(extractor_context)
        bad_extractor = FailingExtractor(extractor_context)

        composer = ComposableExtractor([bad_extractor, good_extractor])

        elem = ET.Element("field")
        elem.set("name", "test_field")

        result = composer.extract_all(elem)

        assert "extracted" in result
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_extract_all_with_empty_list(self, extractor_context):
        """Test extract_all with no extractors."""
        composer = ComposableExtractor([])

        elem = ET.Element("field")
        result = composer.extract_all(elem)

        assert result == {}
