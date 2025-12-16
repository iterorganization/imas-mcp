"""Tests for identifier schema extractor."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.core.extractors.base import ExtractorContext
from imas_codex.core.extractors.identifier_extractor import IdentifierExtractor


@pytest.fixture
def mock_dd_accessor():
    """Create a mock data dictionary accessor."""
    accessor = MagicMock()
    accessor.get_schema = MagicMock(return_value=None)
    return accessor


@pytest.fixture
def sample_ids_xml():
    """Create a sample IDS XML structure."""
    xml_str = """
    <ids name="equilibrium">
        <profiles_2d
            name="profiles_2d"
            path="equilibrium/profiles_2d"
            doc_identifier="equilibrium/equilibrium_profiles_2d_identifier.xml"
        />
        <time_slice name="time_slice" path="equilibrium/time_slice"/>
    </ids>
    """
    return ET.fromstring(xml_str)


@pytest.fixture
def sample_identifier_schema_xml():
    """Create a sample identifier schema XML."""
    xml_str = """
    <identifier documentation="Type of 2D profile">
        <int name="polar" description="Polar coordinates (R,Z)">1</int>
        <int name="flux" description="Flux coordinates">2</int>
        <int name="cartesian" description="Cartesian coordinates (X,Y,Z)">3</int>
    </identifier>
    """
    root = ET.fromstring(xml_str)
    return ET.ElementTree(root)


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


class TestIdentifierExtractor:
    """Tests for IdentifierExtractor."""

    def test_extract_no_doc_identifier(self, extractor_context, sample_ids_xml):
        """Test extraction when element has no doc_identifier."""
        extractor = IdentifierExtractor(extractor_context)

        time_slice_elem = sample_ids_xml.find(".//time_slice")
        result = extractor.extract(time_slice_elem)

        assert result == {}

    def test_extract_with_doc_identifier(
        self, extractor_context, sample_ids_xml, sample_identifier_schema_xml
    ):
        """Test extraction with valid doc_identifier."""
        extractor_context.dd_accessor.get_schema.return_value = (
            sample_identifier_schema_xml
        )

        extractor = IdentifierExtractor(extractor_context)

        profiles_elem = sample_ids_xml.find(".//profiles_2d")
        result = extractor.extract(profiles_elem)

        assert "identifier_schema" in result
        schema = result["identifier_schema"]
        assert (
            schema["schema_path"]
            == "equilibrium/equilibrium_profiles_2d_identifier.xml"
        )

    def test_extract_schema_data(self, extractor_context, sample_identifier_schema_xml):
        """Test _extract_schema_data method."""
        extractor = IdentifierExtractor(extractor_context)

        schema = extractor._extract_schema_data(
            sample_identifier_schema_xml, "test/schema.xml"
        )

        assert schema.schema_path == "test/schema.xml"
        assert len(schema.options) == 3

    def test_extract_identifier_options(
        self, extractor_context, sample_identifier_schema_xml
    ):
        """Test extraction of identifier options."""
        extractor = IdentifierExtractor(extractor_context)

        options = extractor._extract_identifier_options(
            sample_identifier_schema_xml.getroot()
        )

        assert len(options) == 3
        assert options[0].name == "polar"
        assert options[0].index == 1
        assert "Polar" in options[0].description

    def test_extract_imas_int_option(self, extractor_context):
        """Test extraction of IMAS <int> elements."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("int")
        elem.set("name", "test_option")
        elem.set("description", "Test description")
        elem.text = "5"

        option = extractor._extract_imas_int_option(elem)

        assert option.name == "test_option"
        assert option.index == 5
        assert option.description == "Test description"

    def test_extract_imas_int_option_invalid_index(self, extractor_context):
        """Test extraction with invalid index value."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("int")
        elem.set("name", "test_option")
        elem.text = "not_a_number"

        option = extractor._extract_imas_int_option(elem)

        assert option.name == "test_option"
        assert option.index == 0

    def test_extract_imas_int_option_no_name(self, extractor_context):
        """Test extraction with missing name."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("int")
        elem.text = "5"

        option = extractor._extract_imas_int_option(elem)

        assert option is None

    def test_is_identifier_option(self, extractor_context):
        """Test detection of identifier option elements."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("option")
        name_child = ET.SubElement(elem, "child")
        name_child.set("name", "name")
        index_child = ET.SubElement(elem, "child")
        index_child.set("name", "index")
        desc_child = ET.SubElement(elem, "child")
        desc_child.set("name", "description")

        assert extractor._is_identifier_option(elem) is True

    def test_is_identifier_option_missing_fields(self, extractor_context):
        """Test identifier option detection with missing fields."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("option")
        name_child = ET.SubElement(elem, "child")
        name_child.set("name", "name")

        assert extractor._is_identifier_option(elem) is False

    def test_extract_single_option(self, extractor_context):
        """Test extraction of a single identifier option."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("option")

        name_child = ET.SubElement(elem, "field")
        name_child.set("name", "name")
        name_child.set("value", "test_name")

        index_child = ET.SubElement(elem, "field")
        index_child.set("name", "index")
        index_child.text = "42"

        desc_child = ET.SubElement(elem, "field")
        desc_child.set("name", "description")
        desc_child.text = "Test description"

        option = extractor._extract_single_option(elem)

        assert option.name == "test_name"
        assert option.index == 42
        assert option.description == "Test description"

    def test_extract_single_option_no_name(self, extractor_context):
        """Test extraction when option has no name."""
        extractor = IdentifierExtractor(extractor_context)

        elem = ET.Element("option")

        index_child = ET.SubElement(elem, "field")
        index_child.set("name", "index")
        index_child.text = "42"

        option = extractor._extract_single_option(elem)

        assert option is None

    def test_parse_documentation_enums_pattern1(self, extractor_context):
        """Test parsing of ID=value: description pattern."""
        extractor = IdentifierExtractor(extractor_context)

        doc = "ID=1: first option; ID=2: second option; ID=3: third option"
        options = extractor._parse_documentation_enums(doc)

        assert len(options) == 3
        assert options[0].index == 1
        assert options[1].index == 2

    def test_parse_documentation_enums_pattern2(self, extractor_context):
        """Test parsing of value: description pattern."""
        extractor = IdentifierExtractor(extractor_context)

        doc = "1: First option, 2: Second option, 3: Third option"
        options = extractor._parse_documentation_enums(doc)

        assert len(options) == 3

    def test_parse_documentation_enums_empty(self, extractor_context):
        """Test parsing with no enumeration patterns."""
        extractor = IdentifierExtractor(extractor_context)

        doc = "This is just regular documentation text"
        options = extractor._parse_documentation_enums(doc)

        assert len(options) == 0

    def test_extract_alternative_patterns(self, extractor_context):
        """Test extraction of alternative enumeration patterns."""
        extractor = IdentifierExtractor(extractor_context)

        root = ET.Element("identifier")
        root.set("documentation", "ID=1: option1; ID=2: option2")

        options = extractor._extract_alternative_patterns(root)

        assert len(options) >= 2

    def test_parse_identifier_schema_no_accessor(self, extractor_context):
        """Test schema parsing when accessor has no get_schema method."""
        extractor = IdentifierExtractor(extractor_context)

        delattr(extractor_context.dd_accessor, "get_schema")

        result = extractor._parse_identifier_schema("test/schema.xml")

        assert result is None

    def test_parse_identifier_schema_returns_none(self, extractor_context):
        """Test schema parsing when accessor returns None."""
        extractor = IdentifierExtractor(extractor_context)

        extractor_context.dd_accessor.get_schema.return_value = None

        result = extractor._parse_identifier_schema("test/schema.xml")

        assert result is None

    def test_parse_identifier_schema_exception(self, extractor_context):
        """Test schema parsing when exception occurs."""
        extractor = IdentifierExtractor(extractor_context)

        extractor_context.dd_accessor.get_schema.side_effect = Exception("Test error")

        result = extractor._parse_identifier_schema("test/schema.xml")

        assert result is None

    def test_options_sorted_by_index(
        self, extractor_context, sample_identifier_schema_xml
    ):
        """Test that options are sorted by index."""
        extractor = IdentifierExtractor(extractor_context)

        options = extractor._extract_identifier_options(
            sample_identifier_schema_xml.getroot()
        )

        indices = [opt.index for opt in options]
        assert indices == sorted(indices)
