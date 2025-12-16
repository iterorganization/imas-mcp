"""Tests for core/xml_parser.py - DataDictionaryTransformer."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_mcp.core.xml_parser import DataDictionaryTransformer


class MockDDAccessor:
    """Mock DataDictionaryAccessor for testing."""

    def __init__(self, xml_content: str | None = None):
        self._xml_content = xml_content or self._default_xml()
        self._tree = ET.ElementTree(ET.fromstring(self._xml_content))

    def _default_xml(self) -> str:
        """Return minimal valid DD XML for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <data_dictionary>
            <version>4.0.0</version>
            <IDS name="equilibrium" documentation="MHD equilibrium data">
                <field name="time" documentation="Time" data_type="FLT_0D" units="s"/>
                <field name="time_slice" documentation="Time slices" maxoccur="unbounded">
                    <field name="time" documentation="Time of slice" data_type="FLT_0D" units="s"/>
                    <field name="profiles_1d" documentation="1D profiles">
                        <field name="psi" documentation="Poloidal flux" data_type="FLT_1D" units="Wb" coordinate1="1...N"/>
                        <field name="pressure" documentation="Pressure" data_type="FLT_1D" units="Pa" coordinate1="1...N"/>
                    </field>
                    <field name="boundary" documentation="Plasma boundary">
                        <field name="psi" documentation="Boundary psi" data_type="FLT_0D" units="Wb"/>
                        <field name="r" documentation="Major radius" data_type="FLT_1D" units="m"/>
                        <field name="z" documentation="Vertical position" data_type="FLT_1D" units="m"/>
                    </field>
                </field>
            </IDS>
            <IDS name="core_profiles" documentation="Core plasma profiles">
                <field name="time" documentation="Time" data_type="FLT_0D" units="s"/>
                <field name="profiles_1d" documentation="1D profiles" maxoccur="unbounded">
                    <field name="time" documentation="Time" data_type="FLT_0D" units="s"/>
                    <field name="electrons" documentation="Electron profiles">
                        <field name="temperature" documentation="Electron temperature" data_type="FLT_1D" units="eV" coordinate1="rho_tor_norm"/>
                        <field name="density" documentation="Electron density" data_type="FLT_1D" units="m^-3" coordinate1="rho_tor_norm"/>
                    </field>
                    <field name="ion" documentation="Ion profiles" maxoccur="unbounded">
                        <field name="temperature" documentation="Ion temperature" data_type="FLT_1D" units="eV" coordinate1="rho_tor_norm"/>
                    </field>
                </field>
            </IDS>
        </data_dictionary>"""

    def get_xml_tree(self) -> ET.ElementTree:
        return self._tree

    def get_version(self) -> MagicMock:
        version = MagicMock()
        version.public = "4.0.0"
        return version

    def is_available(self) -> bool:
        return True

    def get_schema(self, schema_path: str):
        return None


@pytest.fixture
def mock_dd_accessor():
    """Create a mock DD accessor."""
    return MockDDAccessor()


@pytest.fixture
def transformer(tmp_path, mock_dd_accessor):
    """Create a DataDictionaryTransformer with mocked dependencies."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch(
        "imas_mcp.core.xml_parser.ResourcePathAccessor"
    ) as mock_path_accessor_class:
        mock_path_accessor = MagicMock()
        mock_path_accessor.dd_accessor = mock_dd_accessor
        mock_path_accessor.schemas_dir = output_dir
        mock_path_accessor_class.return_value = mock_path_accessor

        transformer = DataDictionaryTransformer(
            dd_version="4.0.0",
            output_dir=output_dir,
            dd_accessor=mock_dd_accessor,
            use_rich=False,
        )
        return transformer


class TestDataDictionaryTransformerInit:
    """Tests for DataDictionaryTransformer initialization."""

    def test_initialization_with_dd_accessor(self, tmp_path, mock_dd_accessor):
        """Test transformer initializes correctly with provided dd_accessor."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transformer = DataDictionaryTransformer(
            dd_version="4.0.0",
            output_dir=output_dir,
            dd_accessor=mock_dd_accessor,
            use_rich=False,
        )

        assert transformer.dd_accessor == mock_dd_accessor
        assert transformer.output_dir == output_dir
        assert transformer.dd_version == "4.0.0"

    def test_initialization_creates_output_dir(self, tmp_path, mock_dd_accessor):
        """Test transformer creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "output"

        DataDictionaryTransformer(
            dd_version="4.0.0",
            output_dir=output_dir,
            dd_accessor=mock_dd_accessor,
            use_rich=False,
        )

        assert output_dir.exists()

    def test_initialization_with_ids_set(self, tmp_path, mock_dd_accessor):
        """Test transformer respects ids_set filter."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transformer = DataDictionaryTransformer(
            dd_version="4.0.0",
            output_dir=output_dir,
            dd_accessor=mock_dd_accessor,
            ids_set={"equilibrium"},
            use_rich=False,
        )

        assert transformer.ids_set == {"equilibrium"}

    def test_initialization_builds_parent_map(self, transformer):
        """Test transformer builds global parent map on init."""
        assert transformer._global_parent_map is not None
        assert len(transformer._global_parent_map) > 0

    def test_resolved_output_dir(self, transformer):
        """Test resolved_output_dir property."""
        assert transformer.resolved_output_dir is not None
        assert isinstance(transformer.resolved_output_dir, Path)


class TestGlobalParentMap:
    """Tests for parent map building."""

    def test_parent_map_structure(self, transformer):
        """Test parent map maps children to parents correctly."""
        root = transformer._root
        parent_map = transformer._global_parent_map

        # Find a child element and verify its parent
        for parent in root.iter():
            for child in parent:
                if child in parent_map:
                    assert parent_map[child] == parent


class TestElementCaching:
    """Tests for element caching methods."""

    def test_cached_elements_by_name(self, transformer):
        """Test _get_cached_elements_by_name caches results."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        # First call populates cache
        elements1 = transformer._get_cached_elements_by_name(ids_elem, "equilibrium")
        # Second call should use cache
        elements2 = transformer._get_cached_elements_by_name(ids_elem, "equilibrium")

        assert elements1 == elements2
        assert "equilibrium_named_elements" in transformer._element_cache

    def test_cached_elements_by_attribute(self, transformer):
        """Test _get_cached_elements_by_attribute caches results."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        elements = transformer._get_cached_elements_by_attribute(
            ids_elem, "equilibrium", "documentation"
        )

        assert len(elements) > 0
        assert "equilibrium_documentation_elements" in transformer._element_cache


class TestIdsInfoExtraction:
    """Tests for IDS info extraction."""

    def test_extract_ids_info(self, transformer):
        """Test extracting IDS-level information."""
        from imas_mcp.core.extractors import ExtractorContext

        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        context = ExtractorContext(
            dd_accessor=transformer.dd_accessor,
            root=root,
            ids_elem=ids_elem,
            ids_name="equilibrium",
            parent_map=transformer._global_parent_map,
            excluded_patterns=transformer.excluded_patterns,
            include_ggd=transformer.include_ggd,
            include_error_fields=transformer.include_error_fields,
        )

        ids_info = transformer._extract_ids_info(ids_elem, "equilibrium", context)

        assert ids_info["name"] == "equilibrium"
        assert "description" in ids_info
        assert "physics_domain" in ids_info
        assert "max_depth" in ids_info
        assert "leaf_count" in ids_info
        assert "documentation_coverage" in ids_info

    def test_infer_physics_domain(self, transformer):
        """Test physics domain inference returns string."""
        domain = transformer._infer_physics_domain("equilibrium")
        assert isinstance(domain, str)
        assert len(domain) > 0

        domain = transformer._infer_physics_domain("core_profiles")
        assert isinstance(domain, str)
        assert len(domain) > 0


class TestMaxDepthCalculation:
    """Tests for depth calculation."""

    def test_calculate_max_depth(self, transformer):
        """Test max depth calculation."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        depth = transformer._calculate_max_depth(ids_elem)

        assert depth > 0
        assert isinstance(depth, int)

    def test_max_depth_single_element(self, transformer):
        """Test max depth for single element."""
        elem = ET.Element("single")
        depth = transformer._calculate_max_depth(elem)
        assert depth == 0

    def test_max_depth_nested_structure(self, transformer):
        """Test max depth for nested structure."""
        root = ET.Element("root")
        child = ET.SubElement(root, "child")
        ET.SubElement(child, "grandchild")

        depth = transformer._calculate_max_depth(root)
        assert depth == 2


class TestLeafNodeExtraction:
    """Tests for leaf node extraction."""

    def test_get_leaf_nodes(self, transformer):
        """Test leaf node extraction."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        leaves = transformer._get_leaf_nodes(ids_elem)

        assert len(leaves) > 0
        # Leaf nodes should have no children
        for leaf in leaves:
            assert len(list(leaf)) == 0


class TestDocumentationCoverage:
    """Tests for documentation coverage calculation."""

    def test_calculate_documentation_coverage(self, transformer):
        """Test documentation coverage calculation."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        coverage = transformer._calculate_documentation_coverage(ids_elem)

        assert 0.0 <= coverage <= 1.0

    def test_documentation_coverage_empty_element(self, transformer):
        """Test coverage for element with no named children."""
        elem = ET.Element("empty")
        coverage = transformer._calculate_documentation_coverage(elem)
        assert coverage == 0.0


class TestElementPathBuilding:
    """Tests for element path building."""

    def test_build_element_path(self, transformer):
        """Test building element path."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        # Find a nested element
        psi_elem = ids_elem.find(".//field[@name='psi']")

        path = transformer._build_element_path(
            psi_elem, ids_elem, "equilibrium", transformer._global_parent_map
        )

        assert path is not None
        assert "equilibrium" in path
        assert "psi" in path

    def test_build_element_path_caching(self, transformer):
        """Test path caching works correctly."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")
        psi_elem = ids_elem.find(".//field[@name='psi']")

        # First call
        path1 = transformer._build_element_path(
            psi_elem, ids_elem, "equilibrium", transformer._global_parent_map
        )
        # Second call should use cache
        path2 = transformer._build_element_path(
            psi_elem, ids_elem, "equilibrium", transformer._global_parent_map
        )

        assert path1 == path2

    def test_build_element_path_empty(self, transformer):
        """Test building path for element at IDS root returns None."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        # IDS element itself has no path
        path = transformer._build_element_path(
            ids_elem, ids_elem, "equilibrium", transformer._global_parent_map
        )

        assert path is None


class TestElementFiltering:
    """Tests for element filtering."""

    def test_should_skip_element_no_name(self, transformer):
        """Test elements without name are skipped."""
        elem = ET.Element("unnamed")
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        should_skip = transformer._should_skip_element(
            elem, ids_elem, transformer._global_parent_map
        )

        assert should_skip is True

    def test_should_skip_excluded_patterns(self, transformer):
        """Test excluded patterns are skipped."""
        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        # Create element with excluded pattern name
        elem = ET.SubElement(ids_elem, "field")
        elem.set("name", "ids_properties")

        should_skip = transformer._should_skip_element(
            elem, ids_elem, transformer._global_parent_map
        )

        # ids_properties should be excluded
        assert should_skip is True


class TestIdsDataExtraction:
    """Tests for full IDS data extraction."""

    def test_extract_ids_data(self, transformer):
        """Test full IDS data extraction."""
        root = transformer._root
        ids_data = transformer._extract_ids_data(root)

        assert "equilibrium" in ids_data
        assert "core_profiles" in ids_data

        for _ids_name, data in ids_data.items():
            assert "ids_info" in data
            assert "coordinate_systems" in data
            assert "paths" in data
            assert "semantic_groups" in data

    def test_extract_ids_data_with_filter(self, tmp_path, mock_dd_accessor):
        """Test IDS data extraction respects ids_set filter."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transformer = DataDictionaryTransformer(
            dd_version="4.0.0",
            output_dir=output_dir,
            dd_accessor=mock_dd_accessor,
            ids_set={"equilibrium"},
            use_rich=False,
        )

        root = transformer._root
        ids_data = transformer._extract_ids_data(root)

        assert "equilibrium" in ids_data
        assert "core_profiles" not in ids_data


class TestPathExtraction:
    """Tests for path extraction."""

    def test_extract_paths(self, transformer):
        """Test path extraction from IDS element."""
        from imas_mcp.core.extractors import ExtractorContext

        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        context = ExtractorContext(
            dd_accessor=transformer.dd_accessor,
            root=root,
            ids_elem=ids_elem,
            ids_name="equilibrium",
            parent_map=transformer._global_parent_map,
            excluded_patterns=transformer.excluded_patterns,
            include_ggd=transformer.include_ggd,
            include_error_fields=transformer.include_error_fields,
        )

        paths = transformer._extract_paths(ids_elem, "equilibrium", context)

        assert len(paths) > 0
        # Check path structure
        for path, metadata in paths.items():
            assert "equilibrium" in path
            assert "path" in metadata


class TestCoordinateSystemExtraction:
    """Tests for coordinate system extraction."""

    def test_extract_coordinate_systems(self, transformer):
        """Test coordinate system extraction."""
        from imas_mcp.core.extractors import ExtractorContext

        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        context = ExtractorContext(
            dd_accessor=transformer.dd_accessor,
            root=root,
            ids_elem=ids_elem,
            ids_name="equilibrium",
            parent_map=transformer._global_parent_map,
            excluded_patterns=transformer.excluded_patterns,
            include_ggd=transformer.include_ggd,
            include_error_fields=transformer.include_error_fields,
        )

        coord_systems = transformer._extract_coordinate_systems(ids_elem, context)

        assert isinstance(coord_systems, dict)


class TestSemanticGroupExtraction:
    """Tests for semantic group extraction."""

    def test_extract_semantic_groups(self, transformer):
        """Test semantic group extraction."""
        from imas_mcp.core.extractors import ExtractorContext

        root = transformer._root
        ids_elem = root.find(".//IDS[@name='equilibrium']")

        context = ExtractorContext(
            dd_accessor=transformer.dd_accessor,
            root=root,
            ids_elem=ids_elem,
            ids_name="equilibrium",
            parent_map=transformer._global_parent_map,
            excluded_patterns=transformer.excluded_patterns,
            include_ggd=transformer.include_ggd,
            include_error_fields=transformer.include_error_fields,
        )

        # First extract paths
        paths = transformer._extract_paths(ids_elem, "equilibrium", context)
        # Then extract semantic groups
        groups = transformer._extract_semantic_groups(paths, context)

        assert isinstance(groups, dict)


class TestGraphAnalysis:
    """Tests for graph structure analysis."""

    def test_analyze_graph_structure(self, transformer):
        """Test graph structure analysis."""
        root = transformer._root
        ids_data = transformer._extract_ids_data(root)

        graph_data = transformer._analyze_graph_structure(ids_data)

        assert isinstance(graph_data, dict)


class TestCatalogGeneration:
    """Tests for catalog generation."""

    def test_generate_catalog(self, transformer):
        """Test catalog file generation."""
        root = transformer._root
        ids_data = transformer._extract_ids_data(root)
        graph_data = transformer._analyze_graph_structure(ids_data)

        catalog_path = transformer._generate_catalog(ids_data, graph_data)

        assert catalog_path.exists()
        assert catalog_path.name == "ids_catalog.json"

        # Verify catalog content
        with open(catalog_path) as f:
            catalog = json.load(f)

        assert "metadata" in catalog
        assert "ids_catalog" in catalog


class TestDetailedFileGeneration:
    """Tests for detailed IDS file generation."""

    def test_generate_detailed_files(self, transformer):
        """Test detailed file generation."""
        root = transformer._root
        ids_data = transformer._extract_ids_data(root)

        detailed_paths = transformer._generate_detailed_files(ids_data)

        assert len(detailed_paths) > 0

        for path in detailed_paths:
            assert path.exists()
            assert path.suffix == ".json"

            # Verify file content
            with open(path) as f:
                detailed = json.load(f)

            assert "ids_info" in detailed
            assert "paths" in detailed


class TestIdentifierCatalogGeneration:
    """Tests for identifier catalog generation."""

    def test_generate_identifier_catalog(self, transformer):
        """Test identifier catalog generation."""
        root = transformer._root
        ids_data = transformer._extract_ids_data(root)

        catalog_path = transformer._generate_identifier_catalog(ids_data)

        assert catalog_path.exists()
        assert catalog_path.name == "identifier_catalog.json"

        with open(catalog_path) as f:
            catalog = json.load(f)

        assert "metadata" in catalog
        assert "schemas" in catalog
        assert "paths_by_ids" in catalog
        assert "branching_analytics" in catalog

    def test_extract_schema_name(self, transformer):
        """Test schema name extraction."""
        name = transformer._extract_schema_name(
            "equilibrium/equilibrium_profiles_2d_identifier.xml"
        )
        assert "Equilibrium Profiles 2D" in name

        name = transformer._extract_schema_name("")
        assert name == "unknown"


class TestBuild:
    """Tests for full build process."""

    def test_build_returns_transformation_outputs(self, transformer):
        """Test build method returns TransformationOutputs."""
        outputs = transformer.build()

        assert outputs.catalog is not None
        assert outputs.catalog.exists()
        assert len(outputs.detailed) > 0
        assert outputs.identifier_catalog is not None

    def test_build_creates_all_expected_files(self, transformer):
        """Test build creates expected directory structure."""
        outputs = transformer.build()

        # Check catalog
        assert outputs.catalog.exists()

        # Check detailed directory
        detailed_dir = transformer.resolved_output_dir / "detailed"
        assert detailed_dir.exists()

        # Check identifier catalog
        assert outputs.identifier_catalog.exists()

    def test_build_with_none_root_raises(self, tmp_path, mock_dd_accessor):
        """Test build raises when root is None."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        transformer = DataDictionaryTransformer(
            dd_version="4.0.0",
            output_dir=output_dir,
            dd_accessor=mock_dd_accessor,
            use_rich=False,
        )

        # Force root to None
        transformer._root = None

        with pytest.raises(ValueError, match="XML root is None"):
            transformer.build()


class TestStructureAnalysis:
    """Tests for structure analysis generation."""

    def test_generate_structure_analysis(self, transformer):
        """Test structure analysis generation."""
        root = transformer._root
        ids_data = transformer._extract_ids_data(root)

        # Should not raise
        transformer._generate_structure_analysis(ids_data)

    def test_generate_structure_analysis_handles_error(self, transformer, caplog):
        """Test structure analysis handles errors gracefully."""
        with patch("imas_mcp.core.xml_parser.StructureAnalyzer") as mock_analyzer_class:
            mock_analyzer_class.side_effect = Exception("Test error")

            root = transformer._root
            ids_data = transformer._extract_ids_data(root)

            # Should not raise, but log error
            transformer._generate_structure_analysis(ids_data)

            # Check error was logged
            assert "Failed to generate structure analysis" in caplog.text


class TestExclusionChecker:
    """Tests for exclusion checking."""

    def test_exclusion_checker_initialized(self, transformer):
        """Test exclusion checker is initialized."""
        assert transformer._exclusion_checker is not None

    def test_default_excluded_patterns(self, transformer):
        """Test default excluded patterns."""
        assert "ids_properties" in transformer.excluded_patterns
        assert "code" in transformer.excluded_patterns
