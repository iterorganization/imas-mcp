"""Tests for xml_utils.py - XML processing utilities."""

import xml.etree.ElementTree as ET

import pytest

from imas_mcp.core.xml_utils import DocumentationBuilder, XmlTreeUtils


class TestDocumentationBuilderNormalizePunctuation:
    """Tests for _normalize_sentence_punctuation method."""

    def test_adds_period_when_missing(self):
        """Test that period is added when missing."""
        result = DocumentationBuilder._normalize_sentence_punctuation("Hello world")
        assert result == "Hello world."

    def test_preserves_existing_period(self):
        """Test that existing period is preserved."""
        result = DocumentationBuilder._normalize_sentence_punctuation("Hello world.")
        assert result == "Hello world."

    def test_preserves_exclamation_mark(self):
        """Test that exclamation mark is preserved."""
        result = DocumentationBuilder._normalize_sentence_punctuation("Important!")
        assert result == "Important!"

    def test_preserves_question_mark(self):
        """Test that question mark is preserved."""
        result = DocumentationBuilder._normalize_sentence_punctuation(
            "Is this correct?"
        )
        assert result == "Is this correct?"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        result = DocumentationBuilder._normalize_sentence_punctuation("  Hello  ")
        assert result == "Hello."

    def test_empty_string(self):
        """Test handling of empty string."""
        result = DocumentationBuilder._normalize_sentence_punctuation("")
        assert result == ""


class TestDocumentationBuilderHierarchical:
    """Tests for build_hierarchical_documentation method."""

    def test_empty_parts(self):
        """Test with empty documentation parts."""
        result = DocumentationBuilder.build_hierarchical_documentation({})
        assert result == ""

    def test_single_leaf_doc(self):
        """Test with single leaf documentation."""
        parts = {"ids/path/leaf": "Leaf description"}
        result = DocumentationBuilder.build_hierarchical_documentation(parts)
        assert "Leaf description" in result

    def test_hierarchical_docs(self):
        """Test with hierarchical documentation."""
        parts = {
            "equilibrium": "IDS for plasma equilibrium",
            "equilibrium/time_slice": "Time-dependent data",
            "equilibrium/time_slice/boundary": "Plasma boundary data",
        }
        result = DocumentationBuilder.build_hierarchical_documentation(parts)

        # Should contain leaf documentation first
        assert "Plasma boundary data" in result
        # Should contain context from parents
        assert "equilibrium" in result.lower() or "time" in result.lower()

    def test_deepest_path_is_primary(self):
        """Test that deepest path documentation comes first."""
        parts = {
            "ids": "Root level",
            "ids/a": "Level 1",
            "ids/a/b": "Level 2",
            "ids/a/b/c": "This is the leaf",
        }
        result = DocumentationBuilder.build_hierarchical_documentation(parts)

        # Leaf should be at the start
        assert result.startswith("This is the leaf")


class TestDocumentationBuilderCollectHierarchy:
    """Tests for collect_documentation_hierarchy method."""

    @pytest.fixture
    def sample_tree(self):
        """Create a sample XML tree for testing."""
        root = ET.fromstring("""
        <root>
            <IDS name="equilibrium" documentation="Equilibrium IDS documentation">
                <field name="time_slice" documentation="Time slice container">
                    <field name="boundary" documentation="Boundary data">
                        <field name="psi" documentation="Poloidal flux"/>
                    </field>
                </field>
            </IDS>
        </root>
        """)
        return root

    @pytest.fixture
    def parent_map(self, sample_tree):
        """Build parent map for sample tree."""
        return XmlTreeUtils.build_parent_map(sample_tree)

    def test_collects_all_levels(self, sample_tree, parent_map):
        """Test collecting documentation from all hierarchy levels."""
        ids_elem = sample_tree.find(".//IDS[@name='equilibrium']")
        psi_elem = ids_elem.find(".//field[@name='psi']")

        result = DocumentationBuilder.collect_documentation_hierarchy(
            psi_elem, ids_elem, "equilibrium", parent_map
        )

        # Should have documentation for IDS and all parent levels
        assert "equilibrium" in result
        assert any("time_slice" in key for key in result.keys())
        assert any("boundary" in key for key in result.keys())
        assert any("psi" in key for key in result.keys())

    def test_includes_ids_documentation(self, sample_tree, parent_map):
        """Test that IDS documentation is included."""
        ids_elem = sample_tree.find(".//IDS[@name='equilibrium']")
        boundary_elem = ids_elem.find(".//field[@name='boundary']")

        result = DocumentationBuilder.collect_documentation_hierarchy(
            boundary_elem, ids_elem, "equilibrium", parent_map
        )

        assert "equilibrium" in result
        assert "Equilibrium IDS documentation" in result["equilibrium"]


class TestXmlTreeUtils:
    """Tests for XmlTreeUtils class."""

    @pytest.fixture
    def sample_tree(self):
        """Create a sample XML tree for testing."""
        root = ET.fromstring("""
        <root>
            <IDS name="test_ids">
                <field name="level1">
                    <field name="level2">
                        <field name="leaf"/>
                    </field>
                </field>
            </IDS>
        </root>
        """)
        return root

    def test_build_parent_map(self, sample_tree):
        """Test building parent map."""
        parent_map = XmlTreeUtils.build_parent_map(sample_tree)

        # Every non-root element should have a parent
        for elem in sample_tree.iter():
            for child in elem:
                assert child in parent_map
                assert parent_map[child] == elem

    def test_build_element_path(self, sample_tree):
        """Test building element path."""
        parent_map = XmlTreeUtils.build_parent_map(sample_tree)
        ids_elem = sample_tree.find(".//IDS[@name='test_ids']")
        leaf_elem = ids_elem.find(".//field[@name='leaf']")

        path = XmlTreeUtils.build_element_path(
            leaf_elem, ids_elem, "test_ids", parent_map
        )

        assert path == "test_ids/level1/level2/leaf"

    def test_build_element_path_empty(self, sample_tree):
        """Test building path for element without name."""
        parent_map = XmlTreeUtils.build_parent_map(sample_tree)
        ids_elem = sample_tree.find(".//IDS[@name='test_ids']")

        # Create element without name
        unnamed = ET.SubElement(ids_elem, "unnamed")

        path = XmlTreeUtils.build_element_path(
            unnamed, ids_elem, "test_ids", parent_map
        )

        assert path is None

    def test_build_element_path_direct_child(self, sample_tree):
        """Test building path for direct child of IDS."""
        parent_map = XmlTreeUtils.build_parent_map(sample_tree)
        ids_elem = sample_tree.find(".//IDS[@name='test_ids']")
        level1_elem = ids_elem.find("field[@name='level1']")

        path = XmlTreeUtils.build_element_path(
            level1_elem, ids_elem, "test_ids", parent_map
        )

        assert path == "test_ids/level1"
