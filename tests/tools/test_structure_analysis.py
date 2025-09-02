"""
Test enhanced IDS structure analysis functionality.

Tests the analyze_ids_structure tool enhancement with pre-generated static data.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_mcp.models.result_models import StructureResult
from imas_mcp.models.structure_models import (
    DomainDistribution,
    HierarchyMetrics,
    NavigationHints,
    StructureAnalysis,
)
from imas_mcp.tools.analysis_tool import AnalysisTool


class TestStructureAnalysis:
    """Test enhanced structure analysis functionality."""

    @pytest.fixture
    def analysis_tool(self):
        """Create analysis tool with mocked services."""
        tool = AnalysisTool()

        # Mock the services
        tool.documents = MagicMock()
        tool.physics = AsyncMock()
        tool.response = MagicMock()

        return tool

    @pytest.fixture
    def mock_structure_analysis(self):
        """Mock structure analysis data."""
        return StructureAnalysis(
            hierarchy_metrics=HierarchyMetrics(
                total_nodes=150,
                leaf_nodes=85,
                max_depth=6,
                branching_factor=3.2,
                complexity_score=0.65,
            ),
            domain_distribution=[
                DomainDistribution(
                    domain="transport",
                    node_count=45,
                    percentage=30.0,
                    key_paths=["core_profiles/profiles_1d", "transport/model"],
                ),
                DomainDistribution(
                    domain="equilibrium",
                    node_count=35,
                    percentage=23.3,
                    key_paths=["equilibrium/time_slice", "equilibrium/global"],
                ),
            ],
            navigation_hints=NavigationHints(
                entry_points=["core_profiles", "profiles_1d", "time_slice"],
                common_patterns=["Time series data structure", "Profile organization"],
                drill_down_suggestions=[
                    "Explore profiles_1d structure",
                    "Examine time_slice data",
                ],
            ),
            complexity_summary="Moderate complexity structure with 150 nodes, 85 data endpoints, and 6 levels deep. Average branching factor: 3.2",
            organization_pattern="Balanced organization with temporal structure",
        )

    @pytest.fixture
    def mock_documents(self):
        """Mock document data."""
        docs = []
        for i in range(10):
            doc = MagicMock()
            doc.metadata.path_name = f"core_profiles/profiles_1d/path_{i}"
            doc.metadata.physics_domain = "transport" if i < 5 else "equilibrium"
            doc.documentation = f"Documentation for path {i}"
            docs.append(doc)
        return docs

    async def test_analyze_with_enhanced_structure(
        self, analysis_tool, mock_structure_analysis, mock_documents
    ):
        """Test analysis with pre-generated structure data."""
        # Mock services
        analysis_tool.documents.validate_ids = AsyncMock(
            return_value=(["core_profiles"], [])
        )
        analysis_tool.documents.get_documents_safe = AsyncMock(
            return_value=mock_documents
        )
        analysis_tool.physics.enhance_query = AsyncMock(
            return_value={"domain": "transport"}
        )

        # Mock structure analysis loading
        with (
            patch.object(
                analysis_tool,
                "_load_structure_analysis",
                return_value=mock_structure_analysis,
            ),
            patch.object(
                analysis_tool,
                "_perform_graph_analysis",
                return_value={"summary": "mock graph analysis", "metrics": {}},
            ),
        ):
            result = await analysis_tool.analyze_ids_structure("core_profiles")

        # Verify result
        assert isinstance(result, StructureResult)
        assert result.ids_name == "core_profiles"
        assert result.analysis == mock_structure_analysis
        assert result.max_depth == 6
        assert "Moderate complexity" in result.description

        # Verify structure dictionary uses the enhanced analysis data
        assert result.structure["total_nodes"] == 150  # From mock_structure_analysis
        assert (
            result.structure["complexity_score"] == 65
        )  # Converted to int (0.65 * 100)
        assert len(result.structure) >= 5  # Should have multiple structure metrics
        assert result.structure["physics_domains"] == 2  # Count of domains

    async def test_analyze_fallback_to_basic(self, analysis_tool, mock_documents):
        """Test fallback to basic analysis when no pre-generated data."""
        # Mock services
        analysis_tool.documents.validate_ids = AsyncMock(
            return_value=(["test_ids"], [])
        )
        analysis_tool.documents.get_documents_safe = AsyncMock(
            return_value=mock_documents
        )
        analysis_tool.physics.enhance_query = AsyncMock(
            return_value={"domain": "general"}
        )

        # Mock structure analysis loading to return None (no pre-generated data)
        with patch.object(analysis_tool, "_load_structure_analysis", return_value=None):
            result = await analysis_tool.analyze_ids_structure("test_ids")

        # Verify result uses fallback
        assert isinstance(result, StructureResult)
        assert result.ids_name == "test_ids"
        assert result.analysis is None  # No enhanced analysis
        assert "Real-time structural analysis" in result.description
        assert len(result.sample_paths) > 0  # Should have some sample paths

    async def test_analyze_ids_not_found(self, analysis_tool):
        """Test analysis with non-existent IDS."""
        # Mock IDS validation to fail
        error_response = MagicMock()
        analysis_tool.documents.validate_ids = AsyncMock(
            return_value=([], ["invalid_ids"])
        )
        analysis_tool.documents.create_ids_not_found_error = MagicMock(
            return_value=error_response
        )

        result = await analysis_tool.analyze_ids_structure("invalid_ids")

        assert result == error_response
        analysis_tool.documents.create_ids_not_found_error.assert_called_once()

    async def test_load_structure_analysis(
        self, analysis_tool, mock_structure_analysis
    ):
        """Test loading structure analysis from static files."""
        # Mock document store
        mock_store = MagicMock()
        mock_store._data_dir = "/mock/data/dir"
        analysis_tool.documents.store = mock_store

        # Mock StructureAnalyzer
        with patch(
            "imas_mcp.tools.analysis_tool.StructureAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.load_structure_analysis.return_value = mock_structure_analysis
            mock_analyzer_class.return_value = mock_analyzer

            result = await analysis_tool._load_structure_analysis("test_ids")

            assert result == mock_structure_analysis
            mock_analyzer_class.assert_called_once_with("/mock/data/dir")
            mock_analyzer.load_structure_analysis.assert_called_once_with("test_ids")

    async def test_load_structure_analysis_error(self, analysis_tool):
        """Test error handling in structure analysis loading."""
        # Mock document store to raise exception
        mock_store = MagicMock()
        mock_store._data_dir = "/mock/data/dir"
        analysis_tool.documents.store = mock_store

        # Mock StructureAnalyzer to raise exception
        with patch(
            "imas_mcp.tools.analysis_tool.StructureAnalyzer",
            side_effect=Exception("File not found"),
        ):
            result = await analysis_tool._load_structure_analysis("test_ids")

            assert result is None

    def test_prompt_building(self, analysis_tool):
        """Test AI prompt building for structure analysis."""
        prompt = analysis_tool.build_prompt(
            "structure_analysis", {"query": "core_profiles"}
        )

        assert "core_profiles" in prompt
        assert "comprehensive structural analysis" in prompt
        assert "Architecture Overview" in prompt
        assert "Physics Context" in prompt

    def test_prompt_building_simple(self, analysis_tool):
        """Test simple prompt building."""
        prompt = analysis_tool._build_structure_analysis_prompt_simple("equilibrium")

        assert "equilibrium" in prompt
        assert "IMAS IDS Structure Analysis" in prompt
        assert "Data Hierarchy" in prompt
        assert "Usage Patterns" in prompt


class TestStructureAnalysisIntegration:
    """Integration tests for structure analysis."""

    @pytest.mark.integration
    async def test_full_analysis_workflow(self):
        """Test full workflow with real data structure."""
        # This would test with actual static files if available
        tool = AnalysisTool()

        # Mock realistic document store
        with patch.object(
            tool.documents, "validate_ids", return_value=(["core_profiles"], [])
        ):
            with patch.object(tool.documents, "get_documents_safe", return_value=[]):
                with patch.object(tool.physics, "enhance_query", return_value={}):
                    result = await tool.analyze_ids_structure("core_profiles")

                    assert isinstance(result, StructureResult)
                    assert result.ids_name == "core_profiles"

    def test_structure_metrics_calculation(self):
        """Test structure metrics calculation logic."""
        # Test the basic structure analysis fallback
        tool = AnalysisTool()

        # Mock document data
        mock_docs = []
        for i in range(20):
            doc = MagicMock()
            doc.metadata.path_name = f"test/level1/level2/path_{i}"
            doc.raw_data = {
                "identifier_schema": {
                    "options": [{"name": f"opt_{j}"} for j in range(3)]
                }
            }
            mock_docs.append(doc)

        structure = tool._analyze_structure(mock_docs)

        assert structure["document_count"] == 20
        assert structure["max_depth"] == 4  # test/level1/level2/path_N
        assert structure["identifier_nodes"] == 20
        assert structure["branching_complexity"] == 60  # 20 nodes * 3 options each
