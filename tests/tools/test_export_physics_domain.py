"""
Tests for enhanced export_physics_domain tool functionality.

This module tests the Phase 2 improvements to the export_physics_domain tool,
including domain-specific analysis, theoretical foundations, experimental methods,
and cross-domain relationships.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import DomainExport, SearchResult
from imas_mcp.tools.export_tool import ExportTool


class TestExportPhysicsDomain:
    """Test suite for enhanced export_physics_domain functionality."""

    @pytest.fixture
    def export_tool(self):
        """Create export tool instance for testing."""
        tool = ExportTool()

        # Mock the required services
        tool.documents = AsyncMock()
        tool.physics = AsyncMock()
        tool.response = AsyncMock()

        return tool

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results for testing."""
        return [
            MagicMock(
                path="equilibrium/profiles_1d/pressure",
                documentation="Pressure profile data for equilibrium analysis",
                physics_domain="equilibrium",
                data_type="float_1d",
                units="Pa",
                ids_name="equilibrium",
            ),
            MagicMock(
                path="equilibrium/profiles_1d/psi",
                documentation="Poloidal flux surface coordinate",
                physics_domain="equilibrium",
                data_type="float_1d",
                units="Wb",
                ids_name="equilibrium",
            ),
            MagicMock(
                path="transport/model/diffusion_coeff",
                documentation="Transport diffusion coefficients",
                physics_domain="transport",
                data_type="float_1d",
                units="m^2.s^-1",
                ids_name="transport",
            ),
        ]

    @pytest.fixture
    def mock_domain_analysis(self):
        """Create mock domain analysis for testing."""
        return {
            "key_measurements": [
                {
                    "measurement_type": "pressure_measurement",
                    "paths": ["equilibrium/profiles_1d/pressure"],
                    "typical_units": ["Pa"],
                    "description": "Pressure measurements in equilibrium physics context",
                    "data_count": 1,
                }
            ],
            "theoretical_foundations": {
                "primary_phenomena": ["magnetic flux surfaces", "pressure balance"],
                "complexity_level": "intermediate",
                "description": "Magnetohydrodynamic equilibrium and magnetic field configuration",
                "theoretical_context": "Magnetohydrodynamic equilibrium theory and force balance",
            },
            "experimental_methods": [
                {
                    "method": "magnetic diagnostics",
                    "description": "Magnetic coil measurements for equilibrium reconstruction",
                    "typical_outputs": [
                        "magnetic_field",
                        "plasma_current",
                        "equilibrium",
                    ],
                    "applicability": "essential",
                }
            ],
            "cross_domain_links": [
                {
                    "target_domain": "transport",
                    "relationship_type": "causal",
                    "physics_connection": "Equilibrium profiles drive transport gradients",
                    "shared_measurements": ["magnetic diagnostics"],
                }
            ],
            "typical_workflows": [
                {
                    "workflow_name": "equilibrium_reconstruction",
                    "description": "Reconstruct MHD equilibrium from magnetic measurements",
                    "typical_steps": [
                        "magnetic_diagnostics_analysis",
                        "pressure_profile_fitting",
                        "current_profile_optimization",
                        "equilibrium_validation",
                    ],
                    "data_requirements": [
                        "magnetic_field",
                        "pressure_profile",
                        "current_density",
                    ],
                }
            ],
            "data_characteristics": {
                "total_paths": 3,
                "common_units": [("Pa", 1), ("Wb", 1), ("m^2.s^-1", 1)],
                "data_type_distribution": {"float_1d": 3},
                "documentation_quality": "high",
                "complexity_indicators": ["deep_hierarchy"],
            },
            "complexity_assessment": {
                "theoretical_complexity": "intermediate",
                "data_complexity": "intermediate",
                "complexity_factors": ["deep_hierarchy"],
                "recommended_approach": "guided_analysis_workflow",
            },
        }

    async def test_export_physics_domain_basic_functionality(
        self, export_tool, mock_search_results, mock_domain_analysis
    ):
        """Test basic functionality of enhanced export_physics_domain."""
        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value=mock_domain_analysis
        )

        # Execute test
        result = await export_tool.export_physics_domain(
            domain="equilibrium", analysis_depth="focused", max_paths=10
        )

        # Verify result
        assert isinstance(result, DomainExport)
        assert result.domain == "equilibrium"
        assert len(result.domain_info["paths"]) == 3
        assert "key_measurements" in result.domain_info
        assert "theoretical_foundations" in result.domain_info
        assert "experimental_methods" in result.domain_info

        # Verify search was called correctly
        export_tool.execute_search.assert_called_once_with(
            query="equilibrium", search_mode=SearchMode.SEMANTIC, max_results=10
        )

        # Verify domain analyzer was called
        export_tool._domain_analyzer.analyze_domain.assert_called_once_with(
            domain="equilibrium", search_results=mock_search_results, depth="focused"
        )

    async def test_export_physics_domain_comprehensive_analysis(
        self, export_tool, mock_search_results, mock_domain_analysis
    ):
        """Test comprehensive analysis depth functionality."""
        # Add comprehensive analysis features to mock
        comprehensive_analysis = mock_domain_analysis.copy()
        comprehensive_analysis.update(
            {
                "detailed_physics_context": {
                    "fundamental_equations": [
                        "Grad-Shafranov equation",
                        "force balance equation",
                    ],
                    "key_physics_scales": {
                        "spatial": "minor radius",
                        "temporal": "resistive diffusion",
                    },
                    "governing_parameters": [
                        "beta",
                        "safety factor",
                        "pressure gradient",
                    ],
                    "typical_regimes": ["low beta", "high beta", "advanced scenarios"],
                },
                "research_applications": [
                    {
                        "application": "disruption_prediction",
                        "description": "Predict plasma disruptions",
                    },
                    {
                        "application": "scenario_optimization",
                        "description": "Optimize plasma scenarios",
                    },
                ],
                "data_quality_assessment": {
                    "quality": "high",
                    "documentation_coverage": "85.0%",
                    "units_coverage": "100.0%",
                    "total_paths": 3,
                    "recommendations": [
                        "Data suitable for detailed analysis",
                        "Consider advanced workflows",
                    ],
                },
            }
        )

        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value=comprehensive_analysis
        )

        # Execute test with comprehensive depth
        result = await export_tool.export_physics_domain(
            domain="equilibrium", analysis_depth="comprehensive", max_paths=15
        )

        # Verify comprehensive features are included
        assert isinstance(result, DomainExport)
        assert result.domain_info["analysis_depth"] == "comprehensive"
        assert "detailed_physics_context" in result.domain_info
        assert "research_applications" in result.domain_info
        assert "data_quality_assessment" in result.domain_info

        # Verify enhanced path information includes longer documentation
        path_info = result.domain_info["paths"][0]
        assert len(path_info["documentation"]) <= 300  # comprehensive mode limit
        assert "measurement_type" in path_info

    async def test_export_physics_domain_cross_domain_analysis(
        self, export_tool, mock_search_results, mock_domain_analysis
    ):
        """Test cross-domain analysis functionality."""
        # Add cross-domain features to mock
        cross_domain_analysis = mock_domain_analysis.copy()
        cross_domain_analysis.update(
            {
                "measurement_integration": {
                    "measurement_groups": {"pressure": [mock_search_results[0]]},
                    "integration_possibilities": ["pressure_consistency_check"],
                    "validation_strategies": [
                        "cross_measurement_validation",
                        "temporal_consistency_check",
                    ],
                }
            }
        )

        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value=cross_domain_analysis
        )

        # Execute test with cross-domain analysis
        result = await export_tool.export_physics_domain(
            domain="equilibrium", include_cross_domain=True, analysis_depth="focused"
        )

        # Verify cross-domain features are included
        assert isinstance(result, DomainExport)
        assert result.include_cross_domain is True
        assert "cross_domain_links" in result.domain_info
        assert "measurement_integration" in result.domain_info

        # Verify enhancement features metadata
        enhancement_features = result.metadata["enhancement_features"]
        assert "cross_domain_analysis" in enhancement_features
        assert "theoretical_foundations" in enhancement_features

    async def test_export_physics_domain_measurement_classification(
        self, export_tool, mock_search_results
    ):
        """Test measurement type classification functionality."""
        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value={
                "key_measurements": [],
                "theoretical_foundations": {},
                "experimental_methods": [],
                "typical_workflows": [],
                "data_characteristics": {},
                "complexity_assessment": {},
            }
        )

        # Execute test
        result = await export_tool.export_physics_domain(domain="equilibrium")

        # Verify measurement classification
        paths = result.domain_info["paths"]
        assert paths[0]["measurement_type"] == "pressure_measurement"  # pressure path
        assert "measurement_type" in paths[1]  # psi path
        assert (
            paths[2]["measurement_type"] == "general_measurement"
        )  # diffusion_coeff (doesn't match specific patterns)

    async def test_export_physics_domain_no_results(self, export_tool):
        """Test handling when no search results are found."""
        # Setup mock for empty results
        mock_search_result = MagicMock()
        mock_search_result.hits = []
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)

        # Execute test
        result = await export_tool.export_physics_domain(domain="nonexistent_domain")

        # Verify error handling
        assert isinstance(result, ToolError)
        assert "No data found for domain 'nonexistent_domain'" in result.error

    async def test_export_physics_domain_empty_domain(self, export_tool):
        """Test handling of empty domain parameter."""
        # Execute test
        result = await export_tool.export_physics_domain(domain="")

        # Verify error handling (validation decorator catches this)
        assert isinstance(result, ToolError)
        assert "String should have at least 1 character" in result.error

    async def test_export_physics_domain_max_paths_limit(
        self, export_tool, mock_search_results, mock_domain_analysis
    ):
        """Test max_paths limitation functionality."""
        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value=mock_domain_analysis
        )

        # Execute test with high max_paths
        result = await export_tool.export_physics_domain(
            domain="equilibrium",
            max_paths=100,  # Should be limited to 50
        )

        # Verify max_paths was limited
        assert isinstance(result, DomainExport)
        assert result.max_paths == 50

        # Verify search was called with limited max_results
        export_tool.execute_search.assert_called_once_with(
            query="equilibrium", search_mode=SearchMode.SEMANTIC, max_results=50
        )

    async def test_measurement_type_classification_edge_cases(self, export_tool):
        """Test edge cases in measurement type classification."""
        # Test different path patterns
        test_cases = [
            ("equilibrium/density_profile/electron", "density_measurement"),
            ("heating/temperature/ion", "temperature_measurement"),
            ("diagnostics/magnetic_field/bt", "magnetic_field_measurement"),
            ("core_profiles/current/parallel", "current_measurement"),
            ("edge/velocity/toroidal", "velocity_measurement"),
            ("radiation/emission/line", "radiation_measurement"),
            ("structure/position/limiter", "geometric_measurement"),
            ("some/unknown/path", "general_measurement"),
        ]

        for path, expected_type in test_cases:
            mock_result = MagicMock()
            mock_result.path = path
            mock_result.data_type = "float_1d"

            # Test classification
            classified_type = export_tool._classify_measurement_type(mock_result)
            assert classified_type == expected_type, (
                f"Path {path} should be classified as {expected_type}, got {classified_type}"
            )

    async def test_export_physics_domain_metadata_completeness(
        self, export_tool, mock_search_results, mock_domain_analysis
    ):
        """Test completeness of metadata in export results."""
        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value=mock_domain_analysis
        )

        # Execute test
        result = await export_tool.export_physics_domain(
            domain="equilibrium",
            include_cross_domain=True,
            analysis_depth="comprehensive",
        )

        # Verify metadata completeness
        metadata = result.metadata
        assert "total_found" in metadata
        assert "analysis_timestamp" in metadata
        assert "analysis_engine" in metadata
        assert metadata["analysis_engine"] == "PhysicsDomainAnalyzer"

        # Verify enhancement features
        enhancement_features = metadata["enhancement_features"]
        expected_features = [
            "theoretical_foundations",
            "experimental_methods",
            "measurement_classification",
            "workflow_analysis",
            "complexity_assessment",
            "cross_domain_analysis",
        ]
        for feature in expected_features:
            assert feature in enhancement_features

    async def test_domain_analyzer_initialization(self, export_tool):
        """Test that domain analyzer is properly initialized."""
        # Verify domain analyzer exists and is properly initialized
        assert hasattr(export_tool, "_domain_analyzer")
        assert export_tool._domain_analyzer is not None

        # Verify analyzer has required methods
        assert hasattr(export_tool._domain_analyzer, "analyze_domain")
        assert callable(export_tool._domain_analyzer.analyze_domain)

    @pytest.mark.parametrize("analysis_depth", ["overview", "focused", "comprehensive"])
    async def test_export_physics_domain_analysis_depths(
        self, export_tool, mock_search_results, mock_domain_analysis, analysis_depth
    ):
        """Test all analysis depth options."""
        # Setup mocks
        mock_search_result = MagicMock()
        mock_search_result.hits = mock_search_results
        export_tool.execute_search = AsyncMock(return_value=mock_search_result)
        export_tool._domain_analyzer.analyze_domain = MagicMock(
            return_value=mock_domain_analysis
        )

        # Execute test
        result = await export_tool.export_physics_domain(
            domain="equilibrium", analysis_depth=analysis_depth
        )

        # Verify result
        assert isinstance(result, DomainExport)
        assert result.domain_info["analysis_depth"] == analysis_depth

        # Verify analyzer was called with correct depth
        export_tool._domain_analyzer.analyze_domain.assert_called_once_with(
            domain="equilibrium",
            search_results=mock_search_results,
            depth=analysis_depth,
        )

    async def test_export_physics_domain_error_handling(self, export_tool):
        """Test error handling in export_physics_domain."""
        # Setup mock to raise exception
        export_tool.execute_search = AsyncMock(
            side_effect=Exception("Search service error")
        )

        # Execute test
        result = await export_tool.export_physics_domain(domain="equilibrium")

        # Verify error handling
        assert isinstance(result, ToolError)
        assert "Domain export failed" in result.error
        assert "Search service error" in result.error
