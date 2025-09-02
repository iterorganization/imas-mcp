"""
Test export and data access features through user interface.

This module tests export functionality as user-facing features,
focusing on IDS export, domain export, and data formatting.
"""

import json
import time

import pytest

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import (
    DomainExport,
    ExportData,
    IDSExport,
)


class TestExportFeatures:
    """Test export and data access functionality."""

    @pytest.mark.asyncio
    async def test_ids_export_basic(self, tools, mcp_test_context):
        """Test basic IDS export functionality."""
        ids_name = mcp_test_context["test_ids"]
        # export_ids expects ids_list (a list), not ids_name
        result = await tools.export_ids(ids_list=[ids_name])
        assert isinstance(result, IDSExport)

        assert hasattr(result, "ids_names")
        assert ids_name in result.ids_names

    @pytest.mark.asyncio
    async def test_ids_export_data_structure(self, tools, mcp_test_context):
        """Test IDS export data has proper structure."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.export_ids(ids_list=[ids_name])

        assert isinstance(result, IDSExport)
        # Export should have data field
        assert hasattr(result, "data")

        # Export data should be structured
        export_data = result.data
        assert export_data is not None

    @pytest.mark.asyncio
    async def test_physics_domain_export_basic(self, tools):
        """Test basic physics domain export functionality."""
        result = await tools.export_physics_domain(domain="transport")

        assert isinstance(result, DomainExport)

        assert result.domain == "transport"
        assert hasattr(result, "data")
        assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_physics_domain_export_data_structure(self, tools):
        """Test physics domain export data structure."""
        result = await tools.export_physics_domain(domain="transport")

        assert isinstance(result, DomainExport)
        assert isinstance(result.data, ExportData)

    @pytest.mark.asyncio
    async def test_multiple_domain_exports(self, tools):
        """Test export works for multiple physics domains."""
        domains = ["transport", "equilibrium", "core"]

        for domain in domains:
            result = await tools.export_physics_domain(domain=domain)
            assert isinstance(result, DomainExport)
            assert result.domain == domain

    @pytest.mark.asyncio
    async def test_export_metadata_inclusion(self, tools, mcp_test_context):
        """Test export includes relevant metadata."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.export_ids(ids_list=[ids_name])

        assert isinstance(result, IDSExport)
        # Should include export metadata
        assert hasattr(result, "ids_names")
        assert hasattr(result, "data")
        assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_export_format_consistency(self, tools, mcp_test_context):
        """Test export format is consistent across different exports."""
        ids_name = mcp_test_context["test_ids"]

        # Test IDS export format
        ids_result = await tools.export_ids(ids_list=[ids_name])

        # Test domain export format
        domain_result = await tools.export_physics_domain(domain="transport")

        # Both should have consistent top-level structure
        assert isinstance(ids_result, IDSExport)
        assert isinstance(domain_result, DomainExport)

        # Both should have data field
        assert hasattr(ids_result, "data")
        assert hasattr(domain_result, "data")


class TestExportErrorHandling:
    """Test export error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_export_invalid_ids_name(self, tools):
        """Test export handles invalid IDS names gracefully."""
        invalid_ids = ["nonexistent_ids_name"]
        result = await tools.export_ids(ids_list=invalid_ids)
        # The tool returns an ToolError for invalid IDS names
        assert isinstance(result, ToolError)
        assert "not found" in result.error
        assert result.suggestions  # Should provide suggestions
        assert (
            "available_ids" in result.context
        )  # Should include available alternatives

    @pytest.mark.asyncio
    async def test_export_invalid_domain(self, tools):
        """Test export handles invalid physics domains gracefully."""
        invalid_domain = "nonexistent_domain"
        result = await tools.export_physics_domain(domain=invalid_domain)

        assert isinstance(result, DomainExport)
        # Even invalid domains should return a DomainExport object
        assert result.domain == invalid_domain

    @pytest.mark.asyncio
    async def test_export_empty_parameters(self, tools):
        """Test export handles empty parameters gracefully."""
        # Test empty IDS list - should return ToolError due to validation
        result = await tools.export_ids(ids_list=[])
        assert isinstance(result, ToolError)

        # Test empty domain - should return ToolError due to validation
        result = await tools.export_physics_domain(domain="")
        assert isinstance(result, ToolError)


class TestExportDataQuality:
    """Test export data quality and completeness."""

    @pytest.mark.asyncio
    async def test_export_data_completeness(self, tools, mcp_test_context):
        """Test exported data is complete and usable."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.export_ids(ids_list=[ids_name])

        assert isinstance(result, IDSExport)
        export_data = result.data

        # Export data should be substantial
        assert export_data is not None

    @pytest.mark.asyncio
    async def test_export_data_validity(self, tools):
        """Test exported data is valid and properly formatted."""

        result = await tools.export_physics_domain(domain="transport")

        assert isinstance(result, DomainExport)
        export_data = result.data

        # Should be valid JSON-serializable data
        try:
            json.dumps(
                export_data, default=str
            )  # Allow string conversion for non-serializable types
        except (TypeError, ValueError):
            pytest.fail("Export data should be JSON-serializable")

    @pytest.mark.asyncio
    async def test_export_consistency(self, tools, mcp_test_context):
        """Test export results are consistent across calls."""
        ids_name = mcp_test_context["test_ids"]

        # Call export twice
        result1 = await tools.export_ids(ids_list=[ids_name])
        result2 = await tools.export_ids(ids_list=[ids_name])

        # Both should be successful results
        assert isinstance(result1, IDSExport)
        assert isinstance(result2, IDSExport)

        # Core information should be consistent
        assert result1.ids_names == result2.ids_names

        # Export data structure should be consistent
        assert isinstance(result1.data, type(result2.data))


class TestExportPerformance:
    """Test export performance characteristics."""

    @pytest.mark.asyncio
    async def test_export_response_time(self, tools, mcp_test_context):
        """Test export responds in reasonable time."""
        ids_name = mcp_test_context["test_ids"]

        start_time = time.time()
        result = await tools.export_ids(ids_list=[ids_name])
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 10.0, f"Export took {execution_time:.2f}s, too slow"
        assert isinstance(result, IDSExport)

    @pytest.mark.asyncio
    async def test_domain_export_response_time(self, tools):
        """Test domain export responds in reasonable time."""
        start_time = time.time()
        result = await tools.export_physics_domain(domain="transport")
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 10.0, (
            f"Domain export took {execution_time:.2f}s, too slow"
        )

        assert isinstance(result, DomainExport)


class TestExportUsability:
    """Test export usability and practical utility."""

    @pytest.mark.asyncio
    async def test_export_provides_useful_data(self, tools, mcp_test_context):
        """Test export provides data that would be useful to users."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.export_ids(ids_list=[ids_name])

        assert isinstance(result, IDSExport)
        # Should provide actionable export information
        assert hasattr(result, "data")

        # Export should be structured for practical use
        export_data = result.data
        assert export_data is not None

    @pytest.mark.asyncio
    async def test_export_format_readability(self, tools):
        """Test export format is readable and well-structured."""
        result = await tools.export_physics_domain(domain="transport")

        assert isinstance(result, DomainExport)
        export_data = result.data

        # Should be well-structured for human consumption
        assert export_data is not None
        # (This is a qualitative test - specific requirements depend on implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
