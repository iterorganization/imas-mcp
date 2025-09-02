"""Tests for ResponseService."""

from datetime import UTC
from unittest.mock import Mock

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import SearchResult
from imas_mcp.search.search_strategy import SearchHit
from imas_mcp.services.response import ResponseService


class TestResponseService:
    """Test ResponseService functionality."""

    def test_initialization(self):
        """Test ResponseService initializes correctly."""
        service = ResponseService()
        assert service.logger is not None

    def test_build_search_response_basic(self):
        """Test building basic search response."""
        service = ResponseService()

        # Create mock SearchResult objects that return proper SearchHit instances
        mock_result = Mock()
        mock_result.to_hit.return_value = SearchHit(
            path="test.path",
            documentation="Test documentation",
            ids_name="test_ids",
            score=0.9,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="",
        )

        results = [mock_result]  # type: ignore

        response = service.build_search_response(
            results=results, query="test query", search_mode=SearchMode.SEMANTIC
        )

        assert isinstance(response, SearchResult)
        assert response.query == "test query"
        assert response.search_mode == SearchMode.SEMANTIC
        assert len(response.hits) == 1
        assert response.ai_response == {}
        assert response.hits[0].path == "test.path"

    def test_build_search_response_with_insights(self):
        """Test building search response with AI insights."""
        service = ResponseService()

        mock_result = Mock()
        mock_result.to_hit.return_value = SearchHit(
            path="test.path",
            documentation="Test documentation",
            ids_name="test_ids",
            score=0.9,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="",
        )

        results = [mock_result]  # type: ignore
        ai_response = {}  # ai_response should be empty (reserved for LLM sampling)

        response = service.build_search_response(
            results=results,
            query="plasma temperature",
            search_mode=SearchMode.SEMANTIC,
            ai_response=ai_response,
        )

        assert isinstance(response, SearchResult)
        assert response.query == "plasma temperature"
        assert response.search_mode == SearchMode.SEMANTIC
        assert response.ai_response == ai_response
        assert len(response.hits) == 1

    def test_add_standard_metadata(self):
        """Test adding standard metadata to response."""
        service = ResponseService()

        # Create a mock response with metadata attribute
        mock_response = Mock()
        mock_response.metadata = {}

        updated_response = service.add_standard_metadata(mock_response, "test_tool")

        assert updated_response is mock_response
        # Verify metadata was updated with correct structure
        assert hasattr(mock_response, "metadata")
        assert isinstance(mock_response.metadata, dict)
        assert "tool" in mock_response.metadata
        assert "processing_timestamp" in mock_response.metadata
        assert "version" in mock_response.metadata

    def test_add_standard_metadata_content(self):
        """Test metadata content is correctly populated."""
        service = ResponseService()

        mock_response = Mock()
        mock_response.metadata = {}

        service.add_standard_metadata(mock_response, "search_imas")

        metadata = mock_response.metadata

        # Verify tool name
        assert metadata["tool"] == "search_imas"

        # Verify timestamp format (ISO 8601 with timezone)
        timestamp = metadata["processing_timestamp"]
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format contains T
        assert timestamp.endswith("+00:00") or timestamp.endswith("Z")  # UTC timezone

        # Verify version is present and not placeholder
        version = metadata["version"]
        assert isinstance(version, str)
        assert len(version) > 0
        assert version != "1.0"  # Should use dynamic version, not placeholder

    def test_add_standard_metadata_timestamp_freshness(self):
        """Test that timestamps are current, not hardcoded."""
        from datetime import datetime, timezone

        service = ResponseService()
        mock_response = Mock()
        mock_response.metadata = {}

        # Record time before call
        before_time = datetime.now(UTC)

        service.add_standard_metadata(mock_response, "test_tool")

        # Record time after call
        after_time = datetime.now(UTC)

        # Parse the timestamp from metadata
        timestamp_str = mock_response.metadata["processing_timestamp"]
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        # Verify timestamp is between before and after times (within reasonable range)
        assert before_time <= timestamp <= after_time

    def test_add_standard_metadata_no_metadata_attr(self):
        """Test adding metadata when response has no metadata attribute."""
        service = ResponseService()

        # Create a mock response without metadata attribute
        mock_response = Mock()
        del mock_response.metadata  # Ensure no metadata attribute

        updated_response = service.add_standard_metadata(mock_response, "test_tool")

        # Should still return the response without error
        assert updated_response is mock_response

    def test_add_standard_metadata_none_metadata(self):
        """Test adding metadata when metadata is None."""
        service = ResponseService()

        mock_response = Mock()
        mock_response.metadata = None

        updated_response = service.add_standard_metadata(mock_response, "test_tool")

        assert updated_response is mock_response
