"""Tests for DocumentService."""

from unittest.mock import MagicMock

import pytest

from imas_mcp.services.document import DocumentService


class TestDocumentService:
    """Test DocumentService functionality."""

    def test_initialization(self):
        """Test DocumentService initializes correctly."""
        service = DocumentService()
        assert service.logger is not None
        assert service.store is not None

    def test_initialization_with_custom_store(self):
        """Test DocumentService with custom store."""
        mock_store = MagicMock()
        service = DocumentService(document_store=mock_store)
        assert service.store is mock_store

    @pytest.mark.asyncio
    async def test_validate_ids(self):
        """Test IDS validation."""
        service = DocumentService()

        # Mock the store method
        service.store.get_available_ids = MagicMock(
            return_value=["equilibrium", "core_profiles"]
        )

        valid_ids, invalid_ids = await service.validate_ids(
            ["equilibrium", "unknown_ids"]
        )

        assert "equilibrium" in valid_ids
        assert "unknown_ids" in invalid_ids

    @pytest.mark.asyncio
    async def test_get_documents_safe_success(self):
        """Test successful document retrieval."""
        service = DocumentService()

        # Mock successful response
        service.store.get_documents_by_ids = MagicMock(return_value=["doc1", "doc2"])

        result = await service.get_documents_safe("equilibrium")

        assert len(result) == 2
        service.store.get_documents_by_ids.assert_called_once_with("equilibrium")

    @pytest.mark.asyncio
    async def test_get_documents_safe_error(self):
        """Test document retrieval with error."""
        service = DocumentService()

        # Mock error
        service.store.get_documents_by_ids = MagicMock(
            side_effect=Exception("Test error")
        )

        result = await service.get_documents_safe("equilibrium")

        assert result == []

    def test_create_ids_not_found_error(self):
        """Test IDS not found error creation."""
        service = DocumentService()

        # Mock available IDS
        service.store.get_available_ids = MagicMock(
            return_value=["equilibrium", "core_profiles"]
        )

        error_response = service.create_ids_not_found_error("unknown_ids", "test_tool")

        assert hasattr(error_response, "error")
        assert "unknown_ids" in error_response.error
        assert hasattr(error_response, "suggestions")
        assert hasattr(error_response, "context")
