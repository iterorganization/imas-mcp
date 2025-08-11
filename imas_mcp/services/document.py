"""Document store service for IMAS tools."""

from typing import Any

from imas_mcp.models.error_models import ToolError
from imas_mcp.search.document_store import DocumentStore

from .base import BaseService


class DocumentService(BaseService):
    """Service for document store operations."""

    def __init__(self, document_store: DocumentStore | None = None):
        super().__init__()
        self.store = document_store or DocumentStore()

    async def validate_ids(self, ids_names: list[str]) -> tuple[list[str], list[str]]:
        """
        Check IDS names against available IDS.

        Returns:
            Tuple of (valid_ids, invalid_ids)
        """
        available_ids = self.store.get_available_ids()
        valid_ids = [ids for ids in ids_names if ids in available_ids]
        invalid_ids = [ids for ids in ids_names if ids not in available_ids]
        return valid_ids, invalid_ids

    async def get_documents_safe(self, ids_name: str) -> list[Any]:
        """Get documents for IDS with error handling."""
        try:
            return self.store.get_documents_by_ids(ids_name)
        except Exception as e:
            self.logger.error(f"Failed to get documents for {ids_name}: {e}")
            return []

    def create_ids_not_found_error(self, ids_name: str, tool_name: str) -> ToolError:
        """Create standardized IDS not found error."""
        available_ids = self.store.get_available_ids()
        return ToolError(
            error=f"IDS '{ids_name}' not found",
            suggestions=[f"Try: {ids}" for ids in available_ids[:5]],
            context={
                "available_ids": available_ids[:10],
                "ids_name": ids_name,
                "tool": tool_name,
            },
        )
