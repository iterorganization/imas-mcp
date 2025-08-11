"""Physics integration service for IMAS tools."""

from typing import Any

from imas_mcp.models.physics_models import PhysicsSearchResult
from imas_mcp.physics_integration import explain_physics_concept, physics_search

from .base import BaseService


class PhysicsService(BaseService):
    """Service for physics integration and enhancement."""

    async def enhance_query(self, query: str) -> PhysicsSearchResult | None:
        """
        Enhance query with physics context.

        Args:
            query: Search query to enhance

        Returns:
            Physics search result or None if enhancement fails
        """
        try:
            result = physics_search(query)
            self.logger.debug(f"Physics enhancement successful for: {query}")
            return result
        except Exception as e:
            self.logger.warning(f"Physics enhancement failed for '{query}': {e}")
            return None

    async def get_concept_context(
        self, concept: str, detail_level: str = "intermediate"
    ) -> dict[str, Any] | None:
        """Get physics context for a concept."""
        try:
            result = explain_physics_concept(concept, detail_level)
            return {
                "domain": result.domain,
                "description": result.description,
                "phenomena": result.phenomena,
                "typical_units": result.typical_units,
                "complexity_level": result.complexity_level,
            }
        except Exception as e:
            self.logger.warning(f"Concept context failed for '{concept}': {e}")
            return None
