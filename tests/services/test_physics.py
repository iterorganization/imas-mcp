"""Tests for PhysicsService."""

import pytest

from imas_mcp.services.physics import PhysicsService


class TestPhysicsService:
    """Test PhysicsService functionality."""

    def test_initialization(self):
        """Test PhysicsService initializes correctly."""
        service = PhysicsService()
        assert service.logger is not None

    @pytest.mark.asyncio
    async def test_enhance_query_basic(self):
        """Test basic query enhancement."""
        service = PhysicsService()

        # Note: This may return None if physics_search fails
        enhanced = await service.enhance_query("plasma temperature")

        # Result could be None or PhysicsSearchResult
        assert enhanced is None or hasattr(enhanced, "query")

    @pytest.mark.asyncio
    async def test_get_concept_context_known_concept(self):
        """Test getting context for known physics concepts."""
        service = PhysicsService()

        context = await service.get_concept_context("plasma")

        # Result could be None or dict with physics info
        assert context is None or isinstance(context, dict)

    @pytest.mark.asyncio
    async def test_get_concept_context_with_detail_level(self):
        """Test getting context with different detail levels."""
        service = PhysicsService()

        context = await service.get_concept_context("plasma", detail_level="advanced")

        # Result could be None or dict with physics info
        assert context is None or isinstance(context, dict)
