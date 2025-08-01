"""Tests for BaseService."""

import pytest

from imas_mcp.services.base import BaseService


class TestBaseService:
    """Test BaseService functionality."""

    def test_initialization(self):
        """Test BaseService initializes correctly."""
        service = BaseService()
        assert service.logger is not None

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialize method."""
        service = BaseService()
        # Should not raise any exception
        await service.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method."""
        service = BaseService()
        # Should not raise any exception
        await service.cleanup()
