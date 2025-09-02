"""Tests for BaseService."""

import pytest

from imas_mcp.services.base import BaseService


class TestBaseService:
    """Test BaseService functionality."""

    def test_initialization(self):
        """Test BaseService initializes correctly."""
        service = BaseService()
        assert service.logger is not None
