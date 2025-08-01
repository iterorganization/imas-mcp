"""Base service class for dependency injection."""

import logging
from abc import ABC

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Base class for all services with common functionality."""

    def __init__(self):
        self.logger = logger

    async def initialize(self) -> None:
        """Initialize service resources. Override in subclasses."""
        pass

    async def cleanup(self) -> None:
        """Cleanup service resources. Override in subclasses."""
        pass
