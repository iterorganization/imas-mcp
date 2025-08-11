"""Base service class for dependency injection."""

import logging

logger = logging.getLogger(__name__)


class BaseService:
    """Base class for all services with common functionality."""

    def __init__(self):
        self.logger = logger
