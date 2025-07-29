"""
Base tool functionality for IMAS MCP tools.

This module contains common functionality shared across all tool implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all IMAS MCP tools."""

    def __init__(self):
        self.logger = logger

    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool."""
        pass

    def _create_error_response(
        self, error_message: str, query: str = ""
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "error": error_message,
            "query": query,
            "tool": self.get_tool_name(),
            "status": "error",
        }

    def _convert_to_enum(self, value: str | Enum, enum_class: type[Enum]) -> Enum:
        """
        Convert string value to enum, handling both string and enum inputs.

        This utility handles the common pattern of accepting both string and enum
        parameters in tool methods, where validation decorators convert strings to enums
        but the method needs to handle both cases.

        Args:
            value: String value or already-converted enum
            enum_class: Target enum class

        Returns:
            Enum value

        Raises:
            ValueError: If string value is not a valid enum option
            TypeError: If value is neither string nor target enum type
        """
        # Auto-generate parameter name from enum class name
        import re

        parameter_name = re.sub(r"(?<!^)(?=[A-Z])", "_", enum_class.__name__).lower()

        if isinstance(value, enum_class):
            return value

        if isinstance(value, str):
            # Create mapping from string values to enum members
            value_map = {member.value: member for member in enum_class}

            if value not in value_map:
                valid_values = list(value_map.keys())
                raise ValueError(
                    f"Invalid {parameter_name}: {value}. Valid options: {valid_values}"
                )

            return value_map[value]

        raise TypeError(
            f"{parameter_name} must be string or {enum_class.__name__} enum, "
            f"got {type(value).__name__}"
        )
