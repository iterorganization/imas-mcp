"""Validation rules extractor."""

import xml.etree.ElementTree as ET
from typing import Any

from imas_mcp.core.extractors.base import BaseExtractor


class ValidationExtractor(BaseExtractor):
    """Extract validation rules and constraints."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
        """Extract validation rules from element attributes."""
        validation_data = {}
        validation_rules = {}

        # Check data type for validation hints
        data_type = elem.get("data_type")
        if data_type:
            validation_rules["data_type"] = data_type

        # Extract range constraints if available
        min_val = elem.get("min")
        max_val = elem.get("max")
        if min_val is not None:
            validation_rules["min_value"] = min_val
        if max_val is not None:
            validation_rules["max_value"] = max_val

        # Extract units constraints
        units = elem.get("units")
        if units and units not in ["", "1", "mixed"]:
            validation_rules["units_required"] = True
        else:
            validation_rules["units_required"] = False

        # Only add validation_rules if we have any rules
        if validation_rules:
            validation_data["validation_rules"] = validation_rules

        return validation_data
