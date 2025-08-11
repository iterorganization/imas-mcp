"""Coordinate system extractor."""

import xml.etree.ElementTree as ET
from typing import Any

from imas_mcp.core.extractors.base import BaseExtractor


class CoordinateExtractor(BaseExtractor):
    """Extract coordinate system information."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
        """Extract coordinate systems from IDS element."""
        # This operates on IDS level, not individual elements
        return {}

    def extract_coordinate_systems(
        self, ids_elem: ET.Element
    ) -> dict[str, dict[str, Any]]:
        """Extract coordinate systems for an entire IDS."""
        coordinate_systems = {}

        # Find coordinate system definitions
        for coord_elem in ids_elem.findall(".//coordinate"):
            coord_name = coord_elem.get("name")
            if not coord_name:
                continue

            coord_data = {
                "name": coord_name,
                "description": coord_elem.get("documentation", ""),
                "units": coord_elem.get("units", ""),
            }

            # Extract coordinate identifiers
            identifiers = []
            for identifier in coord_elem.findall(".//identifier"):
                id_name = identifier.get("name")
                if id_name:
                    identifiers.append(
                        {
                            "name": id_name,
                            "description": identifier.get("documentation", ""),
                        }
                    )

            coord_data["identifiers"] = identifiers
            coordinate_systems[coord_name] = coord_data

        return coordinate_systems
