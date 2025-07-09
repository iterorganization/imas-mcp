"""Metadata extractor for basic element information."""

import xml.etree.ElementTree as ET
from typing import Any, Dict

from .base import BaseExtractor


class MetadataExtractor(BaseExtractor):
    """Extract basic metadata like documentation, units, coordinates."""

    def extract(self, elem: ET.Element) -> Dict[str, Any]:
        """Extract basic metadata from element."""
        metadata = {}

        # Extract documentation
        doc_text = elem.get("documentation") or elem.text or ""
        if doc_text:
            metadata["documentation"] = doc_text.strip()

        # Extract units
        units = elem.get("units", "")
        metadata["units"] = units

        # Build coordinates list
        coordinates = []
        if elem.get("coordinate1"):
            coordinates.append(elem.get("coordinate1"))
        if elem.get("coordinate2"):
            coordinates.append(elem.get("coordinate2"))
        metadata["coordinates"] = coordinates

        # Extract data type
        data_type = elem.get("data_type")
        if data_type:
            metadata["data_type"] = data_type

        # Extract structure reference
        structure_ref = elem.get("structure_reference")
        if structure_ref:
            metadata["structure_reference"] = structure_ref

        return self._clean_metadata(metadata)

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up None values but keep required fields."""
        cleaned = {}
        required_fields = {"documentation", "units", "coordinates", "data_type"}

        for k, v in metadata.items():
            if k in required_fields or (v is not None and v != ""):
                cleaned[k] = v

        return cleaned
