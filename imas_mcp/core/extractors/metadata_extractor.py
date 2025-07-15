"""Metadata extractor for basic element information."""

import xml.etree.ElementTree as ET
from typing import Any, Dict

from .base import BaseExtractor
from ..xml_utils import DocumentationBuilder


class MetadataExtractor(BaseExtractor):
    """Extract basic metadata like documentation, units, coordinates."""

    def extract(self, elem: ET.Element) -> Dict[str, Any]:
        """Extract basic metadata from element."""
        metadata = {}

        # Extract hierarchical documentation with parent context
        documentation_parts = DocumentationBuilder.collect_documentation_hierarchy(
            elem, self.context.ids_elem, self.context.ids_name, self.context.parent_map
        )

        if documentation_parts:
            # Build LLM-optimized hierarchical documentation
            hierarchical_doc = DocumentationBuilder.build_hierarchical_documentation(
                documentation_parts
            )
            metadata["documentation"] = hierarchical_doc
        else:
            # Fallback to direct documentation
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
