"""Metadata extractor for basic element information."""

import xml.etree.ElementTree as ET
from typing import Any, Dict

from imas_mcp.core.extractors.base import BaseExtractor
from imas_mcp.core.xml_utils import DocumentationBuilder


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
        coordinate1 = elem.get("coordinate1")
        coordinate2 = elem.get("coordinate2")

        if coordinate1:
            coordinates.append(coordinate1)
        if coordinate2:
            coordinates.append(coordinate2)
        metadata["coordinates"] = coordinates

        # Extract individual coordinate fields (these were missing!)
        metadata["coordinate1"] = coordinate1
        metadata["coordinate2"] = coordinate2

        # Extract data type
        data_type = elem.get("data_type")
        if data_type:
            metadata["data_type"] = data_type

        # Extract structure reference
        structure_ref = elem.get("structure_reference")
        if structure_ref:
            metadata["structure_reference"] = structure_ref

        # Extract timebase (this was missing!)
        timebase = elem.get("timebase")
        metadata["timebase"] = timebase

        # Extract type (this was missing!)
        type_attr = elem.get("type")
        metadata["type"] = type_attr

        # Extract introduced_after and introduced_after_version
        introduced_after = elem.get("introduced_after")
        introduced_after_version = elem.get("introduced_after_version")

        # Use introduced_after_version as the primary field, fallback to introduced_after
        if introduced_after_version:
            metadata["introduced_after_version"] = introduced_after_version
        elif introduced_after:
            metadata["introduced_after_version"] = introduced_after

        # Extract lifecycle fields
        lifecycle_status = elem.get("lifecycle_status")
        metadata["lifecycle_status"] = lifecycle_status

        lifecycle_version = elem.get("lifecycle_version")
        metadata["lifecycle_version"] = lifecycle_version

        return self._clean_metadata(metadata)

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up None values but keep required fields."""
        cleaned = {}
        required_fields = {
            "documentation",
            "units",
            "coordinates",
            "data_type",
            "coordinate1",
            "coordinate2",
            "timebase",
            "type",
            "introduced_after_version",
            "lifecycle_status",
            "lifecycle_version",
            "structure_reference",
        }

        for k, v in metadata.items():
            if k in required_fields or (v is not None and v != ""):
                cleaned[k] = v

        return cleaned
