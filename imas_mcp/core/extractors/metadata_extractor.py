"""Metadata extractor for basic element information."""

import re
import xml.etree.ElementTree as ET
from typing import Any

from imas_mcp.core.extractors.base import BaseExtractor
from imas_mcp.core.xml_utils import DocumentationBuilder


class MetadataExtractor(BaseExtractor):
    """Extract basic metadata like documentation, units, coordinates."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
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

        # Extract and resolve units
        units = elem.get("units", "")
        resolved_units = self._resolve_unit_references(units, elem)
        metadata["units"] = resolved_units

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

    def _resolve_unit_references(self, units: str, elem: ET.Element) -> str:
        """Resolve unit references including coordinate systems and process references."""
        if not units or units in ["", "1", "dimensionless", "none"]:
            return units

        # Handle coordinate system unit references
        if "units given by coordinate_system" in units:
            resolved = self._resolve_coordinate_system_units(units)
            # Return resolved unit even if empty (empty means dimensionless)
            return resolved

        # Handle process unit references
        if "units given by process" in units:
            resolved = self._resolve_process_units(units, elem)
            # Return resolved unit even if empty (empty means dimensionless)
            return resolved

        # Handle dimensional units (m^dimension)
        if units.endswith("^dimension"):
            resolved = self._resolve_dimensional_units(units, elem)
            if resolved:
                return resolved

        # Return original units if no resolution needed
        return units

    def _resolve_coordinate_system_units(self, unit_str: str) -> str:
        """
        Resolve coordinate system unit references to actual units.

        Args:
            unit_str: Unit string that may contain coordinate system references

        Returns:
            Resolved unit string or empty string if resolution fails
        """
        # Pattern: "units given by coordinate_system(:)/coordinate(:)/units"
        coord_pattern = (
            r"units given by coordinate_system\([^)]*\)/coordinate\([^)]*\)/units"
        )

        if re.match(coord_pattern, unit_str):
            try:
                # Extract coordinate system and coordinate identifiers
                cs_match = re.search(r"coordinate_system\(([^)]*)\)", unit_str)
                coord_match = re.search(r"coordinate\(([^)]*)\)", unit_str)

                if cs_match and coord_match:
                    coord_identifier = coord_match.group(1)

                    # Handle empty/unspecified coordinate identifiers (often ':')
                    if not coord_identifier or coord_identifier == ":":
                        return ""  # Dimensionless/unspecified

                    # Common coordinate mappings
                    coordinate_units = {
                        "r": "m",
                        "x": "m",
                        "y": "m",
                        "z": "m",
                        "phi": "rad",
                        "theta": "rad",
                        "psi": "Wb",
                        "rho_tor": "1",
                        "rho_pol": "1",
                        "1": "m",
                        "2": "m",
                        "3": "m",
                    }

                    return coordinate_units.get(coord_identifier, "")

            except Exception:
                pass

        return ""

    def _resolve_process_units(self, units: str, elem: ET.Element) -> str:
        """Resolve process-based unit references."""
        # Pattern: "units given by process(:)/results_units" or "units given by process(i1)/results_units"
        # These typically refer to units defined by computational processes
        # For now, return empty string as these are context-dependent and often dimensionless
        return ""

    def _resolve_dimensional_units(self, units: str, elem: ET.Element) -> str:
        """Resolve dimensional units like 'm^dimension'."""
        # Pattern: "m^dimension" where dimension is context-dependent
        if units == "m^dimension":
            # Try to infer dimension from context
            # Look for coordinate information or data structure
            coordinate1 = elem.get("coordinate1")
            coordinate2 = elem.get("coordinate2")

            # If we have coordinates, infer dimensionality
            if coordinate1 and coordinate2:
                return "m^2"  # Area
            elif coordinate1:
                return "m"  # Length
            else:
                # Default to volume for unspecified dimensional units
                return "m^3"

        # Handle other dimensional patterns
        base_unit = units.split("^")[0]
        return base_unit  # Return base unit without dimension

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
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
