"""Path building and semantic analysis extractors."""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from .base import BaseExtractor


class PathExtractor(BaseExtractor):
    """Extract and build element paths."""

    def extract(self, elem: ET.Element) -> Dict[str, Any]:
        """Extract path information."""
        path_data = {}

        # Build full path
        full_path = self._build_element_path(elem)
        if full_path:
            path_data["path"] = full_path

        return path_data

    def _build_element_path(self, elem: ET.Element) -> str:
        """Build full path for element."""
        path_parts = []
        current = elem

        # Walk up the tree to build path
        while current is not None and current != self.context.ids_elem:
            name = current.get("name")
            if name:
                path_parts.append(name)
            current = self.context.parent_map.get(current)

        if not path_parts:
            return f"{self.context.ids_name}/"

        return f"{self.context.ids_name}/{'/'.join(reversed(path_parts))}"


class SemanticExtractor(BaseExtractor):
    """Extract semantic groupings and similarity analysis."""

    def extract(self, elem: ET.Element) -> Dict[str, Any]:
        """Extract semantic information for the element."""
        # This extractor works on collections of paths, not individual elements
        # Implementation would be called at a higher level
        return {}

    def extract_semantic_groups(
        self, paths: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Group paths by semantic similarity."""
        semantic_groups = {}

        # Group by common prefixes and physics concepts
        for path, metadata in paths.items():
            group_key = self._determine_semantic_group(path, metadata)
            if group_key:
                if group_key not in semantic_groups:
                    semantic_groups[group_key] = []
                semantic_groups[group_key].append(path)

        # Filter out single-item groups
        return {k: v for k, v in semantic_groups.items() if len(v) > 1}

    def _determine_semantic_group(self, path: str, metadata: Dict[str, Any]) -> str:
        """Determine semantic group for a path."""
        # Extract physics context
        physics_context = metadata.get("physics_context", {})
        if isinstance(physics_context, dict):
            domain = physics_context.get("domain")
            if domain:
                return f"physics_{domain}"

        # Group by units
        units = metadata.get("units", "")
        if units and units not in ["", "1", "mixed"]:
            return f"units_{units.replace('/', '_').replace('.', '_')}"

        # Group by coordinate system
        coordinates = metadata.get("coordinates", [])
        if coordinates:
            coord_key = "_".join(coordinates[:2])  # Use first two coordinates
            return f"coordinates_{coord_key}"

        # Group by path structure
        path_parts = path.split("/")
        if len(path_parts) >= 3:
            return f"structure_{path_parts[1]}"  # Second level grouping

        return "misc"
