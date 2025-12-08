"""
Path filtering and preprocessing for relationship extraction.
"""

import re
from typing import Any


class PathFilter:
    """Filters and preprocesses paths for relationship extraction."""

    def __init__(self, config):
        """Initialize the path filter with configuration."""
        self.config = config
        self.skip_patterns = [re.compile(pattern) for pattern in config.skip_patterns]

    def filter_meaningful_paths(
        self, ids_data: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Filter paths to include only meaningful ones for relationship extraction."""
        filtered = {}
        total_paths = 0

        for ids_name, ids_info in ids_data.items():
            paths = ids_info.get("paths", {})
            total_paths += len(paths)

            for path, path_data in paths.items():
                if self._should_skip_path(path, path_data):
                    continue

                description = self._build_semantic_description(path, path_data)

                filtered[path] = {
                    "ids": ids_name,
                    "description": description,
                    "data": path_data,
                }

        return filtered

    def _should_skip_path(self, path: str, path_data: dict[str, Any]) -> bool:
        """Check if a path should be skipped during filtering."""
        # Skip if matches skip patterns
        if any(pattern.match(path) for pattern in self.skip_patterns):
            return True

        # Skip if documentation too short
        doc = path_data.get("documentation", "")
        if len(doc.strip()) < self.config.min_documentation_length:
            return True

        # Skip if generic documentation
        if doc.strip() in self.config.generic_docs:
            return True

        return False

    def _build_semantic_description(self, path: str, path_data: dict[str, Any]) -> str:
        """Build semantic description from path and documentation."""
        documentation = path_data.get("documentation", "")
        physics_context = path_data.get("physics_context", "")

        # Clean documentation
        if documentation:
            cleaned_doc = self._clean_documentation(documentation)
        else:
            cleaned_doc = ""

        # Extract meaningful path components
        path_context = self._extract_path_context(path)

        # Combine parts
        description_parts = []
        if cleaned_doc:
            description_parts.append(cleaned_doc)
        if path_context:
            description_parts.append(f"Context: {path_context}")
        if physics_context:
            description_parts.append(f"Physics: {physics_context}")

        return (
            ". ".join(description_parts) if description_parts else "Generic data field"
        )

    def _clean_documentation(self, documentation: str) -> str:
        """Clean and normalize documentation text."""
        # Remove cross-reference sections
        within_pattern = re.compile(
            r"\.\s*Within\s+\w+\s+(IDS|container)\s*:.*?(?=\.|$)",
            re.IGNORECASE | re.DOTALL,
        )
        cleaned_doc = within_pattern.sub("", documentation).strip()
        return re.sub(r"\s+", " ", cleaned_doc).strip()

    def _extract_path_context(self, path: str) -> str:
        """Extract meaningful context from path components."""
        path_parts = path.split("/")
        if len(path_parts) >= 2:
            meaningful_parts = path_parts[1:]  # Skip IDS name
            filtered_parts = [
                part
                for part in meaningful_parts
                if part
                not in ["time_slice", "profiles_1d", "profiles_2d", "global_quantities"]
            ]
            return " ".join(filtered_parts).replace("_", " ")
        return ""


class UnitFamilyBuilder:
    """Builds unit families from filtered path data."""

    def build_unit_families(
        self, filtered_paths: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Build unit families from path data."""
        unit_groups = {}

        for path, path_info in filtered_paths.items():
            units = path_info["data"].get("units", "")
            if units and units not in ["", "1", "-"]:
                if units not in unit_groups:
                    unit_groups[units] = []
                unit_groups[units].append(path)

        # Filter to only include units shared by multiple paths
        unit_families = {}
        for unit, paths in unit_groups.items():
            if len(paths) >= 2:
                unit_families[unit] = {
                    "base_unit": unit,
                    "paths_using": paths,
                }

        return unit_families
