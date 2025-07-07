"""
JSON Data Dictionary accessor for pre-built JSON files stored in resources.

This module provides access to pre-built JSON data structures that are created
during the package build process and stored in the resources directory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .core.data_model import TransformationOutputs

logger = logging.getLogger(__name__)


class JsonDataDictionaryAccessor:
    """Provides access to pre-built JSON data dictionary files."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the accessor.

        Args:
            data_dir: Optional custom data directory. If None, uses the package resources.
        """
        if data_dir is None:
            # Use the package resources directory
            self.data_dir = Path(__file__).resolve().parent / "resources" / "json_data"
        else:
            self.data_dir = data_dir

        self._catalog_cache: Optional[Dict[str, Any]] = None
        self._relationships_cache: Optional[Dict[str, Any]] = None
        self._graph_stats_cache: Optional[Dict[str, Any]] = None

    def is_available(self) -> bool:
        """Check if JSON data is available."""
        return (self.data_dir / "ids_catalog.json").exists()

    def get_catalog(self) -> Dict[str, Any]:
        """Get the IDS catalog data."""
        if self._catalog_cache is None:
            catalog_path = self.data_dir / "ids_catalog.json"
            if not catalog_path.exists():
                raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

            with open(catalog_path, "r", encoding="utf-8") as f:
                self._catalog_cache = json.load(f)

        assert self._catalog_cache is not None
        return self._catalog_cache

    def get_relationships(self) -> Dict[str, Any]:
        """Get the relationships data."""
        if self._relationships_cache is None:
            rel_path = self.data_dir / "relationships.json"
            if not rel_path.exists():
                raise FileNotFoundError(f"Relationships file not found: {rel_path}")

            with open(rel_path, "r", encoding="utf-8") as f:
                self._relationships_cache = json.load(f)

        assert self._relationships_cache is not None
        return self._relationships_cache

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get precomputed graph statistics for all IDS."""
        if self._graph_stats_cache is None:
            catalog = self.get_catalog()
            self._graph_stats_cache = catalog.get("graph_statistics", {})

        assert self._graph_stats_cache is not None
        return self._graph_stats_cache

    def get_ids_graph_stats(self, ids_name: str) -> Dict[str, Any]:
        """Get graph statistics for a specific IDS."""
        graph_stats = self.get_graph_statistics()
        return graph_stats.get(ids_name, {})

    def get_structural_insights(self) -> Dict[str, Any]:
        """Get overall structural insights across all IDS."""
        catalog = self.get_catalog()
        return catalog.get("structural_insights", {})

    def get_available_ids(self) -> List[str]:
        """Get list of available IDS names."""
        catalog = self.get_catalog()
        return list(catalog.get("ids_catalog", {}).keys())

    def get_ids_detailed_data(self, ids_name: str) -> Dict[str, Any]:
        """Get detailed data for a specific IDS.

        Args:
            ids_name: Name of the IDS to get data for

        Returns:
            Dictionary containing detailed IDS data

        Raises:
            FileNotFoundError: If the detailed file for the IDS doesn't exist
            ValueError: If the IDS name is invalid
        """
        if ids_name not in self.get_available_ids():
            raise ValueError(f"IDS '{ids_name}' not found in catalog")

        detailed_file = self.data_dir / "detailed" / f"{ids_name}.json"
        if not detailed_file.exists():
            raise FileNotFoundError(f"Detailed file not found: {detailed_file}")

        with open(detailed_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_ids_paths(self, ids_name: str) -> Dict[str, Dict[str, Any]]:
        """Get all paths for a specific IDS.

        Args:
            ids_name: Name of the IDS to get paths for

        Returns:
            Dictionary mapping path names to path data
        """
        detailed_data = self.get_ids_detailed_data(ids_name)
        return detailed_data.get("paths", {})

    def get_path_documentation(self, ids_name: str, path: str) -> str:
        """Get LLM-optimized documentation for a specific path.

        Args:
            ids_name: Name of the IDS
            path: Full path within the IDS

        Returns:
            LLM-optimized documentation string

        Raises:
            KeyError: If the path doesn't exist in the IDS
        """
        paths = self.get_ids_paths(ids_name)
        if path not in paths:
            raise KeyError(f"Path '{path}' not found in IDS '{ids_name}'")

        return paths[path].get("documentation", "")

    def get_path_units(self, ids_name: str, path: str) -> str:
        """Get units for a specific path.

        Args:
            ids_name: Name of the IDS
            path: Full path within the IDS

        Returns:
            Units string for the path, empty string if no units

        Raises:
            KeyError: If the path doesn't exist in the IDS
        """
        paths = self.get_ids_paths(ids_name)
        if path not in paths:
            raise KeyError(f"Path '{path}' not found in IDS '{ids_name}'")

        return paths[path].get("units", "")

    def search_paths_by_pattern(
        self, pattern: str, ids_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for paths matching a pattern.

        Args:
            pattern: Pattern to search for in path names
            ids_filter: Optional list of IDS names to limit search to

        Returns:
            List of dictionaries containing matching path information
        """
        results = []
        available_ids = self.get_available_ids()

        # Filter IDS list if specified
        if ids_filter:
            available_ids = [ids for ids in available_ids if ids in ids_filter]

        for ids_name in available_ids:
            try:
                paths = self.get_ids_paths(ids_name)
                for path_name, path_data in paths.items():
                    if pattern.lower() in path_name.lower():
                        result = {
                            "ids_name": ids_name,
                            "path": path_name,
                            "documentation": path_data.get("documentation", ""),
                            "units": path_data.get("units", ""),
                        }
                        results.append(result)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not process IDS '{ids_name}': {e}")
                continue

        return results

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the JSON data."""
        catalog = self.get_catalog()
        relationships = self.get_relationships()

        return {
            "catalog_metadata": catalog.get("metadata", {}),
            "relationships_metadata": relationships.get("metadata", {}),
            "available_ids": self.get_available_ids(),
            "total_ids": len(self.get_available_ids()),
            "data_directory": str(self.data_dir),
        }

    def get_transformation_outputs(self) -> TransformationOutputs:
        """Get TransformationOutputs object pointing to the JSON files.

        Returns:
            TransformationOutputs object with paths to the JSON files
        """
        catalog_path = self.data_dir / "ids_catalog.json"
        relationships_path = self.data_dir / "relationships.json"
        detailed_dir = self.data_dir / "detailed"

        # Find all detailed files
        detailed_files = []
        if detailed_dir.exists():
            detailed_files = list(detailed_dir.glob("*.json"))

        return TransformationOutputs(
            catalog=catalog_path,
            relationships=relationships_path,
            detailed=detailed_files,
        )
