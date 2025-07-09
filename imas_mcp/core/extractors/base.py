"""Base classes for composable extractors."""

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ...dd_accessor import ImasDataDictionaryAccessor


@dataclass
class ExtractorContext:
    """Shared context for all extractors."""

    dd_accessor: ImasDataDictionaryAccessor
    root: ET.Element
    ids_elem: ET.Element
    ids_name: str
    parent_map: Dict[ET.Element, ET.Element]

    # Configuration
    excluded_patterns: Set[str]
    skip_ggd: bool = True

    def __post_init__(self):
        """Build parent map for efficient tree traversal."""
        if not hasattr(self, "_parent_map_built"):
            self.parent_map = {c: p for p in self.ids_elem.iter() for c in p}
            self._parent_map_built = True


class BaseExtractor(ABC):
    """Base class for all extractors with filtering capabilities."""

    def __init__(self, context: ExtractorContext):
        self.context = context

    @abstractmethod
    def extract(self, elem: ET.Element) -> Dict[str, Any]:
        """Extract data from an XML element."""
        pass

    def should_filter_element_or_path(
        self,
        elem: Optional[ET.Element] = None,
        path: Optional[str] = None,
        elem_name: Optional[str] = None,
        filter_type: str = "element",
    ) -> bool:
        """Unified filtering logic for elements and paths."""

        # Extract relevant attributes
        if elem is not None:
            path = elem.get("path", "")
            elem_name = elem.get("name", "")

        if not path and not elem_name:
            return True  # Filter out if no identifying information

        path_lower = path.lower() if path else ""
        name_lower = elem_name.lower() if elem_name else ""

        # Filter out excluded patterns
        for pattern in self.context.excluded_patterns:
            if pattern in path_lower or pattern in name_lower:
                return True

        # Filter out GGD if configured
        if self.context.skip_ggd and ("ggd" in path_lower or "ggd" in name_lower):
            return True

        # Filter type-specific logic
        if filter_type == "coordinate":
            return self._filter_coordinate_noise(path, elem_name)
        elif filter_type == "relationship":
            return self._filter_relationship_noise(path, elem_name)
        elif filter_type == "element":
            return self._filter_element_noise(path, elem_name)

        return False

    def _filter_coordinate_noise(
        self, path: Optional[str], elem_name: Optional[str]
    ) -> bool:
        """Filter coordinate-specific noise."""
        if not path:
            return True

        # Skip ids_properties coordinates
        if path.startswith("ids_properties"):
            return True

        return False

    def _filter_relationship_noise(
        self, path: Optional[str], elem_name: Optional[str]
    ) -> bool:
        """Filter relationship-specific noise."""
        if not elem_name:
            return True

        name_lower = elem_name.lower()

        # Filter out generic attributes that add noise to relationships
        generic_noise = {
            "name",
            "description",
            "type",
            "value",
            "index",
            "identifier",
            "label",
            "reference",
            "version",
            "data",
            "units",
            "documentation",
            "comment",
            "note",
            "info",
            "metadata",
            "id",
            "count",
            "number",
        }

        if name_lower in generic_noise:
            return True

        return False

    def _filter_element_noise(
        self, path: Optional[str], elem_name: Optional[str]
    ) -> bool:
        """Filter element-specific noise."""
        # Basic element filtering can be added here
        return False


class ComposableExtractor:
    """Composer that coordinates multiple extractors."""

    def __init__(self, extractors: List[BaseExtractor]):
        self.extractors = extractors

    def extract_all(self, elem: ET.Element) -> Dict[str, Any]:
        """Run all extractors and merge results."""
        result = {}

        for extractor in self.extractors:
            try:
                extracted = extractor.extract(elem)
                result.update(extracted)
            except Exception as e:
                # Log error but continue with other extractors
                print(f"Warning: {extractor.__class__.__name__} failed: {e}")

        return result
