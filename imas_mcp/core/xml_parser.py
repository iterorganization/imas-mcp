"""Enhanced XML parser for IMAS Data Dictionary with relationship extraction."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..dd_accessor import ImasDataDictionaryAccessor
from ..graph_analyzer import analyze_imas_graphs
from .data_model import (
    CatalogMetadata,
    CoordinateSystem,
    DataPath,
    IdsDetailed,
    IdsInfo,
    PhysicsDomain,
    Relationships,
    TransformationOutputs,
    UnitFamily,
)
from .xml_utils import DocumentationBuilder


@dataclass
class DataDictionaryTransformer:
    """Transform IDSDef.xml into layered JSON structure using existing accessor."""

    output_dir: Optional[Path] = None
    dd_accessor: Optional[ImasDataDictionaryAccessor] = None
    ids_set: Optional[Set[str]] = None  # Restrict processing to specific IDS

    # Processing configuration
    excluded_patterns: Set[str] = field(
        default_factory=lambda: {"ids_properties", "code", "error"}
    )
    skip_ggd: bool = True

    def __post_init__(self):
        """Initialize the transformer."""
        if self.dd_accessor is None:
            self.dd_accessor = ImasDataDictionaryAccessor()

        # Default to resources directory if no output_dir specified
        if self.output_dir is None:
            self.output_dir = (
                Path(__file__).resolve().parent.parent / "resources" / "json_data"
            )

        # Ensure output_dir is a Path object for type safety
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Cache XML tree for reuse
        self._tree = self.dd_accessor.get_xml_tree()
        self._root = self._tree.getroot()

    @property
    def resolved_output_dir(self) -> Path:
        """Get the resolved output directory as a Path object."""
        assert self.output_dir is not None, "output_dir should be set in __post_init__"
        return self.output_dir

    def transform_complete(self) -> TransformationOutputs:
        """Transform XML to complete JSON structure."""
        if self._root is None:
            raise ValueError("XML root is None")

        # Extract all IDS information
        ids_data = self._extract_ids_data(self._root)

        # Perform graph analysis on extracted data
        graph_data = self._analyze_graph_structure(ids_data)

        # Generate outputs with graph data
        catalog_path = self._generate_catalog(ids_data, graph_data)
        detailed_paths = self._generate_detailed_files(ids_data)
        relationships_path = self._generate_relationships(ids_data)

        return TransformationOutputs(
            catalog=catalog_path,
            detailed=detailed_paths,
            relationships=relationships_path,
        )

    def _extract_ids_data(self, root: ET.Element) -> Dict[str, Dict[str, Any]]:
        """Extract structured data from XML root."""
        ids_data = {}

        # Find all IDS elements
        for ids_elem in root.findall(".//IDS[@name]"):
            ids_name = ids_elem.get("name")
            if not ids_name:
                continue

            # Skip if not in the specified IDS set
            if self.ids_set is not None and ids_name not in self.ids_set:
                continue

            # Skip GGD nodes if configured
            if self.skip_ggd and ids_name.lower().startswith("ggd"):
                continue

            # Extract IDS information
            ids_info = self._extract_ids_info(ids_elem, ids_name)
            coordinate_systems = self._extract_coordinate_systems(ids_elem)
            paths = self._extract_paths(ids_elem, ids_name)
            semantic_groups = self._extract_semantic_groups(paths)

            ids_data[ids_name] = {
                "ids_info": ids_info,
                "coordinate_systems": coordinate_systems,
                "paths": paths,
                "semantic_groups": semantic_groups,
            }

        return ids_data

    def _extract_ids_info(self, ids_elem: ET.Element, ids_name: str) -> Dict[str, Any]:
        """Extract basic IDS information."""
        return {
            "name": ids_name,
            "description": ids_elem.get("documentation", ""),
            "version": self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            "max_depth": self._calculate_max_depth(ids_elem),
            "leaf_count": len(self._get_leaf_nodes(ids_elem)),
            "physics_domain": self._infer_physics_domain(ids_name),
            "documentation_coverage": self._calculate_documentation_coverage(ids_elem),
            "related_ids": [],  # To be populated by relationship analysis
            "common_use_cases": [],  # To be populated by analysis
        }

    def _extract_coordinate_systems(
        self, ids_elem: ET.Element
    ) -> Dict[str, Dict[str, Any]]:
        """Extract coordinate system information."""
        coordinate_systems = {}

        # Create parent map for filtering
        parent_map = {c: p for p in ids_elem.iter() for c in p}

        # Look for coordinate-related elements
        for elem in ids_elem.findall(".//*[@coordinate1]"):
            # Skip if element should be filtered out
            if self._should_skip_element(elem, ids_elem, parent_map):
                continue

            coord1 = elem.get("coordinate1")
            if coord1 and coord1 not in coordinate_systems:
                # Additional filtering for GGD coordinates
                if self.skip_ggd and "ggd" in coord1.lower():
                    continue

                coordinate_systems[coord1] = {
                    "description": f"Coordinate system: {coord1}",
                    "units": elem.get("units", ""),
                    "range": None,
                    "usage": "Primary coordinate",
                }

        for elem in ids_elem.findall(".//*[@coordinate2]"):
            # Skip if element should be filtered out
            if self._should_skip_element(elem, ids_elem, parent_map):
                continue

            coord2 = elem.get("coordinate2")
            if coord2 and coord2 not in coordinate_systems:
                # Additional filtering for GGD coordinates
                if self.skip_ggd and "ggd" in coord2.lower():
                    continue

                coordinate_systems[coord2] = {
                    "description": f"Coordinate system: {coord2}",
                    "units": elem.get("units", ""),
                    "range": None,
                    "usage": "Secondary coordinate",
                }

        return coordinate_systems

    def _extract_paths(
        self, ids_elem: ET.Element, ids_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Extract all paths with their metadata."""
        paths = {}

        # Create parent map once for this IDS for efficient parent lookup
        parent_map = {c: p for p in ids_elem.iter() for c in p}

        # Get all named elements, excluding excluded patterns
        for elem in ids_elem.findall(".//*[@name]"):
            elem_name = elem.get("name")
            if not elem_name:
                continue

            # Skip excluded patterns
            if self._should_skip_element(elem, ids_elem, parent_map):
                continue

            # Build path
            path = self._build_element_path(elem, ids_elem, ids_name, parent_map)
            if not path:
                continue

            # Extract element metadata with hierarchical documentation
            path_data = self._extract_element_metadata(
                elem, ids_elem, ids_name, parent_map
            )
            path_data["path"] = path  # Add the path field
            paths[path] = path_data

        return paths

    def _extract_element_metadata(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Dict[str, Any]:
        """Extract metadata from XML element attributes with hierarchical documentation."""
        # Handle units properly - convert empty strings to None
        units = elem.get("units", "")
        if units and units.strip():
            units = units.strip()
        else:
            units = None

        # Collect hierarchical documentation
        documentation_parts = DocumentationBuilder.collect_documentation_hierarchy(
            elem, ids_elem, ids_name, parent_map
        )
        hierarchical_doc = DocumentationBuilder.build_hierarchical_documentation(
            documentation_parts
        )

        metadata = {
            "documentation": hierarchical_doc or elem.get("documentation", ""),
            "units": units,  # Always include units field
            "coordinates": [],
            "lifecycle": "active",
            "data_type": elem.get("type", ""),
            "introduced_after": elem.get("introduced_after"),
            "element_type": elem.get("element_type"),
            "coordinate1": elem.get("coordinate1"),
            "coordinate2": elem.get("coordinate2"),
            "timebase": elem.get("timebase"),
            "type": elem.get("type"),
        }

        # Build coordinates list
        coordinates = []
        if elem.get("coordinate1"):
            coordinates.append(elem.get("coordinate1"))
        if elem.get("coordinate2"):
            coordinates.append(elem.get("coordinate2"))
        metadata["coordinates"] = coordinates

        # Clean up None values but keep required fields like 'units'
        cleaned_metadata = {}
        required_fields = {"documentation", "units", "coordinates", "lifecycle"}
        for k, v in metadata.items():
            if k in required_fields or v is not None:
                cleaned_metadata[k] = v

        return cleaned_metadata

    def _build_element_path(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Optional[str]:
        """Build full path for element."""
        path_parts = []
        current = elem

        # Walk up the tree to build path using parent map
        while current is not None and current != ids_elem:
            name = current.get("name")
            if name:
                path_parts.insert(0, name)
            current = parent_map.get(current)

        if not path_parts:
            return None

        return f"{ids_name}/{'/'.join(path_parts)}"

    def _should_skip_element(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> bool:
        """Check if element should be skipped."""
        # Build path to check against excluded patterns using parent map
        path_parts = []
        current = elem

        while current is not None and current != ids_elem:
            name = current.get("name")
            if name:
                path_parts.insert(0, name)
            current = parent_map.get(current)

        # Check if any part matches excluded patterns
        for pattern in self.excluded_patterns:
            if any(pattern in part for part in path_parts):
                return True

        # Enhanced GGD filtering
        if self.skip_ggd:
            # Check for GGD in path parts
            if any("ggd" in part.lower() for part in path_parts):
                return True

            # Check for GGD in element name specifically
            elem_name = elem.get("name", "").lower()
            if "ggd" in elem_name:
                return True

            # Check for grids_ggd patterns
            if any("grids_ggd" in part.lower() for part in path_parts):
                return True

        # Enhanced error node filtering - check for error in element name
        elem_name = elem.get("name", "").lower()
        if "error" in elem_name:
            return True

        return False

    def _extract_semantic_groups(
        self, paths: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Group paths by semantic similarity."""
        semantic_groups = {}

        # Group by common prefixes and physics concepts
        for path, metadata in paths.items():
            # Extract common patterns
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                group_key = "/".join(path_parts[:3])  # First 3 levels
                if group_key not in semantic_groups:
                    semantic_groups[group_key] = []
                semantic_groups[group_key].append(path)

        # Filter out single-item groups
        semantic_groups = {k: v for k, v in semantic_groups.items() if len(v) > 1}

        return semantic_groups

    def _calculate_max_depth(self, ids_elem: ET.Element) -> int:
        """Calculate maximum depth of IDS tree."""
        max_depth = 0

        def get_depth(elem: ET.Element, current_depth: int = 0) -> int:
            """Recursively calculate depth."""
            depth = current_depth
            for child in elem:
                if child.get("name"):
                    child_depth = get_depth(child, current_depth + 1)
                    depth = max(depth, child_depth)
            return depth

        max_depth = get_depth(ids_elem)
        return max_depth

    def _get_leaf_nodes(self, ids_elem: ET.Element) -> List[ET.Element]:
        """Get all leaf nodes (elements with no children)."""
        leaves = []

        for elem in ids_elem.findall(".//*[@name]"):
            if len(list(elem)) == 0:  # No children
                leaves.append(elem)

        return leaves

    def _infer_physics_domain(self, ids_name: str) -> str:
        """Infer physics domain from IDS name."""
        domain_mapping = {
            "core_profiles": PhysicsDomain.TRANSPORT.value,
            "equilibrium": PhysicsDomain.EQUILIBRIUM.value,
            "mhd": PhysicsDomain.MHD.value,
            "heating": PhysicsDomain.HEATING.value,
            "wall": PhysicsDomain.WALL.value,
        }

        return domain_mapping.get(ids_name.lower(), PhysicsDomain.GENERAL.value)

    def _calculate_documentation_coverage(self, ids_elem: ET.Element) -> float:
        """Calculate documentation coverage percentage."""
        total_elements = len(ids_elem.findall(".//*[@name]"))
        documented_elements = len(ids_elem.findall(".//*[@name][@documentation]"))

        if total_elements == 0:
            return 0.0

        return documented_elements / total_elements

    def _analyze_graph_structure(
        self, ids_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform graph analysis on the extracted IDS data."""
        # Create a data structure compatible with analyze_imas_graphs
        data_dict = {
            "ids_catalog": {
                ids_name: {"paths": data["paths"]}
                for ids_name, data in ids_data.items()
            },
            "metadata": {
                "build_time": "",  # Will be set during catalog generation
                "total_ids": len(ids_data),
            },
        }

        # Use the graph analyzer function
        return analyze_imas_graphs(data_dict)

    def _generate_catalog(
        self, ids_data: Dict[str, Dict[str, Any]], graph_data: Dict[str, Any]
    ) -> Path:
        """Generate high-level catalog file with graph statistics."""
        catalog_path = self.resolved_output_dir / "ids_catalog.json"

        # Create catalog metadata
        metadata = CatalogMetadata(
            version=self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            total_ids=len(ids_data),
            total_leaf_nodes=sum(
                data["ids_info"]["leaf_count"] for data in ids_data.values()
            ),
        )

        # Create IDS catalog entries
        catalog_entries = {}
        for ids_name, data in ids_data.items():
            catalog_entries[ids_name] = IdsInfo(**data["ids_info"])

        # Create catalog with graph data
        catalog_dict = {
            "metadata": metadata.model_dump(),
            "ids_catalog": {k: v.model_dump() for k, v in catalog_entries.items()},
        }

        # Merge graph statistics into the catalog
        catalog_dict.update(graph_data)

        # Write to file
        with open(catalog_path, "w", encoding="utf-8") as f:
            import json

            json.dump(catalog_dict, f, indent=2)

        return catalog_path

    def _generate_detailed_files(
        self, ids_data: Dict[str, Dict[str, Any]]
    ) -> List[Path]:
        """Generate detailed IDS files."""
        detailed_dir = self.resolved_output_dir / "detailed"
        detailed_dir.mkdir(exist_ok=True)

        paths = []
        for ids_name, data in ids_data.items():
            detailed_path = detailed_dir / f"{ids_name}.json"

            # Create detailed IDS structure
            detailed = IdsDetailed(
                ids_info=IdsInfo(**data["ids_info"]),
                coordinate_systems={
                    k: CoordinateSystem(**v)
                    for k, v in data["coordinate_systems"].items()
                },
                paths={k: DataPath(**v) for k, v in data["paths"].items()},
                semantic_groups=data["semantic_groups"],
            )
            # Write to file
            with open(detailed_path, "w", encoding="utf-8") as f:
                f.write(detailed.model_dump_json(indent=2))

            paths.append(detailed_path)

        return paths

    def _generate_relationships(self, ids_data: Dict[str, Dict[str, Any]]) -> Path:
        """Generate relationship graph."""
        rel_path = self.resolved_output_dir / "relationships.json"

        # Add basic unit families using proper UnitFamily objects
        unit_families = {}
        for ids_name, data in ids_data.items():
            for path, path_data in data["paths"].items():
                units = path_data.get("units")
                if units and units != "none":
                    if units not in unit_families:
                        unit_families[units] = UnitFamily(
                            base_unit=units,
                            paths_using=[],
                            conversion_factors={},
                        )
                    unit_families[units].paths_using.append(path)

        # Create metadata for relationships
        metadata = CatalogMetadata(
            version=self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            total_ids=len(ids_data),
            total_leaf_nodes=sum(
                data["ids_info"]["leaf_count"] for data in ids_data.values()
            ),
            total_relationships=len(unit_families),  # Count relationships
        )

        # Create basic relationships structure
        relationships = Relationships(metadata=metadata)
        relationships.unit_families = unit_families

        # Write to file
        with open(rel_path, "w", encoding="utf-8") as f:
            f.write(relationships.model_dump_json(indent=2))

        return rel_path
