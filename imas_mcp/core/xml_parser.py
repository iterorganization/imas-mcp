"""Refactored XML parser using composable extractors."""

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
)
from .extractors import (
    ExtractorContext,
    MetadataExtractor,
    LifecycleExtractor,
    PhysicsExtractor,
    ValidationExtractor,
    PathExtractor,
    SemanticExtractor,
    RelationshipExtractor,
    CoordinateExtractor,
)


@dataclass
class DataDictionaryTransformer:
    """Refactored transformer using composable extractors with performance optimizations."""

    output_dir: Optional[Path] = None
    dd_accessor: Optional[ImasDataDictionaryAccessor] = None
    ids_set: Optional[Set[str]] = None

    # Processing configuration
    excluded_patterns: Set[str] = field(
        default_factory=lambda: {"ids_properties", "code"}
    )
    skip_ggd: bool = True

    def __post_init__(self):
        """Initialize the transformer with performance optimizations."""
        if self.dd_accessor is None:
            self.dd_accessor = ImasDataDictionaryAccessor()

        if self.output_dir is None:
            self.output_dir = (
                Path(__file__).resolve().parent.parent / "resources" / "json_data"
            )

        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache XML tree
        self._tree = self.dd_accessor.get_xml_tree()
        self._root = self._tree.getroot()

        # Performance optimization: Build global parent map once
        self._global_parent_map = self._build_global_parent_map()

        # Performance optimization: Pre-cache element lookups
        self._element_cache = {}
        self._path_cache = {}

    def _build_global_parent_map(self) -> Dict[ET.Element, ET.Element]:
        """Build parent map for entire XML tree once for performance."""
        if self._root is None:
            return {}
        return {c: p for p in self._root.iter() for c in p}

    def _get_cached_elements_by_name(
        self, ids_elem: ET.Element, ids_name: str
    ) -> List[ET.Element]:
        """Get all named elements for an IDS with caching."""
        cache_key = f"{ids_name}_named_elements"
        if cache_key not in self._element_cache:
            # Use iter() instead of findall() for better performance
            elements = [elem for elem in ids_elem.iter() if elem.get("name")]
            self._element_cache[cache_key] = elements
        return self._element_cache[cache_key]

    def _get_cached_elements_by_attribute(
        self, ids_elem: ET.Element, ids_name: str, attr: str
    ) -> List[ET.Element]:
        """Get all elements with specific attribute for an IDS with caching."""
        cache_key = f"{ids_name}_{attr}_elements"
        if cache_key not in self._element_cache:
            # Use iter() instead of findall() for better performance
            elements = [elem for elem in ids_elem.iter() if elem.get(attr)]
            self._element_cache[cache_key] = elements
        return self._element_cache[cache_key]

    @property
    def resolved_output_dir(self) -> Path:
        """Get the resolved output directory."""
        assert self.output_dir is not None
        return self.output_dir

    def transform_complete(self) -> TransformationOutputs:
        """Transform XML to complete JSON structure using composable extractors."""
        if self._root is None:
            raise ValueError("XML root is None")

        # Extract all IDS information using new architecture
        ids_data = self._extract_ids_data(self._root)

        # Perform graph analysis
        graph_data = self._analyze_graph_structure(ids_data)

        # Generate outputs
        catalog_path = self._generate_catalog(ids_data, graph_data)
        detailed_paths = self._generate_detailed_files(ids_data)
        relationships_path = self._generate_relationships(ids_data)

        return TransformationOutputs(
            catalog=catalog_path,
            detailed=detailed_paths,
            relationships=relationships_path,
        )

    def _extract_ids_data(self, root: ET.Element) -> Dict[str, Dict[str, Any]]:
        """Extract IDS data using composable extractors with performance optimizations."""
        ids_data = {}

        for ids_elem in root.findall(".//IDS[@name]"):
            ids_name = ids_elem.get("name")
            if not ids_name:
                continue

            if self.ids_set is not None and ids_name not in self.ids_set:
                continue

            print(f"Processing IDS: {ids_name}")

            try:
                # Create context for this IDS with cached parent map
                context = ExtractorContext(
                    dd_accessor=self.dd_accessor,  # type: ignore
                    root=root,
                    ids_elem=ids_elem,
                    ids_name=ids_name,
                    parent_map=self._global_parent_map,  # Use pre-built parent map
                    excluded_patterns=self.excluded_patterns,
                    skip_ggd=self.skip_ggd,
                )

                # Extract IDS-level information
                ids_info = self._extract_ids_info(ids_elem, ids_name, context)
                coordinate_systems = self._extract_coordinate_systems(ids_elem, context)
                paths = self._extract_paths(ids_elem, ids_name, context)
                semantic_groups = self._extract_semantic_groups(paths, context)

                ids_data[ids_name] = {
                    "ids_info": ids_info,
                    "coordinate_systems": coordinate_systems,
                    "paths": paths,
                    "semantic_groups": semantic_groups,
                }

            except Exception as e:
                print(f"Error processing IDS {ids_name}: {e}")
                continue

        return ids_data

    def _extract_ids_info(
        self, ids_elem: ET.Element, ids_name: str, context: ExtractorContext
    ) -> Dict[str, Any]:
        """Extract IDS-level information with optimized element access."""
        # Get cached elements instead of multiple findall() calls
        named_elements = self._get_cached_elements_by_name(ids_elem, ids_name)
        documented_elements = [
            elem for elem in named_elements if elem.get("documentation")
        ]

        return {
            "name": ids_name,
            "description": ids_elem.get("documentation", ""),
            "version": self.dd_accessor.get_version().public  # type: ignore
            if self.dd_accessor
            else "unknown",
            "physics_domain": self._infer_physics_domain(ids_name),
            "max_depth": self._calculate_max_depth(ids_elem),
            "leaf_count": len(
                [elem for elem in named_elements if len(list(elem)) == 0]
            ),
            "documentation_coverage": len(documented_elements) / len(named_elements)
            if named_elements
            else 0.0,
        }

    def _extract_coordinate_systems(
        self, ids_elem: ET.Element, context: ExtractorContext
    ) -> Dict[str, Dict[str, Any]]:
        """Extract coordinate systems using CoordinateExtractor."""
        extractor = CoordinateExtractor(context)
        return extractor.extract_coordinate_systems(ids_elem)

    def _extract_paths(
        self, ids_elem: ET.Element, ids_name: str, context: ExtractorContext
    ) -> Dict[str, Dict[str, Any]]:
        """Extract paths using composable extractors with performance optimizations."""
        paths = {}

        # Set up extractors once
        extractors = [
            MetadataExtractor(context),
            LifecycleExtractor(context),
            PhysicsExtractor(context),
            ValidationExtractor(context),
            PathExtractor(context),
            RelationshipExtractor(context),
        ]

        # Get cached named elements instead of findall()
        named_elements = self._get_cached_elements_by_name(ids_elem, ids_name)

        # Process all elements with names
        for elem in named_elements:
            # Skip if element should be filtered
            if self._should_skip_element(elem, ids_elem, context.parent_map):
                continue

            # Extract path with caching
            path = self._build_element_path(
                elem, ids_elem, ids_name, context.parent_map
            )
            if not path:
                continue

            # Use individual extractors
            try:
                element_metadata = {}
                for extractor in extractors:
                    metadata = extractor.extract(elem)
                    element_metadata.update(metadata)

                # Ensure path is set
                element_metadata["path"] = path

                paths[path] = element_metadata

            except Exception as e:
                print(f"Error extracting metadata for {path}: {e}")
                continue

        return paths

    def _extract_semantic_groups(
        self, paths: Dict[str, Dict[str, Any]], context: ExtractorContext
    ) -> Dict[str, List[str]]:
        """Extract semantic groups using SemanticExtractor."""
        extractor = SemanticExtractor(context)
        return extractor.extract_semantic_groups(paths)

    def _should_skip_element(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> bool:
        """Optimized element filtering with minimal string operations."""
        name = elem.get("name", "")

        # Fast check for excluded patterns (avoid .lower() when possible)
        for pattern in self.excluded_patterns:
            if pattern in name:
                return True

        # Only do lowercase conversion if needed
        if self.skip_ggd and "ggd" in name.lower():
            return True

        return False

    def _build_element_path(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Optional[str]:
        """Build full path for element with caching."""
        # Use element ID as cache key for path building
        elem_id = id(elem)
        cache_key = f"{ids_name}_{elem_id}_path"

        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        path_parts = []
        current = elem

        while current is not None and current != ids_elem:
            name = current.get("name")
            if name:
                path_parts.append(name)
            current = parent_map.get(current)

        if not path_parts:
            self._path_cache[cache_key] = None
            return None

        path = f"{ids_name}/{'/'.join(reversed(path_parts))}"
        self._path_cache[cache_key] = path
        return path

    # Keep existing helper methods for IDS-level analysis
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

    def _calculate_max_depth(self, ids_elem: ET.Element) -> int:
        """Calculate maximum depth using single traversal."""
        max_depth = 0

        def calculate_depth(elem: ET.Element, current_depth: int = 0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            for child in elem:
                calculate_depth(child, current_depth + 1)

        calculate_depth(ids_elem)
        return max_depth

    def _get_leaf_nodes(self, ids_elem: ET.Element) -> List[ET.Element]:
        """Get all leaf nodes using optimized traversal."""
        leaves = []
        for elem in ids_elem.iter():  # Use iter() instead of findall()
            if elem.get("name") and len(list(elem)) == 0:  # No children
                leaves.append(elem)
        return leaves

    def _calculate_documentation_coverage(self, ids_elem: ET.Element) -> float:
        """Calculate documentation coverage using optimized traversal."""
        total_elements = 0
        documented_elements = 0

        for elem in ids_elem.iter():
            if elem.get("name"):
                total_elements += 1
                if elem.get("documentation"):
                    documented_elements += 1

        if total_elements == 0:
            return 0.0

        return documented_elements / total_elements

    # Keep existing graph analysis and output generation methods
    def _analyze_graph_structure(
        self, ids_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform graph analysis on extracted data."""
        data_dict = {
            "ids_catalog": {
                ids_name: {"paths": data["paths"]}
                for ids_name, data in ids_data.items()
            },
            "metadata": {
                "build_time": "",
                "total_ids": len(ids_data),
            },
        }
        return analyze_imas_graphs(data_dict)

    def _generate_catalog(
        self, ids_data: Dict[str, Dict[str, Any]], graph_data: Dict[str, Any]
    ) -> Path:
        """Generate catalog file."""
        catalog_path = self.resolved_output_dir / "ids_catalog.json"

        metadata = CatalogMetadata(
            version=self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            total_ids=len(ids_data),
            total_leaf_nodes=sum(
                data["ids_info"]["leaf_count"] for data in ids_data.values()
            ),
        )

        catalog_entries = {}
        for ids_name, data in ids_data.items():
            catalog_entries[ids_name] = {
                "name": ids_name,
                "description": data["ids_info"]["description"],
                "path_count": len(data["paths"]),
                "physics_domain": data["ids_info"]["physics_domain"],
            }

        catalog_dict = {
            "metadata": metadata.model_dump(),
            "ids_catalog": catalog_entries,
        }
        catalog_dict.update(graph_data)

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

            detailed = IdsDetailed(
                ids_info=IdsInfo(**data["ids_info"]),
                coordinate_systems={
                    k: CoordinateSystem(**v)
                    for k, v in data["coordinate_systems"].items()
                },
                paths={k: DataPath(**v) for k, v in data["paths"].items()},
                semantic_groups=data["semantic_groups"],
            )

            with open(detailed_path, "w", encoding="utf-8") as f:
                f.write(detailed.model_dump_json(indent=2))

            paths.append(detailed_path)

        return paths

    def _generate_relationships(self, ids_data: Dict[str, Dict[str, Any]]) -> Path:
        """Generate relationships file."""
        rel_path = self.resolved_output_dir / "relationships.json"

        metadata = CatalogMetadata(
            version=self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            total_ids=len(ids_data),
            total_leaf_nodes=sum(
                data["ids_info"]["leaf_count"] for data in ids_data.values()
            ),
            total_relationships=0,
        )

        relationships = Relationships(metadata=metadata)

        with open(rel_path, "w", encoding="utf-8") as f:
            f.write(relationships.model_dump_json(indent=2))

        return rel_path
