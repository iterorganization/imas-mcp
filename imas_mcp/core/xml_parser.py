"""Refactored XML parser using composable extractors."""

import json
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from imas_mcp.core.data_model import (
    CatalogMetadata,
    CoordinateSystem,
    IdentifierCatalog,
    IdentifierCatalogSchema,
    IdentifierPath,
    IdsDetailed,
    IdsInfo,
    IdsNode,
    TransformationOutputs,
    UsageExample,
)
from imas_mcp.core.extractors import (
    CoordinateExtractor,
    ExtractorContext,
    IdentifierExtractor,
    LifecycleExtractor,
    MetadataExtractor,
    PathExtractor,
    PhysicsExtractor,
    SemanticExtractor,
    ValidationExtractor,
)
from imas_mcp.core.physics_categorization import physics_categorizer
from imas_mcp.core.progress_monitor import create_progress_monitor
from imas_mcp.dd_accessor import ImasDataDictionaryAccessor
from imas_mcp.graph_analyzer import analyze_imas_graphs
from imas_mcp.structure.structure_analyzer import StructureAnalyzer


@dataclass
class DataDictionaryTransformer:
    """Refactored transformer using composable extractors with performance optimizations."""

    output_dir: Path | None = None
    dd_accessor: ImasDataDictionaryAccessor | None = None
    ids_set: set[str] | None = None

    # Processing configuration
    excluded_patterns: set[str] = field(
        default_factory=lambda: {"ids_properties", "code"}
    )
    skip_ggd: bool = True
    skip_error_fields: bool = True
    use_rich: bool | None = None  # Auto-detect if None

    def __post_init__(self):
        """Initialize the transformer with performance optimizations."""
        if self.dd_accessor is None:
            self.dd_accessor = ImasDataDictionaryAccessor()

        if self.output_dir is None:
            schema_resource = resources.files("imas_mcp") / "resources" / "schemas"
            self.output_dir = Path(str(schema_resource))

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

    def _build_global_parent_map(self) -> dict[ET.Element, ET.Element]:
        """Build parent map for entire XML tree once for performance."""
        if self._root is None:
            return {}
        return {c: p for p in self._root.iter() for c in p}

    def _get_cached_elements_by_name(
        self, ids_elem: ET.Element, ids_name: str
    ) -> list[ET.Element]:
        """Get all named elements for an IDS with caching."""
        cache_key = f"{ids_name}_named_elements"
        if cache_key not in self._element_cache:
            # Use iter() instead of findall() for better performance
            elements = [elem for elem in ids_elem.iter() if elem.get("name")]
            self._element_cache[cache_key] = elements
        return self._element_cache[cache_key]

    def _get_cached_elements_by_attribute(
        self, ids_elem: ET.Element, ids_name: str, attr: str
    ) -> list[ET.Element]:
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

    def build(self) -> TransformationOutputs:
        """Build complete JSON structure using composable extractors."""
        if self._root is None:
            raise ValueError("XML root is None")

        # Extract all IDS information using new architecture
        ids_data = self._extract_ids_data(self._root)

        # Perform graph analysis
        graph_data = self._analyze_graph_structure(ids_data)

        # Generate outputs
        catalog_path = self._generate_catalog(ids_data, graph_data)
        detailed_paths = self._generate_detailed_files(ids_data)

        # Generate identifier catalog
        identifier_catalog_path = self._generate_identifier_catalog(ids_data)

        # Generate enhanced structure analysis
        self._generate_structure_analysis(ids_data)

        return TransformationOutputs(
            catalog=catalog_path,
            detailed=detailed_paths,
            identifier_catalog=identifier_catalog_path,
        )

    def _extract_ids_data(self, root: ET.Element) -> dict[str, dict[str, Any]]:
        """Extract IDS data using composable extractors with performance optimizations."""
        ids_data = {}

        # Collect all IDS names first for progress monitoring
        all_ids_elements = []
        for ids_elem in root.findall(".//IDS[@name]"):
            ids_name = ids_elem.get("name")
            if not ids_name:
                continue
            if self.ids_set is not None and ids_name not in self.ids_set:
                continue
            all_ids_elements.append((ids_elem, ids_name))

        # Extract just the names for progress monitoring
        ids_names = [name for _, name in all_ids_elements]

        # Create progress monitor
        logger = logging.getLogger(__name__)
        progress = create_progress_monitor(
            use_rich=self.use_rich, logger=logger, item_names=ids_names
        )
        progress.start_processing(ids_names, "Processing IDS")

        for ids_elem, ids_name in all_ids_elements:
            try:
                # Set current item before processing
                progress.set_current_item(ids_name)

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

                # Update progress after processing
                progress.update_progress(ids_name)

            except Exception as e:
                # Update progress with error
                progress.update_progress(ids_name, error=str(e))
                continue

        # Finish progress monitoring
        progress.finish_processing()

        return ids_data

    def _extract_ids_info(
        self, ids_elem: ET.Element, ids_name: str, context: ExtractorContext
    ) -> dict[str, Any]:
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
    ) -> dict[str, dict[str, Any]]:
        """Extract coordinate systems using CoordinateExtractor."""
        extractor = CoordinateExtractor(context)
        return extractor.extract_coordinate_systems(ids_elem)

    def _extract_paths(
        self, ids_elem: ET.Element, ids_name: str, context: ExtractorContext
    ) -> dict[str, dict[str, Any]]:
        """Extract paths using composable extractors with performance optimizations."""
        paths = {}

        # Set up extractors once
        extractors = [
            MetadataExtractor(context),
            LifecycleExtractor(context),
            PhysicsExtractor(context),
            ValidationExtractor(context),
            PathExtractor(context),
            IdentifierExtractor(context),
        ]

        # Get cached named elements instead of findall()
        named_elements = self._get_cached_elements_by_name(ids_elem, ids_name)

        # Process all elements with names
        for elem in named_elements:
            # Extract path with caching first
            path = self._build_element_path(
                elem, ids_elem, ids_name, context.parent_map
            )
            if not path:
                continue

            # Skip if element should be filtered
            if self._should_skip_element(elem, ids_elem, context.parent_map):
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
        self, paths: dict[str, dict[str, Any]], context: ExtractorContext
    ) -> dict[str, list[str]]:
        """Extract semantic groups using SemanticExtractor."""
        extractor = SemanticExtractor(context)
        return extractor.extract_semantic_groups(paths)

    def _should_skip_element(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        parent_map: dict[ET.Element, ET.Element],
    ) -> bool:
        """Comprehensive element filtering to exclude unwanted entries."""
        name = elem.get("name", "")
        if not name:
            return True

        # Build path to check for filtering
        path = self._build_element_path(elem, ids_elem, "", parent_map)
        if not path:
            return True

        # Fast check for excluded patterns
        for pattern in self.excluded_patterns:
            if pattern in name or pattern in path:
                return True

        # Skip GGD entries (Grid Geometry Description) - check for various GGD patterns
        if self.skip_ggd and (
            "ggd" in name.lower()
            or "/ggd/" in path.lower()
            or "grids_ggd" in path.lower()
            or path.lower().startswith("grids_ggd")
            or "/grids_ggd/" in path.lower()
        ):
            return True

        # Skip error fields (error_upper, error_lower, error_index, etc.)
        if self.skip_error_fields and (
            "_error_" in name
            or "_error_" in path
            or name.endswith("_error_upper")
            or name.endswith("_error_lower")
            or name.endswith("_error_index")
            or "error_upper" in name
            or "error_lower" in name
            or "error_index" in name
        ):
            return True

        return False

    def _build_element_path(
        self,
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: dict[ET.Element, ET.Element],
    ) -> str | None:
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
        """Infer physics domain from IDS name using enhanced categorization."""
        domain = physics_categorizer.get_domain_for_ids(ids_name)
        return domain.value

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

    def _get_leaf_nodes(self, ids_elem: ET.Element) -> list[ET.Element]:
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
        self, ids_data: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
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
        self, ids_data: dict[str, dict[str, Any]], graph_data: dict[str, Any]
    ) -> Path:
        """Generate catalog file."""
        catalog_path = self.resolved_output_dir / "ids_catalog.json"

        # Calculate total relationships from the paths data
        total_relationships = 0
        total_paths = 0
        for data in ids_data.values():
            paths = data.get("paths", {})
            total_paths += len(paths)  # Add the path count for this IDS
            for path_data in paths.values():
                relationships = path_data.get("relationships", {})
                for _category, rel_list in relationships.items():
                    if isinstance(rel_list, list):
                        total_relationships += len(rel_list)

        metadata = CatalogMetadata(
            version=self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            total_ids=len(ids_data),
            total_leaf_nodes=sum(
                data["ids_info"]["leaf_count"] for data in ids_data.values()
            ),
            total_paths=total_paths,
            total_relationships=total_relationships,
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
            json.dump(catalog_dict, f, indent=2)

        return catalog_path

    def _generate_detailed_files(
        self, ids_data: dict[str, dict[str, Any]]
    ) -> list[Path]:
        """Generate detailed IDS files."""
        detailed_dir = self.resolved_output_dir / "detailed"

        # Clear the detailed directory first to remove any orphaned files
        # This prevents files from previous builds (e.g., removed IDS) from remaining
        if detailed_dir.exists():
            import shutil

            shutil.rmtree(detailed_dir)

        detailed_dir.mkdir(exist_ok=True)

        paths = []
        for ids_name, data in ids_data.items():
            detailed_path = detailed_dir / f"{ids_name}.json"

            # Convert paths to DataPath objects, handling relationships properly
            data_paths = {}
            for path_key, path_data in data["paths"].items():
                # Handle usage_examples conversion
                usage_examples_data = path_data.get("usage_examples", [])
                if usage_examples_data and isinstance(usage_examples_data, list):
                    usage_examples = []
                    for example in usage_examples_data:
                        if isinstance(example, dict):
                            usage_examples.append(UsageExample(**example))
                        else:
                            # Handle string format or other formats
                            usage_examples.append(
                                UsageExample(
                                    scenario="General usage",
                                    code=str(example),
                                    notes="",
                                )
                            )
                    path_data = path_data.copy()
                    path_data["usage_examples"] = usage_examples

                # Create DataPath object
                data_paths[path_key] = IdsNode(**path_data)

            detailed = IdsDetailed(
                ids_info=IdsInfo(**data["ids_info"]),
                coordinate_systems={
                    k: CoordinateSystem(**v)
                    for k, v in data["coordinate_systems"].items()
                },
                paths=data_paths,
                semantic_groups=data["semantic_groups"],
            )

            with open(detailed_path, "w", encoding="utf-8") as f:
                f.write(detailed.model_dump_json(indent=2))

            paths.append(detailed_path)

        return paths

    def _generate_identifier_catalog(self, ids_data: dict[str, dict[str, Any]]) -> Path:
        """Generate identifier catalog from extracted IDS data."""
        identifier_paths = []
        schema_groups = {}

        # Extract all paths with identifier schemas
        for ids_name, data in ids_data.items():
            paths = data.get("paths", {})
            for path, path_data in paths.items():
                identifier_schema = path_data.get("identifier_schema")
                if identifier_schema and identifier_schema.get("options"):
                    # Extract physics domain if available
                    physics_domain = None
                    if "physics_context" in path_data and path_data["physics_context"]:
                        physics_domain = path_data["physics_context"].get("domain")

                    identifier_path = IdentifierPath(
                        path=path,
                        ids_name=ids_name,
                        schema_name=self._extract_schema_name(
                            identifier_schema.get("schema_path", "")
                        ),
                        description=(path_data.get("documentation", "") or "")[
                            :200
                        ],  # First 200 chars
                        option_count=len(identifier_schema.get("options", [])),
                        physics_domain=physics_domain,
                    )
                    identifier_paths.append(identifier_path)

                    # Group by schema path
                    schema_path = identifier_schema.get("schema_path", "unknown")
                    if schema_path not in schema_groups:
                        schema_groups[schema_path] = []
                    schema_groups[schema_path].append(
                        (identifier_path, identifier_schema)
                    )

        # Build schemas
        schemas = {}
        for schema_path, path_schema_list in schema_groups.items():
            if not path_schema_list:
                continue

            first_path, first_schema = path_schema_list[0]
            schema_name = self._extract_schema_name(schema_path)

            # Collect physics domains
            physics_domains = list(
                {
                    path.physics_domain
                    for path, _ in path_schema_list
                    if path.physics_domain
                }
            )

            # Calculate branching complexity (entropy)
            option_count = len(first_schema.get("options", []))
            branching_complexity = math.log2(option_count) if option_count > 1 else 0.0

            schema = IdentifierCatalogSchema(
                schema_name=schema_name,
                schema_path=schema_path,
                description=first_schema.get("documentation", ""),
                total_options=option_count,
                options=first_schema.get("options", []),
                usage_count=len(path_schema_list),
                usage_paths=[path.path for path, _ in path_schema_list],
                physics_domains=physics_domains,
                branching_complexity=branching_complexity,
            )
            schemas[schema_name] = schema

        # Build cross-references
        cross_references = {}
        for schema_name, schema in schemas.items():
            related = []
            for other_name, other_schema in schemas.items():
                if other_name == schema_name:
                    continue
                # Check for physics domain overlap
                if any(
                    domain in other_schema.physics_domains
                    for domain in schema.physics_domains
                ):
                    related.append(other_name)
                # Check for name similarity
                if any(
                    word in other_name.lower() for word in schema_name.lower().split()
                ):
                    if other_name not in related:
                        related.append(other_name)
            cross_references[schema_name] = related

        # Build physics mapping
        physics_mapping = {}
        for schema_name, schema in schemas.items():
            for domain in schema.physics_domains:
                if domain not in physics_mapping:
                    physics_mapping[domain] = []
                physics_mapping[domain].append(schema_name)

        # Build branching analytics
        total_enumeration_space = sum(s.total_options for s in schemas.values())
        complexity_buckets = {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
            "very_complex": 0,
        }

        for schema in schemas.values():
            if schema.branching_complexity < 2:
                complexity_buckets["simple"] += 1
            elif schema.branching_complexity < 4:
                complexity_buckets["moderate"] += 1
            elif schema.branching_complexity < 6:
                complexity_buckets["complex"] += 1
            else:
                complexity_buckets["very_complex"] += 1

        branching_analytics = {
            "total_schemas": len(schemas),
            "total_paths": len(identifier_paths),
            "complexity_distribution": complexity_buckets,
            "most_complex_schemas": [
                {
                    "name": s.schema_name,
                    "complexity": s.branching_complexity,
                    "options": s.total_options,
                }
                for s in sorted(
                    schemas.values(), key=lambda x: x.branching_complexity, reverse=True
                )[:10]
            ],
            "most_used_schemas": [
                {
                    "name": s.schema_name,
                    "usage_count": s.usage_count,
                    "paths": len(s.usage_paths),
                }
                for s in sorted(
                    schemas.values(), key=lambda x: x.usage_count, reverse=True
                )[:10]
            ],
            "option_statistics": {
                "min_options": min(s.total_options for s in schemas.values())
                if schemas
                else 0,
                "max_options": max(s.total_options for s in schemas.values())
                if schemas
                else 0,
                "avg_options": sum(s.total_options for s in schemas.values())
                / len(schemas)
                if schemas
                else 0,
                "total_enumeration_space": total_enumeration_space,
            },
        }

        # Group paths by IDS
        paths_by_ids = {}
        for path in identifier_paths:
            if path.ids_name not in paths_by_ids:
                paths_by_ids[path.ids_name] = []
            paths_by_ids[path.ids_name].append(path)

        # Create catalog
        metadata = CatalogMetadata(
            version=self.dd_accessor.get_version().public
            if self.dd_accessor
            else "unknown",
            total_ids=len(paths_by_ids),
            total_leaf_nodes=len(identifier_paths),
            total_relationships=sum(len(refs) for refs in cross_references.values()),
        )

        catalog = IdentifierCatalog(
            metadata=metadata,
            schemas=schemas,
            paths_by_ids=paths_by_ids,
            cross_references=cross_references,
            physics_mapping=physics_mapping,
            branching_analytics=branching_analytics,
        )

        # Save catalog
        catalog_path = self.resolved_output_dir / "identifier_catalog.json"
        with open(catalog_path, "w", encoding="utf-8") as f:
            f.write(catalog.model_dump_json(indent=2))

        return catalog_path

    def _extract_schema_name(self, schema_path: str) -> str:
        """Extract clean schema name from path."""
        if not schema_path:
            return "unknown"
        return (
            Path(schema_path).stem.replace("_identifier", "").replace("_", " ").title()
        )

    def _generate_structure_analysis(self, ids_data: dict[str, dict[str, Any]]) -> None:
        """Generate enhanced structure analysis for all IDS."""
        try:
            structure_analyzer = StructureAnalyzer(self.resolved_output_dir)
            structure_analyzer.analyze_all_ids(ids_data)
            logger = logging.getLogger(__name__)
            logger.info("Enhanced structure analysis completed")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to generate structure analysis: {e}")
