"""
Mermaid graph generator for IMAS IDS structure visualization.

This module generates Mermaid diagrams representing the hierarchical structure
of IMAS IDS for LLM consumption. The diagrams are optimized for AI understanding
of data relationships and navigation patterns.
"""

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MermaidCache:
    """Cache for generated Mermaid graphs with metadata."""

    graph_count: int = 0
    created_at: float = field(default_factory=time.time)
    ids_set: set | None = None  # IDS set used for this cache
    source_content_hash: str = ""  # Hash of source schema content
    source_max_mtime: float = 0.0  # Maximum modification time of source files

    def is_valid(
        self,
        current_graph_count: int,
        current_ids_set: set | None = None,
        source_data_dir: Path | None = None,
    ) -> bool:
        """Check if cache is valid for current state."""
        return self.validate_with_reason(
            current_graph_count, current_ids_set, source_data_dir
        )[0]

    def validate_with_reason(
        self,
        current_graph_count: int,
        current_ids_set: set | None = None,
        source_data_dir: Path | None = None,
    ) -> tuple[bool, str]:
        """Check if cache is valid and return detailed reason if invalid."""
        # Basic validation checks
        if self.graph_count != current_graph_count:
            return (
                False,
                f"Graph count mismatch: cached={self.graph_count}, current={current_graph_count}",
            )

        if self.ids_set != current_ids_set:
            return (
                False,
                f"IDS set mismatch: cached={self.ids_set}, current={current_ids_set}",
            )

        # Enhanced validation with source file checking
        if source_data_dir is not None:
            # Check if any source files are modified after cache creation
            modified_files = self._get_modified_source_files(source_data_dir)
            if modified_files:
                # Check if catalog was modified
                if "ids_catalog.json" in modified_files:
                    return (
                        False,
                        "Schema catalog has been modified following cache generation",
                    )

                # Check if detailed schema files were modified
                detailed_count = sum(
                    1 for f in modified_files if f.startswith("detailed/")
                )
                if detailed_count > 0:
                    return (
                        False,
                        f"Schema files have been modified following cache generation ({detailed_count} files updated)",
                    )

            # Check source content hash if available
            if self.source_content_hash:
                current_hash = self._compute_source_content_hash(source_data_dir)
                if current_hash != self.source_content_hash:
                    return False, "Source content has changed (hash mismatch)"

        return True, "Cache is valid"

    def _get_modified_source_files(self, source_data_dir: Path) -> list[str]:
        """Get list of source files that have been modified after cache creation."""
        modified_files = []
        try:
            # Check catalog file
            catalog_path = source_data_dir / "ids_catalog.json"
            if catalog_path.exists() and catalog_path.stat().st_mtime > self.created_at:
                modified_files.append("ids_catalog.json")

            # Check detailed files
            detailed_dir = source_data_dir / "detailed"
            if detailed_dir.exists():
                for json_file in detailed_dir.glob("*.json"):
                    if json_file.stat().st_mtime > self.created_at:
                        modified_files.append(f"detailed/{json_file.name}")

        except Exception:
            # If we can't check files, assume they might have been modified
            pass

        return modified_files

    def _compute_source_content_hash(self, source_data_dir: Path) -> str:
        """Compute hash of source schema directory content."""
        hash_data = str(source_data_dir.resolve())

        # Include IDS set in hash for proper cache isolation
        if self.ids_set:
            hash_data += f"_ids_set_{sorted(self.ids_set)}"

        return hashlib.md5(hash_data.encode()).hexdigest()[:16]


class MermaidGraphGenerator:
    """Generate Mermaid diagrams for IMAS IDS structure visualization with caching."""

    def __init__(self, output_dir: Path):
        """Initialize the Mermaid generator with output directory."""
        self.output_dir = output_dir
        self.mermaid_dir = output_dir / "mermaid"
        self.detailed_dir = self.mermaid_dir / "detailed"
        self._cache_path: Path | None = None
        self._cache: MermaidCache | None = None

        # Create directory structure similar to schemas
        self.mermaid_dir.mkdir(exist_ok=True)
        self.detailed_dir.mkdir(exist_ok=True)

    def build(self, ids_set: set[str] | None = None, force: bool = False) -> None:
        """Build all Mermaid graphs from existing schema data with caching.

        Args:
            ids_set: Optional set of IDS names to filter. If None, builds for all available IDS.
            force: If True, force rebuild even if cache is valid.
        """
        import json

        # Set up cache path
        self._setup_cache_path(ids_set)

        # Load catalog and detailed schemas from the schemas directory
        schemas_dir = self.output_dir / "schemas"
        catalog_file = schemas_dir / "ids_catalog.json"
        detailed_dir = schemas_dir / "detailed"

        if not catalog_file.exists():
            logger.warning("IDS catalog not found, skipping Mermaid graph generation")
            return

        if not detailed_dir.exists():
            logger.warning(
                "Detailed schemas directory not found, skipping Mermaid graph generation"
            )
            return

        # Load catalog and extract IDS names
        with open(catalog_file, encoding="utf-8") as f:
            catalog_data = json.load(f)

        # Get IDS names from the ids_catalog section
        ids_catalog = catalog_data.get("ids_catalog", {})
        available_ids = set(ids_catalog.keys())

        # Filter by requested IDS set if provided
        filtered_ids = available_ids
        if ids_set:
            filtered_ids &= ids_set

        if not filtered_ids:
            logger.warning("No matching IDS found for graph generation")
            return

        # Calculate expected graph count
        expected_graph_count = self._calculate_expected_graph_count(filtered_ids)

        # Check if we need to build (cache validation)
        should_build = force or self._should_rebuild_graphs(
            expected_graph_count, ids_set, schemas_dir
        )

        if not should_build:
            logger.info("Mermaid graphs are up to date (cached)")
            return

        # Load detailed schema data
        detailed_data = {}
        for ids_name in filtered_ids:
            schema_file = detailed_dir / f"{ids_name}.json"
            if schema_file.exists():
                with open(schema_file, encoding="utf-8") as f:
                    detailed_data[ids_name] = json.load(f)

        # Generate all graphs
        self.generate_all_graphs(detailed_data)

        # Update cache
        self._update_cache(expected_graph_count, ids_set, schemas_dir)
        logger.info(f"Built Mermaid graphs for {len(detailed_data)} IDS")

    def _setup_cache_path(self, ids_set: set[str] | None) -> None:
        """Set up cache file path based on configuration."""
        # Generate cache filename based on ids_set
        cache_filename = "mermaid_cache"
        if ids_set:
            ids_hash = hashlib.md5(str(sorted(ids_set)).encode()).hexdigest()[:8]
            cache_filename += f"_{ids_hash}"
        cache_filename += ".pkl"

        # Create cache directory
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        self._cache_path = cache_dir / cache_filename

    def _should_rebuild_graphs(
        self, expected_count: int, ids_set: set[str] | None, schemas_dir: Path
    ) -> bool:
        """Check if graphs need to be rebuilt based on cache validation."""
        # Check if cache file exists
        if not self._cache_path or not self._cache_path.exists():
            logger.debug("No cache file found, will build graphs")
            return True

        # Load and validate cache
        try:
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)

            if not isinstance(self._cache, MermaidCache):
                logger.debug("Invalid cache file format, will rebuild graphs")
                return True

            # Validate cache
            is_valid, reason = self._cache.validate_with_reason(
                expected_count, ids_set, schemas_dir
            )

            if not is_valid:
                logger.info(f"Cache invalid: {reason}")
                return True

            return False

        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return True

    def _update_cache(
        self, graph_count: int, ids_set: set[str] | None, schemas_dir: Path
    ) -> None:
        """Update cache with current state."""
        if not self._cache_path:
            return

        try:
            cache = MermaidCache(
                graph_count=graph_count,
                ids_set=ids_set,
                source_content_hash=self._compute_source_hash(schemas_dir),
                source_max_mtime=self._get_max_source_mtime(schemas_dir),
            )

            with open(self._cache_path, "wb") as f:
                pickle.dump(cache, f)

        except Exception as e:
            logger.warning(f"Failed to update cache: {e}")

    def _calculate_expected_graph_count(self, ids_names: set[str]) -> int:
        """Calculate expected number of graph files."""
        # For each IDS: hierarchy + complexity + (optional physics)
        # Plus 1 overview graph
        return len(ids_names) * 2 + 1  # Simplified count, adjust based on actual logic

    def _compute_source_hash(self, schemas_dir: Path) -> str:
        """Compute hash of source schema content."""
        hash_obj = hashlib.md5()

        # Hash catalog file
        catalog_file = schemas_dir / "ids_catalog.json"
        if catalog_file.exists():
            hash_obj.update(catalog_file.read_bytes())

        # Hash detailed files
        detailed_dir = schemas_dir / "detailed"
        if detailed_dir.exists():
            for json_file in sorted(detailed_dir.glob("*.json")):
                hash_obj.update(json_file.read_bytes())

        return hash_obj.hexdigest()[:16]

    def _get_max_source_mtime(self, schemas_dir: Path) -> float:
        """Get maximum modification time of source files."""
        max_mtime = 0.0

        # Check catalog file
        catalog_file = schemas_dir / "ids_catalog.json"
        if catalog_file.exists():
            max_mtime = max(max_mtime, catalog_file.stat().st_mtime)

        # Check detailed files
        detailed_dir = schemas_dir / "detailed"
        if detailed_dir.exists():
            for json_file in detailed_dir.glob("*.json"):
                max_mtime = max(max_mtime, json_file.stat().st_mtime)

        return max_mtime

    def generate_all_graphs(self, ids_data: dict[str, dict[str, Any]]) -> None:
        """Generate Mermaid graphs for all IDS."""
        # Ensure directories exist before generating graphs
        self.mermaid_dir.mkdir(exist_ok=True)
        self.detailed_dir.mkdir(exist_ok=True)

        logger.info("Generating Mermaid graphs for all IDS...")

        for ids_name, data in ids_data.items():
            logger.debug(f"Generating Mermaid graph for {ids_name}")
            self._generate_ids_graph(ids_name, data)

        # Generate summary graph showing all IDS relationships
        self._generate_ids_overview_graph(ids_data)

        logger.info(f"Generated Mermaid graphs for {len(ids_data)} IDS")

    def _generate_ids_graph(self, ids_name: str, ids_data: dict[str, Any]) -> None:
        """Generate Mermaid graph for a single IDS."""
        paths = ids_data.get("paths", {})

        if not paths:
            logger.warning(f"No paths found for {ids_name}, skipping graph generation")
            return

        # Build hierarchy tree for graph generation
        tree = self._build_graph_tree(paths)

        # Generate different graph types for different use cases
        self._generate_hierarchy_graph(ids_name, tree)
        self._generate_physics_domain_graph(ids_name, paths)
        self._generate_complexity_graph(ids_name, tree, paths)

    def _build_graph_tree(self, paths: dict[str, Any]) -> dict[str, Any]:
        """Build tree structure optimized for Mermaid graph generation."""
        tree: dict[str, Any] = {"children": {}, "paths": set()}

        for path, path_data in paths.items():
            # Skip if path_data is None
            if path_data is None:
                logger.warning(f"Skipping path with no data: {path}")
                continue

            # Split path and build tree
            components = path.split("/")
            current = tree

            for i, component in enumerate(components):
                if "children" not in current:
                    current["children"] = {}

                if component not in current["children"]:
                    physics_context = path_data.get("physics_context") or {}
                    current["children"][component] = {
                        "children": {},
                        "paths": set(),
                        "depth": i,
                        "physics_domain": physics_context.get("domain", "unspecified"),
                        "data_type": path_data.get("data_type", ""),
                        "is_leaf": False,  # Will be determined after all paths are processed
                    }

                current["children"][component]["paths"].add(path)
                current = current["children"][component]

        # After processing all paths, determine leaf status based on whether nodes have children
        def mark_leaf_nodes(node: dict[str, Any]) -> None:
            """Recursively mark leaf nodes based on whether they have children."""
            if "children" in node:
                for _child_name, child_node in node["children"].items():
                    # A node is a leaf if it has no children
                    child_node["is_leaf"] = len(child_node.get("children", {})) == 0
                    # Recursively process children
                    mark_leaf_nodes(child_node)

        mark_leaf_nodes(tree)

        return tree

    def _generate_hierarchy_graph(self, ids_name: str, tree: dict[str, Any]) -> None:
        """Generate hierarchical structure Mermaid graph."""
        mermaid_content = [
            "```mermaid",
            "flowchart TD",
            f'    root["{ids_name} IDS"]',
            "",
        ]

        # Generate nodes and connections
        node_id = 0
        id_map = {"root": "root"}

        def process_node(node_data: dict[str, Any], parent_id: str, name: str) -> None:
            nonlocal node_id

            if not node_data.get("children"):
                return

            # Show all children
            children = list(node_data["children"].items())

            for child_name, child_data in children:
                node_id += 1
                current_id = f"n{node_id}"
                id_map[child_name] = current_id

                # Determine node style based on properties
                if child_data.get("is_leaf"):
                    node_style = f"{current_id}[{child_name}]"
                    style_class = "leafNode"
                elif len(child_data.get("children", {})) > 5:
                    node_style = f"{current_id}({child_name})"
                    style_class = "complexNode"
                else:
                    node_style = f"{current_id}[{child_name}]"
                    style_class = "normalNode"

                mermaid_content.append(f"    {node_style}")
                mermaid_content.append(f"    {parent_id} --> {current_id}")

                # Add style class
                mermaid_content.append(f"    class {current_id} {style_class}")

                # Recursively process all children - no depth limit
                process_node(child_data, current_id, child_name)

        # Process tree
        process_node(tree, "root", ids_name)

        # Add styling
        mermaid_content.extend(
            [
                "",
                "    classDef leafNode fill:#e1f5fe",
                "    classDef complexNode fill:#fff3e0",
                "    classDef normalNode fill:#f3e5f5",
                "    classDef moreNode fill:#f5f5f5,stroke-dasharray: 5 5",
                "```",
            ]
        )

        # Save to file
        output_path = self.detailed_dir / f"{ids_name}_hierarchy.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mermaid_content))

        logger.debug(f"Generated hierarchy graph for {ids_name}")

    def _generate_physics_domain_graph(
        self, ids_name: str, paths: dict[str, Any]
    ) -> None:
        """Generate physics domain organization Mermaid graph."""
        # Group paths by physics domain
        domain_paths: dict[str, list[str]] = {}

        for path, path_data in paths.items():
            if path_data is None:
                continue

            physics_context = path_data.get("physics_context") or {}
            domain = physics_context.get("domain", "unspecified")
            if domain not in domain_paths:
                domain_paths[domain] = []
            domain_paths[domain].append(path)

        if len(domain_paths) <= 1:
            logger.debug(
                f"Skipping physics domain graph for {ids_name} (single domain)"
            )
            return

        mermaid_content = [
            "```mermaid",
            "flowchart LR",
            f'    root["{ids_name}"]',
            "",
        ]

        # Generate domain nodes
        domain_id = 0
        for domain, paths_list in domain_paths.items():
            domain_id += 1
            domain_node_id = f"d{domain_id}"

            # Clean domain name for display
            display_domain = domain.replace("_", " ").title()
            if display_domain == "Unspecified":
                display_domain = "Other"

            mermaid_content.append(
                f'    {domain_node_id}["{display_domain}<br/>({len(paths_list)} paths)"]'
            )
            mermaid_content.append(f"    root --> {domain_node_id}")

            # Add representative paths (show more for better coverage)
            paths_to_show = min(len(paths_list), 6)  # Show up to 6 paths instead of 3
            for i, path in enumerate(paths_list[:paths_to_show]):
                path_id = f"p{domain_id}_{i + 1}"
                # Shorten path for display
                display_path = path.split("/")[-1] if "/" in path else path
                if len(display_path) > 20:
                    display_path = display_path[:17] + "..."

                mermaid_content.append(f'    {path_id}["{display_path}"]')
                mermaid_content.append(f"    {domain_node_id} --> {path_id}")
                mermaid_content.append(f"    class {path_id} pathNode")

            # Show count if there are more paths
            if len(paths_list) > paths_to_show:
                more_id = f"more{domain_id}"
                remaining = len(paths_list) - paths_to_show
                mermaid_content.append(
                    f'    {more_id}["...and {remaining} more paths"]'
                )
                mermaid_content.append(f"    {domain_node_id} --> {more_id}")
                mermaid_content.append(f"    class {more_id} moreNode")

            # Add style for domain
            mermaid_content.append(f"    class {domain_node_id} domainNode")

        # Add styling
        mermaid_content.extend(
            [
                "",
                "    classDef domainNode fill:#e8f5e8,stroke:#4caf50",
                "    classDef pathNode fill:#fff8e1,stroke:#ff9800",
                "    classDef moreNode fill:#f5f5f5,stroke-dasharray: 5 5",
                "```",
            ]
        )

        # Save to file
        output_path = self.detailed_dir / f"{ids_name}_physics_domains.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mermaid_content))

        logger.debug(f"Generated physics domain graph for {ids_name}")

    def _generate_complexity_graph(
        self, ids_name: str, tree: dict[str, Any], paths: dict[str, Any]
    ) -> None:
        """Generate complexity visualization Mermaid graph."""
        mermaid_content = [
            "```mermaid",
            "mindmap",
            f"  root(({ids_name}))",
        ]

        # Analyze complexity at each level
        def get_complexity_score(node_data: dict[str, Any]) -> int:
            """Calculate complexity score for a node."""
            children_count = len(node_data.get("children", {}))
            depth = node_data.get("depth", 0)
            path_count = len(node_data.get("paths", set()))

            # Simple complexity formula
            return children_count * 2 + depth + (path_count // 10)

        # Process top-level nodes and their complexity
        if "children" in tree:
            for name, node_data in tree["children"].items():  # Show all nodes
                complexity = get_complexity_score(node_data)
                children_count = len(node_data.get("children", {}))

                if complexity > 15:
                    icon = "🔴"  # High complexity
                elif complexity > 8:
                    icon = "🟡"  # Medium complexity
                else:
                    icon = "🟢"  # Low complexity

                mermaid_content.append(f"    {icon} {name}")

                # Add key children for high complexity nodes
                if children_count > 3 and complexity > 10:
                    for child_name in list(node_data.get("children", {}).keys())[
                        :5
                    ]:  # Show top 5 instead of 3
                        mermaid_content.append(f"      {child_name}")

        mermaid_content.append("```")

        # Save to file
        output_path = self.detailed_dir / f"{ids_name}_complexity.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mermaid_content))

        logger.debug(f"Generated complexity graph for {ids_name}")

    def _generate_ids_overview_graph(self, ids_data: dict[str, dict[str, Any]]) -> None:
        """Generate overview graph showing all IDS relationships."""
        mermaid_content = [
            "```mermaid",
            "graph TB",
            '    IMAS["IMAS Data Dictionary"]',
            "",
        ]

        # Categorize IDS by complexity and domain
        simple_ids = []
        complex_ids = []
        diagnostic_ids = []
        physics_ids = []

        for ids_name, data in ids_data.items():
            path_count = len(data.get("paths", {}))

            # Categorize by complexity
            if path_count < 50:
                simple_ids.append(ids_name)
            else:
                complex_ids.append(ids_name)

            # Categorize by type (simple heuristics)
            if any(
                diag in ids_name
                for diag in ["thomson", "interferometer", "camera", "spectrometer"]
            ):
                diagnostic_ids.append(ids_name)
            elif any(
                phys in ids_name for phys in ["core", "equilibrium", "mhd", "transport"]
            ):
                physics_ids.append(ids_name)

        # Generate category nodes
        if simple_ids:
            mermaid_content.append('    SIMPLE["Simple IDS<br/>(< 50 paths)"]')
            mermaid_content.append("    IMAS --> SIMPLE")

        if complex_ids:
            mermaid_content.append('    COMPLEX["Complex IDS<br/>(≥ 50 paths)"]')
            mermaid_content.append("    IMAS --> COMPLEX")

        # Add all IDS to each category, organized in a more compact way
        for ids_list, category in [(simple_ids, "SIMPLE"), (complex_ids, "COMPLEX")]:
            for i, ids_name in enumerate(ids_list):  # Show all IDS
                node_id = f"{category}_{i + 1}"
                display_name = ids_name.replace("_", " ").title()
                mermaid_content.append(f'    {node_id}["{display_name}"]')
                mermaid_content.append(f"    {category} --> {node_id}")

        # Add styling
        mermaid_content.extend(
            [
                "",
                "    classDef simpleNode fill:#e8f5e8",
                "    classDef complexNode fill:#ffe8e8",
                "    classDef diagnosticNode fill:#e8e8ff",
                "    classDef physicsNode fill:#ffe8ff",
            ]
        )

        # Apply styles to all IDS
        for i in range(len(simple_ids)):
            mermaid_content.append(f"    class SIMPLE_{i + 1} simpleNode")

        for i in range(len(complex_ids)):
            mermaid_content.append(f"    class COMPLEX_{i + 1} complexNode")

        mermaid_content.append("```")

        # Save to file
        output_path = self.mermaid_dir / "ids_overview.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mermaid_content))

        logger.info("Generated IDS overview graph")

    def load_mermaid_graph(
        self, ids_name: str, graph_type: str = "hierarchy"
    ) -> str | None:
        """Load a specific Mermaid graph for an IDS."""
        graph_file = self.detailed_dir / f"{ids_name}_{graph_type}.md"

        if not graph_file.exists():
            logger.warning(f"Mermaid graph not found: {graph_file}")
            return None

        try:
            with open(graph_file, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load Mermaid graph {graph_file}: {e}")
            return None

    def get_available_graphs(self, ids_name: str) -> list[str]:
        """Get list of available graph types for an IDS."""
        graph_types = []
        for graph_type in ["hierarchy", "physics_domains", "complexity"]:
            graph_file = self.detailed_dir / f"{ids_name}_{graph_type}.md"
            if graph_file.exists():
                graph_types.append(graph_type)
        return graph_types

    def get_overview_graph(self) -> str | None:
        """Get the overview graph showing all IDS."""
        overview_file = self.mermaid_dir / "ids_overview.md"

        if not overview_file.exists():
            return None

        try:
            with open(overview_file, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load overview graph: {e}")
            return None
