"""
Enhanced IDS structure analysis with static data generation.

This module provides comprehensive analysis of IMAS IDS structure including:
- Hierarchical structure analysis
- Physics domain distribution
- Complexity metrics
- Navigation optimization
- Tree structure data generation

Generates static analysis data during XML processing for fast runtime access.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imas_mcp.models.structure_models import (
    DomainDistribution,
    HierarchyMetrics,
    MermaidGraphs,
    NavigationHints,
    StructureAnalysis,
)
from imas_mcp.structure.mermaid_generator import MermaidGraphGenerator

logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """Represents a node in the IDS path hierarchy."""

    path: str
    children: dict[str, "PathNode"] = field(default_factory=dict)
    parent: "PathNode | None" = None
    depth: int = 0
    is_leaf: bool = False
    physics_domain: str = ""
    data_type: str = ""
    units: str = ""
    complexity_factors: dict[str, Any] = field(default_factory=dict)

    def add_child(self, name: str, node: "PathNode") -> None:
        """Add a child node."""
        self.children[name] = node
        node.parent = self
        node.depth = self.depth + 1

    def get_subtree_size(self) -> int:
        """Get the total number of nodes in this subtree."""
        size = 1  # This node
        for child in self.children.values():
            size += child.get_subtree_size()
        return size

    def get_leaf_count(self) -> int:
        """Get the number of leaf nodes in this subtree."""
        if self.is_leaf:
            return 1
        return sum(child.get_leaf_count() for child in self.children.values())

    def get_max_depth(self) -> int:
        """Get the maximum depth in this subtree."""
        if not self.children:
            return self.depth
        return max(child.get_max_depth() for child in self.children.values())

    def get_average_branching_factor(self) -> float:
        """Calculate average branching factor for this subtree."""
        if not self.children:
            return 0.0

        total_children = 0
        non_leaf_nodes = 0

        # Count this node if it has children
        total_children += len(self.children)
        non_leaf_nodes += 1

        # Recursively count for children
        for child in self.children.values():
            child_children, child_non_leaf = child._count_branching_stats()
            total_children += child_children
            non_leaf_nodes += child_non_leaf

        return total_children / non_leaf_nodes if non_leaf_nodes > 0 else 0.0

    def _count_branching_stats(self) -> tuple[int, int]:
        """Helper method to count branching statistics."""
        if not self.children:
            return 0, 0

        total_children = len(self.children)
        non_leaf_nodes = 1

        for child in self.children.values():
            child_children, child_non_leaf = child._count_branching_stats()
            total_children += child_children
            non_leaf_nodes += child_non_leaf

        return total_children, non_leaf_nodes


class StructureAnalyzer:
    """Enhanced IDS structure analyzer with static data generation."""

    def __init__(self, data_dir: Path):
        """Initialize analyzer with output directory."""
        self.data_dir = data_dir
        self.structure_dir = data_dir / "structure"
        self.structure_dir.mkdir(exist_ok=True)

        # Initialize Mermaid generator with resources directory (parent of schemas)
        # data_dir is typically the schemas directory, so we need to go up one level
        resources_dir = data_dir.parent
        self.mermaid_generator = MermaidGraphGenerator(resources_dir)

    def analyze_all_ids(self, ids_data: dict[str, dict[str, Any]]) -> None:
        """Analyze structure for all IDS and generate static data files."""
        logger.info("Generating enhanced structure analysis for all IDS...")

        # Generate analysis for each IDS
        for ids_name, data in ids_data.items():
            logger.debug(f"Analyzing structure for {ids_name}")
            analysis = self.analyze_ids_structure(ids_name, data)
            self._save_structure_analysis(ids_name, analysis)

        # Generate Mermaid graphs alongside structure analysis
        self._generate_structure_catalog(ids_data)

        # Generate Mermaid graphs for visual representation
        logger.info("Generating Mermaid graphs for visual analysis...")
        self.mermaid_generator.generate_all_graphs(ids_data)

        logger.info(f"Enhanced structure analysis completed for {len(ids_data)} IDS")

    def analyze_ids_structure(
        self, ids_name: str, ids_data: dict[str, Any]
    ) -> StructureAnalysis:
        """Analyze structure for a single IDS."""
        paths = ids_data.get("paths", {})

        # Build hierarchy tree
        root = self._build_hierarchy_tree(paths)

        # Calculate hierarchy metrics
        hierarchy_metrics = self._calculate_hierarchy_metrics(root)

        # Analyze physics domain distribution
        domain_distribution = self._analyze_domain_distribution(paths)

        # Generate navigation hints
        navigation_hints = self._generate_navigation_hints(root, paths)

        # Generate complexity summary
        complexity_summary = self._generate_complexity_summary(hierarchy_metrics)

        # Identify organization pattern
        organization_pattern = self._identify_organization_pattern(root, paths)

        return StructureAnalysis(
            hierarchy_metrics=hierarchy_metrics,
            domain_distribution=domain_distribution,
            navigation_hints=navigation_hints,
            complexity_summary=complexity_summary,
            organization_pattern=organization_pattern,
        )

    def _build_hierarchy_tree(self, paths: dict[str, Any]) -> PathNode:
        """Build hierarchical tree structure from paths."""
        root = PathNode(path="", depth=0)

        for path, path_data in paths.items():
            # Split path into components
            components = path.split("/")
            current_node = root

            # Build tree structure
            for i, component in enumerate(components):
                if component not in current_node.children:
                    node_path = "/".join(components[: i + 1])
                    new_node = PathNode(
                        path=node_path,
                        depth=i + 1,
                        physics_domain=path_data.get("physics_context", {}).get(
                            "domain", ""
                        ),
                        data_type=path_data.get("data_type", ""),
                        units=path_data.get("units", ""),
                    )
                    current_node.add_child(component, new_node)

                current_node = current_node.children[component]

            # Mark as leaf node
            current_node.is_leaf = True

        return root

    def _calculate_hierarchy_metrics(self, root: PathNode) -> HierarchyMetrics:
        """Calculate hierarchy metrics from tree structure."""
        total_nodes = root.get_subtree_size() - 1  # Exclude root
        leaf_nodes = root.get_leaf_count()
        max_depth = root.get_max_depth()
        branching_factor = root.get_average_branching_factor()

        # Calculate complexity score based on multiple factors
        complexity_score = self._calculate_complexity_score(
            total_nodes, leaf_nodes, max_depth, branching_factor
        )

        return HierarchyMetrics(
            total_nodes=total_nodes,
            leaf_nodes=leaf_nodes,
            max_depth=max_depth,
            branching_factor=branching_factor,
            complexity_score=complexity_score,
        )

    def _calculate_complexity_score(
        self, total_nodes: int, leaf_nodes: int, max_depth: int, branching_factor: float
    ) -> float:
        """Calculate structural complexity score."""
        # Normalize components
        size_factor = min(total_nodes / 1000, 1.0)  # Cap at 1000 nodes
        depth_factor = min(max_depth / 10, 1.0)  # Cap at depth 10
        branching_factor_norm = min(branching_factor / 10, 1.0)  # Cap at factor 10

        # Weight the factors
        complexity = (
            0.4 * size_factor + 0.3 * depth_factor + 0.3 * branching_factor_norm
        )

        return round(complexity, 3)

    def _analyze_domain_distribution(
        self, paths: dict[str, Any]
    ) -> list[DomainDistribution]:
        """Analyze physics domain distribution within the IDS."""
        domain_counts: dict[str, list[str]] = {}

        # Count paths by domain
        for path, path_data in paths.items():
            physics_context = path_data.get("physics_context", {})
            domain = physics_context.get("domain", "unspecified")

            if domain not in domain_counts:
                domain_counts[domain] = []
            domain_counts[domain].append(path)

        # Create distribution objects
        total_paths = len(paths)
        distributions = []

        for domain, domain_paths in domain_counts.items():
            node_count = len(domain_paths)
            percentage = (node_count / total_paths) * 100 if total_paths > 0 else 0

            # Select key representative paths
            key_paths = domain_paths[:5]  # Top 5 paths

            distributions.append(
                DomainDistribution(
                    domain=domain,
                    node_count=node_count,
                    percentage=round(percentage, 1),
                    key_paths=key_paths,
                )
            )

        # Sort by node count (descending)
        distributions.sort(key=lambda x: x.node_count, reverse=True)
        return distributions

    def _generate_navigation_hints(
        self, root: PathNode, paths: dict[str, Any]
    ) -> NavigationHints:
        """Generate navigation hints for the IDS structure."""
        # Find entry points (top-level structures with good documentation)
        entry_points = []
        for _child_name, child_node in root.children.items():
            if child_node.children:  # Has sub-structure
                entry_points.append(child_node.path)

        # Identify common patterns in path structure
        common_patterns = self._identify_common_patterns(paths)

        # Generate drill-down suggestions based on complexity
        drill_down_suggestions = self._generate_drill_down_suggestions(root)

        return NavigationHints(
            entry_points=entry_points[:10],  # Top 10 entry points
            common_patterns=common_patterns[:5],  # Top 5 patterns
            drill_down_suggestions=drill_down_suggestions[:8],  # Top 8 suggestions
        )

    def _identify_common_patterns(self, paths: dict[str, Any]) -> list[str]:
        """Identify common structural patterns in paths."""
        patterns = []

        # Look for time series patterns
        time_paths = [path for path in paths.keys() if "time" in path.lower()]
        if time_paths:
            patterns.append("Time series data structure detected")

        # Look for profile patterns
        profile_paths = [path for path in paths.keys() if "profiles" in path.lower()]
        if profile_paths:
            patterns.append("Profile data organization present")

        # Look for coordinate patterns
        coord_paths = [
            path
            for path in paths.keys()
            if any(coord in path for coord in ["1d", "2d", "3d"])
        ]
        if coord_paths:
            patterns.append("Multi-dimensional coordinate organization")

        # Look for array patterns
        array_paths = [path for path in paths.keys() if "[]" in path or "_" in path]
        if array_paths:
            patterns.append("Array and indexed data structures")

        return patterns

    def _generate_drill_down_suggestions(self, root: PathNode) -> list[str]:
        """Generate drill-down suggestions based on tree structure."""
        suggestions = []

        # Find the most complex subtrees
        complex_nodes = []
        for _child_name, child_node in root.children.items():
            if child_node.children:
                complexity = child_node.get_subtree_size()
                complex_nodes.append((complexity, child_node.path))

        # Sort by complexity and suggest exploration
        complex_nodes.sort(reverse=True)
        for _, path in complex_nodes[:5]:
            suggestions.append(f"Explore {path} structure (high complexity)")

        # Suggest leaf node exploration
        suggestions.append("Examine leaf nodes for data entry points")

        # Suggest domain-specific exploration
        suggestions.append("Filter by physics domain for focused analysis")

        return suggestions

    def _generate_complexity_summary(self, metrics: HierarchyMetrics) -> str:
        """Generate human-readable complexity summary."""
        if metrics.complexity_score < 0.3:
            complexity_level = "Low"
        elif metrics.complexity_score < 0.6:
            complexity_level = "Moderate"
        elif metrics.complexity_score < 0.8:
            complexity_level = "High"
        else:
            complexity_level = "Very High"

        return (
            f"{complexity_level} complexity structure with {metrics.total_nodes} nodes, "
            f"{metrics.leaf_nodes} data endpoints, and {metrics.max_depth} levels deep. "
            f"Average branching factor: {metrics.branching_factor:.1f}"
        )

    def _identify_organization_pattern(
        self, root: PathNode, paths: dict[str, Any]
    ) -> str:
        """Identify the overall organization pattern of the IDS."""
        # Analyze top-level structure
        top_level_count = len(root.children)

        if top_level_count <= 3:
            pattern = "Focused organization"
        elif top_level_count <= 8:
            pattern = "Balanced organization"
        else:
            pattern = "Comprehensive organization"

        # Check for time-based organization
        time_organization = any(
            "time" in child_name.lower() for child_name in root.children.keys()
        )
        if time_organization:
            pattern += " with temporal structure"

        # Check for physics-based organization
        physics_domains = set()
        for path_data in paths.values():
            domain = path_data.get("physics_context", {}).get("domain", "")
            if domain:
                physics_domains.add(domain)

        if len(physics_domains) > 1:
            pattern += " with multi-domain physics coverage"

        return pattern

    def _save_structure_analysis(
        self, ids_name: str, analysis: StructureAnalysis
    ) -> None:
        """Save structure analysis to JSON file."""
        output_path = self.structure_dir / f"{ids_name}.json"

        # Convert to serializable dict
        analysis_dict = analysis.model_dump()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, indent=2)

        logger.debug(f"Saved structure analysis for {ids_name} to {output_path}")

    def _generate_structure_catalog(self, ids_data: dict[str, dict[str, Any]]) -> None:
        """Generate cross-IDS structure catalog."""
        from datetime import UTC, datetime

        catalog = {
            "metadata": {
                "total_ids": len(ids_data),
                "analysis_timestamp": datetime.now(UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "analysis_version": "1.0",
            },
            "structure_summary": {},
            "complexity_rankings": [],
            "organization_patterns": {},
        }

        complexity_scores = []

        # Collect summaries from each IDS
        for ids_name in ids_data.keys():
            analysis_path = self.structure_dir / f"{ids_name}.json"
            if analysis_path.exists():
                with open(analysis_path, encoding="utf-8") as f:
                    analysis_data = json.load(f)

                # Extract key metrics
                metrics = analysis_data.get("hierarchy_metrics", {})
                complexity_score = metrics.get("complexity_score", 0)
                total_nodes = metrics.get("total_nodes", 0)
                max_depth = metrics.get("max_depth", 0)

                catalog["structure_summary"][ids_name] = {
                    "complexity_score": complexity_score,
                    "total_nodes": total_nodes,
                    "max_depth": max_depth,
                    "organization_pattern": analysis_data.get(
                        "organization_pattern", ""
                    ),
                }

                complexity_scores.append((complexity_score, ids_name))

        # Generate complexity rankings
        complexity_scores.sort(reverse=True)
        catalog["complexity_rankings"] = [
            {"ids_name": ids_name, "complexity_score": score}
            for score, ids_name in complexity_scores
        ]

        # Analyze organization patterns
        patterns = {}
        for ids_name, summary in catalog["structure_summary"].items():
            pattern = summary.get("organization_pattern", "unknown")
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(ids_name)

        catalog["organization_patterns"] = patterns

        # Save catalog
        catalog_path = self.structure_dir / "structure_catalog.json"
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Generated structure catalog at {catalog_path}")

    def load_structure_analysis(self, ids_name: str) -> StructureAnalysis | None:
        """Load pre-generated structure analysis from file."""
        analysis_path = self.structure_dir / f"{ids_name}.json"

        if not analysis_path.exists():
            logger.warning(f"No structure analysis found for {ids_name}")
            return None

        try:
            with open(analysis_path, encoding="utf-8") as f:
                analysis_data = json.load(f)

            # Convert back to Pydantic models
            analysis = StructureAnalysis(**analysis_data)

            # Load Mermaid graphs if available
            mermaid_graphs = self._load_mermaid_graphs(ids_name)
            if mermaid_graphs:
                analysis.mermaid_graphs = mermaid_graphs

            return analysis

        except Exception as e:
            logger.error(f"Failed to load structure analysis for {ids_name}: {e}")
            return None

    def _load_mermaid_graphs(self, ids_name: str) -> MermaidGraphs | None:
        """Load Mermaid graphs for an IDS."""
        try:
            available_graphs = self.mermaid_generator.get_available_graphs(ids_name)

            if not available_graphs:
                return None

            mermaid_graphs = MermaidGraphs(available_graphs=available_graphs)

            # Load each available graph type
            if "hierarchy" in available_graphs:
                mermaid_graphs.hierarchy_graph = (
                    self.mermaid_generator.load_mermaid_graph(ids_name, "hierarchy")
                )

            if "physics_domains" in available_graphs:
                mermaid_graphs.physics_domains_graph = (
                    self.mermaid_generator.load_mermaid_graph(
                        ids_name, "physics_domains"
                    )
                )

            if "complexity" in available_graphs:
                mermaid_graphs.complexity_graph = (
                    self.mermaid_generator.load_mermaid_graph(ids_name, "complexity")
                )

            return mermaid_graphs

        except Exception as e:
            logger.error(f"Failed to load Mermaid graphs for {ids_name}: {e}")
            return None

    def get_structure_catalog(self) -> dict[str, Any] | None:
        """Load structure catalog."""
        catalog_path = self.structure_dir / "structure_catalog.json"

        if not catalog_path.exists():
            return None

        try:
            with open(catalog_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load structure catalog: {e}")
            return None
