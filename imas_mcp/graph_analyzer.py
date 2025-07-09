"""
Graph analysis module for IMAS data dictionary structure.
Uses NetworkX to analyze hierarchical relationships and extract insights.
"""

import networkx as nx
from typing import Dict, List, Any
from collections import defaultdict
import statistics


class IMASGraphAnalyzer:
    """Analyzes IMAS data structures using graph theory."""

    def __init__(self):
        self.graphs = {}  # Store graphs per IDS

    def build_ids_graph(self, ids_name: str, paths: Dict[str, Any]) -> nx.DiGraph:
        """Build a directed graph representing the hierarchical structure of an IDS."""
        G = nx.DiGraph()

        # Add nodes and edges based on path hierarchy
        for path in paths.keys():
            # Handle both dot and slash separators
            if "/" in path:
                parts = path.split("/")
            else:
                parts = path.split(".")

            # Add all path segments as nodes
            for i in range(len(parts)):
                if "/" in path:
                    node = "/".join(parts[: i + 1])
                else:
                    node = ".".join(parts[: i + 1])
                G.add_node(node, level=i, name=parts[i])

                # Add edge from parent to child
                if i > 0:
                    if "/" in path:
                        parent = "/".join(parts[:i])
                    else:
                        parent = ".".join(parts[:i])
                    G.add_edge(parent, node)

        return G

    def analyze_ids_structure(
        self, ids_name: str, paths: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive structural analysis of an IDS."""
        G = self.build_ids_graph(ids_name, paths)
        self.graphs[ids_name] = G

        # Basic graph metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Hierarchy analysis
        levels = defaultdict(list)
        for node, data in G.nodes(data=True):
            levels[data.get("level", 0)].append(node)

        max_depth = max(levels.keys()) if levels else 0
        avg_depth = (
            statistics.mean([data.get("level", 0) for _, data in G.nodes(data=True)])
            if num_nodes > 0
            else 0
        )

        # Branching analysis
        out_degrees = [G.out_degree(node) for node in G.nodes()]

        # Calculate branching factors
        non_leaf_nodes = [d for d in out_degrees if d > 0]
        avg_branching = statistics.mean(out_degrees) if out_degrees else 0
        avg_branching_non_leaf = (
            statistics.mean(non_leaf_nodes) if non_leaf_nodes else 0
        )
        max_branching = max(out_degrees) if out_degrees else 0

        # Find leaves and roots
        leaves = [node for node in G.nodes() if G.out_degree(node) == 0]
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]

        # Path complexity analysis
        array_paths = []
        for path, data in paths.items():
            if isinstance(data, dict):
                coords = data.get("coordinates", [])
                coord1 = data.get("coordinate1", "")
                coord2 = data.get("coordinate2", "")

                # Path is an array if it has coordinates
                if coords or coord1 or coord2:
                    array_paths.append(path)

        time_dependent_paths = [p for p in paths.keys() if "time" in p.lower()]

        # Clustering coefficient (treat as undirected for this metric)
        try:
            avg_clustering = (
                nx.average_clustering(G.to_undirected()) if num_nodes > 1 else 0
            )
        except Exception:
            avg_clustering = 0

        # Density
        density = nx.density(G) if num_nodes > 1 else 0

        # Determine separator for deepest paths sorting
        separator = "/" if any("/" in p for p in paths.keys()) else "."

        return {
            "basic_metrics": {
                "total_nodes": num_nodes,
                "total_edges": num_edges,
                "density": round(density, 4),
                "avg_clustering": round(avg_clustering, 4),
            },
            "hierarchy_metrics": {
                "max_depth": max_depth,
                "avg_depth": round(avg_depth, 2),
                "levels": {str(level): len(nodes) for level, nodes in levels.items()},
                "leaf_count": len(leaves),
                "root_count": len(roots),
            },
            "branching_metrics": {
                "avg_branching_factor": round(avg_branching, 2),
                "avg_branching_factor_non_leaf": round(avg_branching_non_leaf, 2),
                "max_branching_factor": max_branching,
                "highly_branched_nodes": len([d for d in out_degrees if d > 5]),
                "non_leaf_nodes": len(non_leaf_nodes),
                "leaf_nodes": len([d for d in out_degrees if d == 0]),
            },
            "complexity_indicators": {
                "array_paths": len(array_paths),
                "array_ratio": round(len(array_paths) / len(paths), 3) if paths else 0,
                "time_dependent_paths": len(time_dependent_paths),
                "time_dependent_ratio": round(len(time_dependent_paths) / len(paths), 3)
                if paths
                else 0,
            },
            "key_nodes": {
                "most_connected": self._get_most_connected_nodes(G, 3),
                "deepest_paths": sorted(
                    leaves, key=lambda x: len(x.split(separator)), reverse=True
                )[:5],
                "root_categories": roots,
            },
        }

    def _get_most_connected_nodes(
        self, G: nx.DiGraph, top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Get nodes with highest degree centrality."""
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Convert to dict to avoid type checking issues
        degree_dict = dict(G.degree)  # type: ignore

        return [
            {
                "node": node,
                "centrality": round(score, 4),
                "degree": degree_dict[node],
                "level": G.nodes[node].get("level", 0),
            }
            for node, score in top_nodes
        ]

    def analyze_cross_ids_patterns(
        self, all_ids_data: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Analyze patterns across all IDS structures."""
        all_stats = {}
        complexity_scores = {}

        for ids_name, paths in all_ids_data.items():
            stats = self.analyze_ids_structure(ids_name, paths)
            all_stats[ids_name] = stats

            # Calculate composite complexity score
            complexity_score = (
                stats["basic_metrics"]["total_nodes"] * 0.3
                + stats["hierarchy_metrics"]["max_depth"] * 10
                + stats["branching_metrics"]["max_branching_factor"] * 5
                + stats["complexity_indicators"]["array_paths"] * 2
                + (1 - stats["basic_metrics"]["avg_clustering"]) * 20
            )
            complexity_scores[ids_name] = round(complexity_score, 2)

        # Overall insights
        total_nodes = sum(
            stats["basic_metrics"]["total_nodes"] for stats in all_stats.values()
        )

        # Handle empty stats case
        depths = [
            stats["hierarchy_metrics"]["avg_depth"] for stats in all_stats.values()
        ]
        avg_depth = statistics.mean(depths) if depths else 0.0

        # Categorize by complexity
        sorted_by_complexity = sorted(
            complexity_scores.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "overview": {
                "total_ids": len(all_stats),
                "total_nodes_all_ids": total_nodes,
                "avg_depth_across_ids": round(avg_depth, 2),
                "complexity_range": {
                    "min": min(complexity_scores.values()) if complexity_scores else 0,
                    "max": max(complexity_scores.values()) if complexity_scores else 0,
                    "avg": round(statistics.mean(complexity_scores.values()), 2)
                    if complexity_scores
                    else 0,
                },
            },
            "complexity_rankings": {
                "most_complex": sorted_by_complexity[:5],
                "least_complex": sorted_by_complexity[-5:],
                "complexity_scores": complexity_scores,
            },
            "structural_patterns": {
                "deepest_ids": sorted(
                    [
                        (name, stats["hierarchy_metrics"]["max_depth"])
                        for name, stats in all_stats.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
                "most_branched": sorted(
                    [
                        (name, stats["branching_metrics"]["max_branching_factor"])
                        for name, stats in all_stats.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
                "array_heavy": sorted(
                    [
                        (name, stats["complexity_indicators"]["array_ratio"])
                        for name, stats in all_stats.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
            },
        }


def analyze_imas_graphs(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to analyze all IMAS structures and return graph statistics."""
    analyzer = IMASGraphAnalyzer()

    # Extract IDS data
    ids_catalog = data_dict.get("ids_catalog", {})

    # Analyze each IDS
    graph_statistics = {}
    all_ids_data = {}

    for ids_name, ids_data in ids_catalog.items():
        if "paths" in ids_data:
            all_ids_data[ids_name] = ids_data["paths"]
            graph_statistics[ids_name] = analyzer.analyze_ids_structure(
                ids_name, ids_data["paths"]
            )

    # Cross-IDS analysis
    structural_insights = analyzer.analyze_cross_ids_patterns(all_ids_data)

    return {
        "graph_statistics": graph_statistics,
        "structural_insights": structural_insights,
        "analysis_metadata": {
            "total_ids_analyzed": len(graph_statistics),
            "analysis_timestamp": data_dict.get("metadata", {}).get("build_time", ""),
            "networkx_version": nx.__version__,
        },
    }
