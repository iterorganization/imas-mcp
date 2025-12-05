"""
Analysis tool implementation with service composition.

This module contains the analyze_ids_structure tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp import dd_version
from imas_mcp.graph_analyzer import IMASGraphAnalyzer
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import AnalysisInput
from imas_mcp.models.result_models import StructureResult
from imas_mcp.resource_path_accessor import ResourcePathAccessor
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)
from imas_mcp.search.decorators.physics_hints import physics_hints
from imas_mcp.search.decorators.tool_hints import tool_hints
from imas_mcp.structure.structure_analyzer import StructureAnalyzer

from .base import BaseTool

logger = logging.getLogger(__name__)


class AnalysisTool(BaseTool):
    """Tool for analyzing IDS structure with service composition."""

    def __init__(self, document_store=None):
        """Initialize the analysis tool with document store access and relationships data."""
        super().__init__(document_store)
        # Note: Structure analysis is now pre-generated during build time
        # We access it through the document store's data directory

        # Initialize graph analyzer for enhanced analysis
        self.graph_analyzer = IMASGraphAnalyzer()

        # Load relationships data for clustering insights
        self._relationships_data = {}
        self._load_relationships_data()

    def _load_relationships_data(self):
        """Load relationships data for clustering-enhanced analysis."""
        try:
            import json

            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            relationships_file = path_accessor.schemas_dir / "relationships.json"
            with relationships_file.open("r", encoding="utf-8") as f:
                self._relationships_data = json.load(f)
                logger.info(
                    "Loaded relationships data for enhanced structural analysis"
                )
        except Exception as e:
            logger.warning(f"Failed to load relationships data for analysis tool: {e}")
            self._relationships_data = {}

    def _enhance_structure_with_clustering(
        self, ids_name: str, ids_documents: list
    ) -> dict:
        """Enhance structural analysis with clustering insights."""
        if not self._relationships_data:
            return {}

        clusters = self._relationships_data.get("clusters", [])
        unit_families = self._relationships_data.get("unit_families", {})

        # Extract paths for this IDS
        ids_paths = [doc.metadata.path_name for doc in ids_documents]

        clustering_insights = {
            "cluster_participation": {
                "participating_clusters": [],
                "cross_ids_clusters": 0,
                "intra_ids_clusters": 0,
                "avg_similarity_score": 0.0,
            },
            "unit_family_analysis": {
                "covered_families": [],
                "total_family_coverage": 0,
            },
            "structural_patterns": {
                "clustered_paths_ratio": 0.0,
                "highest_similarity_cluster": None,
                "structural_coherence": "low",
            },
        }

        participating_clusters = []
        similarity_scores = []

        # Find clusters containing paths from this IDS
        for cluster in clusters:
            cluster_paths = cluster.get("paths", [])
            matching_paths = [p for p in ids_paths if p in cluster_paths]

            if matching_paths:
                cluster_info = {
                    "cluster_id": cluster["id"],
                    "similarity_score": cluster.get("similarity_score", 0.0),
                    "is_cross_ids": cluster.get("is_cross_ids", False),
                    "cluster_size": cluster.get("size", 0),
                    "matching_paths_count": len(matching_paths),
                    "coverage_ratio": len(matching_paths) / len(cluster_paths)
                    if cluster_paths
                    else 0,
                }
                participating_clusters.append(cluster_info)
                similarity_scores.append(cluster.get("similarity_score", 0.0))

                if cluster.get("is_cross_ids", False):
                    clustering_insights["cluster_participation"][
                        "cross_ids_clusters"
                    ] += 1
                else:
                    clustering_insights["cluster_participation"][
                        "intra_ids_clusters"
                    ] += 1

        clustering_insights["cluster_participation"]["participating_clusters"] = (
            participating_clusters
        )
        if similarity_scores:
            clustering_insights["cluster_participation"]["avg_similarity_score"] = sum(
                similarity_scores
            ) / len(similarity_scores)
            clustering_insights["structural_patterns"]["highest_similarity_cluster"] = (
                max(similarity_scores)
            )

        # Analyze unit family coverage
        covered_families = []
        total_coverage = 0

        for unit_name, unit_data in unit_families.items():
            unit_paths = unit_data.get("paths_using", [])
            matching_unit_paths = [p for p in ids_paths if p in unit_paths]

            if matching_unit_paths:
                family_info = {
                    "unit_name": unit_name,
                    "matching_paths_count": len(matching_unit_paths),
                    "total_family_size": len(unit_paths),
                    "coverage_ratio": len(matching_unit_paths) / len(unit_paths)
                    if unit_paths
                    else 0,
                }
                covered_families.append(family_info)
                total_coverage += len(matching_unit_paths)

        clustering_insights["unit_family_analysis"]["covered_families"] = (
            covered_families
        )
        clustering_insights["unit_family_analysis"]["total_family_coverage"] = (
            total_coverage
        )

        # Calculate structural patterns
        clustered_paths_count = sum(
            len(cluster.get("paths", []))
            for cluster in participating_clusters
            if cluster
        )
        clustering_insights["structural_patterns"]["clustered_paths_ratio"] = (
            clustered_paths_count / len(ids_paths) if ids_paths else 0.0
        )

        # Determine structural coherence
        avg_similarity = clustering_insights["cluster_participation"][
            "avg_similarity_score"
        ]
        if avg_similarity > 0.8:
            clustering_insights["structural_patterns"]["structural_coherence"] = (
                "very_high"
            )
        elif avg_similarity > 0.6:
            clustering_insights["structural_patterns"]["structural_coherence"] = "high"
        elif avg_similarity > 0.4:
            clustering_insights["structural_patterns"]["structural_coherence"] = (
                "moderate"
            )
        else:
            clustering_insights["structural_patterns"]["structural_coherence"] = "low"

        return {"clustering_insights": clustering_insights}

    async def _load_structure_analysis(self, ids_name: str):
        """Load pre-generated structure analysis from static files."""
        try:
            # Get the data directory from document store
            data_dir = self.documents.store._data_dir
            structure_analyzer = StructureAnalyzer(data_dir)
            return structure_analyzer.load_structure_analysis(ids_name)
        except Exception as e:
            logger.warning(f"Failed to load structure analysis for {ids_name}: {e}")
            return None

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "analyze_ids_structure"

    @cache_results(ttl=900, key_strategy="path_based")
    @validate_input(schema=AnalysisInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="analysis_suggestions")
    @tool_hints(max_hints=3)
    @physics_hints()
    @mcp_tool("Analyze the internal structure and organization of a specific IMAS IDS")
    async def analyze_ids_structure(
        self, ids_name: str, ctx: Context | None = None
    ) -> StructureResult | ToolError:
        """
        Analyze the internal structure and organization of a specific IMAS IDS.

        Provides comprehensive structural analysis including data hierarchy, branching
        points, identifier schemas, and physics context. Use this tool to understand
        how data is organized within an IDS before accessing specific measurements.

        Args:
            ids_name: Name of the IDS to analyze (e.g., 'equilibrium', 'thomson_scattering')
            ctx: FastMCP context

        Returns:
            StructureResult with detailed analysis
        """
        try:
            # Validate IDS exists using document service
            valid_ids, invalid_ids = await self.documents.validate_ids([ids_name])
            if not valid_ids:
                return self.documents.create_ids_not_found_error(
                    ids_name, self.tool_name
                )

            # Get detailed IDS data from document store
            ids_documents = await self.documents.get_documents_safe(ids_name)

            if not ids_documents:
                result = StructureResult(
                    ids_name=ids_name,
                    description=f"No detailed structure data available for {ids_name}",
                    structure={"total_paths": 0},
                    sample_paths=[],
                    max_depth=0,
                )

                logger.info(f"Structure analysis completed for {ids_name}")
                return result

            # Load pre-generated structure analysis from static files
            structure_analysis = await self._load_structure_analysis(ids_name)

            if structure_analysis:
                # Use pre-generated enhanced analysis with Mermaid graphs
                structure_dict = {
                    "total_nodes": structure_analysis.hierarchy_metrics.total_nodes,
                    "leaf_nodes": structure_analysis.hierarchy_metrics.leaf_nodes,
                    "max_depth": structure_analysis.hierarchy_metrics.max_depth,
                    "branching_factor": int(
                        structure_analysis.hierarchy_metrics.branching_factor
                    ),
                    "complexity_score": int(
                        structure_analysis.hierarchy_metrics.complexity_score * 100
                    ),
                    "physics_domains": len(structure_analysis.domain_distribution),
                }

                sample_paths = structure_analysis.navigation_hints.entry_points[:10]
                max_depth = structure_analysis.hierarchy_metrics.max_depth

                description = (
                    f"Enhanced structural analysis of {ids_name} IDS: "
                    f"{structure_analysis.complexity_summary}"
                )

                # Add graph analysis insights if we have document data
                if ids_documents:
                    graph_insights = self._perform_graph_analysis(
                        ids_name, ids_documents
                    )
                    if graph_insights:
                        description += f" Graph analysis reveals {graph_insights.get('summary', 'additional structural patterns')}."
                        # Merge graph insights into structure_dict with proper type conversion
                        graph_metrics = graph_insights.get("metrics", {})
                        for key, value in graph_metrics.items():
                            if isinstance(value, int | bool):
                                structure_dict[key] = value
                            elif isinstance(value, float):
                                structure_dict[key] = int(value * 1000)

            else:
                # Fallback to real-time analysis using graph analyzer and basic analysis
                logger.info(
                    f"No pre-generated analysis found for {ids_name}, performing real-time analysis"
                )

                if ids_documents:
                    # Perform graph analysis using the existing graph analyzer
                    graph_analysis = self._perform_graph_analysis(
                        ids_name, ids_documents
                    )
                    basic_analysis = self._analyze_structure(ids_documents)

                    # Combine analyses
                    structure_dict = {
                        **basic_analysis,
                        **graph_analysis.get("metrics", {}),
                    }
                    sample_paths = graph_analysis.get(
                        "key_paths",
                        [doc.metadata.path_name for doc in ids_documents[:10]],
                    )
                    max_depth = graph_analysis.get(
                        "max_depth", basic_analysis.get("max_depth", 0)
                    )

                    description = (
                        f"Real-time structural analysis of {ids_name} IDS: "
                        f"{graph_analysis.get('summary', f'Contains {len(ids_documents)} data paths with complex hierarchical organization')}"
                    )
                else:
                    # Last resort - basic fallback
                    structure_dict = {
                        "total_paths": 0,
                        "message": "No analysis data available",
                    }
                    sample_paths = []
                    max_depth = 0
                    description = f"Limited analysis available for {ids_name} IDS"

            # Get physics context
            from imas_mcp.models.physics_models import PhysicsSearchResult

            physics_context = PhysicsSearchResult(
                query=ids_name,
                physics_matches=[],
                concept_suggestions=[],
                unit_suggestions=[],
                symbol_suggestions=[],
                imas_path_suggestions=[],
            )

            # Add mermaid graph references to the description
            mermaid_info = (
                f"\n\n## Visual Structure Analysis\n"
                f"For visual exploration of this IDS structure, use these mermaid resources:\n"
                f"- **Hierarchy**: `mermaid://hierarchy/{ids_name}` - Complete structural hierarchy\n"
                f"- **Physics Domains**: `mermaid://physics/{ids_name}` - Domain organization\n"
                f"- **Complexity**: `mermaid://complexity/{ids_name}` - Complexity analysis\n"
                f"- **Overview**: `mermaid://overview` - All IDS relationships"
            )
            description += mermaid_info

            # Add clustering insights to the analysis
            clustering_enhancement = {}
            if ids_documents:
                clustering_enhancement = self._enhance_structure_with_clustering(
                    ids_name, ids_documents
                )
                if clustering_enhancement:
                    cluster_insights = clustering_enhancement.get(
                        "clustering_insights", {}
                    )
                    cluster_summary = cluster_insights.get("cluster_participation", {})
                    total_clusters = len(
                        cluster_summary.get("participating_clusters", [])
                    )
                    avg_similarity = cluster_summary.get("avg_similarity_score", 0.0)

                    if total_clusters > 0:
                        description += "\n\n## Clustering Analysis\n"
                        description += f"This IDS participates in {total_clusters} similarity clusters with average similarity {avg_similarity:.3f}. "
                        description += f"Structural coherence: {cluster_insights.get('structural_patterns', {}).get('structural_coherence', 'unknown')}."

            result = StructureResult(
                ids_name=ids_name,
                description=description,
                structure={**structure_dict, **clustering_enhancement},
                sample_paths=sample_paths,
                max_depth=max_depth,
                analysis=structure_analysis,
                physics_context=physics_context,
                query=ids_name,
            )

            logger.info(f"Structure analysis completed for {ids_name}")
            return result

        except Exception as e:
            logger.error(f"Structure analysis failed for {ids_name}: {e}")
            return self._create_error_response(f"Analysis failed: {e}", ids_name)

    def _analyze_structure(self, ids_documents):
        """Analyze structure of IDS documents."""
        paths = [doc.metadata.path_name for doc in ids_documents]

        # Analyze identifier schemas
        identifier_nodes = []
        for doc in ids_documents:
            identifier_schema = doc.raw_data.get("identifier_schema")
            if identifier_schema and isinstance(identifier_schema, dict):
                options = identifier_schema.get("options", [])
                identifier_nodes.append(
                    {
                        "path": doc.metadata.path_name,
                        "option_count": len(options),
                    }
                )

        # Build structure analysis
        structure_data = {
            "root_level_paths": len([p for p in paths if "/" not in p.strip("/")]),
            "max_depth": max(len(p.split("/")) for p in paths) if paths else 0,
            "document_count": len(ids_documents),
            "identifier_nodes": len(identifier_nodes),
            "branching_complexity": sum(
                node["option_count"] for node in identifier_nodes
            ),
        }

        return structure_data

    def _perform_graph_analysis(self, ids_name: str, ids_documents) -> dict[str, Any]:
        """Perform enhanced graph analysis using the graph analyzer."""
        try:
            # Convert documents to the format expected by graph analyzer
            paths = {}
            for doc in ids_documents:
                path_data = {
                    "data_type": getattr(doc.raw_data, "data_type", ""),
                    "units": getattr(doc.raw_data, "units", ""),
                    "physics_context": {
                        "domain": getattr(doc.metadata, "physics_domain", "unspecified")
                    },
                }

                # Add coordinate information if available
                if hasattr(doc.raw_data, "coordinates"):
                    path_data["coordinates"] = doc.raw_data.coordinates
                if hasattr(doc.raw_data, "coordinate1"):
                    path_data["coordinate1"] = doc.raw_data.coordinate1
                if hasattr(doc.raw_data, "coordinate2"):
                    path_data["coordinate2"] = doc.raw_data.coordinate2

                paths[doc.metadata.path_name] = path_data

            # Perform graph analysis
            graph_stats = self.graph_analyzer.analyze_ids_structure(ids_name, paths)

            # Extract key insights
            basic_metrics = graph_stats.get("basic_metrics", {})
            hierarchy_metrics = graph_stats.get("hierarchy_metrics", {})
            branching_metrics = graph_stats.get("branching_metrics", {})
            complexity_indicators = graph_stats.get("complexity_indicators", {})
            key_nodes = graph_stats.get("key_nodes", {})

            # Build summary
            total_nodes = basic_metrics.get("total_nodes", 0)
            max_depth = hierarchy_metrics.get("max_depth", 0)
            avg_branching = branching_metrics.get("avg_branching_factor", 0)
            array_ratio = complexity_indicators.get("array_ratio", 0)

            if total_nodes < 50:
                complexity_desc = "simple structure"
            elif total_nodes < 200:
                complexity_desc = "moderate complexity"
            else:
                complexity_desc = "high complexity"

            summary = (
                f"{complexity_desc} with {total_nodes} nodes, "
                f"{max_depth} levels deep, "
                f"average branching factor {avg_branching:.1f}"
            )

            if array_ratio > 0.3:
                summary += ", significant array structures"

            # Get key paths from most connected nodes
            key_paths = []
            for node_info in key_nodes.get("most_connected", []):
                key_paths.append(node_info.get("node", ""))

            # Add deepest paths
            deepest_paths = key_nodes.get("deepest_paths", [])
            key_paths.extend(deepest_paths[:5])

            return {
                "summary": summary,
                "metrics": {
                    "total_nodes": total_nodes,
                    "graph_density": basic_metrics.get("density", 0),
                    "avg_clustering": basic_metrics.get("avg_clustering", 0),
                    "max_depth": max_depth,
                    "avg_depth": hierarchy_metrics.get("avg_depth", 0),
                    "avg_branching_factor": avg_branching,
                    "max_branching_factor": branching_metrics.get(
                        "max_branching_factor", 0
                    ),
                    "array_ratio": array_ratio,
                    "time_dependent_ratio": complexity_indicators.get(
                        "time_dependent_ratio", 0
                    ),
                },
                "key_paths": key_paths[:10],
                "max_depth": max_depth,
            }

        except Exception as e:
            logger.error(f"Graph analysis failed for {ids_name}: {e}")
            return {
                "summary": "graph analysis unavailable",
                "metrics": {},
                "key_paths": [],
                "max_depth": 0,
            }
