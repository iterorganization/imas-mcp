"""
Enhanced IDS Structure Analyzer for comprehensive hierarchical analysis.

This module provides detailed structural analysis capabilities for IMAS IDS,
including hierarchical organization, physics domain mapping, and navigation
optimization for the analyze_ids_structure tool.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from imas_mcp.definitions import load_definition_file
from imas_mcp.graph_analyzer import IMASGraphAnalyzer

logger = logging.getLogger(__name__)


class IDSStructureAnalyzer:
    """Enhanced IDS structure analysis engine with physics domain integration."""

    def __init__(self, definitions_path: Path | None = None):
        """Initialize structure analyzer with physics and hierarchy definitions."""
        self.definitions_path = (
            definitions_path or Path(__file__).parent.parent / "definitions"
        )
        self.graph_analyzer = IMASGraphAnalyzer()

        # Load analysis configurations
        self._hierarchy_patterns = self._load_hierarchy_patterns()
        self._physics_domain_mapping = self._load_physics_domain_mapping()
        self._navigation_templates = self._load_navigation_templates()
        self._complexity_thresholds = self._load_complexity_thresholds()

    def _load_hierarchy_patterns(self) -> dict[str, Any]:
        """Load hierarchy pattern definitions from YAML."""
        try:
            return load_definition_file("imas/hierarchy_patterns.yaml")
        except Exception as e:
            logger.warning(f"Failed to load hierarchy patterns: {e}")
            return self._default_hierarchy_patterns()

    def _load_physics_domain_mapping(self) -> dict[str, Any]:
        """Load physics domain mapping definitions from YAML."""
        try:
            return load_definition_file("physics/domains/ids_mapping.yaml")
        except Exception as e:
            logger.warning(f"Failed to load physics domain mapping: {e}")
            return self._default_physics_domain_mapping()

    def _load_navigation_templates(self) -> dict[str, Any]:
        """Load navigation template definitions from YAML."""
        try:
            return load_definition_file("imas/navigation_templates.yaml")
        except Exception as e:
            logger.warning(f"Failed to load navigation templates: {e}")
            return self._default_navigation_templates()

    def _load_complexity_thresholds(self) -> dict[str, Any]:
        """Load complexity threshold definitions from YAML."""
        try:
            return load_definition_file("imas/complexity_thresholds.yaml")
        except Exception as e:
            logger.warning(f"Failed to load complexity thresholds: {e}")
            return self._default_complexity_thresholds()

    def _default_hierarchy_patterns(self) -> dict[str, Any]:
        """Default hierarchy patterns for fallback."""
        return {
            "common_patterns": {
                "time_slice": {
                    "pattern": "time_slice",
                    "description": "Time-dependent data structure",
                },
                "profiles_1d": {
                    "pattern": "profiles_1d",
                    "description": "1D profile data",
                },
                "profiles_2d": {
                    "pattern": "profiles_2d",
                    "description": "2D profile data",
                },
                "global_quantities": {
                    "pattern": "global",
                    "description": "Global scalar quantities",
                },
                "channels": {
                    "pattern": "channel",
                    "description": "Diagnostic channel data",
                },
                "coordinates": {
                    "pattern": "coordinate",
                    "description": "Coordinate system data",
                },
            },
            "physics_indicators": {
                "equilibrium": ["psi", "q", "safety_factor", "flux"],
                "transport": ["diffusivity", "conductivity", "pinch"],
                "heating": ["power", "efficiency", "absorption"],
                "mhd": ["mode", "amplitude", "frequency"],
                "diagnostics": ["signal", "calibration", "validity"],
            },
        }

    def _default_physics_domain_mapping(self) -> dict[str, Any]:
        """Default physics domain mapping for fallback."""
        return {
            "domains": {
                "equilibrium": {
                    "paths": ["equilibrium", "psi", "q_profile", "flux"],
                    "description": "Plasma equilibrium and magnetic configuration",
                },
                "transport": {
                    "paths": ["transport", "diffusivity", "conductivity", "flux"],
                    "description": "Particle and energy transport",
                },
                "heating": {
                    "paths": ["heating", "power", "nbi", "ecrh", "icrh"],
                    "description": "Auxiliary heating systems",
                },
                "mhd": {
                    "paths": ["mhd", "mode", "instability", "sawteeth"],
                    "description": "Magnetohydrodynamic phenomena",
                },
                "diagnostics": {
                    "paths": [
                        "thomson",
                        "interferometry",
                        "reflectometry",
                        "spectroscopy",
                    ],
                    "description": "Diagnostic measurement systems",
                },
            }
        }

    def _default_navigation_templates(self) -> dict[str, Any]:
        """Default navigation templates for fallback."""
        return {
            "common_workflows": {
                "basic_exploration": {
                    "steps": ["overview", "key_branches", "sample_paths"],
                    "description": "Standard IDS exploration workflow",
                },
                "physics_analysis": {
                    "steps": [
                        "domain_identification",
                        "measurement_paths",
                        "coordinate_systems",
                    ],
                    "description": "Physics-focused analysis workflow",
                },
                "data_access": {
                    "steps": ["entry_points", "array_structures", "time_dependencies"],
                    "description": "Data access optimization workflow",
                },
            }
        }

    def _default_complexity_thresholds(self) -> dict[str, Any]:
        """Default complexity thresholds for fallback."""
        return {
            "thresholds": {
                "depth": {"low": 3, "medium": 6, "high": 10},
                "branching": {"low": 5, "medium": 15, "high": 30},
                "nodes": {"small": 50, "medium": 200, "large": 500},
                "arrays": {"few": 0.1, "some": 0.3, "many": 0.6},
            }
        }

    def analyze_ids_structure(
        self, ids_name: str, ids_documents: list[Any]
    ) -> dict[str, Any]:
        """
        Comprehensive structural analysis of an IDS with enhanced hierarchy and physics integration.

        Args:
            ids_name: Name of the IDS to analyze
            ids_documents: List of IDS documents/paths

        Returns:
            Comprehensive structure analysis including hierarchy, physics domains, and navigation data
        """
        if not ids_documents:
            return self._create_empty_analysis(ids_name)

        # Convert documents to paths dict for graph analysis
        paths = {}
        for doc in ids_documents:
            path_name = getattr(doc.metadata, "path_name", str(doc))
            paths[path_name] = getattr(doc, "raw_data", {})

        # Perform base graph analysis
        graph_analysis = self.graph_analyzer.analyze_ids_structure(ids_name, paths)

        # Build hierarchy tree
        hierarchy_tree = self._build_hierarchy_tree(ids_documents)

        # Analyze physics domain distribution
        physics_domains = self._analyze_physics_domains(ids_documents)

        # Generate navigation optimization data
        navigation_data = self._generate_navigation_data(ids_documents, graph_analysis)

        # Calculate enhanced complexity metrics
        complexity_metrics = self._calculate_enhanced_complexity(
            ids_documents, graph_analysis
        )

        # Identify key structural patterns
        structural_patterns = self._identify_structural_patterns(ids_documents)

        return {
            "basic_structure": graph_analysis,
            "hierarchy_tree": hierarchy_tree,
            "physics_domains": physics_domains,
            "navigation_optimization": navigation_data,
            "enhanced_complexity": complexity_metrics,
            "structural_patterns": structural_patterns,
            "analysis_metadata": {
                "total_documents": len(ids_documents),
                "analysis_completeness": self._assess_analysis_completeness(
                    ids_documents
                ),
                "recommendation_confidence": self._calculate_confidence_score(
                    ids_documents
                ),
            },
        }

    def _create_empty_analysis(self, ids_name: str) -> dict[str, Any]:
        """Create analysis structure for empty IDS."""
        return {
            "basic_structure": {},
            "hierarchy_tree": {"nodes": [], "levels": {}, "relationships": []},
            "physics_domains": [],
            "navigation_optimization": {"entry_points": [], "recommended_paths": []},
            "enhanced_complexity": {"score": 0, "category": "empty"},
            "structural_patterns": [],
            "analysis_metadata": {
                "total_documents": 0,
                "analysis_completeness": 0.0,
                "recommendation_confidence": 0.0,
            },
        }

    def _build_hierarchy_tree(self, ids_documents: list[Any]) -> dict[str, Any]:
        """Build detailed hierarchical tree structure with node relationships."""
        tree_nodes = []
        level_mapping = defaultdict(list)
        parent_child_relationships = []

        # Extract paths and build tree structure
        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))
            parts = path.split("/")

            # Create nodes for each level
            for i, part in enumerate(parts):
                node_path = "/".join(parts[: i + 1])
                level = i

                # Node metadata
                node_info = {
                    "path": node_path,
                    "name": part,
                    "level": level,
                    "is_leaf": i == len(parts) - 1,
                    "parent": "/".join(parts[:i]) if i > 0 else None,
                    "children_count": 0,  # Will be calculated
                    "physics_indicators": self._extract_physics_indicators(part),
                    "data_characteristics": self._analyze_node_characteristics(
                        doc, node_path
                    ),
                }

                if node_info not in tree_nodes:
                    tree_nodes.append(node_info)
                    level_mapping[level].append(node_path)

                # Track parent-child relationships
                if i > 0:
                    parent_path = "/".join(parts[:i])
                    relationship = {
                        "parent": parent_path,
                        "child": node_path,
                        "relationship_type": self._classify_relationship(part),
                    }
                    if relationship not in parent_child_relationships:
                        parent_child_relationships.append(relationship)

        # Calculate children counts
        child_counts = defaultdict(int)
        for rel in parent_child_relationships:
            child_counts[rel["parent"]] += 1

        for node in tree_nodes:
            node["children_count"] = child_counts.get(node["path"], 0)

        return {
            "nodes": tree_nodes,
            "levels": dict(level_mapping),
            "relationships": parent_child_relationships,
            "tree_metrics": {
                "total_nodes": len(tree_nodes),
                "max_depth": max(level_mapping.keys()) if level_mapping else 0,
                "leaf_nodes": len([n for n in tree_nodes if n["is_leaf"]]),
                "branching_points": len(
                    [n for n in tree_nodes if n["children_count"] > 1]
                ),
            },
        }

    def _extract_physics_indicators(self, path_part: str) -> list[str]:
        """Extract physics domain indicators from path component."""
        indicators = []
        path_lower = path_part.lower()

        # Check against physics indicators from patterns
        physics_indicators = self._hierarchy_patterns.get("physics_indicators", {})
        for domain, keywords in physics_indicators.items():
            for keyword in keywords:
                if keyword.lower() in path_lower:
                    indicators.append(domain)

        return list(set(indicators))

    def _analyze_node_characteristics(self, doc: Any, node_path: str) -> dict[str, Any]:
        """Analyze characteristics of a tree node."""
        characteristics = {
            "has_coordinates": False,
            "is_time_dependent": False,
            "is_array": False,
            "measurement_type": None,
            "data_quality_indicators": [],
        }

        try:
            raw_data = getattr(doc, "raw_data", {})

            # Check for coordinates
            if any(
                key in raw_data for key in ["coordinates", "coordinate1", "coordinate2"]
            ):
                characteristics["has_coordinates"] = True

            # Check for time dependence
            if "time" in node_path.lower() or any(
                "time" in str(v).lower() for v in raw_data.values()
            ):
                characteristics["is_time_dependent"] = True

            # Check if it's an array structure
            coords = raw_data.get("coordinates", [])
            if coords or raw_data.get("coordinate1") or raw_data.get("coordinate2"):
                characteristics["is_array"] = True

            # Determine measurement type
            characteristics["measurement_type"] = self._classify_measurement_type(
                node_path, raw_data
            )

            # Check data quality indicators
            if "validity" in raw_data or "validity_timed" in raw_data:
                characteristics["data_quality_indicators"].append("validity_checking")
            if "uncertainty" in raw_data:
                characteristics["data_quality_indicators"].append(
                    "uncertainty_quantification"
                )

        except Exception as e:
            logger.debug(f"Error analyzing node characteristics for {node_path}: {e}")

        return characteristics

    def _classify_relationship(self, path_part: str) -> str:
        """Classify the type of parent-child relationship."""
        part_lower = path_part.lower()

        if part_lower in ["time_slice", "time"]:
            return "temporal"
        elif part_lower in ["profiles_1d", "profiles_2d"]:
            return "spatial_profile"
        elif "channel" in part_lower:
            return "diagnostic_channel"
        elif part_lower in ["coordinate1", "coordinate2", "coordinates"]:
            return "coordinate_system"
        elif part_lower.isdigit():
            return "indexed_collection"
        else:
            return "hierarchical"

    def _classify_measurement_type(self, path: str, raw_data: dict) -> str | None:
        """Classify the type of measurement based on path and data."""
        path_lower = path.lower()

        measurement_patterns = {
            "temperature": ["temperature", "temp", "t_e", "t_i"],
            "density": ["density", "n_e", "n_i", "concentration"],
            "pressure": ["pressure", "p_e", "p_i"],
            "magnetic_field": ["b_field", "magnetic", "b_tor", "b_pol"],
            "current": ["current", "j_", "current_density"],
            "power": ["power", "heating", "absorption"],
            "flux": ["flux", "particle_flux", "heat_flux"],
            "frequency": ["frequency", "freq", "omega"],
            "velocity": ["velocity", "flow", "rotation"],
        }

        for measurement_type, patterns in measurement_patterns.items():
            if any(pattern in path_lower for pattern in patterns):
                return measurement_type

        return None

    def _analyze_physics_domains(
        self, ids_documents: list[Any]
    ) -> list[dict[str, Any]]:
        """Analyze physics domain distribution within the IDS structure."""
        domain_analysis = []
        domain_paths = defaultdict(list)

        # Get domain mapping
        domains = self._physics_domain_mapping.get("domains", {})

        # Analyze each document for physics domain indicators
        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))
            path_lower = path.lower()

            # Check which domains this path belongs to
            for domain_name, domain_info in domains.items():
                domain_paths_list = domain_info.get("paths", [])
                if any(
                    domain_path.lower() in path_lower
                    for domain_path in domain_paths_list
                ):
                    domain_paths[domain_name].append(
                        {
                            "path": path,
                            "relevance_score": self._calculate_domain_relevance(
                                path, domain_paths_list
                            ),
                            "measurement_type": self._classify_measurement_type(
                                path, getattr(doc, "raw_data", {})
                            ),
                        }
                    )

        # Build domain analysis
        for domain_name, paths in domain_paths.items():
            if paths:  # Only include domains with actual paths
                domain_info = domains.get(domain_name, {})
                domain_analysis.append(
                    {
                        "domain": domain_name,
                        "description": domain_info.get(
                            "description", f"{domain_name} physics domain"
                        ),
                        "path_count": len(paths),
                        "coverage_percentage": round(
                            (len(paths) / len(ids_documents)) * 100, 1
                        ),
                        "representative_paths": sorted(
                            paths, key=lambda x: x["relevance_score"], reverse=True
                        )[:5],
                        "measurement_types": list(
                            {
                                p["measurement_type"]
                                for p in paths
                                if p["measurement_type"]
                            }
                        ),
                        "domain_complexity": self._assess_domain_complexity(paths),
                    }
                )

        return sorted(domain_analysis, key=lambda x: x["path_count"], reverse=True)

    def _calculate_domain_relevance(self, path: str, domain_paths: list[str]) -> float:
        """Calculate how relevant a path is to a physics domain."""
        path_lower = path.lower()
        relevance_score = 0.0

        for domain_path in domain_paths:
            domain_path_lower = domain_path.lower()
            if domain_path_lower in path_lower:
                # Higher score for exact matches or matches in path components
                path_parts = path_lower.split("/")
                if domain_path_lower in path_parts:
                    relevance_score += 1.0  # Exact component match
                else:
                    relevance_score += 0.5  # Substring match

        return min(relevance_score, 1.0)  # Cap at 1.0

    def _assess_domain_complexity(self, domain_paths: list[dict]) -> str:
        """Assess the complexity of a physics domain within the IDS."""
        path_count = len(domain_paths)
        unique_measurements = len(
            {p["measurement_type"] for p in domain_paths if p["measurement_type"]}
        )

        if path_count >= 20 and unique_measurements >= 5:
            return "high"
        elif path_count >= 10 and unique_measurements >= 3:
            return "medium"
        else:
            return "low"

    def _generate_navigation_data(
        self, ids_documents: list[Any], graph_analysis: dict
    ) -> dict[str, Any]:
        """Generate navigation optimization data for the IDS."""
        # Identify key entry points
        entry_points = self._identify_entry_points(ids_documents)

        # Generate recommended access paths
        recommended_paths = self._generate_recommended_paths(
            ids_documents, graph_analysis
        )

        # Create navigation patterns
        navigation_patterns = self._extract_navigation_patterns(ids_documents)

        # Drill-down capabilities
        drill_down_maps = self._create_drill_down_maps(ids_documents)

        return {
            "entry_points": entry_points,
            "recommended_paths": recommended_paths,
            "navigation_patterns": navigation_patterns,
            "drill_down_maps": drill_down_maps,
            "optimization_suggestions": self._generate_optimization_suggestions(
                ids_documents, graph_analysis
            ),
        }

    def _identify_entry_points(self, ids_documents: list[Any]) -> list[dict[str, Any]]:
        """Identify optimal entry points for IDS exploration."""
        entry_points = []
        path_frequencies = defaultdict(int)

        # Analyze path patterns to find common entry points
        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))
            parts = path.split("/")

            # Consider top-level and second-level paths as potential entry points
            for i in range(min(3, len(parts))):
                partial_path = "/".join(parts[: i + 1])
                path_frequencies[partial_path] += 1

        # Select entry points based on frequency and importance
        for path, frequency in sorted(
            path_frequencies.items(), key=lambda x: x[1], reverse=True
        ):
            if frequency >= 3:  # Threshold for entry point consideration
                entry_points.append(
                    {
                        "path": path,
                        "frequency": frequency,
                        "importance_score": self._calculate_entry_point_importance(
                            path
                        ),
                        "description": self._describe_entry_point(path),
                        "recommended_usage": self._get_entry_point_usage(path),
                    }
                )

        return entry_points[:10]  # Return top 10 entry points

    def _calculate_entry_point_importance(self, path: str) -> float:
        """Calculate importance score for an entry point."""
        score = 0.0
        path_lower = path.lower()

        # Higher scores for common important paths
        important_patterns = {
            "time_slice": 0.9,
            "profiles_1d": 0.8,
            "global": 0.7,
            "channel": 0.6,
            "coordinate": 0.5,
        }

        for pattern, weight in important_patterns.items():
            if pattern in path_lower:
                score += weight

        return min(score, 1.0)

    def _describe_entry_point(self, path: str) -> str:
        """Provide description for an entry point."""
        path_lower = path.lower()

        descriptions = {
            "time_slice": "Time-dependent data access point",
            "profiles_1d": "1D profile data structure",
            "profiles_2d": "2D profile data structure",
            "global": "Global scalar quantities",
            "channel": "Diagnostic channel data",
            "coordinate": "Coordinate system definitions",
        }

        for pattern, description in descriptions.items():
            if pattern in path_lower:
                return description

        return f"Data access point for {path.split('/')[-1]} data"

    def _get_entry_point_usage(self, path: str) -> str:
        """Get recommended usage pattern for entry point."""
        path_lower = path.lower()

        usage_patterns = {
            "time_slice": "Access time-dependent data by iterating through time slices",
            "profiles_1d": "Extract radial or other 1D profiles for analysis",
            "global": "Retrieve scalar quantities for global plasma parameters",
            "channel": "Access diagnostic measurements from specific channels",
        }

        for pattern, usage in usage_patterns.items():
            if pattern in path_lower:
                return usage

        return "Standard hierarchical data access"

    def _generate_recommended_paths(
        self, ids_documents: list[Any], graph_analysis: dict
    ) -> list[dict[str, Any]]:
        """Generate recommended access paths based on usage patterns."""
        recommended = []

        # Group paths by common patterns
        pattern_groups = defaultdict(list)
        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))
            pattern = self._identify_path_pattern(path)
            pattern_groups[pattern].append(path)

        # Create recommendations for each pattern
        for pattern, paths in pattern_groups.items():
            if len(paths) >= 2:  # Only recommend patterns with multiple paths
                recommended.append(
                    {
                        "pattern": pattern,
                        "path_count": len(paths),
                        "example_paths": paths[:3],
                        "usage_description": self._get_pattern_usage_description(
                            pattern
                        ),
                        "typical_workflow": self._get_pattern_workflow(pattern),
                    }
                )

        return sorted(recommended, key=lambda x: x["path_count"], reverse=True)

    def _identify_path_pattern(self, path: str) -> str:
        """Identify the pattern type of a path."""
        path_lower = path.lower()

        if "time_slice" in path_lower:
            return "temporal_data"
        elif "profiles_1d" in path_lower:
            return "radial_profiles"
        elif "profiles_2d" in path_lower:
            return "spatial_profiles"
        elif "channel" in path_lower:
            return "diagnostic_channels"
        elif "global" in path_lower:
            return "global_quantities"
        else:
            return "hierarchical_data"

    def _get_pattern_usage_description(self, pattern: str) -> str:
        """Get usage description for a path pattern."""
        descriptions = {
            "temporal_data": "Time-evolving measurements across multiple time slices",
            "radial_profiles": "1D profiles typically as functions of radius or flux coordinates",
            "spatial_profiles": "2D spatial distributions and profile data",
            "diagnostic_channels": "Individual diagnostic channel measurements and calibrations",
            "global_quantities": "Scalar plasma parameters and integrated quantities",
            "hierarchical_data": "Structured data with hierarchical organization",
        }
        return descriptions.get(pattern, "Structured data access pattern")

    def _get_pattern_workflow(self, pattern: str) -> list[str]:
        """Get typical workflow steps for a pattern."""
        workflows = {
            "temporal_data": [
                "Select time slice index",
                "Access measurement data",
                "Check validity flags",
                "Extract coordinate information",
            ],
            "radial_profiles": [
                "Access profile data array",
                "Get coordinate grid",
                "Check measurement uncertainties",
                "Validate data quality",
            ],
            "diagnostic_channels": [
                "Select channel index",
                "Access signal data",
                "Apply calibration factors",
                "Check measurement validity",
            ],
        }
        return workflows.get(
            pattern, ["Access data", "Check validity", "Extract coordinates"]
        )

    def _extract_navigation_patterns(
        self, ids_documents: list[Any]
    ) -> list[dict[str, Any]]:
        """Extract common navigation patterns from the IDS structure."""
        patterns = []

        # Load navigation templates
        templates = self._navigation_templates.get("common_workflows", {})

        for workflow_name, workflow_info in templates.items():
            # Check if this workflow is applicable to the current IDS
            applicable_paths = self._find_applicable_paths(ids_documents, workflow_info)

            if applicable_paths:
                patterns.append(
                    {
                        "workflow": workflow_name,
                        "description": workflow_info.get("description", ""),
                        "steps": workflow_info.get("steps", []),
                        "applicable_paths": applicable_paths[:5],
                        "complexity": self._assess_workflow_complexity(
                            applicable_paths
                        ),
                    }
                )

        return patterns

    def _find_applicable_paths(
        self, ids_documents: list[Any], workflow_info: dict
    ) -> list[str]:
        """Find paths applicable to a specific workflow."""
        applicable = []
        steps = workflow_info.get("steps", [])

        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))
            path_lower = path.lower()

            # Check if path matches workflow steps
            for step in steps:
                if any(keyword in path_lower for keyword in step.split("_")):
                    applicable.append(path)
                    break

        return list(set(applicable))

    def _assess_workflow_complexity(self, applicable_paths: list[str]) -> str:
        """Assess complexity of a workflow based on applicable paths."""
        if len(applicable_paths) >= 10:
            return "high"
        elif len(applicable_paths) >= 5:
            return "medium"
        else:
            return "low"

    def _create_drill_down_maps(self, ids_documents: list[Any]) -> dict[str, Any]:
        """Create drill-down capability maps for interactive exploration."""
        drill_down_map = {}
        hierarchy_levels = defaultdict(list)

        # Build hierarchy levels
        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))
            parts = path.split("/")

            for i, part in enumerate(parts):
                level_key = f"level_{i}"
                if part not in hierarchy_levels[level_key]:
                    hierarchy_levels[level_key].append(part)

        # Create drill-down paths
        for level, items in hierarchy_levels.items():
            drill_down_map[level] = {
                "items": items,
                "item_count": len(items),
                "drill_down_options": self._get_drill_down_options(level, items),
            }

        return {
            "hierarchy_levels": dict(hierarchy_levels),
            "drill_down_map": drill_down_map,
            "interactive_features": self._define_interactive_features(hierarchy_levels),
        }

    def _get_drill_down_options(
        self, level: str, items: list[str]
    ) -> list[dict[str, Any]]:
        """Get drill-down options for a hierarchy level."""
        options = []

        for item in items[:10]:  # Limit to top 10 items
            options.append(
                {
                    "item": item,
                    "description": self._describe_hierarchy_item(item),
                    "has_children": self._check_has_children(item, level),
                    "suggested_filters": self._suggest_filters_for_item(item),
                }
            )

        return options

    def _describe_hierarchy_item(self, item: str) -> str:
        """Describe a hierarchy item."""
        item_lower = item.lower()

        descriptions = {
            "time_slice": "Time-dependent data slice",
            "profiles_1d": "1D profile data",
            "channel": "Diagnostic channel",
            "coordinate": "Coordinate system",
            "global": "Global quantities",
        }

        for pattern, description in descriptions.items():
            if pattern in item_lower:
                return description

        return f"Data structure: {item}"

    def _check_has_children(self, item: str, level: str) -> bool:
        """Check if hierarchy item has children (placeholder implementation)."""
        # This would need to be implemented based on actual data structure
        return True  # Simplified assumption

    def _suggest_filters_for_item(self, item: str) -> list[str]:
        """Suggest useful filters for a hierarchy item."""
        item_lower = item.lower()

        filter_suggestions = {
            "time": ["time_range", "time_slice_index"],
            "channel": ["channel_id", "validity"],
            "coordinate": ["coordinate_type", "grid_type"],
            "profile": ["radial_range", "quantity_type"],
        }

        for pattern, filters in filter_suggestions.items():
            if pattern in item_lower:
                return filters

        return ["data_validity", "time_range"]

    def _define_interactive_features(
        self, hierarchy_levels: dict
    ) -> list[dict[str, Any]]:
        """Define interactive features for the hierarchy."""
        features = [
            {
                "feature": "level_navigation",
                "description": "Navigate between hierarchy levels",
                "supported_levels": list(hierarchy_levels.keys()),
                "implementation": "breadcrumb_navigation",
            },
            {
                "feature": "path_filtering",
                "description": "Filter paths by patterns or physics domains",
                "filter_types": [
                    "physics_domain",
                    "measurement_type",
                    "data_structure",
                ],
                "implementation": "dynamic_filtering",
            },
            {
                "feature": "structure_search",
                "description": "Search within IDS structure",
                "search_types": ["path_name", "description", "physics_context"],
                "implementation": "incremental_search",
            },
        ]

        return features

    def _generate_optimization_suggestions(
        self, ids_documents: list[Any], graph_analysis: dict
    ) -> list[dict[str, Any]]:
        """Generate optimization suggestions for data access."""
        suggestions = []

        # Analyze structure characteristics
        total_docs = len(ids_documents)
        complexity_info = graph_analysis.get("complexity_indicators", {})

        # Performance suggestions
        if total_docs > 100:
            suggestions.append(
                {
                    "type": "performance",
                    "suggestion": "Use selective path access for large datasets",
                    "description": "Access specific paths rather than entire IDS to improve performance",
                    "implementation": "path_filtering",
                }
            )

        if complexity_info.get("array_ratio", 0) > 0.5:
            suggestions.append(
                {
                    "type": "data_access",
                    "suggestion": "Optimize array access patterns",
                    "description": "Pre-load coordinate systems for efficient array data access",
                    "implementation": "coordinate_caching",
                }
            )

        # Navigation suggestions
        max_depth = graph_analysis.get("hierarchy_metrics", {}).get("max_depth", 0)
        if max_depth > 6:
            suggestions.append(
                {
                    "type": "navigation",
                    "suggestion": "Use hierarchical entry points",
                    "description": "Deep hierarchy detected - use recommended entry points for efficient navigation",
                    "implementation": "entry_point_navigation",
                }
            )

        return suggestions

    def _calculate_enhanced_complexity(
        self, ids_documents: list[Any], graph_analysis: dict
    ) -> dict[str, Any]:
        """Calculate enhanced complexity metrics."""
        thresholds = self._complexity_thresholds.get("thresholds", {})

        # Get basic metrics
        total_docs = len(ids_documents)
        hierarchy_metrics = graph_analysis.get("hierarchy_metrics", {})
        max_depth = hierarchy_metrics.get("max_depth", 0)

        # Calculate complexity scores
        depth_score = self._score_complexity_dimension(
            max_depth, thresholds.get("depth", {})
        )
        size_score = self._score_complexity_dimension(
            total_docs, thresholds.get("nodes", {})
        )

        # Overall complexity score
        complexity_score = (depth_score + size_score) / 2

        # Complexity category
        if complexity_score >= 0.7:
            category = "high"
        elif complexity_score >= 0.4:
            category = "medium"
        else:
            category = "low"

        return {
            "score": round(complexity_score, 2),
            "category": category,
            "dimensions": {
                "depth_complexity": depth_score,
                "size_complexity": size_score,
            },
            "recommendations": self._get_complexity_recommendations(category),
        }

    def _score_complexity_dimension(self, value: int, thresholds: dict) -> float:
        """Score a complexity dimension against thresholds."""
        if not thresholds:
            return 0.5  # Default medium complexity

        low = thresholds.get("low", 0)
        medium = thresholds.get("medium", low * 2)
        high = thresholds.get("high", medium * 2)

        if value <= low:
            return 0.2
        elif value <= medium:
            return 0.5
        elif value <= high:
            return 0.8
        else:
            return 1.0

    def _get_complexity_recommendations(self, category: str) -> list[str]:
        """Get recommendations based on complexity category."""
        recommendations = {
            "low": [
                "Structure is simple and easy to navigate",
                "Direct path access is recommended",
            ],
            "medium": [
                "Use entry points for efficient navigation",
                "Consider path filtering for focused analysis",
            ],
            "high": [
                "Use recommended navigation patterns",
                "Leverage entry points and drill-down maps",
                "Consider selective data access strategies",
            ],
        }

        return recommendations.get(category, ["Standard navigation recommended"])

    def _identify_structural_patterns(
        self, ids_documents: list[Any]
    ) -> list[dict[str, Any]]:
        """Identify common structural patterns in the IDS."""
        patterns = []
        pattern_counts = defaultdict(int)

        # Analyze path patterns
        for doc in ids_documents:
            path = getattr(doc.metadata, "path_name", str(doc))

            # Identify common patterns
            if "time_slice" in path.lower():
                pattern_counts["temporal_structure"] += 1
            if "profiles_1d" in path.lower():
                pattern_counts["1d_profiles"] += 1
            if "profiles_2d" in path.lower():
                pattern_counts["2d_profiles"] += 1
            if "channel" in path.lower():
                pattern_counts["channel_structure"] += 1
            if "coordinate" in path.lower():
                pattern_counts["coordinate_systems"] += 1

        # Build pattern descriptions
        pattern_descriptions = {
            "temporal_structure": "Time-dependent data organization",
            "1d_profiles": "One-dimensional profile data structures",
            "2d_profiles": "Two-dimensional profile data structures",
            "channel_structure": "Diagnostic channel organization",
            "coordinate_systems": "Coordinate system definitions",
        }

        for pattern, count in pattern_counts.items():
            if count > 0:
                patterns.append(
                    {
                        "pattern": pattern,
                        "description": pattern_descriptions.get(pattern, pattern),
                        "occurrence_count": count,
                        "prevalence": round(count / len(ids_documents), 2)
                        if ids_documents
                        else 0,
                    }
                )

        return sorted(patterns, key=lambda x: x["occurrence_count"], reverse=True)

    def _assess_analysis_completeness(self, ids_documents: list[Any]) -> float:
        """Assess completeness of the analysis."""
        if not ids_documents:
            return 0.0

        # Check various aspects of completeness
        has_hierarchy = len(ids_documents) > 1
        has_physics_indicators = any(
            self._extract_physics_indicators(getattr(doc.metadata, "path_name", ""))
            for doc in ids_documents
        )
        has_coordinates = any(
            self._analyze_node_characteristics(
                doc, getattr(doc.metadata, "path_name", "")
            ).get("has_coordinates", False)
            for doc in ids_documents
        )

        completeness_factors = [has_hierarchy, has_physics_indicators, has_coordinates]
        return sum(completeness_factors) / len(completeness_factors)

    def _calculate_confidence_score(self, ids_documents: list[Any]) -> float:
        """Calculate confidence score for analysis recommendations."""
        if not ids_documents:
            return 0.0

        # Base confidence on data availability and structure consistency
        base_confidence = min(
            1.0, len(ids_documents) / 50
        )  # More documents = higher confidence

        # Adjust for data quality indicators
        quality_indicators = 0
        for doc in ids_documents:
            characteristics = self._analyze_node_characteristics(
                doc, getattr(doc.metadata, "path_name", "")
            )
            if characteristics.get("data_quality_indicators"):
                quality_indicators += 1

        quality_factor = (
            min(1.0, quality_indicators / len(ids_documents)) if ids_documents else 0
        )

        return round((base_confidence + quality_factor) / 2, 2)
