"""
Physics Domain Analyzer for enhanced domain-specific data extraction.

This module provides comprehensive analysis capabilities for physics domains,
including theoretical foundations, experimental methods, and cross-domain
relationships for the export_physics_domain tool.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from imas_mcp.definitions import load_definition_file

logger = logging.getLogger(__name__)


class PhysicsDomainAnalyzer:
    """Enhanced physics domain analysis engine."""

    def __init__(self, definitions_path: Path | None = None):
        """Initialize domain analyzer with physics definitions."""
        self.definitions_path = (
            definitions_path or Path(__file__).parent.parent / "definitions" / "physics"
        )
        self._domain_characteristics = self._load_domain_characteristics()
        self._domain_relationships = self._load_domain_relationships()
        self._measurement_methods = self._extract_measurement_methods()

        # Load additional definition files
        self._measurement_types = self._load_measurement_types()
        self._diagnostic_methods = self._load_diagnostic_methods()
        self._physics_contexts = self._load_physics_contexts()
        self._research_workflows = self._load_research_workflows()

    def _load_measurement_types(self) -> dict[str, Any]:
        """Load measurement type definitions from YAML."""
        try:
            return load_definition_file("physics/domains/measurement_types.yaml")
        except Exception as e:
            logger.error(f"Failed to load measurement types: {e}")
            return {}

    def _load_diagnostic_methods(self) -> dict[str, Any]:
        """Load diagnostic method definitions from YAML."""
        try:
            return load_definition_file("physics/domains/diagnostic_methods.yaml")
        except Exception as e:
            logger.error(f"Failed to load diagnostic methods: {e}")
            return {}

    def _load_physics_contexts(self) -> dict[str, Any]:
        """Load physics context definitions from YAML."""
        try:
            return load_definition_file("physics/domains/physics_contexts.yaml")
        except Exception as e:
            logger.error(f"Failed to load physics contexts: {e}")
            return {}

    def _load_research_workflows(self) -> dict[str, Any]:
        """Load research workflow definitions from YAML."""
        try:
            return load_definition_file("physics/domains/research_workflows.yaml")
        except Exception as e:
            logger.error(f"Failed to load research workflows: {e}")
            return {}

    def _load_domain_characteristics(self) -> dict[str, Any]:
        """Load physics domain characteristics from definitions."""
        try:
            characteristics_file = (
                self.definitions_path / "domains" / "characteristics.yaml"
            )
            if characteristics_file.exists():
                with open(characteristics_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    return data.get("domains", {})
            logger.warning(
                f"Domain characteristics file not found: {characteristics_file}"
            )
            return {}
        except Exception as e:
            logger.error(f"Failed to load domain characteristics: {e}")
            return {}

    def _load_domain_relationships(self) -> dict[str, list[str]]:
        """Load physics domain relationships."""
        try:
            relationships_file = (
                self.definitions_path / "domains" / "relationships.yaml"
            )
            if relationships_file.exists():
                with open(relationships_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    return data.get("relationships", {})
            logger.warning(f"Domain relationships file not found: {relationships_file}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load domain relationships: {e}")
            return {}

    def _extract_measurement_methods(self) -> dict[str, list[str]]:
        """Extract measurement methods for each domain."""
        methods = {}
        for domain, characteristics in self._domain_characteristics.items():
            methods[domain] = characteristics.get("measurement_methods", [])
        return methods

    def analyze_domain(
        self, domain: str, search_results: list[Any], depth: str = "focused"
    ) -> dict[str, Any]:
        """
        Perform comprehensive domain-specific analysis.

        Args:
            domain: Physics domain name
            search_results: Search results from the domain
            depth: Analysis depth ('overview', 'focused', 'comprehensive')

        Returns:
            Comprehensive domain analysis with measurements, theory, and workflows
        """
        # Get domain characteristics
        domain_info = self._domain_characteristics.get(domain, {})

        # Analyze search results
        analysis = {
            "key_measurements": self._extract_measurements(domain, search_results),
            "theoretical_foundations": self._get_theory_base(domain, domain_info),
            "experimental_methods": self._get_measurement_methods(domain, domain_info),
            "cross_domain_links": self._find_domain_bridges(domain),
            "typical_workflows": self._extract_workflows(domain, search_results),
            "data_characteristics": self._analyze_data_characteristics(search_results),
            "complexity_assessment": self._assess_complexity(domain, search_results),
        }

        # Adjust analysis based on depth
        if depth == "overview":
            analysis = self._filter_for_overview(analysis)
        elif depth == "comprehensive":
            analysis = self._expand_for_comprehensive(analysis, domain, search_results)

        return analysis

    def _extract_measurements(
        self, domain: str, search_results: list[Any]
    ) -> list[dict[str, Any]]:
        """Extract key measurements from domain search results."""
        measurements = []

        # Group results by measurement type
        measurement_groups = {}
        for result in search_results:
            # Extract measurement type from path or data type
            measurement_type = self._identify_measurement_type(result)
            if measurement_type not in measurement_groups:
                measurement_groups[measurement_type] = []
            measurement_groups[measurement_type].append(result)

        # Build measurement descriptions
        for mtype, results in measurement_groups.items():
            measurement = {
                "measurement_type": mtype,
                "paths": [r.path for r in results[:3]],  # Limit paths per type
                "typical_units": list({r.units for r in results if r.units})[:3],
                "description": self._describe_measurement(mtype, domain),
                "data_count": len(results),
            }
            measurements.append(measurement)

        return measurements[:10]  # Limit total measurements

    def _identify_measurement_type(self, result: Any) -> str:
        """Identify measurement type from search result using YAML definitions."""
        path = result.path.lower()

        # Get measurement types from YAML
        measurement_types = self._measurement_types.get("measurement_types", {})

        # Check each measurement type for keyword matches
        for mtype, definition in measurement_types.items():
            keywords = definition.get("identification_keywords", [])
            if any(keyword in path for keyword in keywords):
                return mtype

        # Fallback to data type or default
        return result.data_type or "general_measurement"

    def _describe_measurement(self, measurement_type: str, domain: str) -> str:
        """Provide physics-based description of measurement type using YAML definitions."""
        measurement_types = self._measurement_types.get("measurement_types", {})

        if measurement_type in measurement_types:
            template = measurement_types[measurement_type].get(
                "description_template", ""
            )
            return template.format(domain=domain)

        # Fallback to default
        default_template = self._measurement_types.get("default_measurement", {}).get(
            "description_template", ""
        )
        if default_template:
            return default_template.format(domain=domain)

        return f"Measurements related to {domain} physics"

    def _get_theory_base(
        self, domain: str, domain_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Get theoretical foundations for the domain."""
        phenomena = domain_info.get("primary_phenomena", [])
        complexity = domain_info.get("complexity_level", "intermediate")

        return {
            "primary_phenomena": phenomena,
            "complexity_level": complexity,
            "description": domain_info.get("description", f"Physics domain: {domain}"),
            "theoretical_context": self._build_theoretical_context(domain, phenomena),
        }

    def _build_theoretical_context(self, domain: str, phenomena: list[str]) -> str:
        """Build theoretical physics context for domain using YAML definitions."""
        contexts = self._physics_contexts.get("theoretical_contexts", {})

        if domain in contexts:
            context_info = contexts[domain]
            base_context = context_info.get("description", f"Physics domain: {domain}")
        else:
            default_template = self._physics_contexts.get("default_context", {}).get(
                "description_template", ""
            )
            base_context = (
                default_template.format(domain=domain)
                if default_template
                else f"Fundamental physics principles governing {domain}"
            )

        if phenomena:
            phenomena_str = ", ".join(phenomena[:3])
            return f"{base_context}. Key phenomena: {phenomena_str}"
        return base_context

    def _get_measurement_methods(
        self, domain: str, domain_info: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get experimental measurement methods for domain."""
        methods = domain_info.get("measurement_methods", [])

        detailed_methods = []
        for method in methods:
            method_info = {
                "method": method,
                "description": self._describe_measurement_method(method, domain),
                "typical_outputs": self._get_method_outputs(method),
                "applicability": self._assess_method_applicability(method, domain),
            }
            detailed_methods.append(method_info)

        return detailed_methods

    def _describe_measurement_method(self, method: str, domain: str) -> str:
        """Describe measurement method in physics context using YAML definitions."""
        diagnostic_methods = self._diagnostic_methods.get("diagnostic_methods", {})

        if method in diagnostic_methods:
            return diagnostic_methods[method].get(
                "description", f"Diagnostic method used in {domain} physics"
            )

        return f"Diagnostic method used in {domain} physics"

    def _get_method_outputs(self, method: str) -> list[str]:
        """Get typical measurement outputs for method using YAML definitions."""
        diagnostic_methods = self._diagnostic_methods.get("diagnostic_methods", {})

        if method in diagnostic_methods:
            return diagnostic_methods[method].get(
                "typical_outputs", ["measurement_data"]
            )

        return ["measurement_data"]

    def _assess_method_applicability(self, method: str, domain: str) -> str:
        """Assess method applicability to domain using YAML definitions."""
        diagnostic_methods = self._diagnostic_methods.get("diagnostic_methods", {})

        # Check for essential applicability
        if method in diagnostic_methods:
            high_applicability_domains = diagnostic_methods[method].get(
                "high_applicability_domains", []
            )
            if domain in high_applicability_domains:
                return "essential"

        # Check domain defaults
        domain_defaults = self._diagnostic_methods.get(
            "domain_applicability_defaults", {}
        )
        if domain in domain_defaults:
            return domain_defaults[domain].get("default_level", "low")

        # General default
        return domain_defaults.get("default", {}).get("default_level", "low")

    def _find_domain_bridges(self, domain: str) -> list[dict[str, Any]]:
        """Find cross-domain relationship bridges."""
        related_domains = self._domain_relationships.get(domain, [])

        bridges = []
        for related_domain in related_domains:
            bridge = {
                "target_domain": related_domain,
                "relationship_type": self._classify_domain_relationship(
                    domain, related_domain
                ),
                "physics_connection": self._describe_physics_connection(
                    domain, related_domain
                ),
                "shared_measurements": self._find_shared_measurements(
                    domain, related_domain
                ),
            }
            bridges.append(bridge)

        return bridges

    def _classify_domain_relationship(self, domain1: str, domain2: str) -> str:
        """Classify the type of relationship between domains using YAML definitions."""
        physics_connections = self._physics_contexts.get("physics_connections", {})

        # Create connection key (sorted for consistency)
        connection_key = f"{min(domain1, domain2)}_{max(domain1, domain2)}"

        # Check specific connections
        if connection_key in physics_connections:
            return physics_connections[connection_key].get(
                "relationship_type", "correlative"
            )

        # Default to correlative
        return "correlative"

    def _describe_physics_connection(self, domain1: str, domain2: str) -> str:
        """Describe physics connection between domains using YAML definitions."""
        physics_connections = self._physics_contexts.get("physics_connections", {})

        # Create connection key (sorted for consistency)
        connection_key = f"{min(domain1, domain2)}_{max(domain1, domain2)}"

        if connection_key in physics_connections:
            return physics_connections[connection_key].get(
                "description", f"Physics coupling between {domain1} and {domain2}"
            )

        return f"Physics coupling between {domain1} and {domain2}"

    def _find_shared_measurements(self, domain1: str, domain2: str) -> list[str]:
        """Find measurements shared between domains."""
        domain1_methods = set(self._measurement_methods.get(domain1, []))
        domain2_methods = set(self._measurement_methods.get(domain2, []))

        return list(domain1_methods.intersection(domain2_methods))

    def _extract_workflows(
        self, domain: str, search_results: list[Any]
    ) -> list[dict[str, Any]]:
        """Extract typical analysis workflows for domain using YAML definitions."""
        workflows = []

        # Get standard workflows from YAML
        standard_workflows = self._research_workflows.get("standard_workflows", {})

        # Add domain-specific workflows
        for workflow_name, workflow_info in standard_workflows.items():
            if workflow_info.get("domain") == domain:
                workflows.append(
                    {
                        "workflow_name": workflow_name,
                        "description": workflow_info.get("description", ""),
                        "typical_steps": workflow_info.get("typical_steps", []),
                        "data_requirements": workflow_info.get("data_requirements", []),
                        "output_products": workflow_info.get("output_products", []),
                    }
                )

        # Add inferred workflow based on available paths
        if search_results:
            workflows.append(self._infer_workflow_from_paths(domain, search_results))

        return workflows

    def _infer_workflow_from_paths(
        self, domain: str, search_results: list[Any]
    ) -> dict[str, Any]:
        """Infer workflow from available data paths using YAML template."""
        path_types = set()
        for result in search_results:
            path_type = self._identify_measurement_type(result)
            path_types.add(path_type)

        # Get generic template from YAML
        template = self._research_workflows.get("generic_workflow_template", {})

        return {
            "workflow_name": f"{domain}_data_analysis",
            "description": template.get(
                "description_template", "Standard data analysis workflow for {domain}"
            ).format(domain=domain),
            "typical_steps": template.get(
                "typical_steps",
                ["data_validation", "preprocessing", "analysis", "interpretation"],
            ),
            "available_data_types": list(path_types),
            "output_products": template.get(
                "output_products", ["processed_data", "analysis_results"]
            ),
        }

    def _analyze_data_characteristics(
        self, search_results: list[Any]
    ) -> dict[str, Any]:
        """Analyze characteristics of available data."""
        if not search_results:
            return {}

        # Analyze units
        units = [r.units for r in search_results if r.units]
        unit_counts = {}
        for unit in units:
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

        # Analyze data types
        data_types = [r.data_type for r in search_results if r.data_type]
        type_counts = {}
        for dtype in data_types:
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        return {
            "total_paths": len(search_results),
            "common_units": sorted(
                unit_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "data_type_distribution": type_counts,
            "documentation_quality": self._assess_documentation_quality(search_results),
            "complexity_indicators": self._identify_complexity_indicators(
                search_results
            ),
        }

    def _assess_documentation_quality(self, search_results: list[Any]) -> str:
        """Assess quality of documentation in search results."""
        if not search_results:
            return "unknown"

        documented_count = sum(
            1 for r in search_results if r.documentation and len(r.documentation) > 50
        )
        documentation_ratio = documented_count / len(search_results)

        if documentation_ratio > 0.8:
            return "high"
        elif documentation_ratio > 0.5:
            return "moderate"
        else:
            return "limited"

    def _identify_complexity_indicators(self, search_results: list[Any]) -> list[str]:
        """Identify complexity indicators in the data."""
        indicators = []

        # Check for identifier schemas (arrays/enumerations)
        has_identifiers = any("identifier" in r.path.lower() for r in search_results)
        if has_identifiers:
            indicators.append("identifier_schemas")

        # Check for nested structures
        max_depth = (
            max(len(r.path.split("/")) for r in search_results) if search_results else 0
        )
        if max_depth > 4:
            indicators.append("deep_hierarchy")

        # Check for temporal data
        has_temporal = any("time" in r.path.lower() for r in search_results)
        if has_temporal:
            indicators.append("time_dependent")

        return indicators

    def _assess_complexity(
        self, domain: str, search_results: list[Any]
    ) -> dict[str, Any]:
        """Assess overall complexity of domain data."""
        domain_info = self._domain_characteristics.get(domain, {})
        theoretical_complexity = domain_info.get("complexity_level", "intermediate")

        data_complexity = "basic"
        if search_results:
            complexity_indicators = self._identify_complexity_indicators(search_results)
            if len(complexity_indicators) > 2:
                data_complexity = "advanced"
            elif len(complexity_indicators) > 0:
                data_complexity = "intermediate"

        return {
            "theoretical_complexity": theoretical_complexity,
            "data_complexity": data_complexity,
            "complexity_factors": self._identify_complexity_indicators(search_results),
            "recommended_approach": self._recommend_analysis_approach(
                theoretical_complexity, data_complexity
            ),
        }

    def _recommend_analysis_approach(self, theoretical: str, data: str) -> str:
        """Recommend analysis approach based on complexity using YAML definitions."""
        analysis_approaches = self._research_workflows.get("analysis_approaches", {})
        complexity_matrix = analysis_approaches.get("complexity_matrix", {})

        # Build key for complexity combination
        key = f"{theoretical}_theoretical_{data}_data"

        if key in complexity_matrix:
            return complexity_matrix[key].get("approach", "guided_analysis_workflow")

        # Default approach
        default_approach = complexity_matrix.get("default", {})
        return default_approach.get("approach", "guided_analysis_workflow")

    def _build_detailed_physics_context(self, domain: str) -> dict[str, Any]:
        """Build detailed physics context for comprehensive analysis using YAML definitions."""
        contexts = self._physics_contexts.get("theoretical_contexts", {})

        if domain in contexts:
            context_info = contexts[domain]
            return {
                "fundamental_equations": context_info.get("fundamental_equations", []),
                "key_physics_scales": context_info.get("physics_scales", {}),
                "governing_parameters": context_info.get("governing_parameters", []),
                "typical_regimes": context_info.get("typical_regimes", []),
            }

        # Use default context
        default_context = self._physics_contexts.get("default_context", {})
        return {
            "fundamental_equations": default_context.get(
                "fundamental_equations", ["domain-specific equations"]
            ),
            "key_physics_scales": default_context.get(
                "physics_scales",
                {"spatial": "characteristic length", "temporal": "characteristic time"},
            ),
            "governing_parameters": default_context.get(
                "governing_parameters", ["dimensionless parameters"]
            ),
            "typical_regimes": default_context.get(
                "typical_regimes", ["standard operation"]
            ),
        }

    def _get_fundamental_equations(self, domain: str) -> list[str]:
        """Get fundamental equations governing domain using YAML definitions."""
        contexts = self._physics_contexts.get("theoretical_contexts", {})

        if domain in contexts:
            return contexts[domain].get(
                "fundamental_equations", ["domain-specific equations"]
            )

        return ["domain-specific equations"]

    def _get_physics_scales(self, domain: str) -> dict[str, str]:
        """Get characteristic physics scales for domain using YAML definitions."""
        contexts = self._physics_contexts.get("theoretical_contexts", {})

        if domain in contexts:
            return contexts[domain].get(
                "physics_scales",
                {"spatial": "characteristic length", "temporal": "characteristic time"},
            )

        return {"spatial": "characteristic length", "temporal": "characteristic time"}

    def _get_governing_parameters(self, domain: str) -> list[str]:
        """Get key governing parameters for domain using YAML definitions."""
        contexts = self._physics_contexts.get("theoretical_contexts", {})

        if domain in contexts:
            return contexts[domain].get(
                "governing_parameters", ["dimensionless parameters"]
            )

        return ["dimensionless parameters"]

    def _get_typical_regimes(self, domain: str) -> list[str]:
        """Get typical operating regimes for domain using YAML definitions."""
        contexts = self._physics_contexts.get("theoretical_contexts", {})

        if domain in contexts:
            return contexts[domain].get("typical_regimes", ["standard operation"])

        return ["standard operation"]

    def _filter_for_overview(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Filter analysis for overview depth."""
        # Keep only essential information for overview
        filtered = {
            "key_measurements": analysis["key_measurements"][:3],
            "theoretical_foundations": {
                "description": analysis["theoretical_foundations"]["description"],
                "complexity_level": analysis["theoretical_foundations"][
                    "complexity_level"
                ],
            },
            "experimental_methods": analysis["experimental_methods"][:2],
            "cross_domain_links": analysis["cross_domain_links"][:2],
        }
        return filtered

    def _expand_for_comprehensive(
        self, analysis: dict[str, Any], domain: str, search_results: list[Any]
    ) -> dict[str, Any]:
        """Expand analysis for comprehensive depth."""
        # Add detailed sections for comprehensive analysis
        analysis["detailed_physics_context"] = self._build_detailed_physics_context(
            domain
        )
        analysis["measurement_integration"] = self._analyze_measurement_integration(
            search_results
        )
        analysis["research_applications"] = self._identify_research_applications(domain)
        analysis["data_quality_assessment"] = self._assess_data_quality(search_results)

        return analysis

    def _analyze_measurement_integration(
        self, search_results: list[Any]
    ) -> dict[str, Any]:
        """Analyze how measurements integrate for comprehensive analysis."""
        # Group measurements by physics quantity
        quantity_groups = {}
        for result in search_results:
            quantity = self._extract_physics_quantity(result)
            if quantity not in quantity_groups:
                quantity_groups[quantity] = []
            quantity_groups[quantity].append(result)

        integration_analysis = {
            "measurement_groups": quantity_groups,
            "integration_possibilities": self._identify_integration_opportunities(
                quantity_groups
            ),
            "validation_strategies": self._suggest_validation_strategies(
                quantity_groups
            ),
        }

        return integration_analysis

    def _extract_physics_quantity(self, result: Any) -> str:
        """Extract physics quantity from search result."""
        path = result.path.lower()

        if "density" in path:
            return "density"
        elif "temperature" in path:
            return "temperature"
        elif "pressure" in path:
            return "pressure"
        elif "current" in path:
            return "current"
        elif "field" in path:
            return "magnetic_field"
        else:
            return "other"

    def _identify_integration_opportunities(
        self, quantity_groups: dict[str, list[Any]]
    ) -> list[str]:
        """Identify opportunities for measurement integration."""
        opportunities = []

        if "density" in quantity_groups and "temperature" in quantity_groups:
            opportunities.append("pressure_consistency_check")

        if "current" in quantity_groups and "magnetic_field" in quantity_groups:
            opportunities.append("equilibrium_consistency_validation")

        return opportunities

    def _suggest_validation_strategies(
        self, quantity_groups: dict[str, list[Any]]
    ) -> list[str]:
        """Suggest validation strategies for measurements."""
        strategies = []

        if len(quantity_groups) > 2:
            strategies.append("cross_measurement_validation")

        strategies.append("temporal_consistency_check")
        strategies.append("physics_based_validation")

        return strategies

    def _identify_research_applications(self, domain: str) -> list[dict[str, Any]]:
        """Identify research applications for domain using YAML definitions."""
        research_applications = self._research_workflows.get(
            "research_applications", {}
        )

        if domain in research_applications:
            return research_applications[domain]

        # Default application
        return [
            {
                "application": "physics_research",
                "description": f"Research applications in {domain}",
                "data_requirements": ["domain_specific_data"],
                "typical_methods": ["standard_analysis"],
            }
        ]

    def _assess_data_quality(self, search_results: list[Any]) -> dict[str, Any]:
        """Assess overall data quality for comprehensive analysis."""
        if not search_results:
            return {"quality": "unknown", "assessment": "No data available"}

        # Count results with good documentation
        well_documented = sum(
            1 for r in search_results if r.documentation and len(r.documentation) > 100
        )
        documentation_ratio = well_documented / len(search_results)

        # Count results with units
        with_units = sum(1 for r in search_results if r.units)
        units_ratio = with_units / len(search_results)

        # Overall quality assessment
        if documentation_ratio > 0.7 and units_ratio > 0.8:
            quality = "high"
        elif documentation_ratio > 0.4 and units_ratio > 0.5:
            quality = "moderate"
        else:
            quality = "limited"

        return {
            "quality": quality,
            "documentation_coverage": f"{documentation_ratio:.1%}",
            "units_coverage": f"{units_ratio:.1%}",
            "total_paths": len(search_results),
            "recommendations": self._generate_quality_recommendations(quality),
        }

    def _generate_quality_recommendations(self, quality: str) -> list[str]:
        """Generate recommendations based on data quality using YAML definitions."""
        quality_recommendations = self._research_workflows.get(
            "data_quality_recommendations", {}
        )

        if quality in quality_recommendations:
            return quality_recommendations[quality].get("recommendations", [])

        # Default recommendations
        return ["Review data quality", "Consider data improvement"]
