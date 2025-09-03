# File: imas_mcp/resources.py
"""
IMAS MCP Resources Implementation.

This module contains all the MCP resources for IMAS data dictionary schema access.
Resources provide static JSON schema files and reference data.
"""

import json
import logging
from importlib import resources
from pathlib import Path

from fastmcp import FastMCP

from imas_mcp.core.relationships import Relationships
from imas_mcp.providers import MCPProvider
from imas_mcp.structure.mermaid_generator import MermaidGraphGenerator

logger = logging.getLogger(__name__)


def mcp_resource(description: str, uri: str):
    """Decorator to mark methods as MCP resources with description and URI."""

    def decorator(func):
        func._mcp_resource = True
        func._mcp_resource_uri = uri
        func._mcp_resource_description = description
        return func

    return decorator


class Resources(MCPProvider):
    """MCP resources serving existing JSON schema files with LLM-friendly descriptions."""

    def __init__(self, ids_set: set[str] | None = None):
        self.ids_set = ids_set
        schema_resource = resources.files("imas_mcp") / "resources" / "schemas"
        self.schema_dir = Path(str(schema_resource))
        resources_dir = resources.files("imas_mcp") / "resources"
        self.mermaid_generator = MermaidGraphGenerator(Path(str(resources_dir)))

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return "resources"

    def register(self, mcp: FastMCP):
        """Register all IMAS resources with the MCP server."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_mcp_resource") and attr._mcp_resource:
                # Resources need URI and description
                uri = attr._mcp_resource_uri
                description = attr._mcp_resource_description
                mcp.resource(uri=uri, description=description)(attr)

    @mcp_resource(
        "IMAS IDS Catalog - Complete overview of all Interface Data Structures.",
        "ids://catalog",
    )
    async def get_ids_catalog(self) -> str:
        """IMAS IDS Catalog - Complete overview of all Interface Data Structures.

        Use this resource to:
        - Get a quick overview of all available IDS
        - Check document counts and physics domains for each IDS
        - Understand the scope before using search_imas tool
        - Find which IDS contain the most data

        Contains: IDS names, descriptions, path counts, physics domains, metadata.
        Perfect for: Initial orientation, domain mapping, scope assessment.
        """
        try:
            return (self.schema_dir / "ids_catalog.json").read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fall back to latin-1 if utf-8 fails
            return (self.schema_dir / "ids_catalog.json").read_text(encoding="latin-1")

    @mcp_resource(
        "Detailed IDS Structure - Complete schema for a specific IDS.",
        "ids://structure/{ids_name}",
    )
    async def get_ids_structure(self, ids_name: str) -> str:
        """Detailed IDS Structure - Complete schema for a specific IDS.

        Use this resource to:
        - Get the complete structure of a specific IDS
        - Understand data organization before detailed analysis
        - Check available paths and their relationships
        - Identify key physics quantities

        Contains: Full path hierarchy, data types, units, documentation.
        Perfect for: Structure understanding, path exploration, schema validation.
        """
        detailed_file = self.schema_dir / "detailed" / f"{ids_name}.json"
        if detailed_file.exists():
            try:
                return detailed_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Fall back to latin-1 if utf-8 fails
                return detailed_file.read_text(encoding="latin-1")
        return json.dumps(
            {
                "error": f"IDS '{ids_name}' not found",
                "available_ids": "Use ids://catalog resource to see all available IDS",
            }
        )

    @mcp_resource(
        "Identifier Schemas - Enumerated options and branching logic.",
        "ids://identifiers",
    )
    async def get_identifier_catalog(self) -> str:
        """Identifier Schemas - Enumerated options and branching logic.

        Use this resource to:
        - Understand enumerated options for identifiers
        - Check branching complexity and decision points
        - Find most commonly used identifier schemas
        - Analyze data structure complexity

        Contains: Identifier schemas, usage statistics, branching analytics.
        Perfect for: Data validation, option exploration, complexity analysis.
        """
        try:
            return (self.schema_dir / "identifier_catalog.json").read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            # Fall back to latin-1 if utf-8 fails
            return (self.schema_dir / "identifier_catalog.json").read_text(
                encoding="latin-1"
            )

    @mcp_resource(
        "Physics Relationships - Cross-references and measurement dependencies.",
        "ids://relationships",
    )
    async def get_relationships(self) -> str:
        """Physics Relationships - Cross-references and measurement dependencies.

        Use this resource to:
        - Find relationships between different IDS
        - Understand physics dependencies and connections
        - Identify measurement correlations
        - Map physics domain interactions

        Contains: Cross-IDS relationships, physics connections, dependency graphs.
        Perfect for: Multi-IDS analysis, physics correlation, dependency mapping.
        """
        try:
            # Use the relationships manager for better cache management
            relationships = Relationships(ids_set=self.ids_set)

            # Check if rebuild is needed and add warning to output
            if relationships.needs_rebuild():
                logger.warning(
                    "Relationships data may be outdated - consider rebuilding"
                )

            relationships_data = relationships.get_data()
            return json.dumps(relationships_data, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to load relationships data: {e}")
            # Fallback to direct file access
            try:
                return (self.schema_dir / "relationships.json").read_text(
                    encoding="utf-8"
                )
            except UnicodeDecodeError:
                # Fall back to latin-1 if utf-8 fails
                return (self.schema_dir / "relationships.json").read_text(
                    encoding="latin-1"
                )

    @mcp_resource(
        "Mermaid Hierarchy Graph - Visual representation of IDS hierarchical structure.",
        "mermaid://hierarchy/{ids_name}",
    )
    async def get_mermaid_hierarchy(self, ids_name: str) -> str:
        """Mermaid Hierarchy Graph - Visual representation of IDS hierarchical structure.

        Use this resource to:
        - Visualize the complete hierarchical structure of an IDS
        - Understand data organization and nesting patterns
        - Navigate complex IDS structures visually
        - Identify key structural nodes and branching points

        Contains: Complete Mermaid flowchart showing all nodes, relationships, and hierarchy levels.
        Perfect for: Understanding IDS structure, visual navigation, structural analysis.
        """
        graph_content = self.mermaid_generator.load_mermaid_graph(ids_name, "hierarchy")
        if graph_content is None:
            return f"# Hierarchy Graph Not Available\n\nNo hierarchy graph found for '{ids_name}' IDS. The graph may not have been generated yet or the IDS name may be invalid.\n\nAvailable graph types: {', '.join(self.mermaid_generator.get_available_graphs(ids_name))}"
        return graph_content

    @mcp_resource(
        "Mermaid Physics Domains Graph - Visual organization of IDS by physics domains.",
        "mermaid://physics/{ids_name}",
    )
    async def get_mermaid_physics(self, ids_name: str) -> str:
        """Mermaid Physics Domains Graph - Visual organization of IDS by physics domains.

        Use this resource to:
        - Understand physics domain organization within an IDS
        - See how measurement types are grouped by physics context
        - Identify domain-specific data paths and relationships
        - Navigate IDS content by physics domain

        Contains: Mermaid diagram showing physics domain groupings and representative paths.
        Perfect for: Physics-based navigation, domain analysis, measurement categorization.
        """
        graph_content = self.mermaid_generator.load_mermaid_graph(
            ids_name, "physics_domains"
        )
        if graph_content is None:
            return f"# Physics Domains Graph Not Available\n\nNo physics domains graph found for '{ids_name}' IDS. This graph is only generated for IDS with multiple physics domains.\n\nAvailable graph types: {', '.join(self.mermaid_generator.get_available_graphs(ids_name))}"
        return graph_content

    @mcp_resource(
        "Mermaid Complexity Graph - Visual complexity analysis of IDS structure.",
        "mermaid://complexity/{ids_name}",
    )
    async def get_mermaid_complexity(self, ids_name: str) -> str:
        """Mermaid Complexity Graph - Visual complexity analysis of IDS structure.

        Use this resource to:
        - Visualize complexity distribution across IDS structure
        - Identify high-complexity nodes and simple leaf structures
        - Understand complexity patterns and organizational principles
        - Plan navigation strategies based on complexity levels

        Contains: Mermaid mindmap showing complexity indicators and organizational patterns.
        Perfect for: Complexity analysis, navigation planning, structural understanding.
        """
        graph_content = self.mermaid_generator.load_mermaid_graph(
            ids_name, "complexity"
        )
        if graph_content is None:
            return f"# Complexity Graph Not Available\n\nNo complexity graph found for '{ids_name}' IDS. The graph may not have been generated yet or the IDS name may be invalid.\n\nAvailable graph types: {', '.join(self.mermaid_generator.get_available_graphs(ids_name))}"
        return graph_content

    @mcp_resource(
        "Mermaid IDS Overview - Complete overview of all IMAS IDS relationships.",
        "mermaid://overview",
    )
    async def get_mermaid_overview(self) -> str:
        """Mermaid IDS Overview - Complete overview of all IMAS IDS relationships.

        Use this resource to:
        - Get a high-level view of all available IDS
        - Understand IDS categorization by complexity
        - See relationships between different IDS types
        - Navigate the complete IMAS data dictionary structure

        Contains: Mermaid diagram showing all IDS organized by complexity and type.
        Perfect for: IMAS overview, IDS selection, high-level navigation.
        """
        graph_content = self.mermaid_generator.get_overview_graph()
        if graph_content is None:
            return "# IDS Overview Graph Not Available\n\nNo overview graph found. The graph may not have been generated yet. Try running the build-mermaid script to generate all graphs."
        return graph_content

    @mcp_resource(
        "Resource Usage Examples - How to effectively use IMAS MCP resources.",
        "examples://resource-usage",
    )
    async def get_resource_usage_examples(self) -> str:
        """Resource Usage Examples - How to effectively use IMAS MCP resources.

        Use this resource to:
        - Learn when to use resources vs tools
        - See example workflows combining resources and tools
        - Understand resource content and structure
        - Get guidance on efficient IMAS data exploration

        Contains: Usage patterns, workflow examples, best practices.
        Perfect for: Learning optimal usage patterns, workflow design.
        """
        examples = {
            "workflow_patterns": {
                "quick_orientation": {
                    "description": "Start with resources for overview, then use tools for detailed analysis",
                    "steps": [
                        "1. Check ids://catalog for IDS overview and domain mapping",
                        "2. Use ids://structure/{ids_name} for specific IDS structure",
                        "3. View mermaid://hierarchy/{ids_name} for visual structure",
                        "4. Use tools for detailed analysis and relationships",
                    ],
                },
                "visual_exploration": {
                    "description": "Use mermaid graphs for visual understanding of IDS structure",
                    "steps": [
                        "1. Start with mermaid://overview for complete IDS landscape",
                        "2. Use mermaid://hierarchy/{ids_name} for detailed structure",
                        "3. Check mermaid://physics/{ids_name} for domain organization",
                        "4. Use mermaid://complexity/{ids_name} for complexity analysis",
                    ],
                },
                "domain_exploration": {
                    "description": "Explore physics domains efficiently",
                    "steps": [
                        "1. Check ids://relationships for domain connections",
                        "2. Use ids://catalog to identify domain-specific IDS",
                        "3. Use export_physics_domain tool for comprehensive domain data",
                    ],
                },
            },
            "resource_vs_tools": {
                "use_resources_for": [
                    "Quick reference and overview",
                    "Static structure information",
                    "Schema validation and identifiers",
                    "Understanding relationships before analysis",
                ],
                "use_tools_for": [
                    "Dynamic search and filtering",
                    "AI-enhanced analysis and insights",
                    "Complex relationship exploration",
                    "Real-time data processing",
                ],
            },
            "example_resource_content": {
                "ids_catalog_sample": {
                    "core_profiles": {
                        "name": "core_profiles",
                        "description": "Core plasma profiles",
                        "path_count": 175,
                        "physics_domain": "transport",
                    }
                },
                "structure_sample": {
                    "path": "core_profiles/time_slice/profiles_1d/electrons/temperature",
                    "data_type": "FLT_1D",
                    "units": "eV",
                    "documentation": "Electron temperature profile",
                },
            },
        }
        return json.dumps(examples, indent=2)
