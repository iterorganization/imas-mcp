# File: imas_mcp/resources.py
"""
IMAS MCP Resources Implementation.

This module contains all the MCP resources for IMAS data dictionary schema access.
Resources provide static JSON schema files and reference data.
"""

import json
import logging
from pathlib import Path

from fastmcp import FastMCP

from imas_mcp.providers import MCPProvider

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

    def __init__(self):
        self.schema_dir = Path(__file__).parent / "resources" / "schemas"

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
                uri = getattr(attr, "_mcp_resource_uri")
                description = getattr(attr, "_mcp_resource_description")
                mcp.resource(uri=uri, description=description)(attr)

    @mcp_resource(
        "IMAS IDS Catalog - Complete overview of all Interface Data Structures.",
        "ids://catalog",
    )
    async def get_ids_catalog(self) -> str:
        """IMAS IDS Catalog - Complete overview of all Interface Data Structures.

        Use this resource to:
        - Get a quick overview of all available IDS (82 total)
        - Check document counts and physics domains for each IDS
        - Understand the scope before using search_imas tool
        - Find which IDS contain the most data

        Contains: IDS names, descriptions, path counts, physics domains, metadata.
        Perfect for: Initial orientation, domain mapping, scope assessment.
        """
        return (self.schema_dir / "ids_catalog.json").read_text()

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
            return detailed_file.read_text()
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
        return (self.schema_dir / "identifier_catalog.json").read_text()

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
        return (self.schema_dir / "relationships.json").read_text()

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
                        "3. Then use search_imas tool for dynamic queries",
                        "4. Follow with analyze_ids_structure tool for detailed analysis",
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
