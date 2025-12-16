# File: imas_codex/resources.py
"""
IMAS Codex Resources Implementation.

This module contains all the MCP resources for IMAS data dictionary schema access.
Resources provide static JSON schema files and reference data.
"""

import json
import logging

from fastmcp import FastMCP

from imas_codex import dd_version
from imas_codex.core.clusters import Clusters
from imas_codex.providers import MCPProvider
from imas_codex.resource_path_accessor import ResourcePathAccessor

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
        path_accessor = ResourcePathAccessor(dd_version=dd_version)
        self.schema_dir = path_accessor.schemas_dir

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
        "Semantic Clusters - Cross-references and path groupings.",
        "ids://clusters",
    )
    async def get_clusters(self) -> str:
        """Semantic Clusters - Cross-references and path groupings.

        Use this resource to:
        - Find clusters of related paths across different IDS
        - Understand semantic connections between data paths
        - Identify measurement correlations
        - Map physics domain interactions

        Contains: Cross-IDS clusters, semantic groupings, dependency graphs.
        Perfect for: Multi-IDS analysis, physics correlation, dependency mapping.
        """
        try:
            # Use the clusters manager for better cache management
            from imas_codex.embeddings.config import EncoderConfig

            encoder_config = EncoderConfig(
                model_name=None,  # Use env var or fallback
                device=None,
                batch_size=250,
                normalize_embeddings=True,
                use_half_precision=False,
                enable_cache=True,
                cache_dir="embeddings",
                ids_set=self.ids_set,
                use_rich=False,
            )

            clusters = Clusters(encoder_config=encoder_config)

            # Check if rebuild is needed and add warning to output
            if clusters.needs_rebuild():
                logger.warning("Clusters data may be outdated - consider rebuilding")

            clusters_data = clusters.get_data()
            return json.dumps(clusters_data, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to load clusters data: {e}")
            # Fallback to direct file access
            try:
                return (self.schema_dir / "clusters.json").read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Fall back to latin-1 if utf-8 fails
                return (self.schema_dir / "clusters.json").read_text(encoding="latin-1")

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
                        "3. Use search_imas_paths tool to find specific data paths",
                        "4. Use list_imas_identifiers for enumeration options",
                    ],
                },
                "domain_exploration": {
                    "description": "Explore physics domains efficiently",
                    "steps": [
                        "1. Check ids://relationships for domain connections",
                        "2. Use ids://catalog to identify domain-specific IDS",
                        "3. Use search_imas_paths to find domain-specific data",
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
