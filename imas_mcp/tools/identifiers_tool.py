"""
Identifiers tool implementation with catalog-based architecture.

This module provides an intelligent interface to the identifier catalog,
serving as the primary entry point for users to discover and navigate
identifier schemas and enumeration options.
"""

import importlib.resources
import json
import logging

from fastmcp import Context

from imas_mcp.models.constants import IdentifierScope
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import IdentifiersInput
from imas_mcp.models.result_models import IdentifierResult
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class IdentifiersTool(BaseTool):
    """
    IMAS identifier discovery tool for finding valid enumeration values and schemas.

    This tool provides access to the identifier catalog containing all valid enumeration
    values, coordinate systems, and classification schemes used throughout IMAS data.
    Critical for understanding data access patterns and valid parameter values.

    Primary Use Cases:
    - Discover valid values for identifier fields (materials, coordinate systems, etc.)
    - Find enumeration options for array indexing and data selection
    - Understand measurement configuration parameters
    - Explore data classification schemes

    Key Features:
    - 58 identifier schemas covering all IMAS domains
    - 584 total enumeration options across all schemas
    - Scope-based filtering (all, enums, coordinates, etc.)
    - Query-based schema discovery
    - Usage path mapping to show where identifiers are used

    Best Practices for LLMs:
    - Use broad search terms rather than very specific phrases
    - Start with scope="all" for exploration, then narrow with specific scopes
    - Check analytics.enumeration_space to understand data complexity
    - Use returned identifier values in subsequent search_imas() calls
    """

    def __init__(self, *args, **kwargs):
        """Initialize with identifier catalog data loading."""
        super().__init__(*args, **kwargs)
        self._identifier_catalog = {}
        self._load_identifier_catalog()

    def _load_identifier_catalog(self):
        """Load the identifier catalog file specifically."""
        try:
            try:
                catalog_file = (
                    importlib.resources.files("imas_mcp.resources.schemas")
                    / "identifier_catalog.json"
                )
                with catalog_file.open("r", encoding="utf-8") as f:
                    self._identifier_catalog = json.load(f)
                    logger.info("Loaded identifier catalog for identifiers tool")
            except FileNotFoundError:
                logger.warning(
                    "Identifier catalog (identifier_catalog.json) not found in resources/schemas/"
                )

        except Exception as e:
            logger.error(f"Failed to load identifier catalog: {e}")
            self._identifier_catalog = {}

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "explore_identifiers"

    def _filter_schemas_by_query(self, query: str) -> list[str]:
        """Filter identifier schemas based on query terms."""
        if not self._identifier_catalog:
            return []

        query_lower = query.lower()
        relevant_schemas = []
        schemas = self._identifier_catalog.get("schemas", {})

        for schema_name, schema_info in schemas.items():
            # Check name match
            if query_lower in schema_name.lower():
                relevant_schemas.append(schema_name)
                continue

            # Check description match
            description = schema_info.get("description", "").lower()
            if query_lower in description:
                relevant_schemas.append(schema_name)
                continue

            # Check options match
            for option in schema_info.get("options", []):
                option_name = option.get("name", "").lower()
                option_desc = option.get("description", "").lower()
                if query_lower in option_name or query_lower in option_desc:
                    relevant_schemas.append(schema_name)
                    break

        return relevant_schemas

    def _get_scope_filtered_data(
        self, scope: IdentifierScope, query: str | None = None
    ) -> dict:
        """Get data filtered by scope and optional query."""
        if not self._identifier_catalog:
            return {}

        schemas = self._identifier_catalog.get("schemas", {})

        # Filter by query if provided
        if query:
            relevant_schema_names = self._filter_schemas_by_query(query)
            filtered_schemas = {
                name: schemas[name] for name in relevant_schema_names if name in schemas
            }
        else:
            filtered_schemas = schemas

        # Apply scope filtering
        if scope == IdentifierScope.ENUMS:
            # Only schemas with enumeration options
            filtered_schemas = {
                name: schema
                for name, schema in filtered_schemas.items()
                if schema.get("total_options", 0) > 0
            }
        elif scope == IdentifierScope.IDENTIFIERS:
            # Focus on identifier-specific schemas (exclude coordinate systems, etc.)
            filtered_schemas = {
                name: schema
                for name, schema in filtered_schemas.items()
                if "identifier" in name.lower() or "type" in name.lower()
            }
        elif scope == IdentifierScope.COORDINATES:
            # Focus on coordinate system schemas
            filtered_schemas = {
                name: schema
                for name, schema in filtered_schemas.items()
                if any(term in name.lower() for term in ["coordinate", "plane", "grid"])
            }
        elif scope == IdentifierScope.CONSTANTS:
            # Focus on constant/parameter schemas
            filtered_schemas = {
                name: schema
                for name, schema in filtered_schemas.items()
                if any(
                    term in name.lower() for term in ["constant", "parameter", "flag"]
                )
            }
        # IdentifierScope.ALL returns all filtered_schemas as-is

        return filtered_schemas

    def _generate_identifier_recommendations(
        self, query: str | None, schemas: dict
    ) -> list[str]:
        """Generate usage recommendations based on identifier context."""
        recommendations = []

        if query:
            recommendations.append(
                f"🔍 Use search_imas('{query}') to find paths using these identifiers"
            )

            # Schema-specific recommendations
            if any("coordinate" in name.lower() for name in schemas.keys()):
                recommendations.append(
                    "📐 Use analyze_ids_structure() to see how coordinate identifiers affect data structure"
                )

            if any("type" in name.lower() for name in schemas.keys()):
                recommendations.append(
                    "🔧 Use explore_relationships() to find data paths using these type identifiers"
                )

        if schemas:
            schema_names = list(schemas.keys())
            if len(schema_names) > 0:
                first_schema = schema_names[0]
                recommendations.append(
                    f"📋 Use search_imas() with specific values from '{first_schema}' schema"
                )

            if len(schema_names) > 3:
                recommendations.append(
                    "Use explore_relationships() to see how these identifiers connect different IDS"
                )

        # Always include general recommendations
        recommendations.extend(
            [
                "💡 Use get_overview() to understand overall IMAS structure",
                "🌐 Use explore_relationships() to find data connections",
                "📈 Use export_ids() for data extraction with identifier filtering",
                "🔍 Use analyze_ids_structure() to see identifier usage in specific IDS",
            ]
        )

        return recommendations[:6]  # Limit to 6 recommendations

    @cache_results(ttl=3600, key_strategy="semantic")
    @validate_input(schema=IdentifiersInput)
    @handle_errors(fallback="identifiers_suggestions")
    @mcp_tool(
        "Browse IMAS identifier schemas and enumeration options - discover valid values for array indices, coordinate systems, and measurement configurations"
    )
    async def explore_identifiers(
        self,
        query: str | None = None,
        scope: IdentifierScope = IdentifierScope.ALL,
        ctx: Context | None = None,
    ) -> IdentifierResult | ToolError:
        """
        Browse available identifier schemas and enumeration options in IMAS data.

        This tool discovers valid identifier values, coordinate systems, and enumeration
        options that control IMAS data access. Essential for understanding how to properly
        specify array indices, measurement configurations, and data selection criteria.

        Identifier schemas define the valid values for fields that act as discriminators
        or selectors in IMAS data structures. These are critical for accessing the correct
        data arrays and understanding measurement contexts.

        Args:
            query: Search term to filter schemas by schema names/paths (NOT content/descriptions).
                  Filters identifier schemas by matching against schema file names and paths.
                  Use single broad keywords rather than multi-word phrases for best results.
                  Examples: "coordinate", "material", "transport" (not "grid coordinate systems")
                  Leave empty to see all schemas in the specified scope.
            scope: Focus the search on specific identifier types:
                  - "all": All identifier schemas (default, recommended for exploration)
                  - "enums": Only schemas with enumeration options (for discrete choices)
                  - "identifiers": Type and category identifiers (for classification)
                  - "coordinates": Coordinate system and grid identifiers
                  - "constants": Constant and parameter identifiers
            ctx: MCP context for potential AI enhancement

        Returns:
            IdentifierResult containing:
            - schemas: List of identifier schemas with enumeration options
            - paths: IMAS data paths that use these identifiers
            - analytics: Statistics about enumeration spaces and usage

        ## Parameter Interaction Examples

        ### ❌ Common LLM Mistakes:
            # Too specific - will return empty results
            explore_identifiers(query="grid coordinate systems", scope="coordinates")

            # Multi-word technical phrases - may not match schema names
            explore_identifiers(query="plasma equilibrium types")

            # Expecting query to search descriptions - it only filters schema names
            explore_identifiers(query="magnetic flux coordinates")

        ### ✅ Correct Usage Patterns:
            # Broad keyword matching schema names
            explore_identifiers(query="coordinate", scope="coordinates")

            # No query - see all schemas in scope first
            explore_identifiers(scope="coordinates")

            # Single keyword that appears in schema file names
            explore_identifiers(query="material")

        ## Query Parameter Behavior

        The `query` parameter filters identifier schemas by:
        - Schema file names (e.g., "coordinate" matches "coordinate_identifier.xml")
        - Schema paths (e.g., "poloidal" matches "poloidal_plane_coordinates_identifier.xml")

        **NOT by schema content or descriptions**

        ### Best Practices:
        - Use single keywords rather than phrases
        - Use broader terms rather than specific technical phrases
        - Omit query parameter to see all schemas in a scope first
        - Use schema names from results for subsequent filtering

        ## Recommended Usage Pattern for LLMs

        1. **Start broad**: Use scope without query to see available schemas
        2. **Filter by category**: Use scope with broad keywords
        3. **Examine specific schemas**: Use results to understand available options

        ### Example Workflow:
            # Step 1: See all coordinate-related schemas
            explore_identifiers(scope="coordinates")
            → Returns: coordinate_identifier.xml, poloidal_plane_coordinates_identifier.xml

            # Step 2: Filter to specific coordinate types
            explore_identifiers(query="poloidal", scope="coordinates")
            → Returns: poloidal_plane_coordinates_identifier.xml

            # Step 3: Use schema information to configure data paths
            # Now you know valid coordinate options for data access

        ## Scope-Specific Examples

        ### "coordinates" scope:
        - Returns: coordinate_identifier.xml, poloidal_plane_coordinates_identifier.xml
        - Use cases: Finding valid coordinate system options for 2D grids
        - Common options: rectangular, inverse_psi_polar, flux surface types

        ### "enums" scope:
        - Returns: All enumeration schemas (same as "all" - shows enumeration counts)
        - Use cases: Browsing all available identifier options
        - Shows total enumeration space across all schemas

        ### "all" scope:
        - Returns: All 57+ identifier schemas
        - Use cases: General exploration, finding schema categories
        - Shows complete identifier landscape

        Examples:
            # Discover all available identifier schemas
            explore_identifiers()
            → Returns 58 schemas, 146 paths, enumeration space of 584

            # Find material-related identifiers
            explore_identifiers(query="material", scope="enums")
            → Returns materials schema with 31 enumeration options (C, W, SS, etc.)

            # Find coordinate system identifiers
            explore_identifiers(scope="coordinates")
            → Returns 2 coordinate schemas with 68 enumeration options

            # Find plasma-related identifiers
            explore_identifiers(query="plasma")
            → Returns schemas containing "plasma" in their names/paths

            # Find all enumeration schemas
            explore_identifiers(scope="enums")
            → Returns all schemas showing their enumeration option counts

        ## Common LLM Usage Errors to Avoid

        ❌ **Don't**: Use multi-word technical phrases in query
        ❌ **Don't**: Expect query to search schema descriptions
        ❌ **Don't**: Use very specific terminology as query
        ❌ **Don't**: Assume query works like semantic search

        ✅ **Do**: Use single broad keywords
        ✅ **Do**: Start with scope-only calls for exploration
        ✅ **Do**: Use schema names from results for filtering
        ✅ **Do**: Check analytics.enumeration_space to understand complexity

        Usage Tips for LLMs:
            - Use broad search terms: "material", "plasma", "transport", "equilibrium"
            - Avoid overly specific queries like "plasma equilibrium state"
            - Start with scope="all" or scope-only calls to explore available options
            - Use scope="enums" to find discrete choice options
            - Check the analytics.enumeration_space to understand data complexity
            - Use returned paths to understand where identifiers are used in IMAS

        Common Query Patterns:
            - Physics domains: "plasma", "transport", "equilibrium", "sources"
            - Materials: "material", "wall", "divertor"
            - Coordinates: "coordinate", "grid", "geometry"
            - Measurements: "diagnostic", "detector", "sensor"
            - Configuration: "type", "mode", "status"

        Follow-up Actions:
            - Use search_imas() with specific identifier values found here
            - Use analyze_ids_structure() to see identifier usage in specific IDS
            - Use explore_relationships() to find connections between identifiers
        """
        try:
            # Check if identifier catalog is loaded
            if not self._identifier_catalog:
                return ToolError(
                    error="Identifier catalog data not available",
                    suggestions=[
                        "Check if identifier_catalog.json exists in resources/schemas/",
                        "Try restarting the MCP server",
                        "Use search_imas() for direct data access",
                    ],
                    context={
                        "tool": "explore_identifiers",
                        "operation": "catalog_access",
                    },
                )

            # Get filtered schemas based on scope and query
            filtered_schemas = self._get_scope_filtered_data(scope, query)

            # Build schemas list for response
            schemas = []
            total_usage_paths = 0

            for schema_name, schema_info in list(filtered_schemas.items())[
                :20
            ]:  # Limit for performance
                schema_item = {
                    "path": schema_info.get("schema_path", schema_name),
                    "schema_path": schema_info.get("schema_path", ""),
                    "option_count": schema_info.get("total_options", 0),
                    "branching_significance": (
                        "CRITICAL"
                        if schema_info.get("total_options", 0) > 10
                        else "HIGH"
                        if schema_info.get("total_options", 0) > 5
                        else "MODERATE"
                        if schema_info.get("total_options", 0) > 1
                        else "MINIMAL"
                    ),
                    "sample_options": [
                        {
                            "name": opt.get("name", ""),
                            "index": opt.get("index", 0),
                            "description": opt.get("description", ""),
                        }
                        for opt in schema_info.get("options", [])[
                            :5
                        ]  # Limit to 5 sample options
                    ],
                }
                schemas.append(schema_item)
                total_usage_paths += len(schema_info.get("usage_paths", []))

            # Build identifier paths from usage information
            identifier_paths = []
            for schema_name, schema_info in list(filtered_schemas.items())[
                :10
            ]:  # Limit for performance
                for usage_path in schema_info.get("usage_paths", []):
                    identifier_paths.append(
                        {
                            "path": usage_path,
                            "ids_name": usage_path.split("/")[0]
                            if "/" in usage_path
                            else "unknown",
                            "has_identifier": True,
                            "documentation": f"Uses {schema_name} identifier schema",
                        }
                    )

            # Build branching analytics
            branching_analytics = {
                "total_schemas": len(filtered_schemas),
                "total_paths": total_usage_paths,
                "enumeration_space": sum(
                    schema.get("total_options", 0)
                    for schema in filtered_schemas.values()
                ),
                "significance": f"Identifier schemas define {len(filtered_schemas)} critical branching points in IMAS data structures",
                "query_context": query,
                "scope_applied": scope.value,
            }

            # Generate recommendations - used in future enhancements
            # recommendations = self._generate_identifier_recommendations(query, filtered_schemas)

            # Build response using Pydantic
            response = IdentifierResult(
                scope=scope,
                schemas=schemas,
                paths=identifier_paths,
                analytics=branching_analytics,
                ai_response={},  # No AI processing needed for catalog data
            )

            logger.info(f"Identifier exploration completed with scope: {scope.value}")
            return response

        except Exception as e:
            logger.error(f"Catalog-based identifier exploration failed: {e}")
            return ToolError(
                error=str(e),
                suggestions=[
                    "Try a simpler query or different scope",
                    "Use get_overview() for general IMAS exploration",
                    "Check identifier catalog file availability",
                ],
                context={
                    "query": query,
                    "scope": scope.value,
                    "tool": "explore_identifiers",
                    "operation": "catalog_identifiers",
                    "identifier_catalog_loaded": bool(self._identifier_catalog),
                },
            )
