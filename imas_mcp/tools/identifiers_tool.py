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
    Identifier catalog-based tool for IMAS identifier discovery.

    Provides intelligent access to the identifier catalog (identifier_catalog.json),
    serving as the primary interface for users to discover identifier schemas
    and enumeration options.
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
        "Browse available identifier schemas and enumeration options in IMAS data"
    )
    async def explore_identifiers(
        self,
        query: str | None = None,
        scope: IdentifierScope = IdentifierScope.ALL,
        ctx: Context | None = None,
    ) -> IdentifierResult | ToolError:
        """
        Browse available identifier schemas and enumeration options in IMAS data.

        Discovery tool for finding valid identifier values, coordinate systems,
        and enumeration options that control data access. Essential for understanding
        how to properly specify array indices and measurement configurations.

        Args:
            query: Search for specific identifier schemas or enumeration types
            scope: Focus area - all, enums, identifiers, coordinates, or constants
            ctx: MCP context for potential AI enhancement

        Returns:
            IdentifierResult with schemas, enumeration options, and usage guidance
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
