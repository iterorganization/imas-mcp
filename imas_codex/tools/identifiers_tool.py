"""
Identifiers tool implementation with catalog-based architecture.

This module provides an intelligent interface to the identifier catalog,
serving as the primary entry point for users to discover and navigate
identifier schemas and enumeration options.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from imas_codex import dd_version
from imas_codex.models.error_models import ToolError
from imas_codex.models.request_models import IdentifiersInput
from imas_codex.models.result_models import GetIdentifiersResult
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    validate_input,
)

from .base import BaseTool

if TYPE_CHECKING:
    from fastmcp import Context

logger = logging.getLogger(__name__)


class IdentifiersTool(BaseTool):
    """IMAS identifier discovery tool for finding valid enumeration values and schemas."""

    def __init__(self, *args, **kwargs):
        """Initialize with identifier catalog data loading."""
        super().__init__(*args, **kwargs)
        self._identifier_catalog = {}
        self._load_identifier_catalog()

    def _load_identifier_catalog(self):
        """Load the identifier catalog file specifically."""
        try:
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            catalog_file = path_accessor.schemas_dir / "identifier_catalog.json"

            if catalog_file.exists():
                with catalog_file.open("r", encoding="utf-8") as f:
                    self._identifier_catalog = json.load(f)
                    logger.info("Loaded identifier catalog for identifiers tool")
            else:
                logger.warning(f"Identifier catalog not found at {catalog_file}")
        except Exception as e:
            logger.error(f"Failed to load identifier catalog: {e}")
            self._identifier_catalog = {}

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "get_imas_identifiers"

    def _filter_schemas_by_query(self, query: str) -> list[str]:
        """Filter identifier schemas based on query terms using OR logic for multiple keywords."""
        if not self._identifier_catalog:
            return []

        # Split query into keywords and clean them
        import re

        keywords = [
            keyword.strip().lower()
            for keyword in re.split(r"[,\s]+", query)
            if keyword.strip()
        ]

        if not keywords:
            return []

        relevant_schemas = []
        schemas = self._identifier_catalog.get("schemas", {})

        for schema_name, schema_info in schemas.items():
            score = 0
            matched_keywords = []

            # Check each keyword against schema name, description, and options
            for keyword in keywords:
                keyword_matched = False

                # Check name match
                if keyword in schema_name.lower():
                    score += 3  # Higher weight for name matches
                    matched_keywords.append(keyword)
                    keyword_matched = True

                # Check description match
                description = schema_info.get("description", "").lower()
                if keyword in description:
                    score += 2  # Medium weight for description matches
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
                    keyword_matched = True

                # Check options match
                if not keyword_matched:  # Only check options if not already matched
                    for option in schema_info.get("options", []):
                        option_name = option.get("name", "").lower()
                        option_desc = option.get("description", "").lower()
                        if keyword in option_name or keyword in option_desc:
                            score += 1  # Lower weight for option matches
                            if keyword not in matched_keywords:
                                matched_keywords.append(keyword)
                            break

            # Include schema if ANY keyword matched (OR logic)
            if matched_keywords:
                relevant_schemas.append((schema_name, score, matched_keywords))

        # Sort by score (descending) to put best matches first
        relevant_schemas.sort(key=lambda x: x[1], reverse=True)

        # Return just the schema names
        return [
            schema_name for schema_name, score, matched_keywords in relevant_schemas
        ]

    def _get_filtered_schemas(self, query: str | None = None) -> dict:
        """Get schemas filtered by optional query."""
        if not self._identifier_catalog:
            return {}

        schemas = self._identifier_catalog.get("schemas", {})

        # Filter by query if provided
        if query:
            relevant_schema_names = self._filter_schemas_by_query(query)
            return {
                name: schemas[name] for name in relevant_schema_names if name in schemas
            }

        return schemas

    def _generate_identifier_recommendations(
        self, query: str | None, schemas: dict
    ) -> list[str]:
        """Generate usage recommendations based on identifier context."""
        recommendations = []

        if query:
            recommendations.append(
                f"🔍 Use search_imas_paths('{query}') to find paths using these identifiers"
            )

            if any("coordinate" in name.lower() for name in schemas.keys()):
                recommendations.append(
                    "📐 Use list_imas_paths() to see how coordinate identifiers affect data structure"
                )

            if any("type" in name.lower() for name in schemas.keys()):
                recommendations.append(
                    "🔧 Use search_imas_clusters() to find related paths using these type identifiers"
                )

        if schemas:
            schema_names = list(schemas.keys())
            if len(schema_names) > 0:
                first_schema = schema_names[0]
                recommendations.append(
                    f"📋 Use search_imas_paths() with specific values from '{first_schema}' schema"
                )

            if len(schema_names) > 3:
                recommendations.append(
                    "Use search_imas_clusters() to see how these identifiers connect different IDS"
                )

        recommendations.extend(
            [
                "💡 Use get_imas_overview() to understand overall IMAS structure",
                "🌐 Use search_imas_clusters() to find related path clusters",
            ]
        )

        return recommendations[:6]

    @cache_results(ttl=3600, key_strategy="semantic")
    @validate_input(schema=IdentifiersInput)
    @handle_errors(fallback="identifiers_suggestions")
    @mcp_tool(
        "Browse IMAS identifier schemas and enumeration options. "
        "query: Optional keyword filter with OR-logic for multiple terms (e.g., 'coordinate material', 'transport source'). "
        "Returns identifier schemas with valid index values and descriptions. "
        "Key schemas: coordinate_identifier (35 options), core_source_identifier (53 options), ggd_subset_identifier (61 options). "
        "Use for: array indices, coordinate systems, source types, measurement configurations."
    )
    async def get_imas_identifiers(
        self,
        query: str | None = None,
        ctx: Context | None = None,
    ) -> GetIdentifiersResult | ToolError:
        """
        Browse available identifier schemas and enumeration options in IMAS data.

        This tool discovers valid identifier values, coordinate systems, and enumeration
        options that control IMAS data access. Essential for understanding how to properly
        specify array indices, measurement configurations, and data selection criteria.

        Args:
            query: Optional search terms to filter schemas using OR logic for multiple
                  keywords. Supports multiple keywords separated by spaces or commas.
                  Examples: "coordinate material", "transport,diffusion", "plasma"
            ctx: MCP context

        Returns:
            GetIdentifiersResult containing schemas, paths, and analytics

        Examples:
            get_imas_identifiers()  # All schemas
            get_imas_identifiers(query="material")  # Material-related schemas
            get_imas_identifiers(query="coordinate transport")  # OR search
        """
        try:
            if not self._identifier_catalog:
                return ToolError(
                    error="Identifier catalog data not available",
                    suggestions=[
                        "Check if identifier_catalog.json exists in resources/schemas/",
                        "Try restarting the MCP server",
                        "Use search_imas_paths() for direct data access",
                    ],
                    context={
                        "tool": "get_imas_identifiers",
                        "operation": "catalog_access",
                    },
                )

            filtered_schemas = self._get_filtered_schemas(query)

            schemas = []
            total_usage_paths = 0

            for schema_name, schema_info in filtered_schemas.items():
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
                    "options": [
                        {
                            "name": opt.get("name", ""),
                            "index": opt.get("index", 0),
                            "description": opt.get("description", ""),
                        }
                        for opt in schema_info.get("options", [])
                    ],
                }
                schemas.append(schema_item)
                total_usage_paths += len(schema_info.get("usage_paths", []))

            identifier_paths = []
            for schema_name, schema_info in filtered_schemas.items():
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

            branching_analytics = {
                "total_schemas": len(filtered_schemas),
                "total_paths": total_usage_paths,
                "enumeration_space": sum(
                    schema.get("total_options", 0)
                    for schema in filtered_schemas.values()
                ),
                "significance": (
                    f"Identifier schemas define {len(filtered_schemas)} "
                    "critical branching points in IMAS data structures"
                ),
                "query_context": query,
            }

            response = GetIdentifiersResult(
                schemas=schemas,
                paths=identifier_paths,
                analytics=branching_analytics,
            )

            logger.info(f"Identifier listing completed: {len(schemas)} schemas")
            return response

        except Exception as e:
            logger.error(f"Identifier listing failed: {e}")
            return ToolError(
                error=str(e),
                suggestions=[
                    "Try a simpler query",
                    "Use get_imas_overview() for general IMAS exploration",
                    "Check identifier catalog file availability",
                ],
                context={
                    "query": query,
                    "tool": "get_imas_identifiers",
                    "operation": "catalog_identifiers",
                    "identifier_catalog_loaded": bool(self._identifier_catalog),
                },
            )
