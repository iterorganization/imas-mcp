"""
Identifiers tool implementation.

This module contains the explore_identifiers tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Dict, Any, Optional
from fastmcp import Context

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.models.request_models import IdentifiersInputSchema
from imas_mcp.models.constants import IdentifierScope
from imas_mcp.models.response_models import IdentifierResult

# Import all decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    sample,
    recommend_tools,
    measure_performance,
    handle_errors,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""

    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator


class IdentifiersTool(BaseTool):
    """Tool for exploring identifiers."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        """Initialize the identifiers tool."""
        super().__init__()
        self.document_store = document_store or DocumentStore()

    def get_tool_name(self) -> str:
        return "explore_identifiers"

    def _build_identifiers_sample_prompt(
        self, query: Optional[str] = None, scope: str = "all"
    ) -> str:
        """Build sampling prompt for identifier exploration."""
        base_prompt = f"""IMAS Identifier Schema Exploration Request:
Query: {query or "General exploration"}
Scope: {scope}

Please provide comprehensive identifier analysis that includes:

1. **Significance**: Importance of identifier branching logic in IMAS
2. **Key Schemas**: Major identifier schemas and their enumeration options
3. **Physics Implications**: How identifier choices affect physics domain access
4. **Usage Guidance**: Practical guidance for using identifier information
5. **Critical Decisions**: Key branching points for data structure navigation
6. **Enumeration Options**: Available choices at each branching point
7. **Recommendations**: Best practices for identifier-based data access

Focus on providing actionable insights for researchers working with IMAS identifiers.
"""
        return base_prompt

    @cache_results(
        ttl=1200, key_strategy="content_based"
    )  # Longer cache for identifiers
    @validate_input(schema=IdentifiersInputSchema)
    @sample(temperature=0.3, max_tokens=800)  # Balanced creativity for analysis
    @recommend_tools(strategy="identifiers_based", max_tools=3)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="identifiers_suggestions")
    @mcp_tool("Explore IMAS identifier schemas and branching logic")
    async def explore_identifiers(
        self,
        query: Optional[str] = None,
        scope: str = "all",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Explore IMAS identifier schemas and branching logic.

        Provides access to enumerated options and branching logic that define
        critical decision points in the IMAS data structure.

        Args:
            query: Optional search query for specific identifier schemas or options
            scope: Scope of exploration ("all", "schemas", "paths", "summary")
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with identifier schemas, paths, and branching analytics
        """
        try:
            # Get overall identifier branching summary
            summary = self.document_store.get_identifier_branching_summary()

            schemas = []
            identifier_paths = []

            if scope in ["all", "schemas"]:
                # Get identifier schemas
                if query:
                    schema_docs = self.document_store.search_identifier_schemas(query)
                else:
                    schema_docs = self.document_store.get_identifier_schemas()

                for schema_doc in schema_docs[:10]:  # Limit to 10 schemas
                    schema_data = schema_doc.raw_data
                    # Build schema item using dict structure
                    schema_item = {
                        "path": schema_doc.metadata.path_name,
                        "schema_path": schema_data.get("schema_path", "unknown"),
                        "option_count": schema_data.get("total_options", 0),
                        "branching_significance": (
                            "CRITICAL"
                            if schema_data.get("total_options", 0) > 5
                            else "MODERATE"
                            if schema_data.get("total_options", 0) > 1
                            else "MINIMAL"
                        ),
                        "sample_options": [
                            {
                                "name": opt.get("name", ""),
                                "index": opt.get("index", 0),
                                "description": opt.get("description", ""),
                            }
                            for opt in schema_data.get("options", [])[:5]
                        ],
                    }
                    schemas.append(schema_item)

            if scope in ["all", "paths"]:
                # Get identifier paths
                try:
                    all_docs = []
                    for ids_name in self.document_store.get_available_ids()[
                        :5
                    ]:  # Sample 5 IDS
                        docs = self.document_store.get_documents_by_ids(ids_name)
                        all_docs.extend(
                            [d for d in docs if d.raw_data.get("identifier_schema")]
                        )

                    for doc in all_docs[:20]:  # Limit to 20 identifier paths
                        identifier_paths.append(
                            {
                                "path": doc.metadata.path_name,
                                "ids_name": doc.metadata.path_name.split("/")[0]
                                if "/" in doc.metadata.path_name
                                else "unknown",
                                "has_identifier": True,
                                "documentation": doc.metadata.documentation[:100]
                                if doc.metadata.documentation
                                else "",
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get identifier paths: {e}")

            # Prepare branching analytics
            branching_analytics = {
                "total_schemas": summary["total_schemas"],
                "total_paths": summary["total_identifier_paths"],
                "enumeration_space": summary["total_enumeration_options"],
                "significance": "Identifier schemas define critical branching logic and enumeration options in IMAS data structures",
            }

            # Convert scope string to enum
            scope_enum = (
                IdentifierScope.ALL
                if scope == "all"
                else IdentifierScope.ENUMS
                if scope == "enums"
                else IdentifierScope.IDENTIFIERS
                if scope == "identifiers"
                else IdentifierScope.COORDINATES
                if scope == "coordinates"
                else IdentifierScope.CONSTANTS
                if scope == "constants"
                else IdentifierScope.ALL
            )

            # Build final response using Pydantic
            response = IdentifierResult(
                scope=scope_enum,
                schemas=schemas,  # schemas is already dict list
                paths=identifier_paths,
                analytics=branching_analytics,
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"Identifier exploration failed: {e}")
            return {
                "query": query,
                "scope": scope,
                "error": str(e),
                "explanation": "Failed to explore identifiers",
                "suggestions": [
                    "Try scope='summary' for overview",
                    "Use scope='schemas' for schema details",
                    "Use scope='paths' for identifier paths",
                    "Add query to filter results",
                ],
            }
