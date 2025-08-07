"""
Identifiers tool implementation.

This module contains the explore_identifiers tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Optional, Union
from fastmcp import Context

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.models.request_models import IdentifiersInput
from imas_mcp.models.constants import IdentifierScope
from imas_mcp.models.result_models import IdentifierResult
from imas_mcp.models.error_models import ToolError

# Import identifier-appropriate decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
    sample,
    tool_hints,
    mcp_tool,
)

from imas_mcp.tools.base import BaseTool

logger = logging.getLogger(__name__)


class IdentifiersTool(BaseTool):
    """Tool for exploring identifiers using service composition."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        """Initialize the identifiers tool."""
        super().__init__(document_store)
        self.document_store = document_store or DocumentStore()

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
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

    @cache_results(ttl=1200, key_strategy="content_based")
    @validate_input(schema=IdentifiersInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="identifiers_suggestions")
    @tool_hints(max_hints=3)
    @sample(temperature=0.2, max_tokens=650)
    @mcp_tool("Explore IMAS identifier schemas and branching logic")
    async def explore_identifiers(
        self,
        query: Optional[str] = None,
        scope: IdentifierScope = IdentifierScope.ALL,
        ctx: Optional[Context] = None,
    ) -> Union[IdentifierResult, ToolError]:
        """
        Explore IMAS identifier schemas and branching logic using service composition.

        Uses service composition for business logic:
        - DocumentService: Validates queries and retrieves identifier documents
        - PhysicsService: Enhances analysis with physics context
        - ResponseService: Builds standardized Pydantic responses

        Provides access to enumerated options and branching logic that define
        critical decision points in the IMAS data structure.

        Args:
            query: Optional search query for specific identifier schemas or options
            scope: Scope of exploration ("all", "schemas", "paths", "summary")
            ctx: MCP context for AI enhancement

        Returns:
            IdentifierResult with identifier schemas, paths, and branching analytics
        """
        try:
            # Enhance query with physics context using service if provided
            physics_context = None
            if query:
                physics_context = await self.physics.enhance_query(query)

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

            # Prepare branching analytics with enhanced insights
            branching_analytics = {
                "total_schemas": summary["total_schemas"],
                "total_paths": summary["total_identifier_paths"],
                "enumeration_space": summary["total_enumeration_options"],
                "significance": "Identifier schemas define critical branching logic and enumeration options in IMAS data structures",
                "query_context": query,
                "physics_relevance": bool(physics_context),
            }

            # Build response using Pydantic
            response = IdentifierResult(
                scope=scope,
                schemas=schemas,
                paths=identifier_paths,
                analytics=branching_analytics,
                ai_response={},  # Reserved for LLM sampling only
            )

            logger.info(f"Identifier exploration completed with scope: {scope.value}")
            return response

        except Exception as e:
            logger.error(f"Identifier exploration failed: {e}")
            return self._create_error_response(
                f"Identifier exploration failed: {e}", query or "unknown"
            )

    def _build_identifier_guidance(
        self, query: Optional[str], scope: IdentifierScope, schemas: list, paths: list
    ) -> str:
        """Build comprehensive identifier guidance for AI enhancement."""
        return f"""IMAS Identifier Schema Exploration: {query or "General exploration"}

Scope: {scope.value} | Found {len(schemas)} schemas and {len(paths)} identifier paths

Key insights for researchers:
1. **Significance**: Importance of identifier branching logic in IMAS data navigation
2. **Key Schemas**: Major identifier schemas and their enumeration options
3. **Physics Implications**: How identifier choices affect physics domain access
4. **Usage Guidance**: Practical guidance for using identifier information
5. **Critical Decisions**: Key branching points for data structure navigation
6. **Enumeration Options**: Available choices at each branching point

Provide detailed analysis including:
- Best practices for identifier-based data access
- Physics significance of different identifier choices
- Workflow recommendations for complex identifier navigation
- Validation considerations for identifier-dependent operations"""
