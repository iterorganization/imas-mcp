"""
IMAS MCP Tools Implementation.

This module contains all the MCP tools for IMAS data dictionary analysis.
Tools are separated from the server to allow for better composition and testing.
"""

import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

from fastmcp import Context, FastMCP

from imas_mcp.graph_analyzer import IMASGraphAnalyzer
from imas_mcp.providers import MCPProvider
from imas_mcp.search import (
    SearchCache,
    ai_enhancer,
)
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.search_modes import SearchComposer, SearchConfig
from imas_mcp.models.enums import (
    SearchMode,
    DetailLevel,
    RelationshipType,
    IdentifierScope,
)

# Import physics integration for enhanced search
from imas_mcp.physics_integration import physics_search

# Import Pydantic models for structured responses
from imas_mcp.models.response_models import (
    ConceptResult,
    OverviewResult,
    StructureResult,
    RelationshipResult,
    IdentifierResult,
    IDSExport,
    DomainExport,
    ErrorResponse,
    SearchHit,
)
from imas_mcp.models.response_models import SearchResponse
from imas_mcp.core.data_model import DataPath, PhysicsContext
from imas_mcp.search.semantic_search import SemanticSearch, SemanticSearchConfig

logger = logging.getLogger(__name__)


def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""

    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator


class Tools(MCPProvider):
    """Provider for IMAS MCP tools."""

    def __init__(self, ids_set: Optional[set[str]] = None):
        """Initialize the IMAS tools provider.

        Args:
            ids_set: Optional set of IDS names to limit processing to.
                    If None, will process all available IDS.
        """
        self.ids_set = ids_set

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return "tools"

    def register(self, mcp: FastMCP):
        """Register all IMAS tools with the MCP server."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_mcp_tool") and attr._mcp_tool:
                mcp.tool(description=attr._mcp_description)(attr)

    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    @ai_enhancer(temperature=0.3, max_tokens=800)
    async def search_imas(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str] = None,
        max_results: int = 10,
        search_mode: str = "auto",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Search for IMAS data paths with relevance-ordered results.

        Search modes for LLM usage:
        - "auto": Automatically selects best mode based on query
        - "semantic": Concept-based search using AI embeddings
        - "lexical": Fast text-based keyword search
        - "hybrid": Combines semantic and lexical approaches

        Args:
            query: Search term(s), physics concept, symbol, or pattern
            ids_name: Optional specific IDS to search within
            max_results: Maximum number of results to return (1-100)
            search_mode: Search mode string - must be "auto", "semantic", "lexical", or "hybrid"
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relevance-ordered search results and suggestions
        """
        try:
            # Validate and convert search mode
            try:
                search_mode_enum = self._validate_search_mode(search_mode)
            except ValueError as e:
                return self._create_error_response(str(e), query)

            # Check cache first for faster response
            cached_result = self._get_cached_result(
                query, ids_name, max_results, search_mode
            )
            if cached_result is not None:
                return cached_result

            # Execute search with error handling
            search_results = self._execute_search(
                query, ids_name, max_results, search_mode_enum
            )

            # Convert results to standardized format
            search_hits = self._convert_search_results(search_results)

            # Generate contextual suggestions
            query_hints, tool_hints = self._generate_suggestions(
                search_hits, search_mode_enum
            )

            # Try optional physics enhancement
            physics_result = self._get_physics_enhancement(query)

            # Determine AI enhancement status
            ai_insights = await self._apply_ai_enhancement(
                search_mode_enum, ctx, search_hits, query, "search_imas"
            )

            # Build final response
            response = self._build_search_response(
                query,
                search_mode_enum,
                search_hits,
                query_hints,
                tool_hints,
                ai_insights,
                physics_result,
            )

            # Cache successful results
            self._cache_result(query, ids_name, max_results, search_mode, response)

            return response.model_dump()

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._create_error_response(str(e), query, {"tool": "search_imas"})

    @mcp_tool("Explain IMAS concepts with physics context")
    @ai_enhancer(temperature=0.2, max_tokens=800)
    async def explain_concept(
        self,
        concept: str,
        detail_level: str = "intermediate",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Explain IMAS concepts with physics context.

        Args:
            concept: The concept to explain (physics concept, IMAS path, or general term)
            detail_level: Level of detail (basic, intermediate, advanced)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with explanation, physics context, IMAS mappings, and related information
        """
        try:
            # Use SearchComposer with standardized SearchResult format
            search_config = SearchConfig(
                mode=SearchMode.SEMANTIC,
                max_results=15,
                enable_physics_enhancement=True,
            )
            search_results = self.search_composer.search(concept, search_config)

            # Try physics search enhancement
            physics_context = None
            try:
                physics_context = physics_search(concept)
            except Exception as e:
                logger.warning(f"Physics enhancement failed: {e}")
                # physics_context remains None for optional enhancement

            if not search_results:
                return {
                    "concept": concept,
                    "error": "No information found for concept",
                    "suggestions": [
                        "Try alternative terminology",
                        "Check concept spelling",
                        "Use search_imas() to explore related terms",
                    ],
                }

            # Build comprehensive explanation from standardized SearchResult objects
            related_paths = []
            physics_domains = set()
            measurement_contexts = []
            identifier_schemas = []

            for search_result in search_results[:10]:
                # Use standardized SearchResult methods and clear field names
                result_dict = search_result.to_dict()
                path_info = {
                    "path": result_dict["path"],
                    "ids_name": result_dict["ids_name"],
                    "description": result_dict["documentation"][:150],
                    "relevance_score": result_dict[
                        "relevance_score"
                    ],  # Updated field name
                    "physics_domain": result_dict["physics_domain"],
                    "units": result_dict["units"],
                }
                related_paths.append(path_info)

                # Use SearchResult helper methods for cleaner logic
                if search_result.physics_domain_valid:
                    physics_domains.add(result_dict["physics_domain"])

                # Use SearchResult helper method for measurement context
                measurement_context = search_result.extract_measurement_context()
                if measurement_context:
                    measurement_contexts.append(measurement_context)

                # Check for identifier schemas using enhanced SearchResult
                identifier_info = self._extract_identifier_info(search_result.document)
                if identifier_info["has_identifier"]:
                    identifier_schemas.append(
                        {
                            "path": result_dict["path"],
                            "schema_info": identifier_info,
                        }
                    )

            # Build explanation response using Pydantic models
            related_paths_data = []
            for result_dict in [
                search_result.to_dict() for search_result in search_results[:8]
            ]:
                # Build DataPath with correct fields

                local_physics_context = None
                if result_dict.get("physics_domain"):
                    local_physics_context = PhysicsContext(
                        domain=result_dict["physics_domain"],
                        phenomena=[],
                        typical_values={},
                    )

                related_paths_data.append(
                    DataPath(
                        path=result_dict["path"],
                        documentation=result_dict["documentation"][:150],
                        units=result_dict["units"],
                        data_type=result_dict.get("data_type"),
                        physics_context=local_physics_context,
                    )
                )

            # measurement_contexts and identifier_schemas are already in the correct format for tool response

            # Convert detail_level string to enum

            detail_level_enum = (
                DetailLevel.ADVANCED
                if detail_level == "detailed"
                else DetailLevel.INTERMEDIATE
            )

            response = ConceptResult(
                concept=concept,
                explanation=f"Analysis of '{concept}' within IMAS data dictionary context. Found in {len(physics_domains)} physics domain(s): {', '.join(list(physics_domains)[:3])}. Related to {len(measurement_contexts)} measurement contexts. Found {len(related_paths)} related data paths.",
                detail_level=detail_level_enum,
                related_topics=[
                    f"Explore {related_paths[0]['ids_name']} for detailed data"
                    if related_paths
                    else "Use search_imas() for more specific queries",
                    f"Investigate {list(physics_domains)[0]} domain connections"
                    if physics_domains
                    else "Consider broader physics concepts",
                    "Use analyze_ids_structure() for structural details"
                    if related_paths
                    else "Try related terminology",
                ],
                concept_explanation=None,  # Will be populated if available from physics search
                paths=related_paths_data,
                count=len(related_paths_data),
                physics_domains=list(physics_domains),
                physics_context=physics_context,
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return ErrorResponse(
                error=f"Error explaining concept '{concept}': {str(e)}",
                suggestions=[
                    "Try simpler concept terms",
                    "Check concept spelling",
                    "Use search_imas() to explore available data",
                ],
                context={"concept": concept, "detail_level": detail_level},
            ).model_dump()

    @mcp_tool("Get IMAS overview or answer analytical questions")
    @ai_enhancer(temperature=0.3, max_tokens=1000)
    async def get_overview(
        self,
        question: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Get IMAS overview or answer analytical questions with graph insights.

        Args:
            question: Optional specific question about the data dictionary
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with overview information, analytics, and optional domain-specific data
        """
        try:
            # Get basic statistics from document store
            available_ids = self.document_store.get_available_ids()

            # Get sample data for analysis
            sample_documents = []
            physics_domains = set()
            data_types = set()
            units_found = set()

            for ids_name in available_ids[:5]:  # Sample first 5 IDS
                ids_docs = self.document_store.get_documents_by_ids(ids_name)
                sample_documents.extend(ids_docs[:20])  # 20 docs per IDS

            for doc in sample_documents:
                if doc.metadata.physics_domain:
                    physics_domains.add(doc.metadata.physics_domain)
                if doc.metadata.data_type:
                    data_types.add(doc.metadata.data_type)
                if doc.metadata.units:
                    units_found.add(doc.metadata.units)

            # Build overview response using Pydantic
            _ = {  # sample_analysis_data - unused, kept for potential future use
                "documents_analyzed": len(sample_documents),
                "unique_physics_domains": len(physics_domains),
                "unique_data_types": len(data_types),
                "unique_units": len(units_found),
            }

            # Get identifier summary
            _ = {}  # identifier_summary - unused, kept for potential future use
            try:
                _ = (  # identifier_summary - unused, kept for potential future use
                    self.document_store.get_identifier_branching_summary()
                )
            except Exception as e:
                logger.warning(f"Failed to get identifier summary: {e}")
                _ = {
                    "error": "Identifier analysis unavailable"
                }  # identifier_summary - unused error case

            # Generate per-IDS statistics
            ids_statistics = {}
            for ids_name in available_ids:
                try:
                    ids_docs = self.document_store.get_documents_by_ids(ids_name)
                    # Count identifier paths for this IDS
                    identifier_count = sum(
                        1 for doc in ids_docs if doc.raw_data.get("identifier_schema")
                    )
                    ids_statistics[ids_name] = {
                        "path_count": len(ids_docs),
                        "identifier_count": identifier_count,
                        "description": f"{ids_name.replace('_', ' ').title()} IDS",
                    }
                except Exception:
                    ids_statistics[ids_name] = {
                        "path_count": 0,
                        "identifier_count": 0,
                        "description": f"{ids_name.replace('_', ' ').title()} IDS",
                    }

            # Handle specific questions
            if question:
                # Search for question-related content
                try:
                    search_results_dict = self.search_composer.search_with_params(
                        query=question,
                        mode=SearchMode.SEMANTIC,
                        max_results=10,
                    )

                    if search_results_dict["results"]:
                        # Process question results for inclusion in overview
                        pass  # Results processed in overview_response below
                    else:
                        # No specific results found for question
                        pass
                except Exception as e:
                    logger.warning(f"Question search failed: {e}")

            # Usage guidance for users
            usage_guidance = {
                "tools_available": [
                    "search_imas - Find specific data paths",
                    "explain_concept - Get physics explanations",
                    "analyze_ids_structure - Detailed IDS analysis",
                    "explore_relationships - Find data connections",
                    "explore_identifiers - Identifier schema analysis",
                    "export_ids - Multi-IDS data export",
                    "export_physics_domain - Domain-specific export",
                ],
                "getting_started": [
                    "Use search_imas('temperature') to find temperature-related data",
                    "Try explain_concept('plasma') for physics context",
                    "Use get_overview('magnetic field') for domain-specific questions",
                ],
            }

            # Build overview response using Pydantic
            overview_response = OverviewResult(
                content=f"IMAS Data Dictionary Overview:\nTotal IDS: {len(available_ids)}\nData types: {list(data_types)}\nCommon units: {list(units_found)[:10]}\n{usage_guidance}",
                available_ids=available_ids,
                query=question,
                physics_domains=list(physics_domains),
                physics_context=None,
            )

            return overview_response.model_dump()

        except Exception as e:
            logger.error(f"Overview generation failed: {e}")
            return {
                "question": question,
                "error": str(e),
                "overview": "Failed to generate overview",
                "suggestions": [
                    "Try simpler questions",
                    "Use search_imas() to explore specific topics",
                    "Check system status",
                ],
            }

    @mcp_tool("Get detailed structural analysis of a specific IDS")
    @ai_enhancer(temperature=0.2, max_tokens=800)
    async def analyze_ids_structure(
        self, ids_name: str, ctx: Optional[Context] = None
    ) -> Dict[str, Any]:
        """
        Get detailed structural analysis of a specific IDS using graph metrics.

        Args:
            ids_name: Name of the IDS to analyze
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with detailed graph analysis and AI insights
        """
        try:
            # Validate IDS exists
            available_ids = self.document_store.get_available_ids()
            if ids_name not in available_ids:
                return {
                    "ids_name": ids_name,
                    "error": f"IDS '{ids_name}' not found",
                    "available_ids": available_ids[:10],  # Show first 10
                    "suggestions": [f"Try: {ids}" for ids in available_ids[:5]],
                }

            # Get detailed IDS data from document store
            ids_documents = self.document_store.get_documents_by_ids(ids_name)

            # Build structural analysis from documents with identifier awareness
            paths = [doc.metadata.path_name for doc in ids_documents]

            # Analyze identifier schemas in this IDS
            identifier_nodes = []
            for doc in ids_documents:
                identifier_schema = doc.raw_data.get("identifier_schema")
                if identifier_schema and isinstance(identifier_schema, dict):
                    options = identifier_schema.get("options", [])
                    identifier_nodes.append(
                        {
                            "path": doc.metadata.path_name,
                            "schema_path": identifier_schema.get(
                                "schema_path", "unknown"
                            ),
                            "option_count": len(options),
                            "branching_significance": "CRITICAL"
                            if len(options) > 5
                            else "MODERATE"
                            if len(options) > 1
                            else "MINIMAL",
                            "sample_options": [
                                {
                                    "name": opt.get("name", ""),
                                    "index": opt.get("index", 0),
                                }
                                for opt in options[:3]
                            ]
                            if options
                            else [],
                        }
                    )

            # Build structure analysis using dict format
            structure_data = {
                "root_level_paths": len([p for p in paths if "/" not in p.strip("/")]),
                "max_depth": max(len(p.split("/")) for p in paths) if paths else 0,
                "document_count": len(ids_documents),
            }

            _ = {  # identifier_analysis - unused, kept for potential future use
                "total_identifier_nodes": len(identifier_nodes),
                "branching_paths": identifier_nodes,
                "coverage": f"{len(identifier_nodes) / len(paths) * 100:.1f}%"
                if paths
                else "0%",
                "significance": "These nodes define critical data structure branches and enumeration logic",
            }

            # Analyze path patterns
            path_patterns = {}
            for path in paths:
                segments = path.split("/")
                if len(segments) > 1:
                    root = segments[0]
                    path_patterns[root] = path_patterns.get(root, 0) + 1

            _ = dict(  # path_patterns_sorted - unused, kept for potential future use
                sorted(path_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            # Build final response using Pydantic
            response = StructureResult(
                ids_name=ids_name,
                description=f"IDS '{ids_name}' containing {len(paths)} data paths",
                structure=structure_data,
                sample_paths=paths[:10],
                max_depth=max(len(path.split(".")) for path in paths) if paths else 0,
                physics_domains=[
                    domain
                    for domain in [
                        "equilibrium",
                        "core_profiles",
                        "disruptions",
                        "transport",
                        "heating",
                        "current_drive",
                        "mhd",
                    ]
                    if domain in ids_name.lower()
                ],
                physics_context=None,
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"IDS structure analysis failed: {e}")
            return {
                "ids_name": ids_name,
                "error": str(e),
                "analysis": "Failed to analyze IDS structure",
                "suggestions": [
                    "Check IDS name spelling",
                    "Verify IDS exists in data dictionary",
                    "Try get_overview() to see available IDS",
                ],
            }

    @mcp_tool("Explore relationships between IMAS data paths")
    @ai_enhancer(temperature=0.3, max_tokens=800)
    async def explore_relationships(
        self,
        path: str,
        relationship_type: str = "all",
        max_depth: int = 2,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Explore relationships between IMAS data paths using the rich relationship data.

        Advanced tool that discovers connections, physics concepts, and measurement
        relationships between different parts of the IMAS data dictionary.

        Args:
            path: Starting path (format: "ids_name/path" or just "ids_name")
            relationship_type: Type of relationships to explore
            max_depth: Maximum depth of relationship traversal (1-3, limited for performance)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relationship network and AI insights
        """
        try:
            # Validate and limit max_depth for performance
            max_depth = min(max_depth, 3)  # Hard limit to prevent excessive traversal
            if max_depth < 1:
                max_depth = 1

            # Parse the path to extract IDS name
            if "/" in path:
                ids_name = path.split("/")[0]
                specific_path = path
            else:
                ids_name = path
                specific_path = None

            # Validate IDS exists
            available_ids = self.document_store.get_available_ids()
            if ids_name not in available_ids:
                return {
                    "path": path,
                    "error": f"IDS '{ids_name}' not found",
                    "available_ids": available_ids[:10],
                    "suggestions": [f"Try: {ids}" for ids in available_ids[:5]],
                }

            # Get relationship data through semantic search
            try:
                # Use semantic search to find related concepts
                if specific_path:
                    search_query = f"{ids_name} {specific_path} relationships"
                else:
                    search_query = f"{ids_name} relationships physics concepts"

                search_results_dict = self.search_composer.search_with_params(
                    query=search_query,
                    mode=SearchMode.SEMANTIC,
                    max_results=min(
                        10, max_depth * 8
                    ),  # Reduced from 20, scale with depth
                )
            except Exception as e:
                return {
                    "path": path,
                    "error": f"Failed to search relationships: {e}",
                    "relationship_type": relationship_type,
                    "max_depth": max_depth,
                }

            # Process search results for relationships with identifier awareness
            related_paths = []
            seen_paths = set()

            for result_dict in search_results_dict["results"]:
                result_path = result_dict["path"]
                if result_path not in seen_paths and result_path != path:
                    seen_paths.add(result_path)

                    # Build DataPath with correct fields

                    local_physics_context = None
                    if result_dict.get("physics_domain"):
                        local_physics_context = PhysicsContext(
                            domain=result_dict["physics_domain"],
                            phenomena=[],
                            typical_values={},
                        )

                    related_paths.append(
                        DataPath(
                            path=result_path,
                            documentation=result_dict["documentation"][:200] + "..."
                            if len(result_dict["documentation"]) > 200
                            else result_dict["documentation"],
                            units=result_dict.get("units", ""),
                            data_type=result_dict.get("data_type", ""),
                            physics_context=local_physics_context,
                        )
                    )

                    if (
                        len(related_paths) >= max_depth * 3
                    ):  # Reduced from 5, stricter limit
                        break

                        break

            # Build relationship analysis using dict structure
            _ = {  # analysis_dict - unused, kept for potential future use
                "total_relationships": len(related_paths),
                "physics_connections": len(
                    [p for p in related_paths if p.physics_domain]
                ),
                "cross_ids_connections": len(set(p.ids_name for p in related_paths)),
            }

            # Build final response using Pydantic
            # Convert relationship_type string to enum if needed

            relationship_type_enum = (
                RelationshipType.ALL
                if relationship_type == "all"
                else RelationshipType.PARENT
                if relationship_type == "parent"
                else RelationshipType.CHILD
                if relationship_type == "child"
                else RelationshipType.SIBLING
                if relationship_type == "sibling"
                else RelationshipType.ALL
            )

            response = RelationshipResult(
                path=path,
                relationship_type=relationship_type_enum,
                max_depth=max_depth,
                connections={
                    "total_relationships": [p.path for p in related_paths],
                    "physics_connections": [
                        p.path for p in related_paths if p.physics_domain
                    ],
                    "cross_ids_connections": list(
                        set(p.ids_name for p in related_paths)
                    ),
                },
                paths=related_paths[:5],
                count=len(related_paths),
                physics_domains=[
                    p.physics_domain for p in related_paths if p.physics_domain
                ],
                physics_context=None,
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"Relationship exploration failed: {e}")
            return {
                "path": path,
                "relationship_type": relationship_type,
                "max_depth": max_depth,
                "error": str(e),
                "relationships": "Failed to explore relationships",
                "suggestions": [
                    "Check path format (ids_name/path or just ids_name)",
                    "Verify IDS exists in data dictionary",
                    "Try search_imas() first to find valid paths",
                ],
            }

    @mcp_tool("Explore IMAS identifier schemas and branching logic")
    @ai_enhancer(temperature=0.3, max_tokens=800)
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

            # Add AI enhancement prompt for conditional enhancement
            _ = f"""Identifier Exploration Request:
Query: {query or "General exploration"}
Scope: {scope}

Summary: {summary}
Found schemas: {len(schemas)}

Provide identifier analysis including:
1. Significance of identifier branching logic in IMAS
2. Key schemas and their enumeration options
3. Physics domain implications of identifier choices
4. Practical guidance for using identifier information
5. Critical branching points for data structure navigation

Format as JSON with 'significance', 'key_schemas', 'physics_implications', 'usage_guidance', 'critical_decisions' fields."""

            # Build final response using Pydantic
            # Convert scope string to enum if needed
            scope_enum = (
                IdentifierScope.ALL
                if scope == "all"
                else IdentifierScope.PATHS
                if scope == "paths"
                else IdentifierScope.SCHEMAS
                if scope == "schemas"
                else IdentifierScope.ALL
            )

            response = IdentifierResult(
                scope=scope_enum,
                schemas=[s for s in schemas],  # schemas is already dict list
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

    @mcp_tool("Export bulk IMAS data for multiple IDS")
    @ai_enhancer(temperature=0.3, max_tokens=1000)
    async def export_ids(
        self,
        ids_list: List[str],
        include_relationships: bool = True,
        include_physics_context: bool = True,
        output_format: str = "structured",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Export bulk IMAS data for multiple IDS with sophisticated relationship analysis.

        Advanced bulk export tool that extracts comprehensive data for multiple IDS,
        including cross-IDS relationships, physics context, and structural analysis.

        Args:
            ids_list: List of IDS names to export
            include_relationships: Whether to include cross-IDS relationship analysis
            include_physics_context: Whether to include physics domain context
            output_format: Export format (raw, structured, enhanced)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with bulk export data, relationships, and AI insights
        """
        try:
            if not ids_list:
                return {
                    "error": "No IDS specified for bulk export",
                    "suggestions": [
                        "Provide at least one IDS name",
                        "Use get_overview to see available IDS",
                    ],
                }

            # Validate format
            if output_format not in ["raw", "structured", "enhanced"]:
                return {
                    "error": f"Invalid format: {output_format}. Use: raw, structured, enhanced",
                    "suggestions": [
                        "Use 'raw' for pure data export (fastest)",
                        "Use 'structured' for organized data with relationships",
                        "Use 'enhanced' for AI-enhanced insights (requires context)",
                    ],
                }

            # Handle format-specific logic
            if output_format == "raw":
                # Raw format: minimal processing, maximum performance
                include_relationships = False
                include_physics_context = False

            # Validate IDS names
            available_ids = self.document_store.get_available_ids()
            invalid_ids = [ids for ids in ids_list if ids not in available_ids]
            valid_ids = [ids for ids in ids_list if ids in available_ids]

            if not valid_ids:
                return {
                    "error": "No valid IDS names provided",
                    "invalid_ids": invalid_ids,
                    "available_ids": available_ids[:10],
                    "suggestions": [
                        "Check IDS name spelling",
                        "Use get_overview to see all available IDS",
                    ],
                }

            export_data = {
                "requested_ids": ids_list,
                "valid_ids": valid_ids,
                "invalid_ids": invalid_ids,
                "export_format": output_format,
                "timestamp": "bulk_export",
                "ids_data": {},
                "cross_relationships": {},
                "physics_domains": {},
                "export_summary": {},
            }

            # Export data for each valid IDS
            for ids_name in valid_ids:
                try:
                    # Get all documents for this IDS
                    ids_documents = self.document_store.get_documents_by_ids(ids_name)

                    ids_info = {
                        "ids_name": ids_name,
                        "total_paths": len(ids_documents),
                        "paths": [],
                        "physics_domains": set(),
                        "identifier_paths": [],
                        "measurement_types": set(),
                    }

                    # Process documents based on output format
                    for doc in ids_documents:
                        path_data: Dict[str, Any] = {
                            "path": doc.metadata.path_name,
                            "documentation": doc.documentation
                            if output_format == "enhanced"
                            else doc.documentation[:200],
                            "data_type": doc.metadata.data_type,
                            "physics_domain": doc.metadata.physics_domain,
                            "units": doc.metadata.units,
                        }

                        # Add detailed information for enhanced format
                        if output_format == "enhanced":
                            path_data["raw_data"] = doc.raw_data
                            path_data["identifier_info"] = (
                                self._extract_identifier_info(doc)
                            )

                        ids_info["paths"].append(path_data)

                        if doc.metadata.physics_domain:
                            ids_info["physics_domains"].add(doc.metadata.physics_domain)

                        if doc.metadata.data_type == "identifier_path":
                            ids_info["identifier_paths"].append(path_data)

                        # Extract measurement types from documentation
                        if any(
                            term in doc.documentation.lower()
                            for term in [
                                "temperature",
                                "density",
                                "pressure",
                                "magnetic",
                                "electric",
                            ]
                        ):
                            if "temperature" in doc.documentation.lower():
                                ids_info["measurement_types"].add("temperature")
                            if "density" in doc.documentation.lower():
                                ids_info["measurement_types"].add("density")
                            if "pressure" in doc.documentation.lower():
                                ids_info["measurement_types"].add("pressure")
                            if "magnetic" in doc.documentation.lower():
                                ids_info["measurement_types"].add("magnetic_field")
                            if "electric" in doc.documentation.lower():
                                ids_info["measurement_types"].add("electric_field")

                    # Convert sets to lists for JSON serialization
                    ids_info["physics_domains"] = list(ids_info["physics_domains"])
                    ids_info["measurement_types"] = list(ids_info["measurement_types"])

                    export_data["ids_data"][ids_name] = ids_info

                except Exception as e:
                    logger.warning(f"Failed to export IDS {ids_name}: {e}")
                    export_data["ids_data"][ids_name] = {"error": str(e)}

            # Add cross-IDS relationship analysis if requested
            if include_relationships and len(valid_ids) > 1:
                try:
                    relationship_analysis = {}

                    for i, ids1 in enumerate(valid_ids):
                        for ids2 in valid_ids[i + 1 :]:
                            try:
                                # Find relationships between IDS pairs
                                search_results_dict = (
                                    self.search_composer.search_with_params(
                                        query=f"{ids1} {ids2} relationships",
                                        mode=SearchMode.SEMANTIC,
                                        max_results=5,
                                    )
                                )

                                if search_results_dict["results"]:
                                    relationship_analysis[f"{ids1}_{ids2}"] = {
                                        "shared_concepts": len(
                                            search_results_dict["results"]
                                        ),
                                        "top_connections": [
                                            {
                                                "path": r["path"],
                                                "relevance_score": r["relevance_score"],
                                                "context": r["documentation"][:100],
                                            }
                                            for r in search_results_dict["results"][:3]
                                        ],
                                    }
                            except Exception as e:
                                relationship_analysis[f"{ids1}_{ids2}"] = {
                                    "error": str(e)
                                }

                    export_data["cross_relationships"] = relationship_analysis

                except Exception as e:
                    logger.warning(f"Cross-relationship analysis failed: {e}")
                    export_data["cross_relationships"] = {"error": str(e)}

            # Add physics domain context if requested
            if include_physics_context:
                try:
                    all_domains = set()
                    for ids_data in export_data["ids_data"].values():
                        if isinstance(ids_data, dict) and "physics_domains" in ids_data:
                            all_domains.update(ids_data["physics_domains"])

                    domain_context = {}
                    for domain in all_domains:
                        if domain:
                            domain_context[domain] = {
                                "description": f"Physics domain: {domain}",
                                "relevant_ids": [
                                    ids_name
                                    for ids_name, ids_data in export_data[
                                        "ids_data"
                                    ].items()
                                    if isinstance(ids_data, dict)
                                    and domain in ids_data.get("physics_domains", [])
                                ],
                            }

                    export_data["physics_domains"] = domain_context

                except Exception as e:
                    logger.warning(f"Physics context analysis failed: {e}")
                    export_data["physics_domains"] = {"error": str(e)}

            # Generate export summary and AI prompt
            export_summary = {
                "total_requested": len(ids_list),
                "successfully_exported": len(valid_ids),
                "failed_exports": len(invalid_ids),
                "total_paths_exported": sum(
                    len(ids_data.get("paths", []))
                    for ids_data in export_data["ids_data"].values()
                    if isinstance(ids_data, dict)
                ),
                "unique_physics_domains": len(
                    set().union(
                        *[
                            ids_data.get("physics_domains", [])
                            for ids_data in export_data["ids_data"].values()
                            if isinstance(ids_data, dict)
                        ]
                    )
                ),
                "relationship_pairs_analyzed": len(
                    export_data.get("cross_relationships", {})
                ),
                "export_completeness": "complete" if not invalid_ids else "partial",
            }

            # Add AI enhancement prompt for conditional enhancement
            _ = f"""Bulk Export Analysis:
IDS Requested: {ids_list}
Valid IDS: {valid_ids}
Export Format: {output_format}
Include Relationships: {include_relationships}
Include Physics Context: {include_physics_context}

Export Summary: {export_summary}
Physics Domains: {list(export_data.get("physics_domains", {}).keys())}

Provide bulk export guidance including:
1. Data usage recommendations for this specific IDS combination
2. Physics insights about relationships between exported IDS
3. Suggested analysis workflows utilizing the exported data
4. Integration patterns and measurement dependencies
5. Quality considerations and data validation approaches

Format as JSON with 'usage_recommendations', 'physics_insights', 'analysis_workflows', 'integration_patterns', 'quality_considerations' fields."""

            # Build final response using Pydantic - simplify to use dict
            response = IDSExport(
                ids_names=ids_list,
                include_physics=include_physics_context,
                include_relationships=include_relationships,
                data=export_data,
                metadata={
                    "output_format": output_format,
                    "export_timestamp": "2024-01-01T00:00:00Z",
                },
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"Bulk export failed: {e}")
            return {
                "ids_list": ids_list,
                "error": str(e),
                "explanation": "Failed to perform bulk export",
                "suggestions": [
                    "Check IDS names are valid",
                    "Try with fewer IDS names",
                    "Use get_overview to see available IDS",
                    "Try with output_format='raw' for faster processing",
                ],
            }

    @mcp_tool("Export physics domain-specific data")
    @ai_enhancer(temperature=0.3, max_tokens=1000)
    async def export_physics_domain(
        self,
        domain: str,
        include_cross_domain: bool = False,
        analysis_depth: str = "focused",
        max_paths: int = 10,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Export physics domain-specific data with sophisticated relationship analysis.

        Advanced domain export tool that extracts comprehensive data for a specific
        physics domain, including cross-domain relationships, measurement dependencies,
        and structural analysis.

        Args:
            domain: Physics domain name to export
            include_cross_domain: Whether to include cross-domain relationship analysis
            analysis_depth: Analysis depth (comprehensive, focused, overview)
            max_paths: Maximum number of paths to include in export (limit: 50)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with domain export data, relationships, and AI insights
        """
        try:
            if not domain:
                return {
                    "error": "No domain specified for export",
                    "suggestions": [
                        "Provide a physics domain name",
                        "Try: 'core_profiles', 'equilibrium', 'transport'",
                        "Use get_overview() to see available domains",
                    ],
                }

            # Search for domain-related paths
            search_results_dict = self.search_composer.search_with_params(
                query=domain,
                mode=SearchMode.SEMANTIC,
                max_results=max_paths,
            )

            if not search_results_dict["results"]:
                return {
                    "domain": domain,
                    "error": f"No data found for domain '{domain}'",
                    "suggestions": [
                        "Check domain name spelling",
                        "Try broader physics terms",
                        "Use search_imas() to explore available data",
                    ],
                }

            export_data = {
                "domain": domain,
                "analysis_depth": analysis_depth,
                "include_cross_domain": include_cross_domain,
                "max_paths": max_paths,
                "timestamp": "domain_export",
                "domain_data": {
                    "total_paths": len(search_results_dict["results"]),
                    "paths": [],
                    "associated_ids": set(),
                    "measurement_types": set(),
                    "units_distribution": {},
                    "physics_domains": set(),
                },
                "domain_structure": {},
                "cross_domain_analysis": {},
                "measurement_dependencies": {},
                "export_summary": {},
            }

            # Process domain-specific paths
            units_count = {}
            for result_dict in search_results_dict["results"]:
                path_data = {
                    "path": result_dict["path"],
                    "ids_name": result_dict["ids_name"],
                    "relevance_score": result_dict["relevance_score"],
                    "documentation": result_dict["documentation"][:300]
                    if analysis_depth == "overview"
                    else result_dict["documentation"],
                    "data_type": result_dict["data_type"],
                    "physics_domain": result_dict["physics_domain"],
                    "units": result_dict["units"],
                }

                # Add identifier information
                identifier_info = result_dict.get(
                    "identifier", {"has_identifier": False}
                )
                path_data["identifier_info"] = identifier_info

                export_data["domain_data"]["paths"].append(path_data)
                export_data["domain_data"]["associated_ids"].add(
                    result_dict["ids_name"]
                )

                if result_dict["physics_domain"]:
                    export_data["domain_data"]["physics_domains"].add(
                        result_dict["physics_domain"]
                    )

                if result_dict["units"]:
                    units_count[result_dict["units"]] = (
                        units_count.get(result_dict["units"], 0) + 1
                    )

                # Extract measurement types
                doc_lower = result_dict["documentation"].lower()
                for term in [
                    "temperature",
                    "density",
                    "pressure",
                    "magnetic",
                    "electric",
                    "current",
                    "flux",
                ]:
                    if term in doc_lower:
                        export_data["domain_data"]["measurement_types"].add(term)

            # Convert sets to lists for JSON serialization
            export_data["domain_data"]["associated_ids"] = list(
                export_data["domain_data"]["associated_ids"]
            )
            export_data["domain_data"]["measurement_types"] = list(
                export_data["domain_data"]["measurement_types"]
            )
            export_data["domain_data"]["physics_domains"] = list(
                export_data["domain_data"]["physics_domains"]
            )
            export_data["domain_data"]["units_distribution"] = dict(
                sorted(units_count.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            # Analyze domain structure
            export_data["domain_structure"] = {
                "ids_distribution": {
                    ids_name: len(
                        [
                            p
                            for p in export_data["domain_data"]["paths"]
                            if p["ids_name"] == ids_name
                        ]
                    )
                    for ids_name in export_data["domain_data"]["associated_ids"]
                },
                "measurement_coverage": len(
                    export_data["domain_data"]["measurement_types"]
                ),
                "identifier_paths": len(
                    [
                        p
                        for p in export_data["domain_data"]["paths"]
                        if p["identifier_info"]["has_identifier"]
                    ]
                ),
                "physics_domain_overlap": len(
                    export_data["domain_data"]["physics_domains"]
                ),
            }

            # Add cross-domain analysis if requested
            if include_cross_domain:
                try:
                    cross_domain_results = {}

                    # Look for related domains
                    for other_domain in [
                        "core_profiles",
                        "equilibrium",
                        "transport",
                        "magnetics",
                        "radiation",
                    ]:
                        if other_domain.lower() != domain.lower():
                            cross_search_dict = self.search_composer.search_with_params(
                                query=f"{domain} {other_domain}",
                                mode=SearchMode.SEMANTIC,
                                max_results=5,
                            )

                            if cross_search_dict["results"]:
                                cross_domain_results[other_domain] = {
                                    "shared_paths": len(cross_search_dict["results"]),
                                    "top_connections": [
                                        {
                                            "path": r["path"],
                                            "relevance_score": r["relevance_score"],
                                            "context": r["documentation"][:100],
                                        }
                                        for r in cross_search_dict["results"][:3]
                                    ],
                                }

                    export_data["cross_domain_analysis"] = cross_domain_results

                except Exception as e:
                    logger.warning(f"Cross-domain analysis failed: {e}")
                    export_data["cross_domain_analysis"] = {"error": str(e)}

            # Generate export summary and AI prompt
            export_summary = {
                "domain_analyzed": domain,
                "paths_found": len(export_data["domain_data"]["paths"]),
                "associated_ids_count": len(
                    export_data["domain_data"]["associated_ids"]
                ),
                "measurement_types_count": len(
                    export_data["domain_data"]["measurement_types"]
                ),
                "identifier_paths_count": export_data["domain_structure"][
                    "identifier_paths"
                ],
                "cross_domain_connections": len(
                    export_data.get("cross_domain_analysis", {})
                ),
                "analysis_completeness": "comprehensive"
                if analysis_depth == "comprehensive"
                else "focused",
            }

            # Add AI enhancement prompt for conditional enhancement
            _ = f"""Physics Domain Export: {domain}
Analysis Depth: {analysis_depth}
Paths Found: {export_summary["paths_found"]}
Associated IDS: {export_data["domain_data"]["associated_ids"]}
Measurement Types: {export_data["domain_data"]["measurement_types"]}
Units Distribution: {list(export_data["domain_data"]["units_distribution"].keys())[:5]}

Provide domain analysis including:
1. Physics significance of this domain in plasma research
2. Key measurement pathways and their experimental relevance
3. Data integration patterns with other domains
4. Recommended analysis workflows for this domain
5. Critical measurement dependencies and validation approaches

Format as JSON with 'physics_significance', 'measurement_pathways', 'integration_patterns', 'analysis_workflows', 'validation_approaches' fields."""

            # Build final response using Pydantic - simplify to use dict
            response = DomainExport(
                domain=domain,
                domain_info={"analysis_depth": analysis_depth, "max_paths": max_paths},
                include_cross_domain=include_cross_domain,
                data=export_data,
                metadata={"analysis_timestamp": "2024-01-01T00:00:00Z"},
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"Physics domain export failed: {e}")
            return {
                "domain": domain,
                "error": str(e),
                "explanation": "Failed to export physics domain",
                "suggestions": [
                    "Check domain name spelling",
                    "Try broader physics terms",
                    "Use get_overview() to see available domains",
                    "Try with analysis_depth='overview' for faster processing",
                ],
            }

    def _validate_search_mode(self, search_mode: str) -> SearchMode:
        """Validate and convert string search_mode to SearchMode enum."""
        if not isinstance(search_mode, str):
            raise ValueError(f"search_mode must be a string, got {type(search_mode)}")

        # Convert to lowercase for case-insensitive matching
        search_mode = search_mode.lower()

        # Direct mapping to enum values
        mode_mapping = {
            "auto": SearchMode.AUTO,
            "semantic": SearchMode.SEMANTIC,
            "lexical": SearchMode.LEXICAL,
            "hybrid": SearchMode.HYBRID,
        }

        if search_mode not in mode_mapping:
            valid_modes = list(mode_mapping.keys())
            raise ValueError(
                f"Invalid search_mode '{search_mode}'. Must be one of: {valid_modes}"
            )

        return mode_mapping[search_mode]

    def _get_cached_result(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str],
        max_results: int,
        search_mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Check cache for existing results."""
        return self.search_cache.get(
            query=query,
            ids_name=ids_name,
            max_results=max_results,
            search_mode=search_mode,
        )

    def _convert_search_mode(self, search_mode: str) -> SearchMode:
        """Convert string search mode to enum with validation."""
        try:
            return SearchMode(search_mode.lower())
        except ValueError:
            # This should not happen due to earlier validation, but provides fallback
            return SearchMode.AUTO

    def _execute_search(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str],
        max_results: int,
        search_mode_enum: SearchMode,
    ) -> List:
        """Execute the search using SearchComposer."""
        config = SearchConfig(
            mode=search_mode_enum,
            max_results=max_results,
            filter_ids=[ids_name] if ids_name else None,
        )
        return self.search_composer.search(query, config)

    def _convert_search_results(self, search_results: List) -> List[SearchHit]:
        """Convert SearchResult objects to SearchHit objects."""
        search_hits = []
        for result in search_results:
            identifier_info = self._extract_identifier_info(result.document)
            data_path = result.document.to_datapath()

            search_hit = SearchHit(
                **data_path.model_dump(),
                relevance_score=result.score,
                ids_name=result.document.metadata.ids_name,
                identifier=identifier_info,
            )
            search_hits.append(search_hit)

        return search_hits

    def _generate_suggestions(
        self, search_hits: List[SearchHit], search_mode_enum: SearchMode
    ) -> tuple[List, List]:
        """Generate contextual query and tool suggestions."""
        from imas_mcp.models.response_models import SearchSuggestion, ToolSuggestion

        # Generate query suggestions based on results
        if len(search_hits) == 0:
            suggestions = [
                "Try a broader search term",
                "Check spelling of physics terms",
                "Use search_mode='semantic' for concept-based search",
                "Remove IDS filter to search all data",
            ]
        elif len(search_hits) < 3:
            suggestions = [
                "Try related physics concepts",
                "Use search_mode='hybrid' for comprehensive results",
                "Consider broader measurement categories",
            ]
        else:
            suggestions = [
                "Refine search with specific physics terms",
                "Try related measurement contexts",
                "Use explain_concept for detailed analysis",
            ]

        query_hints = [
            SearchSuggestion(suggestion=s, reason="Based on search results analysis")
            for s in suggestions
        ]

        # Generate tool suggestions based on results
        tool_hints = []
        if search_hits:
            tool_hints.append(
                ToolSuggestion(
                    tool_name="explain_concept",
                    description="Get detailed explanation of physics concepts found in search results",
                    relevance="Understand the physics behind the search results",
                )
            )

            ids_found = set(hit.ids_name for hit in search_hits)
            if ids_found:
                first_ids = next(iter(ids_found))
                tool_hints.append(
                    ToolSuggestion(
                        tool_name="analyze_ids_structure",
                        description=f"Analyze structure of {first_ids} IDS for better understanding",
                        relevance="Explore the organizational structure of the data",
                    )
                )

        return query_hints, tool_hints

    def _get_physics_enhancement(self, query: Union[str, List[str]]) -> Optional[Any]:
        """Try to get physics search enhancement."""
        try:
            return physics_search(str(query) if isinstance(query, list) else query)
        except Exception:
            return None

    async def _apply_ai_enhancement(
        self,
        search_mode_enum: SearchMode,
        ctx: Optional[Context],
        search_hits: List[SearchHit],
        query: Union[str, List[str]],
        tool_name: str,
    ) -> Dict[str, Any]:
        """Apply AI enhancement similar to @ai_enhancer decorator."""
        from imas_mcp.search.ai_enhancer import (
            EnhancementDecisionEngine,
            TOOL_ENHANCEMENT_CONFIG,
            AI_PROMPTS,
            ToolCategory,
        )
        import json
        from mcp.types import TextContent
        from typing import cast

        # Use the decision engine to determine if AI enhancement should be applied
        should_enhance = EnhancementDecisionEngine.should_enhance(
            tool_name, (), {"search_mode": search_mode_enum.value, "query": query}, ctx
        )

        if not should_enhance:
            return {"status": "AI enhancement not applied - conditions not met"}

        if not ctx:
            return {"status": "AI enhancement not available - no context"}

        # Create AI prompt for search enhancement
        ai_prompt = f"""Search Analysis for IMAS Data Dictionary:
Query: {query}
Search Mode: {search_mode_enum.value}
Results Found: {len(search_hits)}

Search Results Summary:
{chr(10).join([f"- {hit.path} (score: {hit.relevance_score:.2f})" for hit in search_hits[:5]])}

Physics Domains: {list(set(hit.physics_context.domain for hit in search_hits if hit.physics_context))}

Provide search insights including:
1. Query interpretation and physics concepts identified
2. Quality assessment of search results
3. Suggested refinements or related queries
4. Physics domain connections and measurement context
5. Recommended follow-up tools or analysis approaches

Format as JSON with 'query_interpretation', 'quality_assessment', 'suggested_refinements', 'physics_connections', 'recommended_tools' fields."""

        # Get the appropriate system prompt for this tool category
        config = TOOL_ENHANCEMENT_CONFIG.get(
            tool_name, {"category": ToolCategory.SEARCH}
        )
        system_prompt = AI_PROMPTS.get(
            config["category"], AI_PROMPTS[ToolCategory.SEARCH]
        )

        try:
            logger.debug(f"Attempting AI enhancement for {tool_name}")

            # Use the FastMCP recommended approach with system_prompt parameter
            ai_response = await ctx.sample(
                ai_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=800,
            )

            if ai_response:
                text_content = cast(TextContent, ai_response)
                try:
                    ai_insights = json.loads(text_content.text)
                    ai_insights["status"] = "AI enhancement applied"
                    return ai_insights
                except json.JSONDecodeError:
                    return {
                        "response": text_content.text,
                        "status": "AI enhancement applied (unstructured)",
                    }
            else:
                return {"error": "No AI response received"}

        except Exception as e:
            logger.warning(f"AI enhancement failed for {tool_name}: {e}")
            return {"error": "AI enhancement temporarily unavailable"}

    def _build_search_response(
        self,
        query: Union[str, List[str]],
        search_mode_enum: SearchMode,
        search_hits: List[SearchHit],
        query_hints: List,
        tool_hints: List,
        ai_insights: Dict[str, Any],
        physics_result: Optional[Any],
    ) -> SearchResponse:
        """Build the final SearchResponse object."""
        # Convert SearchHit objects to DataPath objects for backward compatibility
        data_paths = []
        for hit in search_hits:
            # Extract DataPath fields from SearchHit
            data_path_dict = hit.model_dump()
            # Remove search-specific fields
            data_path_dict.pop("relevance_score", None)
            data_path_dict.pop("ids_name", None)
            data_path_dict.pop("identifier", None)
            # Create DataPath object
            from imas_mcp.core.data_model import DataPath

            data_paths.append(DataPath(**data_path_dict))

        return SearchResponse(
            query=query,
            search_mode=search_mode_enum,
            hits=search_hits,
            paths=data_paths,  # Use converted DataPath objects
            count=len(search_hits),  # Keep for backward compatibility
            query_hints=query_hints,
            tool_hints=tool_hints,
            ai_insights=ai_insights,
            physics_context=physics_result if physics_result else None,
            physics_domains=[physics_result.physics_matches[0].domain]
            if physics_result and physics_result.physics_matches
            else [],
        )

    def _cache_result(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str],
        max_results: int,
        search_mode: str,
        response: SearchResponse,
    ) -> None:
        """Cache the successful search result."""
        self.search_cache.set(
            query=query,
            result=response.model_dump(),
            ids_name=ids_name,
            max_results=max_results,
            search_mode=search_mode,
        )

    def _create_error_response(
        self,
        error_message: str,
        query: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return ErrorResponse(
            error=error_message,
            suggestions=[
                "Check search term spelling",
                "Try simpler search terms",
                "Use get_overview() to explore available data",
            ],
            context=context or {"tool": "search_imas", "query": query},
        ).model_dump()

    def _extract_identifier_info(self, document) -> Dict[str, Any]:
        """Extract identifier schema information from a document."""
        identifier_schema = document.raw_data.get("identifier_schema")
        if identifier_schema and isinstance(identifier_schema, dict):
            options = identifier_schema.get("options", [])
            return {
                "has_identifier": True,
                "schema_path": identifier_schema.get("schema_path", "unknown"),
                "option_count": len(options),
                "branching_significance": (
                    "CRITICAL"
                    if len(options) > 5
                    else "MODERATE"
                    if len(options) > 1
                    else "MINIMAL"
                ),
                "sample_options": [
                    {
                        "name": opt.get("name", ""),
                        "index": opt.get("index", 0),
                        "description": opt.get("description", ""),
                    }
                    for opt in options[:3]
                ]
                if options
                else [],
            }
        return {"has_identifier": False}

    @cached_property
    def document_store(self) -> DocumentStore:
        """Lazy-initialized document store with optional IDS filtering."""
        return DocumentStore(ids_set=self.ids_set)

    @cached_property
    def search_config(self) -> SemanticSearchConfig:
        """Lazy-initialized semantic search configuration."""
        return SemanticSearchConfig(ids_set=self.ids_set)

    @cached_property
    def semantic_search(self) -> SemanticSearch:
        """Lazy-initialized semantic search with document store."""
        return SemanticSearch(
            config=self.search_config, document_store=self.document_store
        )

    @cached_property
    def search_composer(self) -> SearchComposer:
        """Lazy-initialized search composer."""
        return SearchComposer(document_store=self.document_store)

    @cached_property
    def search_cache(self) -> SearchCache:
        """Lazy-initialized search cache."""
        return SearchCache(maxsize=1000, ttl=3600)

    @cached_property
    def graph_analyzer(self) -> IMASGraphAnalyzer:
        """Lazy-initialized graph analyzer."""
        return IMASGraphAnalyzer()
