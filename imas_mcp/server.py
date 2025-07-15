"""
IMAS MCP Server with AI Tools - Refactored with Composition Pattern.

This is the principal MCP server for the IMAS data dictionary, providing AI
tools for physics-based search, analysis, and exploration of plasma physics data.
It offers 5 focused tools with intelligent insights and relevance-ranked results
for better LLM usage.

The 5 core tools provide comprehensive coverage:
1. search_imas - Enhanced search with physics concepts, symbols, and units
2. explain_concept - Physics explanations with IMAS mappings and domain context
3. get_overview - General overview with domain analysis and query validation
4. analyze_ids_structure - Detailed structural analysis of specific IDS
5. explore_relationships - Advanced relationship exploration across the data dictionary

This server uses a composition pattern with dedicated search handlers to separate
concerns and make the intent of each search strategy clear.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
from fastmcp import Context, FastMCP

from .json_data_accessor import JsonDataDictionaryAccessor
from .lexicographic_search import LexicographicSearch
from .search import (
    SearchHandler,
    SearchRequest,
    SearchRouter,
    ai_enhancer,
    SEARCH_EXPERT,
    EXPLANATION_EXPERT,
    OVERVIEW_EXPERT,
    STRUCTURE_EXPERT,
    RELATIONSHIP_EXPERT,
)

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Server:
    """AI IMAS MCP Server with structured tool management using composition pattern."""

    mcp: FastMCP = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        self.mcp = FastMCP(name="imas-dd")
        self._register_tools()

    @cached_property
    def data_accessor(self) -> JsonDataDictionaryAccessor:
        """Return the JSON data accessor instance."""
        logger.info("Initializing JSON data accessor")
        accessor = JsonDataDictionaryAccessor()

        if not accessor.is_available():
            raise ValueError(
                "IMAS JSON data is not available. This could be due to:\\n"
                "1. Data dictionary not properly installed\\n"
                "2. JSON files not built during installation\\n"
                "Please reinstall the package or run the build process."
            )

        return accessor

    @cached_property
    def lex_search(self):
        """Return the lexicographic search instance."""
        logger.info("Initializing lexicographic search")
        return LexicographicSearch()

    @cached_property
    def search_handler(self) -> SearchHandler:
        """Return the search handler instance."""
        logger.info("Initializing search handler with composition pattern")
        return SearchHandler(
            json_accessor=self.data_accessor, lex_search=self.lex_search
        )

    @cached_property
    def search_router(self) -> SearchRouter:
        """Return the search router instance."""
        return SearchRouter(
            data_accessor=self.data_accessor, lexicographic_search=self.lex_search
        )

    def _register_tools(self):
        """Register the 5 focused AI MCP tools with the server."""
        # Register the core tools - consolidated to 5 focused tools
        self.mcp.tool(self.search_imas)
        self.mcp.tool(self.explain_concept)
        self.mcp.tool(self.get_overview)
        self.mcp.tool(self.analyze_ids_structure)
        self.mcp.tool(self.explore_relationships)

    @ai_enhancer(SEARCH_EXPERT, "Search analysis", temperature=0.3, max_tokens=500)
    async def search_imas(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str] = None,
        max_results: int = 10,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Search for IMAS data paths with relevance-ordered results and AI enhancement.

        Advanced search tool that finds IMAS data paths, scores them by relevance,
        and optionally enhances results with AI insights when MCP sampling is available.

        Enhanced with physics context - can search by physics concepts, symbols, or units.
        Supports bulk search, boolean operators, wildcards, and field-specific searches.

        Args:
            query: Search term(s), physics concept, symbol, or pattern. Can be:
                   - Single string: "plasma temperature"
                   - List of strings: ["plasma", "temperature", "density"]
                   - Boolean operators: "plasma AND temperature"
                   - Wildcards: "temp*", "ion*" (asterisk supported)
                   - Field-specific: "documentation:temperature", "units:eV", "name:core_profiles"
                   - Fuzzy search: "temperatur~" (with tilde for fuzzy matching)
            ids_name: Optional specific IDS to search within
            max_results: Maximum number of results to return
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relevance-ordered search results, physics mappings, and AI suggestions

        Examples:
            Basic search: search_imas("plasma temperature")
            Bulk search: search_imas(["plasma", "temperature", "density"])
            Boolean: search_imas("plasma AND (temperature OR density)")
            Wildcards: search_imas("temp* OR ion*")
            Field search: search_imas("documentation:electron units:eV")
            Fuzzy search: search_imas("temperatur~ densty~")
        """
        try:
            # Create search request
            request = SearchRequest(
                query=query, ids_name=ids_name, max_results=max_results
            )

            # Auto-detect query features and route to appropriate handler
            if isinstance(query, list):
                # Handle bulk search
                response = self.search_handler.handle_bulk_search(request)
            else:
                # Analyze query features for single query
                features = self.search_router.detect_query_features(query)

                # Route to appropriate search strategy
                if features.is_exact_path:
                    response = self.search_handler.handle_exact_path_search(request)
                elif features.has_boolean:
                    response = self.search_handler.handle_boolean_search(request)
                elif features.has_wildcards:
                    response = self.search_handler.handle_wildcard_search(request)
                elif features.has_field_prefix:
                    response = self.search_handler.handle_field_search(request)
                else:
                    response = self.search_handler.handle_search(request)

            # Convert SearchResult objects to dictionaries for JSON serialization
            results_dict = [
                {
                    "path": result.path,
                    "score": result.score,
                    "documentation": result.documentation,
                    "units": result.units,
                    "ids_name": result.ids_name,
                    "highlights": getattr(result, "highlights", ""),
                }
                for result in response.results
            ]

            # Build response with AI enhancement
            result = {
                "results": results_dict,
                "total_results": response.total_results,
                "search_strategy": response.search_strategy,
                "suggestions": response.suggestions or [],
            }

            # Try physics search enhancement
            try:
                from .physics_integration import physics_search

                physics_result = physics_search(
                    str(query) if isinstance(query, list) else query
                )

                if physics_result.get("physics_matches"):
                    result["physics_matches"] = physics_result["physics_matches"][:3]
                    result["concept_suggestions"] = physics_result.get(
                        "concept_suggestions", []
                    )[:3]
                    result["unit_suggestions"] = physics_result.get(
                        "unit_suggestions", []
                    )[:3]
                    result["symbol_suggestions"] = physics_result.get(
                        "symbol_suggestions", []
                    )[:3]
            except Exception:
                pass  # Physics enhancement is optional

            # AI enhancement if MCP context available - handled by decorator
            if ctx and result["results"]:
                result["ai_prompt"] = (
                    f"Query: {query}\nResults: {result['results'][:3]}\n\nProvide analysis as JSON with 'insights', 'related_terms', and 'suggestions' fields."
                )

            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "results": [],
                "total_results": 0,
                "search_strategy": "error",
                "error": str(e),
                "suggestions": [
                    "Check spelling and try alternative terms",
                    "Use simpler search terms",
                    "Try wildcard patterns like 'term*'",
                ],
            }

    @ai_enhancer(
        EXPLANATION_EXPERT, "Concept explanation", temperature=0.2, max_tokens=800
    )
    async def explain_concept(
        self,
        concept: str,
        detail_level: str = "intermediate",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Explain IMAS concepts with physics context.

        Provides clear explanations of plasma physics concepts as they relate
        to the IMAS data dictionary, enhanced with AI insights.

        Enhanced with comprehensive physics explanations, IMAS mappings, and domain context.

        Args:
            concept: The concept to explain (physics concept, IMAS path, or general term)
            detail_level: Level of detail (basic, intermediate, advanced)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with explanation, physics context, IMAS mappings, and related information
        """
        try:
            # Search for concept in IMAS data
            search_results = await self.search_imas(concept, max_results=5)

            # Try physics search enhancement
            physics_context = {}
            try:
                from .physics_integration import physics_search

                physics_result = physics_search(concept)
                physics_context = {
                    "physics_matches": physics_result.get("physics_matches", [])[:3],
                    "concept_suggestions": physics_result.get(
                        "concept_suggestions", []
                    )[:3],
                    "symbols": physics_result.get("symbol_suggestions", [])[:3],
                    "units": physics_result.get("unit_suggestions", [])[:3],
                }
            except Exception:
                pass  # Physics enhancement is optional

            # Build base response
            result = {
                "concept": concept,
                "detail_level": detail_level,
                "related_paths": [r["path"] for r in search_results["results"][:5]],
                "physics_context": physics_context,
                "search_results_count": len(search_results["results"]),
            }

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                result["ai_prompt"] = f"""Concept: {concept}
Detail Level: {detail_level}
Related IMAS paths: {[r["path"] for r in search_results["results"][:3]]}
Physics context: {physics_context}

Provide a comprehensive explanation including:
1. Physics definition and significance
2. How it relates to plasma physics measurements
3. IMAS data structure implications
4. Practical measurement considerations
5. Related concepts to explore

Format as JSON with 'explanation', 'physics_significance', 'imas_applications', 'related_concepts' fields."""

            return result

        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return {
                "concept": concept,
                "detail_level": detail_level,
                "error": str(e),
                "explanation": "Failed to generate explanation",
                "suggestions": [
                    "Try a more specific concept name",
                    "Check spelling",
                    "Use physics terminology",
                ],
            }

    @ai_enhancer(OVERVIEW_EXPERT, "Overview analysis", temperature=0.3, max_tokens=1000)
    async def get_overview(
        self,
        question: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Get IMAS overview or answer analytical questions with graph insights.

        Provides comprehensive overview of available IDS in the IMAS data dictionary
        or answers specific analytical questions about the data structure.

        Enhanced with physics domain analysis and query validation capabilities.

        Args:
            question: Optional specific question about the data dictionary
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with overview information, analytics, and optional domain-specific data
        """
        try:
            # Get basic statistics from data accessor
            overview_data = {
                "total_documents": len(self.lex_search),
                "available_ids": list(self.data_accessor.get_available_ids()),
                "index_name": self.lex_search.indexname,
            }

            # Add IDS statistics
            ids_stats = {}
            for ids_name in overview_data["available_ids"]:
                try:
                    ids_data = self.data_accessor.get_ids_detailed_data(ids_name)
                    if ids_data:
                        ids_stats[ids_name] = {
                            "path_count": len(ids_data.get("paths", {})),
                            "description": ids_data.get("description", "")[:100]
                            + "...",
                        }
                except Exception:
                    ids_stats[ids_name] = {
                        "path_count": 0,
                        "description": "Error loading",
                    }

            overview_data["ids_statistics"] = ids_stats

            # If specific question provided, search for relevant information
            if question:
                search_results = await self.search_imas(question, max_results=5)
                overview_data["question"] = question
                overview_data["question_results"] = search_results["results"]
                overview_data["search_strategy"] = search_results["search_strategy"]

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                overview_data["ai_prompt"] = f"""IMAS Data Dictionary Overview:
Total documents: {overview_data["total_documents"]}
Available IDS: {overview_data["available_ids"]}
Question: {question or "General overview"}

{f"Relevant search results: {overview_data.get('question_results', [])[:3]}" if overview_data.get("question_results") else ""}

Provide analysis including:
1. Overview of IMAS structure and purpose
2. Key IDS and their physics domains  
3. Data organization principles
4. Usage recommendations for researchers
{f"5. Specific answer to: {question}" if question else "5. Getting started guidance"}

Format as JSON with 'overview', 'key_concepts', 'recommendations', 'navigation_tips' fields."""

            return overview_data

        except Exception as e:
            logger.error(f"Overview generation failed: {e}")
            return {
                "question": question,
                "error": str(e),
                "overview": "Failed to generate overview",
                "basic_info": {
                    "total_documents": len(self.lex_search)
                    if hasattr(self, "lex_search")
                    else 0,
                    "status": "error",
                },
            }

    @ai_enhancer(
        STRUCTURE_EXPERT, "Structure analysis", temperature=0.2, max_tokens=800
    )
    async def analyze_ids_structure(
        self,
        ids_name: str,
        ctx: Optional[Context] = None,
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
            available_ids = self.data_accessor.get_available_ids()
            if ids_name not in available_ids:
                return {
                    "ids_name": ids_name,
                    "error": f"IDS '{ids_name}' not found",
                    "available_ids": available_ids[:10],  # Show first 10
                    "suggestions": [f"Try: {ids}" for ids in available_ids[:5]],
                }

            # Get detailed IDS data
            ids_data = self.data_accessor.get_ids_detailed_data(ids_name)
            ids_paths = self.data_accessor.get_ids_paths(ids_name)

            # Get graph statistics if available
            try:
                graph_stats = self.data_accessor.get_ids_graph_stats(ids_name)
            except Exception:
                graph_stats = {"error": "Graph statistics not available"}

            # Build structural analysis
            result = {
                "ids_name": ids_name,
                "total_paths": len(ids_paths),
                "description": ids_data.get("description", ""),
                "structure": {
                    "root_level_paths": len(
                        [p for p in ids_paths.keys() if "/" not in p.strip("/")]
                    ),
                    "max_depth": max(len(p.split("/")) for p in ids_paths.keys())
                    if ids_paths
                    else 0,
                    "graph_statistics": graph_stats,
                },
                "sample_paths": list(ids_paths.keys())[
                    :10
                ],  # First 10 paths as examples
            }

            # Analyze path patterns
            path_patterns = {}
            for path in ids_paths.keys():
                segments = path.split("/")
                if len(segments) > 1:
                    root = segments[0]
                    path_patterns[root] = path_patterns.get(root, 0) + 1

            result["path_patterns"] = dict(
                sorted(path_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                result["ai_prompt"] = f"""IDS Structure Analysis: {ids_name}
Total paths: {result["total_paths"]}
Max depth: {result["structure"]["max_depth"]}
Top path patterns: {list(result["path_patterns"].keys())[:5]}
Sample paths: {result["sample_paths"][:5]}
Description: {result["description"][:200]}

Provide structural analysis including:
1. Physics domain and measurement purpose
2. Data organization principles used
3. Key measurement paths and their significance  
4. Typical usage patterns for researchers
5. Related IDS for comprehensive analysis

Format as JSON with 'physics_domain', 'organization_analysis', 'key_paths', 'usage_patterns', 'related_ids' fields."""

            return result

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

    @ai_enhancer(
        RELATIONSHIP_EXPERT, "Relationship analysis", temperature=0.3, max_tokens=800
    )
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
            max_depth: Maximum depth of relationship traversal (1-3)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relationship network and AI insights
        """
        try:
            # Parse the path to extract IDS name
            if "/" in path:
                ids_name = path.split("/")[0]
                specific_path = path
            else:
                ids_name = path
                specific_path = None

            # Validate IDS exists
            available_ids = self.data_accessor.get_available_ids()
            if ids_name not in available_ids:
                return {
                    "path": path,
                    "error": f"IDS '{ids_name}' not found",
                    "available_ids": available_ids[:10],
                    "suggestions": [f"Try: {ids}" for ids in available_ids[:5]],
                }

            # Get relationship data
            try:
                structural_insights = self.data_accessor.get_structural_insights()
            except Exception as e:
                return {
                    "path": path,
                    "error": f"Failed to load relationship data: {e}",
                    "relationship_type": relationship_type,
                    "max_depth": max_depth,
                }

            # Find related paths through search
            search_results = []
            if specific_path:
                # Search for paths related to the specific path
                path_parts = specific_path.split("/")
                search_terms = (
                    path_parts[-2:] if len(path_parts) > 1 else path_parts
                )  # Last 2 segments
                for term in search_terms:
                    if len(term) > 2:  # Skip very short terms
                        results = await self.search_imas(term, max_results=5)
                        search_results.extend(results["results"])
            else:
                # Search for paths in the same IDS
                results = await self.search_imas(f"ids_name:{ids_name}", max_results=10)
                search_results = results["results"]

            # Remove duplicates and filter
            seen_paths = set()
            unique_results = []
            for result in search_results:
                if result["path"] not in seen_paths and result["path"] != path:
                    seen_paths.add(result["path"])
                    unique_results.append(result)
                if len(unique_results) >= 20:  # Limit to reasonable number
                    break

            # Build relationship analysis
            result = {
                "path": path,
                "relationship_type": relationship_type,
                "max_depth": max_depth,
                "ids_name": ids_name,
                "related_paths": unique_results[:10],
                "relationship_count": len(unique_results),
                "analysis": {
                    "same_ids_paths": len(
                        [r for r in unique_results if r["path"].startswith(ids_name)]
                    ),
                    "cross_ids_paths": len(
                        [
                            r
                            for r in unique_results
                            if not r["path"].startswith(ids_name)
                        ]
                    ),
                    "structural_insights": structural_insights.get(ids_name, {}),
                },
            }

            # Add physics-based relationships if available
            try:
                from .physics_integration import physics_search

                physics_result = physics_search(
                    path.split("/")[-1]
                )  # Search on last path segment
                if physics_result.get("physics_matches"):
                    result["physics_relationships"] = {
                        "concepts": physics_result.get("physics_matches", [])[:3],
                        "related_symbols": physics_result.get("symbol_suggestions", [])[
                            :3
                        ],
                        "related_units": physics_result.get("unit_suggestions", [])[:3],
                    }
            except Exception:
                pass  # Physics enhancement is optional

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                result["ai_prompt"] = f"""Relationship Analysis for: {path}
IDS: {ids_name}
Related paths found: {len(unique_results)}
Same IDS: {result["analysis"]["same_ids_paths"]}
Cross IDS: {result["analysis"]["cross_ids_paths"]}
Top related paths: {[r["path"] for r in unique_results[:5]]}

Provide relationship analysis including:
1. Physics-based connections and measurement dependencies
2. Data flow patterns and coupling mechanisms  
3. Experimental workflow relationships
4. Recommended exploration paths for comprehensive analysis
5. Critical measurement combinations

Format as JSON with 'physics_connections', 'data_dependencies', 'workflow_patterns', 'exploration_recommendations', 'measurement_combinations' fields."""

            return result

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

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
        """
        Run the AI-enhanced MCP server.

        Args:
            transport: Transport protocol to use
            host: Host to bind to (for sse and streamable-http transports)
            port: Port to bind to (for sse and streamable-http transports)
        """
        try:
            match transport:
                case "stdio":
                    self.mcp.run(transport="stdio")
                case "sse":
                    self.mcp.run(transport=transport, host=host, port=port)
                case "streamable-http":
                    self._run_http_with_health(host=host, port=port)
        except KeyboardInterrupt:
            logger.info("Stopping AI-enhanced MCP server...")

    def _run_http_with_health(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the MCP server with streamable-http transport and add health endpoint."""
        try:
            import uvicorn
            from fastapi import FastAPI
            from starlette.middleware.cors import CORSMiddleware
            from starlette.responses import JSONResponse

            app = FastAPI(title="IMAS MCP Server", version=self._get_version())

            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Add health endpoint
            @app.get("/health")
            async def health():
                return JSONResponse(
                    {"status": "healthy", "version": self._get_version()}
                )

            # Mount MCP app
            app.mount("/", self.mcp.create_app())

        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        uvicorn.run(app, host=host, port=port, log_level="info")

    def _get_version(self) -> str:
        """Get the package version."""
        try:
            return importlib.metadata.version("imas-mcp")
        except Exception:
            return "unknown"


def main():
    """Run the server with stdio transport."""
    server = Server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
    """
    Entry point for running the AI-enhanced server with specified transport.

    Args:
        transport: Transport protocol to use
        host: Host to bind to (for sse and streamable-http transports)
        port: Port to bind to (for sse and streamable-http transports)
    """
    server = Server()
    server.run(transport=transport, host=host, port=port)
