"""
IMAS MCP Server with AI Tools.

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

This server replaces the legacy lexicographic server (now lexicographic_server.py)
with capabilities including graph analysis, physics context, and AI-powered
explanations.
"""

import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional, cast

import nest_asyncio
from fastmcp import Context, FastMCP
from mcp.types import TextContent

from .json_data_accessor import JsonDataDictionaryAccessor

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI prompts focused on specific tasks with relationship awareness
SEARCH_EXPERT = """You are an IMAS search expert with deep knowledge of data relationships. Analyze relevance-ranked search results and provide:
1. 5 related search terms for plasma physics research, considering cross-references and physics concepts
2. Brief physics insights about the found data paths and their measurement context
3. Suggestions for complementary searches based on measurement relationships and connected IDS

The search results are ordered by relevance considering exact matches, path position, 
documentation content, path specificity, physics concepts, and cross-reference connectivity. 
Focus on practical physics relationships and measurement considerations that would lead to 
valuable follow-up searches using the rich relationship network."""

EXPLANATION_EXPERT = """You are a plasma physics expert with access to IMAS relationship data. Explain concepts clearly with:
1. Physics significance and context from physics concepts database
2. How data paths relate to measurements, modeling, and cross-referenced paths
3. Related concepts researchers should explore via the relationship network
4. Cross-domain connections revealed by the physics concepts and unit families

Focus on clarity, actionable guidance, and leveraging the relationship data to show 
connections between different measurement systems and modeling approaches."""

OVERVIEW_EXPERT = """You are an IMAS analytics expert with comprehensive relationship insights. Provide insights about:
1. Data structure patterns, organization, and relationship connectivity (~90,000 total relationships)
2. Which IDS are most important for specific research areas based on cross-references
3. Statistical insights about data distribution, physics domains, and measurement relationships
4. Recommendations for exploration using the relationship network and connectivity patterns

Focus on quantitative insights with physics context, emphasizing how the relationship 
data reveals measurement interdependencies and research pathways."""


@dataclass
class Server:
    """AI IMAS MCP Server with structured tool management."""

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
                "IMAS JSON data is not available. This could be due to:\n"
                "1. Data dictionary not properly installed\n"
                "2. JSON files not built during installation\n"
                "Please reinstall the package or run the build process."
            )

        return accessor

    def _register_tools(self):
        """Register the 5 focused AI MCP tools with the server."""
        # Register the core tools - consolidated to 5 focused tools
        self.mcp.tool(self.search_imas)
        self.mcp.tool(self.explain_concept)
        self.mcp.tool(self.get_overview)
        self.mcp.tool(self.analyze_ids_structure)
        self.mcp.tool(self.explore_relationships)

    async def search_imas(
        self,
        query: str,
        ids_name: Optional[str] = None,
        max_results: int = 10,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Search for IMAS data paths with relevance-ordered results and AI enhancement.

        Advanced search tool that finds IMAS data paths, scores them by relevance,
        and optionally enhances results with AI insights when MCP sampling is available.

        Enhanced with physics context - can search by physics concepts, symbols, or units.

        Args:
            query: Search term, physics concept, symbol, or pattern
            ids_name: Optional specific IDS to search within
            max_results: Maximum number of results to return
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relevance-ordered search results, physics mappings, and AI suggestions
        """
        # Try physics search first for enhanced results
        try:
            from .physics_integration import physics_search

            physics_result = physics_search(query)

            # If physics search found results, enhance the standard search
            if physics_result.get("physics_matches"):
                standard_result = await search_imas(query, ids_name, max_results, ctx)

                # Merge physics context into standard results
                standard_result["physics_matches"] = physics_result["physics_matches"][
                    :3
                ]
                standard_result["concept_suggestions"] = physics_result.get(
                    "concept_suggestions", []
                )[:3]
                standard_result["unit_suggestions"] = physics_result.get(
                    "unit_suggestions", []
                )[:3]
                standard_result["symbol_suggestions"] = physics_result.get(
                    "symbol_suggestions", []
                )[:3]

                return standard_result

        except Exception:
            pass  # Fall back to standard search

        return await search_imas(query, ids_name, max_results, ctx)

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
        # Try physics-enhanced explanation first
        try:
            from .physics_integration import (
                explain_physics_concept,
                get_concept_imas_mapping,
            )

            physics_explanation = explain_physics_concept(concept, detail_level)

            # If physics explanation succeeded, merge with standard explanation
            if "error" not in physics_explanation:
                standard_result = await explain_concept(concept, detail_level, ctx)

                # Merge physics context into standard results
                standard_result["physics_explanation"] = physics_explanation
                standard_result["imas_mapping"] = get_concept_imas_mapping(concept)

                return standard_result

        except Exception:
            pass  # Fall back to standard explanation

        return await explain_concept(concept, detail_level, ctx)

    async def get_overview(
        self, question: Optional[str] = None, ctx: Optional[Context] = None
    ) -> Dict[str, Any]:
        """
        Get IMAS overview or answer analytical questions with graph insights.

        Provides comprehensive overview of available IDS in the IMAS data dictionary
        or answers specific analytical questions about the data structure.

        Enhanced with physics domain analysis and query validation capabilities.

        Args:
            question: Optional specific question about the data dictionary (can be a physics domain or validation query)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with overview information, analytics, and optional domain-specific data
        """
        # Handle domain-specific questions
        if question:
            try:
                from .core.domain_loader import get_domain_loader

                domain_loader = get_domain_loader()
                domain_characteristics = domain_loader.load_domain_characteristics()

                # Simple domain matching
                question_lower = question.lower()
                for domain_name, domain_info in domain_characteristics.items():
                    if domain_name.lower() in question_lower:
                        # Get basic overview and add domain info
                        basic_result = await get_overview(None, ctx)
                        basic_result["physics_domain_overview"] = {
                            "domain": domain_name,
                            "characteristics": domain_info,
                        }
                        basic_result["domain_specific_query"] = True
                        return basic_result

                # Check if question is asking for query validation
                if any(
                    word in question_lower
                    for word in ["validate", "check", "verify", "suggest"]
                ):
                    from .physics_integration import get_physics_integration

                    integration = get_physics_integration()
                    validation = integration.validate_physics_query(question)

                    standard_result = await get_overview(question, ctx)
                    standard_result["query_validation"] = validation
                    return standard_result

            except Exception:
                pass  # Fall back to standard overview

        return await get_overview(question, ctx)

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
        return await analyze_ids_structure(ids_name, ctx)

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
            relationship_type: Type of relationships to explore ("cross_references", "physics_concepts", "units", "all")
            max_depth: Maximum depth of relationship traversal (1-3)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relationship network and AI insights
        """
        if not data_accessor.is_available():
            return {
                "path": path,
                "relationships": [],
                "error": "Data not available - run build process",
            }

        # Get relationship data
        try:
            relationships = data_accessor.get_relationships()
            physics_concepts = relationships.get("physics_concepts", {})
            cross_refs = relationships.get("cross_references", {})
            unit_families = relationships.get("unit_families", {})
        except (FileNotFoundError, Exception):
            return {
                "path": path,
                "relationships": [],
                "error": "Relationship data not available",
            }

        def explore_path_relationships(
            start_path: str, current_depth: int = 0
        ) -> Dict[str, Any]:
            """Recursively explore relationships from a given path."""
            if current_depth >= max_depth:
                return {}

            path_relationships = {
                "path": start_path,
                "depth": current_depth,
                "physics_concepts": {},
                "cross_references": {},
                "unit_families": {},
                "connected_paths": [],
            }

            # Get physics concepts
            if (
                relationship_type in ["all", "physics_concepts"]
                and start_path in physics_concepts
            ):
                path_relationships["physics_concepts"] = physics_concepts[start_path]

            # Get cross-references
            if (
                relationship_type in ["all", "cross_references"]
                and start_path in cross_refs
            ):
                cross_ref_data = cross_refs[start_path]
                path_relationships["cross_references"] = cross_ref_data

                # Recursively explore connected paths
                related_paths = cross_ref_data.get("related_paths", [])[
                    :5
                ]  # Limit to 5 to avoid explosion
                for related_path in related_paths:
                    if current_depth < max_depth - 1:
                        connected = explore_path_relationships(
                            related_path, current_depth + 1
                        )
                        if connected:
                            path_relationships["connected_paths"].append(connected)

            # Get unit families
            if relationship_type in ["all", "units"] and start_path in unit_families:
                path_relationships["unit_families"] = unit_families[start_path]

            return path_relationships

        # Start exploration
        exploration_result = explore_path_relationships(path)

        # Calculate network statistics
        all_paths_found = set()

        def collect_paths(node: Dict[str, Any]):
            if "path" in node:
                all_paths_found.add(node["path"])
            for connected in node.get("connected_paths", []):
                collect_paths(connected)

        collect_paths(exploration_result)

        # Find physics domains involved
        domains_involved = set()
        for path_key in all_paths_found:
            if path_key in physics_concepts:
                domain = physics_concepts[path_key].get("domain", "")
                if domain:
                    domains_involved.add(domain)

        result = {
            "starting_path": path,
            "relationship_type": relationship_type,
            "max_depth": max_depth,
            "exploration_tree": exploration_result,
            "network_statistics": {
                "total_paths_discovered": len(all_paths_found),
                "physics_domains_involved": list(domains_involved),
                "relationship_density": len(
                    [p for p in all_paths_found if p in cross_refs]
                ),
                "physics_coverage": len(
                    [p for p in all_paths_found if p in physics_concepts]
                ),
            },
            "all_paths_in_network": list(all_paths_found)[:20],  # Limit output size
        }

        # AI enhancement if context available
        if ctx:
            try:
                ai_prompt = f"""
                Analyze this relationship network starting from IMAS path "{path}":
                
                Network discovered:
                - Total paths: {len(all_paths_found)}
                - Physics domains: {list(domains_involved)}
                - Relationship density: {result["network_statistics"]["relationship_density"]} connected paths
                - Physics coverage: {result["network_statistics"]["physics_coverage"]} paths with physics concepts
                
                Key relationships found:
                {exploration_result.get("cross_references", {}).get("relationship_type", "N/A")}
                
                Provide insights about:
                1. What this relationship network reveals about plasma physics connections
                2. Most important paths to explore for understanding this measurement/concept
                3. How these relationships help with data analysis workflows
                4. Complementary measurements or modeling approaches
                
                Return as JSON: {{
                    "physics_insights": "explanation of physics connections",
                    "key_pathways": ["path1", "path2", "path3"],
                    "workflow_suggestions": ["suggestion1", "suggestion2"],
                    "measurement_context": "how these relate to actual measurements"
                }}
                """

                ai_response = await ctx.sample(
                    ai_prompt,
                    system_prompt=SEARCH_EXPERT,
                    temperature=0.3,
                    max_tokens=500,
                )

                if ai_response:
                    text_content = cast(TextContent, ai_response)
                    ai_data = json.loads(text_content.text)
                    result["ai_insights"] = ai_data
            except Exception:
                pass

        return result

    def run(
        self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000
    ) -> None:
        """Run the AI-enhanced MCP server.

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
        """Run the MCP server with streamable-http transport and add health endpoint.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            # Get the FastMCP ASGI app
            app = self.mcp.http_app()

            # Add health endpoint using Starlette routing
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        async def health_endpoint(request):
            """Health check endpoint that verifies the data accessor is working."""
            try:
                # Get overview for essential metrics
                overview = await self.get_overview(ctx=None)
                metadata = overview.get("metadata", {})
                structural_overview = overview.get("structural_overview", {})

                return JSONResponse(
                    {
                        "status": "healthy",
                        "service": "imas-mcp",
                        "version": self._get_version(),
                        "creation_date": metadata.get("generation_date", "unknown"),
                        "dd_version": metadata.get("version", "unknown"),
                        "total_ids": overview.get("total_ids", 0),
                        "total_data_nodes": structural_overview.get(
                            "total_nodes_all_ids", 0
                        ),
                        "transport": "streamable-http",
                        "available_tools": [
                            "search_imas",
                            "explain_concept",
                            "get_overview",
                            "analyze_ids_structure",
                            "explore_relationships",
                        ],
                    }
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    {
                        "status": "unhealthy",
                        "service": "imas-mcp",
                        "error": str(e),
                        "transport": "streamable-http",
                    },
                    status_code=503,
                )

        # Add the health route to the existing app
        health_route = Route("/health", health_endpoint, methods=["GET"])
        app.routes.append(health_route)

        logger.info(
            f"Starting AI-enhanced MCP server with health endpoint at http://{host}:{port}/health"
        )

        # Run with uvicorn
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        uvicorn.run(app, host=host, port=port, log_level="info")

    def _get_version(self) -> str:
        """Get the package version."""
        try:
            import importlib.metadata

            return importlib.metadata.version("imas-mcp")
        except Exception:
            return "unknown"


# Initialize data accessor for standalone functions (backward compatibility)
data_accessor = JsonDataDictionaryAccessor()


async def search_imas(
    query: str,
    ids_name: Optional[str] = None,
    max_results: int = 10,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Search for IMAS data paths with relevance-ordered results and AI enhancement.

    Advanced search tool that finds IMAS data paths, scores them by relevance,
    and optionally enhances results with AI insights when MCP sampling is available.

    Args:
        query: Search term or pattern
        ids_name: Optional specific IDS to search within
        max_results: Maximum number of results to return
        ctx: MCP context for AI enhancement

    Returns:
        Dictionary with relevance-ordered search results and AI suggestions
    """
    if not data_accessor.is_available():
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "error": "Data not available - run build process",
        }

    def calculate_relevance_score(
        path: str,
        documentation: str,
        query_terms: list,
        physics_concepts: Dict[str, Any],
        cross_refs: Dict[str, Any],
        result: Dict[str, Any],
    ) -> float:
        """Calculate relevance score for a search result with relationship data."""
        score = 0.0
        path_lower = path.lower()
        doc_lower = documentation.lower() if documentation else ""
        full_path = f"{result.get('ids_name', '')}/{path}"

        for term in query_terms:
            term_lower = term.lower()

            # Exact matches in path get highest score
            if term_lower == path_lower.split("/")[-1]:  # exact field name match
                score += 10.0
            elif term_lower in path_lower:
                # Weighted by position - earlier matches score higher
                position_weight = 1.0 / (path_lower.find(term_lower) + 1)
                score += 5.0 * position_weight

            # Documentation matches
            if term_lower in doc_lower:
                # Count occurrences in documentation
                occurrences = doc_lower.count(term_lower)
                score += 2.0 * occurrences

            # Partial matches in path components
            path_components = path_lower.split("/")
            for component in path_components:
                if term_lower in component and term_lower != component:
                    score += 1.0

            # Physics concepts boost - if this path has physics concepts related to query
            if full_path in physics_concepts:
                concepts = physics_concepts[full_path]
                concept_text = " ".join(
                    [
                        concepts.get("concept", ""),
                        concepts.get("domain", ""),
                        concepts.get("description", ""),
                    ]
                ).lower()
                if term_lower in concept_text:
                    score += 3.0  # Boost for physics concept relevance

            # Cross-reference boost - if this path references other related paths
            if full_path in cross_refs:
                cross_ref_data = cross_refs[full_path]
                ref_text = " ".join(
                    [
                        cross_ref_data.get("target_path", ""),
                        cross_ref_data.get("relationship_type", ""),
                        cross_ref_data.get("description", ""),
                    ]
                ).lower()
                if term_lower in ref_text:
                    score += 2.0  # Boost for cross-reference relevance

        # Connectivity bonus - highly connected paths are often more important
        if full_path in cross_refs:
            connection_count = len(cross_refs[full_path].get("related_paths", []))
            if connection_count > 5:
                score += 1.0  # Bonus for highly connected paths

        # Bonus for shorter paths (more specific)
        path_depth = len(path.split("/"))
        if path_depth <= 3:
            score += 1.0
        elif path_depth >= 6:
            score -= 0.5

        return score

    # Prepare query terms
    query_terms = [term.strip() for term in query.lower().split() if term.strip()]

    # Get relationship data for enhanced scoring
    try:
        relationships = data_accessor.get_relationships()
        physics_concepts = relationships.get("physics_concepts", {})
        cross_refs = relationships.get("cross_references", {})
    except (FileNotFoundError, Exception):
        physics_concepts = {}
        cross_refs = {}

    # Search logic with relevance scoring
    if ids_name:
        paths = data_accessor.get_ids_paths(ids_name)
        candidate_results = [
            {"ids_name": ids_name, "path": path}
            for path in paths.keys()
            if any(term in path.lower() for term in query_terms)
        ]
    else:
        # Use broader search then filter
        all_results = data_accessor.search_paths_by_pattern(query)
        candidate_results = all_results

    # Build results with enhanced relevance scoring
    scored_results = []
    for result in candidate_results:
        documentation = data_accessor.get_path_documentation(
            result["ids_name"], result["path"]
        )
        try:
            units = data_accessor.get_path_units(result["ids_name"], result["path"])
        except KeyError:
            units = ""

        relevance_score = calculate_relevance_score(
            result["path"],
            documentation,
            query_terms,
            physics_concepts,
            cross_refs,
            result,
        )

        # Add relationship information to results
        full_path = f"{result['ids_name']}/{result['path']}"
        related_paths = []
        physics_info = {}

        # Get cross-references for this path
        if full_path in cross_refs:
            cross_ref_data = cross_refs[full_path]
            related_paths = cross_ref_data.get("related_paths", [])[
                :3
            ]  # Top 3 related paths

        # Get physics concept information
        if full_path in physics_concepts:
            concepts = physics_concepts[full_path]
            physics_info = {
                "concept": concepts.get("concept", ""),
                "domain": concepts.get("domain", ""),
                "description": concepts.get("description", ""),
            }

        scored_results.append(
            {
                "ids_name": result["ids_name"],
                "path": result["path"],
                "documentation": documentation,
                "units": units,
                "relevance_score": relevance_score,
                "related_paths": related_paths,  # Cross-references
                "physics_info": physics_info,  # Physics concepts
            }
        )

    # Sort by relevance score (descending) and take top results
    scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = scored_results[:max_results]

    # Remove relevance_score from final output (internal use only)
    for result in results:
        result.pop("relevance_score", None)

    # AI enhancement if context available
    ai_suggestions = []
    if ctx:
        try:
            # Enhanced AI prompt with relevance context
            top_paths = [f"{r['ids_name']}.{r['path']}" for r in results[:3]]
            total_candidates = len(scored_results)

            ai_prompt = f"""
            Analyze this relevance-ranked IMAS search for "{query}":
            - Found {total_candidates} total matches, showing top {len(results)}
            - Top ranked paths: {top_paths}
            - Query terms: {query_terms}
            
            The results are ordered by relevance based on:
            1. Exact field name matches (highest score)
            2. Position of matches in path (earlier = better)
            3. Documentation content matches
            4. Path specificity (shorter paths preferred)
            
            Provide 5 related search terms that would find complementary IMAS data paths.
            Focus on physics relationships and measurement techniques.
            Return as JSON: {{"suggested_related": ["term1", "term2", "term3", "term4", "term5"]}}
            """

            ai_response = await ctx.sample(
                ai_prompt,
                system_prompt=SEARCH_EXPERT,
                temperature=0.3,
                max_tokens=300,
            )

            if ai_response:
                text_content = cast(TextContent, ai_response)
                ai_data = json.loads(text_content.text)
                ai_suggestions = ai_data.get("suggested_related", [])
        except Exception:
            pass

    return {
        "query": query,
        "results": results,
        "total_found": len(scored_results),  # Total matches before limiting
        "returned_count": len(results),  # Actual results returned
        "suggested_related": ai_suggestions,
        "search_info": {
            "query_terms": query_terms,
            "relevance_ranked": True,
            "max_results": max_results,
        },
    }


async def explain_concept(
    concept: str, detail_level: str = "intermediate", ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Explain IMAS concepts with physics context.

    Provides clear explanations of plasma physics concepts as they relate
    to the IMAS data dictionary, enhanced with AI insights.

    Args:
        concept: The concept to explain
        detail_level: Level of detail (basic, intermediate, advanced)
        ctx: MCP context for AI enhancement

    Returns:
        Dictionary with explanation and related information
    """
    if not data_accessor.is_available():
        return {
            "concept": concept,
            "explanation": "IMAS data not available - run build process first",
            "related_paths": [],
            "physics_context": "",
        }

    # Find related paths
    search_results = data_accessor.search_paths_by_pattern(concept)
    related_paths = [f"{r['ids_name']}.{r['path']}" for r in search_results[:10]]

    # Get relationship data for enhanced explanations
    try:
        relationships = data_accessor.get_relationships()
        physics_concepts = relationships.get("physics_concepts", {})
        cross_refs = relationships.get("cross_references", {})
        unit_families = relationships.get("unit_families", {})
    except (FileNotFoundError, Exception):
        physics_concepts = {}
        cross_refs = {}
        unit_families = {}

    # Find physics concepts related to this concept
    concept_matches = []
    cross_reference_info = []

    for path, concept_data in physics_concepts.items():
        concept_text = " ".join(
            [
                concept_data.get("concept", ""),
                concept_data.get("domain", ""),
                concept_data.get("description", ""),
            ]
        ).lower()

        if concept.lower() in concept_text:
            concept_matches.append(
                {
                    "path": path,
                    "concept": concept_data.get("concept", ""),
                    "domain": concept_data.get("domain", ""),
                    "description": concept_data.get("description", ""),
                }
            )

    # Find cross-references that mention this concept
    for path, cross_ref_data in cross_refs.items():
        if concept.lower() in path.lower():
            cross_reference_info.append(
                {
                    "path": path,
                    "related_paths": cross_ref_data.get("related_paths", [])[:3],
                    "relationship_type": cross_ref_data.get("relationship_type", ""),
                }
            )

    # Find relevant unit families
    related_units = []
    for unit_path, unit_data in unit_families.items():
        if (
            concept.lower() in unit_path.lower()
            or concept.lower() in str(unit_data).lower()
        ):
            related_units.append({"path": unit_path, "unit_info": unit_data})

    # Basic explanation without AI
    basic_explanation = {
        "concept": concept,
        "explanation": f"'{concept}' is a concept in plasma physics with {len(related_paths)} related data paths in IMAS.",
        "related_paths": related_paths,
        "physics_context": "This concept relates to plasma physics modeling and measurements.",
        "suggested_searches": [],
        "physics_concepts": concept_matches[:5],  # Top 5 physics concept matches
        "cross_references": cross_reference_info[
            :5
        ],  # Top 5 cross-reference connections
        "measurement_domains": list(
            set([c.get("domain", "") for c in concept_matches if c.get("domain")])
        )[:3],
        "related_units": related_units[:3],  # Related unit families
    }

    # AI enhancement if context available
    if ctx:
        try:
            ai_prompt = f"""
            Explain the plasma physics concept "{concept}" in the context of IMAS data structures.
            
            Detail level: {detail_level}
            Found {len(related_paths)} related data paths
            Sample paths: {related_paths[:3]}
            
            Provide:
            1. Clear explanation appropriate for {detail_level} level
            2. Physics context and significance
            3. How this relates to IMAS data organization
            4. 3 related concepts to explore
            5. 3 suggested follow-up searches
            
            Return as JSON:
            {{
                "explanation": "detailed explanation here",
                "physics_context": "physics significance and context",
                "related_concepts": ["concept1", "concept2", "concept3"],
                "suggested_searches": ["search1", "search2", "search3"]
            }}
            """

            ai_response = await ctx.sample(
                ai_prompt,
                system_prompt=EXPLANATION_EXPERT,
                temperature=0.2,
                max_tokens=800,
            )

            if ai_response:
                text_content = cast(TextContent, ai_response)
                ai_data = json.loads(text_content.text)

                basic_explanation.update(
                    {
                        "explanation": ai_data.get(
                            "explanation", basic_explanation["explanation"]
                        ),
                        "physics_context": ai_data.get(
                            "physics_context", basic_explanation["physics_context"]
                        ),
                        "related_concepts": ai_data.get("related_concepts", []),
                        "suggested_searches": ai_data.get("suggested_searches", []),
                    }
                )
        except Exception:
            pass

    return basic_explanation


async def get_overview(
    question: Optional[str] = None, ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Get IMAS overview or answer analytical questions with graph insights.

    Provides comprehensive overview of available IDS in the IMAS data dictionary
    or answers specific analytical questions about the data structure.

    Args:
        question: Optional specific question about the data dictionary
        ctx: MCP context for AI enhancement

    Returns:
        Dictionary with overview information and analytics
    """
    if not data_accessor.is_available():
        return {
            "status": "error",
            "message": "IMAS data not available - run build process first",
        }

    available_ids = data_accessor.get_available_ids()
    catalog = data_accessor.get_catalog()
    metadata = catalog.get("metadata", {})

    # Get graph statistics
    graph_stats = data_accessor.get_graph_statistics()
    structural_insights = data_accessor.get_structural_insights()

    # Get relationship data for enhanced overview
    try:
        relationships = data_accessor.get_relationships()
        physics_concepts = relationships.get("physics_concepts", {})
        cross_refs = relationships.get("cross_references", {})
        unit_families = relationships.get("unit_families", {})
        rel_metadata = relationships.get("metadata", {})
    except (FileNotFoundError, Exception):
        physics_concepts = {}
        cross_refs = {}
        unit_families = {}
        rel_metadata = {}

    # Calculate relationship statistics
    total_relationships = rel_metadata.get("total_relationships", 0)
    total_physics_concepts = len(physics_concepts)
    total_cross_refs = len(cross_refs)
    total_unit_families = len(unit_families)

    # Find most connected paths
    connection_counts = {}
    for path, ref_data in cross_refs.items():
        connection_counts[path] = len(ref_data.get("related_paths", []))

    most_connected = sorted(
        connection_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]

    # Analyze physics domains
    domain_counts = {}
    for concept_data in physics_concepts.values():
        domain = concept_data.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Get basic IDS information enhanced with graph data
    ids_details = {}
    complexity_scores = structural_insights.get("complexity_rankings", {}).get(
        "complexity_scores", {}
    )

    for ids_name in available_ids:
        try:
            paths = data_accessor.get_ids_paths(ids_name)
            ids_stats = graph_stats.get(ids_name, {})
            ids_details[ids_name] = {
                "path_count": len(paths),
                "complexity_score": complexity_scores.get(ids_name, 0),
                "max_depth": ids_stats.get("hierarchy_metrics", {}).get("max_depth", 0),
                "branching_factor": ids_stats.get("branching_metrics", {}).get(
                    "avg_branching_factor_non_leaf", 0
                ),
            }
        except Exception:
            ids_details[ids_name] = {
                "path_count": 0,
                "complexity_score": 0,
                "max_depth": 0,
                "branching_factor": 0,
            }

    # Sort by complexity or size - THIS IS THE KEY RANKING FUNCTIONALITY
    largest_ids = sorted(
        ids_details.items(), key=lambda x: x[1]["path_count"], reverse=True
    )[:5]
    most_complex = sorted(
        ids_details.items(), key=lambda x: x[1]["complexity_score"], reverse=True
    )[:5]

    basic_overview = {
        "total_ids": len(available_ids),
        "ids_names": sorted(available_ids),
        "largest_ids": [
            {
                "name": name,
                "path_count": details["path_count"],
                "complexity_score": details["complexity_score"],
                "max_depth": details["max_depth"],
            }
            for name, details in largest_ids
        ],
        "most_complex_ids": [
            {
                "name": name,
                "complexity_score": details["complexity_score"],
                "path_count": details["path_count"],
                "branching_factor": details["branching_factor"],
            }
            for name, details in most_complex
        ],
        "structural_overview": {
            "total_nodes_all_ids": structural_insights.get("overview", {}).get(
                "total_nodes_all_ids", 0
            ),
            "avg_depth_across_ids": structural_insights.get("overview", {}).get(
                "avg_depth_across_ids", 0
            ),
            "complexity_range": structural_insights.get("overview", {}).get(
                "complexity_range", {}
            ),
            "deepest_ids": structural_insights.get("structural_patterns", {}).get(
                "deepest_ids", []
            )[:3],
            "most_branched": structural_insights.get("structural_patterns", {}).get(
                "most_branched", []
            )[:3],
        },
        "relationship_overview": {
            "total_relationships": total_relationships,
            "physics_concepts_count": total_physics_concepts,
            "cross_references_count": total_cross_refs,
            "unit_families_count": total_unit_families,
            "most_connected_paths": [
                {"path": path, "connection_count": count}
                for path, count in most_connected
            ],
            "physics_domains": dict(
                sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            ),
        },
        "metadata": metadata,
        "description": "IMAS provides standardized data structures for fusion plasma modeling.",
    }

    # Enhanced AI prompts with graph data
    if ctx:
        try:
            if question:
                ai_prompt = f"""
                Answer this question about the IMAS data dictionary: "{question}"
                
                Available context with graph analysis:
                - Total IDS: {len(available_ids)}
                - Total nodes across all IDS: {structural_insights.get("overview", {}).get("total_nodes_all_ids", 0)}
                - Most complex IDS: {[f"{name} (score: {details['complexity_score']})" for name, details in most_complex[:3]]}
                - Deepest hierarchies: {structural_insights.get("structural_patterns", {}).get("deepest_ids", [])[:3]}
                - Structural patterns: {structural_insights.get("structural_patterns", {})}
                
                Use graph metrics to provide insights about data relationships and navigation complexity.
                Return as JSON: {{"answer": "detailed answer", "insights": ["insight1", "insight2"], "navigation_tips": ["tip1", "tip2"]}}
                """
            else:
                ai_prompt = f"""
                Provide insights about this IMAS data dictionary using graph analysis:
                
                - Total IDS: {len(available_ids)} 
                - Average hierarchy depth: {structural_insights.get("overview", {}).get("avg_depth_across_ids", 0)}
                - Complexity range: {structural_insights.get("overview", {}).get("complexity_range", {})}
                - Most complex structures: {[name for name, _ in most_complex[:3]]}
                - Deepest hierarchies: {[name for name, _ in structural_insights.get("structural_patterns", {}).get("deepest_ids", [])[:3]]}
                
                Provide:
                1. Explanation of structural complexity patterns
                2. Most important IDS for different complexity preferences
                3. Navigation strategies based on graph metrics
                4. Insights about data organization principles
                
                Return as JSON:
                {{
                    "structural_explanation": "how IMAS is organized hierarchically",
                    "for_beginners": ["simple IDS to start with"],
                    "for_advanced": ["complex IDS for detailed analysis"], 
                    "navigation_strategies": ["strategy1", "strategy2"],
                    "organization_insights": ["insight1", "insight2"]
                }}
                """

            ai_response = await ctx.sample(
                ai_prompt,
                system_prompt=OVERVIEW_EXPERT,
                temperature=0.2,
                max_tokens=800,
            )

            if ai_response:
                text_content = cast(TextContent, ai_response)
                ai_data = json.loads(text_content.text)
                basic_overview["ai_insights"] = ai_data

                if question:
                    basic_overview["question"] = question
                    basic_overview["answer"] = ai_data.get(
                        "answer", "AI analysis not available"
                    )
        except Exception:
            pass

    return basic_overview


async def analyze_ids_structure(
    ids_name: str, ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Get detailed structural analysis of a specific IDS using graph metrics.

    Args:
        ids_name: Name of the IDS to analyze
        ctx: MCP context for AI enhancement

    Returns:
        Dictionary with detailed graph analysis and AI insights
    """
    if not data_accessor.is_available():
        return {"error": "Data not available"}

    if ids_name not in data_accessor.get_available_ids():
        return {"error": f"IDS '{ids_name}' not found"}

    # Get graph statistics for this IDS
    ids_graph_stats = data_accessor.get_ids_graph_stats(ids_name)
    paths = data_accessor.get_ids_paths(ids_name)
    structural_insights = data_accessor.get_structural_insights()
    complexity_scores = structural_insights.get("complexity_rankings", {}).get(
        "complexity_scores", {}
    )

    # Get relationship data for this specific IDS
    try:
        relationships = data_accessor.get_relationships()
        physics_concepts = relationships.get("physics_concepts", {})
        cross_refs = relationships.get("cross_references", {})
        unit_families = relationships.get("unit_families", {})
    except (FileNotFoundError, Exception):
        physics_concepts = {}
        cross_refs = {}
        unit_families = {}

    # Analyze relationships for this IDS
    ids_physics_concepts = {}
    ids_cross_refs = {}
    ids_unit_families = {}
    external_connections = []

    for path_key, concept_data in physics_concepts.items():
        if path_key.startswith(f"{ids_name}/"):
            ids_physics_concepts[path_key] = concept_data

    for path_key, ref_data in cross_refs.items():
        if path_key.startswith(f"{ids_name}/"):
            ids_cross_refs[path_key] = ref_data
            # Find external IDS connections
            related_paths = ref_data.get("related_paths", [])
            for related_path in related_paths:
                if not related_path.startswith(f"{ids_name}/"):
                    external_ids = (
                        related_path.split("/")[0]
                        if "/" in related_path
                        else related_path
                    )
                    if external_ids not in external_connections:
                        external_connections.append(external_ids)

    for path_key, unit_data in unit_families.items():
        if path_key.startswith(f"{ids_name}/"):
            ids_unit_families[path_key] = unit_data

    # Calculate relationship statistics for this IDS
    physics_domains_in_ids = {}
    for concept_data in ids_physics_concepts.values():
        domain = concept_data.get("domain", "unknown")
        physics_domains_in_ids[domain] = physics_domains_in_ids.get(domain, 0) + 1

    analysis = {
        "ids_name": ids_name,
        "total_paths": len(paths),
        "graph_metrics": ids_graph_stats,
        "complexity_score": complexity_scores.get(ids_name, 0),
        "navigation_complexity": "unknown",
        "relationship_analysis": {
            "physics_concepts_count": len(ids_physics_concepts),
            "cross_references_count": len(ids_cross_refs),
            "unit_families_count": len(ids_unit_families),
            "external_ids_connections": external_connections[
                :10
            ],  # Top 10 connected IDS
            "physics_domains": physics_domains_in_ids,
            "most_connected_paths": [
                {"path": path, "connections": len(ref_data.get("related_paths", []))}
                for path, ref_data in sorted(
                    ids_cross_refs.items(),
                    key=lambda x: len(x[1].get("related_paths", [])),
                    reverse=True,
                )[:5]
            ],
        },
    }

    # Determine navigation complexity
    if ids_graph_stats:
        max_depth = ids_graph_stats.get("hierarchy_metrics", {}).get("max_depth", 0)
        branching = ids_graph_stats.get("branching_metrics", {}).get(
            "avg_branching_factor_non_leaf", 0
        )

        if max_depth <= 3 and branching <= 2:
            analysis["navigation_complexity"] = "simple"
        elif max_depth <= 5 and branching <= 4:
            analysis["navigation_complexity"] = "moderate"
        else:
            analysis["navigation_complexity"] = "complex"

    # AI enhancement
    if ctx and ids_graph_stats:
        try:
            ai_prompt = f"""
            Analyze the structure of IMAS IDS "{ids_name}" using these graph metrics:
            
            Hierarchy: {ids_graph_stats.get("hierarchy_metrics", {})}
            Branching: {ids_graph_stats.get("branching_metrics", {})} 
            Complexity: {ids_graph_stats.get("complexity_indicators", {})}
            Key nodes: {ids_graph_stats.get("key_nodes", {})}
            
            Provide practical guidance for researchers:
            1. How to navigate this IDS effectively
            2. Which paths are most important to start with
            3. Complexity insights and potential challenges
            4. Related IDS that complement this one
            
            Return as JSON:
            {{
                "navigation_guide": "step-by-step navigation approach",
                "important_starting_paths": ["path1", "path2", "path3"],
                "complexity_insights": "what makes this IDS complex/simple",
                "complementary_ids": ["ids1", "ids2"]
            }}
            """

            ai_response = await ctx.sample(
                ai_prompt,
                system_prompt=EXPLANATION_EXPERT,
                temperature=0.2,
                max_tokens=600,
            )

            if ai_response:
                text_content = cast(TextContent, ai_response)
                ai_data = json.loads(text_content.text)
                analysis["ai_guidance"] = ai_data

        except Exception:
            pass

    return analysis


def create_server() -> FastMCP:
    """
    Create and configure the AI-enhanced FastMCP server with IMAS tools.

    Returns:
        Configured FastMCP server instance with AI-enhanced tools including graph analysis
    """
    # Create server instance and return the FastMCP app
    server = Server()
    return server.mcp


def main():
    """Main entry point for running the server."""
    server = Server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()


# Additional entry point for backward compatibility and testing
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
