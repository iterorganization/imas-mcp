"""
Simplified MCP Server with AI-Enhanced IMAS Tools.

This module provides 3 focused tools for the IMAS data dictionary with better LLM usage.
"""

import json
from typing import Any, Dict, Optional, cast

from fastmcp import Context, FastMCP
from mcp.types import TextContent

from .json_data_accessor import JsonDataDictionaryAccessor

# Simplified AI prompts focused on specific tasks
SEARCH_EXPERT = """You are an IMAS search expert. Analyze relevance-ranked search results and provide:
1. 5 related search terms for plasma physics research
2. Brief physics insights about the found data paths  
3. Suggestions for complementary searches based on measurement relationships

The search results are ordered by relevance considering exact matches, path position, 
documentation content, and path specificity. Focus on practical physics relationships 
and measurement considerations that would lead to valuable follow-up searches."""

EXPLANATION_EXPERT = """You are a plasma physics expert. Explain IMAS concepts clearly with:
1. Physics significance and context
2. How data paths relate to measurements and modeling
3. Related concepts researchers should explore
4. Cross-domain connections

Focus on clarity and actionable guidance."""

OVERVIEW_EXPERT = """You are an IMAS analytics expert. Provide insights about:
1. Data structure patterns and organization
2. Which IDS are most important for specific research areas
3. Statistical insights about data distribution
4. Recommendations for exploration

Focus on quantitative insights with physics context."""

# Initialize data accessor
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
        path: str, documentation: str, query_terms: list
    ) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        path_lower = path.lower()
        doc_lower = documentation.lower() if documentation else ""

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

        # Bonus for shorter paths (more specific)
        path_depth = len(path.split("/"))
        if path_depth <= 3:
            score += 1.0
        elif path_depth >= 6:
            score -= 0.5

        return score

    # Prepare query terms
    query_terms = [term.strip() for term in query.lower().split() if term.strip()]

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

    # Build results with relevance scoring
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
            result["path"], documentation, query_terms
        )

        scored_results.append(
            {
                "ids_name": result["ids_name"],
                "path": result["path"],
                "documentation": documentation,
                "units": units,
                "relevance_score": relevance_score,
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

    # Basic explanation without AI
    basic_explanation = {
        "concept": concept,
        "explanation": f"'{concept}' is a concept in plasma physics with {len(related_paths)} related data paths in IMAS.",
        "related_paths": related_paths,
        "physics_context": "This concept relates to plasma physics modeling and measurements.",
        "suggested_searches": [],
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

    # Sort by complexity or size
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

    analysis = {
        "ids_name": ids_name,
        "total_paths": len(paths),
        "graph_metrics": ids_graph_stats,
        "complexity_score": complexity_scores.get(ids_name, 0),
        "navigation_complexity": "unknown",
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
    Create and configure the simplified FastMCP server with IMAS tools.

    Returns:
        Configured FastMCP server instance with AI-enhanced tools including graph analysis
    """
    app = FastMCP("AI-Enhanced IMAS Data Dictionary MCP Server")

    # Register the enhanced tools including graph analysis
    app.tool(search_imas)
    app.tool(explain_concept)
    app.tool(get_overview)
    app.tool(analyze_ids_structure)

    return app


def main():
    """Main entry point for running the server."""
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
