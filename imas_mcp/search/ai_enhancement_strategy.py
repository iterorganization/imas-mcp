"""
Selective AI Enhancement Strategy for IMAS MCP Tools.

This module implements conditional AI enhancement to optimize performance by applying
AI insights only when they provide value, reducing unnecessary latency.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# AI enhancement strategy mapping
AI_ENHANCEMENT_STRATEGY = {
    "search_imas": "conditional",  # Only for complex queries or when requested
    "explain_concept": "always",  # Core value-add
    "get_overview": "always",  # Benefits from synthesis
    "analyze_ids_structure": "conditional",  # Only for complex IDS
    "explore_relationships": "conditional",  # Only for deep analysis
    "explore_identifiers": "never",  # Structured data doesn't need AI
    "export_ids_bulk": "conditional",  # Only for enhanced format
    "export_physics_domain": "conditional",  # Only for workflow guidance
}


def should_apply_ai_enhancement(
    func_name: str, args: tuple, kwargs: dict, ctx: Optional[Any] = None
) -> bool:
    """
    Determine if AI enhancement should be applied based on tool strategy and context.

    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        ctx: MCP context for AI enhancement

    Returns:
        Boolean indicating whether AI enhancement should be applied
    """
    # No AI enhancement if no context available
    if not ctx:
        return False

    # Get strategy for this tool
    strategy = AI_ENHANCEMENT_STRATEGY.get(func_name, "always")

    if strategy == "never":
        return False
    elif strategy == "always":
        return True
    elif strategy == "conditional":
        return _evaluate_conditional_enhancement(func_name, args, kwargs)

    return True  # Default to enhancement


def _evaluate_conditional_enhancement(
    func_name: str, args: tuple, kwargs: dict
) -> bool:
    """
    Evaluate conditional AI enhancement based on specific context.

    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        Boolean indicating whether AI enhancement should be applied
    """
    try:
        if func_name == "search_imas":
            return _should_enhance_search(args, kwargs)
        elif func_name == "analyze_ids_structure":
            return _should_enhance_structure_analysis(args, kwargs)
        elif func_name == "explore_relationships":
            return _should_enhance_relationships(args, kwargs)
        elif func_name == "export_ids_bulk":
            return _should_enhance_bulk_export(args, kwargs)
        elif func_name == "export_physics_domain":
            return _should_enhance_physics_domain(args, kwargs)

    except Exception as e:
        logger.warning(f"Error evaluating conditional enhancement for {func_name}: {e}")
        return True  # Default to enhancement on error

    return True  # Default to enhancement for unknown tools


def _should_enhance_search(args: tuple, kwargs: dict) -> bool:
    """Determine if search_imas should use AI enhancement."""
    # Check search mode - only enhance for comprehensive mode
    search_mode = kwargs.get("search_mode", "auto")
    if search_mode in ["comprehensive", "semantic"]:
        return True

    # Check if query is complex (multiple terms, boolean operators, etc.)
    query = args[1] if len(args) > 1 else kwargs.get("query", "")
    if isinstance(query, list) and len(query) > 2:
        return True
    if isinstance(query, str) and any(
        op in query.upper() for op in ["AND", "OR", "NOT"]
    ):
        return True
    if isinstance(query, str) and len(query.split()) > 3:
        return True

    # Check if max_results is high (indicates need for detailed analysis)
    max_results = kwargs.get("max_results", 10)
    if max_results > 15:
        return True

    return False


def _should_enhance_structure_analysis(args: tuple, kwargs: dict) -> bool:
    """Determine if analyze_ids_structure should use AI enhancement."""
    # This would need access to the IDS data to determine complexity
    # For now, use a simple heuristic based on IDS name patterns
    ids_name = args[1] if len(args) > 1 else kwargs.get("ids_name", "")

    # Complex IDS names that typically have more structure
    complex_ids_patterns = [
        "core_profiles",
        "equilibrium",
        "transport",
        "edge_profiles",
        "mhd",
        "disruption",
        "pellets",
        "wall",
        "ec_launchers",
    ]

    if any(pattern in ids_name.lower() for pattern in complex_ids_patterns):
        return True

    return False


def _should_enhance_relationships(args: tuple, kwargs: dict) -> bool:
    """Determine if explore_relationships should use AI enhancement."""
    # Check max_depth - enhance for deep analysis only
    max_depth = kwargs.get("max_depth", 2)
    if max_depth >= 3:
        return True

    # Check if path suggests complex physics domain
    path = args[1] if len(args) > 1 else kwargs.get("path", "")
    complex_path_patterns = [
        "profiles",
        "transport",
        "equilibrium",
        "mhd",
        "disruption",
        "heating",
        "current_drive",
        "edge",
        "pedestal",
    ]

    if any(pattern in path.lower() for pattern in complex_path_patterns):
        return True

    # Check relationship type - enhance for specific complex types only
    relationship_type = kwargs.get("relationship_type", "all")
    if relationship_type in ["physics", "measurement_dependencies"]:
        return True

    return False


def _should_enhance_bulk_export(args: tuple, kwargs: dict) -> bool:
    """Determine if export_ids_bulk should use AI enhancement."""
    # Check output format first - raw format always disables AI
    output_format = kwargs.get("output_format", "structured")
    if output_format == "raw":
        return False

    # Check output format - only enhance for enhanced format
    if output_format == "enhanced":
        return True

    # Check if multiple IDS are being exported (complex analysis)
    ids_list = args[1] if len(args) > 1 else kwargs.get("ids_list", [])
    if len(ids_list) > 3:  # Only 4+ IDS enable AI automatically
        return True

    # Check if relationships are included AND we have multiple IDS (needs AI for interpretation)
    include_relationships = kwargs.get("include_relationships", True)
    include_physics_context = kwargs.get("include_physics_context", True)
    if (
        include_relationships and include_physics_context and len(ids_list) > 2
    ):  # Need at least 3 IDS for relationship analysis
        return True

    return False


def _should_enhance_physics_domain(args: tuple, kwargs: dict) -> bool:
    """Determine if export_physics_domain should use AI enhancement."""
    # Check analysis depth - enhance for comprehensive analysis
    analysis_depth = kwargs.get("analysis_depth", "focused")
    if analysis_depth == "comprehensive":
        return True

    # Check if cross-domain analysis is requested
    include_cross_domain = kwargs.get("include_cross_domain", False)
    if include_cross_domain:
        return True

    # Check max_paths - enhance for large exports
    max_paths = kwargs.get("max_paths", 10)
    if max_paths > 20:
        return True

    return False


def suggest_follow_up_tools(
    results: Dict[str, Any], func_name: str
) -> List[Dict[str, str]]:
    """
    Suggest relevant follow-up tools based on current results.

    Args:
        results: Results from the current tool execution
        func_name: Name of the function that generated the results

    Returns:
        List of tool suggestions with sample calls
    """
    suggestions = []

    try:
        if func_name == "search_imas":
            suggestions.extend(_suggest_search_follow_ups(results))
        elif func_name == "explain_concept":
            suggestions.extend(_suggest_concept_follow_ups(results))
        elif func_name == "analyze_ids_structure":
            suggestions.extend(_suggest_structure_follow_ups(results))
        elif func_name == "explore_relationships":
            suggestions.extend(_suggest_relationship_follow_ups(results))
        elif func_name == "export_ids_bulk":
            suggestions.extend(_suggest_bulk_export_follow_ups(results))

    except Exception as e:
        logger.warning(f"Error generating tool suggestions for {func_name}: {e}")

    return suggestions


def _suggest_search_follow_ups(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest follow-up tools for search results."""
    suggestions = []

    # Physics concept suggestions
    if results.get("physics_matches"):
        suggestions.append(
            {
                "tool": "explain_concept",
                "reason": "Get detailed physics explanations for found concepts",
                "sample_call": f"explain_concept('{results['physics_matches'][0].get('concept', 'plasma')}')",
            }
        )

    # Relationship analysis suggestions
    if len(results.get("results", [])) > 1:
        unique_ids = set(r.get("ids_name", "") for r in results["results"])
        if len(unique_ids) > 1:
            first_path = results["results"][0].get("path", "")
            suggestions.append(
                {
                    "tool": "explore_relationships",
                    "reason": "Analyze relationships between found paths across IDS",
                    "sample_call": f"explore_relationships('{first_path}')",
                }
            )

    # Bulk export suggestions
    if len(results.get("results", [])) > 5:
        unique_ids = list(set(r.get("ids_name", "") for r in results["results"]))
        if len(unique_ids) > 1:
            suggestions.append(
                {
                    "tool": "export_ids_bulk",
                    "reason": f"Export bulk data for {len(unique_ids)} IDS with relationships",
                    "sample_call": f"export_ids_bulk({unique_ids[:3]})",
                }
            )

    # Structure analysis for single IDS with many results
    single_ids_results = {}
    for result in results.get("results", []):
        ids_name = result.get("ids_name", "")
        if ids_name:
            single_ids_results[ids_name] = single_ids_results.get(ids_name, 0) + 1

    for ids_name, count in single_ids_results.items():
        if count > 3:
            suggestions.append(
                {
                    "tool": "analyze_ids_structure",
                    "reason": f"Analyze structure of {ids_name} IDS with {count} matching paths",
                    "sample_call": f"analyze_ids_structure('{ids_name}')",
                }
            )
            break  # Only suggest one structure analysis

    return suggestions


def _suggest_concept_follow_ups(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest follow-up tools for concept explanation results."""
    suggestions = []

    # Search for related paths
    if results.get("related_paths"):
        suggestions.append(
            {
                "tool": "search_imas",
                "reason": "Search for more paths related to this concept",
                "sample_call": f"search_imas('{results.get('concept', 'plasma')} related')",
            }
        )

    # Relationship analysis if multiple paths found
    if len(results.get("related_paths", [])) > 1:
        first_path = results["related_paths"][0]
        suggestions.append(
            {
                "tool": "explore_relationships",
                "reason": "Explore relationships between concept-related paths",
                "sample_call": f"explore_relationships('{first_path}')",
            }
        )

    return suggestions


def _suggest_structure_follow_ups(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest follow-up tools for structure analysis results."""
    suggestions = []

    # Relationship exploration for complex structures
    if results.get("total_paths", 0) > 10:
        ids_name = results.get("ids_name", "")
        suggestions.append(
            {
                "tool": "explore_relationships",
                "reason": f"Explore relationships within complex {ids_name} structure",
                "sample_call": f"explore_relationships('{ids_name}')",
            }
        )

    # Search for specific path patterns
    if results.get("path_patterns"):
        top_pattern = list(results["path_patterns"].keys())[0]
        suggestions.append(
            {
                "tool": "search_imas",
                "reason": f"Search for more paths with pattern '{top_pattern}'",
                "sample_call": f"search_imas('{top_pattern}')",
            }
        )

    return suggestions


def _suggest_relationship_follow_ups(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest follow-up tools for relationship exploration results."""
    suggestions = []

    # Bulk export if multiple IDS involved
    if results.get("analysis", {}).get("cross_ids_paths", 0) > 0:
        related_ids = set()
        for path_info in results.get("related_paths", []):
            if path_info.get("ids_name"):
                related_ids.add(path_info["ids_name"])

        if len(related_ids) > 1:
            suggestions.append(
                {
                    "tool": "export_ids_bulk",
                    "reason": f"Export data for {len(related_ids)} related IDS",
                    "sample_call": f"export_ids_bulk({list(related_ids)[:3]})",
                }
            )

    # Concept explanation for physics relationships
    if results.get("physics_relationships"):
        concepts = results["physics_relationships"].get("concepts", [])
        if concepts:
            suggestions.append(
                {
                    "tool": "explain_concept",
                    "reason": "Get detailed explanation of related physics concepts",
                    "sample_call": f"explain_concept('{concepts[0].get('concept', 'plasma')}')",
                }
            )

    return suggestions


def _suggest_bulk_export_follow_ups(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest follow-up tools for bulk export results."""
    suggestions = []

    # Physics domain analysis for exported data
    physics_domains = set()
    for ids_data in results.get("ids_data", {}).values():
        if isinstance(ids_data, dict):
            domains = ids_data.get("physics_domains", [])
            physics_domains.update(domains)

    for domain in list(physics_domains)[:2]:  # Suggest up to 2 domains
        if domain:
            suggestions.append(
                {
                    "tool": "export_physics_domain",
                    "reason": f"Analyze {domain} physics domain in detail",
                    "sample_call": f"export_physics_domain('{domain}')",
                }
            )

    return suggestions
