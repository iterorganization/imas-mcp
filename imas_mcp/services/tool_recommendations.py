"""Tool recommendation service for intelligent workflow guidance."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

from imas_mcp.models.suggestion_models import ToolSuggestion
from imas_mcp.models.result_models import SearchResult
from .base import BaseService


class RecommendationStrategy(Enum):
    """Tool recommendation strategy options."""

    SEARCH_BASED = "search_based"
    CONCEPT_BASED = "concept_based"
    ANALYSIS_BASED = "analysis_based"
    EXPORT_BASED = "export_based"
    OVERVIEW_BASED = "overview_based"
    RELATIONSHIPS_BASED = "relationships_based"
    IDENTIFIERS_BASED = "identifiers_based"


class ToolRecommendationService(BaseService):
    """Service for generating intelligent tool recommendations."""

    def apply_recommendations(
        self,
        result: BaseModel,
        strategy: RecommendationStrategy = RecommendationStrategy.SEARCH_BASED,
        max_tools: int = 4,
        query: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        """
        Generate and attach tool recommendations to appropriate result types.

        Args:
            result: The tool result to analyze and modify
            strategy: The recommendation strategy to use
            max_tools: Maximum number of recommendations
            query: Original query context
            **kwargs: Additional parameters

        Returns:
            Original result with recommendations attached (if applicable)
        """
        # Only apply recommendations to SearchResult instances
        if not isinstance(result, SearchResult):
            return result

        try:
            recommendations = self.generate_recommendations(
                result, strategy, max_tools, query, **kwargs
            )

            if recommendations:
                tool_hints = [
                    ToolSuggestion(
                        tool_name=rec["tool"],
                        description=rec.get("description", ""),
                        relevance=rec.get("reason", ""),
                    )
                    for rec in recommendations
                ]
                result.tool_hints = tool_hints

        except Exception as e:
            self.logger.warning(f"Failed to apply recommendations: {e}")

        return result

    def generate_recommendations(
        self,
        result: BaseModel,
        strategy: RecommendationStrategy = RecommendationStrategy.SEARCH_BASED,
        max_tools: int = 4,
        query: Optional[str] = None,
        context: Optional[dict] = None,
        **kwargs,
    ) -> List[dict]:
        """
        Generate tool recommendations based on result analysis.

        Replaces the functionality of @recommend_tools decorator.
        """
        try:
            if self._has_errors(result):
                return self._generate_error_recommendations()

            if strategy == RecommendationStrategy.SEARCH_BASED:
                return self._generate_search_recommendations(result, query, max_tools)
            elif strategy == RecommendationStrategy.CONCEPT_BASED:
                return self._generate_concept_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.ANALYSIS_BASED:
                return self._generate_analysis_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.EXPORT_BASED:
                return self._generate_export_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.OVERVIEW_BASED:
                return self._generate_overview_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.RELATIONSHIPS_BASED:
                return self._generate_relationships_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.IDENTIFIERS_BASED:
                return self._generate_identifiers_recommendations(result, max_tools)

            return self._generate_generic_recommendations(max_tools)

        except Exception as e:
            self.logger.warning(f"Tool recommendation generation failed: {e}")
            return self._generate_fallback_recommendations()

    def _generate_search_recommendations(
        self, result: BaseModel, query: Optional[str], max_tools: int
    ) -> List[dict]:
        """Generate recommendations for search results."""
        recommendations = []

        # Extract search context
        hits = getattr(result, "hits", [])
        hit_count = len(hits) if hits else 0

        if hit_count > 0:
            # Extract IDS names and domains from results
            ids_names = set()
            domains = set()

            for hit in hits[:5]:  # Analyze top 5 hits
                path = getattr(hit, "path", "")
                if path and "/" in path:
                    ids_names.add(path.split("/")[0])

                physics_domain = getattr(hit, "physics_domain", None)
                if physics_domain:
                    domains.add(physics_domain)

            # Suggest structure analysis for found IDS
            for ids_name in list(ids_names)[:2]:
                recommendations.append(
                    {
                        "tool": "analyze_ids_structure",
                        "reason": f"Analyze detailed structure of {ids_name} IDS",
                        "description": f"Get comprehensive structural analysis of {ids_name}",
                    }
                )

            # Suggest concept explanation for domains
            for domain in list(domains)[:2]:
                recommendations.append(
                    {
                        "tool": "explain_concept",
                        "reason": f"Learn more about {domain} physics domain",
                        "description": f"Get detailed explanation of {domain} concepts",
                    }
                )

            # Suggest relationship exploration
            if hit_count >= 3:
                recommendations.append(
                    {
                        "tool": "explore_relationships",
                        "reason": f"Explore data relationships for the {hit_count} found paths",
                        "description": "Discover how these data paths connect to other IMAS structures",
                    }
                )

            # Suggest export for large result sets
            if hit_count >= 5:
                recommendations.append(
                    {
                        "tool": "export_ids",
                        "reason": f"Export data for the {len(ids_names)} IDS found",
                        "description": "Export structured data for use in analysis workflows",
                    }
                )

        else:
            # No results - suggest broader exploration
            recommendations.extend(
                [
                    {
                        "tool": "get_overview",
                        "reason": "No results found - get overview of available data",
                        "description": "Explore IMAS data structure and available concepts",
                    },
                    {
                        "tool": "explore_identifiers",
                        "reason": "Search for related terms and identifiers",
                        "description": "Discover alternative search terms and data identifiers",
                    },
                ]
            )

            if query:
                recommendations.append(
                    {
                        "tool": "explain_concept",
                        "reason": f'Learn about "{query}" concept in fusion physics',
                        "description": "Get conceptual understanding and context",
                    }
                )

        return recommendations[:max_tools]

    def _generate_concept_recommendations(
        self, result: BaseModel, max_tools: int
    ) -> List[dict]:
        """Generate recommendations for concept explanation results."""
        concept = getattr(result, "concept", "physics concept")

        recommendations = [
            {
                "tool": "search_imas",
                "reason": f"Find data paths related to {concept}",
                "description": f"Search for IMAS data containing {concept} measurements",
            },
            {
                "tool": "explore_identifiers",
                "reason": f"Explore identifiers and terms related to {concept}",
                "description": "Discover related concepts and terminology",
            },
        ]

        # Add domain-specific suggestions
        concept_lower = str(concept).lower()

        if any(
            term in concept_lower for term in ["temperature", "density", "pressure"]
        ):
            recommendations.append(
                {
                    "tool": "search_imas",
                    "reason": "Explore core plasma profiles",
                    "description": "Search for core_profiles data containing plasma parameters",
                }
            )

        if any(term in concept_lower for term in ["magnetic", "field", "equilibrium"]):
            recommendations.append(
                {
                    "tool": "analyze_ids_structure",
                    "reason": "Analyze equilibrium IDS structure",
                    "description": "Examine magnetic equilibrium data organization",
                }
            )

        return recommendations[:max_tools]

    def _generate_analysis_recommendations(
        self, result: BaseModel, max_tools: int
    ) -> List[dict]:
        """Generate recommendations for analysis results."""
        return [
            {
                "tool": "export_ids",
                "reason": "Export analysis results for further processing",
                "description": "Save structured analysis data for workflows",
            },
            {
                "tool": "explore_relationships",
                "reason": "Explore relationships between analyzed components",
                "description": "Understand connections and dependencies",
            },
            {
                "tool": "search_imas",
                "reason": "Search for related data paths",
                "description": "Find relevant IMAS data",
            },
        ][:max_tools]

    def _generate_export_recommendations(
        self, result: BaseModel, max_tools: int
    ) -> List[dict]:
        """Generate recommendations for export results."""
        return [
            {
                "tool": "explore_relationships",
                "reason": "Understand data relationships in exported content",
                "description": "Analyze connections between exported IDS",
            }
        ][:max_tools]

    def _generate_overview_recommendations(
        self, result: BaseModel, max_tools: int
    ) -> List[dict]:
        """Generate recommendations for overview results."""
        return [
            {
                "tool": "search_imas",
                "reason": "Search for specific data based on overview",
                "description": "Find detailed data paths in areas of interest",
            }
        ][:max_tools]

    def _generate_relationships_recommendations(
        self, result: BaseModel, max_tools: int
    ) -> List[dict]:
        """Generate recommendations for relationship exploration results."""
        return [
            {
                "tool": "export_ids",
                "reason": "Export related IDS for analysis",
                "description": "Save relationship data for workflows",
            }
        ][:max_tools]

    def _generate_identifiers_recommendations(
        self, result: BaseModel, max_tools: int
    ) -> List[dict]:
        """Generate recommendations for identifier exploration results."""
        return [
            {
                "tool": "search_imas",
                "reason": "Search using discovered identifiers",
                "description": "Find data paths using the identified terms",
            }
        ][:max_tools]

    def _generate_generic_recommendations(self, max_tools: int) -> List[dict]:
        """Generate generic recommendations when strategy is unknown."""
        return [
            {
                "tool": "search_imas",
                "reason": "Search for specific data paths",
                "description": "Find relevant IMAS data",
            },
            {
                "tool": "get_overview",
                "reason": "Get overview of IMAS structure",
                "description": "Understand available data and capabilities",
            },
        ][:max_tools]

    def _generate_error_recommendations(self) -> List[dict]:
        """Generate recommendations for error cases."""
        return [
            {
                "tool": "get_overview",
                "reason": "Get overview of available data and functionality",
                "description": "Explore IMAS capabilities and data structure",
            }
        ]

    def _generate_fallback_recommendations(self) -> List[dict]:
        """Generate fallback recommendations when generation fails."""
        return [
            {
                "tool": "search_imas",
                "reason": "Search for specific data paths",
                "description": "Find relevant IMAS data for your research",
            },
            {
                "tool": "get_overview",
                "reason": "Get overview of IMAS structure",
                "description": "Understand available data and capabilities",
            },
        ]

    def _has_errors(self, result: BaseModel) -> bool:
        """Check if result contains errors."""
        if hasattr(result, "error") or (isinstance(result, dict) and "error" in result):
            return True
        return False
