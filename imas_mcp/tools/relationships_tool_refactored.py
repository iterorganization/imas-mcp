"""
IMAS Relationships Tool.

Discovers and analyzes relationships between IMAS data paths using
semantic analysis and physics-based categorization.
"""

import importlib.resources
import json
import logging
from typing import Any

from fastmcp import Context

from imas_mcp.core.data_model import PhysicsContext
from imas_mcp.models.constants import RelationshipType
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import RelationshipsInput
from imas_mcp.models.result_models import RelationshipResult
from imas_mcp.physics_extraction.relationship_analysis import (
    RelationshipEngine,
    create_relationship_nodes,
)
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class RelationshipsTool(BaseTool):
    """
    Relationships catalog-based tool for IMAS data relationship discovery.

    Provides intelligent access to the relationships catalog (relationships.json)
    with semantic analysis, physics domain mapping, and strength-based
    scoring for relationship discovery.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with relationships catalog data loading and relationship engine."""
        super().__init__(*args, **kwargs)
        self._relationships_catalog = {}
        self._relationship_engine = None
        self._load_relationships_catalog()

    def _load_relationships_catalog(self):
        """Load the relationships catalog file and initialize relationship engine."""
        try:
            try:
                catalog_file = (
                    importlib.resources.files("imas_mcp.resources.schemas")
                    / "relationships.json"
                )
                with catalog_file.open("r", encoding="utf-8") as f:
                    self._relationships_catalog = json.load(f)

                # Initialize relationship engine
                if self._relationships_catalog:
                    self._relationship_engine = RelationshipEngine(
                        self._relationships_catalog
                    )
                    logger.info("Loaded relationships catalog with relationship engine")
                else:
                    logger.warning("Empty relationships catalog loaded")

            except FileNotFoundError:
                logger.warning(
                    "Relationships catalog (relationships.json) not found in resources/schemas/"
                )

        except Exception as e:
            logger.error(f"Failed to load relationships catalog: {e}")
            self._relationships_catalog = {}
            self._relationship_engine = None

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "explore_relationships"

    @mcp_tool(
        name="explore_relationships",
        description="Discover connections and cross-references between IMAS data paths using semantic analysis and physics domain mapping",
    )
    @validate_input(RelationshipsInput)
    @handle_errors
    @cache_results(
        cache_key_func=lambda path,
        relationship_type,
        max_depth: f"rel_{path}_{relationship_type.value}_{max_depth}"
    )
    async def explore_relationships(
        self,
        context: Context,
        path: str,
        relationship_type: RelationshipType = RelationshipType.ALL,
        max_depth: int = 2,
    ) -> RelationshipResult:
        """
        Discover relationships and connections for an IMAS data path.

        Args:
            path: IMAS data path to explore relationships for
            relationship_type: Type of relationships to discover (all, semantic, structural, physics, measurement)
            max_depth: Maximum depth for relationship traversal (default: 2)

        Returns:
            RelationshipResult containing discovered relationships with physics context
        """
        logger.info(
            f"Exploring relationships for path: {path} (type: {relationship_type.value}, depth: {max_depth})"
        )

        if not self._relationship_engine:
            logger.warning(
                "Relationship engine not available, falling back to basic catalog search"
            )
            return await self._basic_relationship_search(
                path, relationship_type, max_depth
            )

        try:
            # Use relationship engine for discovery
            relationships = self._relationship_engine.discover_relationships(
                path, relationship_type, max_depth
            )

            # Generate physics context
            physics_context = self._relationship_engine.generate_physics_context(
                path, relationships
            )

            # Create relationship nodes
            related_nodes = create_relationship_nodes(
                path, relationships, physics_context
            )

            # Generate AI response with insights
            ai_response = await self._generate_relationship_insights(
                path, relationships, physics_context
            )

            return RelationshipResult(
                path=path,
                connections=relationships,
                nodes=related_nodes,
                physics_domains=self._extract_physics_domains(relationships),
                total_relationships=sum(
                    len(rel_list) for rel_list in relationships.values()
                ),
                ai_response=ai_response,
            )

        except Exception as e:
            logger.error(f"Error in relationship discovery: {e}")
            raise ToolError(
                f"Failed to discover relationships for {path}: {str(e)}"
            ) from e

    async def _basic_relationship_search(
        self, path: str, relationship_type: RelationshipType, max_depth: int
    ) -> RelationshipResult:
        """Fallback basic relationship search when engine unavailable."""
        if not self._relationships_catalog:
            return RelationshipResult(
                path=path,
                connections={
                    "semantic": [],
                    "structural": [],
                    "physics": [],
                    "measurement": [],
                },
                nodes=[],
                physics_domains=[],
                total_relationships=0,
                ai_response={
                    "summary": "No relationships catalog available",
                    "insights": [],
                    "physics_analysis": "Physics analysis unavailable without relationship engine",
                },
            )

        # Basic catalog search
        cross_references = self._relationships_catalog.get("cross_references", {})
        relationships = {
            "semantic": [],
            "structural": [],
            "physics": [],
            "measurement": [],
        }

        if path in cross_references:
            for rel in cross_references[path].get("relationships", []):
                relationships["structural"].append(
                    {
                        "path": rel["path"],
                        "type": rel.get("type", "cross_reference"),
                        "strength": 0.7,
                        "distance": 1,
                        "source": "catalog",
                    }
                )

        return RelationshipResult(
            path=path,
            connections=relationships,
            nodes=[],
            physics_domains=[],
            total_relationships=len(relationships["structural"]),
            ai_response={
                "summary": f"Found {len(relationships['structural'])} basic relationships",
                "insights": ["Using basic catalog search - limited functionality"],
                "physics_analysis": "Physics analysis requires relationship engine",
            },
        )

    async def _generate_relationship_insights(
        self,
        path: str,
        relationships: dict[str, list[dict[str, Any]]],
        physics_context: PhysicsContext | None,
    ) -> dict[str, Any]:
        """Generate AI insights about discovered relationships."""
        total_relationships = sum(len(rel_list) for rel_list in relationships.values())

        insights = []
        if relationships["semantic"]:
            insights.append(
                f"Found {len(relationships['semantic'])} semantic relationships based on physics concepts"
            )
        if relationships["physics"]:
            insights.append(
                f"Identified {len(relationships['physics'])} physics domain connections"
            )
        if relationships["structural"]:
            insights.append(
                f"Discovered {len(relationships['structural'])} structural relationships from catalog"
            )
        if relationships["measurement"]:
            insights.append(
                f"Located {len(relationships['measurement'])} measurement chain relationships"
            )

        physics_analysis = "No physics context available"
        if physics_context:
            physics_analysis = (
                f"Primary physics domain: {physics_context.domain}. "
                f"Key phenomena: {', '.join(physics_context.phenomena[:3])}. "
                f"Related domains: {', '.join(physics_context.related_domains[:3])}"
            )

        return {
            "summary": f"Discovered {total_relationships} relationships across {len([k for k, v in relationships.items() if v])} relationship types",
            "insights": insights,
            "physics_analysis": physics_analysis,
            "relationship_strength_distribution": self._analyze_strength_distribution(
                relationships
            ),
        }

    def _analyze_strength_distribution(
        self, relationships: dict[str, list[dict[str, Any]]]
    ) -> dict[str, int]:
        """Analyze the distribution of relationship strengths."""
        distribution = {
            "very_strong": 0,
            "strong": 0,
            "moderate": 0,
            "weak": 0,
            "very_weak": 0,
        }

        for rel_list in relationships.values():
            for rel in rel_list:
                strength = rel.get("strength", 0)
                if strength >= 0.8:
                    distribution["very_strong"] += 1
                elif strength >= 0.6:
                    distribution["strong"] += 1
                elif strength >= 0.4:
                    distribution["moderate"] += 1
                elif strength >= 0.2:
                    distribution["weak"] += 1
                else:
                    distribution["very_weak"] += 1

        return distribution

    def _extract_physics_domains(
        self, relationships: dict[str, list[dict[str, Any]]]
    ) -> list[str]:
        """Extract unique physics domains from relationships."""
        domains = set()

        for rel_list in relationships.values():
            for rel in rel_list:
                if "physics_domain_source" in rel:
                    domains.add(rel["physics_domain_source"])
                if "physics_domain_target" in rel:
                    domains.add(rel["physics_domain_target"])

        return list(domains)

    def _find_related_paths(
        self, path: str, relationship_type: RelationshipType, max_depth: int
    ) -> list[dict]:
        """Find related paths from the catalog based on the input path."""
        if not self._relationships_catalog:
            return []

        cross_references = self._relationships_catalog.get("cross_references", {})
        physics_concepts = self._relationships_catalog.get("physics_concepts", {})

        related_paths = []

        # Direct cross-references
        if path in cross_references:
            for relationship in cross_references[path].get("relationships", []):
                related_paths.append(
                    {
                        "path": relationship["path"],
                        "type": relationship.get("type", "cross_reference"),
                        "strength": 0.8,
                        "source": "cross_reference",
                    }
                )

        # Physics concept relationships
        if path in physics_concepts:
            for concept_path in physics_concepts[path].get("relevant_paths", []):
                related_paths.append(
                    {
                        "path": concept_path,
                        "type": "physics_concept",
                        "strength": 0.6,
                        "source": "physics_concept",
                    }
                )

        return related_paths

    def get_tool_definitions(self) -> list:
        """Get the tool definitions for this tool."""
        return [
            {
                "name": "explore_relationships",
                "description": "Discover connections and cross-references between IMAS data paths using semantic analysis and physics domain mapping",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "IMAS data path to explore relationships for",
                        },
                        "relationship_type": {
                            "type": "string",
                            "enum": [
                                "all",
                                "semantic",
                                "structural",
                                "physics",
                                "measurement",
                            ],
                            "description": "Type of relationships to discover",
                            "default": "all",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth for relationship traversal",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["path"],
                },
            }
        ]
