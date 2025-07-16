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

This server uses cached properties for lazy initialization of DocumentStore and
SemanticSearch components for efficient and maintainable data access.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
from fastmcp import Context, FastMCP

from .search.document_store import DocumentStore
from .search.semantic_search import SemanticSearch, SemanticSearchConfig
from .search import (
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
    """IMAS MCP Server with cached properties for lazy initialization."""

    # Configuration parameters
    ids_set: Optional[set[str]] = None
    search_config: Optional[SemanticSearchConfig] = None

    # Internal fields
    mcp: FastMCP = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        if self.search_config is None:
            self.search_config = SemanticSearchConfig()
        self.mcp = FastMCP(name="imas-dd")
        self._register_tools()

    @cached_property
    def document_store(self) -> DocumentStore:
        """Lazily initialize and cache the document store."""
        if self.ids_set:
            # When ids_set is provided, use filtered initialization
            store = DocumentStore(ids_set=self.ids_set)
        else:
            # When no ids_set, load all documents
            store = DocumentStore()
        return store

    @cached_property
    def semantic_search(self) -> SemanticSearch:
        """Lazily initialize and cache the semantic search with document store."""
        # Ensure we have a valid config (should be set in __post_init__)
        config = self.search_config or SemanticSearchConfig()
        return SemanticSearch(config=config, document_store=self.document_store)

    def _register_tools(self):
        """Register the MCP tools with the server."""
        self.mcp.tool(self.search_imas)
        self.mcp.tool(self.explain_concept)
        self.mcp.tool(self.get_overview)
        self.mcp.tool(self.analyze_ids_structure)
        self.mcp.tool(self.explore_relationships)
        self.mcp.tool(self.explore_identifiers)

    def _extract_identifier_info(self, document) -> Dict[str, Any]:
        """Extract identifier schema information from a document."""
        identifier_schema = document.raw_data.get("identifier_schema")
        if identifier_schema and isinstance(identifier_schema, dict):
            options = identifier_schema.get("options", [])
            return {
                "has_identifier": True,
                "schema_path": identifier_schema.get("schema_path", "unknown"),
                "option_count": len(options),
                "branching_logic": len(options) > 1,
                "critical_node": len(options) > 5,
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
            # Use semantic search for enhanced physics-aware searching
            search_results = self.semantic_search.search(
                query=query if isinstance(query, str) else " ".join(query),
                top_k=max_results,
                filter_ids=[ids_name] if ids_name else None,
            )

            # Convert semantic search results to standard format
            results_dict = []
            for result in search_results:
                result_item = {
                    "path": result.document.metadata.path_name,
                    "score": result.similarity_score,
                    "documentation": result.document.documentation,
                    "units": result.document.units.unit_str
                    if result.document.units
                    else "",
                    "ids_name": result.document.metadata.ids_name,
                    "highlights": "",  # Can be enhanced later if needed
                }

                # Add identifier information if present - critical for branching logic
                result_item["identifier"] = self._extract_identifier_info(
                    result.document
                )
                results_dict.append(result_item)

            # Build response with enhanced semantic context
            result = {
                "results": results_dict,
                "total_results": len(results_dict),
                "search_strategy": "semantic_search",
                "suggestions": [],
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

            # Build base response with identifier awareness
            identifier_info = []
            for r in search_results["results"][:5]:
                path_name = r["path"]
                # Get the document to check for identifier schema
                all_docs = self.document_store.get_all_documents()
                matching_doc = next(
                    (doc for doc in all_docs if doc.metadata.path_name == path_name),
                    None,
                )

                if matching_doc:
                    identifier_schema = matching_doc.raw_data.get("identifier_schema")
                    if identifier_schema and isinstance(identifier_schema, dict):
                        options = identifier_schema.get("options", [])
                        identifier_info.append(
                            {
                                "path": path_name,
                                "schema_type": identifier_schema.get(
                                    "schema_path", "unknown"
                                ),
                                "branching_options": len(options),
                                "is_critical_node": len(options) > 5,
                            }
                        )

            result = {
                "concept": concept,
                "detail_level": detail_level,
                "related_paths": [r["path"] for r in search_results["results"][:5]],
                "physics_context": physics_context,
                "search_results_count": len(search_results["results"]),
                "identifier_context": {
                    "related_identifier_nodes": identifier_info,
                    "branching_significance": "Some related paths have enumerated options that define data structure logic"
                    if identifier_info
                    else "No related identifier schemas found",
                },
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
            # Get basic statistics from document store
            available_ids = self.document_store.get_available_ids()
            all_documents = self.document_store.get_all_documents()
            total_documents = len(all_documents)

            overview_data = {
                "total_documents": total_documents,
                "available_ids": available_ids,
                "index_name": "semantic_search_index",
            }

            # Get identifier information directly from the document store
            identifier_summary = self.document_store.get_identifier_branching_summary()

            # Add IDS statistics using document store data
            ids_stats = {}
            for ids_name in overview_data["available_ids"]:
                try:
                    ids_documents = self.document_store.get_documents_by_ids(ids_name)
                    if ids_documents:
                        # Get identifier count for this IDS from the identifier summary
                        ids_identifier_count = len(
                            identifier_summary["by_ids"].get(ids_name, [])
                        )

                        ids_stats[ids_name] = {
                            "path_count": len(ids_documents),
                            "identifier_count": ids_identifier_count,
                            "description": f"IDS containing {len(ids_documents)} data paths, {ids_identifier_count} with branching logic",
                        }
                except Exception:
                    ids_stats[ids_name] = {
                        "path_count": 0,
                        "identifier_count": 0,
                        "description": "Error loading",
                    }

            overview_data["ids_statistics"] = ids_stats
            overview_data["identifier_summary"] = {
                "total_identifiers": identifier_summary["total_identifier_paths"],
                "total_schemas": identifier_summary["total_schemas"],
                "enumeration_space": identifier_summary["total_enumeration_options"],
                "identifier_coverage": f"{identifier_summary['total_identifier_paths'] / total_documents * 100:.1f}%"
                if total_documents > 0
                else "0%",
                "significance": "These paths define critical branching logic in the data structure",
                "complexity_metrics": identifier_summary["complexity_metrics"],
                "by_physics_domain": identifier_summary["by_physics_domain"],
            }

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
                    "total_documents": len(self.document_store.get_all_documents()),
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

            result = {
                "ids_name": ids_name,
                "total_paths": len(paths),
                "description": f"IDS '{ids_name}' containing {len(paths)} data paths",
                "structure": {
                    "root_level_paths": len(
                        [p for p in paths if "/" not in p.strip("/")]
                    ),
                    "max_depth": max(len(p.split("/")) for p in paths) if paths else 0,
                    "document_count": len(ids_documents),
                },
                "identifier_analysis": {
                    "total_identifier_nodes": len(identifier_nodes),
                    "branching_paths": identifier_nodes,
                    "coverage": f"{len(identifier_nodes) / len(paths) * 100:.1f}%"
                    if paths
                    else "0%",
                    "significance": "These nodes define critical data structure branches and enumeration logic",
                },
                "sample_paths": paths[:10],  # First 10 paths as examples
            }

            # Analyze path patterns
            path_patterns = {}
            for path in paths:
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
Description: {result["description"]}

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

                search_results = self.semantic_search.search(
                    query=search_query,
                    top_k=20,  # Get more results for relationship analysis
                )
            except Exception as e:
                return {
                    "path": path,
                    "error": f"Failed to search relationships: {e}",
                    "relationship_type": relationship_type,
                    "max_depth": max_depth,
                }

            # Process search results for relationships with identifier awareness
            relationships = []
            seen_paths = set()
            identifier_relationships = []

            for result in search_results:
                result_path = result.document.metadata.path_name
                if result_path not in seen_paths and result_path != path:
                    seen_paths.add(result_path)

                    relationship_item = {
                        "path": result_path,
                        "score": result.similarity_score,
                        "relationship_type": "semantic_similarity",
                        "ids_name": result.document.metadata.ids_name,
                        "documentation": result.document.documentation[:200] + "..."
                        if len(result.document.documentation) > 200
                        else result.document.documentation,
                    }

                    # Check for identifier information in related paths
                    identifier_info = self._extract_identifier_info(result.document)
                    relationship_item["identifier"] = identifier_info

                    if identifier_info["has_identifier"]:
                        identifier_relationships.append(relationship_item)

                    relationships.append(relationship_item)

                    if len(relationships) >= max_depth * 5:  # Limit results
                        break

            # Build relationship analysis with identifier awareness
            result = {
                "path": path,
                "relationship_type": relationship_type,
                "max_depth": max_depth,
                "ids_name": ids_name,
                "related_paths": relationships[:10],
                "relationship_count": len(relationships),
                "analysis": {
                    "same_ids_paths": len(
                        [r for r in relationships if r["path"].startswith(ids_name)]
                    ),
                    "cross_ids_paths": len(
                        [r for r in relationships if not r["path"].startswith(ids_name)]
                    ),
                    "semantic_search_relationships": len(relationships),
                    "identifier_related_paths": len(identifier_relationships),
                },
                "identifier_context": {
                    "related_identifier_nodes": identifier_relationships[:5],
                    "significance": f"Found {len(identifier_relationships)} related paths with branching logic"
                    if identifier_relationships
                    else "No related identifier schemas in nearby paths",
                    "branching_implications": "These identifier nodes define critical data structure options"
                    if identifier_relationships
                    else None,
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
Related paths found: {len(relationships)}
Same IDS: {result["analysis"]["same_ids_paths"]}
Cross IDS: {result["analysis"]["cross_ids_paths"]}
Top related paths: {[r["path"] for r in relationships[:5]]}

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

    @ai_enhancer(
        SEARCH_EXPERT, "Identifier exploration", temperature=0.3, max_tokens=800
    )
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
            result = {}

            if scope in ["all", "summary"]:
                # Get overall identifier branching summary
                summary = self.document_store.get_identifier_branching_summary()
                result["summary"] = summary
                result["overview"] = {
                    "total_schemas": summary["total_schemas"],
                    "total_paths": summary["total_identifier_paths"],
                    "enumeration_space": summary["total_enumeration_options"],
                    "significance": "Identifier schemas define critical branching logic and enumeration options in IMAS data structures",
                }

            if scope in ["all", "schemas"]:
                # Get identifier schemas
                if query:
                    schemas = self.document_store.search_identifier_schemas(query)
                else:
                    schemas = self.document_store.get_identifier_schemas()

                result["schemas"] = []
                for schema_doc in schemas[:10]:  # Limit to 10 schemas
                    schema_data = schema_doc.raw_data
                    result["schemas"].append(
                        {
                            "name": schema_doc.metadata.path_name,
                            "description": schema_data.get("description", "")[:200],
                            "total_options": schema_data.get("total_options", 0),
                            "complexity": schema_data.get("branching_complexity", 0),
                            "usage_count": schema_data.get("usage_count", 0),
                            "physics_domains": schema_data.get("physics_domains", []),
                            "sample_options": schema_data.get("options", [])[
                                :5
                            ],  # First 5 options
                            "usage_paths": schema_data.get("usage_paths", [])[
                                :5
                            ],  # First 5 usage paths
                        }
                    )

            if scope in ["all", "paths"]:
                # Get identifier paths
                if query:
                    # Search within identifier path documents
                    search_results = await self.search_imas(
                        f"identifier {query}", max_results=20
                    )
                    identifier_paths = [
                        r
                        for r in search_results["results"]
                        if r.get("identifier", {}).get("has_identifier", False)
                    ]
                else:
                    # Get all identifier paths using document store
                    path_docs = self.document_store.get_identifier_paths()
                    identifier_paths = []
                    for doc in path_docs[:20]:  # Limit to 20 paths
                        if doc.metadata.data_type == "identifier_path":
                            identifier_paths.append(
                                {
                                    "path": doc.metadata.path_name,
                                    "ids_name": doc.metadata.ids_name,
                                    "description": doc.documentation[:200],
                                    "schema_name": doc.raw_data.get("schema_name", ""),
                                    "option_count": doc.raw_data.get("option_count", 0),
                                    "physics_domain": doc.metadata.physics_domain,
                                }
                            )

                result["identifier_paths"] = identifier_paths

            # Add query-specific results if query provided
            if query:
                result["query"] = query
                result["query_note"] = f"Results filtered for: {query}"

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                result["ai_prompt"] = f"""Identifier Exploration Request:
Query: {query or "General exploration"}
Scope: {scope}

Summary: {result.get("summary", {})}
Found schemas: {len(result.get("schemas", []))}
Found paths: {len(result.get("identifier_paths", []))}

Provide identifier analysis including:
1. Significance of identifier branching logic in IMAS
2. Key schemas and their enumeration options
3. Physics domain implications of identifier choices
4. Practical guidance for using identifier information
5. Critical branching points for data structure navigation

Format as JSON with 'significance', 'key_schemas', 'physics_implications', 'usage_guidance', 'critical_decisions' fields."""

            return result

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
