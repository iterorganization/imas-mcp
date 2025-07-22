"""
IMAS MCP Server with AI Tools.

This is the principal MCP server for the IMAS data dictionary, providing AI
tools for physics-based search, analysis, and exploration of plasma physics data.
It offers 8 focused tools with intelligent insights and relevance-ranked results
for better LLM usage.

The 8 core tools provide comprehensive coverage:
1. search_imas - Enhanced search with physics concepts, symbols, and units
2. explain_concept - Physics explanations with IMAS mappings and domain context
3. get_overview - General overview with domain analysis and query validation
4. analyze_ids_structure - Detailed structural analysis of specific IDS
5. explore_relationships - Advanced relationship exploration across the data dictionary
6. explore_identifiers - Identifier schema exploration and branching logic analysis
7. export_ids_bulk - Bulk export of multiple IDS with cross-relationships analysis
8. export_physics_domain - Physics domain-specific export with measurement dependencies

This server uses cached properties for lazy initialization of DocumentStore,
SemanticSearch, and IMASGraphAnalyzer components for efficient and maintainable data access.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
from fastmcp import Context, FastMCP

from imas_mcp.graph_analyzer import IMASGraphAnalyzer
from imas_mcp.search import (
    BULK_EXPORT_EXPERT,
    EXPLANATION_EXPERT,
    OVERVIEW_EXPERT,
    PHYSICS_DOMAIN_EXPERT,
    RELATIONSHIP_EXPERT,
    SEARCH_EXPERT,
    STRUCTURE_EXPERT,
    ai_enhancer,
    SearchCache,
    SearchComposer,
    SearchConfig,
    SearchMode,
    suggest_follow_up_tools,
)
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.semantic_search import SemanticSearch, SemanticSearchConfig

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
            self.search_config = SemanticSearchConfig(ids_set=self.ids_set)
        else:
            assert self.ids_set == self.search_config.ids_set, (
                "If search_config is provided, it must match the ids_set."
            )
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

    @cached_property
    def graph_analyzer(self) -> IMASGraphAnalyzer:
        """Lazily initialize and cache the graph analyzer."""
        return IMASGraphAnalyzer()

    @cached_property
    def search_composer(self) -> SearchComposer:
        """Lazily initialize and cache the search composer."""
        return SearchComposer(self.document_store)

    @cached_property
    def search_cache(self) -> SearchCache:
        """Lazily initialize and cache the search cache."""
        return SearchCache(maxsize=1000, ttl=3600)

    def _register_tools(self):
        """Register the MCP tools with the server."""
        self.mcp.tool(self.search_imas)
        self.mcp.tool(self.explain_concept)
        self.mcp.tool(self.get_overview)
        self.mcp.tool(self.analyze_ids_structure)
        self.mcp.tool(self.explore_relationships)
        self.mcp.tool(self.explore_identifiers)
        self.mcp.tool(self.export_ids_bulk)
        self.mcp.tool(self.export_physics_domain)

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
        search_mode: str = "auto",
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
            search_mode: Search mode - "auto", "semantic", "lexical", or "hybrid"
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
            Lexical search: search_imas("plasma temperature", search_mode="lexical")
        """
        try:
            # Check cache first for faster response
            cached_result = self.search_cache.get(
                query=query,
                ids_name=ids_name,
                max_results=max_results,
                search_mode=search_mode,
            )
            if cached_result is not None:
                return cached_result

            # Parse search mode
            try:
                mode = SearchMode(search_mode.lower())
            except ValueError:
                mode = SearchMode.AUTO

            # Configure search
            config = SearchConfig(
                mode=mode,
                max_results=max_results,
                filter_ids=[ids_name] if ids_name else None,
            )

            # Execute search using composer
            search_results = self.search_composer.search(query, config)

            # Convert results to standard format
            results_dict = []
            for result in search_results:
                result_item = result.to_dict()

                # Add identifier information if present - critical for branching logic
                result_item["identifier"] = self._extract_identifier_info(
                    result.document
                )
                results_dict.append(result_item)

            # Build response with enhanced context
            result = {
                "results": results_dict,
                "total_results": len(results_dict),
                "search_strategy": mode.value,
                "suggestions": [],
            }

            # Add tool suggestions based on results
            try:
                result["suggested_tools"] = suggest_follow_up_tools(
                    result, "search_imas"
                )
            except Exception as e:
                logger.warning(f"Failed to generate tool suggestions: {e}")
                result["suggested_tools"] = []

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

            # AI enhancement if MCP context available - handled by decorator with selective strategy
            # Add AI prompt for conditional enhancement - decorator will handle cases without context
            if result["results"]:
                # The selective AI enhancement strategy will determine if this should be processed
                result["ai_prompt"] = (
                    f"Query: {query}\nSearch Mode: {mode.value}\nResults: {result['results'][:3]}\n\n"
                    f"Provide analysis as JSON with 'insights', 'related_terms', and 'suggestions' fields."
                )

            # Cache successful results for future use
            self.search_cache.set(
                query=query,
                result=result,
                ids_name=ids_name,
                max_results=max_results,
                search_mode=search_mode,
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

                search_results = self.semantic_search.search(
                    query=search_query,
                    top_k=min(10, max_depth * 8),  # Reduced from 20, scale with depth
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

                    if (
                        len(relationships) >= max_depth * 3
                    ):  # Reduced from 5, stricter limit
                        break

            # Build relationship analysis with identifier awareness
            result = {
                "path": path,
                "relationship_type": relationship_type,
                "max_depth": max_depth,
                "ids_name": ids_name,
                "related_paths": relationships[:5],  # Reduced from 10 for performance
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

    @ai_enhancer(
        BULK_EXPORT_EXPERT, "Bulk export analysis", temperature=0.3, max_tokens=1000
    )
    async def export_ids_bulk(
        self,
        ids_list: List[str],
        include_relationships: bool = True,
        include_physics_context: bool = True,
        output_format: str = "structured",  # raw, structured, enhanced
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Export bulk IMAS data for multiple IDS with sophisticated relationship analysis.

        Advanced bulk export tool that extracts comprehensive data for multiple IDS,
        including cross-IDS relationships, physics context, and structural analysis
        using IMASGraphAnalyzer for sophisticated relationship embedding.

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
                            if output_format == "comprehensive"
                            else doc.documentation[:200],
                            "data_type": doc.metadata.data_type,
                            "physics_domain": doc.metadata.physics_domain,
                            "units": doc.metadata.units,
                        }

                        # Add detailed information for comprehensive format
                        if output_format == "comprehensive":
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
                    # Use graph analyzer for sophisticated relationship analysis
                    relationship_analysis = {}

                    for i, ids1 in enumerate(valid_ids):
                        for ids2 in valid_ids[i + 1 :]:
                            try:
                                # Analyze structural patterns between IDS
                                # Build the data structure expected by analyze_cross_ids_patterns
                                ids_data_for_analysis = {}
                                for ids in [ids1, ids2]:
                                    if ids in export_data["ids_data"] and isinstance(
                                        export_data["ids_data"][ids], dict
                                    ):
                                        ids_data_for_analysis[ids] = {
                                            path["path"]: path
                                            for path in export_data["ids_data"][
                                                ids
                                            ].get("paths", [])
                                        }

                                if len(ids_data_for_analysis) == 2:
                                    patterns = (
                                        self.graph_analyzer.analyze_cross_ids_patterns(
                                            ids_data_for_analysis
                                        )
                                    )
                                else:
                                    patterns = {
                                        "note": "Insufficient data for cross-IDS analysis"
                                    }

                                relationship_key = f"{ids1}_{ids2}"
                                relationship_analysis[relationship_key] = {
                                    "ids_pair": [ids1, ids2],
                                    "patterns": patterns,
                                    "common_physics_domains": list(
                                        set(
                                            export_data["ids_data"][ids1].get(
                                                "physics_domains", []
                                            )
                                        )
                                        & set(
                                            export_data["ids_data"][ids2].get(
                                                "physics_domains", []
                                            )
                                        )
                                    ),
                                    "complementary_measurements": list(
                                        set(
                                            export_data["ids_data"][ids1].get(
                                                "measurement_types", []
                                            )
                                        )
                                        | set(
                                            export_data["ids_data"][ids2].get(
                                                "measurement_types", []
                                            )
                                        )
                                    ),
                                }
                            except Exception as e:
                                logger.warning(
                                    f"Failed to analyze relationship {ids1}-{ids2}: {e}"
                                )
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
                        if domain:  # Skip empty domains
                            domain_search_results = self.semantic_search.search(
                                domain, top_k=5
                            )
                            domain_context[domain] = {
                                "related_paths": [
                                    r.document.metadata.path_name
                                    for r in domain_search_results[:3]
                                ],
                                "measurement_scope": f"Physics domain: {domain}",
                                "associated_ids": [
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

            # Generate export summary
            export_data["export_summary"] = {
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

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                export_data["ai_prompt"] = f"""Bulk Export Analysis:
IDS Requested: {ids_list}
Valid IDS: {valid_ids}
Export Format: {output_format}
Include Relationships: {include_relationships}
Include Physics Context: {include_physics_context}

Export Summary: {export_data["export_summary"]}
Physics Domains: {list(export_data.get("physics_domains", {}).keys())}

Provide bulk export guidance including:
1. Data usage recommendations for this specific IDS combination
2. Physics insights about relationships between exported IDS
3. Suggested analysis workflows utilizing the exported data
4. Integration patterns and measurement dependencies
5. Quality considerations and data validation approaches

Format as JSON with 'usage_recommendations', 'physics_insights', 'analysis_workflows', 'integration_patterns', 'quality_considerations' fields."""

            return export_data

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
                    "Try with output_format='minimal' for faster processing",
                ],
            }

    @ai_enhancer(
        PHYSICS_DOMAIN_EXPERT,
        "Physics domain analysis",
        temperature=0.3,
        max_tokens=1000,
    )
    async def export_physics_domain(
        self,
        domain: str,
        include_cross_domain: bool = False,  # Default to False to avoid performance issues
        analysis_depth: str = "focused",  # Default to focused instead of comprehensive
        max_paths: int = 10,  # Reduced from 20 to prevent excessive processing
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Export physics domain-specific data with sophisticated relationship analysis.

        Advanced domain export tool that extracts comprehensive data for a specific
        physics domain, including cross-domain relationships, measurement dependencies,
        and structural analysis using IMASGraphAnalyzer for domain-aware insights.

        Args:
            domain: Physics domain name to export (e.g., 'core_profiles', 'equilibrium', 'transport')
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
                    "error": "No physics domain specified",
                    "suggestions": [
                        "Provide a physics domain name",
                        "Use get_overview to see available domains",
                    ],
                }

            # Search for domain-related paths
            domain_search_results = self.semantic_search.search(domain, top_k=max_paths)

            if not domain_search_results:
                return {
                    "domain": domain,
                    "error": "No data found for specified domain",
                    "suggestions": [
                        "Try alternative domain names",
                        "Use search_imas to explore available domains",
                        "Check domain spelling",
                    ],
                }

            export_data = {
                "domain": domain,
                "analysis_depth": analysis_depth,
                "include_cross_domain": include_cross_domain,
                "max_paths": max_paths,
                "timestamp": "domain_export",
                "domain_data": {
                    "total_paths": len(domain_search_results),
                    "paths": [],
                    "associated_ids": set(),
                    "measurement_types": set(),
                    "units_distribution": {},
                    "lifecycle_stages": set(),
                },
                "domain_structure": {},
                "cross_domain_analysis": {},
                "measurement_dependencies": {},
                "export_summary": {},
            }

            # Process domain-specific paths
            units_count = {}
            for result in domain_search_results:
                path_info = {
                    "path": result.document.metadata.path_name,
                    "ids_name": result.ids_name,
                    "documentation": result.document.documentation
                    if analysis_depth == "comprehensive"
                    else result.document.documentation[:200],
                    "physics_domain": result.document.metadata.physics_domain,
                    "units": result.document.metadata.units,
                    "data_type": result.document.metadata.data_type,
                    "relevance_score": result.similarity_score,
                }

                # Add detailed analysis for comprehensive depth
                if analysis_depth == "comprehensive":
                    path_info["raw_data"] = result.document.raw_data
                    path_info["identifier_info"] = self._extract_identifier_info(
                        result.document
                    )

                export_data["domain_data"]["paths"].append(path_info)
                export_data["domain_data"]["associated_ids"].add(result.ids_name)

                if result.document.metadata.units:
                    units_count[result.document.metadata.units] = (
                        units_count.get(result.document.metadata.units, 0) + 1
                    )

                # Extract measurement types from documentation
                doc_lower = result.document.documentation.lower()
                if "temperature" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add("temperature")
                if "density" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add("density")
                if "pressure" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add("pressure")
                if "magnetic" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add(
                        "magnetic_field"
                    )
                if "electric" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add(
                        "electric_field"
                    )
                if "flux" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add("flux")
                if "current" in doc_lower:
                    export_data["domain_data"]["measurement_types"].add("current")

            # Convert sets to lists and finalize data
            export_data["domain_data"]["associated_ids"] = list(
                export_data["domain_data"]["associated_ids"]
            )
            export_data["domain_data"]["measurement_types"] = list(
                export_data["domain_data"]["measurement_types"]
            )
            export_data["domain_data"]["lifecycle_stages"] = list(
                export_data["domain_data"]["lifecycle_stages"]
            )
            export_data["domain_data"]["units_distribution"] = dict(
                sorted(units_count.items(), key=lambda x: x[1], reverse=True)
            )

            # Perform domain structure analysis using graph analyzer
            try:
                associated_ids = export_data["domain_data"]["associated_ids"]
                if len(associated_ids) > 1:
                    # Build the data structure expected by analyze_cross_ids_patterns
                    ids_data_for_analysis = {}
                    for ids_name in associated_ids[:5]:  # Limit to 5 IDS
                        ids_documents = self.document_store.get_documents_by_ids(
                            ids_name
                        )
                        if ids_documents:
                            ids_data_for_analysis[ids_name] = {
                                doc.metadata.path_name: {
                                    "path": doc.metadata.path_name,
                                    "documentation": doc.documentation,
                                    "data_type": doc.metadata.data_type,
                                    "physics_domain": doc.metadata.physics_domain,
                                    "units": doc.metadata.units,
                                }
                                for doc in ids_documents[:20]  # Limit paths per IDS
                            }

                    if len(ids_data_for_analysis) > 1:
                        structure_patterns = (
                            self.graph_analyzer.analyze_cross_ids_patterns(
                                ids_data_for_analysis
                            )
                        )
                        export_data["domain_structure"] = {
                            "structural_patterns": structure_patterns,
                            "domain_coherence": f"Domain spans {len(associated_ids)} IDS with structured relationships",
                            "key_ids_for_domain": associated_ids[:5],
                        }
                    else:
                        export_data["domain_structure"] = {
                            "note": "Insufficient IDS data for structural analysis",
                            "single_ids_analysis": f"Domain primarily contained in: {list(ids_data_for_analysis.keys())}",
                        }
                else:
                    export_data["domain_structure"] = {
                        "note": f"Domain primarily contained in single IDS: {associated_ids[0] if associated_ids else 'unknown'}",
                        "structural_analysis": "Limited cross-IDS structure analysis available",
                    }

            except Exception as e:
                logger.warning(f"Domain structure analysis failed: {e}")
                export_data["domain_structure"] = {"error": str(e)}

            # Cross-domain analysis if requested
            if include_cross_domain:
                try:
                    # Find related domains through measurement types and units
                    related_domains = set()
                    # Limit measurement types to prevent exponential search
                    measurement_types = export_data["domain_data"]["measurement_types"][
                        :3
                    ]

                    for measurement in measurement_types:
                        related_search = self.semantic_search.search(
                            measurement,
                            top_k=5,  # Reduced from 10 to 5
                        )
                        for result in related_search[:3]:  # Reduced from 5 to 3
                            if (
                                result.document.metadata.physics_domain
                                and result.document.metadata.physics_domain != domain
                            ):
                                related_domains.add(
                                    result.document.metadata.physics_domain
                                )

                    cross_domain_data = {}
                    for related_domain in list(
                        related_domains
                    )[
                        :2  # Reduced from 3 to 2 related domains
                    ]:  # Limit to 3 related domains
                        related_search = self.semantic_search.search(
                            related_domain, top_k=10
                        )
                        cross_domain_data[related_domain] = {
                            "connection_strength": len(
                                [
                                    r
                                    for r in related_search
                                    if any(
                                        mt in r.document.documentation.lower()
                                        for mt in export_data["domain_data"][
                                            "measurement_types"
                                        ]
                                    )
                                ]
                            ),
                            "shared_measurements": [
                                mt
                                for mt in export_data["domain_data"][
                                    "measurement_types"
                                ]
                                if any(
                                    mt in r.document.documentation.lower()
                                    for r in related_search[:5]
                                )
                            ],
                            "sample_connections": [
                                r.document.metadata.path_name
                                for r in related_search[:2]
                            ],
                        }

                    export_data["cross_domain_analysis"] = cross_domain_data

                except Exception as e:
                    logger.warning(f"Cross-domain analysis failed: {e}")
                    export_data["cross_domain_analysis"] = {"error": str(e)}

            # Analyze measurement dependencies
            try:
                dependencies = {}
                measurement_types = export_data["domain_data"]["measurement_types"]

                for measurement in measurement_types:
                    # Find paths that might depend on this measurement
                    dep_search = self.semantic_search.search(
                        f"{measurement} dependency", top_k=5
                    )
                    dependencies[measurement] = {
                        "dependent_paths": [
                            r.document.metadata.path_name for r in dep_search[:3]
                        ],
                        "measurement_context": f"Analysis of {measurement} dependencies within {domain} domain",
                    }

                export_data["measurement_dependencies"] = dependencies

            except Exception as e:
                logger.warning(f"Measurement dependency analysis failed: {e}")
                export_data["measurement_dependencies"] = {"error": str(e)}

            # Generate export summary
            export_data["export_summary"] = {
                "domain": domain,
                "total_paths_found": len(export_data["domain_data"]["paths"]),
                "associated_ids_count": len(
                    export_data["domain_data"]["associated_ids"]
                ),
                "unique_measurement_types": len(
                    export_data["domain_data"]["measurement_types"]
                ),
                "unique_units": len(export_data["domain_data"]["units_distribution"]),
                "cross_domain_connections": len(
                    export_data.get("cross_domain_analysis", {})
                ),
                "analysis_completeness": "complete"
                if analysis_depth == "comprehensive"
                else analysis_depth,
                "domain_coverage": "extensive"
                if len(export_data["domain_data"]["paths"]) > 20
                else "focused",
            }

            # AI enhancement if MCP context available - handled by decorator
            if ctx:
                export_data["ai_prompt"] = f"""Physics Domain Export Analysis:
Domain: {domain}
Analysis Depth: {analysis_depth}
Include Cross-Domain: {include_cross_domain}

Domain Data Summary: {export_data["export_summary"]}
Associated IDS: {export_data["domain_data"]["associated_ids"]}
Measurement Types: {export_data["domain_data"]["measurement_types"]}
Units Distribution: {list(export_data["domain_data"]["units_distribution"].keys())[:5]}

Provide domain-specific analysis including:
1. Comprehensive physics context for the {domain} domain
2. Key measurement relationships and dependencies within the domain
3. Suggested analysis approaches for domain-specific research
4. Integration patterns with other physics domains
5. Practical considerations for domain-focused data analysis

Format as JSON with 'physics_context', 'measurement_relationships', 'analysis_approaches', 'integration_patterns', 'practical_considerations' fields."""

            return export_data

        except Exception as e:
            logger.error(f"Physics domain export failed: {e}")
            return {
                "domain": domain,
                "error": str(e),
                "explanation": "Failed to export physics domain data",
                "suggestions": [
                    "Check domain name spelling",
                    "Try with analysis_depth='overview' for faster processing",
                    "Use search_imas to explore available domains",
                    "Reduce max_paths for better performance",
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
