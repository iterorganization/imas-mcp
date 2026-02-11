"""
Smolagents tools for agent workflows.

Converts existing tool functions to smolagents @tool decorated functions.
Tools should:
- Have clear docstrings with Args/Returns
- Log extensively with print() for agent debugging
- Return descriptive error messages (not raise)

Architecture:
    Tools are stateless functions decorated with @tool.
    For stateful tools (e.g., facility-bound), use Tool class subclasses.
"""

from __future__ import annotations

import logging
from typing import Any

from smolagents import Tool, tool

from imas_codex.graph import GraphClient
from imas_codex.remote.tools import run

logger = logging.getLogger(__name__)


# =============================================================================
# Graph Tools
# =============================================================================


@tool
def query_neo4j(cypher: str, params: str = "") -> str:
    """Execute a read-only Cypher query against the Neo4j graph database.

    Use this to explore TreeNodes, code examples, facility data, and relationships.
    The graph contains MDSplus tree structures, code chunks with embeddings, and wiki content.

    Args:
        cypher: A READ-ONLY Cypher query (MATCH, RETURN, CALL only)
        params: Optional JSON string of query parameters

    Returns:
        Query results as formatted string (max 20 rows), or error message
    """
    import json

    print(f"Executing Cypher query: {cypher[:100]}...")

    # Parse params if provided
    query_params: dict[str, Any] = {}
    if params:
        try:
            query_params = json.loads(params)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON params: {e}"

    # Safety check for mutations
    upper = cypher.upper()
    if any(
        kw in upper for kw in ["CREATE", "DELETE", "SET", "MERGE", "REMOVE", "DROP"]
    ):
        return "Error: Only read-only queries are allowed (MATCH, RETURN, CALL)"

    try:
        with GraphClient() as gc:
            result = gc.query(cypher, **query_params)
            if not result:
                print("Query returned no results")
                return "No results found"

            output = []
            for r in result[:20]:
                output.append(str(dict(r)))
            if len(result) > 20:
                output.append(f"... and {len(result) - 20} more rows")

            print(f"Query returned {len(result)} rows")
            full_output = "\n".join(output)
            if len(full_output) > 10000:
                return full_output[:10000] + "... (truncated)"
            return full_output
    except Exception as e:
        return f"Query error: {e}"


# =============================================================================
# Code Search Tools
# =============================================================================


@tool
def search_code_examples(query: str, limit: int = 5, min_score: float = 0.5) -> str:
    """Search code examples using semantic vector search.

    Finds code that semantically matches your query even without exact keyword matches.
    Use this to find how paths are used in real code before enriching.

    Args:
        query: Natural language query or path fragment (e.g., 'electron density profile')
        limit: Maximum number of results (default: 5)
        min_score: Minimum similarity score 0-1 (default: 0.5)

    Returns:
        Matching code snippets with file names, function names, and scores
    """
    print(f"Searching code examples for: {query}")

    try:
        from imas_codex.ingestion.search import ChunkSearch

        searcher = ChunkSearch()
        results = searcher.search(query=query, top_k=limit, min_score=min_score)

        if not results:
            print("No code examples found")
            return f"No code examples found matching '{query}'"

        print(f"Found {len(results)} code examples")
        output = []
        for r in results:
            header = f"=== {r.source_file} (score: {r.score:.2f}) ==="
            if r.function_name:
                header = (
                    f"=== {r.source_file}::{r.function_name} (score: {r.score:.2f}) ==="
                )
            output.append(header)
            content = r.content[:600] + "..." if len(r.content) > 600 else r.content
            output.append(content)
            if r.related_ids:
                output.append(f"  Related IDS: {', '.join(r.related_ids)}")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


# =============================================================================
# IMAS Tools (singleton DocumentStore)
# =============================================================================

_imas_tools_instance = None


def _get_imas_tools():
    """Get or create singleton Tools instance with shared DocumentStore."""
    global _imas_tools_instance
    if _imas_tools_instance is None:
        from imas_codex.tools import Tools

        _imas_tools_instance = Tools()
    return _imas_tools_instance


def _run_async(coro):
    """Run async coroutine in sync context."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


@tool
def search_imas_paths(query: str, ids_filter: str = "", max_results: int = 10) -> str:
    """Search IMAS Data Dictionary for physics paths using semantic search.

    Use this to find IMAS paths that might correspond to MDSplus TreeNodes,
    enabling semantic mapping between local data and IMAS standard.

    Args:
        query: Natural language query (e.g., 'electron temperature profile')
        ids_filter: Optional IDS name filter (e.g., 'core_profiles equilibrium')
        max_results: Maximum results to return (default: 10)

    Returns:
        Matching IMAS paths with descriptions, units, and similarity scores
    """
    print(f"Searching IMAS paths for: {query}")

    try:
        tools = _get_imas_tools()
        result = _run_async(
            tools.search_imas_paths(
                query=query,
                ids_filter=ids_filter if ids_filter else None,
                max_results=max_results,
            )
        )
        if not result.hits:
            print("No IMAS paths found")
            return f"No IMAS paths found for '{query}'"

        print(f"Found {len(result.hits)} IMAS paths")
        output = []
        for hit in result.hits:
            line = f"{hit.full_path} (score: {hit.score:.2f})"
            output.append(line)
            if hit.documentation:
                output.append(f"  {hit.documentation[:150]}...")
            if hit.units:
                output.append(f"  Units: {hit.units}")
            output.append("")
        return "\n".join(output)
    except Exception as e:
        return f"IMAS search error: {e}"


@tool
def check_imas_paths(paths: str) -> str:
    """Validate IMAS paths for existence in the Data Dictionary.

    Use this to quickly verify if specific paths exist and get basic info.

    Args:
        paths: Space-delimited IMAS paths (e.g., 'equilibrium/time_slice/global_quantities/ip')

    Returns:
        Validation results with existence status and basic info
    """
    print(f"Checking IMAS paths: {paths[:100]}...")

    try:
        tools = _get_imas_tools()
        result = _run_async(tools.check_imas_paths(paths=paths))
        return str(result)
    except Exception as e:
        return f"Path check error: {e}"


@tool
def fetch_imas_paths(paths: str) -> str:
    """Get full documentation for IMAS paths.

    Use this to get detailed documentation, units, coordinates, and data types.

    Args:
        paths: Space-delimited IMAS paths

    Returns:
        Detailed path documentation including units, coordinates, type
    """
    print(f"Fetching IMAS path docs: {paths[:100]}...")

    try:
        tools = _get_imas_tools()
        result = _run_async(tools.fetch_imas_paths(paths=paths))
        return str(result)
    except Exception as e:
        return f"Fetch error: {e}"


# =============================================================================
# Wiki Tools
# =============================================================================


@tool
def search_wiki(query: str, limit: int = 5, facility: str = "tcv") -> str:
    """Search wiki documentation using semantic vector search.

    Use for official signal descriptions, sign conventions, and diagnostic specs.
    Wiki content is authoritative documentation from facility experts.

    Args:
        query: Natural language query (e.g., 'Thomson scattering calibration')
        limit: Maximum number of results (default: 5)
        facility: Facility ID (default: 'tcv')

    Returns:
        Matching wiki chunks with page titles, sections, and content
    """
    print(f"Searching wiki for: {query}")

    try:
        from imas_codex.embeddings import get_embed_model

        embed_model = get_embed_model()
        query_embedding = embed_model.get_text_embedding(query)

        with GraphClient() as gc:
            result = gc.query(
                """
                CALL db.index.vector.queryNodes('wiki_chunk_embedding', $limit, $embedding)
                YIELD node, score
                MATCH (p:WikiPage)-[:HAS_CHUNK]->(node)
                WHERE p.facility_id = $facility
                RETURN p.title AS page_title, p.url AS url,
                       node.content AS content, score
                ORDER BY score DESC
                """,
                limit=limit,
                embedding=query_embedding,
                facility=facility,
            )

            if not result:
                return f"No wiki content found for '{query}' in {facility}"

            print(f"Found {len(result)} wiki chunks")
            output = []
            for r in result:
                output.append(f"=== {r['page_title']} (score: {r['score']:.3f}) ===")
                output.append(f"URL: {r['url']}")
                content = (
                    r["content"][:500] + "..."
                    if len(r["content"]) > 500
                    else r["content"]
                )
                output.append(content)
                output.append("")

            return "\n".join(output)
    except Exception as e:
        return f"Wiki search error: {e}"


@tool
def get_wiki_context_for_path(path: str, facility: str = "tcv") -> str:
    """Get wiki documentation for a specific MDSplus path.

    Searches for WikiChunks that document a given path. Use during enrichment
    to add authoritative descriptions from facility documentation.

    Args:
        path: MDSplus path (e.g., '\\RESULTS::THOMSON:NE')
        facility: Facility ID (default: 'tcv')

    Returns:
        Wiki content that documents this path
    """
    print(f"Looking up wiki context for: {path}")

    try:
        normalized = path.lstrip("\\").upper()

        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (wc:WikiChunk)-[:DOCUMENTS]->(t:TreeNode)
                WHERE t.path CONTAINS $path OR t.canonical_path = $normalized
                MATCH (wp:WikiPage)-[:HAS_CHUNK]->(wc)
                RETURN wp.title AS page_title, wc.content AS content
                LIMIT 5
                """,
                path=path,
                normalized=normalized,
            )

            if not result:
                # Try fuzzy match
                result = gc.query(
                    """
                    MATCH (wc:WikiChunk {facility_id: $facility})
                    WHERE ANY(p IN wc.mdsplus_paths_mentioned WHERE p CONTAINS $path_part)
                    MATCH (wp:WikiPage)-[:HAS_CHUNK]->(wc)
                    RETURN wp.title AS page_title, wc.content AS content
                    LIMIT 5
                    """,
                    facility=facility,
                    path_part=path.split("::")[-1] if "::" in path else path,
                )

            if not result:
                return f"No wiki documentation found for {path}"

            print(f"Found {len(result)} wiki chunks for path")
            output = [f"Wiki documentation for {path}:", ""]
            for r in result:
                output.append(f"From: {r['page_title']}")
                content = (
                    r["content"][:400] + "..."
                    if len(r["content"]) > 400
                    else r["content"]
                )
                output.append(content)
                output.append("")

            return "\n".join(output)
    except Exception as e:
        return f"Wiki context error: {e}"


# =============================================================================
# Facility-Bound Tool Classes (for exploration)
# =============================================================================


class RunCommandTool(Tool):
    """Execute shell commands on a specific facility.

    This is a Tool class (not @tool function) because it's bound to a
    specific facility at creation time.
    """

    name = "run_command"
    description = (
        "Execute a shell command on the facility. "
        "Use rg, fd, dust for file discovery. Auto-detects local vs SSH."
    )
    inputs = {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default: 60)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(self, command: str, timeout: int = 60) -> str:
        """Execute the command."""
        print(f"Running on {self.facility}: {command[:80]}...")
        try:
            result = run(command, facility=self.facility, timeout=timeout)
            lines = result.strip().split("\n")
            print(f"Command returned {len(lines)} lines")
            return result
        except TimeoutError:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Command error: {e}"


class QueueFilesTool(Tool):
    """Queue source files for ingestion into the knowledge graph."""

    name = "queue_files"
    description = (
        "Queue source files for ingestion. MUST call this to persist discoveries. "
        "Set interest_score based on physics value (0.9+ for IMAS, 0.7+ for physics codes)."
    )
    inputs = {
        "file_paths": {
            "type": "array",
            "description": "List of absolute file paths on the facility",
        },
        "interest_score": {
            "type": "number",
            "description": "Priority score 0.0-1.0 (default: 0.7)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility
        self.files_queued: list[str] = []

    def forward(self, file_paths: list[str], interest_score: float = 0.7) -> str:
        """Queue the files."""
        from imas_codex.ingestion import queue_source_files

        if not file_paths:
            return "No files provided"

        print(f"Queueing {len(file_paths)} files with score {interest_score}")

        try:
            result = queue_source_files(
                facility=self.facility,
                file_paths=file_paths,
                interest_score=interest_score,
                discovered_by="explore_agent",
            )
            self.files_queued.extend(file_paths[: result["discovered"]])

            summary = (
                f"Queued: {result['discovered']}, "
                f"Skipped: {result['skipped']} (already discovered), "
                f"Errors: {len(result['errors'])}"
            )
            print(summary)
            return summary
        except Exception as e:
            return f"Queue error: {e}"


class AddNoteTool(Tool):
    """Add a timestamped exploration note for a facility."""

    name = "add_note"
    description = (
        "Add timestamped exploration note. Use for significant findings "
        "like IMAS patterns, code locations, conventions."
    )
    inputs = {
        "note": {"type": "string", "description": "The observation to record"},
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility
        self.notes: list[str] = []

    def forward(self, note: str) -> str:
        """Add the note."""
        from imas_codex.discovery import add_exploration_note

        print(f"Adding note: {note[:80]}...")

        try:
            add_exploration_note(self.facility, note)
            self.notes.append(note)
            return f"Note added for {self.facility}"
        except Exception as e:
            return f"Failed to add note: {e}"


class GetFacilityInfoTool(Tool):
    """Get current facility configuration and exploration status."""

    name = "get_facility_info"
    description = (
        "Get current facility config and exploration status. "
        "Check before exploring to see what's already known."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(self) -> str:
        """Get facility info."""
        from imas_codex.discovery import get_facility

        print(f"Getting info for facility: {self.facility}")

        try:
            info = get_facility(self.facility)

            output = [f"Facility: {self.facility}"]
            output.append(f"SSH host: {info.get('ssh_host', self.facility)}")

            if tools := info.get("tools", {}):
                tool_strs = []
                for k, v in tools.items():
                    version = v.get("version", "?") if isinstance(v, dict) else v
                    tool_strs.append(f"{k}={version}")
                output.append(f"Tools: {', '.join(tool_strs)}")

            if paths := info.get("paths", {}):
                output.append("Known paths:")
                for category, path_dict in paths.items():
                    if isinstance(path_dict, dict):
                        for p, desc in list(path_dict.items())[:3]:
                            output.append(f"  [{category}] {p}: {desc}")

            if notes := info.get("exploration_notes", []):
                output.append(f"Notes: {len(notes)} exploration notes")

            return "\n".join(output)
        except Exception as e:
            return f"Failed to get facility info: {e}"


# =============================================================================
# Tool Collection Factories
# =============================================================================


def get_enrichment_tools() -> list[Tool]:
    """Get tools optimized for enrichment tasks.

    Returns tools for:
    - Neo4j graph queries (sibling nodes, existing metadata)
    - Code example search (usage patterns)
    - Wiki context lookup (authoritative descriptions)
    - IMAS path search (mapping targets)
    """
    return [
        query_neo4j,
        search_code_examples,
        get_wiki_context_for_path,
        search_imas_paths,
        search_wiki,
    ]


def get_exploration_tools(facility: str) -> list[Tool]:
    """Get tools configured for facility exploration.

    Args:
        facility: Facility ID to bind tools to

    Returns:
        List of tools including:
        - run_command: Shell execution on facility
        - query_graph: Neo4j queries
        - queue_files: Persist discoveries
        - add_note: Record observations
        - get_facility_info: Check current state
    """
    return [
        RunCommandTool(facility),
        query_neo4j,
        QueueFilesTool(facility),
        AddNoteTool(facility),
        GetFacilityInfoTool(facility),
    ]


def get_all_tools(facility: str = "tcv") -> list[Tool]:
    """Get all available tools.

    Warning: Includes IMAS tools which have ~30s startup cost for embeddings.

    Args:
        facility: Default facility for facility-bound tools

    Returns:
        Combined list of all tools
    """
    return [
        # Graph
        query_neo4j,
        # Code search
        search_code_examples,
        # IMAS
        search_imas_paths,
        check_imas_paths,
        fetch_imas_paths,
        # Wiki
        search_wiki,
        get_wiki_context_for_path,
        # Facility-bound
        RunCommandTool(facility),
        QueueFilesTool(facility),
        AddNoteTool(facility),
        GetFacilityInfoTool(facility),
    ]
