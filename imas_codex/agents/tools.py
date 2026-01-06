"""
Reusable tools for LlamaIndex agents.

Provides FunctionTools for:
- Neo4j graph queries
- MDSplus SSH queries
- Code example search
- IMAS DD search (direct, no MCP overhead)

Architecture:
    Tools are exposed as simple sync functions that LlamaIndex can wrap.
    The IMAS search tools use a singleton DocumentStore to avoid repeated
    embedding model loading (~30s startup cost paid once).
"""

import asyncio
import subprocess
from typing import Any

from llama_index.core.tools import FunctionTool

from imas_codex.graph import GraphClient

# =============================================================================
# Singleton IMAS Tools (avoids repeated DocumentStore initialization)
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


def _query_neo4j(cypher: str, params: dict[str, Any] | None = None) -> str:
    """
    Execute a Cypher query against the Neo4j graph database.

    Use this to explore TreeNode metadata, relationships, and patterns.
    The graph contains MDSplus tree structures, code examples, and facility data.

    Args:
        cypher: A READ-ONLY Cypher query
        params: Optional query parameters

    Returns:
        Query results as formatted string (max 20 rows)

    Examples:
        - MATCH (n:TreeNode) WHERE n.path CONTAINS 'ASTRA' RETURN n.path, n.description LIMIT 10
        - MATCH (t:MDSplusTree) RETURN t.name, t.node_count_total
        - MATCH (c:CodeChunk)-[:CONTAINS_REF]->(d:DataReference) RETURN d.raw_string LIMIT 5
    """
    try:
        with GraphClient() as gc:
            result = gc.query(cypher, **(params or {}))
            if not result:
                return "No results found"
            output = []
            for r in result[:20]:
                output.append(str(dict(r)))
            if len(result) > 20:
                output.append(f"... and {len(result) - 20} more rows")
            return "\n".join(output)
    except Exception as e:
        return f"Query error: {e}"


def _ssh_mdsplus_query(
    tree_name: str,
    path: str,
    shot: int = 80000,
    facility: str = "epfl",
) -> str:
    """
    Query MDSplus database via SSH for node metadata.

    Use this to get the actual description, units, and usage type
    of an MDSplus node directly from the TCV database.

    Args:
        tree_name: MDSplus tree name (e.g., 'results', 'magnetics', 'tcv_shot')
        path: MDSplus path (e.g., '\\RESULTS::TOP.ASTRA')
        shot: Shot number to query (default: 80000)
        facility: SSH host alias (default: 'epfl')

    Returns:
        Node metadata: usage, description, units, dtype
    """
    escaped_path = path.replace("\\", "\\\\").replace("'", "\\'")
    cmd = f'''python3 -c "
import MDSplus
try:
    tree = MDSplus.Tree('{tree_name}', {shot})
    node = tree.getNode('{escaped_path}')
    print('usage:', node.usage)
    print('description:', str(node.description) if node.description else 'None')
    print('units:', str(node.units) if node.units else 'None')
    print('dtype:', node.dtype_str if hasattr(node, 'dtype_str') else 'unknown')
    # Get children if it's a structure node
    try:
        children = node.getChildren()
        if children:
            print('children:', ', '.join(c.node_name for c in children[:10]))
            if len(children) > 10:
                print(f'  ... and {{len(children) - 10}} more')
    except:
        pass
except Exception as e:
    print('error:', str(e))
"'''
    try:
        result = subprocess.run(
            ["ssh", facility, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "SSH query timed out after 30s"
    except Exception as e:
        return f"SSH error: {e}"


def _ssh_command(command: str, facility: str = "epfl", timeout: int = 60) -> str:
    """
    Execute an arbitrary SSH command on a remote facility.

    Use this for exploration tasks like listing directories, searching files,
    or running analysis tools.

    Args:
        command: Shell command to execute
        facility: SSH host alias (default: 'epfl')
        timeout: Command timeout in seconds (default: 60)

    Returns:
        Command output (stdout + stderr)

    Examples:
        - ls -la /home/codes/liuqe
        - rg -l 'equilibrium' /home/codes --max-depth 3
        - python3 -c "import MDSplus; print(MDSplus.__version__)"
    """
    try:
        result = subprocess.run(
            ["ssh", facility, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"SSH error: {e}"


def _search_code_examples(query: str, limit: int = 5, min_score: float = 0.5) -> str:
    """
    Search code examples using semantic vector search.

    Uses embedded code chunks for similarity search, finding code that
    semantically matches your query even without exact keyword matches.

    Args:
        query: Natural language query or path fragment (e.g., 'electron density profile')
        limit: Maximum number of results (default: 5)
        min_score: Minimum similarity score 0-1 (default: 0.5)

    Returns:
        Matching code snippets with file names, function names, and scores
    """
    try:
        from imas_codex.code_examples.search import CodeExampleSearch

        searcher = CodeExampleSearch()
        results = searcher.search(query=query, top_k=limit, min_score=min_score)

        if not results:
            return f"No code examples found matching '{query}'"

        output = []
        for r in results:
            header = f"=== {r.source_file} (score: {r.score:.2f}) ==="
            if r.function_name:
                header = (
                    f"=== {r.source_file}::{r.function_name} (score: {r.score:.2f}) ==="
                )
            output.append(header)
            # Truncate long content
            content = r.content[:600] + "..." if len(r.content) > 600 else r.content
            output.append(content)
            if r.related_ids:
                output.append(f"  Related IDS: {', '.join(r.related_ids)}")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


def _search_imas_paths(
    query: str, ids_filter: str | None = None, max_results: int = 10
) -> str:
    """
    Search IMAS Data Dictionary for physics paths using semantic search.

    Use this to find IMAS paths that might correspond to MDSplus TreeNodes,
    enabling semantic mapping between local data and IMAS standard.

    Args:
        query: Natural language query (e.g., 'electron temperature profile')
        ids_filter: Optional IDS name filter (e.g., 'core_profiles equilibrium')
        max_results: Maximum results to return (default: 10)

    Returns:
        Matching IMAS paths with descriptions, units, and scores
    """
    try:
        tools = _get_imas_tools()
        result = _run_async(
            tools.search_imas_paths(
                query=query,
                ids_filter=ids_filter,
                max_results=max_results,
            )
        )
        if not result.hits:
            return f"No IMAS paths found for '{query}'"
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


def _check_imas_paths(paths: str) -> str:
    """
    Validate IMAS paths for existence in the Data Dictionary.

    Use this to quickly verify if specific paths exist and get basic info.

    Args:
        paths: Space-delimited IMAS paths (e.g., 'equilibrium/time_slice/global_quantities/ip')

    Returns:
        Validation results with existence status and basic info
    """
    try:
        tools = _get_imas_tools()
        result = _run_async(tools.check_imas_paths(paths=paths))
        return str(result)
    except Exception as e:
        return f"Path check error: {e}"


def _fetch_imas_paths(paths: str) -> str:
    """
    Get full documentation for IMAS paths.

    Use this to get detailed documentation, units, coordinates, and data types.

    Args:
        paths: Space-delimited IMAS paths

    Returns:
        Detailed path documentation including units, coordinates, type
    """
    try:
        tools = _get_imas_tools()
        result = _run_async(tools.fetch_imas_paths(paths=paths))
        return str(result)
    except Exception as e:
        return f"Fetch error: {e}"


def _list_imas_paths(paths: str, leaf_only: bool = True, max_paths: int = 100) -> str:
    """
    List data paths in IDS with minimal overhead.

    Use for structure exploration - returns paths only, no descriptions.

    Args:
        paths: Space-separated IDS names or path prefixes (e.g., 'equilibrium core_profiles')
        leaf_only: If True, only return data fields, not intermediate nodes
        max_paths: Limit output size

    Returns:
        Tree structure of paths in YAML format
    """
    try:
        tools = _get_imas_tools()
        result = _run_async(
            tools.list_imas_paths(
                paths=paths,
                leaf_only=leaf_only,
                max_paths=max_paths,
            )
        )
        return str(result)
    except Exception as e:
        return f"List error: {e}"


def _get_imas_overview(query: str | None = None) -> str:
    """
    Get high-level overview of IMAS Data Dictionary structure.

    Use this first to understand what IDS are available and their complexity.

    Args:
        query: Optional keyword filter for IDS names/descriptions

    Returns:
        Overview including IDS list, physics domains, statistics
    """
    try:
        tools = _get_imas_tools()
        result = _run_async(tools.get_imas_overview(query=query))
        return str(result)
    except Exception as e:
        return f"Overview error: {e}"


def _search_imas_clusters(query: str, ids_filter: str | None = None) -> str:
    """
    Find clusters of semantically related IMAS paths.

    Use this to discover related data structures across diagnostics.

    Args:
        query: Natural language or IMAS path (e.g., 'electron density measurements')
        ids_filter: Optional IDS filter

    Returns:
        Clusters of related paths - cross_ids and intra_ids groupings
    """
    try:
        tools = _get_imas_tools()
        result = _run_async(
            tools.search_imas_clusters(query=query, ids_filter=ids_filter)
        )
        return str(result)
    except Exception as e:
        return f"Cluster search error: {e}"


def _get_tree_structure(tree_name: str, path_prefix: str = "") -> str:
    """
    Get TreeNode structure from the graph for a specific tree or subtree.

    Use this to understand the hierarchical structure of MDSplus trees
    stored in the knowledge graph.

    Args:
        tree_name: MDSplus tree name (e.g., 'results', 'magnetics')
        path_prefix: Optional path prefix to filter (e.g., '\\RESULTS::ASTRA')

    Returns:
        Tree structure with paths and descriptions
    """
    try:
        with GraphClient() as gc:
            if path_prefix:
                result = gc.query(
                    """
                    MATCH (n:TreeNode {tree_name: $tree})
                    WHERE n.path STARTS WITH $prefix
                    RETURN n.path AS path, n.description AS desc, n.units AS units
                    ORDER BY n.path
                    LIMIT 50
                    """,
                    tree=tree_name,
                    prefix=path_prefix,
                )
            else:
                result = gc.query(
                    """
                    MATCH (n:TreeNode {tree_name: $tree})
                    RETURN n.path AS path, n.description AS desc, n.units AS units
                    ORDER BY n.path
                    LIMIT 50
                    """,
                    tree=tree_name,
                )
            if not result:
                return f"No TreeNodes found for tree '{tree_name}'"
            output = []
            for r in result:
                line = r["path"]
                if r["desc"] and r["desc"] != "None":
                    line += f" - {r['desc']}"
                if r["units"] and r["units"] != "dimensionless":
                    line += f" [{r['units']}]"
                output.append(line)
            if len(result) == 50:
                output.append("... (limited to 50 results)")
            return "\n".join(output)
    except Exception as e:
        return f"Query error: {e}"


# Create FunctionTools for use with LlamaIndex agents
def get_exploration_tools() -> list[FunctionTool]:
    """
    Get the standard set of exploration tools for agents.

    Returns:
        List of FunctionTool instances for:
        - query_neo4j: Graph queries
        - ssh_mdsplus_query: MDSplus metadata lookup
        - ssh_command: Arbitrary SSH commands
        - search_code_examples: Code search
        - get_tree_structure: Tree structure lookup
    """
    return [
        FunctionTool.from_defaults(
            fn=_query_neo4j,
            name="query_neo4j",
            description=(
                "Execute a Cypher query against the Neo4j graph database. "
                "Use for exploring TreeNodes, code examples, and facility data."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_ssh_mdsplus_query,
            name="ssh_mdsplus_query",
            description=(
                "Query MDSplus database via SSH for node metadata. "
                "Gets description, units, usage type directly from TCV."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_ssh_command,
            name="ssh_command",
            description=(
                "Execute an SSH command on a remote facility. "
                "Use for exploration: ls, rg, fd, python scripts."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_search_code_examples,
            name="search_code_examples",
            description=(
                "Semantic search over code examples using vector embeddings. "
                "Finds code that matches meaning, not just keywords."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_get_tree_structure,
            name="get_tree_structure",
            description=(
                "Get TreeNode structure from the graph. "
                "Shows hierarchical MDSplus tree structure."
            ),
        ),
    ]


def get_imas_tools() -> list[FunctionTool]:
    """
    Get IMAS Data Dictionary tools (direct, no MCP overhead).

    These tools provide semantic search and exploration of the IMAS
    Data Dictionary. They use a singleton DocumentStore so embedding
    models are loaded only once.

    Returns:
        List of FunctionTool instances for:
        - search_imas_paths: Semantic search across DD
        - check_imas_paths: Fast path validation
        - fetch_imas_paths: Full path documentation
        - list_imas_paths: Structure exploration
        - get_imas_overview: High-level DD summary
        - search_imas_clusters: Find related paths
    """
    return [
        FunctionTool.from_defaults(
            fn=_search_imas_paths,
            name="search_imas_paths",
            description=(
                "Search IMAS Data Dictionary for physics paths using semantic search. "
                "Args: query (str), ids_filter (str, optional), max_results (int). "
                "Use to find IMAS equivalents for MDSplus data."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_check_imas_paths,
            name="check_imas_paths",
            description=(
                "Validate IMAS paths for existence. "
                "Args: paths (str, space-delimited). Fast existence check."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_fetch_imas_paths,
            name="fetch_imas_paths",
            description=(
                "Get full documentation for IMAS paths. "
                "Args: paths (str, space-delimited). Returns units, coords, type."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_list_imas_paths,
            name="list_imas_paths",
            description=(
                "List paths in IDS for structure exploration. "
                "Args: paths (str), leaf_only (bool), max_paths (int)."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_get_imas_overview,
            name="get_imas_overview",
            description=(
                "Get high-level overview of IMAS Data Dictionary. "
                "Args: query (str, optional filter). Start here to explore DD."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_search_imas_clusters,
            name="search_imas_clusters",
            description=(
                "Find clusters of semantically related IMAS paths. "
                "Args: query (str), ids_filter (str, optional). "
                "Discover related data across diagnostics."
            ),
        ),
    ]


# Convenience: individual tool getters for custom agent configurations
def get_graph_tool() -> FunctionTool:
    """Get just the Neo4j query tool."""
    return FunctionTool.from_defaults(fn=_query_neo4j, name="query_neo4j")


def get_ssh_tools() -> list[FunctionTool]:
    """Get SSH-related tools (MDSplus query and general command)."""
    return [
        FunctionTool.from_defaults(fn=_ssh_mdsplus_query, name="ssh_mdsplus_query"),
        FunctionTool.from_defaults(fn=_ssh_command, name="ssh_command"),
    ]


def get_search_tools() -> list[FunctionTool]:
    """Get fast search tools (code examples only, no IMAS DD).

    For IMAS DD search tools, use get_imas_tools() separately.
    Note: IMAS tools have ~30s startup cost for embedding model loading.
    """
    return [
        FunctionTool.from_defaults(
            fn=_search_code_examples, name="search_code_examples"
        ),
    ]


def get_all_tools() -> list[FunctionTool]:
    """
    Get all available tools for a fully-capable agent.

    WARNING: This includes IMAS DD tools which have ~30s startup cost
    for embedding model loading. For fast agent startup, use
    get_exploration_tools() instead.

    Combines:
    - Exploration tools (graph, SSH, code search)
    - IMAS DD tools (semantic search, path validation, structure)

    The IMAS tools use a singleton DocumentStore, so embedding models
    are loaded only once regardless of how many times this is called.

    Returns:
        Combined list of all FunctionTools (11 total)
    """
    return get_exploration_tools() + get_imas_tools()
