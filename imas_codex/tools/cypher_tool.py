"""Read-only Cypher query tool for graph-native MCP access.

Enables agents to compose arbitrary read-only queries against the IMAS
Data Dictionary knowledge graph. Mutations are blocked at the query
level before execution.
"""

import logging
import re

from fastmcp import Context

from imas_codex.search.decorators import mcp_tool

logger = logging.getLogger(__name__)

# Cypher keywords that indicate a write operation
_MUTATION_KEYWORDS = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|LOAD\s+CSV|FOREACH|CALL\s*\{)\b",
    re.IGNORECASE,
)

# Maximum rows returned to prevent context overflow
MAX_RESULT_ROWS = 200


def _is_read_only(cypher: str) -> bool:
    """Check if a Cypher query contains only read operations.

    Strips string literals and comments before checking for mutation
    keywords to avoid false positives from quoted text.
    """
    # Remove single-line comments
    cleaned = re.sub(r"//.*$", "", cypher, flags=re.MULTILINE)
    # Remove block comments
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    # Remove string literals (both single and double quoted)
    cleaned = re.sub(r"'[^']*'", "''", cleaned)
    cleaned = re.sub(r'"[^"]*"', '""', cleaned)

    return not _MUTATION_KEYWORDS.search(cleaned)


class CypherTool:
    """Read-only Cypher query tool for the IMAS DD graph."""

    def __init__(self, graph_client):
        self.graph_client = graph_client

    @mcp_tool(
        "Execute a read-only Cypher query against the IMAS Data Dictionary graph. "
        "Use this for flexible graph traversal: version evolution, path history, "
        "relationship discovery, and cross-referencing. "
        "cypher (required): A read-only Cypher query (MATCH, RETURN, WITH, WHERE, "
        "ORDER BY, LIMIT, CALL db.index.vector.queryNodes). "
        "Mutations (CREATE, DELETE, SET, MERGE, DROP, REMOVE) are rejected. "
        "Results are truncated to 200 rows. Use 'LIMIT' for smaller result sets. "
        "Call get_dd_graph_schema() first to understand available node types, "
        "relationships, and properties before composing queries."
    )
    async def query_imas_graph(
        self,
        cypher: str,
        ctx: Context | None = None,
    ) -> dict:
        """Execute a read-only Cypher query against the graph.

        Args:
            cypher: Read-only Cypher query string.
            ctx: FastMCP context.

        Returns:
            Query results as a dict with 'rows', 'columns', and 'truncated' keys.
        """
        if not cypher or not cypher.strip():
            return {"error": "Empty query", "rows": [], "columns": []}

        if not _is_read_only(cypher):
            return {
                "error": "Query rejected: only read-only Cypher is allowed. "
                "Mutations (CREATE, DELETE, SET, MERGE, DROP, REMOVE) are blocked.",
                "rows": [],
                "columns": [],
            }

        try:
            results = self.graph_client.query(cypher)
        except Exception as e:
            return {"error": str(e), "rows": [], "columns": []}

        if not results:
            return {"rows": [], "columns": [], "truncated": False}

        columns = list(results[0].keys())
        truncated = len(results) > MAX_RESULT_ROWS
        rows = [dict(row) for row in results[:MAX_RESULT_ROWS]]

        # Strip embedding vectors from results to reduce payload size
        for row in rows:
            for key in list(row.keys()):
                if isinstance(row[key], list) and len(row[key]) > 20:
                    row[key] = f"[vector dim={len(row[key])}]"

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }
