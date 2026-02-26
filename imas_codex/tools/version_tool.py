"""Version metadata tool for graph-derived DD version reporting.

Queries DDVersion nodes from the graph to provide version range,
current version, and count — replacing the hardcoded dd_version constant
as the source of truth for version information.
"""

import logging

from imas_codex.search.decorators import mcp_tool

logger = logging.getLogger(__name__)


class VersionTool:
    """DD version metadata tool backed by graph DDVersion nodes."""

    def __init__(self, graph_client):
        self.graph_client = graph_client

    @mcp_tool(
        "Get IMAS Data Dictionary version metadata from the knowledge graph. "
        "Returns the current DD version, full version range, total version count, "
        "and the version chain. Use this to understand what DD versions are "
        "available for version-scoped queries."
    )
    async def get_dd_versions(self) -> dict:
        """Query DDVersion nodes for version metadata.

        Returns:
            Dict with current_version, version_range, version_count,
            and versions list.
        """
        try:
            results = self.graph_client.query(
                "MATCH (d:DDVersion) "
                "RETURN d.id AS id, d.major AS major, d.minor AS minor, "
                "d.patch AS patch, d.is_current AS is_current "
                "ORDER BY d.major, d.minor, d.patch"
            )
        except Exception as e:
            return {"error": f"Failed to query DDVersion nodes: {e}"}

        if not results:
            return {
                "current_version": None,
                "version_range": None,
                "version_count": 0,
                "versions": [],
            }

        versions = [row["id"] for row in results]
        current = next(
            (row["id"] for row in results if row.get("is_current")),
            versions[-1],  # fallback to latest by sort order
        )

        return {
            "current_version": current,
            "version_range": f"{versions[0]} - {versions[-1]}",
            "version_count": len(versions),
            "versions": versions,
        }
