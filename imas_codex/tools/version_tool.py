"""Version metadata tool for graph-derived DD version reporting.

Queries DDVersion nodes from the graph to provide version range,
current version, and count — replacing the hardcoded dd_version constant
as the source of truth for version information.
"""

import logging
from typing import Any

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

    @mcp_tool(
        "Get version change history for specific IMAS paths, or list all changes of a specific type across the Data Dictionary. "
        "Mode 1 (per-path): Provide paths to get version history for specific paths. "
        "Mode 2 (bulk query): Omit paths and set change_type_filter to list all changes of that type. "
        "paths: One or more IMAS paths to check version history. Optional when change_type_filter is set. "
        "change_type_filter: Filter to specific change type when querying without paths "
        "(e.g., 'path_renamed', 'units', 'data_type', 'cocos_label_transformation', 'added'). "
        "ids_filter: Limit to specific IDS. "
        "from_version: Start version for range filter (exclusive). "
        "to_version: End version for range filter (inclusive). "
        "follow_rename_chains: When True, traverse RENAMED_TO graph edges to return full multi-hop rename lineages."
    )
    async def get_dd_version_context(
        self,
        paths: str | list[str] | None = None,
        change_type_filter: str | None = None,
        ids_filter: str | None = None,
        from_version: str | None = None,
        to_version: str | None = None,
        follow_rename_chains: bool = False,
    ) -> dict[str, Any]:
        """Get version change context for IMAS paths.

        Mode 1 (per-path): Provide paths to get version history per path.
        Mode 2 (bulk query): Omit paths and set change_type_filter to list all
        changes of that type across the Data Dictionary.

        When follow_rename_chains is True, an additional query traverses
        RENAMED_TO edges to return multi-hop rename lineages.
        """
        # Normalise paths into a list
        if paths is not None:
            if isinstance(paths, str):
                path_list: list[str] = [
                    p.strip() for p in paths.replace(",", " ").split() if p.strip()
                ]
            else:
                path_list = list(paths)
        else:
            path_list = []

        # ── Mode 2: bulk query ────────────────────────────────────────────────
        if not path_list:
            if not change_type_filter and not follow_rename_chains:
                return {
                    "error": (
                        "Provide either 'paths' for per-path history, "
                        "or 'change_type_filter' for a bulk query."
                    )
                }
            if follow_rename_chains and not change_type_filter:
                # Rename-chain only mode
                return await self._rename_chain_query(ids_filter=ids_filter)
            result = await self._bulk_query(
                change_type_filter=change_type_filter,
                ids_filter=ids_filter,
                from_version=from_version,
                to_version=to_version,
            )
            if follow_rename_chains:
                chain_result = await self._rename_chain_query(ids_filter=ids_filter)
                result["rename_chains"] = chain_result.get("rename_chains", [])
            return result

        # ── Mode 1: per-path ─────────────────────────────────────────────────
        try:
            results = self.graph_client.query(
                """
                UNWIND $path_ids AS pid
                MATCH (p:IMASNode {id: pid})
                OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv:DDVersion)
                OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dv:DDVersion)
                OPTIONAL MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p)
                OPTIONAL MATCH (change)-[:IN_VERSION]->(v:DDVersion)
                RETURN p.id AS id,
                       p.lifecycle_status AS lifecycle_status,
                       iv.id AS introduced_in,
                       dv.id AS deprecated_in,
                       count(change) AS change_count,
                       collect({
                           version: v.id,
                           change_type: change.change_type,
                           semantic_type: change.semantic_type,
                           old_value: change.old_value,
                           new_value: change.new_value
                       })[..20] AS changes
                """,
                path_ids=path_list,
            )
        except Exception as e:
            return {"error": f"Failed to query version context: {e}", "paths": {}}

        path_ctx: dict[str, Any] = {}
        for r in results or []:
            # Filter out null entries produced when OPTIONAL MATCH finds nothing
            changes = [
                c for c in (r.get("changes") or []) if c.get("version") is not None
            ]
            path_ctx[r["id"]] = {
                "lifecycle_status": r.get("lifecycle_status"),
                "introduced_in": r.get("introduced_in"),
                "deprecated_in": r.get("deprecated_in"),
                "change_count": len(changes),
                "changes": changes,
            }

        # Report paths not found in graph
        found = set(path_ctx.keys())
        not_found = [p for p in path_list if p not in found]
        paths_without_changes = sorted(
            path_id for path_id, ctx in path_ctx.items() if ctx["change_count"] == 0
        )
        graph_change_nodes_seen = sum(ctx["change_count"] for ctx in path_ctx.values())

        return {
            "paths": path_ctx,
            "total_paths": len(path_list),
            "paths_found": sorted(found),
            "paths_without_changes": paths_without_changes,
            "paths_with_changes": sum(
                1 for v in path_ctx.values() if v["change_count"] > 0
            ),
            "graph_change_nodes_seen": graph_change_nodes_seen,
            "not_found": not_found,
        }

    async def _rename_chain_query(
        self, ids_filter: str | None = None
    ) -> dict[str, Any]:
        """Traverse RENAMED_TO edges to return multi-hop rename lineages.

        Finds all root nodes (no incoming RENAMED_TO) and follows the chain
        up to 10 hops deep, returning the full path lineage per chain.

        Args:
            ids_filter: Restrict chains to paths belonging to this IDS.

        Returns:
            Dict with 'rename_chains' list of {chain, hops} dicts.
        """
        if ids_filter:
            cypher = """
            MATCH chain = (old:IMASNode)-[:RENAMED_TO*1..10]->(current:IMASNode)
            WHERE old.ids = $ids_filter
              AND NOT EXISTS { ()-[:RENAMED_TO]->(old) }
            RETURN [n IN nodes(chain) | n.id] AS rename_chain,
                   length(chain) AS hops
            ORDER BY hops DESC
            """
            params: dict[str, Any] = {"ids_filter": ids_filter}
        else:
            cypher = """
            MATCH chain = (old:IMASNode)-[:RENAMED_TO*1..10]->(current:IMASNode)
            WHERE NOT EXISTS { ()-[:RENAMED_TO]->(old) }
            RETURN [n IN nodes(chain) | n.id] AS rename_chain,
                   length(chain) AS hops
            ORDER BY hops DESC
            """
            params = {}

        try:
            rows = self.graph_client.query(cypher, **params)
        except Exception as e:
            return {"error": f"Failed to query rename chains: {e}", "rename_chains": []}

        chains = [
            {"chain": list(r["rename_chain"]), "hops": r["hops"]}
            for r in (rows or [])
            if r.get("rename_chain")
        ]
        return {"rename_chains": chains}

    @mcp_tool(
        "Rank IMAS Data Dictionary paths by how much they have changed across DD versions. "
        "Returns a volatility-scored table showing which paths changed most often, "
        "what types of changes occurred, and whether they were renamed. "
        "ids_filter: restrict to one IDS. "
        "from_version: start of version range (exclusive). "
        "to_version: end of version range (inclusive). "
        "limit: max results (default 50)."
    )
    async def get_dd_changelog(
        self,
        ids_filter: str | None = None,
        from_version: str | None = None,
        to_version: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Rank paths by change frequency and type diversity across DD versions.

        Returns a volatility-scored ranking. Volatility score = change_count
        + (type_variety * 2) + (was_renamed * 3).

        Args:
            ids_filter: Restrict to one IDS (e.g. 'equilibrium').
            from_version: Start of version range filter (exclusive).
            to_version: End of version range filter (inclusive).
            limit: Maximum number of results to return (default 50).

        Returns:
            Dict with 'results' list and metadata.
        """
        cypher = """
        MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE ($ids_filter IS NULL OR p.ids = $ids_filter)
          AND ($from_version IS NULL OR change.version > $from_version)
          AND ($to_version IS NULL OR change.version <= $to_version)
        WITH p, count(change) AS change_count,
             count(DISTINCT change.semantic_change_type) AS type_variety,
             collect(DISTINCT change.semantic_change_type) AS change_types
        OPTIONAL MATCH (p)-[:RENAMED_TO]->()
        WITH p, change_count, type_variety, change_types,
             CASE WHEN EXISTS { (p)-[:RENAMED_TO]->() } THEN 1 ELSE 0 END AS was_renamed
        RETURN p.id AS path, p.ids AS ids, p.lifecycle_status AS lifecycle_status,
               change_count, type_variety, change_types, was_renamed,
               change_count + (type_variety * 2) + (was_renamed * 3) AS volatility_score
        ORDER BY volatility_score DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "ids_filter": ids_filter,
            "from_version": from_version,
            "to_version": to_version,
            "limit": limit,
        }

        try:
            rows = self.graph_client.query(cypher, **params)
        except Exception as e:
            return {"error": f"Failed to query changelog: {e}"}

        results = [dict(r) for r in (rows or [])]

        version_range: dict[str, str] | None = None
        if from_version or to_version:
            version_range = {
                "from": from_version or "",
                "to": to_version or "",
            }

        return {
            "results": results,
            "total": len(results),
            "ids_filter": ids_filter,
            "version_range": version_range,
            "limit": limit,
        }

    async def _bulk_query(
        self,
        change_type_filter: str,
        ids_filter: str | None,
        from_version: str | None,
        to_version: str | None,
    ) -> dict[str, Any]:
        """Execute a bulk query for all changes of a specific type."""
        # Build WHERE clauses
        where_clauses = ["c.change_type = $change_type"]
        params: dict[str, Any] = {"change_type": change_type_filter}

        if from_version:
            where_clauses.append("v.id > $from_version")
            params["from_version"] = from_version
        if to_version:
            where_clauses.append("v.id <= $to_version")
            params["to_version"] = to_version

        ids_where = ""
        if ids_filter:
            ids_where = "WHERE p.ids = $ids_filter"
            params["ids_filter"] = ids_filter

        where_str = "WHERE " + " AND ".join(where_clauses)

        cypher = f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        {where_str}
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        {ids_where}
        RETURN p.id AS path, p.ids AS ids,
               c.old_value AS old_value, c.new_value AS new_value,
               v.id AS version, c.change_type AS change_type,
               coalesce(c.breaking_level, 'informational') AS severity
        ORDER BY v.id, p.ids, p.id
        LIMIT 200
        """

        try:
            rows = self.graph_client.query(cypher, **params)
        except Exception as e:
            return {"error": f"Failed to execute bulk query: {e}"}

        changes = [dict(r) for r in (rows or [])]
        ids_affected = sorted({r["ids"] for r in changes if r.get("ids")})
        version_range: dict[str, str] | None = None
        if from_version or to_version:
            version_range = {
                "from": from_version or "",
                "to": to_version or "",
            }

        return {
            "mode": "bulk_query",
            "change_type_filter": change_type_filter,
            "change_count": len(changes),
            "changes": changes,
            "ids_affected": ids_affected,
            "version_range": version_range,
        }
