"""Neo4j graph health check utilities.

Provides structured health information for local and remote Neo4j instances,
complementing the embedding server's ``/health`` endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GraphHealth:
    """Health status of a Neo4j graph instance."""

    status: str  # "ok", "stopped", "unreachable"
    name: str
    location: str
    host: str | None
    bolt_port: int
    http_port: int
    bolt_url: str
    http_url: str
    node_count: int | None = None
    relationship_count: int | None = None
    store_size: str | None = None
    neo4j_version: str | None = None
    uptime: str | None = None
    facilities: list[str] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def check_graph_health(*, timeout: float = 5.0) -> GraphHealth:
    """Check Neo4j graph health and return structured status.

    Resolves the active graph profile then probes the Neo4j instance
    for liveness, statistics, and metadata.

    Args:
        timeout: Connection timeout in seconds.

    Returns:
        GraphHealth with status and available metadata.
    """
    from imas_codex.graph.profiles import resolve_neo4j

    try:
        profile = resolve_neo4j()
    except Exception as e:
        return GraphHealth(
            status="unreachable",
            name="unknown",
            location="unknown",
            host=None,
            bolt_port=7687,
            http_port=7474,
            bolt_url="bolt://localhost:7687",
            http_url="http://localhost:7474",
            error=str(e),
        )

    bolt_url = profile.uri
    http_url = f"http://localhost:{profile.http_port}"

    health = GraphHealth(
        status="stopped",
        name=profile.name,
        location=profile.location,
        host=profile.host,
        bolt_port=profile.bolt_port,
        http_port=profile.http_port,
        bolt_url=bolt_url,
        http_url=http_url,
    )

    # Quick HTTP liveness check
    try:
        import urllib.request

        urllib.request.urlopen(http_url + "/", timeout=timeout)
    except Exception:
        return health

    health.status = "ok"

    # Detailed stats via Bolt
    try:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient.from_profile()
        stats = gc.get_stats()
        health.node_count = stats.get("nodes")
        health.relationship_count = stats.get("relationships")

        # Graph meta
        try:
            from imas_codex.graph.meta import get_graph_meta

            meta = get_graph_meta(gc)
            if meta:
                health.facilities = meta.get("facilities") or []
        except Exception:
            pass

        # Neo4j version and store size from procedures
        try:
            result = gc.query(
                "CALL dbms.components() YIELD name, versions "
                "RETURN versions[0] AS version"
            )
            if result:
                health.neo4j_version = result[0].get("version")
        except Exception:
            pass

        gc.close()
    except Exception as e:
        health.error = f"Stats unavailable: {e}"

    return health


__all__ = ["GraphHealth", "check_graph_health"]
