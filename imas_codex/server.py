"""
IMAS Codex Server - Graph-Native MCP Server.

MCP server for the IMAS data dictionary backed by Neo4j.
All data — paths, clusters, versions, embeddings — lives in the graph.

The server integrates:
- Tools: 10 graph-backed tools for search, exploration, and schema introspection
- Resources: Usage examples for MCP clients
"""

import importlib.metadata
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import nest_asyncio
from fastmcp import FastMCP

from imas_codex.graph.client import GraphClient
from imas_codex.health import HealthEndpoint
from imas_codex.resource_provider import Resources
from imas_codex.tools import Tools

# apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

logging.basicConfig(
    level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Suppress FastMCP startup messages
logging.getLogger("FastMCP.fastmcp.server.server").setLevel(logging.ERROR)
logging.getLogger("FastMCP").setLevel(logging.WARNING)


def _connect_graph() -> GraphClient:
    """Connect to Neo4j. Raises RuntimeError if unavailable."""
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")

    gc = GraphClient(uri=uri, username=username, password=password)
    gc.query("RETURN 1")
    logger.info(f"Connected to Neo4j at {uri}")
    return gc


def _server_name(graph_client: GraphClient) -> str:
    """Derive server name from DDVersion nodes in the graph."""
    try:
        result = graph_client.query(
            "MATCH (v:DDVersion {is_current: true}) RETURN v.id"
        )
        if result:
            return f"imas-data-dictionary-{result[0]['v.id']}"
    except Exception:
        pass
    return "imas-data-dictionary"


@dataclass
class Server:
    """IMAS Codex MCP Server backed by Neo4j."""

    ids_set: set[str] | None = None
    graph_client: GraphClient | None = None

    mcp: FastMCP = field(init=False, repr=False)
    tools: Tools = field(init=False, repr=False)
    resources: Resources = field(init=False, repr=False)
    started_at: datetime = field(init=False, repr=False)
    _started_monotonic: float = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server."""
        if self.graph_client is None:
            self.graph_client = _connect_graph()

        self.mcp = FastMCP(name=_server_name(self.graph_client))
        self.tools = Tools(ids_set=self.ids_set, graph_client=self.graph_client)
        self.resources = Resources()

        self.tools.register(self.mcp)
        self.resources.register(self.mcp)

        self.started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Run the server with the specified transport."""
        if transport == "stdio":
            logger.setLevel(logging.WARNING)
            self.mcp.run(transport=transport)
        elif transport in ["sse", "streamable-http"]:
            logger.setLevel(logging.INFO)
            logger.info(f"Starting IMAS Codex server with {transport} on {host}:{port}")
            try:
                HealthEndpoint(self).attach()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to attach /health: {e}")
            self.mcp.run(transport=transport, host=host, port=port)
        else:
            raise ValueError(
                f"Unsupported transport: {transport}. "
                f"Supported: stdio, sse, streamable-http"
            )

    def _get_version(self) -> str:
        try:
            return importlib.metadata.version("imas-codex")
        except Exception:
            return "unknown"

    def uptime_seconds(self) -> float:
        try:
            return max(0.0, time.monotonic() - self._started_monotonic)
        except Exception:  # pragma: no cover - defensive
            return 0.0


def main():
    """Run the server with streamable-http transport."""
    server = Server()
    server.run(transport="streamable-http")


def run_server(
    transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Entry point for running the server with specified transport."""
    server = Server()
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
