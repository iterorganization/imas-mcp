"""
IMAS MCP Server - Composable Integrator.

This is the principal MCP server for the IMAS data dictionary that uses
composition to combine tools and resources from separate providers.
This architecture enables clean separation of concerns and better maintainability.

The server integrates:
- Tools: 8 core tools for physics-based search and analysis
- Resources: Static JSON schema resources for reference data

Each component is accessible via server.tools and server.resources properties.
"""

import importlib.metadata
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import nest_asyncio
from fastmcp import FastMCP

from imas_mcp.embeddings.embeddings import Embeddings
from imas_mcp.health import HealthEndpoint
from imas_mcp.resource_provider import Resources
from imas_mcp.search.semantic_search import SemanticSearch as _ServerSemanticSearch
from imas_mcp.tools import Tools

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging with specific control over different components
# Note: Default to WARNING but allow CLI to override this
logging.basicConfig(
    level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s"
)

# Set our application logger to WARNING for stdio transport to prevent
# INFO messages from appearing as warnings in MCP clients
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Backward compatibility for tests that patch imas_mcp.server.SemanticSearch
# The semantic search initialization logic moved to the Embeddings dataclass,
# but tests still reference this symbol on the server module for mocking.
SemanticSearch = _ServerSemanticSearch  # type: ignore

# Suppress FastMCP startup messages by setting to ERROR level
# This prevents the "Starting MCP server" message from appearing as a warning
fastmcp_server_logger = logging.getLogger("FastMCP.fastmcp.server.server")
fastmcp_server_logger.setLevel(logging.ERROR)

# General FastMCP logger can stay at WARNING
fastmcp_logger = logging.getLogger("FastMCP")
fastmcp_logger.setLevel(logging.WARNING)


@dataclass
class Server:
    """IMAS MCP Server - Composable integrator using composition pattern."""

    # Configuration parameters
    ids_set: set[str] | None = None
    use_rich: bool = True

    # Internal fields
    mcp: FastMCP = field(init=False, repr=False)
    tools: Tools = field(init=False, repr=False)
    resources: Resources = field(init=False, repr=False)
    embeddings: Embeddings = field(init=False, repr=False)
    started_at: datetime = field(init=False, repr=False)
    _started_monotonic: float = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        self.mcp = FastMCP(name="imas")

        # Initialize components
        self.tools = Tools(ids_set=self.ids_set)
        self.resources = Resources(ids_set=self.ids_set)
        # Compose embeddings manager
        self.embeddings = Embeddings(
            document_store=self.tools.document_store,
            ids_set=self.ids_set,
            use_rich=self.use_rich,
        )

        # Register components with MCP server
        self._register_components()
        # Capture start times (wall clock + monotonic for stable uptime)
        self.started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()

        logger.debug("IMAS MCP Server initialized with tools and resources")

    def _register_components(self):
        """Register tools and resources with the MCP server."""
        logger.debug("Registering tools component")
        self.tools.register(self.mcp)

        logger.debug("Registering resources component")
        self.resources.register(self.mcp)

        logger.debug("Successfully registered all components")

    # Embedding initialization logic encapsulated in Embeddings dataclass (embeddings/embeddings.py)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Run the server with the specified transport.

        Args:
            transport: Transport protocol to use
            host: Host to bind to (for HTTP transports)
            port: Port to bind to (for HTTP transports)
        """
        # Adjust logging level based on transport
        # For stdio transport, suppress INFO logs to prevent them appearing as warnings in MCP clients
        # For HTTP transport, allow INFO logs for useful debugging information
        if transport == "stdio":
            logger.setLevel(logging.WARNING)
            logger.debug("Starting IMAS MCP server with stdio transport")
            self.mcp.run(transport=transport)
        elif transport in ["sse", "streamable-http"]:
            logger.setLevel(logging.INFO)
            logger.info(
                f"Starting IMAS MCP server with {transport} transport on {host}:{port}"
            )
            # Use stateful HTTP mode to support sampling functionality
            logger.info("Using stateful HTTP mode to support MCP sampling")
            # Attach minimal /health endpoint (same port) for HTTP transports
            try:
                HealthEndpoint(self).attach()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to attach /health: {e}")
            self.mcp.run(
                transport=transport, host=host, port=port, stateless_http=False
            )
        else:
            raise ValueError(
                f"Unsupported transport: {transport}. "
                f"Supported transports: stdio, sse, streamable-http"
            )

    def _get_version(self) -> str:
        """Get the package version."""
        try:
            return importlib.metadata.version("imas-mcp")
        except Exception:
            return "unknown"

    def uptime_seconds(self) -> float:
        """Return process uptime in seconds using monotonic clock."""
        try:
            return max(0.0, time.monotonic() - self._started_monotonic)
        except Exception:  # pragma: no cover - defensive
            return 0.0


def main():
    """Run the server with stdio transport."""
    server = Server()
    server.run(transport="stdio")


def run_server(
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """
    Entry point for running the server with specified transport.

    Args:
        transport: Either 'stdio', 'sse', or 'streamable-http'
        host: Host for HTTP transport
        port: Port for HTTP transport
    """
    server = Server()
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
