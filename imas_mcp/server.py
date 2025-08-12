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
from dataclasses import dataclass, field

import nest_asyncio
from fastmcp import FastMCP

# Import Resources from the resource_provider module
from imas_mcp.resource_provider import Resources
from imas_mcp.search.semantic_search import SemanticSearch, SemanticSearchConfig
from imas_mcp.tools import Tools

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging with specific control over different components
logging.basicConfig(
    level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s"
)

# Set our application logger to WARNING for stdio transport to prevent
# INFO messages from appearing as warnings in MCP clients
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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

    # Internal fields
    mcp: FastMCP = field(init=False, repr=False)
    tools: Tools = field(init=False, repr=False)
    resources: Resources = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        self.mcp = FastMCP(name="imas")

        # Initialize components
        self.tools = Tools(ids_set=self.ids_set)
        self.resources = Resources()

        # Pre-build embeddings during server initialization to avoid delays on
        # first request
        self._initialize_embeddings()

        # Register components with MCP server
        self._register_components()

        logger.debug("IMAS MCP Server initialized with tools and resources")

    def _register_components(self):
        """Register tools and resources with the MCP server."""
        logger.debug("Registering tools component")
        self.tools.register(self.mcp)

        logger.debug("Registering resources component")
        self.resources.register(self.mcp)

        logger.debug("Successfully registered all components")

    def _initialize_embeddings(self):
        """Pre-build embeddings during server initialization to avoid delays."""
        try:
            # Create semantic search configuration matching the server's ids_set
            config = SemanticSearchConfig(ids_set=self.ids_set)
            semantic_search = SemanticSearch(
                config=config, document_store=self.tools.document_store
            )

            # Force initialization which will build/load embeddings
            semantic_search._initialize()

        except Exception as e:
            logger.warning(f"Could not pre-build embeddings during startup: {e}")
            logger.warning("Embeddings will be built on first semantic search request")

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
        """Run the server with the specified transport."""
        # Adjust logging level based on transport
        # For stdio transport, suppress INFO logs to prevent them appearing as warnings in MCP clients
        # For HTTP transport, allow INFO logs for useful debugging information
        if transport == "stdio":
            logger.setLevel(logging.WARNING)
            logger.debug("Starting IMAS MCP server with stdio transport")
            self.mcp.run()
        elif transport == "http":
            logger.setLevel(logging.INFO)
            logger.info(
                f"Starting IMAS MCP server with HTTP transport on {host}:{port}"
            )
            self._run_http(host, port)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    def _run_http(self, host: str, port: int):
        """Run the server with HTTP transport."""
        try:
            import uvicorn

            # Note: HTTP transport import path may need adjustment based on FastMCP version
            from fastmcp.transports.http import create_app
        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        app = create_app(self.mcp)
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


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
    """
    Entry point for running the server with specified transport.

    Args:
        transport: Either 'stdio' or 'http'
        host: Host for HTTP transport
        port: Port for HTTP transport
    """
    server = Server()
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
