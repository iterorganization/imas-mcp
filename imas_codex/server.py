"""
IMAS Codex Server - Composable Integrator.

This is the principal MCP server for the IMAS data dictionary that uses
composition to combine tools and resources from separate providers.
This architecture enables clean separation of concerns and better maintainability.

The server integrates
- Tools: 8 core tools for physics-based search and analysis
- Resources: Static JSON schema resources for reference data

Each component is accessible via server.tools and server.resources properties.
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

from imas_codex import dd_version
from imas_codex.embeddings.embeddings import Embeddings
from imas_codex.health import HealthEndpoint
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.resource_provider import Resources
from imas_codex.search.semantic_search import SemanticSearch as _ServerSemanticSearch
from imas_codex.services.docs_server_manager import DocsServerManager
from imas_codex.tools import Tools

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

# Backward compatibility for tests that patch imas_codex.server.SemanticSearch
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
    """IMAS Codex Server - Composable integrator using composition pattern."""

    # Configuration parameters
    ids_set: set[str] | None = None
    use_rich: bool = True
    start_docs_server: bool = True  # Flag to control docs server startup

    # Internal fields
    mcp: FastMCP = field(init=False, repr=False)
    tools: Tools = field(init=False, repr=False)
    resources: Resources = field(init=False, repr=False)
    embeddings: Embeddings = field(init=False, repr=False)
    docs_manager: DocsServerManager = field(init=False, repr=False)
    started_at: datetime = field(init=False, repr=False)
    _started_monotonic: float = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        # Include DD version in server name so clients/LLMs know which version they're using
        server_name = f"imas-data-dictionary-{dd_version}"
        self.mcp = FastMCP(name=server_name)

        # Validate schemas exist before initialization (fail fast)
        self._validate_schemas_available()

        # Initialize docs manager first
        self.docs_manager = DocsServerManager()

        # Initialize components with dependency injection
        self.tools = Tools(ids_set=self.ids_set, docs_manager=self.docs_manager)
        self.resources = Resources(ids_set=self.ids_set)
        # Compose embeddings manager
        self.embeddings = Embeddings(
            document_store=self.tools.document_store,
            ids_set=self.ids_set,
            use_rich=self.use_rich,
        )

        # Register components with MCP server
        self._register_components()

        logger.debug("IMAS Codex Server initialized with tools and resources")

        # Capture start times (wall clock + monotonic for stable uptime)
        self.started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown including docs server."""
        if hasattr(self.docs_manager, "setup_signal_handlers"):
            self.docs_manager.setup_signal_handlers()

    async def cleanup(self):
        """Clean up resources including docs server."""
        try:
            if hasattr(self, "docs_manager"):
                await self.docs_manager.stop_server()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup when Server instance is destroyed."""
        try:
            if hasattr(self, "docs_manager") and hasattr(self.docs_manager, "process"):
                # Force terminate the docs server process if it exists
                # We can't await async cleanup in __del__, so just terminate synchronously
                if self.docs_manager.process is not None:
                    try:
                        # Synchronous termination (no await)
                        self.docs_manager.process.terminate()
                        logger.debug("Terminated docs server process in __del__")
                    except Exception as e:
                        logger.debug(f"Could not terminate docs server in __del__: {e}")
        except Exception:
            pass  # Ignore errors during destruction

    # Context manager support
    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        await self.cleanup()

    def _register_components(self):
        """Register tools and resources with the MCP server."""
        logger.debug("Registering tools component")
        self.tools.register(self.mcp)

        logger.debug("Registering resources component")
        self.resources.register(self.mcp)

        logger.debug("Successfully registered all components")

    def _build_schemas_if_missing(self) -> bool:
        """Automatically build schemas if they're missing.

        Returns:
            True if schemas exist or were built successfully, False otherwise.
        """
        path_accessor = ResourcePathAccessor(dd_version=dd_version)
        catalog_path = path_accessor.schemas_dir / "ids_catalog.json"
        detailed_dir = path_accessor.schemas_dir / "detailed"

        # Check if schemas already exist and are complete
        if (
            catalog_path.exists()
            and detailed_dir.exists()
            and list(detailed_dir.glob("*.json"))
        ):
            return True

        # Schemas are missing - attempt to build them
        logger.info(
            f"Schemas not found for DD version '{dd_version}'. Building schemas automatically..."
        )

        try:
            from imas_codex.core.xml_parser import DataDictionaryTransformer
            from imas_codex.dd_accessor import ImasDataDictionariesAccessor

            # Create DD accessor
            dd_accessor = ImasDataDictionariesAccessor(dd_version)

            # Build schemas (this is what the build hook does)
            logger.info(
                f"Building schemas for DD version {dd_version}. This may take a minute..."
            )

            json_transformer = DataDictionaryTransformer(
                dd_accessor=dd_accessor, ids_set=self.ids_set, use_rich=False
            )
            json_transformer.build()

            logger.info(f"✓ Schemas built successfully for DD version '{dd_version}'")
            return True

        except Exception as e:
            logger.error(f"Failed to auto-build schemas: {e}")
            return False

    def _validate_schemas_available(self):
        """Validate that schema files exist for the current DD version.

        Attempts to auto-build schemas if they're missing (useful for editable installs).

        Raises:
            RuntimeError: If required schema files are missing and cannot be built.
        """
        # Try to auto-build schemas if missing
        if self._build_schemas_if_missing():
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            detailed_dir = path_accessor.schemas_dir / "detailed"
            detailed_files = list(detailed_dir.glob("*.json"))
            logger.debug(
                f"Schema validation passed: {len(detailed_files)} IDS schemas found for DD version '{dd_version}'"
            )
            return

        # Auto-build failed - provide helpful error message
        path_accessor = ResourcePathAccessor(dd_version=dd_version)
        is_dev_version = "dev" in dd_version.lower()
        build_cmd = "dd-version dev" if is_dev_version else f"dd-version {dd_version}"

        imas_dd_version_env = os.environ.get("IMAS_DD_VERSION", "(not set)")
        ids_filter_env = os.environ.get("IDS_FILTER", "(not set)")

        error_msg = (
            f"\n\n"
            f"Environment variables:\n"
            f"  IMAS_DD_VERSION: {imas_dd_version_env}\n"
            f"  IDS_FILTER: {ids_filter_env}\n\n"
            f"Schema files not found for DD version '{dd_version}'.\n"
            f"Expected location: {path_accessor.schemas_dir}\n\n"
            f"Auto-build failed. To build schemas manually, run:\n"
            f"  {build_cmd}\n\n"
            f"To list all available versions:\n"
            f"  dd-version --list\n\n"
            f"To use a different DD version:\n"
            f"  dd-version <version>  # e.g., dd-version 3.42.2\n"
            f"  dd-version dev        # for development version\n"
        )
        raise RuntimeError(error_msg)

    # Embedding initialization logic encapsulated in Embeddings dataclass (embeddings/embeddings.py)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
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
            logger.debug("Starting IMAS Codex server with stdio transport")
            self.mcp.run(transport=transport)
        elif transport in ["sse", "streamable-http"]:
            logger.setLevel(logging.INFO)
            logger.info(
                f"Starting IMAS Codex server with {transport} transport on {host}:{port}"
            )
            # Attach minimal /health endpoint (same port) for HTTP transports
            try:
                HealthEndpoint(self).attach()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to attach /health: {e}")
            self.mcp.run(transport=transport, host=host, port=port)
        else:
            raise ValueError(
                f"Unsupported transport: {transport}. "
                f"Supported transports: stdio, sse, streamable-http"
            )

    def _get_version(self) -> str:
        """Get the package version."""
        try:
            return importlib.metadata.version("imas-codex")
        except Exception:
            return "unknown"

    def uptime_seconds(self) -> float:
        """Return process uptime in seconds using monotonic clock."""
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
