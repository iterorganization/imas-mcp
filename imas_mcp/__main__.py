"""
IMAS MCP Server Module Entry Point

This module provides the entry point when running the IMAS MCP server as a Python module.
The full CLI interface is exposed through this entry point.

Usage:
    python -m imas_mcp [OPTIONS]

Options:
    --transport [stdio|sse|streamable-http]  Transport protocol [default: stdio]
    --host TEXT                              Host to bind to [default: 127.0.0.1]
    --port INTEGER                           Port to bind to [default: 8000]
    --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]  Log level [default: INFO]
    --help                                   Show help message

Examples:
    # Run with default stdio transport
    python -m imas_mcp

    # Show all CLI options
    python -m imas_mcp --help

    # Run with custom options
    python -m imas_mcp --transport sse --port 8080 --log-level DEBUG

    # Run HTTP server for API access
    python -m imas_mcp --transport streamable-http --host 0.0.0.0 --port 9000

See __main__.py in the root directory for complete documentation and examples.
"""

from imas_mcp.cli import main

if __name__ == "__main__":
    # Expose the full Click CLI interface
    main()
