"""
IMAS Codex Server Module Entry Point

This module provides the entry point when running the IMAS Codex server as a Python module.
The full CLI interface is exposed through this entry point.

Usage:
    python -m imas_codex [OPTIONS]

Options:
    --transport [stdio|sse|streamable-http]  Transport protocol [default: streamable-http]
    --host TEXT                              Host to bind to [default: 127.0.0.1]
    --port INTEGER                           Port to bind to [default: 8000]
    --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]  Log level [default: INFO]
    --help                                   Show help message

Examples:
    # Run with default streamable-http transport
    python -m imas_codex

    # Show all CLI options
    python -m imas_codex --help

    # Run with custom options
    python -m imas_codex --transport sse --port 8080 --log-level DEBUG

    # Run HTTP server for API access
    python -m imas_codex --transport streamable-http --host 0.0.0.0 --port 9000

See __main__.py in the root directory for complete documentation and examples.
"""

from imas_codex.cli import main

if __name__ == "__main__":
    # Expose the full Click CLI interface
    main()
