"""
IMAS MCP Server - Model Context Protocol Server for IMAS Data Dictionary

This server provides access to the IMAS (ITER Integrated Modelling & Analysis Suite)
Data Dictionary through the Model Context Protocol (MCP). It enables AI assistants
and other MCP clients to search, explore, and query IMAS data structures.

Command Line Interface:
    The server exposes a full CLI with options for transport, host, port, and logging.    Usage:
      python -m imas_mcp [OPTIONS]   # Module execution (recommended)
      imas-mcp [OPTIONS]             # Console script (after pip install)

    Options:
      --transport [stdio|sse|streamable-http]
                                      Transport protocol to use (stdio, sse, or
                                      streamable-http)  [default: stdio]
      --host TEXT                     Host to bind to (for sse and streamable-http
                                      transports)  [default: 127.0.0.1]
      --port INTEGER                  Port to bind to (for sse and streamable-http
                                      transports)  [default: 8000]
      --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                      Set the logging level  [default: INFO]
      --help                          Show this message and exit.

Usage Examples:
    # Module execution (recommended)    python -m imas_mcp                          # Default stdio transport
    python -m imas_mcp --help                   # Show help
    python -m imas_mcp --log-level DEBUG        # Debug logging    # After installation with pip, use console scripts
    imas-mcp                                     # Default stdio transport
    imas-mcp --transport sse --port 8080        # Server-Sent Events

    # Different transport protocols    python -m imas_mcp --transport stdio                    # Default: stdio for MCP clients
    python -m imas_mcp --transport sse --port 8080          # Server-Sent Events
    python -m imas_mcp --transport streamable-http --port 9000  # HTTP streaming

    # Different log levels    python -m imas_mcp --log-level DEBUG        # Verbose debugging output
    python -m imas_mcp --log-level WARNING      # Only warnings and errors

    # Custom host/port (for sse and streamable-http)
    python -m imas_mcp --transport sse --host 0.0.0.0 --port 8080

    # Multiple options
    python -m imas_mcp --transport streamable-http --host 0.0.0.0 --port 8080 --log-level DEBUG

Available Tools:
    The server provides the following MCP tools for querying IMAS data:

    - ids_names(): List all available IDS names
    - ids_info(): Get Data Dictionary metadata and version info
    - search_by_keywords(): Full-text search across documentation and paths
    - search_by_exact_path(): Look up specific IDS paths
    - search_by_path_prefix(): Find all paths under a given prefix
    - filter_search_results(): Filter search results by field criteria
    - get_ids_structure(): Explore hierarchical structure of an IDS
    - get_index_stats(): Get search index statistics
    - get_common_units(): Get unit usage statistics

Transport Protocols:
    - stdio: Standard input/output (default, recommended for MCP clients)
      * Use this for VS Code extensions and other MCP clients
      * Communication via stdin/stdout
      * No network configuration required

    - sse: Server-Sent Events over HTTP
      * Use for web applications requiring real-time updates
      * Accessible via HTTP GET requests
      * Supports browser-based clients

    - streamable-http: HTTP streaming protocol
      * Use for direct HTTP API access
      * Supports POST requests with JSON payloads
      * Good for testing and development

Installation & CLI:
    # Install from source
    pip install -e .    # Install from Git repository
    pip install git+https://github.com/your-repo/imas-mcp.git    # After installation, console scripts are available:
    imas-mcp --help

    # Module execution works without installation:
    python -m imas_mcp --help

    The CLI is built with Click and supports:
    - Full argument validation and type checking
    - Help documentation (--help)
    - Environment variable support
    - Shell completion (if configured)

Environment:
    The server loads IMAS Data Dictionary data from the local environment.
    Ensure IMAS is properly installed and configured before running.

    Environment variables (optional):
    - IMAS_MCP_HOST: Default host (overridden by --host)
    - IMAS_MCP_PORT: Default port (overridden by --port)
    - IMAS_MCP_LOG_LEVEL: Default log level (overridden by --log-level)

More Information:
    - IMAS Documentation: https://imas.iter.org/
    - MCP Protocol: https://modelcontextprotocol.io/
    - GitHub Repository: https://github.com/your-repo/imas-mcp
"""

if __name__ == "__main__":
    from imas_mcp.cli import main

    # Expose the Click CLI interface directly
    main()
