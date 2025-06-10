"""
A simple entry point for the imas-mcp command.
"""

if __name__ == "__main__":
    from .mcp_imas import run_server

    run_server(transport="streamable-http", host="127.0.0.1", port=8000)
