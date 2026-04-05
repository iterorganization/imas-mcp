"""Server cold-start benchmarks.

Measures MCP server initialization time — critical for container
boot and first-request latency.
"""

from __future__ import annotations


class ServerStartupBenchmarks:
    """Benchmark AgentsServer construction and first tool call."""

    timeout = 120
    number = 1
    repeat = 3
    warmup_time = 0

    def time_server_cold_start(self):
        """Full server construction from scratch."""
        from imas_codex.llm.server import AgentsServer

        AgentsServer()

    def time_mcp_tool_registration(self):
        """Tool registration via _register_tools()."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer.__new__(AgentsServer)
        from fastmcp import FastMCP

        server.mcp = FastMCP("imas-codex-bench")
        server.read_only = False
        server._register_tools()

    def time_first_tool_call(self):
        """First search_dd_paths after cold start (lazy init triggers)."""
        import asyncio

        from fastmcp import Client

        from imas_codex.llm.server import AgentsServer

        srv = AgentsServer()
        client = Client(srv.mcp)

        async def _call():
            async with client:
                return await client.call_tool(
                    "search_dd_paths", {"query": "temperature", "k": 1}
                )

        asyncio.run(_call())
