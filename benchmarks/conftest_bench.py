"""Shared benchmark infrastructure for ASV performance suites.

Provides lazy-loaded MCP fixtures, shared constants, and async helpers
that all benchmark modules import.
"""

from __future__ import annotations

import asyncio
from functools import cached_property


class MCPFixture:
    """Lazy-loaded MCP server + client for benchmarks."""

    @cached_property
    def server(self):
        from imas_codex.llm.server import AgentsServer

        return AgentsServer()

    @cached_property
    def client(self):
        from fastmcp import Client

        return Client(self.server.mcp)

    @cached_property
    def graph_client(self):
        from imas_codex.graph.client import GraphClient

        return GraphClient.from_profile()


_fixture = MCPFixture()


SEARCH_QUERIES: dict[str, str] = {
    "simple": "electron temperature",
    "multi_term": "magnetic field equilibrium boundary",
    "physics_specific": "poloidal flux gradient",
    "cross_ids": "temperature AND pressure profiles",
    "complex": "magnetic field topology near X-point separatrix",
}

IMAS_PATHS: dict[str, str] = {
    "leaf": "core_profiles/profiles_1d/electrons/temperature",
    "branch": "equilibrium/time_slice/profiles_1d",
    "ids": "equilibrium",
    "short": "core_profiles",
}

IDS_NAMES: dict[str, str] = {
    "small": "core_profiles",
    "large": "equilibrium",
    "domain": "magnetics",
}

UNIT_STRINGS: list[str] = [
    "eV",
    "m",
    "m^-3",
    "T",
    "Pa",
    "s",
    "A",
    "V",
    "W",
    "m.s^-1",
    "T.m^2",
    "keV",
    "m^-2.s^-1",
    "ohm.m",
    "-",
    "mixed",
    "dimensionless",
]


def run_tool(tool_name: str, arguments: dict | None = None):
    """Run an MCP tool call synchronously. Used by ASV benchmark methods."""

    async def _call():
        async with _fixture.client:
            return await _fixture.client.call_tool(tool_name, arguments or {})

    return asyncio.run(_call())
