"""Tests for MCP tool schema compatibility.

Validates that tool definitions emitted by the server conform to MCP protocol
constraints and are compatible with clients negotiating older protocol versions
(e.g., GitHub Copilot using 2024-11-05).
"""

from __future__ import annotations

import json

import pytest

from imas_codex.llm.server import AgentsServer

SN_TOOLS = {"search_standard_names", "fetch_standard_names", "list_standard_names"}


def _tool_names(server: AgentsServer) -> set[str]:
    return {
        k.removeprefix("tool:").rstrip("@")
        for k in server.mcp._local_provider._components
        if k.startswith("tool:")
    }


@pytest.fixture(scope="module")
def server():
    """Create an AgentsServer in DD-only mode (no Neo4j needed for schema checks)."""
    return AgentsServer(dd_only=True)


@pytest.fixture(scope="module")
def tool_schemas(server) -> list[dict]:
    """Get MCP tool definitions as dicts."""
    tools = []
    for key, component in server.mcp._local_provider._components.items():
        if key.startswith("tool:"):
            mcp_tool = component.to_mcp_tool()
            tools.append(mcp_tool.model_dump(exclude_none=True))
    return tools


@pytest.fixture(scope="module")
def rw_server():
    """Create a read-write AgentsServer for testing all tools including REPL."""
    return AgentsServer(read_only=False, dd_only=False)


@pytest.fixture(scope="module")
def rw_tool_schemas(rw_server) -> list[dict]:
    """Get MCP tool definitions from read-write server."""
    tools = []
    for key, component in rw_server.mcp._local_provider._components.items():
        if key.startswith("tool:"):
            mcp_tool = component.to_mcp_tool()
            tools.append(mcp_tool.model_dump(exclude_none=True))
    return tools


class TestToolDescriptions:
    """Every tool must have a non-empty description."""

    def test_all_tools_have_descriptions(self, tool_schemas):
        missing = [t["name"] for t in tool_schemas if not t.get("description")]
        assert not missing, f"Tools missing description: {missing}"

    def test_rw_tools_have_descriptions(self, rw_tool_schemas):
        missing = [t["name"] for t in rw_tool_schemas if not t.get("description")]
        assert not missing, f"Tools missing description: {missing}"

    def test_python_tool_has_description(self, rw_tool_schemas):
        """Regression: FastMCP 3.0 captures docstrings eagerly at decoration time."""
        repl_tools = [t for t in rw_tool_schemas if t["name"] == "repl"]
        assert repl_tools, "repl tool not found"
        desc = repl_tools[0].get("description", "")
        assert len(desc) > 100, f"repl tool description too short: {len(desc)} chars"
        assert "REPL" in desc, "repl tool description should mention REPL"


class TestOutputSchema:
    """outputSchema must not be present — incompatible with MCP 2024-11-05."""

    def test_no_output_schema(self, tool_schemas):
        offenders = [t["name"] for t in tool_schemas if "outputSchema" in t]
        assert not offenders, (
            f"Tools with outputSchema (breaks MCP 2024-11-05 clients): {offenders}"
        )

    def test_no_output_schema_rw(self, rw_tool_schemas):
        offenders = [t["name"] for t in rw_tool_schemas if "outputSchema" in t]
        assert not offenders, (
            f"Tools with outputSchema (breaks MCP 2024-11-05 clients): {offenders}"
        )


class TestInputSchema:
    """inputSchema must be well-formed JSON Schema."""

    def test_all_have_input_schema(self, tool_schemas):
        missing = [t["name"] for t in tool_schemas if "inputSchema" not in t]
        assert not missing, f"Tools missing inputSchema: {missing}"

    def test_input_schemas_are_objects(self, tool_schemas):
        bad = [
            t["name"]
            for t in tool_schemas
            if t.get("inputSchema", {}).get("type") != "object"
        ]
        assert not bad, f"Tools with non-object inputSchema: {bad}"

    def test_input_schemas_serializable(self, tool_schemas):
        """All input schemas must be JSON-serializable."""
        for t in tool_schemas:
            try:
                json.dumps(t["inputSchema"])
            except (TypeError, ValueError) as e:
                pytest.fail(f"Tool {t['name']} inputSchema not serializable: {e}")


class TestToolPayloadSize:
    """Guard against oversized tool payloads that may be rejected by APIs."""

    MAX_SINGLE_TOOL_BYTES = 10_000
    MAX_TOTAL_PAYLOAD_BYTES = 200_000

    def test_individual_tool_size(self, rw_tool_schemas):
        oversized = []
        for t in rw_tool_schemas:
            size = len(json.dumps(t))
            if size > self.MAX_SINGLE_TOOL_BYTES:
                oversized.append(f"{t['name']} ({size} bytes)")
        assert not oversized, f"Oversized tools: {oversized}"

    def test_total_payload_size(self, rw_tool_schemas):
        total = sum(len(json.dumps(t)) for t in rw_tool_schemas)
        assert total < self.MAX_TOTAL_PAYLOAD_BYTES, (
            f"Total tools payload {total} bytes exceeds {self.MAX_TOTAL_PAYLOAD_BYTES}"
        )


class TestIncludeStandardNames:
    """Standard name tools must only be registered when --include-standard-names is set."""

    def test_sn_tools_absent_by_default(self):
        """Standard name tools must NOT appear unless include_standard_names=True."""
        server = AgentsServer(dd_only=True, include_standard_names=False)
        names = _tool_names(server)
        present = SN_TOOLS & names
        assert not present, (
            f"Standard name tools registered by default (should be opt-in): {present}"
        )

    def test_sn_tools_absent_in_rw_server(self):
        """Standard name tools must NOT appear in the default read-write server."""
        server = AgentsServer(
            read_only=False, dd_only=False, include_standard_names=False
        )
        names = _tool_names(server)
        present = SN_TOOLS & names
        assert not present, (
            f"Standard name tools registered by default (should be opt-in): {present}"
        )

    def test_sn_tools_present_when_flag_set(self):
        """All three standard name tools must be registered when include_standard_names=True."""
        server = AgentsServer(dd_only=True, include_standard_names=True)
        names = _tool_names(server)
        missing = SN_TOOLS - names
        assert not missing, f"Expected standard name tools not registered: {missing}"
