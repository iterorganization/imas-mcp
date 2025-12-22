"""
Agents MCP Server - Prompts for LLM-driven facility exploration.

This server provides MCP prompts that inject context and pointers
for exploring remote fusion facilities. The LLM orchestrates
exploration via terminal commands and reads source files for details.

Minimal prompt approach: provide entry points and file locations,
let the agent discover details via --help and read_file.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from fastmcp import FastMCP

from imas_codex.discovery import get_config, list_facilities

logger = logging.getLogger(__name__)


@dataclass
class AgentsServer:
    """
    MCP server for remote facility exploration prompts.

    Provides minimal prompts that teach the LLM how to explore
    remote facilities via terminal commands. The LLM:

    1. Runs commands via `uv run imas-codex <facility> "cmd"`
    2. Uses --help to discover CLI options
    3. Reads Pydantic models for artifact schemas
    4. Captures learnings via `--capture`
    """

    mcp: FastMCP = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._register_prompts()
        logger.debug("Agents MCP server initialized")

    def _register_prompts(self):
        """Register exploration prompts."""

        @self.mcp.prompt()
        def explore() -> str:
            """
            Explore a remote fusion facility via SSH.

            Returns guidance pointer and facility list with SSH hosts.
            """
            facilities = list_facilities()

            lines = [
                "# Facility Exploration",
                "",
                "Read `imas_codex/config/README.md` for comprehensive guidance on:",
                "- Batch command patterns (critical for performance)",
                "- Category-specific exploration scripts",
                "- Safety rules",
                "- Capture syntax and examples",
                "",
                "## Available Facilities",
                "",
            ]

            if facilities:
                for name in sorted(facilities):
                    try:
                        config = get_config(name)
                        lines.append(f"### {name}")
                        lines.append(f"**{config.description}**")
                        lines.append("```bash")
                        lines.append(f'ssh {config.ssh_host} "your commands here"')
                        lines.append("```")
                        lines.append("")
                    except Exception:
                        lines.append(f"### {name}")
                        lines.append("(config error)")
                        lines.append("")
            else:
                lines.append("No facilities configured.")

            lines.extend(
                [
                    "## Artifact Schemas",
                    "",
                    "See `imas_codex/discovery/models/*.py` for Pydantic models:",
                    "- `environment.py` - Python, OS, compilers, modules",
                    "- `tools.py` - CLI tool availability",
                    "- `filesystem.py` - Directory structure, paths",
                    "- `data.py` - MDSplus, HDF5, NetCDF patterns",
                ]
            )

            return "\n".join(lines)

        @self.mcp.prompt()
        def list_facilities_prompt() -> str:
            """
            List all available facilities for exploration.
            """
            facilities = list_facilities()

            if not facilities:
                return "No facilities configured."

            lines = ["# Available Facilities\n"]

            for name in sorted(facilities):
                try:
                    config = get_config(name)
                    lines.append(f"## {name}")
                    lines.append(f"**{config.description}**\n")
                    lines.append(f"SSH Host: `{config.ssh_host}`\n")
                except Exception as e:
                    lines.append(f"## {name}")
                    lines.append(f"Error loading config: {e}\n")

            return "\n".join(lines)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8001,
    ):
        """Run the agents server."""
        if transport == "stdio":
            logger.debug("Starting Agents server with stdio transport")
            self.mcp.run(transport=transport)
        else:
            logger.info(f"Starting Agents server on {host}:{port}")
            self.mcp.run(transport=transport, host=host, port=port)
