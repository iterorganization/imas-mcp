"""
Agents MCP Server - Lightweight server for facility exploration prompts.

This server provides MCP prompts that enable the Commander LLM to orchestrate
specialist subagents for remote facility exploration.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from fastmcp import FastMCP

from imas_codex.discovery import get_config, list_facilities
from imas_codex.discovery.prompts import get_prompt_loader

logger = logging.getLogger(__name__)


@dataclass
class AgentsServer:
    """
    MCP server providing prompts for remote facility exploration.

    This is a lightweight server that serves prompts only - no tools.
    The prompts guide the Commander LLM to orchestrate subagents.
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
        def explore_files(facility: str, path: str = "/") -> str:
            """
            Explore filesystem structure on a remote facility.

            Generates a prompt that guides the Commander LLM to orchestrate
            the File Explorer subagent for mapping directory structures.

            Args:
                facility: Facility identifier (e.g., 'epfl')
                path: Starting path to explore (default: /)
            """
            try:
                config = get_config(facility)
                prompts = get_prompt_loader()

                # Load common agent instructions
                common_prompt = prompts.load("common")
                common_instructions = common_prompt.render(**config.to_context())

                # Load file explorer specialist instructions
                explorer_prompt = prompts.load("file_explorer")
                specialist_instructions = explorer_prompt.render(
                    **config.to_context(),
                    target_path=path,
                )

                # Compose the full prompt
                return f"""# File Explorer Agent Task

## Facility
- **Name**: {config.facility}
- **Description**: {config.description}
- **SSH Host**: {config.ssh_host}

## Target Path
{path}

---

{common_instructions}

---

{specialist_instructions}
"""
            except ValueError as e:
                available = list_facilities()
                return (
                    f"Error: {e}\n\n"
                    f"Available facilities: {', '.join(available) if available else 'none configured'}"
                )

        @self.mcp.prompt()
        def list_agents() -> str:
            """
            List available exploration agents and their capabilities.

            Returns information about the specialist subagents that can be
            dispatched for remote facility exploration.
            """
            return """# Available Exploration Agents

## File Explorer Agent
**Status**: Available
**Purpose**: Map filesystem structure on remote facilities
**Prompt**: `/explore_files`
**Capabilities**:
- Navigate directory hierarchies
- Identify file types and patterns
- Discover code repositories and data locations
- Report filesystem structure as JSON

## Code Search Agent
**Status**: Planned
**Purpose**: Search for code patterns across facility codebases
**Prompt**: `/search_code` (coming soon)

## Data Inspector Agent
**Status**: Planned
**Purpose**: Inspect data files (HDF5, NetCDF, MDSplus)
**Prompt**: `/inspect_data` (coming soon)

## Environment Probe Agent
**Status**: Planned
**Purpose**: Discover system capabilities and installed tools
**Prompt**: `/probe_environment` (coming soon)

---

To use an agent, invoke its prompt with the facility name and parameters.
"""

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
