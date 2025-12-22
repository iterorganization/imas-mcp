"""
Agents MCP Server - Prompts for LLM-driven facility exploration.

This server provides MCP prompts that inject context and instructions
for exploring remote fusion facilities. The Cursor chat LLM orchestrates
exploration via terminal commands.

No subagents. No tools. Just prompt-driven LLM orchestration.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from fastmcp import FastMCP

from imas_codex.discovery import get_config, list_facilities
from imas_codex.discovery.prompts import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentsServer:
    """
    MCP server for remote facility exploration prompts.

    Provides prompts that teach the Cursor chat LLM how to explore
    remote facilities via terminal commands. The LLM:

    1. Runs commands via `uv run imas-codex <facility> "cmd"`
    2. Observes outputs
    3. Decides what's worth learning
    4. Persists learnings via `--finish`
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
        def explore(facility: str) -> str:
            """
            Explore a remote fusion facility via SSH.

            Injects facility context and instructions for LLM-driven exploration.
            The LLM runs terminal commands to discover the environment and
            persists learnings when done.

            Args:
                facility: Facility identifier (e.g., 'epfl', 'ipp')
            """
            # Validate facility exists
            available = list_facilities()
            if facility not in available:
                return f"""# Error: Unknown Facility

Facility '{facility}' not found.

Available facilities: {", ".join(available)}

Use one of the available facilities, e.g., `/explore epfl`
"""

            config = get_config(facility)

            # Load knowledge from facility config
            knowledge = None
            try:
                from imas_codex.remote.finish import load_facility_yaml

                raw_config = load_facility_yaml(facility)
                knowledge = raw_config.get("knowledge", {})
                if not knowledge or not any(v for v in knowledge.values() if v):
                    knowledge = None
            except Exception:
                pass

            # Build context for template
            context = {
                "facility": facility,
                "description": config.description,
                "ssh_host": config.ssh_host,
                "knowledge": knowledge,
                "paths": config.paths.model_dump(),
                "known_systems": config.known_systems.model_dump(),
                "exploration_hints": config.exploration_hints,
            }

            # Load and render the explore template
            prompt = load_prompt("agents/explore")
            return prompt.render(**context)

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
                    lines.append(f"To explore: `/explore {name}`\n")
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
