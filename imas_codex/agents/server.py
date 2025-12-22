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

import yaml
from fastmcp import FastMCP

from imas_codex.discovery import get_config, list_facilities

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

            # Format knowledge section
            knowledge_yaml = ""
            try:
                from imas_codex.remote.finish import load_facility_yaml

                raw_config = load_facility_yaml(facility)
                knowledge = raw_config.get("knowledge", {})
                if knowledge and any(v for v in knowledge.values() if v):
                    knowledge_yaml = yaml.dump(
                        knowledge, default_flow_style=False, sort_keys=False
                    )
                else:
                    knowledge_yaml = "None yet - you're the first to explore!"
            except Exception:
                knowledge_yaml = "Unable to load knowledge section"

            # Format paths
            paths_yaml = yaml.dump(
                config.paths.model_dump(), default_flow_style=False, sort_keys=False
            )

            # Format known systems
            systems_yaml = yaml.dump(
                config.known_systems.model_dump(),
                default_flow_style=False,
                sort_keys=False,
            )

            # Format exploration hints
            hints = "\n".join(f"- {hint}" for hint in config.exploration_hints)

            return f"""# Exploring {config.description}

You are exploring the **{facility}** facility via SSH. Use terminal commands to discover the environment and persist what you learn.

## Commands

Execute commands on the remote facility:
```bash
uv run imas-codex {facility} "your command here"
```

Check what you've run in this session:
```bash
uv run imas-codex {facility} --status
```

When done, persist your learnings:
```bash
uv run imas-codex {facility} --finish << 'EOF'
python:
  version: "3.9.21"
  path: "/usr/bin/python3"
tools:
  rg: unavailable
  grep: available
paths:
  data_dir: /path/to/data
notes:
  - "Any freeform observations"
EOF
```

Or discard the session without saving:
```bash
uv run imas-codex {facility} --discard
```

## Current Knowledge

{knowledge_yaml}

## Known Paths

{paths_yaml}

## Known Systems

{systems_yaml}

## Exploration Hints

{hints}

## Exploration Guidelines

1. **Start with environment basics**: Python version, available tools (rg, grep, tree, find, h5dump, ncdump)
2. **Explore known paths**: Check what's in the data/code directories listed above
3. **Look for documentation**: README files, wikis, important scripts
4. **Test data access**: Try listing MDSplus trees, HDF5 files, etc.
5. **Note anything useful**: Paths, tool availability, data organization patterns

## Learning Categories

Use these categories in your `--finish` YAML:

- `python`: version, path, available packages
- `tools`: available/unavailable CLI tools (rg, grep, tree, h5dump, etc.)
- `paths`: important directories discovered
- `data`: data organization patterns, file formats
- `mdsplus`: tree names, server info, signal structure
- `notes`: freeform observations (as a list)

## Safety Notes

- All commands are validated for read-only operations
- Destructive commands (rm, mv, chmod, etc.) are blocked
- Large outputs are automatically truncated in logs
"""

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
