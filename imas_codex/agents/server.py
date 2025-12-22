"""
Agents MCP Server - Server for facility exploration tools and prompts.

This server provides MCP tools that dispatch autonomous subagents for
remote facility exploration, plus prompts for documentation.

Supports streaming progress notifications for real-time visibility
into the subagent's train of thought during exploration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from fastmcp import Context, FastMCP

from imas_codex.agents.exploration import ExplorationAgent
from imas_codex.agents.knowledge import persist_learnings
from imas_codex.discovery import get_config, list_facilities

logger = logging.getLogger(__name__)


@dataclass
class AgentsServer:
    """
    MCP server for remote facility exploration.

    Provides:
    - `explore` tool: Dispatches autonomous subagents for natural language tasks
    - `list_facilities` tool: Lists available facility configurations
    - Prompts for documentation and guidance
    """

    mcp: FastMCP = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._register_tools()
        self._register_prompts()
        logger.debug("Agents MCP server initialized")

    def _register_tools(self):
        """Register exploration tools."""

        @self.mcp.tool()
        async def explore(
            facility: str,
            task: str,
            mode: str = "auto",
            max_iterations: int = 10,
            ctx: Context | None = None,
        ) -> dict[str, Any]:
            """
            Explore a remote facility with a natural language task.

            Dispatches an autonomous subagent that connects to the facility via SSH,
            runs a ReAct loop with a frontier LLM (Claude Opus 4.5), and returns
            structured findings.

            The agent uses accumulated knowledge from previous explorations and
            persists new discoveries for future runs.

            Progress notifications are streamed in real-time, showing the agent's
            train of thought (reasoning, script execution, observations).

            Args:
                facility: Facility identifier (e.g., 'epfl', 'ipp')
                task: Natural language description of what to find/explore.
                    Examples:
                    - "find Python code related to equilibrium reconstruction"
                    - "identify all HDF5 data files and their structure"
                    - "what analysis tools are available?"
                mode: Exploration mode hint (optional):
                    - "auto": Let the agent decide based on the task (default)
                    - "code": Focus on finding source code and packages
                    - "data": Focus on inspecting data files and formats
                    - "env": Focus on system capabilities and tools
                    - "filesystem": Focus on directory structure mapping
                max_iterations: Maximum ReAct loop iterations (default: 10)
                ctx: MCP context for progress reporting (injected automatically)

            Returns:
                Dictionary containing:
                - success: Whether the exploration completed successfully
                - findings: Structured findings from the exploration
                - learnings: New discoveries that were persisted
                - iterations: Number of ReAct iterations used
                - errors: Any errors encountered
            """
            try:
                # Validate facility exists
                try:
                    get_config(facility)
                except ValueError as e:
                    available = list_facilities()
                    return {
                        "success": False,
                        "error": str(e),
                        "available_facilities": available,
                    }

                # Create progress callback if context available
                async def on_progress(
                    progress: float, total: float, message: str
                ) -> None:
                    """Report progress to the MCP client."""
                    if ctx:
                        await ctx.report_progress(progress, total, message)

                async def on_log(level: str, title: str, content: str) -> None:
                    """Stream detailed logs to the MCP client."""
                    if ctx:
                        # Format as markdown for readability
                        if level == "stream":
                            # Streaming tokens - show as-is with truncation
                            # Only show last portion to avoid flooding
                            if len(content) > 500:
                                preview = "..." + content[-500:]
                            else:
                                preview = content
                            formatted = f"**{title}**\n```\n{preview}\n```"
                        elif level == "thought":
                            # Agent reasoning - show full text
                            formatted = f"**{title}**\n\n{content}"
                        elif level == "script":
                            # Script - show as code block
                            formatted = f"**{title}**\n\n```bash\n{content}\n```"
                        elif level == "output":
                            # Output - show as code block (truncate if huge)
                            if len(content) > 5000:
                                content = content[:5000] + "\n... (truncated)"
                            formatted = f"**{title}**\n\n```\n{content}\n```"
                        else:
                            # Error or other
                            formatted = f"**{title}**: {content}"
                        await ctx.info(formatted)

                # Dispatch the exploration agent with progress callback
                agent = ExplorationAgent(facility)
                result = await agent.run(
                    task=task,
                    mode=mode,
                    max_iterations=max_iterations,
                    on_progress=on_progress,
                    on_log=on_log,
                )

                # Persist any learnings
                if result.learnings:
                    persist_learnings(facility, result.learnings)
                    logger.info(
                        f"Persisted {len(result.learnings)} learnings for {facility}"
                    )

                return result.to_dict()

            except Exception as e:
                logger.exception(f"Exploration failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "task": task,
                    "facility": facility,
                }

        @self.mcp.tool()
        def get_facilities() -> dict[str, Any]:
            """
            List available facility configurations.

            Returns:
                Dictionary with list of facility names and their descriptions
            """
            facilities = list_facilities()
            result = {"facilities": []}

            for name in facilities:
                try:
                    config = get_config(name)
                    result["facilities"].append(
                        {
                            "name": name,
                            "description": config.description,
                            "ssh_host": config.ssh_host,
                        }
                    )
                except Exception:
                    result["facilities"].append(
                        {
                            "name": name,
                            "description": "(error loading config)",
                        }
                    )

            return result

    def _register_prompts(self):
        """Register documentation prompts."""

        @self.mcp.prompt()
        def exploration_guide() -> str:
            """
            Guide for using the exploration tools.

            Provides documentation on how to use the explore tool effectively.
            """
            return """# Facility Exploration Guide

## The `explore` Tool

Use the `explore` tool to dispatch an autonomous subagent that will:
1. Connect to a remote fusion facility via SSH
2. Run a ReAct loop with Claude Opus 4.5 to execute your task
3. Generate bash scripts, observe results, and iterate
4. Return structured findings and persist learnings

### Basic Usage

```
explore(facility="epfl", task="find Python code related to equilibrium")
```

### Parameters

- **facility** (required): The facility identifier (e.g., "epfl", "ipp")
- **task** (required): Natural language description of what to explore
- **mode** (optional): Hint for exploration focus
  - `"auto"` (default): Agent decides based on task
  - `"code"`: Focus on source code and packages
  - `"data"`: Focus on data files and formats
  - `"env"`: Focus on system capabilities
  - `"filesystem"`: Focus on directory structure
- **max_iterations** (optional): Max ReAct iterations (default: 10)

### Example Tasks

**Code exploration:**
- "find Python code related to equilibrium reconstruction"
- "identify all MATLAB scripts in /common/tcv/codes"
- "what packages are available for Thomson scattering analysis?"

**Data exploration:**
- "what HDF5 files exist and what's their structure?"
- "how is shot data organized in this facility?"
- "find MDSplus trees and list their signals"

**Environment exploration:**
- "what Python version and modules are available?"
- "is ripgrep (rg) installed? what search tools exist?"
- "what data analysis tools are available?"

### Parallel Exploration

You can run multiple explorations in parallel:

```
# These run concurrently
explore(facility="epfl", task="find equilibrium code")
explore(facility="epfl", task="find diagnostic data")
explore(facility="ipp", task="find equilibrium code")
```

### Knowledge Persistence

The agent automatically:
- Loads knowledge from previous explorations
- Persists new discoveries (e.g., "rg not available, use grep -r")
- Tracks novelty to avoid redundant exploration

## Available Facilities

Use `get_facilities()` to list configured facilities.
"""

        @self.mcp.prompt()
        def list_agents() -> str:
            """
            List available exploration agents and their capabilities.
            """
            return """# Exploration Agents

## General Exploration Agent

**Tool**: `explore(facility, task, mode?, max_iterations?)`
**Status**: ✅ Available

A general-purpose agent that accepts natural language tasks and autonomously
explores remote facilities. The agent:

- Connects via SSH to the target facility
- Runs a ReAct loop with Claude Opus 4.5
- Generates and executes bash scripts
- Tracks novelty to know when to stop
- Persists learnings for future runs

### Modes

| Mode | Focus |
|------|-------|
| `auto` | Agent decides based on task |
| `code` | Source files, packages, imports |
| `data` | HDF5, NetCDF, MDSplus files |
| `env` | System tools, Python, modules |
| `filesystem` | Directory structure |

## Planned Specializations

Future versions may add specialized agents for:
- Deep code analysis with AST parsing
- Interactive MDSplus tree exploration
- Dependency graph mapping
- Documentation extraction

For now, the general `explore` tool handles all these via natural language.
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
