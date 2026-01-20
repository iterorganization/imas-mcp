"""
Facility exploration agent with ReAct architecture.

Provides an interactive agent for exploring remote facilities, discovering
source files, and populating the knowledge graph.

The agent uses:
- Fast CLI tools (rg, fd, dust) for efficient file discovery
- SSH for remote facility access (auto-detected)
- Neo4j for knowledge graph queries and persistence
- MCP tools for infrastructure and file tracking

Usage:
    # CLI
    imas-codex explore iter --prompt "Find IMAS integration code"

    # Python
    from imas_codex.agentic.explore import run_exploration
    result = await run_exploration("iter", prompt="Find equilibrium codes")
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from imas_codex.agentic.llm import get_llm, get_model_for_task
from imas_codex.agentic.prompt_loader import load_prompts
from imas_codex.code_examples import queue_source_files
from imas_codex.discovery import get_facility
from imas_codex.graph import GraphClient
from imas_codex.remote.tools import run

logger = logging.getLogger(__name__)

# Load prompts
_PROMPTS = load_prompts()


def _get_prompt(name: str) -> str:
    """Get prompt by name with fallback."""
    prompt = _PROMPTS.get(name)
    if prompt is None:
        logger.warning(f"Prompt '{name}' not found")
        return ""
    return prompt.content


@dataclass
class ExplorationResult:
    """Result of a facility exploration session."""

    facility: str
    files_queued: int
    paths_discovered: int
    notes_added: int
    summary: str
    error: str | None = None


# =============================================================================
# Exploration Tools
# =============================================================================


def _create_exploration_tools(facility: str) -> list[FunctionTool]:
    """Create tools bound to a specific facility for exploration.

    Args:
        facility: Facility ID to explore

    Returns:
        List of FunctionTools for exploration
    """

    def run_command(command: str, timeout: int = 60) -> str:
        """Execute a shell command on the facility.

        Use this for exploration tasks like listing directories, searching files,
        or running analysis tools. The command runs on the target facility
        (locally if you're there, via SSH if remote).

        Args:
            command: Shell command to execute (use rg, fd, dust, etc.)
            timeout: Command timeout in seconds (default: 60)

        Returns:
            Command output (stdout + stderr)

        Examples:
            - rg -l 'IMAS' /work/codes -g '*.py'
            - fd -e py /work/projects | head -50
            - dust -d 2 /work
        """
        try:
            return run(command, facility=facility, timeout=timeout)
        except TimeoutError:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Command error: {e}"

    def query_graph(cypher: str) -> str:
        """Execute a read-only Cypher query against the knowledge graph.

        Use this to check what's already known about the facility, find
        existing source files, or explore discovered paths.

        Args:
            cypher: READ-ONLY Cypher query

        Returns:
            Query results (max 20 rows)

        Examples:
            MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: 'iter'})
            RETURN sf.path, sf.status LIMIT 10
        """
        # Safety check
        upper = cypher.upper()
        if any(kw in upper for kw in ["CREATE", "DELETE", "SET", "MERGE", "REMOVE"]):
            return "Error: Only read-only queries allowed"

        try:
            with GraphClient() as gc:
                result = gc.query(cypher)
                if not result:
                    return "No results"
                output = [str(dict(r)) for r in result[:20]]
                if len(result) > 20:
                    output.append(f"... and {len(result) - 20} more")
                return "\n".join(output)
        except Exception as e:
            return f"Query error: {e}"

    def queue_files(file_paths: list[str], interest_score: float = 0.7) -> str:
        """Queue source files for ingestion into the knowledge graph.

        Call this to persist discovered files. The ingestion pipeline will
        later process them, extract code chunks, and generate embeddings.

        Args:
            file_paths: List of absolute file paths on the facility
            interest_score: Priority score 0.0-1.0 (higher = more important)
                - 0.9+: IMAS integration, IDS writers
                - 0.7+: Physics codes (equilibrium, transport)
                - 0.5+: General analysis
                - <0.5: Utilities, config

        Returns:
            Summary of queued files

        Example:
            queue_files(["/work/codes/liuqe/liuqe.py", "/work/codes/chease/chease.py"], 0.85)
        """
        if not file_paths:
            return "No files provided"

        try:
            result = queue_source_files(
                facility=facility,
                file_paths=file_paths,
                interest_score=interest_score,
                discovered_by="explore_agent",
            )
            return (
                f"Queued: {result['discovered']}, "
                f"Skipped: {result['skipped']} (already discovered), "
                f"Errors: {len(result['errors'])}"
            )
        except Exception as e:
            return f"Queue error: {e}"

    def add_note(note: str) -> str:
        """Add a timestamped exploration note for this facility.

        Use this to record significant findings, patterns, or observations
        that should be preserved for future reference.

        Args:
            note: The observation to record

        Returns:
            Confirmation message
        """
        try:
            from imas_codex.discovery import add_exploration_note

            add_exploration_note(facility, note)
            return f"Note added for {facility}"
        except Exception as e:
            return f"Failed to add note: {e}"

    def update_infrastructure(data: dict[str, Any]) -> str:
        """Update facility infrastructure data (tools, paths, OS info).

        Use this to persist discovered infrastructure details like:
        - Tool versions: {"tools": {"rg": "14.1.1", "fd": "10.2.0"}}
        - Important paths: {"paths": {"imas": {"/work/imas": "IMAS root"}}}
        - System info: {"os": {"version": "RHEL 9.2"}}

        Args:
            data: Dictionary to merge into infrastructure config

        Returns:
            Confirmation message
        """
        try:
            from imas_codex.discovery import update_infrastructure as update_infra

            update_infra(facility, data)
            return f"Infrastructure updated for {facility}"
        except Exception as e:
            return f"Failed to update infrastructure: {e}"

    def get_facility_info() -> str:
        """Get current facility configuration and exploration status.

        Returns the merged public + private configuration, including
        any previously discovered paths, tools, and exploration notes.

        Returns:
            Facility configuration as formatted string
        """
        try:
            info = get_facility(facility)

            output = [f"Facility: {facility}"]
            output.append(f"SSH host: {info.get('ssh_host', facility)}")

            if tools := info.get("tools", {}):
                tool_strs = []
                for k, v in tools.items():
                    version = v.get("version", "?") if isinstance(v, dict) else v
                    tool_strs.append(f"{k}={version}")
                output.append(f"Tools: {', '.join(tool_strs)}")

            if paths := info.get("paths", {}):
                output.append("Known paths:")
                for category, path_dict in paths.items():
                    if isinstance(path_dict, dict):
                        for p, desc in list(path_dict.items())[:3]:
                            output.append(f"  [{category}] {p}: {desc}")

            if notes := info.get("exploration_notes", []):
                output.append(f"Notes: {len(notes)} exploration notes")

            return "\n".join(output)
        except Exception as e:
            return f"Failed to get facility info: {e}"

    return [
        FunctionTool.from_defaults(
            fn=run_command,
            name="run_command",
            description=(
                "Execute shell command on facility. Use rg, fd, dust for exploration. "
                "Auto-detects local vs SSH execution."
            ),
        ),
        FunctionTool.from_defaults(
            fn=query_graph,
            name="query_graph",
            description=(
                "Query Neo4j knowledge graph. Check existing discoveries, "
                "find source files by status, explore relationships."
            ),
        ),
        FunctionTool.from_defaults(
            fn=queue_files,
            name="queue_files",
            description=(
                "Queue source files for ingestion. MUST call this to persist "
                "file discoveries. Set interest_score based on physics value."
            ),
        ),
        FunctionTool.from_defaults(
            fn=add_note,
            name="add_note",
            description=(
                "Add timestamped exploration note. Use for significant findings "
                "like IMAS patterns, code locations, conventions."
            ),
        ),
        FunctionTool.from_defaults(
            fn=update_infrastructure,
            name="update_infrastructure",
            description=(
                "Update facility infrastructure data (tools, paths, OS). "
                "Persists to private config."
            ),
        ),
        FunctionTool.from_defaults(
            fn=get_facility_info,
            name="get_facility_info",
            description=(
                "Get current facility config and exploration status. "
                "Check before exploring to see what's already known."
            ),
        ),
    ]


# =============================================================================
# Exploration Agent
# =============================================================================


def create_exploration_agent(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> ReActAgent:
    """Create a ReAct agent configured for facility exploration.

    Args:
        facility: Facility ID to explore
        prompt: Optional guidance for the exploration
        verbose: Enable verbose agent output
        model: LLM model to use (default: from config)

    Returns:
        Configured ReActAgent
    """
    # Get model from config if not specified
    if model is None:
        model = get_model_for_task("exploration")

    # Load and customize system prompt
    system_prompt = _get_prompt("explore-facility")
    if not system_prompt:
        system_prompt = "You are an expert at exploring fusion facility codebases."

    # Add facility context
    facility_context = f"\n\n## Current Session\nFacility: {facility}\n"
    if prompt:
        facility_context += f"Exploration guidance: {prompt}\n"

    system_prompt += facility_context

    # Create tools bound to this facility
    tools = _create_exploration_tools(facility)

    # Create LLM
    llm = get_llm(model=model, temperature=0.3, max_tokens=16384)

    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        system_prompt=system_prompt,
        max_iterations=30,  # Allow extended exploration
    )


async def run_exploration(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> ExplorationResult:
    """Run an exploration session for a facility.

    Args:
        facility: Facility ID to explore
        prompt: Optional guidance (e.g., "Find IMAS integration code")
        verbose: Enable verbose output
        model: LLM model to use

    Returns:
        ExplorationResult with summary and statistics
    """
    agent = create_exploration_agent(
        facility=facility,
        prompt=prompt,
        verbose=verbose,
        model=model,
    )

    # Build the initial task
    task = f"Explore the {facility} facility and discover code files for ingestion."
    if prompt:
        task += f" Focus on: {prompt}"
    task += (
        "\n\nStart by checking what's already known (get_facility_info, query_graph), "
        "then explore systematically. Queue discovered files and add notes for "
        "significant findings. End with a summary."
    )

    try:
        # Run the agent
        response = await agent.run(task)
        summary = str(response)

        # Get statistics from graph
        with GraphClient() as gc:
            stats = gc.query(
                """
                MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE sf.discovered_by = 'explore_agent'
                RETURN count(sf) AS files_queued
                """,
                facility=facility,
            )
            files_queued = stats[0]["files_queued"] if stats else 0

        return ExplorationResult(
            facility=facility,
            files_queued=files_queued,
            paths_discovered=0,  # Could enhance to track
            notes_added=0,  # Could enhance to track
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Exploration failed: {e}")
        return ExplorationResult(
            facility=facility,
            files_queued=0,
            paths_discovered=0,
            notes_added=0,
            summary="",
            error=str(e),
        )


def run_exploration_sync(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> ExplorationResult:
    """Synchronous wrapper for run_exploration."""
    return asyncio.run(run_exploration(facility, prompt, verbose, model))
