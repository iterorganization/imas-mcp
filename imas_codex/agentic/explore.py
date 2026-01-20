"""
Facility exploration agent with ReAct architecture.

Provides a stateful agent for exploring remote facilities, discovering
source files, and populating the knowledge graph.

The agent uses:
- Fast CLI tools (rg, fd, dust) for efficient file discovery
- SSH for remote facility access (auto-detected)
- Neo4j for knowledge graph queries and persistence
- MCP tools for infrastructure and file tracking

Usage:
    # CLI
    imas-codex explore iter --prompt "Find IMAS integration code"

    # Python - class-based (recommended)
    from imas_codex.agentic.explore import ExplorationAgent

    async with ExplorationAgent("iter") as agent:
        result = await agent.explore(prompt="Find equilibrium codes")
        print(agent.progress)

    # Python - convenience function
    from imas_codex.agentic.explore import run_exploration
    result = await run_exploration("iter", prompt="Find equilibrium codes")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from imas_codex.agentic.llm import get_llm, get_model_for_task
from imas_codex.agentic.prompt_loader import load_prompts
from imas_codex.code_examples import queue_source_files
from imas_codex.discovery import (
    add_exploration_note,
    get_facility,
    update_infrastructure,
)
from imas_codex.graph import GraphClient
from imas_codex.remote.tools import run

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

# Load prompts at module level
_PROMPTS = load_prompts()


def _get_prompt(name: str) -> str:
    """Get prompt by name with fallback."""
    prompt = _PROMPTS.get(name)
    if prompt is None:
        logger.warning(f"Prompt '{name}' not found")
        return ""
    return prompt.content


@dataclass
class ExplorationProgress:
    """Progress metrics for an exploration session."""

    files_queued: int = 0
    files_skipped: int = 0
    searches_run: int = 0
    notes_added: int = 0
    elapsed_seconds: float = 0.0

    @property
    def rate(self) -> float:
        """Files queued per minute."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return (self.files_queued / self.elapsed_seconds) * 60


@dataclass
class ExplorationResult:
    """Result of a facility exploration session."""

    facility: str
    files_queued: int
    paths_discovered: int
    notes_added: int
    summary: str
    error: str | None = None
    progress: ExplorationProgress = field(default_factory=ExplorationProgress)


# =============================================================================
# ExplorationAgent Class
# =============================================================================


class ExplorationAgent:
    """Stateful agent for facility exploration with lifecycle management.

    Encapsulates exploration state, tools, and the underlying ReActAgent.
    Use as a context manager for proper lifecycle management.

    Attributes:
        facility: Facility ID being explored
        config: Merged facility configuration
        model: LLM model identifier
        verbose: Whether verbose output is enabled
        progress: Current exploration metrics

    Example:
        async with ExplorationAgent("iter") as agent:
            # Full exploration
            result = await agent.explore(prompt="Find IMAS code")

            # Targeted search
            files = await agent.find_pattern("imas.DBEntry", "/work/codes")

            # Check progress
            print(f"Queued: {agent.progress.files_queued}")
    """

    def __init__(
        self,
        facility: str,
        model: str | None = None,
        verbose: bool = False,
        max_iterations: int = 30,
    ) -> None:
        """Initialize the exploration agent.

        Args:
            facility: Facility ID to explore (e.g., "iter", "epfl")
            model: LLM model to use (default: from config)
            verbose: Enable verbose agent output
            max_iterations: Maximum agent iterations (default: 30)
        """
        self.facility = facility
        self.model = model or get_model_for_task("exploration")
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Load facility config (validates facility exists)
        self.config = get_facility(facility)

        # State tracking
        self._files_queued: list[str] = []
        self._files_skipped: int = 0
        self._notes: list[str] = []
        self._searches_run: int = 0
        self._start_time: float | None = None

        # Lazy-initialized components
        self._agent: ReActAgent | None = None
        self._tools: list[FunctionTool] | None = None
        self._graph_client: GraphClient | None = None

    # -------------------------------------------------------------------------
    # Context Manager Protocol
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> ExplorationAgent:
        """Async context manager entry."""
        self._start_time = time.monotonic()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - cleanup resources."""
        await self._cleanup()

    def __enter__(self) -> ExplorationAgent:
        """Sync context manager entry."""
        self._start_time = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Sync context manager exit."""
        # Run cleanup synchronously
        if self._graph_client is not None:
            self._graph_client.close()
            self._graph_client = None
        self._agent = None
        self._tools = None

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._graph_client is not None:
            self._graph_client.close()
            self._graph_client = None
        self._agent = None
        self._tools = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def progress(self) -> ExplorationProgress:
        """Get current exploration progress metrics."""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time

        return ExplorationProgress(
            files_queued=len(self._files_queued),
            files_skipped=self._files_skipped,
            searches_run=self._searches_run,
            notes_added=len(self._notes),
            elapsed_seconds=elapsed,
        )

    @property
    def agent(self) -> ReActAgent:
        """Get or create the underlying ReActAgent (lazy initialization)."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    @property
    def tools(self) -> list[FunctionTool]:
        """Get or create exploration tools (lazy initialization)."""
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools

    # -------------------------------------------------------------------------
    # Tool Creation
    # -------------------------------------------------------------------------

    def _create_tools(self) -> list[FunctionTool]:
        """Create tools bound to this facility for exploration."""
        facility = self.facility
        agent_self = self  # Capture self for state tracking in closures

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
            agent_self._searches_run += 1
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
            upper = cypher.upper()
            if any(
                kw in upper for kw in ["CREATE", "DELETE", "SET", "MERGE", "REMOVE"]
            ):
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
                queue_files(["/work/codes/liuqe.py", "/work/codes/chease.py"], 0.85)
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
                agent_self._files_queued.extend(file_paths[: result["discovered"]])
                agent_self._files_skipped += result["skipped"]

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
                add_exploration_note(facility, note)
                agent_self._notes.append(note)
                return f"Note added for {facility}"
            except Exception as e:
                return f"Failed to add note: {e}"

        def update_infra(data: dict[str, Any]) -> str:
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
                update_infrastructure(facility, data)
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
                    "Execute shell command on facility. Use rg, fd, dust for "
                    "exploration. Auto-detects local vs SSH execution."
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
                fn=update_infra,
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

    # -------------------------------------------------------------------------
    # Agent Creation
    # -------------------------------------------------------------------------

    def _create_agent(self, prompt: str | None = None) -> ReActAgent:
        """Create the underlying ReActAgent with system prompt.

        Args:
            prompt: Optional exploration guidance to include in system prompt

        Returns:
            Configured ReActAgent
        """
        system_prompt = _get_prompt("explore-facility")
        if not system_prompt:
            system_prompt = "You are an expert at exploring fusion facility codebases."

        facility_context = f"\n\n## Current Session\nFacility: {self.facility}\n"
        if prompt:
            facility_context += f"Exploration guidance: {prompt}\n"
        system_prompt += facility_context

        llm = get_llm(model=self.model, temperature=0.3, max_tokens=16384)

        return ReActAgent(
            tools=self.tools,
            llm=llm,
            verbose=self.verbose,
            system_prompt=system_prompt,
            max_iterations=self.max_iterations,
        )

    # -------------------------------------------------------------------------
    # Exploration Methods
    # -------------------------------------------------------------------------

    async def explore(self, prompt: str | None = None) -> ExplorationResult:
        """Run a full exploration session.

        Args:
            prompt: Optional guidance (e.g., "Find IMAS integration code")

        Returns:
            ExplorationResult with summary and statistics
        """
        if prompt:
            self._agent = self._create_agent(prompt)

        task = (
            f"Explore the {self.facility} facility and discover code files "
            "for ingestion."
        )
        if prompt:
            task += f" Focus on: {prompt}"
        task += (
            "\n\nStart by checking what's already known (get_facility_info, "
            "query_graph), then explore systematically. Queue discovered files "
            "and add notes for significant findings. End with a summary."
        )

        try:
            # agent.run() returns a WorkflowHandler which is awaitable
            handler = self.agent.run(task)
            response = await handler
            summary = str(response)

            return ExplorationResult(
                facility=self.facility,
                files_queued=len(self._files_queued),
                paths_discovered=self._searches_run,
                notes_added=len(self._notes),
                summary=summary,
                progress=self.progress,
            )

        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            return ExplorationResult(
                facility=self.facility,
                files_queued=len(self._files_queued),
                paths_discovered=self._searches_run,
                notes_added=len(self._notes),
                summary="",
                error=str(e),
                progress=self.progress,
            )

    async def find_pattern(
        self,
        pattern: str,
        path: str,
        file_types: str = "py",
    ) -> list[str]:
        """Targeted search for files matching a pattern.

        A simpler alternative to full exploration - just find files matching
        a specific pattern and optionally queue them.

        Args:
            pattern: Search pattern (regex for rg)
            path: Directory path to search
            file_types: File extensions to search (default: "py")

        Returns:
            List of matching file paths
        """
        cmd = f"rg -l '{pattern}' {path} -g '*.{file_types}'"
        self._searches_run += 1

        try:
            result = run(cmd, facility=self.facility, timeout=60)
            return [f.strip() for f in result.strip().split("\n") if f.strip()]
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []

    def explore_sync(self, prompt: str | None = None) -> ExplorationResult:
        """Synchronous wrapper for explore()."""
        return asyncio.run(self.explore(prompt))


# =============================================================================
# Convenience Functions (backward compatibility)
# =============================================================================


def create_exploration_agent(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> ReActAgent:
    """Create a ReAct agent configured for facility exploration.

    Note: Consider using ExplorationAgent class directly for better state
    management and lifecycle control.

    Args:
        facility: Facility ID to explore
        prompt: Optional guidance for the exploration
        verbose: Enable verbose agent output
        model: LLM model to use (default: from config)

    Returns:
        Configured ReActAgent
    """
    agent = ExplorationAgent(facility=facility, model=model, verbose=verbose)
    return agent._create_agent(prompt)


async def run_exploration(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> ExplorationResult:
    """Run an exploration session for a facility.

    Convenience function that wraps ExplorationAgent for simple use cases.

    Args:
        facility: Facility ID to explore
        prompt: Optional guidance (e.g., "Find IMAS integration code")
        verbose: Enable verbose output
        model: LLM model to use

    Returns:
        ExplorationResult with summary and statistics
    """
    async with ExplorationAgent(
        facility=facility, model=model, verbose=verbose
    ) as agent:
        return await agent.explore(prompt=prompt)


def run_exploration_sync(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> ExplorationResult:
    """Synchronous wrapper for run_exploration."""
    return asyncio.run(run_exploration(facility, prompt, verbose, model))
