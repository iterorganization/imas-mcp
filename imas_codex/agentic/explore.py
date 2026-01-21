"""
Facility exploration agent.

Provides autonomous exploration using CodeAgent that generates Python code
to invoke tools, enabling adaptive problem-solving.

Usage:
    # Async context manager (recommended)
    async with ExplorationAgent("iter") as agent:
        result = await agent.explore("Find IMAS integration code")

    # Simple function
    result = await explore_facility("iter", "Find equilibrium codes")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smolagents import CodeAgent

from imas_codex.agentic.agents import create_litellm_model, get_model_for_task
from imas_codex.agentic.monitor import AgentMonitor, create_step_callback
from imas_codex.agentic.prompt_loader import load_prompts
from imas_codex.agentic.tools import (
    AddNoteTool,
    GetFacilityInfoTool,
    QueueFilesTool,
    RunCommandTool,
    query_neo4j,
)
from imas_codex.discovery import get_facility

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
    cost_usd: float = 0.0
    steps: int = 0

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
    cost_usd: float = 0.0


class ExplorationAgent:
    """Smolagents-based agent for autonomous facility exploration.

    Uses CodeAgent to generate Python code for tool invocation,
    enabling adaptive problem-solving and self-debugging.

    Attributes:
        facility: Facility ID being explored
        model: LLM model identifier
        verbose: Whether verbose output is enabled
        progress: Current exploration metrics

    Example:
        async with ExplorationAgent("iter") as agent:
            result = await agent.explore("Find IMAS integration code")
            print(f"Queued: {agent.progress.files_queued} files")
    """

    def __init__(
        self,
        facility: str,
        model: str | None = None,
        verbose: bool = False,
        max_steps: int = 30,
        cost_limit_usd: float = 5.0,
    ) -> None:
        """Initialize the exploration agent.

        Args:
            facility: Facility ID to explore (e.g., "iter", "epfl")
            model: LLM model to use (default: exploration model from config)
            verbose: Enable verbose output
            max_steps: Maximum agent iterations
            cost_limit_usd: Budget limit in USD
        """
        self.facility = facility
        self.model = model or get_model_for_task("exploration")
        self.verbose = verbose
        self.max_steps = max_steps
        self.cost_limit_usd = cost_limit_usd

        # Validate facility exists
        self.config = get_facility(facility)

        # State tracking
        self._start_time: float | None = None
        self._agent: CodeAgent | None = None
        self._monitor: AgentMonitor | None = None

        # Stateful tools (track their own state)
        self._run_tool: RunCommandTool | None = None
        self._queue_tool: QueueFilesTool | None = None
        self._note_tool: AddNoteTool | None = None

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
        """Async context manager exit."""
        self._agent = None
        self._monitor = None

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
        self._agent = None
        self._monitor = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def progress(self) -> ExplorationProgress:
        """Get current exploration progress."""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time

        files_queued = 0
        notes_added = 0
        if self._queue_tool:
            files_queued = len(self._queue_tool.files_queued)
        if self._note_tool:
            notes_added = len(self._note_tool.notes)

        cost = 0.0
        steps = 0
        if self._monitor:
            cost = self._monitor.total_cost_usd
            steps = self._monitor.step_count

        return ExplorationProgress(
            files_queued=files_queued,
            files_skipped=0,  # Not tracked in new architecture
            searches_run=steps,
            notes_added=notes_added,
            elapsed_seconds=elapsed,
            cost_usd=cost,
            steps=steps,
        )

    # -------------------------------------------------------------------------
    # Agent Creation
    # -------------------------------------------------------------------------

    def _create_agent(self, prompt: str | None = None) -> CodeAgent:
        """Create the CodeAgent with tools and monitoring."""
        # Create stateful tools bound to facility
        self._run_tool = RunCommandTool(self.facility)
        self._queue_tool = QueueFilesTool(self.facility)
        self._note_tool = AddNoteTool(self.facility)
        info_tool = GetFacilityInfoTool(self.facility)

        tools = [
            self._run_tool,
            query_neo4j,
            self._queue_tool,
            self._note_tool,
            info_tool,
        ]

        # Create monitor for cost tracking
        self._monitor = AgentMonitor(
            agent_name=f"exploration_{self.facility}",
            cost_limit_usd=self.cost_limit_usd,
        )
        self._monitor._model = self.model

        # Create LLM
        llm = create_litellm_model(
            model=self.model,
            task="exploration",
            temperature=0.3,
            max_tokens=16384,
        )

        # Build system prompt
        system_prompt = _get_prompt("scout-facility")
        if not system_prompt:
            system_prompt = "You are an expert at exploring fusion facility codebases."

        facility_context = f"\n\n## Current Session\nFacility: {self.facility}\n"
        if prompt:
            facility_context += f"Exploration guidance: {prompt}\n"
        system_prompt += facility_context

        # Create callbacks
        callbacks = [create_step_callback(self._monitor)]

        return CodeAgent(
            tools=tools,
            model=llm,
            instructions=system_prompt,
            max_steps=self.max_steps,
            planning_interval=5,  # Re-plan every 5 steps
            step_callbacks=callbacks,
            name=f"explorer_{self.facility}",
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
        # Create agent with prompt context
        self._agent = self._create_agent(prompt)

        # Build task
        task = (
            f"Explore the {self.facility} facility and discover code files "
            "for ingestion into the knowledge graph.\n\n"
            "## Instructions\n"
            "1. Start by checking what's already known (get_facility_info)\n"
            "2. Query the graph for existing discoveries\n"
            "3. Use run_command with rg, fd, dust to explore the filesystem\n"
            "4. Queue discovered files with appropriate interest scores:\n"
            "   - 0.9+: IMAS integration, IDS writers\n"
            "   - 0.7+: Physics codes (equilibrium, transport)\n"
            "   - 0.5+: General analysis\n"
            "5. Add notes for significant findings\n"
            "6. End with a summary of what you found\n"
        )
        if prompt:
            task += f"\n## Focus\n{prompt}\n"

        try:
            # CodeAgent.run is synchronous, run in executor for async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._agent.run, task)
            summary = str(response)

            progress = self.progress
            return ExplorationResult(
                facility=self.facility,
                files_queued=progress.files_queued,
                paths_discovered=progress.searches_run,
                notes_added=progress.notes_added,
                summary=summary,
                progress=progress,
                cost_usd=progress.cost_usd,
            )

        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            progress = self.progress
            return ExplorationResult(
                facility=self.facility,
                files_queued=progress.files_queued,
                paths_discovered=progress.searches_run,
                notes_added=progress.notes_added,
                summary="",
                error=str(e),
                progress=progress,
                cost_usd=progress.cost_usd,
            )

    def explore_sync(self, prompt: str | None = None) -> ExplorationResult:
        """Synchronous wrapper for explore()."""
        return asyncio.run(self.explore(prompt))


# =============================================================================
# Convenience Functions
# =============================================================================


async def explore_facility(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
    cost_limit_usd: float = 5.0,
) -> ExplorationResult:
    """Run an exploration session for a facility.

    Convenience function that wraps ExplorationAgent.

    Args:
        facility: Facility ID to explore
        prompt: Optional guidance (e.g., "Find IMAS integration code")
        verbose: Enable verbose output
        model: LLM model to use
        cost_limit_usd: Budget limit in USD

    Returns:
        ExplorationResult with summary and statistics
    """
    async with ExplorationAgent(
        facility=facility,
        model=model,
        verbose=verbose,
        cost_limit_usd=cost_limit_usd,
    ) as agent:
        return await agent.explore(prompt=prompt)


def explore_facility_sync(
    facility: str,
    prompt: str | None = None,
    verbose: bool = False,
    model: str | None = None,
    cost_limit_usd: float = 5.0,
) -> ExplorationResult:
    """Synchronous wrapper for explore_facility."""
    return asyncio.run(
        explore_facility(facility, prompt, verbose, model, cost_limit_usd)
    )
