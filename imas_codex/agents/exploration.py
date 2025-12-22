"""
Exploration Agent for natural language facility exploration.

This agent runs a ReAct loop to explore remote facilities via SSH,
using a frontier LLM for decision making and tracking novelty to
know when exploration is complete.

Supports progress streaming via callback for real-time visibility
into the agent's train of thought.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.agents.knowledge import FacilityKnowledge, load_knowledge
from imas_codex.discovery.config import MergedConfig, get_config
from imas_codex.discovery.connection import FacilityConnection, ScriptResult
from imas_codex.discovery.llm import (
    AgentLLM,
    assistant_message,
    system_message,
    user_message,
)
from imas_codex.discovery.prompts import get_prompt_loader

# Type alias for progress callback
# Args: (progress, total, message)
ProgressCallback = Callable[[float, float, str], Awaitable[None]]

# Type alias for detailed log callback
# Args: (level, title, content) where level is "thought", "script", "output", "error"
LogCallback = Callable[[str, str, str], Awaitable[None]]


@dataclass
class ExplorationState:
    """
    Tracks exploration progress and novelty for boredom detection.

    The agent uses this to determine when it's finding diminishing returns
    and should wrap up the exploration.
    """

    seen_paths: set[str] = field(default_factory=set)
    seen_file_types: set[str] = field(default_factory=set)
    seen_patterns: set[str] = field(default_factory=set)
    iterations_without_novelty: int = 0

    # Regex patterns for extracting paths and file types from output
    PATH_PATTERN: re.Pattern = field(
        default=re.compile(r"(?:/[\w.-]+)+/?"),
        init=False,
        repr=False,
    )
    FILE_EXT_PATTERN: re.Pattern = field(
        default=re.compile(r"\.\w{1,10}(?:\s|$|:|\n)"),
        init=False,
        repr=False,
    )

    def assess_novelty(self, observation: str) -> float:
        """
        Assess novelty of an observation and update state.

        Returns:
            Novelty score from 0.0 (bored) to 1.0 (novel)
        """
        # Extract paths from observation
        new_paths = set(self.PATH_PATTERN.findall(observation)) - self.seen_paths

        # Extract file extensions
        extensions = self.FILE_EXT_PATTERN.findall(observation)
        new_types = {ext.strip() for ext in extensions} - self.seen_file_types

        # Update tracking
        if new_paths or new_types:
            self.iterations_without_novelty = 0
            self.seen_paths.update(new_paths)
            self.seen_file_types.update(new_types)
            # More novel if we found many new things
            novelty = min(1.0, (len(new_paths) + len(new_types)) / 10)
            return max(0.5, novelty)  # At least 0.5 if we found something
        else:
            self.iterations_without_novelty += 1
            # Decay novelty based on iterations without discovery
            return max(0.0, 1.0 - (self.iterations_without_novelty * 0.4))

    @property
    def explored_paths(self) -> list[str]:
        """Get sorted list of explored paths."""
        return sorted(self.seen_paths)


@dataclass
class ExplorationResult:
    """Result of an exploration run."""

    success: bool
    findings: dict[str, Any]
    learnings: list[str]
    iterations: int
    task: str = ""
    mode: str = "auto"
    novelty_scores: list[float] = field(default_factory=list)
    scripts_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "task": self.task,
            "mode": self.mode,
            "findings": self.findings,
            "learnings": self.learnings,
            "iterations": self.iterations,
            "scripts_executed": len(self.scripts_executed),
            "errors": self.errors,
            "novelty_scores": self.novelty_scores,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


@dataclass
class ExplorationAgent:
    """
    General-purpose exploration agent for natural language tasks.

    Uses a ReAct loop with a frontier LLM to explore remote facilities
    via SSH, returning structured findings and accumulated learnings.

    The agent tracks novelty across iterations to detect when exploration
    is yielding diminishing returns (the "boredom" metric).

    Example:
        agent = ExplorationAgent("epfl")
        result = await agent.run("find Python code related to equilibrium")
        print(result.findings)
        print(result.learnings)  # Discoveries for persistence
    """

    facility: str
    model: str | None = None
    api_key: str | None = None

    # Internal fields
    config: MergedConfig = field(init=False, repr=False)
    connection: FacilityConnection = field(init=False, repr=False)
    llm: AgentLLM = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the agent."""
        self.config = get_config(self.facility)
        self.connection = FacilityConnection(self.config.ssh_host)

        model = self.model or self.config.discovery.model
        self.llm = AgentLLM(model=model, api_key=self.api_key)

    async def run(
        self,
        task: str,
        mode: str = "auto",
        max_iterations: int = 10,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> ExplorationResult:
        """
        Execute a natural language exploration task.

        Args:
            task: Natural language description of what to find/explore
            mode: Exploration mode - "auto", "code", "data", "env", "filesystem"
            max_iterations: Maximum ReAct loop iterations
            on_progress: Optional async callback for progress updates.
                Called with (current, total, message) during each ReAct step.
            on_log: Optional async callback for detailed logs.
                Called with (level, title, content) for full visibility into:
                - "thought": Agent's reasoning
                - "script": Bash script being executed
                - "output": Script stdout/stderr
                - "error": Errors encountered

        Returns:
            ExplorationResult with findings and learnings
        """
        result = ExplorationResult(
            success=False,
            findings={},
            learnings=[],
            iterations=0,
            task=task,
            mode=mode,
        )

        async def emit_progress(current: float, total: float, message: str) -> None:
            """Emit progress if callback is registered."""
            if on_progress:
                await on_progress(current, total, message)

        async def emit_log(level: str, title: str, content: str) -> None:
            """Emit detailed log if callback is registered."""
            if on_log:
                await on_log(level, title, content)

        # Initialize exploration state for novelty tracking
        state = ExplorationState()

        # Load knowledge from config if available
        knowledge = self._load_knowledge()

        # Build initial prompt
        prompts = get_prompt_loader()
        explore_prompt = prompts.load("explore")

        # Build context for prompt rendering
        context = self._build_context(
            task=task,
            mode=mode,
            knowledge=knowledge,
            state=state,
            iteration=0,
            max_iterations=max_iterations,
        )

        # Load and render the common (system) prompt
        common_prompt = prompts.load("common")
        system_msg = common_prompt.render(**context)

        # Render the explore (user) prompt
        user_msg = explore_prompt.render(**context)

        messages = [system_message(system_msg), user_message(user_msg)]

        with self.connection.session():
            for iteration in range(max_iterations):
                result.iterations = iteration + 1
                iter_label = f"[{iteration + 1}/{max_iterations}]"

                # Emit progress: starting iteration
                await emit_progress(
                    iteration,
                    max_iterations,
                    f"{iter_label} Thinking...",
                )

                # Create streaming callback if logging is enabled
                stream_callback = None
                if on_log:
                    # Capture iter_label value for this iteration
                    current_iter_label = iter_label
                    last_streamed_len = 0

                    async def stream_callback(
                        content: str, label: str = current_iter_label
                    ) -> None:
                        nonlocal last_streamed_len
                        # Only emit new content (delta)
                        if len(content) > last_streamed_len:
                            # Emit periodically as content grows
                            await emit_log(
                                "stream",
                                f"{label} Generating...",
                                content,
                            )
                            last_streamed_len = len(content)

                # Get LLM response with optional streaming
                try:
                    response = await self.llm.chat(messages, on_stream=stream_callback)
                except Exception as e:
                    result.errors.append(f"LLM error: {e}")
                    await emit_log("error", f"{iter_label} LLM Error", str(e))
                    break

                # Emit the agent's full reasoning (final, after streaming)
                if response.reasoning:
                    await emit_log(
                        "thought",
                        f"{iter_label} Agent Reasoning",
                        response.reasoning,
                    )

                # Check if done
                if response.done:
                    result.success = True
                    result.findings = response.findings or {}
                    if "learnings" in result.findings:
                        result.learnings = result.findings.pop("learnings", [])
                    await emit_progress(
                        max_iterations,
                        max_iterations,
                        f"Complete ({iteration + 1} iterations)",
                    )
                    break

                # No script generated - prompt for action
                if not response.script:
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(
                            "Please generate a bash script to continue exploration, "
                            "or signal completion with a JSON block containing your findings."
                        )
                    )
                    await emit_log(
                        "error",
                        f"{iter_label} No Script",
                        "Agent did not generate a script, prompting for action.",
                    )
                    continue

                # Emit the full script being executed
                await emit_log(
                    "script",
                    f"{iter_label} Executing Script",
                    response.script,
                )

                # Execute the script
                try:
                    script_result = self.connection.run_script(response.script)
                    result.scripts_executed.append(response.script)
                except ValueError as e:
                    result.errors.append(f"Script rejected: {e}")
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(
                            f"Script rejected by safety sandbox: {e}\n\n"
                            "Please generate a safer script that only reads data."
                        )
                    )
                    await emit_log(
                        "error",
                        f"{iter_label} Script Rejected",
                        str(e),
                    )
                    continue
                except Exception as e:
                    result.errors.append(f"Execution error: {e}")
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(
                            f"Script execution failed: {e}\n\n"
                            "Please try a different approach."
                        )
                    )
                    await emit_log(
                        "error",
                        f"{iter_label} Execution Error",
                        str(e),
                    )
                    continue

                # Assess novelty of the observation
                novelty = state.assess_novelty(script_result.stdout)
                result.novelty_scores.append(novelty)

                # Emit the full output
                output_summary = (
                    f"Exit code: {script_result.return_code} | "
                    f"Novelty: {novelty:.0%} | "
                    f"Lines: {len(script_result.stdout.splitlines())}"
                )
                full_output = script_result.stdout
                if script_result.stderr:
                    full_output += f"\n\n--- stderr ---\n{script_result.stderr}"
                await emit_log(
                    "output",
                    f"{iter_label} Output ({output_summary})",
                    full_output,
                )

                # Build observation with progress update
                observation = self._format_observation(
                    script_result=script_result,
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                    novelty=novelty,
                    state=state,
                )

                # Feed results back to LLM
                messages.append(assistant_message(response.raw))
                messages.append(user_message(observation))

        result.completed_at = datetime.now(UTC)
        return result

    def _load_knowledge(self) -> FacilityKnowledge:
        """Load accumulated knowledge from facility config."""
        return load_knowledge(self.facility)

    def _build_context(
        self,
        task: str,
        mode: str,
        knowledge: FacilityKnowledge,
        state: ExplorationState,
        iteration: int,
        max_iterations: int,
    ) -> dict[str, Any]:
        """Build the template context for prompt rendering."""
        # Start with facility context
        context = self.config.to_context()

        # Add exploration-specific context
        context.update(
            {
                "task": task,
                "mode": mode,
                "knowledge": knowledge.to_dict() if not knowledge.is_empty() else None,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "novelty_score": 1.0,  # Start optimistic
                "iterations_without_novelty": 0,
                "explored_paths": state.explored_paths,
            }
        )

        return context

    def _format_observation(
        self,
        script_result: ScriptResult,
        iteration: int,
        max_iterations: int,
        novelty: float,
        state: ExplorationState,
    ) -> str:
        """Format script result as an observation with progress info."""
        # Truncate output if too long
        stdout = script_result.stdout
        if len(stdout) > 10000:
            stdout = stdout[:10000] + "\n\n...[output truncated at 10KB]..."

        stderr = script_result.stderr
        if stderr and len(stderr) > 2000:
            stderr = stderr[:2000] + "\n\n...[stderr truncated]..."

        # Build observation
        parts = [
            "## Observation",
            "",
            f"**Exit code**: {script_result.return_code}",
            f"**Iteration**: {iteration} / {max_iterations}",
            f"**Novelty**: {novelty:.2f}",
            "",
        ]

        if novelty < 0.3:
            parts.extend(
                [
                    f"⚠️ **Diminishing returns**: {state.iterations_without_novelty} "
                    "iterations without new discoveries. Consider wrapping up.",
                    "",
                ]
            )

        parts.append("### stdout")
        parts.append("```")
        parts.append(stdout if stdout else "(empty)")
        parts.append("```")

        if stderr:
            parts.append("")
            parts.append("### stderr")
            parts.append("```")
            parts.append(stderr)
            parts.append("```")

        parts.extend(
            [
                "",
                "---",
                "",
                "Continue exploring or signal completion with a JSON block.",
            ]
        )

        return "\n".join(parts)
