"""
File Explorer Subagent for remote filesystem exploration.

This agent runs a ReAct loop to explore filesystem structures on remote
facilities via SSH, using a frontier LLM for decision making.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.discovery.config import MergedConfig, get_config
from imas_codex.discovery.connection import FacilityConnection, ScriptResult
from imas_codex.discovery.llm import (
    AgentLLM,
    assistant_message,
    system_message,
    user_message,
)
from imas_codex.discovery.prompts import PromptLoader, get_prompt_loader


@dataclass
class ExplorationResult:
    """Result of a file exploration run."""

    success: bool
    findings: dict[str, Any]
    learnings: list[str]  # Discoveries to potentially persist as knowledge
    iterations: int
    scripts_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "findings": self.findings,
            "learnings": self.learnings,
            "iterations": self.iterations,
            "scripts_executed": len(self.scripts_executed),
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


@dataclass
class FileExplorerAgent:
    """
    File Explorer subagent for filesystem exploration.

    Uses a ReAct loop with a frontier LLM to explore remote filesystems
    via SSH, returning structured findings and accumulated learnings.

    Example:
        agent = FileExplorerAgent("epfl")
        result = await agent.explore("/common/tcv/codes")
        print(result.findings)
        print(result.learnings)  # Discoveries for Commander to persist
    """

    facility: str
    model: str | None = None
    api_key: str | None = None
    max_iterations: int = 5

    # Internal fields
    config: MergedConfig = field(init=False, repr=False)
    connection: FacilityConnection = field(init=False, repr=False)
    llm: AgentLLM = field(init=False, repr=False)
    prompts: PromptLoader = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the agent."""
        self.config = get_config(self.facility)
        self.connection = FacilityConnection(self.config.ssh_host)
        self.prompts = get_prompt_loader()

        model = self.model or self.config.discovery.model
        self.llm = AgentLLM(model=model, api_key=self.api_key)

    async def explore(
        self,
        path: str,
        max_depth: int = 3,
    ) -> ExplorationResult:
        """
        Explore a filesystem path on the remote facility.

        Args:
            path: Starting path to explore
            max_depth: Maximum directory depth

        Returns:
            ExplorationResult with findings and learnings
        """
        result = ExplorationResult(
            success=False,
            findings={},
            learnings=[],
            iterations=0,
        )

        # Load prompts from markdown files
        context = self.config.to_context()
        common_prompt = self.prompts.load("common")
        explorer_prompt = self.prompts.load("file_explorer")

        system_msg = common_prompt.render(**context)
        user_msg = explorer_prompt.render(
            **context,
            target_path=path,
            max_depth=max_depth,
        )
        messages = [system_message(system_msg), user_message(user_msg)]

        with self.connection.session():
            for iteration in range(self.max_iterations):
                result.iterations = iteration + 1

                # Get LLM response
                try:
                    response = await self.llm.chat(messages)
                except Exception as e:
                    result.errors.append(f"LLM error: {e}")
                    break

                # Check if done
                if response.done:
                    result.success = True
                    result.findings = response.findings or {}
                    # Extract learnings from findings if present
                    if "learnings" in result.findings:
                        result.learnings = result.findings.pop("learnings", [])
                    break

                # No script generated - prompt for action
                if not response.script:
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(
                            "Please generate a bash script to continue exploration, "
                            "or signal completion with a JSON block."
                        )
                    )
                    continue

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
                            "Please generate a safer script."
                        )
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
                    continue

                # Feed results back to LLM
                messages.append(assistant_message(response.raw))
                messages.append(user_message(self._format_observation(script_result)))

        result.completed_at = datetime.now(UTC)
        return result

    def _format_observation(self, result: ScriptResult) -> str:
        """Format script result as an observation using the observation template."""
        # Truncate output if too long
        stdout = result.stdout
        if len(stdout) > 10000:
            stdout = stdout[:10000] + "\n...[truncated]..."

        stderr = result.stderr
        if stderr and len(stderr) > 2000:
            stderr = stderr[:2000] + "\n...[truncated]..."

        # Load and render the observation template
        observation_prompt = self.prompts.load("observation")
        return observation_prompt.render(
            exit_code=result.return_code,
            stdout=stdout if stdout else None,
            stderr=stderr if stderr else None,
        )
