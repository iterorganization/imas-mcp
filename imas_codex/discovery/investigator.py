"""
Agentic Investigator for remote facility exploration.

The Investigator runs an LLM-driven loop where Claude generates bash scripts,
executes them on remote facilities, and iteratively refines its exploration
based on the results.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.discovery.config import get_config
from imas_codex.discovery.connection import FacilityConnection, ScriptResult
from imas_codex.discovery.llm import (
    AgentLLM,
    Message,
    assistant_message,
    system_message,
    user_message,
)
from imas_codex.discovery.prompts import get_prompt_loader


@dataclass
class InvestigationResult:
    """Result of an investigation run."""

    success: bool
    findings: dict[str, Any]
    iterations: int
    messages: list[Message] = field(default_factory=list)
    scripts_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "findings": self.findings,
            "iterations": self.iterations,
            "scripts_executed": len(self.scripts_executed),
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


class Investigator:
    """
    Agentic investigator for remote facility exploration.

    Uses an LLM (Claude Opus 4.5 by default) to generate bash scripts,
    execute them on remote facilities via SSH, and iteratively explore
    based on the results.

    Example:
        inv = Investigator("epfl")
        result = await inv.explore()
        print(result.findings)
    """

    def __init__(
        self,
        facility: str,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the investigator.

        Args:
            facility: Facility identifier (e.g., 'epfl')
            model: LLM model to use (default from config)
            api_key: OpenRouter API key (default from env)
        """
        self.config = get_config(facility)
        self.connection = FacilityConnection(self.config.ssh_host)
        self.prompts = get_prompt_loader()

        # Use model from config if not specified
        model = model or self.config.discovery.model
        self.llm = AgentLLM(model=model, api_key=api_key)

    async def run(
        self,
        task: str,
        prompt_name: str | None = None,
        max_iterations: int | None = None,
        **context: Any,
    ) -> InvestigationResult:
        """
        Run an agentic investigation.

        Args:
            task: Description of what to investigate
            prompt_name: Optional prompt template to use
            max_iterations: Max iterations (default from config)
            **context: Additional context for prompt rendering

        Returns:
            InvestigationResult with findings and metadata
        """
        max_iter = max_iterations or self.config.discovery.max_iterations
        result = InvestigationResult(
            success=False,
            findings={},
            iterations=0,
        )

        # Build initial messages
        if prompt_name:
            system_msg, user_msg = self.prompts.render(
                prompt_name,
                **self.config.to_context(),
                task=task,
                **context,
            )
        else:
            system_msg = self._build_freeform_system()
            user_msg = self._build_freeform_user(task)

        messages = [
            system_message(system_msg),
            user_message(user_msg),
        ]
        result.messages = messages.copy()

        with self.connection.session():
            for iteration in range(max_iter):
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
                    break

                # Check if no script generated
                if not response.script:
                    # LLM is thinking/explaining, prompt for action
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
                    # Script validation failed
                    result.errors.append(f"Script rejected: {e}")
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(
                            f"Script rejected by safety sandbox: {e}\n\n"
                            "Please generate a safer script that avoids this pattern."
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
                messages.append(user_message(self._format_script_result(script_result)))

        result.completed_at = datetime.now(UTC)
        result.messages = messages
        return result

    def run_sync(
        self,
        task: str,
        prompt_name: str | None = None,
        max_iterations: int | None = None,
        **context: Any,
    ) -> InvestigationResult:
        """
        Synchronous version of run().

        Uses the sync LLM client internally.
        """
        max_iter = max_iterations or self.config.discovery.max_iterations
        result = InvestigationResult(
            success=False,
            findings={},
            iterations=0,
        )

        # Build initial messages
        if prompt_name:
            system_msg, user_msg = self.prompts.render(
                prompt_name,
                **self.config.to_context(),
                task=task,
                **context,
            )
        else:
            system_msg = self._build_freeform_system()
            user_msg = self._build_freeform_user(task)

        messages = [
            system_message(system_msg),
            user_message(user_msg),
        ]

        with self.connection.session():
            for iteration in range(max_iter):
                result.iterations = iteration + 1

                try:
                    response = self.llm.chat_sync(messages)
                except Exception as e:
                    result.errors.append(f"LLM error: {e}")
                    break

                if response.done:
                    result.success = True
                    result.findings = response.findings or {}
                    break

                if not response.script:
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(
                            "Please generate a bash script or signal completion."
                        )
                    )
                    continue

                try:
                    script_result = self.connection.run_script(response.script)
                    result.scripts_executed.append(response.script)
                except ValueError as e:
                    result.errors.append(f"Script rejected: {e}")
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(f"Script rejected: {e}\nPlease try again.")
                    )
                    continue
                except Exception as e:
                    result.errors.append(f"Execution error: {e}")
                    messages.append(assistant_message(response.raw))
                    messages.append(
                        user_message(f"Execution failed: {e}\nPlease try again.")
                    )
                    continue

                messages.append(assistant_message(response.raw))
                messages.append(user_message(self._format_script_result(script_result)))

        result.completed_at = datetime.now(UTC)
        result.messages = messages
        return result

    async def explore(self) -> InvestigationResult:
        """
        Run environment exploration using the explore_environment prompt.

        Returns:
            InvestigationResult with RemoteEnvironment findings
        """
        return await self.run(
            task="Discover the system environment",
            prompt_name="explore_environment",
        )

    async def survey(self, path: str, max_depth: int = 3) -> InvestigationResult:
        """
        Survey a directory structure.

        Args:
            path: Directory path to survey
            max_depth: Maximum directory depth

        Returns:
            InvestigationResult with SurveyResult findings
        """
        return await self.run(
            task=f"Survey directory: {path}",
            prompt_name="directory_survey",
            target_path=path,
            max_depth=max_depth,
        )

    async def search(
        self,
        pattern: str,
        file_types: list[str] | None = None,
        search_paths: list[str] | None = None,
    ) -> InvestigationResult:
        """
        Search for code patterns.

        Args:
            pattern: Regex pattern to search for
            file_types: File extensions to search (e.g., ["*.py"])
            search_paths: Paths to search in

        Returns:
            InvestigationResult with SearchResult findings
        """
        return await self.run(
            task=f"Search for: {pattern}",
            prompt_name="code_search",
            pattern=pattern,
            file_types=file_types or ["*.py"],
            search_paths=search_paths or self.config.paths.code,
        )

    async def ask(self, question: str) -> InvestigationResult:
        """
        Ask a freeform question about the facility.

        Args:
            question: Natural language question

        Returns:
            InvestigationResult with answer in findings
        """
        return await self.run(task=question)

    def _build_freeform_system(self) -> str:
        """Build system message for freeform queries."""
        return f"""You are an expert system administrator exploring a remote fusion research facility.

Facility: {self.config.facility}
Description: {self.config.description}
SSH Host: {self.config.ssh_host}

You operate in an agentic loop:
1. Generate a bash script to gather information
2. I will execute it and show you the results
3. You may generate follow-up scripts
4. When done, output JSON with {{"done": true, "findings": {{...}}}}

Safety rules:
- All scripts must be READ-ONLY
- Never use: rm, mv, cp, chmod, sudo, dd, or destructive commands
- Handle errors gracefully
"""

    def _build_freeform_user(self, task: str) -> str:
        """Build user message for freeform queries."""
        context = self.config.to_context()
        hints = "\n".join(f"- {h}" for h in context.get("exploration_hints", []))

        return f"""Task: {task}

Known information about this facility:
{hints}

Known data paths: {context.get("paths", {}).get("data", [])}
Known code paths: {context.get("paths", {}).get("code", [])}

Please generate your first exploration script.
"""

    def _format_script_result(self, result: ScriptResult) -> str:
        """Format script execution result for LLM feedback."""
        parts = [f"**Exit code:** {result.return_code}"]

        if result.stdout:
            # Truncate if too long
            stdout = result.stdout
            if len(stdout) > 10000:
                stdout = stdout[:10000] + "\n...[truncated]..."
            parts.append(f"**stdout:**\n```\n{stdout}\n```")

        if result.stderr:
            stderr = result.stderr
            if len(stderr) > 2000:
                stderr = stderr[:2000] + "\n...[truncated]..."
            parts.append(f"**stderr:**\n```\n{stderr}\n```")

        parts.append(
            "\nAnalyze these results. You may:\n"
            "- Generate another script to explore further\n"
            "- Refine your approach if there were errors\n"
            '- Signal completion with `{"done": true, "findings": {...}}`'
        )

        return "\n\n".join(parts)
