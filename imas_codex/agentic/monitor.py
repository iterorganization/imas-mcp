"""
Cost monitoring and budget enforcement for smolagents.

Provides:
- AgentMonitor: Track costs, tokens, and enforce budget limits
- Step callbacks for real-time monitoring
- BudgetExhaustedError for budget enforcement

Usage:
    monitor = AgentMonitor(agent_name="enrichment", cost_limit_usd=5.0)
    callback = create_step_callback(monitor)
    agent = CodeAgent(tools=tools, model=llm, step_callbacks=[callback])

    # Check status
    if monitor.is_exhausted():
        raise BudgetExhaustedError(monitor.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from smolagents.memory import MemoryStep

logger = logging.getLogger(__name__)


# Approximate costs per 1M tokens (USD) - OpenRouter pricing
# Updated 2026-01 from https://openrouter.ai/models
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_cost_per_1M, output_cost_per_1M)
    "google/gemini-3-flash-preview": (0.10, 0.40),
    "google/gemini-3-pro-preview": (1.25, 10.00),
    "anthropic/claude-haiku-4.5": (0.80, 4.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-opus-4.5": (15.00, 75.00),
}

# Default fallback cost estimate
DEFAULT_COST_PER_1M = (1.00, 5.00)


class BudgetExhaustedError(Exception):
    """Raised when cost limit is exceeded."""

    def __init__(self, summary: str):
        self.summary = summary
        super().__init__(f"Budget exhausted: {summary}")


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost for a request.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier (with or without openrouter/ prefix)

    Returns:
        Estimated cost in USD
    """
    # Normalize model name (remove openrouter/ prefix if present)
    model_key = model.replace("openrouter/", "")
    input_cost, output_cost = MODEL_COSTS.get(model_key, DEFAULT_COST_PER_1M)
    return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000


@dataclass
class AgentMonitor:
    """Monitor agent execution with cost tracking and budget enforcement.

    Attributes:
        agent_name: Name of the agent being monitored
        cost_limit_usd: Maximum cost in USD (None = no limit)
        total_cost_usd: Running total of costs incurred
        input_tokens: Total input tokens processed
        output_tokens: Total output tokens generated
        step_count: Number of agent steps executed
        tool_calls: Count of tool invocations
        start_time: Timestamp when monitoring started
        errors: List of errors encountered
    """

    agent_name: str
    cost_limit_usd: float | None = None
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    step_count: int = 0
    tool_calls: int = 0
    start_time: float = field(default_factory=time.monotonic)
    errors: list[str] = field(default_factory=list)
    _model: str = "unknown"

    def record_step(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_name: str | None = None,
        error: str | None = None,
    ) -> float:
        """Record a step and return its cost.

        Args:
            input_tokens: Number of input tokens in this step
            output_tokens: Number of output tokens in this step
            tool_name: Name of tool called (if any)
            error: Error message if step failed

        Returns:
            Cost of this step in USD
        """
        cost = estimate_cost(input_tokens, output_tokens, self._model)
        self.total_cost_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.step_count += 1

        if tool_name:
            self.tool_calls += 1

        if error:
            self.errors.append(error)

        return cost

    def is_exhausted(self) -> bool:
        """Check if budget limit has been reached."""
        if self.cost_limit_usd is None:
            return False
        return self.total_cost_usd >= self.cost_limit_usd

    def remaining_usd(self) -> float | None:
        """Return remaining budget in USD, or None if unlimited."""
        if self.cost_limit_usd is None:
            return None
        return max(0.0, self.cost_limit_usd - self.total_cost_usd)

    def elapsed_seconds(self) -> float:
        """Return elapsed time since monitoring started."""
        return time.monotonic() - self.start_time

    def summary(self) -> str:
        """Return human-readable monitoring summary."""
        elapsed = self.elapsed_seconds()
        parts = [
            f"Agent: {self.agent_name}",
            f"Cost: ${self.total_cost_usd:.4f}",
        ]
        if self.cost_limit_usd is not None:
            remaining = self.remaining_usd()
            parts.append(f"(${remaining:.2f} remaining of ${self.cost_limit_usd:.2f})")

        parts.extend(
            [
                f"| {self.step_count} steps",
                f"| {self.tool_calls} tool calls",
                f"| {self.input_tokens:,} in / {self.output_tokens:,} out tokens",
                f"| {elapsed:.1f}s elapsed",
            ]
        )

        if self.errors:
            parts.append(f"| {len(self.errors)} errors")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Export monitor state as dictionary."""
        return {
            "agent_name": self.agent_name,
            "cost_limit_usd": self.cost_limit_usd,
            "total_cost_usd": self.total_cost_usd,
            "remaining_usd": self.remaining_usd(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "step_count": self.step_count,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": self.elapsed_seconds(),
            "errors": self.errors,
            "is_exhausted": self.is_exhausted(),
        }

    def check_budget(self) -> None:
        """Raise if budget is exhausted.

        Raises:
            BudgetExhaustedError: If budget limit has been reached
        """
        if self.is_exhausted():
            raise BudgetExhaustedError(self.summary())


def create_step_callback(monitor: AgentMonitor) -> Callable[[MemoryStep], None]:
    """Create a step callback that updates the monitor.

    The callback extracts token usage from LLM responses and tracks
    tool calls for cost estimation.

    Args:
        monitor: AgentMonitor to update

    Returns:
        Callback function for smolagents step_callbacks

    Example:
        monitor = AgentMonitor("enrichment", cost_limit_usd=5.0)
        callback = create_step_callback(monitor)
        agent = CodeAgent(tools=tools, model=llm, step_callbacks=[callback])
    """

    def step_callback(step: MemoryStep) -> None:
        """Process a step and update the monitor."""
        # Extract token usage from step if available
        input_tokens = 0
        output_tokens = 0
        tool_name = None
        error = None

        # ActionStep contains LLM response with token info
        if hasattr(step, "model_output") and step.model_output:
            # Estimate tokens from output length (rough approximation)
            # LiteLLM doesn't always provide usage in streaming mode
            output_tokens = len(step.model_output) // 4

        # Check for tool execution
        if hasattr(step, "tool_calls") and step.tool_calls:
            for tool_call in step.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")

        # Check for errors
        if hasattr(step, "error") and step.error:
            error = str(step.error)

        # Record the step
        cost = monitor.record_step(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_name=tool_name,
            error=error,
        )

        # Log if verbose
        if cost > 0.001:  # Only log significant costs
            logger.debug(
                f"Step {monitor.step_count}: ${cost:.4f} "
                f"(total: ${monitor.total_cost_usd:.4f})"
            )

        # Check budget and raise if exhausted
        # This will stop the agent mid-execution
        monitor.check_budget()

    return step_callback


# =============================================================================
# Rich Progress Display
# =============================================================================


@dataclass
class AgentProgressDisplay:
    """Rich progress display for agent execution.

    Provides a composable live display with:
    - Progress bar with estimated work
    - Command ticker showing current operation (e.g., `ssh rg ...`)
    - Clipped output summary in a styled box
    - Cost and step metrics

    Usage:
        with AgentProgressDisplay(monitor, total_steps=10) as display:
            for step in steps:
                display.update_command(f"ssh tcv rg -l 'IMAS' /path")
                display.update_output("Found 5 files matching pattern")
                display.advance()  # Move progress bar
    """

    monitor: AgentMonitor
    total_steps: int | None = None
    show_output_box: bool = True
    title: str = "Agent Progress"
    _live: Any = field(default=None, repr=False)
    _progress: Any = field(default=None, repr=False)
    _task_id: Any = field(default=None, repr=False)
    _current_command: str = ""
    _current_output: str = ""
    _console: Any = field(default=None, repr=False)

    def __post_init__(self):
        from rich.console import Console

        self._console = Console()

    def _build_display(self):
        """Build the Rich display layout."""
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # Build metrics table
        metrics = Table.grid(padding=(0, 2))
        metrics.add_row(
            f"[cyan]Steps:[/cyan] {self.monitor.step_count}",
            f"[cyan]Tool calls:[/cyan] {self.monitor.tool_calls}",
            f"[cyan]Cost:[/cyan] ${self.monitor.total_cost_usd:.4f}",
            f"[cyan]Time:[/cyan] {self.monitor.elapsed_seconds():.1f}s",
        )
        if self.monitor.cost_limit_usd:
            remaining = self.monitor.remaining_usd() or 0
            metrics.add_row(
                "",
                "",
                f"[dim](${remaining:.2f} remaining)[/dim]",
                "",
            )

        # Build command display
        command_text = Text()
        if self._current_command:
            command_text.append("$ ", style="green bold")
            # Clip command to fit
            cmd = self._current_command
            if len(cmd) > 80:
                cmd = cmd[:77] + "..."
            command_text.append(cmd, style="white")
        else:
            command_text.append("(waiting...)", style="dim")

        # Build output box
        output_content = self._current_output or "(no output yet)"
        if len(output_content) > 200:
            output_content = output_content[:197] + "..."

        # Combine into layout
        layout = Layout()
        layout.split_column(
            Layout(metrics, name="metrics", size=3),
            Layout(
                Panel(command_text, title="Current Command", border_style="blue"),
                name="command",
                size=3,
            ),
        )
        if self.show_output_box and self._current_output:
            layout.add_split(
                Layout(
                    Panel(output_content, title="Output", border_style="dim"),
                    name="output",
                    size=5,
                )
            )

        return Panel(layout, title=self.title, border_style="green")

    def __enter__(self) -> AgentProgressDisplay:
        """Start the live display."""
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        # Create progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
        )

        total = self.total_steps if self.total_steps else None
        self._task_id = self._progress.add_task(
            self.monitor.agent_name,
            total=total,
        )

        # Start live display
        self._live = Live(
            self._build_display(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display."""
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        return False

    def update_command(self, command: str) -> None:
        """Update the current command display."""
        self._current_command = command
        if self._live:
            self._live.update(self._build_display())

    def update_output(self, output: str) -> None:
        """Update the output box with clipped content."""
        self._current_output = output
        if self._live:
            self._live.update(self._build_display())

    def advance(self, steps: int = 1) -> None:
        """Advance the progress bar."""
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, steps)
        if self._live:
            self._live.update(self._build_display())

    def set_total(self, total: int) -> None:
        """Update the total steps for the progress bar."""
        self.total_steps = total
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, total=total)

    def log(self, message: str, style: str = "") -> None:
        """Log a message below the progress display."""
        if self._console:
            self._console.log(message, style=style)


def create_progress_callback(
    monitor: AgentMonitor,
    display: AgentProgressDisplay | None = None,
) -> Callable[[MemoryStep], None]:
    """Create a step callback that updates both monitor and display.

    This is a convenience wrapper that combines cost monitoring with
    live progress display.

    Args:
        monitor: AgentMonitor for cost tracking
        display: Optional AgentProgressDisplay for live updates

    Returns:
        Callback function for smolagents step_callbacks
    """
    base_callback = create_step_callback(monitor)

    def progress_callback(step: MemoryStep) -> None:
        # Run base monitoring
        base_callback(step)

        # Update display if available
        if display:
            # Extract command from tool calls
            if hasattr(step, "tool_calls") and step.tool_calls:
                for tool_call in step.tool_calls:
                    name = getattr(tool_call, "name", "")
                    if name == "run_command":
                        # Extract the command argument
                        args = getattr(tool_call, "arguments", {})
                        cmd = args.get("command", "")
                        if cmd:
                            display.update_command(cmd)

            # Extract output from observations
            if hasattr(step, "observations") and step.observations:
                output = str(step.observations)
                display.update_output(output)

            display.advance()

    return progress_callback


# =============================================================================
# Convenience Functions
# =============================================================================


def estimate_task_cost(
    paths_count: int,
    batch_size: int = 100,
    model: str = "google/gemini-3-pro-preview",
) -> dict[str, float]:
    """Estimate cost for a batch enrichment task.

    Args:
        paths_count: Number of paths to enrich
        batch_size: Paths per LLM request
        model: Model to use

    Returns:
        Dict with num_batches, input_tokens, output_tokens, estimated_cost
    """
    num_batches = (paths_count + batch_size - 1) // batch_size
    # Estimate ~12 tokens per path input, ~150 tokens per path output
    input_tokens = num_batches * (500 + batch_size * 12)
    output_tokens = num_batches * batch_size * 150
    cost = estimate_cost(input_tokens, output_tokens, model)

    return {
        "num_batches": num_batches,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }
