"""Rich Live display for stateless scout progress."""

from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    pass


@dataclass
class ScoutDisplay:
    """Live display for scout exploration progress.

    Uses Rich Live to show fixed panels that update in-place
    instead of scrolling output.
    """

    facility: str
    max_steps: int
    console: Console = field(default_factory=Console)
    _live: Live | None = field(default=None, repr=False)
    _progress: Progress | None = field(default=None, repr=False)
    _task_id: TaskID | None = field(default=None, repr=False)
    _current_step: int = 0
    _current_path: str = ""
    _current_action: str = ""
    _history: list[str] = field(default_factory=list)
    _start_time: float = field(default_factory=time)

    # Summary stats
    _total_paths: int = 0
    _remaining: int = 0
    _explored: int = 0
    _files_queued: int = 0
    _dry_run: bool = False

    def _make_layout(self) -> Group:
        """Create the display layout."""
        # Header panel
        elapsed = time() - self._start_time
        header_text = Text()
        header_text.append(f"Scout: {self.facility}", style="bold blue")
        header_text.append(f"  |  Step {self._current_step}/{self.max_steps}")
        header_text.append(f"  |  {elapsed:.1f}s elapsed")
        if self._dry_run:
            header_text.append("  |  ", style="dim")
            header_text.append("DRY RUN", style="yellow bold")

        # Stats table
        stats_table = Table.grid(expand=True)
        stats_table.add_column(justify="left")
        stats_table.add_column(justify="right", style="cyan")
        stats_table.add_column(justify="left", width=4)
        stats_table.add_column(justify="left")
        stats_table.add_column(justify="right", style="cyan")

        stats_table.add_row(
            "Total paths: ",
            str(self._total_paths),
            "   ",
            "Explored: ",
            str(self._explored),
        )
        stats_table.add_row(
            "Remaining: ",
            str(self._remaining),
            "   ",
            "Files queued: ",
            str(self._files_queued),
        )

        # Current action panel
        action_text = Text()
        if self._current_path:
            action_text.append("Path: ", style="dim")
            action_text.append(self._current_path, style="green")
            action_text.append("\n")
        if self._current_action:
            action_text.append("Action: ", style="dim")
            action_text.append(self._current_action, style="yellow")
        else:
            action_text.append("Waiting...", style="dim italic")

        action_panel = Panel(
            action_text,
            title="Current",
            border_style="blue",
            height=4,
        )

        # History panel (last 5 actions)
        history_text = Text()
        for item in self._history[-5:]:
            history_text.append("• ", style="dim")
            history_text.append(item)
            history_text.append("\n")
        if not self._history:
            history_text.append("No actions yet...", style="dim italic")

        history_panel = Panel(
            history_text,
            title="Recent Actions",
            border_style="dim",
            height=7,
        )

        # Progress bar
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            )
            self._task_id = self._progress.add_task(
                "Exploring...", total=self.max_steps
            )

        return Group(
            header_text,
            stats_table,
            "",
            action_panel,
            history_panel,
            self._progress,
        )

    def start(self, dry_run: bool = False) -> None:
        """Start the live display."""
        self._dry_run = dry_run
        self._start_time = time()
        self._live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=4,
            transient=False,  # Keep final state visible
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_step(
        self,
        step: int,
        path: str = "",
        action: str = "",
    ) -> None:
        """Update the current step display."""
        self._current_step = step
        self._current_path = path
        self._current_action = action
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=step)
        if self._live:
            self._live.update(self._make_layout())

    def update_stats(
        self,
        total_paths: int = 0,
        remaining: int = 0,
        explored: int = 0,
        files_queued: int = 0,
    ) -> None:
        """Update the summary statistics."""
        self._total_paths = total_paths
        self._remaining = remaining
        self._explored = explored
        self._files_queued = files_queued
        if self._live:
            self._live.update(self._make_layout())

    def add_history(self, action: str) -> None:
        """Add an action to the history."""
        self._history.append(action)
        if self._live:
            self._live.update(self._make_layout())

    def show_result(self, status: str, message: str = "") -> None:
        """Show a result status."""
        style = (
            "green" if status == "success" else "red" if status == "error" else "yellow"
        )
        result_text = f"[{style}]{status}[/{style}]"
        if message:
            result_text += f": {message}"
        self._current_action = result_text
        if self._live:
            self._live.update(self._make_layout())

    def __enter__(self) -> "ScoutDisplay":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()
