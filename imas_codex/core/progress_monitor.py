"""Progress monitoring with clean Rich display.

Provides a unified progress monitor that uses Rich transient progress bars
to show build phases without logging interruptions. Falls back to structured
logging when Rich is unavailable or the terminal is non-interactive.
"""

import logging
from contextlib import contextmanager
from typing import Any

try:
    from rich.console import Console
    from rich.markup import escape as _rich_escape
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    BarColumn = Progress = SpinnerColumn = TaskID = TextColumn = TimeRemainingColumn = (
        TimeElapsedColumn
    ) = Console = Table = Any

    def _rich_escape(text: str) -> str:  # type: ignore[misc]
        return text


def _resolve_rich(use_rich: bool | None) -> bool:
    """Determine whether to use Rich display."""
    if use_rich is not None:
        return use_rich and RICH_AVAILABLE
    from imas_codex.cli.rich_output import should_use_rich

    return RICH_AVAILABLE and should_use_rich()


class ProgressMonitor:
    """Single-phase progress monitor with Rich transient bar or logging fallback.

    The progress bar is transient (disappears on completion) so it never
    collides with subsequent log output.
    """

    def __init__(
        self,
        use_rich: bool | None = None,
        logger: logging.Logger | None = None,
        item_names: list[str] | None = None,
        description_template: str = "Processing: {item}",
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.description_template = description_template
        self._use_rich = _resolve_rich(use_rich)

        self._progress: Progress | None = None
        self._console: Console | None = None
        self._task_id: TaskID | None = None
        self._current_total: int = 0
        self._current_completed: int = 0
        self._current_description: str = ""

        self._max_name_length = (
            max(len(name) for name in item_names) if item_names else 0
        )

    def start_processing(
        self, items: list[str], description: str = "Processing"
    ) -> None:
        """Start tracking a list of items."""
        self._current_total = len(items)
        self._current_completed = 0
        self._current_description = description

        if self._max_name_length == 0 and items:
            self._max_name_length = max(len(item) for item in items)

        if self._use_rich:
            self._console = Console(force_terminal=True)
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
                transient=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                description, total=self._current_total
            )
        else:
            self.logger.info(f"{description}: {self._current_total} items")

    def set_current_item(self, item_name: str) -> None:
        """Update the current item description."""
        if self._use_rich and self._progress and self._task_id is not None:
            padded = item_name.ljust(self._max_name_length)
            self._progress.update(
                self._task_id,
                description=self.description_template.format(item=padded),
            )

    def update_progress(self, item_name: str, error: str | None = None) -> None:
        """Mark one item completed."""
        self._current_completed += 1

        if self._use_rich and self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)
            if error and self._console:
                self._console.print(
                    f"  [red]Error: {_rich_escape(item_name)}: "
                    f"{_rich_escape(error)}[/red]"
                )
        else:
            if error:
                self.logger.error(f"{self._current_description}: {item_name}: {error}")
            elif "/" in item_name and item_name.count("/") == 1:
                self.logger.info(f"{self._current_description}: {item_name}")
            else:
                self.logger.info(
                    f"{self._current_description}: {item_name} "
                    f"({self._current_completed}/{self._current_total})"
                )

    def finish_processing(self) -> None:
        """Complete the current tracking phase."""
        if self._use_rich and self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        else:
            self.logger.info(
                f"Completed {self._current_description}: "
                f"{self._current_completed}/{self._current_total}"
            )

    def log_info(self, message: str) -> None:
        """Log an info message compatible with Rich display."""
        if self._use_rich and self._progress:
            self._progress.console.print(f"  [dim]{_rich_escape(message)}[/dim]")
        else:
            self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        if self._use_rich and self._progress:
            self._progress.console.print(f"  [red]{_rich_escape(message)}[/red]")
        else:
            self.logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        if self._use_rich and self._progress:
            self._progress.console.print(f"  [yellow]{_rich_escape(message)}[/yellow]")
        else:
            self.logger.warning(message)


class BuildProgressMonitor:
    """Multi-phase progress monitor for the DD build pipeline.

    Shows transient per-phase progress bars and prints a compact
    summary table on completion. Logging handlers are temporarily
    raised to WARNING while Rich bars are active so they cannot
    break the display.
    """

    def __init__(
        self,
        use_rich: bool | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self._use_rich = _resolve_rich(use_rich)
        self._console: Console | None = None
        self._phases: list[dict[str, Any]] = []
        self._suppressed_handlers: dict[int, int] = {}

    def status(self, message: str) -> None:
        """Print a status message (visible between phases)."""
        if self._use_rich and self._console:
            self._console.print(f"  [dim]{message}[/dim]")
        else:
            self.logger.info(message)

    @contextmanager
    def managed_build(self, title: str = "IMAS DD Build"):
        """Context manager that provides clean Rich output for the entire build.

        Suppresses root logger handlers during the build so log lines
        cannot break progress bars. Restores them on exit and prints a
        summary table.
        """
        if self._use_rich:
            self._console = Console(force_terminal=True)
            self._console.print(f"\n[bold blue]{title}[/bold blue]")
            self._suppress_logging()
            try:
                yield self
            finally:
                self._restore_logging()
                self._print_summary()
        else:
            yield self

    def _suppress_logging(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers:
            self._suppressed_handlers[id(handler)] = handler.level
            handler.setLevel(logging.WARNING)

    def _restore_logging(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers:
            orig = self._suppressed_handlers.get(id(handler))
            if orig is not None:
                handler.setLevel(orig)
        self._suppressed_handlers.clear()

    def _print_summary(self) -> None:
        if not self._console or not self._phases:
            return

        table = Table(title="Build Summary", show_header=True, padding=(0, 1))
        table.add_column("Phase", style="cyan", no_wrap=True)
        table.add_column("Items", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Detail", style="dim")

        for phase in self._phases:
            status = "[green]done[/green]" if phase.get("ok") else "[red]fail[/red]"
            detail = phase.get("detail", "")
            count = str(phase.get("count", ""))
            table.add_row(phase["name"], count, status, detail)

        self._console.print(table)
        self._console.print()

    def phase(
        self,
        name: str,
        items: list[str] | None = None,
        total: int | None = None,
        description_template: str = "{item}",
        item_label: str = "items",
    ) -> "PhaseTracker":
        """Create a tracked phase of the build.

        Args:
            name: Phase name shown in summary (e.g., "Extract paths")
            items: List of item names if known
            total: Total count if items are not enumerated
            description_template: Template with {item} placeholder
            item_label: Label for the count display (e.g., "versions")

        Returns:
            PhaseTracker context manager
        """
        phase_info: dict[str, Any] = {"name": name, "ok": False}
        self._phases.append(phase_info)
        return PhaseTracker(
            parent=self,
            phase_info=phase_info,
            items=items,
            total=total,
            description_template=description_template,
            item_label=item_label,
        )


class PhaseTracker:
    """Tracks a single build phase with optional Rich progress bar."""

    def __init__(
        self,
        parent: BuildProgressMonitor,
        phase_info: dict[str, Any],
        items: list[str] | None = None,
        total: int | None = None,
        description_template: str = "{item}",
        item_label: str = "items",
    ):
        self._parent = parent
        self._phase = phase_info
        self._items = items
        self._total = total or (len(items) if items else 0)
        self._template = description_template
        self._item_label = item_label
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._completed = 0
        self._max_width = max((len(i) for i in items), default=0) if items else 20

    def __enter__(self) -> "PhaseTracker":
        self._phase["count"] = self._total
        if self._parent._use_rich and self._parent._console and self._total > 0:
            self._parent._console.print(
                f"  [bold]{self._phase['name']}[/bold] [dim]({self._total} {self._item_label})[/dim]"
            )
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._parent._console,
                transient=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self._phase["name"], total=self._total
            )
        elif self._parent._use_rich and self._parent._console:
            # Phase with no items (spinner only)
            self._parent._console.print(
                f"  [bold]{self._phase['name']}[/bold] [dim]...[/dim]"
            )
        elif not self._parent._use_rich:
            self._parent.logger.info(
                f"{self._phase['name']}: {self._total} {self._item_label}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._progress:
            self._progress.stop()
            self._progress = None
        self._phase["ok"] = exc_type is None

        # Print persistent completion line so phases leave a visible trace
        if self._parent._use_rich and self._parent._console:
            detail = self._phase.get("detail", "")
            detail_str = f" [dim]({detail})[/dim]" if detail else ""
            if exc_type is None:
                self._parent._console.print(
                    f"    [green]\u2713[/green] {self._completed}/{self._total}"
                    f"{detail_str}"
                )
            else:
                self._parent._console.print(
                    f"    [red]\u2717[/red] failed at {self._completed}/{self._total}"
                )
        elif not self._parent._use_rich and exc_type is None:
            self._parent.logger.info(
                f"  {self._phase['name']}: {self._completed}/{self._total} done"
            )

    def update(self, item: str | None = None, error: str | None = None) -> None:
        """Advance by one item."""
        self._completed += 1
        if self._progress and self._task_id is not None:
            desc = self._template.format(item=(item or "").ljust(self._max_width))
            self._progress.update(self._task_id, advance=1, description=desc)
            if error:
                self._progress.console.print(
                    f"  [red]{_rich_escape(item or '')}: {_rich_escape(error)}[/red]"
                )
        elif not self._parent._use_rich and item:
            if error:
                self._parent.logger.error(f"  {self._phase['name']}: {item}: {error}")

    def set_detail(self, detail: str) -> None:
        """Set detail text shown in build summary."""
        self._phase["detail"] = detail

    def set_description(self, desc: str) -> None:
        """Update the progress bar description mid-phase."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=desc)

    def log(self, message: str) -> None:
        """Print a message within this phase (compatible with progress bar)."""
        if self._progress:
            self._progress.console.print(f"  [dim]{_rich_escape(message)}[/dim]")
        else:
            self._parent.logger.info(f"  {message}")


def create_progress_monitor(
    use_rich: bool | None = None,
    logger: logging.Logger | None = None,
    item_names: list[str] | None = None,
    description_template: str = "Processing: {item}",
) -> ProgressMonitor:
    """Create a single-phase progress monitor (backward compatible)."""
    return ProgressMonitor(
        use_rich=use_rich,
        logger=logger,
        item_names=item_names,
        description_template=description_template,
    )


def create_build_monitor(
    use_rich: bool | None = None,
    logger: logging.Logger | None = None,
) -> BuildProgressMonitor:
    """Create a multi-phase build monitor for the DD pipeline."""
    return BuildProgressMonitor(use_rich=use_rich, logger=logger)
