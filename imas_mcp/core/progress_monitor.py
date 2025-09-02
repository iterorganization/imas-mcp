"""Progress monitoring utilities with rich fallback to logging."""

import logging
from typing import Any

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy types for type hints when rich is not available
    BarColumn = Progress = SpinnerColumn = TaskID = TextColumn = TimeRemainingColumn = (
        Console
    ) = Any


class ProgressMonitor:
    """Progress monitor that uses rich when available, falls back to logging."""

    def __init__(
        self,
        use_rich: bool | None = None,
        logger: logging.Logger | None = None,
        item_names: list[str] | None = None,
        description_template: str = "Processing IDS: {item}",
    ):
        """Initialize progress monitor.

        Args:
            use_rich: Force use of rich (True) or logging (False).
                If None, auto-detect.
            logger: Logger instance to use for fallback. If None, creates one.
            item_names: List of item names for calculating padding.
            description_template: Template for progress description.
                Should contain {item} placeholder.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.description_template = description_template

        # Determine if we should use rich
        if use_rich is None:
            self._use_rich = RICH_AVAILABLE and self._is_interactive()
        else:
            self._use_rich = use_rich and RICH_AVAILABLE

        self._progress = None
        self._console = None
        self._task_id = None
        self._current_total: int = 0
        self._current_completed: int = 0
        self._current_description: str = ""  # Store current processing description

        # Calculate maximum name length for space-padding
        if item_names:
            self._max_name_length = max(len(name) for name in item_names)
        else:
            self._max_name_length = 0

    def _is_interactive(self) -> bool:
        """Check if we're in an interactive environment."""
        try:
            import sys

            return sys.stdout.isatty() and sys.stderr.isatty()
        except Exception:
            return False

    def start_processing(
        self, items: list[str], description: str = "Processing"
    ) -> None:
        """Start processing a list of items.

        Args:
            items: List of item names to process
            description: Description for the progress bar
        """
        self._current_total = len(items)
        self._current_completed = 0
        self._current_description = description  # Store description for logging

        # Use pre-calculated max length or calculate it now
        if self._max_name_length == 0 and items:
            self._max_name_length = max(len(item) for item in items)

        if self._use_rich and RICH_AVAILABLE:
            self._console = Console()  # type: ignore
            self._progress = Progress(  # type: ignore
                SpinnerColumn(),  # type: ignore
                TextColumn("[progress.description]{task.description}"),  # type: ignore
                BarColumn(),  # type: ignore
                TimeRemainingColumn(),  # type: ignore
                console=self._console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                description, total=self._current_total
            )
        else:
            # Use INFO level for normal progress information
            self.logger.info(f"{description} {self._current_total} items...")

    def set_current_item(self, item_name: str) -> None:
        """Set the current item being processed (updates description).

        Args:
            item_name: Name of the item currently being processed
        """
        if self._use_rich and self._progress and self._task_id is not None:
            # Space-pad the name to keep progress bar position fixed
            padded_name = item_name.ljust(self._max_name_length)
            self._progress.update(
                self._task_id,
                description=self.description_template.format(item=padded_name),
            )
        else:
            # For logging mode, we'll log this when we complete the item
            pass

    def update_progress(self, item_name: str, error: str | None = None) -> None:
        """Update progress for a completed item.

        Args:
            item_name: Name of the item that was processed
            error: Error message if processing failed
        """
        self._current_completed += 1

        if self._use_rich and self._progress and self._task_id is not None:
            # Just advance the progress bar, description was already set
            self._progress.update(self._task_id, advance=1)
            if error and self._console:
                self._console.print(f"[red]Error processing {item_name}: {error}[/red]")
        else:
            if error:
                self.logger.error(f"Error processing IDS {item_name}: {error}")
            else:
                # Use INFO level for consistency with rich mode
                # Skip the repetitive count if item_name already contains progress info
                if "/" in item_name and item_name.count("/") == 1:
                    # Item name already contains progress (e.g., "250/13457")
                    self.logger.info(f"{self._current_description}: {item_name}")
                else:
                    # Item name doesn't contain progress, add it
                    self.logger.info(
                        f"{self._current_description}: {item_name} "
                        f"({self._current_completed}/{self._current_total})"
                    )

    def finish_processing(self) -> None:
        """Finish processing and clean up."""
        if self._use_rich and self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        else:
            # Use INFO level for normal completion information
            self.logger.info(
                f"IMAS-MCP: Completed processing "
                f"{self._current_completed}/{self._current_total} items"
            )

    def log_info(self, message: str) -> None:
        """Log an info message."""
        if self._use_rich and self._console:
            self._console.print(f"[blue]INFO[/blue]: {message}")
        else:
            self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        if self._use_rich and self._console:
            self._console.print(f"[red]ERROR[/red]: {message}")
        else:
            self.logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        if self._use_rich and self._console:
            self._console.print(f"[yellow]WARNING[/yellow]: {message}")
        else:
            self.logger.warning(message)


def create_progress_monitor(
    use_rich: bool | None = None,
    logger: logging.Logger | None = None,
    item_names: list[str] | None = None,
    description_template: str = "Processing IDS: {item}",
) -> ProgressMonitor:
    """Create a progress monitor with appropriate settings.

    Args:
        use_rich: Force use of rich (True) or logging (False).
            If None, auto-detect.
        logger: Logger instance to use for fallback.
        item_names: List of item names for calculating padding.
        description_template: Template for progress description.
            Should contain {item} placeholder.

    Returns:
        Configured ProgressMonitor instance
    """
    return ProgressMonitor(
        use_rich=use_rich,
        logger=logger,
        item_names=item_names,
        description_template=description_template,
    )
