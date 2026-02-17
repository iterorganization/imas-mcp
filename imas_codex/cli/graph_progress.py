"""Rich progress display for graph push/pull/fetch/export operations.

Provides :class:`GraphProgress` — a context manager that shows a
multi-step progress display with spinners, elapsed time, and file-size
information for each phase of a graph lifecycle operation.

Falls back to plain ``click.echo`` when Rich is unavailable or the
terminal is non-interactive.
"""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Generator
from contextlib import contextmanager

from imas_codex.cli.rich_output import should_use_rich


class GraphProgress:
    """Multi-phase progress display for graph operations.

    Usage::

        with GraphProgress("pull") as gp:
            gp.start_phase("Resolving version")
            version = resolve(...)
            gp.complete_phase()

            gp.start_phase("Fetching from GHCR")
            run_oras(...)
            gp.complete_phase(detail="123.4 MB")

            gp.start_phase("Loading into Neo4j")
            load(...)
            gp.complete_phase()
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self._use_rich = should_use_rich()
        self._console = None
        self._status = None
        self._current_phase: str | None = None
        self._phase_count = 0
        self._total_phases = 0

    def __enter__(self) -> GraphProgress:
        if self._use_rich:
            try:
                from rich.console import Console

                self._console = Console()
            except ImportError:
                self._use_rich = False
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def set_total_phases(self, total: int) -> None:
        """Set the expected number of phases for step counting."""
        self._total_phases = total

    def start_phase(self, description: str) -> None:
        """Begin a new operation phase with a spinner."""
        self._phase_count += 1
        if self._total_phases:
            label = f"[{self._phase_count}/{self._total_phases}] {description}"
        else:
            label = description
        self._current_phase = description

        if self._use_rich and self._console:
            if self._status is not None:
                self._status.stop()
            self._status = self._console.status(
                f"[bold blue]{label}[/]", spinner="dots"
            )
            self._status.start()
        else:
            import click

            click.echo(f"  {label}...")

    def update_phase(self, description: str) -> None:
        """Update the current phase description."""
        if self._total_phases:
            label = f"[{self._phase_count}/{self._total_phases}] {description}"
        else:
            label = description

        if self._use_rich and self._status:
            self._status.update(f"[bold blue]{label}[/]")
        else:
            import click

            click.echo(f"  {label}...")

    def complete_phase(self, detail: str | None = None) -> None:
        """Mark the current phase as complete."""
        if self._status is not None:
            self._status.stop()
            self._status = None

        msg = self._current_phase or "Done"
        if detail:
            msg = f"{msg} ({detail})"

        if self._use_rich and self._console:
            self._console.print(f"  [green]✓[/] {msg}")
        else:
            import click

            click.echo(f"  ✓ {msg}")

    def fail_phase(self, detail: str | None = None) -> None:
        """Mark the current phase as failed."""
        if self._status is not None:
            self._status.stop()
            self._status = None

        msg = self._current_phase or "Failed"
        if detail:
            msg = f"{msg}: {detail}"

        if self._use_rich and self._console:
            self._console.print(f"  [red]✗[/] {msg}")
        else:
            import click

            click.echo(f"  ✗ {msg}", err=True)

    def print(self, message: str) -> None:
        """Print a message within the progress context."""
        if self._status is not None:
            self._status.stop()
        if self._use_rich and self._console:
            self._console.print(message)
        else:
            import click

            # Strip Rich markup tags for plain output
            plain = re.sub(r"\[/?[a-z ]+\]", "", message)
            click.echo(plain)
        if self._status is not None:
            self._status.start()


def run_oras_with_progress(
    cmd: list[str],
    *,
    progress: GraphProgress | None = None,
    phase_description: str = "ORAS transfer",
) -> subprocess.CompletedProcess:
    """Run an oras command, showing its native progress output.

    Instead of ``capture_output=True`` (which hides progress), this
    lets oras output flow to the terminal.  Stderr is still captured
    for error reporting on failure.

    Args:
        cmd: The full oras command (e.g. ``["oras", "push", ...]``).
        progress: Optional :class:`GraphProgress` to pause spinner during oras output.
        phase_description: Description for logging on non-Rich terminals.

    Returns:
        The completed process.

    Raises:
        click.ClickException: If the command fails.
    """
    import click

    # Pause the spinner so oras progress output is readable
    if progress and progress._status is not None:
        progress._status.stop()
        progress._status = None

    # Let oras write directly to the terminal for its progress display.
    # Capture stderr via a pipe so we can report errors.
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise click.ClickException(
            f"oras command failed: {result.stderr or 'unknown error'}"
        )

    return result


@contextmanager
def streaming_remote_script(
    script: str,
    ssh_host: str,
    *,
    timeout: int = 600,
    progress: GraphProgress | None = None,
) -> Generator[subprocess.Popen, None, None]:
    """Run a remote script over SSH with streaming output.

    Yields the ``Popen`` handle so callers can read stdout line-by-line
    and update progress.  Stderr is merged into stdout.

    Args:
        script: Bash script to execute.
        ssh_host: SSH host alias.
        timeout: Kill the process after this many seconds.
        progress: Optional progress display to update.

    Yields:
        A running ``Popen`` process.
    """
    from imas_codex.remote.executor import SSH_PATH_DIRS, _ensure_ssh_healthy_once

    _ensure_ssh_healthy_once(ssh_host)

    path_dirs = ":".join(SSH_PATH_DIRS)
    remote_script = f'export PATH="{path_dirs}:$PATH"\n{script}'

    proc = subprocess.Popen(
        ["ssh", "-T", ssh_host, "bash -s"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.stdin is not None  # guaranteed by stdin=PIPE
    assert proc.stdout is not None  # guaranteed by stdout=PIPE

    # Send script and close stdin
    proc.stdin.write(remote_script)
    proc.stdin.close()

    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    try:
        yield proc
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


def remote_operation_streaming(
    script: str,
    ssh_host: str,
    *,
    progress: GraphProgress | None = None,
    progress_markers: dict[str, str] | None = None,
    timeout: int = 600,
) -> str:
    """Run a remote script with streaming progress updates.

    The script should emit progress markers on stdout (lines starting
    with ``PROGRESS:``).  Known markers are mapped to friendly
    descriptions via ``progress_markers``.

    Args:
        script: Bash script to execute.
        ssh_host: SSH host alias.
        progress: Progress display to update.
        progress_markers: Mapping of marker text → phase description
            (e.g. ``{"STOPPING": "Stopping Neo4j"}``).
        timeout: Timeout in seconds.

    Returns:
        Full captured output.
    """
    markers = progress_markers or {}
    output_lines: list[str] = []

    with streaming_remote_script(
        script, ssh_host, timeout=timeout, progress=progress
    ) as proc:
        assert proc.stdout is not None  # guaranteed by stdout=PIPE
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            output_lines.append(stripped)

            # Check for progress markers
            if stripped.startswith("PROGRESS:"):
                marker = stripped.split(":", 1)[1].strip()
                if marker in markers and progress:
                    progress.update_phase(markers[marker])

        proc.wait()

    output = "\n".join(output_lines)
    if proc.returncode != 0:
        import click

        raise click.ClickException(
            f"Remote operation on {ssh_host} failed (exit {proc.returncode}):\n{output}"
        )

    return output
