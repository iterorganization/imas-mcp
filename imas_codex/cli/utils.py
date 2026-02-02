"""CLI utilities and shared helpers."""

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

console = Console()


def validate_directory(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> Path | None:
    """Validate that a path is a directory."""
    if value is None:
        return None
    path = Path(value)
    if not path.is_dir():
        raise click.BadParameter(f"Directory not found: {value}")
    return path


def validate_file(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> Path | None:
    """Validate that a path is a file."""
    if value is None:
        return None
    path = Path(value)
    if not path.is_file():
        raise click.BadParameter(f"File not found: {value}")
    return path


def output_json(data: Any, pretty: bool = True) -> None:
    """Output data as JSON."""
    if pretty:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.echo(json.dumps(data, default=str))


def output_table(
    title: str,
    columns: list[tuple[str, str]],
    rows: list[list[str]],
) -> None:
    """Output data as a rich table.

    Args:
        title: Table title
        columns: List of (name, style) tuples
        rows: List of row data (each row is a list of strings)
    """
    from rich.table import Table

    table = Table(title=title)
    for name, style in columns:
        table.add_column(name, style=style)

    for row in rows:
        table.add_row(*row)

    console.print(table)


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt user for confirmation."""
    return click.confirm(message, default=default)


def get_facility_or_fail(name: str) -> Any:
    """Get facility config or raise ClickException."""
    from imas_codex.remote.facilities import FacilityManager

    manager = FacilityManager()
    config = manager.get_facility(name)

    if config is None:
        available = ", ".join(manager.list_facilities().keys())
        raise click.ClickException(f"Unknown facility '{name}'. Available: {available}")
    return config


def run_async(coro: Any) -> Any:
    """Run an async coroutine in the current event loop or create one."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # Already in async context - shouldn't happen in CLI but handle it
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop - normal CLI case
        return asyncio.run(coro)


def setup_logging(log_level: str) -> None:
    """Configure logging for CLI commands."""
    import logging

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))


def get_workspace_root() -> Path:
    """Get the imas-codex workspace root directory."""
    # Look for pyproject.toml going up from current file
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback to cwd
    return Path.cwd()


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class ProgressReporter:
    """Simple progress reporter that works with --no-rich flag."""

    def __init__(self, use_rich: bool = True, total: int | None = None):
        self.use_rich = use_rich
        self.total = total
        self.current = 0
        self._progress = None

    def __enter__(self):
        if self.use_rich and self.total:
            from rich.progress import Progress

            self._progress = Progress()
            self._progress.__enter__()
            self._task = self._progress.add_task("Processing...", total=self.total)
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.__exit__(*args)

    def update(self, advance: int = 1, description: str | None = None) -> None:
        self.current += advance
        if self._progress:
            self._progress.update(self._task, advance=advance, description=description)
        else:
            # Simple text progress
            if description:
                click.echo(f"  {description}")


@click.command("setup-age")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path (default: ~/.config/imas-codex/age-key.txt)",
)
def setup_age(output: str | None) -> None:
    """Generate an age encryption key for private data.

    Creates a new age key pair for encrypting private facility YAML files.
    Store this key securely - you'll need it to decrypt your data.

    Examples:
        imas-codex setup-age
        imas-codex setup-age -o ~/my-age-key.txt
    """
    import shutil
    import subprocess

    if not shutil.which("age-keygen"):
        click.echo("Error: age-keygen not found in PATH", err=True)
        click.echo("Install age:")
        click.echo("  brew install age     # macOS")
        click.echo("  apt install age      # Debian/Ubuntu")
        click.echo("  cargo install rage   # Rust alternative")
        raise SystemExit(1)

    output_path = (
        Path(output)
        if output
        else Path.home() / ".config" / "imas-codex" / "age-key.txt"
    )

    if output_path.exists():
        click.echo(f"Key already exists: {output_path}")
        click.echo("Delete it first if you want to regenerate.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["age-keygen", "-o", str(output_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        click.echo(f"Error generating key: {result.stderr}", err=True)
        raise SystemExit(1)

    # Extract and display public key
    content = output_path.read_text()
    public_key = None
    for line in content.splitlines():
        if line.startswith("# public key:"):
            public_key = line.split(":")[-1].strip()
            break

    click.echo(f"Age key generated: {output_path}")
    click.echo(f"Public key: {public_key}")
    click.echo()
    click.echo("IMPORTANT: Back up this key to a password manager.")
    click.echo("You need it to decrypt your private data on other machines.")
    click.echo()
    click.echo("To use, add to your .env:")
    click.echo(f"  IMAS_AGE_KEY_FILE={output_path}")
