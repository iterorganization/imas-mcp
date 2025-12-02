"""Standalone CLI for managing documentation libraries."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from imas_mcp.services.docs_cli_helpers import (
    build_docs_server_command,
    get_npx_executable,
)
from imas_mcp.services.docs_server_manager import DocsServerManager

# Load environment variables from .env file
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_store_path() -> Path:
    """Get the docs store path from environment or use default."""
    store_path_env = os.environ.get("DOCS_MCP_STORE_PATH")
    if store_path_env:
        return Path(store_path_env)
    return Path.cwd() / "docs-data"


def cleanup_orphaned_servers(verbose: bool = False) -> int:
    """Kill orphaned docs-mcp-server processes."""
    if verbose:
        click.echo("Checking for orphaned docs-mcp-server processes...")

    killed = DocsServerManager.kill_orphaned_servers(verbose=verbose)

    if verbose and killed > 0:
        click.echo(f"Cleaned up {killed} orphaned process(es)")
    elif verbose:
        click.echo("No orphaned processes found")

    return killed


@click.group()
def main():
    """Manage documentation libraries and cache."""
    pass


@main.command("list")
@click.option(
    "--docs-store-path",
    envvar="DOCS_MCP_STORE_PATH",
    type=click.Path(path_type=Path),
    help="Path for docs-mcp-server data storage (env: DOCS_MCP_STORE_PATH) (default: ./docs-data)",
)
def docs_list(docs_store_path: Path | None):
    """List all indexed documentation libraries."""
    store_path = docs_store_path or get_store_path()
    store_path.mkdir(parents=True, exist_ok=True)

    try:
        npx = get_npx_executable()
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Build the npx command using helper
    cmd = build_docs_server_command(npx, "list")

    # Set environment variable for store path
    env = os.environ.copy()
    env["DOCS_MCP_STORE_PATH"] = str(store_path)

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env
        )

        # Parse the output
        output = result.stdout.strip()
        if "No libraries indexed yet" in output or not output or output == "[]":
            click.echo("No documentation libraries indexed yet.")
            click.echo("\nTo add a library:")
            click.echo("  docs add <library> <url> [--max-pages N]")
            click.echo("\nExamples:")
            click.echo("  docs add numpy https://numpy.org/doc/stable/")
            click.echo(
                "  docs add pandas https://pandas.pydata.org/docs/ --max-pages 500"
            )
            click.echo("  docs add python https://docs.python.org/3/")
        else:
            # Pretty-print the JSON output
            import json

            try:
                data = json.loads(output)
                if not data:
                    click.echo("No documentation libraries indexed yet.")
                    click.echo("\nTo add a library:")
                    click.echo("  docs add <library> <url>")
                else:
                    click.echo(f"\nIndexed libraries ({len(data)}):")
                    click.echo()
                    for lib in data:
                        click.echo(f"• {lib['name']}")
                        for ver in lib.get("versions", []):
                            version_str = (
                                f" (v{ver['version']})" if ver.get("version") else ""
                            )
                            doc_count = ver.get("documentCount", 0)
                            status = ver.get("status", "unknown")
                            click.echo(
                                f"  {version_str} - {doc_count} documents, status: {status}"
                            )
                    click.echo()
            except json.JSONDecodeError:
                # Fallback to raw output
                click.echo(output)

    except subprocess.CalledProcessError as e:
        click.echo(f"Error listing libraries: {e}", err=True)
        if e.stderr:
            click.echo(e.stderr, err=True)
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(
            "Error: npx command not found. Please ensure Node.js and npm are installed.",
            err=True,
        )
        sys.exit(1)


@main.command("add")
@click.argument("library")
@click.argument("url")
@click.option(
    "-v",
    "--version",
    type=str,
    help="Version of the library (optional)",
)
@click.option(
    "--max-pages",
    type=int,
    help="Maximum number of pages to scrape",
)
@click.option(
    "--docs-store-path",
    envvar="DOCS_MCP_STORE_PATH",
    type=click.Path(path_type=Path),
    help="Path for docs-mcp-server data storage (env: DOCS_MCP_STORE_PATH) (default: ./docs-data)",
)
def docs_add(
    library: str,
    url: str,
    version: str | None,
    max_pages: int | None,
    docs_store_path: Path | None,
):
    """Add a new documentation library by scraping a URL.

    LIBRARY: Name for the library
    URL: Documentation URL to scrape

    Examples:
        docs add numpy https://numpy.org/doc/stable/
        docs add numpy https://numpy.org/doc/stable/ --version 1.24.0
        docs add python https://docs.python.org/3/ -v 3.11
    """
    # Clean up orphaned servers first
    cleanup_orphaned_servers(verbose=False)

    store_path = docs_store_path or get_store_path()
    store_path.mkdir(parents=True, exist_ok=True)

    try:
        npx = get_npx_executable()
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Build the npx command using helper
    cmd = build_docs_server_command(npx, "scrape", library, url)

    if version:
        cmd.extend(["--version", version])

    if max_pages:
        cmd.extend(["--max-pages", str(max_pages)])

    # Set environment variable for store path
    env = os.environ.copy()
    env["DOCS_MCP_STORE_PATH"] = str(store_path)

    version_str = f" (version: {version})" if version else ""
    click.echo(f"Adding library '{library}'{version_str} from {url}...")
    if max_pages:
        click.echo(f"Maximum pages: {max_pages}")

    try:
        # Run the command with live output
        result = subprocess.run(cmd, check=True, env=env)
        click.echo(f"\n✓ Successfully indexed '{library}'{version_str}")
        sys.exit(result.returncode)

    except subprocess.CalledProcessError as e:
        click.echo("\n✗ Failed to add library", err=True)
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(
            "Error: npx command not found. Please ensure Node.js and npm are installed.",
            err=True,
        )
        sys.exit(1)


@main.command("remove")
@click.argument("library")
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--docs-store-path",
    envvar="DOCS_MCP_STORE_PATH",
    type=click.Path(path_type=Path),
    help="Path for docs-mcp-server data storage (env: DOCS_MCP_STORE_PATH) (default: ./docs-data)",
)
def docs_remove(library: str, force: bool, docs_store_path: Path | None):
    """Remove a documentation library.

    LIBRARY: Name of the library to remove
    """
    # Clean up orphaned servers first
    cleanup_orphaned_servers(verbose=False)

    # Confirm removal if not forced
    if not force:
        click.echo(
            f"This will remove library '{library}' and all its indexed documents."
        )
        if not click.confirm("Are you sure?"):
            click.echo("Cancelled.")
            return

    store_path = docs_store_path or get_store_path()
    store_path.mkdir(parents=True, exist_ok=True)

    try:
        npx = get_npx_executable()
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Build the npx command using helper
    cmd = build_docs_server_command(npx, "remove", library)

    # Set environment variable for store path
    env = os.environ.copy()
    env["DOCS_MCP_STORE_PATH"] = str(store_path)

    click.echo(f"Removing library '{library}'...")

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env
        )
        click.echo(f"\n✓ Successfully removed '{library}'")
        if result.stdout:
            click.echo(result.stdout)

    except subprocess.CalledProcessError as e:
        click.echo("\n✗ Failed to remove library", err=True)
        if e.stderr:
            click.echo(e.stderr, err=True)
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(
            "Error: npx command not found. Please ensure Node.js and npm are installed.",
            err=True,
        )
        sys.exit(1)


@main.command("clear-all")
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--docs-store-path",
    envvar="DOCS_MCP_STORE_PATH",
    type=click.Path(path_type=Path),
    help="Path for docs-mcp-server data storage (env: DOCS_MCP_STORE_PATH) (default: ./docs-data)",
)
@click.option(
    "--keep-locked",
    is_flag=True,
    help="Keep locked files and continue (useful when database is in use)",
)
def docs_clear_all(force: bool, docs_store_path: Path | None, keep_locked: bool):
    """Clear all documentation cache and libraries.

    This will remove all indexed documentation and delete the docs-data directory.
    Use --keep-locked to work around locked database files.
    """
    store_path = docs_store_path or get_store_path()

    # Confirm clearing if not forced
    if not force:
        click.echo(
            "⚠️  WARNING: This will remove ALL documentation libraries and cache."
        )
        click.echo(f"Store path: {store_path.absolute()}")
        if not click.confirm("Are you sure you want to continue?"):
            click.echo("Cancelled.")
            return

    # Clean up orphaned servers first - this is critical to release file locks
    click.echo("Terminating all docs-mcp-server processes...")
    cleanup_orphaned_servers(verbose=True)

    # Give processes time to fully terminate and release file handles
    import time

    time.sleep(2)

    click.echo("Clearing all documentation cache...")

    # Remove the entire docs-data directory
    if store_path.exists():
        try:
            removed_count = 0
            locked_files = []

            # Remove files, tracking locked ones
            for root, dirs, files in os.walk(store_path, topdown=False):
                for name in files:
                    file_path = Path(root) / name
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except PermissionError:
                        locked_files.append(str(file_path))

                # Try to remove empty directories
                for name in dirs:
                    dir_path = Path(root) / name
                    try:
                        dir_path.rmdir()
                    except (PermissionError, OSError):
                        pass  # Directory not empty or locked

            # Try to remove the main directory
            try:
                store_path.rmdir()
            except (PermissionError, OSError):
                pass  # Directory not empty or locked

            click.echo(f"Removed {removed_count} file(s)")

            if locked_files:
                if keep_locked:
                    click.echo(
                        f"\n⚠️  {len(locked_files)} file(s) are locked and could not be removed:"
                    )
                    for f in locked_files[:5]:  # Show first 5
                        click.echo(f"  • {f}")
                    if len(locked_files) > 5:
                        click.echo(f"  ... and {len(locked_files) - 5} more")
                    click.echo("\nThese files are likely database files still in use.")
                    click.echo("They will be cleared on next server restart.")
                else:
                    click.echo(
                        f"\n✗ Error: {len(locked_files)} file(s) are locked and could not be removed",
                        err=True,
                    )
                    click.echo(
                        "\nSome files in the docs-data directory are currently in use.",
                        err=True,
                    )
                    click.echo("This can happen if:", err=True)
                    click.echo(
                        "  • A docs-mcp-server is still running (try: docs cleanup)",
                        err=True,
                    )
                    click.echo("  • An IDE or file explorer has files open", err=True)
                    click.echo(
                        "  • Database connections haven't been released", err=True
                    )
                    click.echo(
                        "\nUse --keep-locked flag to remove unlocked files:",
                        err=True,
                    )
                    click.echo("  docs clear-all --force --keep-locked", err=True)
                    sys.exit(1)

        except Exception as e:
            click.echo(f"\n✗ Error removing directory: {e}", err=True)
            sys.exit(1)

    # Recreate empty directory with original name
    original_store_path = docs_store_path or get_store_path()
    original_store_path.mkdir(parents=True, exist_ok=True)

    click.echo("\n✓ Successfully cleared all documentation cache")


@main.command("cleanup")
def docs_cleanup():
    """Kill all orphaned docs-mcp-server processes."""
    click.echo("Searching for orphaned docs-mcp-server processes...")

    processes = DocsServerManager.find_docs_server_processes()

    if not processes:
        click.echo("No orphaned docs-mcp-server processes found.")
        return

    click.echo(f"\nFound {len(processes)} docs-mcp-server process(es):")
    for proc in processes:
        click.echo(f"  • Port {proc['port']}, PID {proc['pid']}")

    if not click.confirm("\nTerminate all these processes?"):
        click.echo("Cancelled.")
        return

    killed = cleanup_orphaned_servers(verbose=True)
    click.echo(f"\n✓ Terminated {killed} process(es)")


@main.command("serve")
@click.option(
    "--port",
    envvar="DOCS_SERVER_PORT",
    default=6280,
    type=int,
    help="Port for docs-mcp-server (env: DOCS_SERVER_PORT) (default: 6280)",
)
@click.option(
    "--host",
    envvar="DOCS_SERVER_HOST",
    default="127.0.0.1",
    help="Host to bind (env: DOCS_SERVER_HOST) (default: 127.0.0.1)",
)
@click.option(
    "--docs-store-path",
    envvar="DOCS_MCP_STORE_PATH",
    type=click.Path(path_type=Path),
    help="Path for docs-mcp-server data storage (env: DOCS_MCP_STORE_PATH) (default: ./docs-data)",
)
def docs_serve(port: int, host: str, docs_store_path: Path | None):
    """Start the docs-mcp-server in standalone mode.

    This command launches a persistent docs-mcp-server instance that can be
    used for documentation search and retrieval.

    Examples:
        # Start with default settings
        docs serve

        # Start on custom port
        docs serve --port 6281

        # Start with custom data path
        docs serve --docs-store-path /path/to/docs-data
    """
    import asyncio
    import signal

    store_path = docs_store_path or get_store_path()
    store_path.mkdir(parents=True, exist_ok=True)

    click.echo("Starting docs-mcp-server...")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Store Path: {store_path.absolute()}")
    click.echo()

    # Clean up orphaned servers first
    cleanup_orphaned_servers(verbose=False)

    # Create server manager
    manager = DocsServerManager(
        default_port=port,
        store_path=store_path,
    )

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        click.echo("\nShutting down docs server...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    async def run_server():
        try:
            await manager.start_server()

            click.echo("✓ Docs server started successfully")
            click.echo(f"  URL: {manager.base_url}")
            click.echo(f"  MCP Endpoint: {manager.base_url}/mcp")
            click.echo()
            click.echo("Press Ctrl+C to stop the server")

            # Wait for shutdown signal
            await shutdown_event.wait()

        except Exception as e:
            click.echo(f"\n✗ Error starting server: {e}", err=True)
            sys.exit(1)
        finally:
            await manager.stop_server()
            click.echo("Server stopped")

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        click.echo("\nServer stopped")


if __name__ == "__main__":
    main()
