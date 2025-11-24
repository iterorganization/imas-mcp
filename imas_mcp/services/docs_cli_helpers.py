"""Helper utilities for docs-mcp-server CLI operations."""

import shutil
from pathlib import Path


def get_npx_executable() -> str:
    """Find npx executable or raise error.

    Returns:
        Path to npx executable (absolute path)

    Raises:
        RuntimeError: If npx is not found in PATH
    """
    npx_executable = shutil.which("npx")
    if not npx_executable:
        raise RuntimeError(
            "npx executable not found in PATH. "
            "Please ensure Node.js and npm are properly installed."
        )
    # Resolve to absolute path to avoid PATH issues in subprocesses
    return str(Path(npx_executable).resolve())


def build_docs_server_command(
    npx_path: str,
    command: str,
    *args: str,
    store_path: Path | None = None,
    port: int | None = None,
    host: str | None = None,
) -> list[str]:
    """Build standardized docs-mcp-server command.

    Args:
        npx_path: Path to npx executable
        command: Command to run (e.g., 'list', 'scrape', 'remove')
        *args: Additional arguments for the command
        store_path: Optional store path for server mode
        port: Optional port for server mode
        host: Optional host for server mode

    Returns:
        List of command arguments ready for subprocess
    """
    cmd = [
        npx_path,
        "--yes",  # Use long form for clarity
        "@arabold/docs-mcp-server@latest",
    ]

    # For server mode, add protocol/host/port flags before command
    if port is not None and host is not None:
        cmd.extend(
            [
                "--protocol",
                "http",
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
        if store_path:
            cmd.extend(["--store-path", str(store_path)])
    else:
        # For CLI mode, add command first, then args
        cmd.append(command)
        cmd.extend(args)

    return cmd
