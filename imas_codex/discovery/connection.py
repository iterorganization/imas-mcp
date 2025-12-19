"""
Fabric-based connection management for remote facilities.

This module provides a wrapper around Fabric's Connection class that
integrates with the command sandbox and facility configuration.
"""

import io
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fabric import Connection
from invoke.runners import Result

from imas_codex.discovery.sandbox import CommandSandbox


@dataclass
class ScriptResult:
    """Result of executing an ephemeral script."""

    return_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if script executed successfully."""
        return self.return_code == 0

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n".join(parts)


@dataclass
class FacilityConnection:
    """
    Manages SSH connections to remote fusion facilities via Fabric.

    This class wraps Fabric's Connection with:
    - Command sandbox enforcement
    - Automatic timeout handling
    - Output size limiting
    - Connection lifecycle management

    Attributes:
        host: SSH host alias (from ~/.ssh/config) or hostname
        sandbox: Command sandbox for validation
        connect_timeout: Connection timeout in seconds
        connect_kwargs: Additional kwargs for Fabric Connection
    """

    host: str
    sandbox: CommandSandbox = field(default_factory=CommandSandbox)
    connect_timeout: int = 30
    connect_kwargs: dict = field(default_factory=dict)

    _connection: Connection | None = field(default=None, init=False, repr=False)

    @property
    def connection(self) -> Connection:
        """Get or create the Fabric connection."""
        if self._connection is None:
            self._connection = Connection(
                host=self.host,
                connect_timeout=self.connect_timeout,
                connect_kwargs=self.connect_kwargs,
            )
        return self._connection

    def open(self) -> None:
        """Open the SSH connection."""
        if not self.connection.is_connected:
            self.connection.open()

    def close(self) -> None:
        """Close the SSH connection."""
        if self._connection is not None and self._connection.is_connected:
            self._connection.close()

    @contextmanager
    def session(self) -> Generator["FacilityConnection", None, None]:
        """
        Context manager for connection lifecycle.

        Usage:
            with FacilityConnection("epfl").session() as conn:
                result = conn.run("ls -la")
        """
        try:
            self.open()
            yield self
        finally:
            self.close()

    def run(
        self,
        command: str,
        *,
        timeout: int | None = None,
        warn: bool = True,
        hide: bool = True,
        validate: bool = True,
    ) -> Result:
        """
        Run a command on the remote facility.

        Args:
            command: Shell command to execute
            timeout: Command timeout (uses sandbox default if None)
            warn: If True, don't raise on non-zero exit codes
            hide: If True, don't print stdout/stderr
            validate: If True, validate command against sandbox

        Returns:
            Invoke Result object with stdout, stderr, return_code

        Raises:
            ValueError: If command fails sandbox validation
            Exception: If connection or execution fails
        """
        if validate:
            is_valid, error = self.sandbox.validate(command)
            if not is_valid:
                raise ValueError(f"Command rejected by sandbox: {error}")

        # Wrap with timeout
        wrapped = self.sandbox.wrap_with_timeout(command, timeout)

        # Ensure connection is open
        self.open()

        return self.connection.run(wrapped, warn=warn, hide=hide)

    def run_unchecked(
        self,
        command: str,
        *,
        timeout: int | None = None,
        hide: bool = True,
    ) -> Result:
        """
        Run a command without sandbox validation.

        Use with caution - only for trusted, known-safe commands.

        Args:
            command: Shell command to execute
            timeout: Command timeout
            hide: If True, don't print stdout/stderr

        Returns:
            Invoke Result object
        """
        return self.run(command, timeout=timeout, hide=hide, validate=False, warn=True)

    def exists(self, path: str) -> bool:
        """Check if a path exists on the remote."""
        result = self.run(f"test -e {path}", warn=True, validate=False)
        return result.return_code == 0

    def is_file(self, path: str) -> bool:
        """Check if path is a regular file."""
        result = self.run(f"test -f {path}", warn=True, validate=False)
        return result.return_code == 0

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        result = self.run(f"test -d {path}", warn=True, validate=False)
        return result.return_code == 0

    def read_file(
        self,
        path: str,
        max_bytes: int | None = None,
    ) -> str:
        """
        Read contents of a remote file.

        Args:
            path: Path to the file
            max_bytes: Maximum bytes to read (default: sandbox limit)

        Returns:
            File contents as string
        """
        limit = max_bytes or self.sandbox.max_output_size
        result = self.run(f"head -c {limit} {path}")
        return result.stdout

    def list_dir(self, path: str = ".") -> list[str]:
        """
        List directory contents.

        Args:
            path: Directory path (default: current directory)

        Returns:
            List of entry names
        """
        result = self.run(f"ls -1 {path}")
        return [line for line in result.stdout.strip().split("\n") if line]

    def which(self, command: str) -> str | None:
        """
        Find the path to a command.

        Args:
            command: Command name to find

        Returns:
            Path to command or None if not found
        """
        result = self.run(f"which {command}", warn=True)
        if result.return_code == 0:
            return result.stdout.strip()
        return None

    def get_home(self) -> Path:
        """Get the remote user's home directory."""
        result = self.run("echo $HOME")
        return Path(result.stdout.strip())

    def get_cwd(self) -> Path:
        """Get the remote current working directory."""
        result = self.run("pwd")
        return Path(result.stdout.strip())

    def run_script(
        self,
        script: str,
        *,
        timeout: int | None = None,
        validate: bool = True,
    ) -> ScriptResult:
        """
        Execute an ephemeral bash script via stdin.

        The script is piped to `bash -s` and executed. This is the core
        mechanism for LLM-generated exploration scripts.

        Args:
            script: Bash script content to execute
            timeout: Execution timeout in seconds
            validate: If True, validate script against sandbox rules

        Returns:
            ScriptResult with stdout, stderr, and return code

        Raises:
            ValueError: If script fails sandbox validation
        """
        if validate:
            self.sandbox.validate_script(script)

        # Ensure connection is open
        self.open()

        # Prepare timeout
        t = timeout or self.sandbox.timeout

        # Execute script via stdin to bash -s
        # Using bash -s reads script from stdin
        result = self.connection.run(
            f"timeout {t} bash -s",
            in_stream=io.StringIO(script),
            warn=True,
            hide=True,
        )

        return ScriptResult(
            return_code=result.return_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
