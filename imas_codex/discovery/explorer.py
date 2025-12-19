"""
Remote facility explorer for interactive investigation.

This module provides a high-level interface for exploring remote facilities,
discovering available tools, and gathering environment information.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.discovery.connection import FacilityConnection


@dataclass
class ToolInfo:
    """Information about a discovered tool."""

    name: str
    path: str | None
    version: str | None
    available: bool


@dataclass
class RemoteEnvironment:
    """Snapshot of a remote facility's environment."""

    host: str
    probe_timestamp: datetime
    available_tools: list[ToolInfo]
    python_version: str | None
    shell: str | None
    os_info: str | None
    data_libraries: list[str]


@dataclass
class RemoteExplorer:
    """
    High-level interface for exploring remote fusion facilities.

    Provides methods to probe the remote environment, discover available
    tools, and gather system information.

    Attributes:
        connection: FacilityConnection instance
    """

    connection: FacilityConnection

    # Tools to probe for, organized by category
    tool_categories: dict[str, list[str]] = field(
        default_factory=lambda: {
            "search": ["rg", "ag", "grep", "ack", "fgrep", "egrep"],
            "tree": ["tree", "find", "fd", "exa", "ls"],
            "code_analysis": ["tree-sitter", "ctags", "cscope", "clang"],
            "data_access": ["h5dump", "h5ls", "ncdump", "mdsplus"],
            "text_processing": ["jq", "yq", "awk", "sed", "cut", "sort"],
            "compression": ["gzip", "bzip2", "xz", "zstd", "tar", "unzip"],
            "version_control": ["git", "svn", "hg"],
            "python": ["python3", "python", "pip3", "pip", "uv", "conda"],
            "system": ["file", "stat", "du", "df", "lsb_release", "uname"],
        }
    )

    def probe_tool(self, tool_name: str) -> ToolInfo:
        """
        Probe for a specific tool on the remote system.

        Args:
            tool_name: Name of the tool to probe

        Returns:
            ToolInfo with availability and version information
        """
        path = self.connection.which(tool_name)
        available = path is not None
        version = None

        if available:
            # Try to get version
            version = self._get_tool_version(tool_name)

        return ToolInfo(
            name=tool_name,
            path=path,
            version=version,
            available=available,
        )

    def _get_tool_version(self, tool_name: str) -> str | None:
        """Try to get version string for a tool."""
        # Common version flags to try
        version_flags = ["--version", "-V", "-v", "version"]

        for flag in version_flags:
            try:
                result = self.connection.run(
                    f"{tool_name} {flag} 2>&1 | head -1",
                    timeout=5,
                    warn=True,
                )
                if result.return_code == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                continue

        return None

    def probe_all_tools(self) -> dict[str, list[ToolInfo]]:
        """
        Probe all tools in all categories.

        Returns:
            Dict mapping category names to lists of ToolInfo
        """
        results: dict[str, list[ToolInfo]] = {}

        for category, tools in self.tool_categories.items():
            results[category] = [self.probe_tool(tool) for tool in tools]

        return results

    def get_python_version(self) -> str | None:
        """Get Python version on remote system."""
        for python in ["python3", "python"]:
            result = self.connection.run(
                f"{python} --version 2>&1",
                warn=True,
            )
            if result.return_code == 0:
                return result.stdout.strip()
        return None

    def get_shell(self) -> str | None:
        """Get the default shell on remote system."""
        result = self.connection.run("echo $SHELL", warn=True)
        if result.return_code == 0:
            return result.stdout.strip()
        return None

    def get_os_info(self) -> str | None:
        """Get OS information from remote system."""
        # Try lsb_release first
        result = self.connection.run(
            "lsb_release -d 2>/dev/null || cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'",
            warn=True,
        )
        if result.return_code == 0 and result.stdout.strip():
            # Clean up lsb_release output
            output = result.stdout.strip()
            if output.startswith("Description:"):
                output = output[12:].strip()
            return output

        # Fallback to uname
        result = self.connection.run("uname -a", warn=True)
        if result.return_code == 0:
            return result.stdout.strip()

        return None

    def probe_data_libraries(self) -> list[str]:
        """Check for data access libraries (MDSplus, HDF5, etc.)."""
        libraries = []

        # Check library versions using Python imports
        # Use newline instead of semicolon for multi-statement Python
        lib_checks = [
            ("MDSplus", "from MDSplus import __version__ as v\nprint(v)"),
            ("h5py", "from h5py import __version__ as v\nprint(v)"),
            ("netCDF4", "from netCDF4 import __version__ as v\nprint(v)"),
            ("xarray", "from xarray import __version__ as v\nprint(v)"),
        ]

        for lib_name, check_code in lib_checks:
            # Use echo with pipe to avoid semicolons in shell command
            result = self.connection.run(
                f"python3 -c '{check_code}' 2>/dev/null",
                warn=True,
            )
            if result.return_code == 0 and result.stdout.strip():
                libraries.append(f"{lib_name} {result.stdout.strip()}")

        return libraries

    def probe_environment(self) -> RemoteEnvironment:
        """
        Perform a complete environment probe.

        Returns:
            RemoteEnvironment with all discovered information
        """
        tool_results = self.probe_all_tools()

        # Flatten tools list
        all_tools = []
        for tools in tool_results.values():
            all_tools.extend(tools)

        return RemoteEnvironment(
            host=self.connection.host,
            probe_timestamp=datetime.now(UTC),
            available_tools=all_tools,
            python_version=self.get_python_version(),
            shell=self.get_shell(),
            os_info=self.get_os_info(),
            data_libraries=self.probe_data_libraries(),
        )

    def find_directories(
        self,
        path: str,
        pattern: str | None = None,
        max_depth: int = 3,
    ) -> list[str]:
        """
        Find directories matching a pattern.

        Args:
            path: Root path to search
            pattern: Optional name pattern (glob)
            max_depth: Maximum depth to search

        Returns:
            List of matching directory paths
        """
        cmd = f"find {path} -maxdepth {max_depth} -type d"
        if pattern:
            cmd += f" -name '{pattern}'"
        cmd += " 2>/dev/null"

        result = self.connection.run(cmd, warn=True)
        if result.return_code != 0:
            return []

        return [line for line in result.stdout.strip().split("\n") if line]

    def find_files(
        self,
        path: str,
        pattern: str | None = None,
        max_depth: int = 3,
        max_results: int = 1000,
    ) -> list[str]:
        """
        Find files matching a pattern.

        Args:
            path: Root path to search
            pattern: Optional name pattern (glob)
            max_depth: Maximum depth to search
            max_results: Maximum number of results

        Returns:
            List of matching file paths
        """
        cmd = f"find {path} -maxdepth {max_depth} -type f"
        if pattern:
            cmd += f" -name '{pattern}'"
        cmd += f" 2>/dev/null | head -n {max_results}"

        result = self.connection.run(cmd, warn=True)
        if result.return_code != 0:
            return []

        return [line for line in result.stdout.strip().split("\n") if line]

    def search_content(
        self,
        path: str,
        pattern: str,
        file_pattern: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search file contents using ripgrep or grep.

        Args:
            path: Root path to search
            pattern: Regex pattern to search for
            file_pattern: Optional glob pattern for files
            max_results: Maximum number of results

        Returns:
            List of dicts with file, line_number, content
        """
        # Prefer ripgrep, fall back to grep
        rg_path = self.connection.which("rg")

        if rg_path:
            cmd = f"rg --json -m {max_results} '{pattern}' {path}"
            if file_pattern:
                cmd += f" -g '{file_pattern}'"
        else:
            cmd = f"grep -rn '{pattern}' {path}"
            if file_pattern:
                cmd += f" --include='{file_pattern}'"
            cmd += f" | head -n {max_results}"

        result = self.connection.run(cmd, warn=True)
        if result.return_code != 0:
            return []

        # Parse results (simplified - just return raw lines for now)
        matches = []
        for line in result.stdout.strip().split("\n"):
            if line:
                matches.append({"raw": line})

        return matches

    def tree(self, path: str, max_depth: int = 2) -> str:
        """
        Get a tree view of a directory.

        Args:
            path: Root path
            max_depth: Maximum depth

        Returns:
            Tree output as string
        """
        tree_path = self.connection.which("tree")

        if tree_path:
            cmd = f"tree -L {max_depth} {path}"
        else:
            # Fallback using find
            cmd = f"find {path} -maxdepth {max_depth} -print | head -500"

        result = self.connection.run(cmd, warn=True)
        return result.stdout

    def disk_usage(self, path: str) -> dict[str, Any]:
        """
        Get disk usage for a path.

        Args:
            path: Path to check

        Returns:
            Dict with size information
        """
        result = self.connection.run(f"du -sh {path} 2>/dev/null", warn=True)
        if result.return_code != 0:
            return {"path": path, "error": "Could not get disk usage"}

        parts = result.stdout.strip().split("\t")
        return {
            "path": path,
            "size": parts[0] if parts else "unknown",
        }
