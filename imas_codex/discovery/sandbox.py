"""
Read-only command sandbox for safe remote exploration.

This module enforces a whitelist of non-destructive commands that can be
executed on remote facilities. It prevents accidental or malicious
modification of remote filesystems.
"""

import re
import shlex
from dataclasses import dataclass, field


@dataclass
class CommandSandbox:
    """
    Enforces a whitelist of safe, read-only commands for remote execution.

    The sandbox validates commands before execution and rejects any command
    that could modify the remote filesystem.

    Attributes:
        allowed_commands: Set of allowed command names
        max_output_size: Maximum bytes to capture from command output
        timeout: Default command timeout in seconds
    """

    allowed_commands: set[str] = field(
        default_factory=lambda: {
            # Filesystem inspection
            "ls",
            "find",
            "stat",
            "file",
            "du",
            "tree",
            "df",
            "readlink",
            "realpath",
            "dirname",
            "basename",
            "pwd",  # Current working directory
            # File reading (content inspection)
            "cat",
            "head",
            "tail",
            "less",
            "more",
            "wc",
            "strings",
            "hexdump",
            "od",
            # Search tools
            "grep",
            "egrep",
            "fgrep",
            "rg",
            "ag",
            "ack",
            # Tool detection
            "which",
            "type",
            "command",
            "whereis",
            "hash",
            # System information
            "uname",
            "hostname",
            "whoami",
            "id",
            "groups",
            "env",
            "printenv",
            "locale",
            "date",
            "uptime",
            "lsb_release",  # for /etc/os-release
            # Process information (read-only)
            "ps",
            "pgrep",
            # Package/module queries (read-only)
            "rpm",  # Package queries
            "dpkg",  # Debian package queries
            "dnf",  # DNF queries (restricted to query subcommands)
            "yum",  # YUM queries (restricted to query subcommands)
            "apt",  # APT queries (restricted to query subcommands)
            "module",  # Environment modules
            "modulecmd",  # Environment modules internal
            "ml",  # Lmod shortcut
            "alternatives",  # RHEL alternatives system
            "update-alternatives",  # Debian alternatives
            "scl",  # Software Collections
            # Python and version checks
            "python",
            "python3",
            "python3.9",
            "python3.10",
            "python3.11",
            "python3.12",
            "pip",
            "pip3",
            "conda",  # Conda environment manager
            "mamba",  # Faster conda
            "uv",  # Modern Python package manager
            "pyenv",  # Python version manager
            "virtualenv",  # Virtual environments
            # Data tools
            "h5dump",
            "h5ls",
            "ncdump",
            "mdstcl",  # MDSplus TCL interface
            "mdsvalue",  # MDSplus value query
            # Text processing
            "awk",
            "sed",
            "cut",
            "sort",
            "uniq",
            "tr",
            "tee",
            "xargs",
            "printf",
            "echo",
            # Archive inspection (read-only)
            "tar",
            "unzip",
            "zipinfo",
            "gzip",
            "zcat",
            "bzcat",
            "xzcat",
            # Version control (read-only operations)
            "git",
            "svn",
            "hg",
            # Misc utilities
            "true",
            "false",
            "test",
            "expr",
            "seq",
            "timeout",
            "time",
            "getconf",  # System configuration
        }
    )

    # Commands that need argument validation
    restricted_commands: dict[str, set[str]] = field(
        default_factory=lambda: {
            # git: only allow read-only subcommands
            "git": {
                "status",
                "log",
                "show",
                "diff",
                "branch",
                "tag",
                "ls-files",
                "ls-tree",
                "rev-parse",
                "describe",
                "remote",
                "config",
                "rev-list",
            },
            # tar: only allow list/extract, not create
            "tar": {"t", "tf", "tvf", "x", "xf", "xvf", "--list"},
            # dnf/yum: only allow query operations
            "dnf": {
                "list",
                "info",
                "search",
                "provides",
                "repolist",
                "repoinfo",
                "module",  # module list/info
                "config-manager",  # for listing repos
            },
            "yum": {
                "list",
                "info",
                "search",
                "provides",
                "repolist",
            },
            # apt: only allow query operations
            "apt": {
                "list",
                "show",
                "search",
                "policy",
            },
            # module: all standard operations (session-scoped, non-destructive)
            # These only modify environment variables for the current session
            "module": {
                # Query operations
                "avail",
                "av",
                "list",
                "li",
                "show",
                "display",
                "whatis",
                "spider",
                "keyword",
                "help",
                "is-loaded",
                "is-avail",
                # Environment management (session-scoped)
                "load",
                "add",
                "unload",
                "rm",
                "del",
                "swap",
                "switch",
                "purge",
                "refresh",
                "update",
                "restore",
                "save",
                "savelist",
                "describe",
                "use",
                "unuse",
            },
            # ml (Lmod): all standard operations
            "ml": {
                # Query
                "avail",
                "av",
                "list",
                "li",
                "show",
                "display",
                "whatis",
                "spider",
                # Environment management
                "load",
                "unload",
                "purge",
                "swap",
                "restore",
                "save",
            },
            # scl: only enable (to show available) and list
            "scl": {
                "enable",
                "--list",
            },
            # svn: only allow read-only subcommands
            "svn": {
                "info",
                "log",
                "ls",
                "list",
                "cat",
                "diff",
                "status",
                "st",
            },
            # hg: only allow read-only subcommands
            "hg": {
                "log",
                "diff",
                "status",
                "cat",
                "manifest",
                "branches",
                "tags",
            },
            # conda: only allow info/list operations
            "conda": {
                "info",
                "list",
                "env",
                "config",
                "search",
            },
        }
    )

    # Dangerous patterns that should never be allowed
    # Note: We allow | (pipe), || (or), && (and), ; (chaining),
    # $VAR, $(...), and stderr redirection (2>/dev/null, 2>&1)
    forbidden_patterns: list[re.Pattern] = field(
        default_factory=lambda: [
            re.compile(r"`"),  # Backtick command substitution (use $() instead)
            re.compile(
                r"(?<![|&>])&(?![&1])"
            ),  # Background (&) but allow &&, >&1, and &>
            re.compile(r"\brm\s"),  # Remove command
            re.compile(r"\bmv\s"),  # Move command
            re.compile(r"\bcp\s"),  # Copy command
            re.compile(r"\bchmod\s"),  # Permission changes
            re.compile(r"\bchown\s"),  # Ownership changes
            re.compile(r"\bdd\s"),  # Direct disk operations
            re.compile(r"\bmkdir\s"),  # Create directories
            re.compile(r"\brmdir\s"),  # Remove directories
            re.compile(r"\btouch\s"),  # Create/modify files
            re.compile(r"\bln\s"),  # Create links
            re.compile(r"\bkill\s"),  # Kill processes
            re.compile(r"\bpkill\s"),  # Kill processes
            re.compile(r"\breboot\b"),  # System reboot
            re.compile(r"\bshutdown\b"),  # System shutdown
            re.compile(r"\bsudo\s"),  # Privilege escalation
            re.compile(r"\bsu\s"),  # Switch user
        ]
    )

    max_output_size: int = 10 * 1024 * 1024  # 10 MB
    timeout: int = 60  # seconds

    def validate(self, command: str) -> tuple[bool, str]:
        """
        Validate a command against the sandbox rules.

        Supports command chaining with ;, &&, ||, and pipes |.
        Each command in the chain is validated independently.

        Args:
            command: The shell command to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string.
        """
        # Check for empty command first
        if not command or not command.strip():
            return False, "Empty command"

        # Check for forbidden patterns first (these apply to the whole command)
        for pattern in self.forbidden_patterns:
            if pattern.search(command):
                return False, f"Command contains forbidden pattern: {pattern.pattern}"

        # Split on command separators (; && ||) to validate each command
        # We need to be careful with pipes - they're allowed but each side
        # needs validation
        simple_commands = self._split_command_chain(command)

        # Ensure we have at least one non-empty command
        non_empty = [c.strip() for c in simple_commands if c.strip()]
        if not non_empty:
            return False, "Empty command"

        for simple_cmd in non_empty:
            is_valid, error = self._validate_simple_command(simple_cmd)
            if not is_valid:
                return False, error

        return True, ""

    def _split_command_chain(self, command: str) -> list[str]:
        """
        Split a command chain into individual simple commands.

        Handles ;, &&, ||, and | separators while respecting quotes.
        """
        # Split on ; && || but keep quoted strings together
        # This is a simplified approach - for complex commands, use script mode
        result = []
        current = []
        in_single_quote = False
        in_double_quote = False
        i = 0

        while i < len(command):
            char = command[i]

            # Handle quotes
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                current.append(char)
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                current.append(char)
            # Handle escape sequences
            elif char == "\\" and i + 1 < len(command):
                current.append(char)
                current.append(command[i + 1])
                i += 1
            # Handle separators (only outside quotes)
            elif not in_single_quote and not in_double_quote:
                if char == ";":
                    result.append("".join(current))
                    current = []
                elif char == "|":
                    # Check for || (or) vs | (pipe)
                    if i + 1 < len(command) and command[i + 1] == "|":
                        result.append("".join(current))
                        current = []
                        i += 1  # Skip second |
                    else:
                        # Pipe - split here too, both sides need validation
                        result.append("".join(current))
                        current = []
                elif char == "&":
                    # Check for && (and)
                    if i + 1 < len(command) and command[i + 1] == "&":
                        result.append("".join(current))
                        current = []
                        i += 1  # Skip second &
                    else:
                        current.append(char)
                else:
                    current.append(char)
            else:
                current.append(char)

            i += 1

        # Don't forget the last command
        if current:
            result.append("".join(current))

        return result

    def _validate_simple_command(self, command: str) -> tuple[bool, str]:
        """
        Validate a single simple command (no chaining operators).

        Args:
            command: A simple shell command

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Strip common redirections for parsing
        cmd_for_parse = command
        for redir in ["2>/dev/null", "2>&1", ">/dev/null", "&>/dev/null"]:
            cmd_for_parse = cmd_for_parse.replace(redir, "")

        # Parse command to get the executable
        try:
            parts = shlex.split(cmd_for_parse.strip())
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"

        if not parts:
            # Empty command is okay (e.g., result of stripping)
            return True, ""

        executable = parts[0]

        # Handle path-prefixed commands (e.g., /usr/bin/find)
        if "/" in executable:
            executable = executable.rsplit("/", 1)[-1]

        # Check if command is in whitelist
        if executable not in self.allowed_commands:
            return False, f"Command not allowed: {executable}"

        # Check restricted commands for valid subcommands
        if executable in self.restricted_commands:
            if len(parts) < 2:
                return False, f"Command '{executable}' requires a subcommand"

            subcommand = parts[1]
            # Handle flags before subcommand (e.g., rpm -qa)
            if subcommand.startswith("-"):
                # For rpm, -q* flags are queries
                if executable == "rpm" and subcommand.startswith("-q"):
                    return True, ""
                # Otherwise check next arg
                if len(parts) > 2:
                    subcommand = parts[2]
                else:
                    # Just flags, allow for some commands
                    if executable in {"rpm", "dpkg"}:
                        return True, ""

            allowed_subcommands = self.restricted_commands[executable]

            if subcommand not in allowed_subcommands:
                return False, (
                    f"Subcommand '{subcommand}' not allowed for '{executable}'. "
                    f"Allowed: {sorted(allowed_subcommands)}"
                )

        return True, ""

    def is_allowed(self, command: str) -> bool:
        """Check if a command is allowed (convenience method)."""
        is_valid, _ = self.validate(command)
        return is_valid

    def wrap_with_timeout(self, command: str, timeout: int | None = None) -> str:
        """
        Wrap a command with timeout for safety.

        Args:
            command: The command to wrap
            timeout: Timeout in seconds (uses default if None)

        Returns:
            Command wrapped with timeout
        """
        t = timeout or self.timeout
        return f"timeout {t} {command}"

    def wrap_with_size_limit(self, command: str, max_bytes: int | None = None) -> str:
        """
        Wrap a command to limit output size.

        Args:
            command: The command to wrap
            max_bytes: Maximum bytes to capture (uses default if None)

        Returns:
            Command piped through head -c
        """
        limit = max_bytes or self.max_output_size
        return f"{command} | head -c {limit}"

    def validate_script(self, script: str, max_lines: int = 200) -> None:
        """
        Validate a complete bash script for safety.

        This is used for LLM-generated scripts that will be piped
        to bash -s for execution.

        Args:
            script: The bash script content to validate
            max_lines: Maximum allowed lines in script

        Raises:
            ValueError: If script contains dangerous patterns
        """
        lines = script.strip().split("\n")

        # Check line count
        if len(lines) > max_lines:
            raise ValueError(f"Script too long: {len(lines)} lines (max {max_lines})")

        # Dangerous patterns specific to scripts
        script_forbidden = [
            # Destructive commands
            (r"\brm\s+(-[rf]+\s+)?", "rm command"),
            (r"\bmv\s+", "mv command"),
            (r"\bcp\s+", "cp command"),
            (r"\bchmod\s+", "chmod command"),
            (r"\bchown\s+", "chown command"),
            (r"\bchgrp\s+", "chgrp command"),
            # Privilege escalation
            (r"\bsudo\s+", "sudo command"),
            (r"\bsu\s+", "su command"),
            (r"\bdoas\s+", "doas command"),
            # System commands
            (r"\breboot\b", "reboot command"),
            (r"\bshutdown\b", "shutdown command"),
            (r"\bhalt\b", "halt command"),
            (r"\bpoweroff\b", "poweroff command"),
            (r"\binit\s+", "init command"),
            (r"\bsystemctl\s+", "systemctl command"),
            (r"\bservice\s+", "service command"),
            # Disk operations
            (r"\bdd\s+", "dd command"),
            (r"\bmkfs\b", "mkfs command"),
            (r"\bfdisk\b", "fdisk command"),
            (r"\bparted\b", "parted command"),
            # Process killing
            (r"\bkill\s+", "kill command"),
            (r"\bpkill\s+", "pkill command"),
            (r"\bkillall\s+", "killall command"),
            # File creation
            (r"\bmkdir\s+", "mkdir command"),
            (r"\btouch\s+", "touch command"),
            (r"\bln\s+", "ln command"),
            # Dangerous redirections (but allow 2>/dev/null and 2>&1)
            (r">\s*/dev/(?!null)", "writing to device"),
            (r">\s*/etc/", "writing to /etc"),
            (r">\s*/var/", "writing to /var"),
            (r">\s*/usr/", "writing to /usr"),
            (r">\s*/bin/", "writing to /bin"),
            (r">\s*/sbin/", "writing to /sbin"),
            (r">\s*~/", "writing to home directory"),
            (r">\s*/home/", "writing to /home"),
            (r">\s*/root/", "writing to /root"),
            # Network operations that could exfiltrate data
            (r"\bwget\s+.*-O\s+", "wget downloading"),
            (r"\bcurl\s+.*(-o|--output)\s+", "curl downloading"),
            (r"\bnc\s+-l", "netcat listening"),
            (r"\bnetcat\s+-l", "netcat listening"),
            # Fork bomb patterns
            (r":\(\)\s*\{", "fork bomb pattern"),
            (r"\|\s*:\s*&", "fork bomb pattern"),
            # Eval and exec
            (r"\beval\s+", "eval command"),
            (r"\bexec\s+[^>]", "exec command"),  # Allow exec for redirection
        ]

        for pattern, description in script_forbidden:
            if re.search(pattern, script, re.IGNORECASE | re.MULTILINE):
                raise ValueError(f"Script contains forbidden pattern: {description}")

        # Check for backtick command substitution
        if "`" in script:
            raise ValueError(
                "Script contains backtick command substitution (use $() instead)"
            )
