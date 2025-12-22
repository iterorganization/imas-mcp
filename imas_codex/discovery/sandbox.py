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
            # Python and version checks
            "python",
            "python3",
            "python3.12",
            "pip",
            "pip3",
            # Data tools
            "h5dump",
            "h5ls",
            "ncdump",
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
            # Version control (read-only operations)
            "git",
            # Misc utilities
            "true",
            "false",
            "test",
            "expr",
            "seq",
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
        }
    )

    # Dangerous patterns that should never be allowed
    # Note: We allow | (pipe), || (or), $VAR for environment variables,
    # and 2>/dev/null or 2>&1 for stderr redirection
    forbidden_patterns: list[re.Pattern] = field(
        default_factory=lambda: [
            re.compile(r";"),  # Command chaining with semicolon
            re.compile(r"`"),  # Backtick command substitution
            re.compile(r"\$\("),  # Command substitution $(...)
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

        Args:
            command: The shell command to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string.
        """
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern.search(command):
                return False, f"Command contains forbidden pattern: {pattern.pattern}"

        # Parse command to get the executable
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"

        if not parts:
            return False, "Empty command"

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
