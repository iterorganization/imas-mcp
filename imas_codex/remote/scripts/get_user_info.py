#!/usr/bin/env python3
"""Remote user info extraction script.

This script is executed on remote facilities via SSH. It extracts user
information from getent passwd for a list of usernames, with fallbacks.

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- Uses getent passwd (preferred) or /etc/passwd as fallback

Usage:
    echo '{"usernames": ["dubrovm", "abalestr"]}' | python3 get_user_info.py

Input (JSON on stdin):
    {
        "usernames": ["username1", "username2", ...]
    }

Output (JSON on stdout):
    {
        "users": [
            {
                "username": "dubrovm",
                "name": "Dubrov Maksim EXT",
                "home": "/home/dubrovm",
                "shell": "/bin/bash",
                "source": "getent"
            },
            ...
        ],
        "errors": ["username3: not found", ...]
    }

Name Format Notes:
- ITER format: "Last First [EXT]" (e.g., "Dubrov Maksim EXT")
- EPFL format: "First Last" (e.g., "Alessandro Balestri")
- The caller should handle format parsing based on facility
"""

import json
import os
import subprocess
import sys
from typing import Any


def has_command(cmd: str) -> bool:
    """Check if command exists in PATH."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for path_dir in path_dirs:
        if os.path.isfile(os.path.join(path_dir, cmd)):
            return True
    return False


def get_user_via_getent(username: str) -> dict[str, Any] | None:
    """Get user info via getent passwd (POSIX standard)."""
    try:
        proc = subprocess.run(
            ["getent", "passwd", username],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            # passwd format: username:password:uid:gid:gecos:home:shell
            parts = proc.stdout.strip().split(":")
            if len(parts) >= 6:
                return {
                    "username": parts[0],
                    "name": parts[4] if len(parts) > 4 else "",
                    "home": parts[5] if len(parts) > 5 else "",
                    "shell": parts[6] if len(parts) > 6 else "",
                    "source": "getent",
                }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_user_via_passwd_file(username: str) -> dict[str, Any] | None:
    """Get user info via /etc/passwd (fallback)."""
    try:
        with open("/etc/passwd") as f:
            for line in f:
                parts = line.strip().split(":")
                if parts and parts[0] == username and len(parts) >= 6:
                    return {
                        "username": parts[0],
                        "name": parts[4] if len(parts) > 4 else "",
                        "home": parts[5] if len(parts) > 5 else "",
                        "shell": parts[6] if len(parts) > 6 else "",
                        "source": "passwd",
                    }
    except (PermissionError, FileNotFoundError):
        pass
    return None


def get_user_via_id(username: str) -> dict[str, Any] | None:
    """Get minimal user info via id command (last resort)."""
    try:
        proc = subprocess.run(
            ["id", username],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            # id output: uid=1234(username) gid=1234(group) groups=...
            # We can at least confirm the user exists
            return {
                "username": username,
                "name": "",  # id doesn't provide GECOS
                "home": "",
                "shell": "",
                "source": "id",
            }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_user_info(username: str) -> dict[str, Any] | None:
    """Get user info with cascading fallbacks.

    Order of preference:
    1. getent passwd (standard POSIX, includes LDAP/NIS/SSSD)
    2. /etc/passwd file (local users only)
    3. id command (existence check only, no GECOS)
    """
    # Try getent first (handles LDAP, NIS, SSSD)
    info = get_user_via_getent(username)
    if info:
        return info

    # Fallback to /etc/passwd
    info = get_user_via_passwd_file(username)
    if info:
        return info

    # Last resort: id command (just confirms existence)
    info = get_user_via_id(username)
    if info:
        return info

    return None


def main() -> None:
    """Read usernames from stdin, lookup info, output JSON to stdout."""
    # Read configuration from stdin
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        result = {"error": f"Invalid JSON input: {e}", "users": [], "errors": []}
        print(json.dumps(result))
        sys.exit(1)

    usernames: list[str] = config.get("usernames", [])

    users: list[dict[str, Any]] = []
    errors: list[str] = []

    for username in usernames:
        info = get_user_info(username)
        if info:
            users.append(info)
        else:
            errors.append(f"{username}: not found")

    result = {"users": users, "errors": errors}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
