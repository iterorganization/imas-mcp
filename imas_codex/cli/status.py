"""CLI status command - show current node hostname and load."""

from __future__ import annotations

import socket
import subprocess

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_load_info() -> dict:
    """Get CPU load and memory usage for the current node.

    Returns dict with keys: hostname, load_1m, load_5m, load_15m,
    cpu_count, mem_used_mb, mem_total_mb, users.
    """
    import os

    hostname = socket.gethostname()
    info: dict = {"hostname": hostname}

    # CPU load averages
    try:
        load_1, load_5, load_15 = os.getloadavg()
        info["load_1m"] = load_1
        info["load_5m"] = load_5
        info["load_15m"] = load_15
    except OSError:
        info["load_1m"] = info["load_5m"] = info["load_15m"] = 0.0

    # CPU count
    try:
        info["cpu_count"] = os.cpu_count() or 1
    except Exception:
        info["cpu_count"] = 1

    # Memory from /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    meminfo[key] = int(parts[1])  # kB
        total_kb = meminfo.get("MemTotal", 0)
        available_kb = meminfo.get("MemAvailable", 0)
        info["mem_total_mb"] = total_kb / 1024
        info["mem_used_mb"] = (total_kb - available_kb) / 1024
    except Exception:
        info["mem_total_mb"] = 0
        info["mem_used_mb"] = 0

    # Logged-in users
    try:
        result = subprocess.run(
            ["who"], capture_output=True, text=True, timeout=5
        )
        info["users"] = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
    except Exception:
        info["users"] = 0

    return info


def _get_remote_load_info(ssh_host: str, timeout: int = 10) -> dict | None:
    """Get load info from a remote node via SSH.

    Returns dict with same keys as _get_load_info(), or None on failure.
    """
    # Single SSH call to get all info at once
    script = (
        "hostname -f; "
        "cat /proc/loadavg; "
        "nproc; "
        "awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} "
        "END{printf \"%d %d\\n\", t, a}' /proc/meminfo; "
        "who | wc -l"
    )
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                "-o", "StrictHostKeyChecking=accept-new",
                ssh_host,
                script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().splitlines()
        if len(lines) < 4:
            return None

        hostname = lines[0].strip()
        loadavg_parts = lines[1].split()
        cpu_count = int(lines[2].strip())
        mem_parts = lines[3].split()
        users = int(lines[4].strip()) if len(lines) > 4 else 0

        total_kb = int(mem_parts[0]) if len(mem_parts) >= 2 else 0
        avail_kb = int(mem_parts[1]) if len(mem_parts) >= 2 else 0

        return {
            "hostname": hostname,
            "load_1m": float(loadavg_parts[0]),
            "load_5m": float(loadavg_parts[1]),
            "load_15m": float(loadavg_parts[2]),
            "cpu_count": cpu_count,
            "mem_total_mb": total_kb / 1024,
            "mem_used_mb": (total_kb - avail_kb) / 1024,
            "users": users,
        }
    except (subprocess.TimeoutExpired, Exception):
        return None


def _colored_bar(used: float, limit: float, width: int = 20) -> str:
    """Render a colored usage bar: [████████░░░░] 40%."""
    if limit <= 0:
        return ""
    ratio = min(used / limit, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    if ratio > 0.8:
        color = "red"
    elif ratio > 0.5:
        color = "yellow"
    else:
        color = "green"
    bar = click.style("█" * filled, fg=color) + click.style("░" * empty, dim=True)
    pct = click.style(f"{ratio * 100:.0f}%", fg=color)
    return f"[{bar}] {pct}"


def _format_load_row(info: dict) -> list[str]:
    """Format a load info dict into a table row."""
    cpu_count = info.get("cpu_count", 1)
    load_1m = info.get("load_1m", 0)
    load_5m = info.get("load_5m", 0)

    # CPU load as percentage of total cores
    cpu_pct = (load_1m / cpu_count) * 100 if cpu_count > 0 else 0
    cpu_bar = _colored_bar(load_1m, cpu_count)

    # Memory bar
    mem_total = info.get("mem_total_mb", 0)
    mem_used = info.get("mem_used_mb", 0)
    mem_bar = _colored_bar(mem_used, mem_total)
    mem_label = f"{mem_used / 1024:.1f}/{mem_total / 1024:.0f} GB"

    # Load averages
    load_str = f"{load_1m:.1f} / {load_5m:.1f} / {info.get('load_15m', 0):.1f}"

    users = str(info.get("users", 0))

    return [
        info.get("hostname", "unknown"),
        f"{cpu_bar}  {load_str} ({cpu_count} cores)",
        f"{mem_bar}  {mem_label}",
        users,
    ]


@click.command("status")
def status() -> None:
    """Show current node hostname and system load.

    Displays CPU load, memory usage, and user count for the node
    this CLI session is running on.

    Examples:
        imas-codex status
    """
    info = _get_load_info()

    hostname = info["hostname"]
    click.echo(f"\nNode: {click.style(hostname, fg='cyan', bold=True)}")

    cpu_count = info["cpu_count"]
    load_1m = info["load_1m"]
    load_5m = info["load_5m"]
    load_15m = info["load_15m"]

    # CPU load bar (as fraction of total cores)
    cpu_bar = _colored_bar(load_1m, cpu_count)
    click.echo(
        f"  CPU:  {cpu_bar}  "
        f"load {load_1m:.1f} / {load_5m:.1f} / {load_15m:.1f}  "
        f"({cpu_count} cores)"
    )

    # Memory bar
    mem_total = info["mem_total_mb"]
    mem_used = info["mem_used_mb"]
    if mem_total > 0:
        mem_bar = _colored_bar(mem_used, mem_total)
        click.echo(
            f"  Mem:  {mem_bar}  "
            f"{mem_used / 1024:.1f} / {mem_total / 1024:.0f} GB"
        )

    # Users
    click.echo(f"  Users: {info['users']}")
    click.echo()


def discover_login_nodes(ssh_host: str, timeout: int = 10) -> list[str]:
    """Discover available login nodes by reading /etc/hosts on the facility.

    Parses /etc/hosts for nodes matching the facility's login_nodes patterns
    (e.g. 98dci4-srv-*), excluding GPU and matlab nodes.

    Args:
        ssh_host: SSH host alias to connect through.
        timeout: SSH timeout.

    Returns:
        List of short hostnames (e.g. ["98dci4-srv-1001", ...]).
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                ssh_host,
                "grep '98dci4-srv-' /etc/hosts | awk '{print $2}'",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0:
            return []

        nodes = []
        for line in result.stdout.strip().splitlines():
            fqdn = line.strip()
            # Filter to login nodes only (skip gpu, matlab)
            if fqdn and "srv" in fqdn and "gpu" not in fqdn:
                # Extract short hostname
                short = fqdn.split(".")[0]
                if short not in nodes:
                    nodes.append(short)
        return sorted(nodes)
    except Exception:
        return []
