"""Host CLI - node load monitoring, process display, and SSH target management.

Unified command for monitoring compute node health and managing SSH jump targets.

\b
Usage:
  imas-codex host                      Local node load + imas-codex processes
  imas-codex host iter                 Survey ITER login nodes with load + processes
  imas-codex host iter --set-default 3
                                       Set iter SSH target to 3rd node
  imas-codex host iter --set-default   Auto-select least loaded node

Login nodes are discovered dynamically from /etc/hosts on the facility
gateway — no individual node configuration needed.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import socket
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


# ── Load info helpers ────────────────────────────────────────────────────


def _get_load_info() -> dict:
    """Get CPU load and memory usage for the current node."""
    hostname = socket.gethostname()
    info: dict = {"hostname": hostname}

    try:
        load_1, load_5, load_15 = os.getloadavg()
        info["load_1m"] = load_1
        info["load_5m"] = load_5
        info["load_15m"] = load_15
    except OSError:
        info["load_1m"] = info["load_5m"] = info["load_15m"] = 0.0

    try:
        info["cpu_count"] = os.cpu_count() or 1
    except Exception:
        info["cpu_count"] = 1

    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    meminfo[key] = int(parts[1])
        total_kb = meminfo.get("MemTotal", 0)
        available_kb = meminfo.get("MemAvailable", 0)
        info["mem_total_mb"] = total_kb / 1024
        info["mem_used_mb"] = (total_kb - available_kb) / 1024
    except Exception:
        info["mem_total_mb"] = 0
        info["mem_used_mb"] = 0

    try:
        result = subprocess.run(
            ["who"], capture_output=True, text=True, timeout=5
        )
        info["users"] = (
            len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
        )
    except Exception:
        info["users"] = 0

    return info


# ── Process detection ────────────────────────────────────────────────────

# Patterns that identify imas-codex processes in ps output.
_CODEX_PATTERNS = [
    "imas-codex",
    "imas_codex",
    "litellm",
    "neo4j",
    "sentence-transformers",
    "uvicorn.*embed",
]

_CODEX_RE = re.compile("|".join(_CODEX_PATTERNS), re.IGNORECASE)

# ps column spec: user pid %cpu %mem vsz rss etimes args
_PS_CMD_LOCAL = ["ps", "-eo", "user,pid,%cpu,%mem,vsz,rss,etimes,args"]
_PS_CMD_REMOTE = "ps -eo user,pid,%cpu,%mem,vsz,rss,etimes,args"


def _format_age(seconds: int) -> str:
    """Format elapsed seconds as human-readable age.

    Examples: "2d 5h", "3h 12m", "45m", "30s".
    """
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_m = minutes % 60
    if hours < 24:
        return f"{hours}h {remaining_m}m" if remaining_m else f"{hours}h"
    days = hours // 24
    remaining_h = hours % 24
    return f"{days}d {remaining_h}h" if remaining_h else f"{days}d"


def _parse_duration(spec: str) -> int:
    """Parse a duration string like '1h', '2d', '30m', '1d12h' to seconds.

    Raises ValueError for invalid input.
    """
    pattern = re.compile(r"(\d+)\s*([smhd])", re.IGNORECASE)
    matches = pattern.findall(spec)
    if not matches:
        raise ValueError(
            f"Invalid duration: '{spec}'. "
            f"Use combinations of <N>s, <N>m, <N>h, <N>d "
            f"(e.g. '1h', '2d', '30m', '1d12h')"
        )
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    total = 0
    for value, unit in matches:
        total += int(value) * multipliers[unit.lower()]
    return total


def _get_codex_processes_local() -> list[dict]:
    """Find imas-codex related processes on the local node."""
    try:
        result = subprocess.run(
            _PS_CMD_LOCAL,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        procs = _parse_ps_output(result.stdout)
        # Exclude this process and its parent (uv run wrapper)
        self_pids = {str(os.getpid()), str(os.getppid())}
        return [p for p in procs if p["pid"] not in self_pids]
    except Exception:
        return []


def _parse_ps_output(ps_output: str) -> list[dict]:
    """Parse 'ps -eo user,pid,%cpu,%mem,vsz,rss,etimes,args' output."""
    procs = []
    lines = ps_output.strip().splitlines()
    if not lines:
        return procs
    for line in lines[1:]:  # Skip header
        if not _CODEX_RE.search(line):
            continue
        # Skip the ps command itself and grep
        if "ps -eo" in line or "ps aux" in line or "grep" in line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        try:
            age_seconds = int(parts[6])
        except (ValueError, IndexError):
            age_seconds = 0
        procs.append({
            "user": parts[0],
            "pid": parts[1],
            "cpu": parts[2],
            "mem": parts[3],
            "vsz_mb": int(parts[4]) / 1024,
            "rss_mb": int(parts[5]) / 1024,
            "age_seconds": age_seconds,
            "command": parts[7][:120],
        })
    return procs


# ── Display helpers ──────────────────────────────────────────────────────


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
    bar = click.style("█" * filled, fg=color) + click.style(
        "░" * empty, dim=True
    )
    pct = click.style(f"{ratio * 100:.0f}%", fg=color)
    return f"[{bar}] {pct}"


def _format_load_row(info: dict) -> list[str]:
    """Format a load info dict into a table row."""
    cpu_count = info.get("cpu_count", 1)
    load_1m = info.get("load_1m", 0)
    load_5m = info.get("load_5m", 0)

    cpu_bar = _colored_bar(load_1m, cpu_count)
    mem_total = info.get("mem_total_mb", 0)
    mem_used = info.get("mem_used_mb", 0)
    mem_bar = _colored_bar(mem_used, mem_total)
    mem_label = f"{mem_used / 1024:.1f}/{mem_total / 1024:.0f} GB"
    load_str = f"{load_1m:.1f} / {load_5m:.1f} / {info.get('load_15m', 0):.1f}"
    users = str(info.get("users", 0))

    return [
        info.get("hostname", "unknown"),
        f"{cpu_bar}  {load_str} ({cpu_count} cores)",
        f"{mem_bar}  {mem_label}",
        users,
    ]


def _show_local_load(info: dict) -> None:
    """Print local node load in compact format."""
    hostname = info["hostname"]
    click.echo(f"\nNode: {click.style(hostname, fg='cyan', bold=True)}")

    cpu_count = info["cpu_count"]
    load_1m = info["load_1m"]
    load_5m = info["load_5m"]
    load_15m = info["load_15m"]

    cpu_bar = _colored_bar(load_1m, cpu_count)
    click.echo(
        f"  CPU:  {cpu_bar}  "
        f"load {load_1m:.1f} / {load_5m:.1f} / {load_15m:.1f}  "
        f"({cpu_count} cores)"
    )

    mem_total = info["mem_total_mb"]
    mem_used = info["mem_used_mb"]
    if mem_total > 0:
        mem_bar = _colored_bar(mem_used, mem_total)
        click.echo(
            f"  Mem:  {mem_bar}  "
            f"{mem_used / 1024:.1f} / {mem_total / 1024:.0f} GB"
        )

    click.echo(f"  Users: {info['users']}")


def _build_process_table(
    procs_by_node: dict[str, list[dict]],
    title: str = "imas-codex Processes",
) -> Table | None:
    """Build Rich Table for processes, optionally grouped by node."""
    active = {n: ps for n, ps in procs_by_node.items() if ps}
    if not active:
        return None

    multi = len(active) > 1
    table = Table(title=title, show_edge=False, pad_edge=False)
    if multi:
        table.add_column("Node", style="cyan")
    table.add_column("PID", style="dim")
    table.add_column("CPU%", justify="right")
    table.add_column("Mem%", justify="right")
    table.add_column("RSS", justify="right", style="dim")
    table.add_column("Age", justify="right")
    table.add_column("Command")

    first_node = True
    for node, procs in active.items():
        if multi and not first_node:
            table.add_section()
        first_node = False
        for i, p in enumerate(procs):
            cpu_str = f"[red]{p['cpu']}%[/red]" if float(p["cpu"]) > 50 else f"{p['cpu']}%"
            mem_str = f"[yellow]{p['mem']}%[/yellow]" if float(p["mem"]) > 10 else f"{p['mem']}%"
            age_s = p.get("age_seconds", 0)
            age_str = _format_age(age_s)
            if age_s > 86400:
                age_str = f"[yellow]{age_str}[/yellow]"
            row: list[str] = []
            if multi:
                row.append(node if i == 0 else "")
            row.extend([p["pid"], cpu_str, mem_str, f"{p['rss_mb']:.0f}M", age_str, p["command"]])
            table.add_row(*row)

    return table


def _show_processes(
    procs_by_node: dict[str, list[dict]],
    title: str = "imas-codex Processes",
) -> None:
    """Print imas-codex processes in a Rich Table."""
    table = _build_process_table(procs_by_node, title)
    if table is None:
        console.print("[dim]No imas-codex processes running[/dim]")
    else:
        console.print(table)
    click.echo()


# ── Remote node discovery and querying ───────────────────────────────────


def _discover_login_nodes(ssh_host: str, timeout: int = 10) -> list[str]:
    """Discover available login nodes by reading /etc/hosts on the facility.

    Returns list of short hostnames (e.g. ["98dci4-srv-1001", ...]).
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
            if fqdn and "srv" in fqdn and "gpu" not in fqdn:
                short = fqdn.split(".")[0]
                if short not in nodes:
                    nodes.append(short)
        return sorted(nodes)
    except Exception:
        return []


def _query_node(
    node: str,
    gateway: str,
    user: str,
    timeout: int = 10,
) -> tuple[str, dict | None]:
    """SSH to a specific node via gateway and get load + codex processes.

    Returns (node_short_name, info_dict_or_None).
    """
    fqdn = f"{node}.iter.org"
    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=accept-new",
        "-J", gateway,
    ]
    if user:
        ssh_cmd.append(f"{user}@{fqdn}")
    else:
        ssh_cmd.append(fqdn)

    # Single command: load info + codex processes
    script = (
        "hostname -f; "
        "cat /proc/loadavg; "
        "nproc; "
        "awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} "
        "END{printf \"%d %d\\n\", t, a}' /proc/meminfo; "
        "who | wc -l; "
        "echo '---PROCS---'; "
        + _PS_CMD_REMOTE
    )
    ssh_cmd.append(script)

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0:
            return node, None

        output = result.stdout.strip()
        # Split into load section and process section
        if "---PROCS---" in output:
            load_section, ps_section = output.split("---PROCS---", 1)
        else:
            load_section = output
            ps_section = ""

        lines = load_section.strip().splitlines()
        if len(lines) < 4:
            return node, None

        loadavg_parts = lines[1].split()
        cpu_count = int(lines[2].strip())
        mem_parts = lines[3].split()
        users = int(lines[4].strip()) if len(lines) > 4 else 0
        total_kb = int(mem_parts[0]) if len(mem_parts) >= 2 else 0
        avail_kb = int(mem_parts[1]) if len(mem_parts) >= 2 else 0

        procs = _parse_ps_output(ps_section) if ps_section.strip() else []

        return node, {
            "hostname": lines[0].strip(),
            "load_1m": float(loadavg_parts[0]),
            "load_5m": float(loadavg_parts[1]),
            "load_15m": float(loadavg_parts[2]),
            "cpu_count": cpu_count,
            "mem_total_mb": total_kb / 1024,
            "mem_used_mb": (total_kb - avail_kb) / 1024,
            "users": users,
            "codex_procs": procs,
        }
    except Exception:
        return node, None


# ── Survey data helpers ──────────────────────────────────────────────────


def _gather_survey_data(
    nodes: list[str],
    gateway: str,
    user: str,
    timeout: int,
) -> dict[str, dict | None]:
    """Query all nodes in parallel and return results."""
    results: dict[str, dict | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futures = {
            pool.submit(_query_node, n, gateway, user, timeout): n
            for n in nodes
        }
        for future in concurrent.futures.as_completed(futures):
            node, info = future.result()
            results[node] = info
    return results


def _build_survey_table(
    results: dict[str, dict | None],
    facility: str,
    current_target: str | None,
) -> tuple[Table, str | None, float]:
    """Build the summary Rich Table from survey results.

    Returns (table, best_node, best_load_pct).
    """
    table = Table(title=f"Login Nodes — {facility.upper()}")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Node", style="cyan")
    table.add_column("CPU Load", min_width=40)
    table.add_column("Memory", min_width=35)
    table.add_column("Users", justify="right")
    table.add_column("Codex Procs", justify="right")

    best_node = None
    best_load = float("inf")

    for idx, node in enumerate(sorted(results), 1):
        info = results[node]
        if info is None:
            table.add_row(
                str(idx), node, "[red]unreachable[/red]", "", "", ""
            )
            continue

        row = _format_load_row(info)
        cpu_pct = (
            (info["load_1m"] / info["cpu_count"]) * 100
            if info["cpu_count"] > 0
            else 100
        )
        n_procs = len(info.get("codex_procs", []))

        if cpu_pct < best_load:
            best_load = cpu_pct
            best_node = node

        # Mark current SSH target
        hostname = info.get("hostname", "")
        is_current = current_target and (
            hostname.startswith(current_target.split(".")[0])
            or current_target.startswith(node)
        )
        marker = " ◀" if is_current else ""
        node_display = f"{row[0]}{marker}"

        proc_str = f"[green]{n_procs}[/green]" if n_procs > 0 else "0"

        table.add_row(
            str(idx), node_display, row[1], row[2], row[3], proc_str
        )

    return table, best_node, best_load


def _collect_procs(results: dict[str, dict | None]) -> dict[str, list[dict]]:
    """Collect codex processes from survey results, keyed by node."""
    procs: dict[str, list[dict]] = {}
    for node in sorted(results):
        info = results[node]
        if info is not None:
            ps = info.get("codex_procs", [])
            if ps:
                procs[node] = ps
    return procs


def _build_survey_display(
    results: dict[str, dict | None],
    facility: str,
    current_target: str | None,
) -> tuple[list, str | None, float]:
    """Build complete survey display as a list of Rich renderables.

    Returns (renderables, best_node, best_load_pct).
    """
    table, best_node, best_load = _build_survey_table(
        results, facility, current_target
    )
    parts: list = [table]

    procs_by_node = _collect_procs(results)
    proc_table = _build_process_table(procs_by_node)
    if proc_table:
        parts.append(proc_table)

    if best_node:
        parts.append(
            Text.from_markup(
                f"\n  Least loaded: [green bold]{best_node}[/green bold] "
                f"({best_load:.0f}% CPU)"
            )
        )

    if current_target:
        parts.append(
            Text.from_markup(
                f"  SSH target:   {facility} → "
                f"[cyan]{current_target}[/cyan]"
            )
        )

    return parts, best_node, best_load


# ── SSH config management ────────────────────────────────────────────────


def _get_ssh_hostname(alias: str) -> str | None:
    """Read the current HostName for an SSH alias from ~/.ssh/config."""
    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return None

    content = ssh_config.read_text()
    in_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("Host ") and not stripped.startswith("Host *"):
            host_names = stripped.split()[1:]
            in_block = alias in host_names
            continue
        if in_block and stripped.lower().startswith("hostname "):
            return stripped.split(None, 1)[1].strip()
        if stripped.startswith("Host "):
            in_block = False

    return None


def _set_ssh_hostname(alias: str, new_hostname: str) -> bool:
    """Update the HostName for an SSH alias in ~/.ssh/config.

    Finds the `Host <alias>` block that contains a HostName directive
    and replaces it. Only modifies the FIRST matching HostName in the
    correct Host block.

    Returns True on success.
    """
    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return False

    content = ssh_config.read_text()
    lines = content.splitlines(keepends=True)
    new_lines = []
    in_target_block = False
    replaced = False

    for line in lines:
        stripped = line.strip()

        # Detect Host block boundaries
        if stripped.startswith("Host ") and not stripped.startswith("Host *"):
            host_names = stripped.split()[1:]
            # We need a block where the alias appears ALONE or first
            # to avoid matching shared blocks like "Host iter sdcc"
            # The HostName should be in a standalone "Host iter" block
            in_target_block = (
                alias in host_names and not replaced
            )

        if (
            in_target_block
            and not replaced
            and stripped.lower().startswith("hostname ")
        ):
            # Preserve indentation
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}HostName {new_hostname}\n")
            replaced = True
            continue

        new_lines.append(line)

    if not replaced:
        return False

    # Atomic write: write to temp first, then rename
    tmp_path = ssh_config.with_suffix(".tmp")
    tmp_path.write_text("".join(new_lines))
    tmp_path.rename(ssh_config)
    os.chmod(ssh_config, 0o600)
    return True


# ── CLI ──────────────────────────────────────────────────────────────────


class _HostGroup(click.Group):
    """Group that falls through to the survey command for unknown names.

    Known subcommands (status, add) are dispatched normally.
    Anything else is treated as a facility name and routed to the
    hidden ``survey`` subcommand, giving us:

        imas-codex host              → local load
        imas-codex host iter         → survey ITER nodes
        imas-codex host status       → SSH connectivity
        imas-codex host add myhost   → add SSH entry
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # If the first arg looks like a facility (not a known command),
        # inject "survey" so Click routes to the survey subcommand.
        if (
            args
            and args[0] not in self.commands
            and not args[0].startswith("-")
        ):
            args = ["survey"] + args
        return super().parse_args(ctx, args)


@click.group(cls=_HostGroup, invoke_without_command=True)
@click.pass_context
def host(ctx: click.Context) -> None:
    """Node load monitoring and SSH target management.

    \b
      imas-codex host                    Local node load + processes
      imas-codex host iter               Survey ITER login nodes
      imas-codex host iter --set-default 3
      imas-codex host status             SSH connectivity check
      imas-codex host add <name>         Add SSH host entry
    """
    if ctx.invoked_subcommand is not None:
        return

    # Default: show local load + processes
    info = _get_load_info()
    _show_local_load(info)
    click.echo()
    procs = _get_codex_processes_local()
    _show_processes({info["hostname"]: procs})


@host.command("survey", hidden=True)
@click.argument("facility")
@click.option("--timeout", default=10, help="SSH timeout per node in seconds")
@click.option(
    "--watch", "-w", is_flag=True, help="Repeat every 30s (Ctrl+C to stop)"
)
@click.option(
    "--set-default",
    "set_default",
    default=None,
    is_flag=False,
    flag_value="auto",
    help=(
        "Set SSH target: node index, hostname, or omit value "
        "to auto-select least loaded"
    ),
)
def host_survey(
    facility: str,
    timeout: int,
    watch: bool,
    set_default: str | None,
) -> None:
    """Survey login nodes on a facility with load and process info.

    Discovers login nodes dynamically via /etc/hosts on the facility
    gateway.  Shows CPU load, memory, users, and imas-codex processes
    per node.

    Use --set-default to update ~/.ssh/config to target a specific
    login node for all future SSH/cx connections.

    \b
    Examples:
        imas-codex host iter                    # Survey ITER login nodes
        imas-codex host iter --watch            # Continuous monitoring
        imas-codex host iter --set-default 3    # Pick node #3
        imas-codex host iter --set-default      # Auto-pick least loaded
    """
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    ssh_host = config.get("ssh_host")
    if not ssh_host:
        raise click.ClickException(
            f"Facility '{facility}' has no ssh_host configured"
        )

    gateway = config.get("ssh_gateway", "sdcc-login.iter.org")
    user = config.get("ssh_user", "")

    def _discover_and_survey():
        """Discover nodes, query them, display results. Return results dict."""
        click.echo(
            f"Discovering login nodes on "
            f"{click.style(facility, fg='cyan')}…"
        )
        nodes = _discover_login_nodes(ssh_host, timeout=timeout)
        if not nodes:
            raise click.ClickException(
                f"Could not discover login nodes for '{facility}'. "
                f"Ensure SSH access to '{ssh_host}' is working."
            )

        click.echo(f"Found {len(nodes)} login nodes, querying…\n")
        results = _gather_survey_data(nodes, gateway, user, timeout)
        current_target = _get_ssh_hostname(facility)

        parts, _, _ = _build_survey_display(
            results, facility, current_target
        )
        for part in parts:
            if isinstance(part, (Table, Text)):
                console.print(part)
        click.echo()

        return results, nodes

    if set_default is not None:
        results, nodes = _discover_and_survey()
        sorted_nodes = sorted(results)

        # Resolve target
        target_fqdn = None

        if set_default == "auto":
            best = None
            best_cpu = float("inf")
            for node in sorted_nodes:
                info = results[node]
                if info is None:
                    continue
                cpu_pct = (
                    (info["load_1m"] / info["cpu_count"]) * 100
                    if info["cpu_count"] > 0
                    else 100
                )
                if cpu_pct < best_cpu:
                    best_cpu = cpu_pct
                    best = node
            if best:
                target_fqdn = f"{best}.iter.org"
            else:
                raise click.ClickException("No reachable nodes found")
        else:
            try:
                idx = int(set_default)
                if 1 <= idx <= len(sorted_nodes):
                    target_fqdn = f"{sorted_nodes[idx - 1]}.iter.org"
                else:
                    raise click.ClickException(
                        f"Index {idx} out of range (1-{len(sorted_nodes)})"
                    )
            except ValueError:
                candidate = set_default.strip()
                if "." not in candidate:
                    matches = [
                        n for n in sorted_nodes if candidate in n
                    ]
                    if len(matches) == 1:
                        target_fqdn = f"{matches[0]}.iter.org"
                    elif len(matches) > 1:
                        raise click.ClickException(
                            f"Ambiguous hostname '{candidate}', "
                            f"matches: {', '.join(matches)}"
                        )
                    else:
                        raise click.ClickException(
                            f"No node matching '{candidate}'"
                        )
                else:
                    target_fqdn = candidate

        if target_fqdn:
            target_short = target_fqdn.split(".")[0]
            if target_short in results and results[target_short] is None:
                raise click.ClickException(
                    f"Node '{target_short}' is unreachable — refusing "
                    f"to switch. Pick a reachable node."
                )

            current = _get_ssh_hostname(facility)
            if current == target_fqdn:
                click.echo(
                    f"  Already set: {facility} → {target_fqdn}"
                )
            elif _set_ssh_hostname(facility, target_fqdn):
                click.echo(
                    click.style("  ✓ ", fg="green")
                    + f"Updated: {facility} → "
                    + click.style(target_fqdn, fg="cyan", bold=True)
                )
                click.echo(
                    "  Note: existing SSH sessions continue on the "
                    "old node until closed."
                )
            else:
                raise click.ClickException(
                    f"Could not update SSH config for '{facility}'. "
                    f"Ensure a 'Host {facility}' block with a HostName "
                    f"directive exists in ~/.ssh/config"
                )
        return

    if watch:
        import time

        from rich.console import Group
        from rich.live import Live

        click.echo(
            f"Discovering login nodes on "
            f"{click.style(facility, fg='cyan')}…"
        )
        nodes = _discover_login_nodes(ssh_host, timeout=timeout)
        if not nodes:
            raise click.ClickException(
                f"Could not discover login nodes for '{facility}'. "
                f"Ensure SSH access to '{ssh_host}' is working."
            )
        click.echo(
            f"Found {len(nodes)} login nodes. "
            f"Live monitoring… (Ctrl+C to stop)\n"
        )

        try:
            with Live(console=console, refresh_per_second=0.5) as live:
                while True:
                    results = _gather_survey_data(
                        nodes, gateway, user, timeout
                    )
                    current_target = _get_ssh_hostname(facility)
                    parts, _, _ = _build_survey_display(
                        results, facility, current_target
                    )
                    live.update(Group(*parts))
                    time.sleep(30)
        except KeyboardInterrupt:
            click.echo("\nStopped.")
    else:
        _discover_and_survey()


@host.command("status")
@click.argument("facility", required=False)
@click.option("--timeout", default=5, help="Connection timeout in seconds")
def host_status(facility: str | None, timeout: int) -> None:
    """Check SSH connectivity to configured facilities.

    If FACILITY is provided, checks only that facility. Otherwise checks all.
    """
    import time

    from imas_codex.discovery.base.facility import get_facility, list_facilities

    facility_names = list_facilities()

    if facility:
        if facility not in facility_names:
            raise click.ClickException(f"Unknown facility: {facility}")
        facility_names = [facility]

    table = Table(title="SSH Connectivity")
    table.add_column("Facility", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Latency", style="dim")

    for name in facility_names:
        config = get_facility(name)
        ssh_host = config.get("ssh_host")
        if not ssh_host:
            table.add_row(name, "[dim]No SSH host[/dim]", "-")
            continue

        start = time.time()
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "BatchMode=yes",
                    "-o", f"ConnectTimeout={timeout}",
                    "-o", "StrictHostKeyChecking=accept-new",
                    ssh_host,
                    "echo ok",
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 2,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                table.add_row(
                    name,
                    "[green]✓ Connected[/green]",
                    f"{elapsed * 1000:.0f}ms",
                )
            else:
                error = (
                    result.stderr.strip().split("\n")[0]
                    if result.stderr
                    else "Failed"
                )
                table.add_row(name, f"[red]✗ {error}[/red]", "-")

        except subprocess.TimeoutExpired:
            table.add_row(name, "[red]✗ Timeout[/red]", "-")
        except Exception as e:
            table.add_row(name, f"[red]✗ {e}[/red]", "-")

    console.print(table)


@host.command("add")
@click.argument("name")
@click.option("--hostname", required=True, help="Remote hostname or IP")
@click.option("--user", required=True, help="SSH username")
@click.option("--port", default=22, help="SSH port")
@click.option("--identity", help="Path to SSH private key")
@click.option("--proxy", help="ProxyJump host for bastion")
def host_add(
    name: str,
    hostname: str,
    user: str,
    port: int,
    identity: str | None,
    proxy: str | None,
) -> None:
    """Add a new SSH host entry to ~/.ssh/config.

    Examples:
        imas-codex host add tcv --hostname tcv.epfl.ch --user smith
        imas-codex host add iter --hostname iter.org --user john --proxy gateway
    """
    ssh_config = Path.home() / ".ssh" / "config"
    ssh_dir = ssh_config.parent

    if not ssh_dir.exists():
        ssh_dir.mkdir(mode=0o700)

    lines = [f"\nHost {name}"]
    lines.append(f"    HostName {hostname}")
    lines.append(f"    User {user}")
    if port != 22:
        lines.append(f"    Port {port}")
    if identity:
        lines.append(f"    IdentityFile {identity}")
    if proxy:
        lines.append(f"    ProxyJump {proxy}")
    lines.append("")

    entry = "\n".join(lines)

    if ssh_config.exists():
        content = ssh_config.read_text()
        if f"Host {name}" in content or f"Host {name}\n" in content:
            raise click.ClickException(
                f"Host '{name}' already exists in ~/.ssh/config"
            )

    with open(ssh_config, "a") as f:
        f.write(entry)

    os.chmod(ssh_config, 0o600)

    console.print(f"[green]✓[/green] Added SSH host '{name}'")
    console.print(f"[dim]Test with: ssh {name}[/dim]")


@host.command("kill")
@click.argument("facility")
@click.option(
    "--include", "-i", "include_pattern",
    help="Only kill processes matching this regex pattern",
)
@click.option(
    "--exclude", "-e", "exclude_pattern",
    help="Exclude processes matching this regex pattern",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--node", "-n", "target_node",
    help="Target specific node (index or name)",
)
@click.option(
    "--signal", "-s", "sig", default="TERM",
    help="Signal to send (default: TERM)",
)
@click.option(
    "--older-than", "-o", "older_than",
    help="Only kill processes older than this duration (e.g. 1h, 2d, 30m)",
)
@click.option("--timeout", default=10, help="SSH timeout per node")
def host_kill(
    facility: str,
    include_pattern: str | None,
    exclude_pattern: str | None,
    yes: bool,
    target_node: str | None,
    sig: str,
    older_than: str | None,
    timeout: int,
) -> None:
    """Kill imas-codex processes on remote facility nodes.

    Shows matching processes first, then kills with confirmation.
    Uses the same process detection as 'host <facility>'.

    \b
    Examples:
        imas-codex host kill iter                        # Preview all
        imas-codex host kill iter -y                     # Kill all
        imas-codex host kill iter -i litellm -y          # Kill litellm
        imas-codex host kill iter -e "neo4j|serve" -y    # Kill all except
        imas-codex host kill iter -n 2 -y                # Kill on node #2
        imas-codex host kill iter -i discover -s KILL    # SIGKILL discover
        imas-codex host kill iter -o 2d -y               # Kill procs >2 days old
        imas-codex host kill iter -o 1h -e serve -y      # Kill >1h except serve
    """
    valid_signals = {
        "TERM", "KILL", "HUP", "INT", "QUIT",
        "USR1", "USR2", "STOP", "CONT",
    }
    sig = sig.upper()
    if sig not in valid_signals:
        raise click.ClickException(
            f"Invalid signal: {sig}. "
            f"Valid: {', '.join(sorted(valid_signals))}"
        )

    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    ssh_host = config.get("ssh_host")
    if not ssh_host:
        raise click.ClickException(
            f"Facility '{facility}' has no ssh_host configured"
        )

    gateway = config.get("ssh_gateway", "sdcc-login.iter.org")
    user = config.get("ssh_user", "")

    click.echo(
        f"Discovering login nodes on "
        f"{click.style(facility, fg='cyan')}…"
    )
    nodes = _discover_login_nodes(ssh_host, timeout=timeout)
    if not nodes:
        raise click.ClickException(
            f"Could not discover login nodes for '{facility}'."
        )

    # Filter to specific node if requested
    if target_node:
        try:
            idx = int(target_node)
            if 1 <= idx <= len(nodes):
                nodes = [sorted(nodes)[idx - 1]]
            else:
                raise click.ClickException(
                    f"Node index {idx} out of range (1-{len(nodes)})"
                )
        except ValueError:
            matches = [n for n in nodes if target_node in n]
            if len(matches) == 1:
                nodes = matches
            elif not matches:
                raise click.ClickException(
                    f"No node matching '{target_node}'"
                )
            else:
                raise click.ClickException(
                    f"Ambiguous: {', '.join(matches)}"
                )

    click.echo(f"Querying {len(nodes)} node(s)…\n")
    results = _gather_survey_data(nodes, gateway, user, timeout)

    # Compile patterns
    include_re = (
        re.compile(include_pattern, re.IGNORECASE)
        if include_pattern else None
    )
    exclude_re = (
        re.compile(exclude_pattern, re.IGNORECASE)
        if exclude_pattern else None
    )

    # Filter processes
    kill_targets: dict[str, list[dict]] = {}
    for node in sorted(results):
        info = results[node]
        if info is None:
            continue
        procs = info.get("codex_procs", [])
        matched = []
        for p in procs:
            cmd = p["command"]
            if include_re and not include_re.search(cmd):
                continue
            if exclude_re and exclude_re.search(cmd):
                continue
            if min_age_seconds and p.get("age_seconds", 0) < min_age_seconds:
                continue
            matched.append(p)
        if matched:
            kill_targets[node] = matched

    if not kill_targets:
        click.echo("No matching processes found.")
        return

    total = sum(len(ps) for ps in kill_targets.values())
    _show_processes(
        kill_targets, title=f"Processes to kill ({total} total, SIG{sig})"
    )

    if not yes:
        click.confirm(
            f"Kill {total} process(es) with SIG{sig}?", abort=True
        )

    # Send kill signals
    for node, procs in kill_targets.items():
        pids = " ".join(p["pid"] for p in procs)
        fqdn = f"{node}.iter.org"
        kill_cmd = f"kill -{sig} {pids} 2>/dev/null; echo done"

        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={timeout}",
            "-o", "StrictHostKeyChecking=accept-new",
            "-J", gateway,
        ]
        if user:
            ssh_cmd.append(f"{user}@{fqdn}")
        else:
            ssh_cmd.append(fqdn)
        ssh_cmd.append(kill_cmd)

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True,
                timeout=timeout + 5,
            )
            if result.returncode == 0:
                click.echo(
                    f"  {click.style('✓', fg='green')} "
                    f"{node}: killed {len(procs)} process(es)"
                )
            else:
                click.echo(
                    f"  {click.style('✗', fg='red')} "
                    f"{node}: {result.stderr.strip()[:60]}"
                )
        except Exception as e:
            click.echo(
                f"  {click.style('✗', fg='red')} {node}: {e}"
            )
