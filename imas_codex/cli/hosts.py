"""Hosts commands - SSH host configuration management."""

import os
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def hosts() -> None:
    """Manage SSH host configurations for remote facilities.

    \b
      imas-codex hosts list       List configured SSH hosts
      imas-codex hosts status     Check SSH connectivity
      imas-codex hosts load       Survey login node load across a facility
      imas-codex hosts add        Add a new SSH host entry
    """
    pass


@hosts.command("list")
def hosts_list() -> None:
    """List configured SSH hosts from ~/.ssh/config."""
    ssh_config = Path.home() / ".ssh" / "config"

    if not ssh_config.exists():
        console.print("[yellow]No SSH config found at ~/.ssh/config[/yellow]")
        return

    hosts_found: list[dict] = []
    current_host: dict | None = None

    with open(ssh_config) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Host ") and not line.startswith("Host *"):
                if current_host:
                    hosts_found.append(current_host)
                host_name = line.split(None, 1)[1].split()[0]  # First host pattern
                current_host = {"name": host_name, "hostname": "", "user": ""}
            elif current_host and line.startswith("HostName "):
                current_host["hostname"] = line.split(None, 1)[1]
            elif current_host and line.startswith("User "):
                current_host["user"] = line.split(None, 1)[1]

        if current_host:
            hosts_found.append(current_host)

    table = Table(title="SSH Hosts")
    table.add_column("Alias", style="cyan")
    table.add_column("Hostname", style="white")
    table.add_column("User", style="dim")

    for host in hosts_found:
        table.add_row(host["name"], host["hostname"], host["user"])

    console.print(table)


@hosts.command("status")
@click.argument("host", required=False)
@click.option("--timeout", default=5, help="Connection timeout in seconds")
def hosts_status(host: str | None, timeout: int) -> None:
    """Check SSH connectivity to configured hosts.

    If HOST is provided, checks only that host. Otherwise checks all.
    """
    import time

    from imas_codex.discovery.base.facility import get_facility, list_facilities

    facility_names = list_facilities()

    if host:
        # Check specific host
        if host not in facility_names:
            raise click.ClickException(f"Unknown facility: {host}")
        facility_names = [host]

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

        # Try SSH connection
        start = time.time()
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={timeout}",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
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
                    name, "[green]✓ Connected[/green]", f"{elapsed * 1000:.0f}ms"
                )
            else:
                error = (
                    result.stderr.strip().split("\n")[0] if result.stderr else "Failed"
                )
                table.add_row(name, f"[red]✗ {error}[/red]", "-")

        except subprocess.TimeoutExpired:
            table.add_row(name, "[red]✗ Timeout[/red]", "-")
        except Exception as e:
            table.add_row(name, f"[red]✗ {e}[/red]", "-")

    console.print(table)


@hosts.command("add")
@click.argument("name")
@click.option("--hostname", required=True, help="Remote hostname or IP")
@click.option("--user", required=True, help="SSH username")
@click.option("--port", default=22, help="SSH port")
@click.option("--identity", help="Path to SSH private key")
@click.option("--proxy", help="ProxyJump host for bastion")
def hosts_add(
    name: str,
    hostname: str,
    user: str,
    port: int,
    identity: str | None,
    proxy: str | None,
) -> None:
    """Add a new SSH host entry to ~/.ssh/config.

    Examples:
        imas-codex hosts add tcv --hostname tcv.epfl.ch --user smith
        imas-codex hosts add iter --hostname iter.org --user john --proxy gateway
    """
    ssh_config = Path.home() / ".ssh" / "config"
    ssh_dir = ssh_config.parent

    # Ensure .ssh directory exists with correct permissions
    if not ssh_dir.exists():
        ssh_dir.mkdir(mode=0o700)

    # Build host entry
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

    # Check if host already exists
    if ssh_config.exists():
        content = ssh_config.read_text()
        if f"Host {name}" in content or f"Host {name}\n" in content:
            raise click.ClickException(f"Host '{name}' already exists in ~/.ssh/config")

    # Append to config
    with open(ssh_config, "a") as f:
        f.write(entry)

    # Fix permissions
    os.chmod(ssh_config, 0o600)

    console.print(f"[green]✓[/green] Added SSH host '{name}'")
    console.print("[dim]Test with: ssh {name}[/dim]")


@hosts.command("load")
@click.argument("facility", default="iter")
@click.option("--timeout", default=10, help="SSH timeout per node in seconds")
@click.option(
    "--watch", "-w", is_flag=True, help="Repeat every 30 seconds (Ctrl+C to stop)"
)
def hosts_load(facility: str, timeout: int, watch: bool) -> None:
    """Survey login node load across a facility.

    Discovers all login nodes via /etc/hosts on the facility and queries
    each for CPU load, memory usage, and logged-in users. Highlights the
    least loaded node for optimal session targeting.

    Nodes are discovered dynamically by SSHing to the facility's gateway
    and reading /etc/hosts — no need to configure individual node addresses.

    \b
    Examples:
        imas-codex hosts load              # Survey ITER login nodes
        imas-codex hosts load --watch      # Continuous monitoring
        imas-codex hosts load iter --timeout 15
    """
    import concurrent.futures
    import time

    from imas_codex.cli.status import (
        _colored_bar,
        _format_load_row,
        _get_remote_load_info,
        discover_login_nodes,
    )
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    ssh_host = config.get("ssh_host")
    if not ssh_host:
        raise click.ClickException(f"Facility '{facility}' has no ssh_host configured")

    def _survey_once() -> None:
        # Discover login nodes via /etc/hosts on the facility
        click.echo(f"Discovering login nodes on {click.style(facility, fg='cyan')}…")
        nodes = discover_login_nodes(ssh_host, timeout=timeout)

        if not nodes:
            raise click.ClickException(
                f"Could not discover login nodes for '{facility}'. "
                f"Ensure SSH access to '{ssh_host}' is working."
            )

        click.echo(f"Found {len(nodes)} login nodes, querying load…\n")

        # Query all nodes in parallel via SSH through the gateway
        results: dict[str, dict | None] = {}

        # SSH directly to each node's FQDN through the gateway
        gateway = config.get("ssh_gateway", "sdcc-login.iter.org")
        user = config.get("ssh_user", "")

        def _query_node(node: str) -> tuple[str, dict | None]:
            # Build SSH command that goes through the gateway to the node
            fqdn = f"{node}.iter.org"
            ssh_cmd = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                "-o", "StrictHostKeyChecking=accept-new",
                "-J", gateway,
            ]
            if user:
                ssh_cmd.extend([f"{user}@{fqdn}"])
            else:
                ssh_cmd.append(fqdn)

            script = (
                "hostname -f; "
                "cat /proc/loadavg; "
                "nproc; "
                "awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} "
                "END{printf \"%d %d\\n\", t, a}' /proc/meminfo; "
                "who | wc -l"
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

                lines = result.stdout.strip().splitlines()
                if len(lines) < 4:
                    return node, None

                loadavg_parts = lines[1].split()
                cpu_count = int(lines[2].strip())
                mem_parts = lines[3].split()
                users = int(lines[4].strip()) if len(lines) > 4 else 0
                total_kb = int(mem_parts[0]) if len(mem_parts) >= 2 else 0
                avail_kb = int(mem_parts[1]) if len(mem_parts) >= 2 else 0

                return node, {
                    "hostname": lines[0].strip(),
                    "load_1m": float(loadavg_parts[0]),
                    "load_5m": float(loadavg_parts[1]),
                    "load_15m": float(loadavg_parts[2]),
                    "cpu_count": cpu_count,
                    "mem_total_mb": total_kb / 1024,
                    "mem_used_mb": (total_kb - avail_kb) / 1024,
                    "users": users,
                }
            except Exception:
                return node, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            futures = {pool.submit(_query_node, n): n for n in nodes}
            for future in concurrent.futures.as_completed(futures):
                node, info = future.result()
                results[node] = info

        # Build results table
        table = Table(
            title=f"Login Node Load — {facility.upper()}",
        )
        table.add_column("Node", style="cyan")
        table.add_column("CPU Load", min_width=40)
        table.add_column("Memory", min_width=35)
        table.add_column("Users", justify="right")
        table.add_column("", style="dim")  # Recommendation column

        best_node = None
        best_load = float("inf")

        for node in sorted(results):
            info = results[node]
            if info is None:
                table.add_row(
                    node, "[red]unreachable[/red]", "", "", ""
                )
                continue

            row = _format_load_row(info)
            cpu_pct = (info["load_1m"] / info["cpu_count"]) * 100 if info["cpu_count"] > 0 else 100

            if cpu_pct < best_load:
                best_load = cpu_pct
                best_node = node

            table.add_row(row[0], row[1], row[2], row[3], "")

        console.print(table)

        if best_node:
            console.print(
                f"\n  Least loaded: [bold green]{best_node}[/bold green] "
                f"({best_load:.0f}% CPU)\n"
            )

    if watch:
        try:
            while True:
                _survey_once()
                click.echo("Refreshing in 30s… (Ctrl+C to stop)\n")
                time.sleep(30)
                # Clear screen for clean refresh
                click.clear()
        except KeyboardInterrupt:
            click.echo("\nStopped.")
    else:
        _survey_once()
