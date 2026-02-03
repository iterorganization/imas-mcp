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
