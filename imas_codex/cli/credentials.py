"""Credential management CLI commands.

Provides commands for storing, retrieving, and managing service credentials
used by various imas-codex subsystems (wiki authentication, API access, etc.).

Credentials are stored securely in the system keyring with fallbacks
to environment variables for headless/CI environments.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _resolve_service(service: str) -> str:
    """Resolve a facility name or service name to a credential service name.

    If the argument matches a facility ID, looks up its credential service(s).
    If only one service is found, returns it. Otherwise returns the input unchanged.
    """
    try:
        from imas_codex.discovery.base.facility import get_facility, list_facilities

        if service in list_facilities():
            config = get_facility(service)
            credential_services = []
            for site in config.get("wiki_sites", []):
                svc = site.get("credential_service")
                if svc and svc not in credential_services:
                    credential_services.append(svc)
            if len(credential_services) == 1:
                resolved = credential_services[0]
                if resolved != service:
                    console.print(
                        f"[dim]Resolved facility '{service}' → service '{resolved}'[/dim]"
                    )
                return resolved
            if len(credential_services) > 1:
                console.print(
                    f"[yellow]Facility '{service}' has multiple credential services: "
                    f"{', '.join(credential_services)}[/yellow]\n"
                    "Please specify one directly."
                )
                raise SystemExit(1)
            # No credential services for this facility
            console.print(
                f"[yellow]Facility '{service}' has no credential services "
                "(all sites use auth_type: none).[/yellow]"
            )
            raise SystemExit(1)
    except ImportError:
        pass
    return service


@click.group()
def credentials():
    """Manage service credentials.

    \b
    Commands:
      set     Store credentials for a service
      get     Check if credentials are stored
      delete  Remove stored credentials
      list    List known credential services
      status  Show keyring status and troubleshooting info

    Credentials are stored securely in your system keyring
    (GNOME Keyring, macOS Keychain, or Windows Credential Locker).

    \b
    On headless systems, use environment variables instead:
      export SERVICE_NAME_USERNAME=your_username
      export SERVICE_NAME_PASSWORD=your_password
    """
    pass


@credentials.command("set")
@click.argument("service")
@click.option("--username", "-u", help="Username (prompts if not provided)")
@click.option("--password", "-p", help="Password (prompts if not provided)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing credentials")
def credentials_set(
    service: str,
    username: str | None,
    password: str | None,
    force: bool,
) -> None:
    """Store credentials for a service.

    SERVICE is a facility name (e.g., "tcv", "iter", "jet").

    \b
    Examples:
      imas-codex credentials set tcv
      imas-codex credentials set iter -u myuser
    """
    import getpass

    from imas_codex.discovery.wiki.auth import CredentialManager

    service = _resolve_service(service)
    creds = CredentialManager()

    # Check for existing credentials
    if not force and creds.has_credentials(service):
        if not click.confirm(
            f"Credentials already exist for {service}. Overwrite?",
            default=False,
        ):
            console.print("[yellow]Aborted.[/yellow]")
            raise SystemExit(0)

    # Prompt for missing values
    if not username:
        username = click.prompt("Username")

    if not password:
        password = getpass.getpass("Password: ")

    if not username or not password:
        console.print("[red]Username and password are required.[/red]")
        raise SystemExit(1)

    # Store credentials
    if creds.set_credentials(service, username, password):
        console.print(f"[green]✓ Credentials stored for {service}[/green]")
    else:
        console.print(f"[red]✗ Failed to store credentials for {service}[/red]")
        console.print(
            "\n[yellow]Hint: Check that your system keyring is available.[/yellow]\n"
            "On headless systems, use environment variables instead:\n"
            f"  export {service.upper().replace('-', '_')}_USERNAME=your_username\n"
            f"  export {service.upper().replace('-', '_')}_PASSWORD=your_password"
        )
        raise SystemExit(1)


@credentials.command("get")
@click.argument("service")
def credentials_get(
    service: str,
) -> None:
    """Retrieve credentials for a service.

    Shows credential status and storage source. Passwords are always
    masked — manage passwords via your system keyring or environment
    variables directly.

    SERVICE is a facility name (e.g., "tcv", "iter", "jet").

    \b
    Examples:
      imas-codex credentials get tcv
      imas-codex credentials get iter
    """
    import os

    from imas_codex.discovery.wiki.auth import CredentialManager

    service = _resolve_service(service)
    creds = CredentialManager()

    result = creds.get_credentials(service, prompt_if_missing=False)
    if result is None:
        console.print(f"[yellow]No credentials found for {service}[/yellow]")
        console.print(f"\nTo set credentials: imas-codex credentials set {service}")
        raise SystemExit(1)

    username, password = result

    # Determine storage source
    username_var = creds._env_var_name(service, "username")
    password_var = creds._env_var_name(service, "password")
    if service in CredentialManager._memory_cache:
        source = "memory cache (this process only)"
    elif os.environ.get(username_var) and os.environ.get(password_var):
        source = f"environment ({username_var}, {password_var})"
    else:
        source = "system keyring"

    console.print(f"[green]✓ Credentials stored for {service}[/green]")
    console.print(f"  Username: {username}")
    console.print("  Password: ********")
    console.print(f"  Source:   {source}")


@credentials.command("delete")
@click.argument("service")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def credentials_delete(
    service: str,
    yes: bool,
) -> None:
    """Delete stored credentials for a service.

    Also clears any cached session for the service.

    SERVICE is a facility name (e.g., "tcv", "iter", "jet").

    \b
    Examples:
      imas-codex credentials delete tcv
      imas-codex credentials delete iter --yes
    """
    from imas_codex.discovery.wiki.auth import CredentialManager

    service = _resolve_service(service)
    creds = CredentialManager()

    if not creds.has_credentials(service):
        console.print(f"[yellow]No credentials found for {service}[/yellow]")
        raise SystemExit(0)

    if not yes:
        if not click.confirm(f"Delete credentials for {service}?", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            raise SystemExit(0)

    # Delete credentials and session
    creds.delete_credentials(service)
    creds.delete_session(service)
    console.print(f"[green]✓ Deleted credentials and session for {service}[/green]")


@credentials.command("list")
@click.argument("facility", required=False)
@click.option(
    "--remote",
    is_flag=True,
    help="Check credential status on the remote facility via SSH.",
)
def credentials_list(facility: str | None, remote: bool) -> None:
    """List known credential services.

    Shows services from facility configurations with access requirements
    and credential status.

    \b
    Examples:
      imas-codex credentials list          # All facilities
      imas-codex credentials list iter     # ITER services only
      imas-codex credentials list iter --remote  # Check ITER's keyring via SSH
    """
    import os

    from imas_codex.discovery.wiki.auth import CredentialManager

    # Remote mode: check credentials on the facility host via SSH
    if remote:
        if not facility:
            console.print(
                "[red]--remote requires a facility argument "
                "(e.g., credentials list iter --remote)[/red]"
            )
            raise SystemExit(1)
        _list_remote_credentials(facility)
        return

    creds = CredentialManager()

    # Collect known services from facility configs
    services: dict[str, dict] = {}

    try:
        from imas_codex.discovery.base.facility import get_facility, list_facilities

        facilities = [facility] if facility else list_facilities()
        for facility_id in facilities:
            try:
                config = get_facility(facility_id)
                wiki_sites = config.get("wiki_sites", [])
                for site in wiki_sites:
                    auth_type = site.get("auth_type", "none")
                    access = site.get("access_method", "direct")
                    ssh_avail = site.get("ssh_available", False)
                    needs_creds = auth_type not in ("none",)
                    service = site.get("credential_service") or (
                        f"{facility_id}-wiki" if needs_creds else None
                    )
                    services[site.get("url", facility_id)] = {
                        "service": service,
                        "facility": facility_id,
                        "site_type": site.get("site_type", "mediawiki"),
                        "auth_type": auth_type,
                        "access_method": access,
                        "ssh_available": ssh_avail,
                        "url": site.get("url", ""),
                        "has_creds": (
                            creds.has_credentials(service)
                            if service
                            else None  # Not applicable
                        ),
                    }
            except Exception:
                pass
    except Exception:
        pass

    # Also check for environment-based credentials not in config
    if not facility:
        for key in os.environ:
            if key.endswith("_USERNAME"):
                service_name = key.replace("_USERNAME", "").lower().replace("_", "-")
                if not any(s.get("service") == service_name for s in services.values()):
                    password_key = key.replace("_USERNAME", "_PASSWORD")
                    if password_key in os.environ:
                        services[service_name] = {
                            "service": service_name,
                            "facility": "env",
                            "site_type": "environment",
                            "auth_type": "env",
                            "access_method": "direct",
                            "url": "",
                            "has_creds": True,
                        }

    if not services:
        if facility:
            console.print(
                f"[yellow]No credential services configured for {facility}.[/yellow]"
            )
        else:
            console.print("[yellow]No credential services configured.[/yellow]")
        raise SystemExit(0)

    # Display table
    title = f"Credential Services — {facility}" if facility else "Credential Services"
    table = Table(title=title)
    table.add_column("Facility", style="bold")
    table.add_column("Access")
    table.add_column("SSH")
    table.add_column("Auth")
    table.add_column("Service", style="cyan")
    table.add_column("Credentials")
    table.add_column(
        "URL", style="dim", no_wrap=not bool(facility), min_width=20 if facility else 0
    )

    # Group sites with identical patterns for compact display
    # e.g., JET's 16 wikis all share the same access/auth/service pattern
    grouped: list[dict] = []
    seen_groups: dict[str, dict] = {}

    for _key, info in sorted(services.items(), key=lambda x: x[1]["facility"]):
        group_key = (
            f"{info['facility']}|{info['access_method']}|"
            f"{info['auth_type']}|{info.get('service') or ''}"
        )

        if info.get("service"):
            # Sites with credential services always get individual rows
            if group_key in seen_groups:
                seen_groups[group_key]["_count"] += 1
                seen_groups[group_key]["_urls"].append(info["url"])
            else:
                entry = {**info, "_count": 1, "_urls": [info["url"]]}
                seen_groups[group_key] = entry
                grouped.append(entry)
        elif group_key in seen_groups:
            seen_groups[group_key]["_count"] += 1
            seen_groups[group_key]["_urls"].append(info["url"])
        else:
            entry = {**info, "_count": 1, "_urls": [info["url"]]}
            seen_groups[group_key] = entry
            grouped.append(entry)

    for info in grouped:
        # Credential status
        if info["has_creds"] is None:
            cred_display = "[dim]n/a[/dim]"
        elif info["has_creds"]:
            cred_display = "[green]✓ stored[/green]"
        else:
            cred_display = "[yellow]✗ missing[/yellow]"

        # Access method display
        access = info["access_method"]
        access_display = {
            "direct": "direct",
            "vpn": "[magenta]vpn[/magenta]",
        }.get(access, access)

        # SSH availability
        ssh_display = (
            "[green]✓[/green]" if info.get("ssh_available") else "[dim]—[/dim]"
        )

        urls = [u for u in info.get("_urls", []) if u]
        count = info.get("_count", 1)
        if facility and urls:
            # When a specific facility is requested, show all URLs stacked
            url_display = "\n".join(urls)
        elif count > 1:
            url_display = f"{count} sites"
        elif urls:
            url = urls[0]
            url_display = url[:40] + "…" if len(url) > 40 else url
        else:
            url_display = ""

        table.add_row(
            info["facility"],
            access_display,
            ssh_display,
            info["auth_type"],
            info.get("service") or "—",
            cred_display,
            url_display,
        )

    console.print(table)


def _list_remote_credentials(facility: str) -> None:
    """Check credential status on a remote facility via SSH."""
    from imas_codex.discovery.base.facility import get_facility

    try:
        config = get_facility(facility)
    except Exception as e:
        console.print(f"[red]Unknown facility: {facility}[/red]")
        raise SystemExit(1) from e

    ssh_host = config.get("ssh_host", facility)
    venv_python = "~/.local/share/imas-codex/venv/bin/python"

    # Build a small Python script to check keyring status remotely
    check_script = """
import json, sys
try:
    import keyring
    backend = keyring.get_keyring().__class__.__name__
except Exception as e:
    print(json.dumps({"error": f"keyring unavailable: {e}"}))
    sys.exit(0)

services = %s
results = {}
for svc in services:
    try:
        val = keyring.get_password(f"imas-codex/{svc}", "credentials")
        results[svc] = bool(val)
    except Exception:
        results[svc] = False

print(json.dumps({"backend": backend, "results": results}))
"""

    # Collect credential services for this facility
    credential_services = []
    for site in config.get("wiki_sites", []):
        svc = site.get("credential_service")
        if svc:
            credential_services.append(svc)

    if not credential_services:
        console.print(
            f"[yellow]No credential services configured for {facility}.[/yellow]"
        )
        return

    import subprocess

    script = check_script % repr(credential_services)
    cmd = [
        "ssh",
        ssh_host,
        f"{venv_python} -c {repr(script)}",
    ]

    console.print(
        f"Checking keyring on [bold]{ssh_host}[/bold] "
        f"({len(credential_services)} services)…\n"
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No such file" in stderr or "not found" in stderr:
                console.print(
                    f"[yellow]imas-codex venv not installed on {ssh_host}.[/yellow]\n"
                    f"Install with: imas-codex tools install {facility}"
                )
            else:
                console.print(
                    f"[red]SSH command failed:[/red] {stderr or 'unknown error'}"
                )
            return

        import json

        data = json.loads(result.stdout.strip())
        if "error" in data:
            console.print(
                f"[yellow]Remote keyring: {data['error']}[/yellow]\n"
                "Credentials on this host must use environment variables."
            )
            return

        table = Table(title=f"Remote Credentials — {facility} ({ssh_host})")
        table.add_column("Service", style="cyan")
        table.add_column("Keyring Backend")
        table.add_column("Credentials")

        for svc, has_creds in data["results"].items():
            status = (
                "[green]✓ stored[/green]" if has_creds else "[yellow]✗ missing[/yellow]"
            )
            table.add_row(svc, data["backend"], status)

        console.print(table)
        console.print(
            f"\n[dim]To set credentials on {ssh_host}: "
            f'ssh {ssh_host} "~/.local/share/imas-codex/venv/bin/imas-codex '
            f'credentials set <service>"[/dim]'
        )

    except subprocess.TimeoutExpired:
        console.print(f"[red]SSH connection to {ssh_host} timed out.[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@credentials.command("status")
def credentials_status() -> None:
    """Show keyring status and troubleshooting info.

    Helps diagnose credential storage issues, especially on WSL.
    """
    import sys

    from imas_codex.discovery.wiki.auth import CredentialManager, _is_wsl

    console.print("[bold]Keyring Status[/bold]\n")

    # Platform info
    console.print(f"Platform: {sys.platform}")
    console.print(f"WSL detected: {_is_wsl()}")

    # Try to initialize credential manager
    creds = CredentialManager()
    console.print(f"Keyring available: {creds._keyring_available}")

    if creds._keyring_available:
        import keyring

        backend = keyring.get_keyring()
        console.print(f"Backend: {backend.__class__.__name__}")
        console.print("\n[green]✓ Keyring is working![/green]")
        console.print("  Credentials will be stored securely in the system keyring.")
    else:
        console.print("\n[yellow]⚠ Keyring not available[/yellow]")
        console.print("  Credentials will fall back to environment variables.")

        if _is_wsl():
            console.print("\n[bold]WSL Troubleshooting:[/bold]")
            console.print(
                "  If GNOME Keyring prompts for an unlock password you don't know:"
            )
            console.print("\n  [cyan]Option 1: Reset keyring (recommended)[/cyan]")
            console.print("    rm -rf ~/.local/share/keyrings/*")
            console.print("    # Next credential access will create a new keyring")
            console.print("    # Use your WSL login password for auto-unlock")
            console.print("\n  [cyan]Option 2: Use encrypted file backend[/cyan]")
            console.print("    pip install keyrings.alt")
            console.print(
                "    mkdir -p ~/.config && cat > ~/.config/keyringrc.cfg << EOF"
            )
            console.print("    [backend]")
            console.print("    default-keyring=keyrings.alt.file.EncryptedKeyring")
            console.print("    EOF")
            console.print("\n  [cyan]Option 3: Use environment variables[/cyan]")
            console.print("    export TCV_WIKI_USERNAME=your_username")
            console.print("    export TCV_WIKI_PASSWORD=your_password")
