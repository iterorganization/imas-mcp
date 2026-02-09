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

    SERVICE is the credential service name (e.g., "tcv-wiki", "iter-confluence").

    \b
    Examples:
      imas-codex credentials set tcv-wiki
      imas-codex credentials set iter-confluence -u myuser
    """
    import getpass

    from imas_codex.discovery.wiki.auth import CredentialManager

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
@click.option("--show-password", is_flag=True, help="Show password (security risk)")
def credentials_get(
    service: str,
    show_password: bool,
) -> None:
    """Check if credentials exist for a service.

    SERVICE is the credential service name (e.g., "tcv-wiki", "iter-confluence").

    \b
    Examples:
      imas-codex credentials get tcv-wiki
      imas-codex credentials get iter-confluence --show-password
    """
    from imas_codex.discovery.wiki.auth import CredentialManager

    creds = CredentialManager()

    result = creds.get_credentials(service, prompt_if_missing=False)
    if result is None:
        console.print(f"[yellow]No credentials found for {service}[/yellow]")
        console.print(f"\nTo set credentials: imas-codex credentials set {service}")
        raise SystemExit(1)

    username, password = result
    console.print(f"[green]✓ Credentials exist for {service}[/green]")
    console.print(f"  Username: {username}")
    if show_password:
        console.print(f"  Password: {password}")
    else:
        console.print(f"  Password: {'*' * len(password)}")


@credentials.command("delete")
@click.argument("service")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def credentials_delete(
    service: str,
    yes: bool,
) -> None:
    """Delete stored credentials for a service.

    Also clears any cached session for the service.

    SERVICE is the credential service name (e.g., "tcv-wiki", "iter-confluence").

    \b
    Examples:
      imas-codex credentials delete tcv-wiki
      imas-codex credentials delete iter-confluence --yes
    """
    from imas_codex.discovery.wiki.auth import CredentialManager

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
def credentials_list() -> None:
    """List known credential services.

    Shows services from facility configurations and environment variables.
    """
    import os

    from imas_codex.discovery.wiki.auth import CredentialManager

    creds = CredentialManager()

    # Collect known services from facility configs
    services: dict[str, dict] = {}

    try:
        from imas_codex.discovery.base.facility import list_facilities

        for facility_id in list_facilities():
            from imas_codex.discovery.base.facility import get_facility

            try:
                config = get_facility(facility_id)
                wiki_sites = config.get("wiki_sites", [])
                for site in wiki_sites:
                    service = site.get("credential_service") or f"{facility_id}-wiki"
                    services[service] = {
                        "facility": facility_id,
                        "source": site.get("site_type", "mediawiki"),
                        "url": site.get("url", ""),
                        "has_creds": False,
                    }
            except Exception:
                pass
    except Exception:
        pass

    # Check which have stored credentials
    for service in services:
        services[service]["has_creds"] = creds.has_credentials(service)

    # Also check for environment-based credentials
    for key in os.environ:
        if key.endswith("_USERNAME"):
            service_name = key.replace("_USERNAME", "").lower().replace("_", "-")
            if service_name not in services:
                password_key = key.replace("_USERNAME", "_PASSWORD")
                if password_key in os.environ:
                    services[service_name] = {
                        "facility": "env",
                        "source": "environment",
                        "url": "",
                        "has_creds": True,
                    }

    if not services:
        console.print("[yellow]No credential services configured.[/yellow]")
        console.print(
            "\nConfigure wiki sites in facility YAML files under 'wiki_sites:'"
        )
        raise SystemExit(0)

    # Display table
    table = Table(title="Credential Services")
    table.add_column("Service", style="cyan")
    table.add_column("Facility")
    table.add_column("Source")
    table.add_column("Credentials", style="green")
    table.add_column("URL", style="dim")

    for service, info in sorted(services.items()):
        status = "✓ stored" if info["has_creds"] else "✗ missing"
        style = "green" if info["has_creds"] else "yellow"
        table.add_row(
            service,
            info["facility"],
            info["source"],
            f"[{style}]{status}[/{style}]",
            info["url"][:50] + "..." if len(info["url"]) > 50 else info["url"],
        )

    console.print(table)


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
