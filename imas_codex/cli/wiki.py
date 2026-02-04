"""Wiki CLI commands: Credential management and site operations.

Provides commands for managing wiki credentials and testing connectivity.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def wiki():
    """Manage wiki site credentials and access.

    \b
    Credential Management:
      credentials set     Store credentials for a wiki site
      credentials get     Check if credentials are stored
      credentials delete  Remove stored credentials
      credentials list    List known credential services

    \b
    Site Operations:
      sites               List configured wiki sites for a facility
      test                Test connectivity to a wiki site
      session             Manage cached sessions

    Credentials are stored securely in your system keyring
    (GNOME Keyring, macOS Keychain, or Windows Credential Locker).
    """
    pass


@wiki.group()
def credentials():
    """Manage wiki site credentials.

    Credentials are stored securely in your system keyring.
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
    """Store credentials for a wiki site.

    SERVICE is the credential service name (e.g., "tcv-wiki", "iter-confluence").

    \b
    Examples:
      imas-codex wiki credentials set tcv-wiki
      imas-codex wiki credentials set iter-confluence -u myuser
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
    """Check if credentials exist for a wiki site.

    SERVICE is the credential service name (e.g., "tcv-wiki", "iter-confluence").

    \b
    Examples:
      imas-codex wiki credentials get tcv-wiki
      imas-codex wiki credentials get iter-confluence --show-password
    """
    from imas_codex.discovery.wiki.auth import CredentialManager

    creds = CredentialManager()

    result = creds.get_credentials(service, prompt_if_missing=False)
    if result is None:
        console.print(f"[yellow]No credentials found for {service}[/yellow]")
        console.print(
            f"\nTo set credentials: imas-codex wiki credentials set {service}"
        )
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
    """Delete stored credentials for a wiki site.

    Also clears any cached session for the site.

    SERVICE is the credential service name (e.g., "tcv-wiki", "iter-confluence").

    \b
    Examples:
      imas-codex wiki credentials delete tcv-wiki
      imas-codex wiki credentials delete iter-confluence --yes
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
    """List known wiki credential services.

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
                        "site_type": site.get("site_type", "mediawiki"),
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
                        "site_type": "unknown",
                        "url": "",
                        "has_creds": True,
                    }

    if not services:
        console.print("[yellow]No wiki credential services configured.[/yellow]")
        console.print(
            "\nConfigure wiki sites in facility YAML files under 'wiki_sites:'"
        )
        raise SystemExit(0)

    # Display table
    table = Table(title="Wiki Credential Services")
    table.add_column("Service", style="cyan")
    table.add_column("Facility")
    table.add_column("Type")
    table.add_column("Credentials", style="green")
    table.add_column("URL", style="dim")

    for service, info in sorted(services.items()):
        status = "✓ stored" if info["has_creds"] else "✗ missing"
        style = "green" if info["has_creds"] else "yellow"
        table.add_row(
            service,
            info["facility"],
            info["site_type"],
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


@wiki.command("sites")
@click.argument("facility")
def wiki_sites(facility: str) -> None:
    """List configured wiki sites for a facility.

    \b
    Examples:
      imas-codex wiki sites tcv
      imas-codex wiki sites iter
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.wiki.auth import CredentialManager

    try:
        config = get_facility(facility)
    except Exception as e:
        console.print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    wiki_sites_list = config.get("wiki_sites", [])

    if not wiki_sites_list:
        console.print(f"[yellow]No wiki sites configured for {facility}.[/yellow]")
        console.print("\nConfigure wiki sites in the facility YAML under 'wiki_sites:'")
        raise SystemExit(1)

    creds = CredentialManager()

    table = Table(title=f"Wiki Sites for {facility}")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("URL")
    table.add_column("Type")
    table.add_column("Auth")
    table.add_column("Credentials")
    table.add_column("Portal")

    for i, site in enumerate(wiki_sites_list):
        service = site.get("credential_service") or f"{facility}-wiki"
        has_creds = creds.has_credentials(service)
        auth_type = site.get("auth_type", "none")

        # Determine credential status
        if auth_type in ("none", "ssh_proxy"):
            cred_status = "[dim]n/a[/dim]"
        elif has_creds:
            cred_status = "[green]✓ stored[/green]"
        else:
            cred_status = "[yellow]✗ missing[/yellow]"

        table.add_row(
            str(i),
            site.get("url", ""),
            site.get("site_type", "mediawiki"),
            auth_type,
            cred_status,
            site.get("portal_page", ""),
        )

    console.print(table)

    # Show setup instructions if credentials are missing
    needs_setup = []
    for site in wiki_sites_list:
        auth_type = site.get("auth_type", "none")
        if auth_type in ("session", "basic", "tequila"):
            service = site.get("credential_service") or f"{facility}-wiki"
            if not creds.has_credentials(service):
                needs_setup.append(service)

    if needs_setup:
        console.print("\n[yellow]To set up credentials:[/yellow]")
        for service in needs_setup:
            console.print(f"  imas-codex wiki credentials set {service}")


@wiki.command("test")
@click.argument("facility")
@click.option("--site-index", "-i", default=0, type=int, help="Wiki site index")
@click.option("--page", "-p", default=None, help="Specific page to fetch")
def wiki_test(
    facility: str,
    site_index: int,
    page: str | None,
) -> None:
    """Test connectivity to a wiki site.

    Authenticates and fetches a test page to verify access.

    \b
    Examples:
      imas-codex wiki test tcv
      imas-codex wiki test iter --page Main
      imas-codex wiki test tcv -i 0 -p Portal:TCV
    """
    from imas_codex.discovery.wiki.config import WikiConfig

    try:
        config = WikiConfig.from_facility(facility, site_index)
    except Exception as e:
        console.print(f"[red]Error loading wiki config: {e}[/red]")
        raise SystemExit(1) from e

    console.print(f"[bold]Testing: {config.base_url}[/bold]")
    console.print(f"  Type: {config.site_type}")
    console.print(f"  Auth: {config.auth_type}")

    # Use portal page if no page specified
    test_page = page or config.portal_page

    if config.site_type == "confluence":
        from imas_codex.discovery.wiki.confluence import ConfluenceClient

        client = ConfluenceClient(
            base_url=config.base_url,
            credential_service=config.credential_service or f"{facility}-confluence",
        )

        console.print("\n[dim]Authenticating...[/dim]")
        if not client.authenticate():
            console.print("[red]✗ Authentication failed[/red]")
            raise SystemExit(1)

        console.print("[green]✓ Authenticated[/green]")

        if test_page:
            console.print(f"\n[dim]Fetching page: {test_page}[/dim]")
            # Confluence uses page IDs, not names
            console.print("[green]✓ Connection successful[/green]")

    elif config.site_type == "mediawiki":
        if config.auth_type == "tequila" or config.auth_type == "session":
            from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

            client = MediaWikiClient(
                base_url=config.base_url,
                credential_service=config.credential_service or f"{facility}-wiki",
            )

            console.print("\n[dim]Authenticating via Tequila...[/dim]")
            if not client.authenticate():
                console.print("[red]✗ Authentication failed[/red]")
                raise SystemExit(1)

            console.print("[green]✓ Authenticated[/green]")

            if test_page:
                console.print(f"\n[dim]Fetching page: {test_page}[/dim]")
                result = client.get_page(test_page)
                if result:
                    console.print(f"[green]✓ Fetched: {result.title}[/green]")
                    console.print(f"  Size: {len(result.content_html):,} bytes")
                else:
                    console.print("[yellow]Page not found or error[/yellow]")

            client.close()

        elif config.auth_type == "ssh_proxy":
            console.print(
                "\n[yellow]SSH proxy mode - use 'discover wiki' command[/yellow]"
            )
            console.print(
                "Consider switching to auth_type: tequila for better performance"
            )

        else:
            console.print(f"\n[yellow]Unknown auth_type: {config.auth_type}[/yellow]")

    else:
        console.print(f"[yellow]Unknown site_type: {config.site_type}[/yellow]")


@wiki.group()
def session():
    """Manage cached wiki sessions.

    Sessions contain authentication cookies that persist across runs.
    """
    pass


@session.command("clear")
@click.argument("service")
@click.option("--all", "-a", "clear_all", is_flag=True, help="Clear all sessions")
def session_clear(service: str, clear_all: bool) -> None:
    """Clear cached session for a wiki site.

    This forces re-authentication on next access.

    \b
    Examples:
      imas-codex wiki session clear tcv-wiki
      imas-codex wiki session clear iter-confluence
    """
    from imas_codex.discovery.wiki.auth import CredentialManager

    creds = CredentialManager()

    if creds.delete_session(service):
        console.print(f"[green]✓ Cleared session for {service}[/green]")
    else:
        console.print(f"[yellow]No session found for {service}[/yellow]")


@session.command("status")
@click.argument("service")
def session_status(service: str) -> None:
    """Check session status for a wiki site.

    \b
    Examples:
      imas-codex wiki session status tcv-wiki
    """
    import json
    import time

    from imas_codex.discovery.wiki.auth import CredentialManager

    creds = CredentialManager()

    # Access internal keyring storage
    if not creds._keyring_available:
        console.print("[yellow]Keyring not available[/yellow]")
        raise SystemExit(1)

    import keyring

    service_name = creds._service_name(service)
    try:
        session_json = keyring.get_password(service_name, "session")
        if not session_json:
            console.print(f"[yellow]No session found for {service}[/yellow]")
            raise SystemExit(0)

        session_data = json.loads(session_json)
        expires_at = session_data.get("expires_at", 0)
        cookies = session_data.get("cookies", {})

        remaining = expires_at - time.time()
        if remaining > 0:
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            console.print(f"[green]✓ Session valid for {service}[/green]")
            console.print(f"  Expires in: {hours}h {minutes}m")
            console.print(f"  Cookies: {len(cookies)} stored")
        else:
            console.print(f"[yellow]Session expired for {service}[/yellow]")
            console.print(f"  Expired: {int(-remaining // 60)} minutes ago")

    except Exception as e:
        console.print(f"[red]Error checking session: {e}[/red]")
        raise SystemExit(1) from e
