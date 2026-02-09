"""Wiki CLI commands: Site operations and session management.

Provides commands for managing wiki sites, testing connectivity,
and managing cached authentication sessions.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def wiki():
    """Manage wiki sites and sessions.

    \b
    Site Operations:
      sites               List configured wiki sites for a facility
      test                Test connectivity to a wiki site
      session             Manage cached sessions

    For credential management, use: imas-codex credentials
    """
    pass


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
            console.print(f"  imas-codex credentials set {service}")


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


@wiki.command("clear")
@click.argument("facility")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def wiki_clear(facility: str, force: bool) -> None:
    """Clear all wiki data for a facility.

    Deletes WikiPages, WikiChunks, and WikiArtifacts for the facility.


    \b
    Examples:
      imas-codex wiki clear tcv
      imas-codex wiki clear jt60sa --force
    """
    from imas_codex.discovery.wiki import clear_facility_wiki, get_wiki_stats
    from imas_codex.graph import GraphClient

    wiki_stats = get_wiki_stats(facility)
    pages = wiki_stats.get("pages", 0)
    chunks = wiki_stats.get("chunks", 0)

    with GraphClient() as gc:
        artifact_result = gc.query(
            "MATCH (wa:WikiArtifact {facility_id: $f}) RETURN count(wa) AS cnt",
            f=facility,
        )
        artifacts = artifact_result[0]["cnt"] if artifact_result else 0

    if pages == 0 and artifacts == 0:
        console.print(f"No wiki data to clear for {facility}")
        return

    parts = []
    if pages:
        parts.append(f"{pages:,} pages")
    if chunks:
        parts.append(f"{chunks:,} chunks")
    if artifacts:
        parts.append(f"{artifacts:,} artifacts")
    summary = ", ".join(parts)

    if not force:
        click.confirm(
            f"This will delete {summary} for {facility}. Continue?",
            abort=True,
        )

    result = clear_facility_wiki(facility)
    deleted_parts = []
    if result.get("pages_deleted"):
        deleted_parts.append(f"{result['pages_deleted']:,} pages")
    if result.get("chunks_deleted"):
        deleted_parts.append(f"{result['chunks_deleted']:,} chunks")
    if result.get("artifacts_deleted"):
        deleted_parts.append(f"{result['artifacts_deleted']:,} artifacts")
    console.print(f"[green]✓ Deleted {', '.join(deleted_parts)} for {facility}[/green]")


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
      imas-codex wiki session clear tcv
      imas-codex wiki session clear iter
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
      imas-codex wiki session status tcv
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
