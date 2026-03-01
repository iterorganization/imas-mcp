"""Wiki discovery command: Page scanning, scoring and ingestion."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time

import click
from rich.console import Console
from rich.markup import escape as rich_escape

logger = logging.getLogger(__name__)


@click.command()
@click.argument("facility")
@click.option("--source", "-s", help="Specific wiki site URL or index")
@click.option(
    "--cost-limit", "-c", type=float, default=10.0, help="Maximum LLM spend in USD"
)
@click.option(
    "--max-pages", "-n", type=int, default=None, help="Maximum pages to process"
)
@click.option(
    "--max-depth", type=int, default=None, help="Maximum link depth from portal"
)
@click.option(
    "--focus", "-f", help="Focus discovery (e.g., 'equilibrium', 'diagnostics')"
)
@click.option(
    "--scan-only", is_flag=True, help="Only scan pages, skip scoring and ingestion"
)
@click.option(
    "--score-only",
    is_flag=True,
    help="Only score already-discovered pages, skip ingestion",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.option(
    "--rescan",
    is_flag=True,
    default=False,
    help="Re-scan pages even if already in graph",
)
@click.option(
    "--score-workers",
    type=int,
    default=2,
    help="Number of parallel score workers (default: 2)",
)
@click.option(
    "--ingest-workers",
    type=int,
    default=4,
    help="Number of parallel ingest workers (default: 4)",
)
@click.option(
    "--rescan-artifacts",
    is_flag=True,
    default=False,
    help="Re-scan artifacts even if already in graph",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes (e.g., 5). Discovery halts when time expires.",
)
@click.option(
    "--store-images",
    is_flag=True,
    default=False,
    help="Keep image bytes in graph after VLM scoring (default: clear to save storage)",
)
def wiki(
    facility: str,
    source: str | None,
    cost_limit: float,
    max_pages: int | None,
    max_depth: int | None,
    focus: str | None,
    scan_only: bool,
    score_only: bool,
    verbose: bool,
    rescan: bool,
    score_workers: int,
    ingest_workers: int,
    rescan_artifacts: bool,
    time_limit: int | None,
    store_images: bool,
) -> None:
    """Discover wiki pages and build documentation graph.

    Runs parallel wiki discovery workers:

    \b
    - SCAN: Enumerate all pages per site (runs once, cached in graph)
    - SCORE: LLM relevance evaluation with content fetch
    - INGEST: Chunk and embed high-score pages
    - FILE: Score and embed wiki file attachments (PDFs, CSVs, etc.)

    Page scanning runs automatically on first invocation. Use --rescan
    to re-enumerate pages (adds new pages, keeps existing).

    \b
    Examples:
      imas-codex discover wiki jt-60sa              # Full discovery
      imas-codex discover wiki jt-60sa --scan-only  # Scan pages only
      imas-codex discover wiki tcv -f equilibrium  # Focus scoring
      imas-codex discover wiki jet -c 5.0          # $5 budget
    """
    from imas_codex.cli.logging import configure_cli_logging

    # Auto-detect if rich can run
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.wiki import get_wiki_stats
    from imas_codex.discovery.wiki.graph_ops import (
        recover_failed_pages,
        reset_transient_pages,
    )
    from imas_codex.discovery.wiki.parallel import run_parallel_wiki_discovery

    use_rich = should_use_rich()

    # Always configure file logging (DEBUG level to disk)
    configure_cli_logging("wiki", facility=facility, verbose=verbose)

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    wiki_logger = logging.getLogger("imas_codex.discovery.wiki")
    if not use_rich:
        wiki_logger.setLevel(logging.INFO)

    def log_print(msg: str, style: str = "") -> None:
        """Print to console or log, stripping rich markup."""
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            wiki_logger.info(clean_msg)

    try:
        config = get_facility(facility)
        wiki_sites = config.get("wiki_sites", [])
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    if not wiki_sites:
        log_print(
            f"[yellow]No wiki sites configured for {facility}.[/yellow]\n"
            "Configure wiki sites in the facility YAML under 'wiki_sites:'"
        )
        raise SystemExit(1)

    # Check embedding server availability upfront for ingestion mode.
    if not scan_only and not score_only:
        from imas_codex.embeddings.config import EmbeddingBackend
        from imas_codex.settings import get_embedding_location

        backend_str = get_embedding_location()
        if backend_str == "local":
            backend = EmbeddingBackend.LOCAL
        else:
            backend = EmbeddingBackend.REMOTE

        if backend == EmbeddingBackend.REMOTE:
            from imas_codex.embeddings.readiness import ensure_embedding_ready

            if use_rich and console:
                from rich.status import Status

                with Status(
                    "[cyan]Checking embedding server...[/cyan]",
                    console=console,
                    spinner="dots",
                ) as status:

                    def _readiness_status(msg: str) -> None:
                        status.update(f"[cyan]{rich_escape(msg)}[/cyan]")

                    ok, msg = ensure_embedding_ready(
                        status_fn=_readiness_status, timeout=60.0
                    )

                if ok:
                    log_print(f"[green]✓ {rich_escape(msg)}[/green]")
                else:
                    log_print(f"[red]{rich_escape(msg)}[/red]")
                    log_print(
                        "[dim]Or use --scan-only / --score-only to skip embedding[/dim]"
                    )
                    raise SystemExit(1)
            else:

                def _readiness_log(msg: str, style: str = "info") -> None:
                    log_print(rich_escape(msg))

                ok, msg = ensure_embedding_ready(log_fn=_readiness_log, timeout=60.0)
                if not ok:
                    log_print(msg)
                    raise SystemExit(1)

    # Display wiki sites
    log_print(f"[bold]Documentation sources for {facility}:[/bold]")
    for i, site in enumerate(wiki_sites):
        from urllib.parse import urlparse as _parse_url

        url = site.get("url", "")
        desc = site.get("description", "")
        parsed = _parse_url(url)
        name = parsed.path.rstrip("/").rsplit("/", 1)[-1] or url
        if desc:
            log_print(f"  [{i}] {name}: {desc}")
        else:
            log_print(f"  [{i}] {url}")

    # Show aggregated auth info when all sites share the same auth
    _auth_types = {s.get("auth_type") for s in wiki_sites if s.get("auth_type")}
    _show_per_site_auth = len(_auth_types) > 1
    if _auth_types and not _show_per_site_auth:
        auth_label = next(iter(_auth_types))
        log_print(f"[dim]Auth: {auth_label}[/dim]")
    if use_rich:
        console.print()

    site_indices = list(range(len(wiki_sites)))
    if source:
        try:
            idx = int(source)
            if 0 <= idx < len(wiki_sites):
                site_indices = [idx]
            else:
                log_print(f"[red]Invalid site index: {idx}[/red]")
                raise SystemExit(1)
        except ValueError:
            matched = [
                i
                for i, s in enumerate(wiki_sites)
                if source.lower() in s.get("url", "").lower()
            ]
            if matched:
                site_indices = matched
            else:
                log_print(f"[red]No site matching '{source}'[/red]")
                raise SystemExit(1) from None

    # ================================================================
    # Phase 1: Preflight - validate credentials, bulk discovery
    # ================================================================

    # Check wiki stats once (facility-level, not per-site)
    wiki_stats = get_wiki_stats(facility)
    existing_pages = wiki_stats.get("pages", 0) or wiki_stats.get("total", 0)
    should_bulk_discover = rescan or existing_pages == 0

    # Check existing artifact count
    existing_artifacts = 0
    _site_page_counts: list[tuple[str, int]] = []
    _site_artifact_counts: list[tuple[str, int]] = []
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as _gc:
            _art_result = _gc.query(
                "MATCH (wa:WikiArtifact {facility_id: $f}) RETURN count(wa) AS cnt",
                f=facility,
            )
            existing_artifacts = _art_result[0]["cnt"] if _art_result else 0

            # Per-site page counts
            _page_rows = _gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $f})
                WITH wp.url AS url
                RETURN url, count(*) AS cnt
                """,
                f=facility,
            )
            _site_urls = [s.get("url", "").rstrip("/") for s in wiki_sites]
            # Sort by longest prefix first so tfe1 matches before tfe
            _site_order = sorted(
                range(len(_site_urls)), key=lambda i: len(_site_urls[i]), reverse=True
            )
            _site_names = []
            for s in wiki_sites:
                from urllib.parse import urlparse as _parse_site_url

                _p = _parse_site_url(s.get("url", ""))
                _site_names.append(
                    _p.path.rstrip("/").rsplit("/", 1)[-1] or s.get("url", "")
                )
            _spc: dict[int, int] = {}
            for row in _page_rows:
                purl = row["url"] or ""
                for si in _site_order:
                    surl = _site_urls[si]
                    if purl == surl or purl.startswith(surl + "/"):
                        _spc[si] = _spc.get(si, 0) + row["cnt"]
                        break
            _site_page_counts = [
                (_site_names[si], cnt) for si, cnt in sorted(_spc.items())
            ]

            # Per-site artifact counts
            _art_rows = _gc.query(
                """
                MATCH (wa:WikiArtifact {facility_id: $f})<-[:HAS_ARTIFACT]-(wp:WikiPage)
                WITH wp.url AS url, count(wa) AS cnt
                RETURN url, cnt
                """,
                f=facility,
            )
            _sac: dict[int, int] = {}
            for row in _art_rows:
                purl = row["url"] or ""
                for si in _site_order:
                    surl = _site_urls[si]
                    if purl == surl or purl.startswith(surl + "/"):
                        _sac[si] = _sac.get(si, 0) + row["cnt"]
                        break
            _site_artifact_counts = [
                (_site_names[si], cnt) for si, cnt in sorted(_sac.items())
            ]
    except Exception:
        pass

    if existing_pages > 0 and not should_bulk_discover:
        log_print(f"[dim]Found {existing_pages:,} pages in graph, skipping scan[/dim]")
        if _site_page_counts and len(_site_page_counts) > 1:
            for sname, cnt in _site_page_counts:
                log_print(f"[dim]  {sname}: {cnt:,} pages[/dim]")
        log_print("[dim]Use --rescan to re-enumerate pages[/dim]")

        if existing_artifacts > 0:
            log_print("")
            log_print(f"[dim]Found {existing_artifacts:,} artifacts in graph[/dim]")
            if _site_artifact_counts and len(_site_artifact_counts) > 1:
                for sname, cnt in _site_artifact_counts:
                    log_print(f"[dim]  {sname}: {cnt:,} artifacts[/dim]")
            log_print("[dim]Use --rescan-artifacts to re-enumerate artifacts[/dim]")
    elif rescan and existing_pages > 0:
        log_print(
            f"[yellow]Rescan: adding new pages (keeping {existing_pages} existing)[/yellow]"
        )

    # Reset orphaned pages once (facility-level)
    reset_counts = reset_transient_pages(facility, silent=True)
    if any(reset_counts.values()):
        total_reset = sum(reset_counts.values())
        log_print(f"[dim]Reset {total_reset} orphaned pages from previous run[/dim]")

    # Recover pages that were marked failed due to transient fetch failures.
    # These pages have exhausted fetch_retries but have no actual error — they
    # just couldn't be fetched (e.g. auth session expired). Reset them back to
    # scanned so they get another chance.
    recovered = recover_failed_pages(facility)
    if recovered > 0:
        log_print(
            f"[dim]Recovered {recovered:,} pages from transient fetch failures[/dim]"
        )

    # Pre-warm SSH ControlMaster for all sites that need SSH access.
    # This prevents race conditions when bulk_discover_pages tries SSH
    # before the ControlMaster is established.
    _ssh_hosts_warmed: set[str] = set()
    for _site in wiki_sites:
        _am = _site.get("access_method", "direct")
        if _am in ("vpn", "tunnel") or _site.get("ssh_available", False):
            _sh = _site.get("ssh_host") or config.get("ssh_host")
            if _sh and _sh not in _ssh_hosts_warmed:
                try:
                    subprocess.run(
                        ["ssh", "-O", "check", _sh],
                        capture_output=True,
                        timeout=5,
                    )
                    _ssh_hosts_warmed.add(_sh)
                except Exception:
                    try:
                        subprocess.run(
                            ["ssh", _sh, "true"],
                            capture_output=True,
                            timeout=30,
                        )
                        _ssh_hosts_warmed.add(_sh)
                        log_print(f"[dim]SSH ControlMaster established for {_sh}[/dim]")
                    except Exception as _e:
                        log_print(
                            f"[yellow]Warning: SSH to {_sh} failed: {_e}[/yellow]"
                        )

    # Validate all sites and run bulk discovery for each
    site_configs: list[dict] = []
    validated_cred_services: set[str] = set()

    # Pre-check Keycloak domain reachability once (avoids 5s timeout × N sites)
    _keycloak_unreachable_domains: set[str] = set()
    _keycloak_domains = set()
    for _ws in wiki_sites:
        if _ws.get("auth_type") == "keycloak":
            from urllib.parse import urlparse as _kc_parse

            _kc_domain = _kc_parse(_ws.get("url", "")).netloc
            if _kc_domain:
                _keycloak_domains.add(_kc_domain)
    for _kc_dom in _keycloak_domains:
        import socket as _sock

        try:
            _kc_sock = _sock.create_connection((_kc_dom, 443), timeout=5)
            _kc_sock.close()
        except Exception:
            _keycloak_unreachable_domains.add(_kc_dom)
            log_print(
                f"[yellow]{_kc_dom} not reachable directly, will use SSH for page discovery[/yellow]"
            )

    for site_idx in site_indices:
        site = wiki_sites[site_idx]
        site_type = site.get("site_type", "mediawiki")
        base_url = site.get("url", "")
        portal_page = site.get("portal_page", "Main_Page")
        auth_type = site.get("auth_type")
        credential_service = site.get("credential_service")
        access_method = site.get("access_method", "direct")

        # Determine SSH host
        ssh_host = None
        if access_method in ("vpn", "tunnel") or site.get("ssh_available", False):
            ssh_host = site.get("ssh_host")
            if not ssh_host:
                ssh_host = config.get("ssh_host")

        # Short name for multi-site display
        from urllib.parse import urlparse as _urlparse

        parsed_url = _urlparse(base_url)
        short_name = parsed_url.path.rstrip("/").rsplit("/", 1)[-1] or base_url
        site_desc = site.get("description", "")

        # Site header
        if site_desc:
            log_print(f"\n[bold cyan]{short_name}[/bold cyan] \u2014 {site_desc}")
        else:
            log_print(f"\n[bold cyan]{base_url}[/bold cyan]")

        if _show_per_site_auth:
            if site_type == "twiki":
                log_print("  [dim]TWiki via SSH[/dim]")
            elif auth_type in ("tequila", "session"):
                log_print("  [dim]Tequila auth[/dim]")
            elif auth_type == "basic" and credential_service:
                log_print(f"  [dim]HTTP Basic ({credential_service})[/dim]")
            elif auth_type == "keycloak":
                log_print("  [dim]keycloak[/dim]")
            elif access_method in ("vpn", "tunnel") and ssh_host:
                log_print(f"  [dim]VPN via {ssh_host}[/dim]")

        # Validate credentials once per credential_service
        if auth_type in ("tequila", "session", "basic") and credential_service:
            if credential_service not in validated_cred_services:
                from imas_codex.discovery.wiki.auth import CredentialManager

                cred_mgr = CredentialManager()
                creds = cred_mgr.get_credentials(
                    credential_service, prompt_if_missing=True
                )
                if creds is None:
                    log_print(
                        f"[red]Credentials required for {credential_service}.[/red]\n"
                        f"Set them with: imas-codex credentials set {credential_service}\n"
                        f"Or set environment variables:\n"
                        f"  export {cred_mgr._env_var_name(credential_service, 'username')}=your_username\n"
                        f"  export {cred_mgr._env_var_name(credential_service, 'password')}=your_password"
                    )
                    raise SystemExit(1)
                log_print(f"  [dim]Authenticated as: {creds[0]}[/dim]")
                validated_cred_services.add(credential_service)

        # Bulk page discovery
        bulk_discovered = 0
        if should_bulk_discover and not score_only:
            from imas_codex.discovery.wiki.parallel import bulk_discover_pages

            # Skip Keycloak auth if the domain is unreachable
            effective_auth_type = auth_type or "none"
            if auth_type == "keycloak" and ssh_host:
                from urllib.parse import urlparse as _auth_parse

                domain = _auth_parse(base_url).netloc
                if domain in _keycloak_unreachable_domains:
                    effective_auth_type = "none"

            discover_kwargs: dict = {
                "facility": facility,
                "site_type": site_type,
                "base_url": base_url,
                "ssh_host": ssh_host,
                "auth_type": effective_auth_type,
                "credential_service": credential_service,
                "access_method": access_method,
            }

            if site_type == "confluence":
                discover_kwargs["space_key"] = portal_page
            elif site_type == "twiki":
                discover_kwargs["webs"] = site.get("webs", ["Main"])
            elif site_type == "twiki_raw":
                discover_kwargs["data_path"] = site.get("data_path", base_url)
                discover_kwargs["web_name"] = site.get("web_name", "Main")
                discover_kwargs["exclude_patterns"] = site.get("exclude_patterns")
            elif site_type == "static_html":
                from imas_codex.discovery.wiki.parallel import (
                    _get_exclude_prefixes,
                )

                discover_kwargs["exclude_prefixes"] = _get_exclude_prefixes(
                    facility, base_url
                )

            def bulk_progress_log(msg, _):
                wiki_logger.info(f"BULK: {msg}")

            try:
                if use_rich:
                    from rich.status import Status

                    with Status(
                        f"[cyan]Bulk discovery: {rich_escape(base_url)}...[/cyan]",
                        console=console,
                        spinner="dots",
                    ) as status:

                        def bulk_progress_rich(msg, _, _url=base_url):
                            safe_url = rich_escape(_url)
                            safe_msg = rich_escape(msg)
                            if "pages" in msg:
                                status.update(f"[cyan]{safe_url}: {safe_msg}[/cyan]")
                            elif "creating" in msg:
                                status.update(f"[cyan]{safe_url}: {safe_msg}[/cyan]")
                            elif "created" in msg:
                                status.update(f"[green]{safe_url}: {safe_msg}[/green]")

                        bulk_discovered = bulk_discover_pages(
                            **discover_kwargs, on_progress=bulk_progress_rich
                        )

                    if bulk_discovered > 0:
                        log_print(f"  [green]{bulk_discovered:,} pages[/green]")
                else:
                    bulk_discovered = bulk_discover_pages(
                        **discover_kwargs, on_progress=bulk_progress_log
                    )
                    if bulk_discovered > 0:
                        wiki_logger.info(f"{short_name}: {bulk_discovered} pages")
            except Exception as e:
                log_print(f"  [red]Bulk discovery failed: {e}[/red]")
                wiki_logger.warning("Bulk discovery failed for %s: %s", base_url, e)
                continue

        # Artifact scanning
        should_discover_artifacts_site = (
            rescan_artifacts
            or (bulk_discovered > 0)
            or (existing_pages > 0 and existing_artifacts == 0)
        ) and not score_only
        if should_discover_artifacts_site:
            from imas_codex.discovery.wiki.parallel import bulk_discover_artifacts

            def artifact_progress_log(msg, _):
                wiki_logger.info(f"ARTIFACTS: {msg}")

            wiki_client = None
            if site_type == "mediawiki" and credential_service:
                if auth_type == "tequila":
                    from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

                    wiki_client = MediaWikiClient(
                        base_url=base_url,
                        credential_service=credential_service,
                        verify_ssl=False,
                    )
                    wiki_client.authenticate()
                elif auth_type == "basic":
                    from imas_codex.discovery.wiki.mediawiki import (
                        BasicAuthWikiClient,
                    )

                    wiki_client = BasicAuthWikiClient(
                        base_url=base_url,
                        credential_service=credential_service,
                        verify_ssl=False,
                    )
                    wiki_client.authenticate()
                elif auth_type == "keycloak":
                    from imas_codex.discovery.wiki.keycloak import (
                        KeycloakWikiClient,
                    )

                    kc = KeycloakWikiClient(
                        base_url=base_url,
                        credential_service=credential_service,
                    )
                    if kc.authenticate():
                        wiki_client = kc
                    else:
                        log_print(
                            "  [yellow]Keycloak auth failed, artifact discovery will use SSH[/yellow]"
                        )

            artifacts_discovered = 0
            page_artifacts = {}

            if use_rich:
                from rich.status import Status

                with Status(
                    f"[cyan]Artifact discovery: {rich_escape(base_url)}...[/cyan]",
                    console=console,
                    spinner="dots",
                ) as status:

                    def artifact_progress_rich(msg, _, _url=base_url):
                        safe_url = rich_escape(_url)
                        safe_msg = rich_escape(msg)
                        if "scanned" in msg or "scanning" in msg:
                            status.update(f"[cyan]{safe_url}: {safe_msg}[/cyan]")
                        elif "batch" in msg:
                            status.update(
                                f"[cyan]{safe_url} artifacts: {safe_msg}[/cyan]"
                            )
                        elif "created" in msg or "discovered" in msg:
                            status.update(f"[green]{safe_url}: {safe_msg}[/green]")

                    try:
                        artifacts_discovered, page_artifacts = bulk_discover_artifacts(
                            facility=facility,
                            base_url=base_url,
                            site_type=site_type,
                            ssh_host=ssh_host,
                            wiki_client=wiki_client,
                            credential_service=credential_service,
                            access_method=access_method,
                            data_path=site.get("data_path"),
                            pub_path=site.get("pub_path"),
                            space_key=portal_page
                            if site_type == "confluence"
                            else None,
                            on_progress=artifact_progress_rich,
                        )
                    except Exception:
                        logger.exception("Artifact discovery failed for %s", base_url)
                        log_print(
                            f"  [red]Artifact discovery failed for {short_name}[/red]"
                        )
            else:
                try:
                    artifacts_discovered, page_artifacts = bulk_discover_artifacts(
                        facility=facility,
                        base_url=base_url,
                        site_type=site_type,
                        ssh_host=ssh_host,
                        wiki_client=wiki_client,
                        credential_service=credential_service,
                        access_method=access_method,
                        data_path=site.get("data_path"),
                        pub_path=site.get("pub_path"),
                        space_key=portal_page if site_type == "confluence" else None,
                        on_progress=artifact_progress_log,
                    )
                except Exception:
                    logger.exception("Artifact discovery failed for %s", base_url)
                    log_print(
                        f"  [red]Artifact discovery failed for {short_name}[/red]"
                    )

            if wiki_client:
                wiki_client.close()

            if artifacts_discovered > 0:
                log_print(f"  [green]{artifacts_discovered:,} artifacts[/green]")

        site_configs.append(
            {
                "site_type": site_type,
                "base_url": base_url,
                "portal_page": portal_page,
                "ssh_host": ssh_host,
                "auth_type": auth_type,
                "credential_service": credential_service,
                "short_name": short_name,
                "max_wiki_connections": site.get("max_wiki_connections", 10),
            }
        )

    if not site_configs:
        log_print("[yellow]No sites to process[/yellow]")
        raise SystemExit(1)

    # ================================================================
    # Phase 2: Score/ingest all sites with unified progress display
    # ================================================================

    worker_parts = []
    if not scan_only:
        worker_parts.append(f"{score_workers} score")
        worker_parts.append(f"{ingest_workers} ingest")
        worker_parts.append("2 artifact")
    log_print(f"\nWorkers: {', '.join(worker_parts)}")
    if not scan_only:
        log_print(f"Cost limit: ${cost_limit:.2f}")
    if max_pages:
        log_print(f"Page limit: {max_pages}")
    if time_limit is not None:
        log_print(f"Time limit: {time_limit} min")
    if focus and not scan_only:
        log_print(f"Focus: {focus}")
    if len(site_configs) > 1:
        log_print(f"Sites to process: {len(site_configs)}")

    try:

        async def run_all_sites_unified(
            _facility: str,
            _site_configs: list[dict],
            _cost_limit: float,
            _max_pages: int | None,
            _max_depth: int | None,
            _focus: str | None,
            _scan_only: bool,
            _score_only: bool,
            _use_rich: bool,
            _num_score_workers: int,
            _num_ingest_workers: int,
            _deadline: float | None = None,
            _store_images: bool = False,
        ):
            combined: dict = {
                "scanned": 0,
                "scored": 0,
                "ingested": 0,
                "artifacts": 0,
                "images_scored": 0,
                "cost": 0.0,
                "elapsed_seconds": 0.0,
            }
            remaining_budget = _cost_limit
            remaining_pages = _max_pages

            # Logging-only callbacks for non-rich mode
            def log_on_scan(msg, stats, results=None):
                if msg != "idle":
                    wiki_logger.info(f"SCAN: {msg}")

            def log_on_score(msg, stats, results=None):
                if msg == "provider_budget_exhausted":
                    wiki_logger.warning(
                        "API key budget exhausted — LLM workers halting"
                    )
                elif msg != "idle":
                    wiki_logger.info(f"SCORE: {msg}")

            def log_on_ingest(msg, stats, results=None):
                if msg != "idle":
                    wiki_logger.info(f"INGEST: {msg}")

            if not _use_rich:
                _multi = len(_site_configs) > 1
                for i, sc in enumerate(_site_configs):
                    if remaining_budget <= 0:
                        wiki_logger.info("Budget exhausted, skipping remaining sites")
                        break
                    if remaining_pages is not None and remaining_pages <= 0:
                        wiki_logger.info("Page limit reached, skipping remaining sites")
                        break
                    if _deadline is not None and time.time() >= _deadline:
                        wiki_logger.info("Time limit reached, skipping remaining sites")
                        break

                    wiki_logger.info(
                        "Processing site %d/%d: %s",
                        i + 1,
                        len(_site_configs),
                        sc["base_url"],
                    )

                    try:
                        result = await run_parallel_wiki_discovery(
                            facility=_facility,
                            site_type=sc["site_type"],
                            base_url=sc["base_url"],
                            portal_page=sc["portal_page"],
                            ssh_host=sc["ssh_host"],
                            auth_type=sc["auth_type"],
                            credential_service=sc["credential_service"],
                            cost_limit=remaining_budget,
                            page_limit=remaining_pages,
                            max_depth=_max_depth,
                            focus=_focus,
                            num_scan_workers=1,
                            num_score_workers=_num_score_workers,
                            num_ingest_workers=_num_ingest_workers,
                            scan_only=_scan_only,
                            score_only=_score_only,
                            store_images=_store_images,
                            bulk_discover=False,
                            skip_reset=_multi,
                            deadline=_deadline,
                            on_scan_progress=log_on_scan,
                            on_score_progress=log_on_score,
                            on_ingest_progress=log_on_ingest,
                            max_wiki_connections=sc.get("max_wiki_connections", 10),
                        )
                    except Exception as e:
                        wiki_logger.warning("Site %s failed: %s", sc["base_url"], e)
                        continue

                    for key in ("scanned", "scored", "ingested", "artifacts"):
                        combined[key] += result.get(key, 0)
                    combined["cost"] += result.get("cost", 0)
                    combined["elapsed_seconds"] += result.get("elapsed_seconds", 0)

                    remaining_budget -= result.get("cost", 0)
                    if remaining_pages is not None:
                        remaining_pages -= result.get("scored", 0)

                return combined

            # Rich mode: single unified display across all sites
            from imas_codex.cli.discover.common import create_discovery_monitor
            from imas_codex.discovery.wiki.progress import WikiProgressDisplay

            multi_site = len(_site_configs) > 1

            service_monitor = create_discovery_monitor(
                config,
                check_graph=True,
                check_embed=not (_scan_only or _score_only),
                check_model=not _scan_only,
            )

            # Suppress noisy INFO logs during rich display.
            # Note: discovery.wiki.state is NOT suppressed — its WARNING-level
            # messages about wiki overwhelm, backoff and recovery are critical.
            for mod in (
                "imas_codex.embeddings",
                "imas_codex.discovery.wiki.adapters",
                "imas_codex.discovery.wiki.mediawiki",
                "imas_codex.discovery.wiki.pipeline",
                "imas_codex.discovery.wiki.scoring",
                "imas_codex.discovery.wiki.prefetch",
            ):
                logging.getLogger(mod).setLevel(logging.WARNING)

            display = WikiProgressDisplay(
                facility=_facility,
                cost_limit=_cost_limit,
                page_limit=_max_pages,
                focus=_focus or "",
                console=console,
                scan_only=_scan_only,
                score_only=_score_only,
            )
            # Set service_monitor BEFORE entering the Live context so the
            # SERVERS row renders from the very first frame.
            display.service_monitor = service_monitor
            await service_monitor.__aenter__()

            with display:
                if multi_site:
                    display.set_site_info(
                        site_name=_site_configs[0]["base_url"],
                        site_index=0,
                        total_sites=len(_site_configs),
                    )

                async def refresh_graph_state():
                    from imas_codex.discovery.wiki.parallel import (
                        get_wiki_discovery_stats,
                    )

                    while True:
                        try:
                            stats = get_wiki_discovery_stats(_facility)
                            display.update_from_graph(
                                total_pages=stats.get("total", 0),
                                pages_scanned=stats.get("scanned", 0),
                                pages_scored=stats.get("scored", 0),
                                pages_ingested=stats.get("ingested", 0),
                                pages_skipped=stats.get("skipped", 0),
                                pages_failed=stats.get("failed", 0),
                                pending_score=stats.get(
                                    "pending_score",
                                    stats.get("scanned", 0),
                                ),
                                pending_ingest=stats.get("pending_ingest", 0),
                                pending_artifact_score=stats.get(
                                    "pending_artifact_score", 0
                                ),
                                pending_artifact_ingest=stats.get(
                                    "pending_artifact_ingest", 0
                                ),
                                accumulated_cost=stats.get("accumulated_cost", 0.0),
                                accumulated_page_cost=stats.get(
                                    "accumulated_page_cost", 0.0
                                ),
                                accumulated_artifact_cost=stats.get(
                                    "accumulated_artifact_cost", 0.0
                                ),
                                accumulated_image_cost=stats.get(
                                    "accumulated_image_cost", 0.0
                                ),
                                total_artifacts=stats.get("total_artifacts", 0),
                                docs_ingested=stats.get("docs_ingested", 0),
                                docs_scored=stats.get("artifacts_scored", 0),
                                artifacts_failed=stats.get("artifacts_failed", 0),
                                artifacts_deferred=stats.get("artifacts_deferred", 0),
                                artifacts_skipped=stats.get("artifacts_skipped", 0),
                                images_scored=stats.get("images_scored", 0),
                                pending_image_score=stats.get("pending_image_score", 0),
                                historic_score_rate=stats.get("historic_score_rate"),
                                historic_ingest_rate=stats.get("historic_ingest_rate"),
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            wiki_logger.debug("Graph refresh failed: %s", e)
                        await asyncio.sleep(0.5)

                async def queue_ticker():
                    while True:
                        try:
                            display.tick()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            wiki_logger.debug("Display tick failed: %s", e)
                        await asyncio.sleep(0.15)

                refresh_task = asyncio.create_task(refresh_graph_state())
                ticker_task = asyncio.create_task(queue_ticker())

                def on_scan(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "title": r.get("id", "?").split(":")[-1],
                                "out_links": r.get("out_degree", 0),
                                "depth": r.get("depth", 0),
                            }
                            for r in results[:5]
                        ]
                    display.update_scan(msg, stats, result_dicts)

                def on_score(msg, stats, results=None):
                    if msg == "provider_budget_exhausted":
                        display.state.provider_budget_exhausted = True
                        return
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "title": r.get("id", "?").split(":")[-1],
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                                "is_physics": r.get("is_physics", False),
                                "skipped": r.get("skipped", False),
                                "skip_reason": r.get("skip_reason", ""),
                            }
                            for r in results[:5]
                        ]
                    display.update_score(msg, stats, result_dicts)

                def on_ingest(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "title": r.get("id", "?").split(":")[-1],
                                "score": r.get("score"),
                                "description": r.get("description", ""),
                                "physics_domain": r.get("physics_domain"),
                                "chunk_count": r.get("chunk_count", 0),
                            }
                            for r in results[:5]
                        ]
                    display.update_ingest(msg, stats, result_dicts)

                def on_artifact(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "filename": r.get("filename", "unknown"),
                                "artifact_type": r.get("artifact_type", ""),
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                                "chunk_count": r.get("chunk_count", 0),
                            }
                            for r in results[:5]
                        ]
                    display.update_docs(msg, stats, result_dicts)

                def on_artifact_score(msg, stats, results=None):
                    if msg == "provider_budget_exhausted":
                        display.state.provider_budget_exhausted = True
                        return
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "filename": r.get("filename", "unknown"),
                                "artifact_type": r.get("artifact_type", ""),
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                            }
                            for r in results[:5]
                        ]
                    display.update_docs_score(msg, stats, result_dicts)

                def on_image(msg, stats, results=None):
                    if msg == "provider_budget_exhausted":
                        display.state.provider_budget_exhausted = True
                        return
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "id": r.get("id", "unknown"),
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                                "purpose": r.get("purpose", ""),
                            }
                            for r in results[:5]
                        ]
                    display.update_image(msg, stats, result_dicts)

                def on_worker_status(worker_group):
                    display.update_worker_status(worker_group)

                try:
                    for i, sc in enumerate(_site_configs):
                        if i > 0 and multi_site:
                            display.advance_site(sc["base_url"], i)

                        if remaining_budget <= 0:
                            break
                        if remaining_pages is not None and remaining_pages <= 0:
                            break
                        if _deadline is not None and time.time() >= _deadline:
                            break

                        try:
                            result = await run_parallel_wiki_discovery(
                                facility=_facility,
                                site_type=sc["site_type"],
                                base_url=sc["base_url"],
                                portal_page=sc["portal_page"],
                                ssh_host=sc["ssh_host"],
                                auth_type=sc["auth_type"],
                                credential_service=sc["credential_service"],
                                cost_limit=remaining_budget,
                                page_limit=remaining_pages,
                                max_depth=_max_depth,
                                focus=_focus,
                                num_scan_workers=1,
                                num_score_workers=_num_score_workers,
                                num_ingest_workers=_num_ingest_workers,
                                scan_only=_scan_only,
                                score_only=_score_only,
                                store_images=_store_images,
                                bulk_discover=False,
                                skip_reset=multi_site,
                                deadline=_deadline,
                                on_scan_progress=on_scan,
                                on_score_progress=on_score,
                                on_ingest_progress=on_ingest,
                                on_docs_progress=on_artifact,
                                on_artifact_score_progress=on_artifact_score,
                                on_image_progress=on_image,
                                on_worker_status=on_worker_status,
                                service_monitor=service_monitor,
                                max_wiki_connections=sc.get("max_wiki_connections", 10),
                            )
                        except Exception as e:
                            wiki_logger.warning("Site %s failed: %s", sc["base_url"], e)
                            continue

                        for key in (
                            "scanned",
                            "scored",
                            "ingested",
                            "artifacts",
                            "images_scored",
                        ):
                            combined[key] += result.get(key, 0)
                        combined["cost"] += result.get("cost", 0)
                        combined["elapsed_seconds"] += result.get("elapsed_seconds", 0)

                        remaining_budget -= result.get("cost", 0)
                        if remaining_pages is not None:
                            remaining_pages -= result.get("scored", 0)

                finally:
                    # Final graph refresh so progress bars show 100% state
                    try:
                        from imas_codex.discovery.wiki.parallel import (
                            get_wiki_discovery_stats,
                        )

                        stats = get_wiki_discovery_stats(_facility)
                        display.update_from_graph(
                            total_pages=stats.get("total", 0),
                            pages_scanned=stats.get("scanned", 0),
                            pages_scored=stats.get("scored", 0),
                            pages_ingested=stats.get("ingested", 0),
                            pages_skipped=stats.get("skipped", 0),
                            pages_failed=stats.get("failed", 0),
                            pending_score=stats.get(
                                "pending_score",
                                stats.get("scanned", 0),
                            ),
                            pending_ingest=stats.get("pending_ingest", 0),
                            pending_artifact_score=stats.get(
                                "pending_artifact_score", 0
                            ),
                            pending_artifact_ingest=stats.get(
                                "pending_artifact_ingest", 0
                            ),
                            accumulated_cost=stats.get("accumulated_cost", 0.0),
                            accumulated_page_cost=stats.get(
                                "accumulated_page_cost", 0.0
                            ),
                            accumulated_artifact_cost=stats.get(
                                "accumulated_artifact_cost", 0.0
                            ),
                            accumulated_image_cost=stats.get(
                                "accumulated_image_cost", 0.0
                            ),
                            total_artifacts=stats.get("total_artifacts", 0),
                            docs_ingested=stats.get("docs_ingested", 0),
                            docs_scored=stats.get("artifacts_scored", 0),
                            artifacts_failed=stats.get("artifacts_failed", 0),
                            artifacts_deferred=stats.get("artifacts_deferred", 0),
                            artifacts_skipped=stats.get("artifacts_skipped", 0),
                            images_scored=stats.get("images_scored", 0),
                            pending_image_score=stats.get("pending_image_score", 0),
                            historic_score_rate=stats.get("historic_score_rate"),
                            historic_ingest_rate=stats.get("historic_ingest_rate"),
                        )
                        display.tick()
                    except Exception:
                        pass
                    refresh_task.cancel()
                    ticker_task.cancel()
                    try:
                        await refresh_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await ticker_task
                    except asyncio.CancelledError:
                        pass
                    await service_monitor.__aexit__(None, None, None)

            # Print summary AFTER Live display exits
            display.print_summary()

            return combined

        deadline: float | None = None
        if time_limit is not None:
            deadline = time.time() + (time_limit * 60)

        result = asyncio.run(
            run_all_sites_unified(
                _facility=facility,
                _site_configs=site_configs,
                _cost_limit=cost_limit,
                _max_pages=max_pages,
                _max_depth=max_depth,
                _focus=focus,
                _scan_only=scan_only,
                _score_only=score_only,
                _use_rich=use_rich,
                _num_score_workers=score_workers,
                _num_ingest_workers=ingest_workers,
                _deadline=deadline,
                _store_images=store_images,
            )
        )

        # Display final results
        log_print(
            f"  [green]{result.get('scanned', 0)} pages scanned, "
            f"{result.get('scored', 0)} scored, "
            f"{result.get('ingested', 0)} ingested, "
            f"{result.get('artifacts', 0)} artifacts[/green]"
        )
        log_print(
            f"  [dim]Cost: ${result.get('cost', 0):.2f}, "
            f"Time: {result.get('elapsed_seconds', 0):.1f}s[/dim]"
        )
    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()

    log_print("\n[green]Documentation discovery complete.[/green]")
