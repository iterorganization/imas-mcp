"""Parallel wiki discovery engine.

Main entry point for wiki discovery with async workers. Orchestrates:
- Bulk page/artifact discovery via platform APIs
- Supervised async worker pool (score, ingest, artifact, image)
- Progress reporting and cost tracking

Use run_parallel_wiki_discovery() as the main entry point.
For bulk discovery without scoring, use bulk_discover_pages().
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import urllib.parse
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.embed_worker import embed_description_worker
from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
    supervised_worker,
)
from imas_codex.graph import GraphClient
from imas_codex.graph.models import WikiArtifactStatus, WikiPageStatus

from .graph_ops import (
    INGESTABLE_ARTIFACT_TYPES,
    SCORABLE_ARTIFACT_TYPES,
    _bulk_create_wiki_artifacts,
    _bulk_create_wiki_pages,
    reset_transient_pages,
)
from .state import WikiDiscoveryState
from .workers import (
    artifact_score_worker,
    artifact_worker,
    image_score_worker,
    ingest_worker,
    score_worker,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _persist_discovered_pages(
    facility: str,
    pages: list,
    on_progress: Callable | None = None,
) -> int:
    """Persist discovered pages to graph as 'scanned' status.

    Converts DiscoveredPage instances to batch_data and calls
    _bulk_create_wiki_pages for efficient UNWIND insertion.

    Args:
        facility: Facility ID
        pages: List of DiscoveredPage instances (name, url attrs)
        on_progress: Optional progress callback

    Returns:
        Number of pages created/updated
    """
    from imas_codex.discovery.wiki.scraper import canonical_page_id

    if not pages:
        return 0

    batch_data = [
        {
            "id": canonical_page_id(page.name, facility),
            "title": page.name,
            "url": page.url,
        }
        for page in pages
    ]

    with GraphClient() as gc:
        created = _bulk_create_wiki_pages(
            gc, facility, batch_data, on_progress=on_progress
        )

    logger.info("Created/updated %d pages in graph (scanned status)", created)
    if on_progress:
        on_progress(f"created {created} pages", None)

    return created


def bulk_discover_pages(
    facility: str,
    site_type: str,
    base_url: str,
    *,
    ssh_host: str | None = None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    access_method: str = "direct",
    webs: list[str] | None = None,
    data_path: str | None = None,
    web_name: str = "Main",
    exclude_patterns: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
    max_depth: int | None = None,
    space_key: str | None = None,
    on_progress: Callable | None = None,
) -> int:
    """Bulk discover all wiki pages for a site using the appropriate adapter.

    Unified entry point replacing the per-platform bulk_discover_all_pages_*
    functions. Creates the adapter, runs discovery, and persists to graph.

    Args:
        facility: Facility ID
        site_type: Wiki platform (mediawiki, twiki, twiki_static, twiki_raw,
            static_html)
        base_url: Wiki base URL
        ssh_host: SSH host for proxied access
        auth_type: Authentication type (tequila, keycloak, basic, session, etc.)
        credential_service: Keyring service for credentials
        access_method: Access method ("direct" or "vpn")
        webs: TWiki web names (for twiki sites)
        data_path: TWiki data directory path (for twiki_raw)
        web_name: TWiki web name (for twiki_raw, default "Main")
        exclude_patterns: Topic name regex patterns to skip (for twiki_raw)
        exclude_prefixes: URL path prefixes to exclude (for static_html)
        max_depth: BFS depth limit (for static_html, default 3)
        on_progress: Progress callback(msg, stats)

    Returns:
        Number of pages discovered
    """
    from imas_codex.discovery.wiki.adapters import get_adapter

    logger.info(
        "Starting bulk page discovery: site_type=%s, base_url=%s, auth=%s",
        site_type,
        base_url,
        auth_type,
    )

    # Build adapter kwargs
    adapter_kwargs: dict[str, Any] = {
        "ssh_host": ssh_host,
        "credential_service": credential_service,
    }

    # Auth session setup
    session = None
    close_session = False

    if site_type == "confluence":
        if auth_type == "session" and credential_service:
            from imas_codex.discovery.wiki.confluence import ConfluenceClient

            confluence_client = ConfluenceClient(
                base_url=base_url,
                credential_service=credential_service,
            )
            if not confluence_client.authenticate():
                logger.error("Confluence session auth failed for %s", base_url)
                return 0
            session = confluence_client._get_session()
            adapter_kwargs["session"] = session
            close_session = True
    elif site_type == "mediawiki":
        if auth_type == "tequila" and credential_service:
            # Tequila uses MediaWikiClient (handled by adapter)
            from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

            wiki_client = MediaWikiClient(
                base_url=base_url,
                credential_service=credential_service,
                verify_ssl=False,
            )
            if not wiki_client.authenticate():
                logger.error("Tequila authentication failed for %s", base_url)
                return 0
            adapter_kwargs["wiki_client"] = wiki_client
        elif auth_type == "keycloak" and credential_service:
            from imas_codex.discovery.wiki.keycloak import KeycloakSession

            ks = KeycloakSession(credential_service)
            try:
                session = ks.login(f"{base_url}/")
                adapter_kwargs["session"] = session
                close_session = True
            except Exception as e:
                if ssh_host:
                    logger.warning(
                        "Keycloak login failed (%s: %s), falling back to SSH via %s",
                        type(e).__name__,
                        e,
                        ssh_host,
                    )
                else:
                    logger.error("Keycloak authentication failed: %s", e)
                    return 0
        elif auth_type == "basic" and credential_service:
            import requests

            from imas_codex.discovery.wiki.auth import CredentialManager

            cred_mgr = CredentialManager()
            creds = cred_mgr.get_credentials(credential_service)
            if not creds:
                logger.error("No credentials for %s", credential_service)
                return 0
            session = requests.Session()
            session.auth = (creds[0], creds[1])
            session.verify = False
            adapter_kwargs["session"] = session
            close_session = True

    # Platform-specific adapter kwargs
    if site_type == "confluence":
        if space_key:
            adapter_kwargs["space_key"] = space_key
    elif site_type == "twiki":
        adapter_kwargs["webs"] = webs or ["Main"]
        adapter_kwargs["base_url"] = base_url
    elif site_type == "twiki_static":
        adapter_kwargs["base_url"] = base_url
        adapter_kwargs["access_method"] = access_method
    elif site_type == "twiki_raw":
        adapter_kwargs["data_path"] = data_path or base_url
        adapter_kwargs["web_name"] = web_name
        adapter_kwargs["exclude_patterns"] = exclude_patterns
    elif site_type == "static_html":
        adapter_kwargs["base_url"] = base_url
        adapter_kwargs["access_method"] = access_method
        adapter_kwargs["exclude_prefixes"] = exclude_prefixes
        adapter_kwargs["max_depth"] = max_depth

    try:
        adapter = get_adapter(site_type, **adapter_kwargs)

        pages = adapter.bulk_discover_pages(facility, base_url, on_progress)

        if not pages:
            logger.warning("No pages discovered from %s (%s)", base_url, site_type)
            return 0

        logger.info("Discovered %d pages from %s", len(pages), base_url)
        return _persist_discovered_pages(facility, pages, on_progress)
    finally:
        if close_session and session is not None:
            session.close()


# =============================================================================
# Bulk Artifact Discovery via MediaWiki API
# =============================================================================


def bulk_discover_artifacts(
    facility: str,
    base_url: str,
    site_type: str = "mediawiki",
    ssh_host: str | None = None,
    wiki_client: Any = None,
    credential_service: str | None = None,
    access_method: str = "direct",
    data_path: str | None = None,
    pub_path: str | None = None,
    session: Any = None,
    space_key: str | None = None,
    on_progress: Callable | None = None,
) -> tuple[int, dict[str, list[str]]]:
    """Bulk discover all wiki artifacts via platform API.

    This is much faster than scanning pages - uses dedicated APIs:
    - MediaWiki: list=allimages API (returns all files in one call)
    - TWiki: /pub/ directory listing
    - TWiki static: Parse topic pages for linked files
    - TWiki raw: List files in pub directory via SSH
    - Confluence: /rest/api/content/{id}/child/attachment

    Args:
        facility: Facility ID
        base_url: Wiki base URL
        site_type: Wiki platform type
        ssh_host: SSH host for proxied access
        wiki_client: Authenticated MediaWikiClient (for Tequila)
        credential_service: Keyring service name
        access_method: Access method ("direct" or "vpn")
        data_path: TWiki data directory path (for twiki_raw)
        pub_path: TWiki pub directory path (for twiki_raw)
        session: Authenticated requests.Session (for Confluence)
        space_key: Confluence space key (for scoped discovery)
        on_progress: Progress callback

    Returns:
        Tuple of (count, page_artifacts) where page_artifacts maps
        page names to lists of artifact filenames discovered on that page.
    """
    from imas_codex.discovery.wiki.adapters import get_adapter

    logger.debug(f"Starting bulk artifact discovery for {site_type}...")

    # For Confluence with session auth, create session if not provided
    close_session = False
    if site_type == "confluence" and not session and not ssh_host:
        if credential_service:
            try:
                from imas_codex.discovery.wiki.confluence import ConfluenceClient

                confluence_client = ConfluenceClient(
                    base_url=base_url,
                    credential_service=credential_service,
                )
                if confluence_client.authenticate():
                    session = confluence_client._get_session()
                    close_session = True
                    logger.info("Created Confluence session for artifact discovery")
                else:
                    logger.warning(
                        "Confluence auth failed for artifact discovery at %s",
                        base_url,
                    )
            except Exception as e:
                logger.warning("Failed to create Confluence session: %s", e)

    # Get the appropriate adapter
    adapter = get_adapter(
        site_type=site_type,
        ssh_host=ssh_host,
        wiki_client=wiki_client,
        credential_service=credential_service,
        base_url=base_url,  # Needed for static site adapters
        access_method=access_method,
        data_path=data_path,
        pub_path=pub_path,
        session=session,
        space_key=space_key,
    )

    try:
        # Discover artifacts
        artifacts = adapter.bulk_discover_artifacts(facility, base_url, on_progress)
    finally:
        if close_session and session is not None:
            try:
                session.close()
            except Exception:
                pass

    if not artifacts:
        logger.debug("No artifacts discovered")
        return 0, {}

    logger.debug(f"Discovered {len(artifacts)} artifacts")

    # Create artifact nodes in graph using shared helper (UNWIND + AT_FACILITY)
    # Include linked_pages for HAS_ARTIFACT relationship creation
    batch_data = [
        {
            "id": f"{facility}:{a.filename}",
            "filename": a.filename,
            "url": a.url,
            "artifact_type": a.artifact_type,
            "size_bytes": a.size_bytes,
            "mime_type": a.mime_type,
            "linked_pages": a.linked_pages,
        }
        for a in artifacts
    ]

    with GraphClient() as gc:
        created = _bulk_create_wiki_artifacts(
            gc, facility, batch_data, on_progress=on_progress
        )

    logger.info(f"Created/updated {created} artifact nodes in graph")
    if on_progress:
        on_progress(f"created {created} artifacts", None)

    # Build page → artifact filenames mapping for CLI display
    page_artifacts: dict[str, list[str]] = {}
    for a in artifacts:
        if a.linked_pages:
            for page in a.linked_pages:
                page_artifacts.setdefault(page, []).append(a.filename)
        else:
            page_artifacts.setdefault("(unlinked)", []).append(a.filename)

    return created, page_artifacts


# =============================================================================


async def run_parallel_wiki_discovery(
    facility: str,
    site_type: str,
    base_url: str,
    portal_page: str,
    ssh_host: str | None = None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    cost_limit: float = 10.0,
    page_limit: int | None = None,
    max_depth: int | None = None,
    focus: str | None = None,
    num_scan_workers: int = 1,
    num_score_workers: int = 2,
    num_ingest_workers: int = 4,
    scan_only: bool = False,
    score_only: bool = False,
    store_images: bool = False,
    bulk_discover: bool = True,
    ingest_artifacts: bool = True,
    skip_reset: bool = False,
    deadline: float | None = None,
    on_scan_progress: Callable | None = None,
    on_score_progress: Callable | None = None,
    on_ingest_progress: Callable | None = None,
    on_artifact_progress: Callable | None = None,
    on_artifact_score_progress: Callable | None = None,
    on_image_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    service_monitor: Any = None,
) -> dict[str, Any]:
    """Run parallel wiki discovery with async workers.

    Args:
        ssh_host: SSH host for proxied access (for non-public wikis)
        auth_type: Authentication type (tequila, session, ssh_proxy, or None)
        credential_service: Keyring service name for credentials
        bulk_discover: If True (default), use Special:AllPages for fast
            discovery of all pages upfront. This is 100-300x faster than
            crawling links page-by-page. Only works for MediaWiki sites.
        ingest_artifacts: If True (default), start artifact worker to
            download and ingest artifacts discovered during scanning.
            Artifacts go through score→ingest pipeline:
            discovered → scored (LLM) → ingested (embed)
        skip_reset: If True, skip orphan recovery on entry. The reset
            is now timeout-based and parallel-safe, so this is mainly
            a performance optimization for multi-site loops.
        on_artifact_progress: Progress callback for artifact ingest worker.
        on_artifact_score_progress: Progress callback for artifact score worker.
        on_worker_status: Callback for worker status changes. Called with
            SupervisedWorkerGroup for live status display.
        service_monitor: ServiceMonitor instance for health monitoring.
            When provided, workers gate on service health (pause when
            SSH/VPN is down). Passed through to WikiDiscoveryState.

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

    # Release ALL claims from previous runs.  At startup, this process is the
    # only one that should own claims for this facility, so force-clearing is
    # safe and prevents stale claims from a recently-crashed run blocking work.
    if not skip_reset:
        reset_transient_pages(facility, force=True)

    # Ensure Facility node exists so AT_FACILITY relationships don't fail
    with GraphClient() as gc:
        gc.ensure_facility(facility)

    # Initialize state with auth info
    state = WikiDiscoveryState(
        facility=facility,
        site_type=site_type,
        base_url=base_url,
        portal_page=portal_page,
        ssh_host=ssh_host,
        auth_type=auth_type,
        credential_service=credential_service,
        cost_limit=cost_limit,
        page_limit=page_limit,
        max_depth=max_depth,
        focus=focus,
        service_monitor=service_monitor,
        score_only=score_only,
        store_images=store_images,
        deadline=deadline,
    )

    # Pre-warm SSH ControlMaster if using SSH transport.
    # Ensures the master connection is established before concurrent workers
    # spawn multiple ssh subprocesses (which would race to create it).
    if ssh_host:
        logger.info("Pre-warming SSH ControlMaster to %s...", ssh_host)
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["ssh", "-O", "check", ssh_host],
                capture_output=True,
                timeout=10,
            )
            logger.info("SSH ControlMaster active for %s", ssh_host)
        except Exception:
            # No master yet — establish one with a quick command
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["ssh", ssh_host, "true"],
                    capture_output=True,
                    timeout=30,
                )
                logger.info("SSH ControlMaster established for %s", ssh_host)
            except Exception as e:
                logger.warning("Failed to pre-warm SSH to %s: %s", ssh_host, e)

    # Create worker group for status tracking
    worker_group = SupervisedWorkerGroup()

    # Bulk discovery: use adapter-based APIs to find all pages instantly
    # This replaces the slow scan phase (crawling links page-by-page)
    bulk_discovered = 0
    if bulk_discover and not score_only:

        def bulk_progress(msg, _stats):
            if on_scan_progress:
                on_scan_progress(f"bulk: {msg}", state.scan_stats)

        # Extract data_path for twiki_raw from base_url
        data_path = None
        if site_type == "twiki_raw":
            data_path = base_url
            if data_path.startswith("ssh://"):
                data_path = data_path.split("/", 3)[-1]
                if not data_path.startswith("/"):
                    data_path = "/" + data_path

        # Compute exclude prefixes and max_depth for static_html
        exclude_prefixes = None
        bfs_max_depth = None
        if site_type == "static_html":
            exclude_prefixes = _get_exclude_prefixes(facility, base_url)
            bfs_max_depth = _get_site_max_depth(facility, base_url)

        bulk_discovered = await asyncio.to_thread(
            bulk_discover_pages,
            facility,
            site_type,
            base_url,
            ssh_host=ssh_host,
            auth_type=state.auth_type,
            credential_service=state.credential_service,
            data_path=data_path,
            exclude_prefixes=exclude_prefixes,
            max_depth=bfs_max_depth,
            on_progress=bulk_progress,
        )

        if bulk_discovered:
            logger.info(f"Bulk discovery found {bulk_discovered} pages")
        state.scan_stats.processed = bulk_discovered

    # Bulk artifact discovery: use platform API to find all artifacts
    # This is much faster than scanning each page for links
    bulk_artifacts_discovered = 0
    if bulk_discover and ingest_artifacts and not score_only:
        logger.info("Starting bulk artifact discovery via API...")

        def artifact_progress(msg, _stats):
            if on_artifact_progress:
                on_artifact_progress(f"bulk: {msg}", state.artifact_stats)

        # Determine space_key for Confluence sites (stored on portal_page)
        _space_key = portal_page if site_type == "confluence" else None

        # Bulk discovery uses SSH when available, session for Confluence
        bulk_result = await asyncio.to_thread(
            bulk_discover_artifacts,
            facility,
            base_url,
            site_type,
            ssh_host=ssh_host,
            wiki_client=None,  # removed, use SSH path
            credential_service=state.credential_service,
            access_method="vpn" if ssh_host else "direct",
            space_key=_space_key,
            on_progress=artifact_progress,
        )
        # bulk_discover_artifacts returns (count, page_artifacts_dict)
        if isinstance(bulk_result, tuple):
            bulk_artifacts_discovered, _page_artifacts = bulk_result
        else:
            bulk_artifacts_discovered = bulk_result or 0

        if bulk_artifacts_discovered:
            logger.info(f"Bulk discovery found {bulk_artifacts_discovered} artifacts")
            state.artifact_stats.processed = bulk_artifacts_discovered

    # Create portal page if not exists (may already exist from bulk discovery)
    _seed_portal_page(facility, portal_page, base_url, site_type)

    # Start supervised workers with status tracking
    # Note: scan_worker was removed — bulk_discover_pages() replaces link-crawling.
    # num_scan_workers and scan_only params are kept for CLI backwards compat.
    #
    # --scan-only: return immediately (bulk discovery already ran above)
    # --score-only: start score + artifact_score + image_score workers only
    # default: start all workers (score, ingest, artifact, image)

    if scan_only:
        # Bulk discovery already ran above — nothing more to do.
        elapsed = time.time() - start_time
        return {
            "scanned": state.scan_stats.processed,
            "scored": 0,
            "ingested": 0,
            "artifacts": 0,
            "images_scored": 0,
            "cost": 0.0,
            "elapsed_seconds": elapsed,
            "scan_rate": state.scan_stats.rate,
            "score_rate": 0.0,
        }

    # --- LLM scoring workers (run in both default and score_only modes) ---

    for i in range(num_score_workers):
        worker_name = f"score_worker_{i}"
        status = worker_group.create_status(worker_name, group="triage")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    score_worker,
                    worker_name,
                    state,
                    state.should_stop_scoring,
                    on_progress=on_score_progress,
                    status_tracker=status,
                )
            )
        )

    if ingest_artifacts:
        # Artifact score worker: LLM scoring with text preview extraction
        artifact_score_status = worker_group.create_status(
            "artifact_score_worker", group="docs"
        )
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    artifact_score_worker,
                    "artifact_score_worker",
                    state,
                    state.should_stop_artifact_scoring,
                    on_progress=on_artifact_score_progress,
                    status_tracker=artifact_score_status,
                )
            )
        )

    # Image score worker: VLM caption + scoring for ingested images
    image_score_status = worker_group.create_status(
        "image_score_worker", group="images"
    )
    worker_group.add_task(
        asyncio.create_task(
            supervised_worker(
                image_score_worker,
                "image_score_worker",
                state,
                state.should_stop_image_scoring,
                on_progress=on_image_progress,
                status_tracker=image_score_status,
            )
        )
    )

    # Embed description worker: embeds descriptions for all wiki node types
    # Runs continuously, picking up newly-described nodes from score/ingest workers
    embed_status = worker_group.create_status("embed_worker", group="pages")
    worker_group.add_task(
        asyncio.create_task(
            supervised_worker(
                embed_description_worker,
                "embed_worker",
                state,
                state.should_stop,
                labels=["WikiPage", "WikiArtifact", "Image"],
                status_tracker=embed_status,
            )
        )
    )

    # --- I/O ingest workers (skipped in score_only mode) ---

    if not score_only:
        for i in range(num_ingest_workers):
            worker_name = f"ingest_worker_{i}"
            ingest_status = worker_group.create_status(worker_name, group="pages")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        ingest_worker,
                        worker_name,
                        state,
                        state.should_stop_ingesting,
                        on_progress=on_ingest_progress,
                        status_tracker=ingest_status,
                    )
                )
            )

        if ingest_artifacts:
            # Artifact ingest worker: download and embed scored artifacts
            artifact_ingest_status = worker_group.create_status(
                "artifact_worker", group="docs"
            )
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        artifact_worker,
                        "artifact_worker",
                        state,
                        state.should_stop_artifact_worker,
                        on_progress=on_artifact_progress,
                        status_tracker=artifact_ingest_status,
                    )
                )
            )
    else:
        # In score_only mode, mark I/O worker phases as done so should_stop()
        # doesn't hang waiting for workers that were never started.
        state.ingest_phase.mark_done()
        state.artifact_phase.mark_done()

    # Scan workers were removed (replaced by bulk discovery above).
    # Mark scan phase as done so should_stop() doesn't block on a non-existent worker.
    state.scan_phase.mark_done()

    logger.info(
        f"Started {worker_group.get_active_count()} workers: "
        f"score_only={score_only}, scan_only={scan_only}, "
        f"ingest_artifacts={ingest_artifacts}"
    )

    # Periodic orphan recovery during discovery (every 60s)
    orphan_tick = make_orphan_recovery_tick(
        facility,
        [
            OrphanRecoverySpec("WikiPage"),
            OrphanRecoverySpec("WikiArtifact"),
        ],
    )

    # Run supervision loop — handles status updates and clean shutdown
    await run_supervised_loop(
        worker_group,
        state.should_stop,
        on_worker_status=on_worker_status,
        on_tick=orphan_tick,
    )
    state.stop_requested = True

    # Clean up async wiki client
    await state.close_async_wiki_client()
    await state.close_confluence_client()
    await state.close_keycloak_client()
    await state.close_basic_auth_client()

    elapsed = time.time() - start_time

    return {
        "scanned": state.scan_stats.processed,
        "scored": state.score_stats.processed,
        "ingested": state.ingest_stats.processed,
        "artifacts": state.artifact_stats.processed,
        "images_scored": state.image_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_rate": state.scan_stats.rate,
        "score_rate": state.score_stats.rate,
    }


def _get_exclude_prefixes(facility: str, current_base_url: str) -> list[str]:
    """Get URL path prefixes to exclude from crawling.

    Combines two sources of exclusions:
    1. Paths of OTHER wiki_sites on the same origin (auto-computed)
    2. Explicit exclude_prefixes from the site config (e.g., vendor docs)

    For example, jt-60sa has:
      - https://nakasvr23.iferc.org/twiki_html (twiki_static)
      - https://nakasvr23.iferc.org (static_html with exclude_prefixes)

    When crawling the root site, /twiki_html is auto-excluded, and
    /matlab, /idl, etc. are excluded via config.

    Args:
        facility: Facility ID
        current_base_url: Base URL of the site being crawled

    Returns:
        List of URL path prefixes to exclude
    """
    import urllib.parse as urlparse

    try:
        from imas_codex.discovery.base.facility import get_facility

        config = get_facility(facility)
        wiki_sites = config.get("wiki_sites", [])
    except Exception:
        return []

    current_parsed = urlparse.urlparse(current_base_url.rstrip("/"))
    current_origin = f"{current_parsed.scheme}://{current_parsed.netloc}"

    prefixes = []

    # Find the current site config to get explicit exclude_prefixes
    current_site_config = None
    for site in wiki_sites:
        site_url = site.get("url", "").rstrip("/")
        if site_url == current_base_url.rstrip("/"):
            current_site_config = site
            break

    # Add explicit exclude_prefixes from site config (vendor docs, etc.)
    if current_site_config:
        explicit_excludes = current_site_config.get("exclude_prefixes", [])
        prefixes.extend(explicit_excludes)

    # Add paths of other wiki_sites on same origin
    for site in wiki_sites:
        site_url = site.get("url", "").rstrip("/")
        if not site_url or site_url == current_base_url.rstrip("/"):
            continue

        site_parsed = urlparse.urlparse(site_url)
        site_origin = f"{site_parsed.scheme}://{site_parsed.netloc}"

        # Same origin, different path → exclude that path
        # Append trailing '/' to prevent /tfe from matching /tfe1, /tfe2 etc.
        if site_origin == current_origin and site_parsed.path:
            path = site_parsed.path.rstrip("/") + "/"
            prefixes.append(path)

    if prefixes:
        logger.info(
            "Excluding paths from crawl: %s (belong to other wiki_sites)", prefixes
        )

    return prefixes


def _get_site_max_depth(facility: str, current_base_url: str) -> int | None:
    """Get BFS max_depth from the wiki_sites config entry, if specified.

    Args:
        facility: Facility ID
        current_base_url: Base URL of the site being crawled

    Returns:
        max_depth value from config, or None to use adapter default
    """
    try:
        from imas_codex.discovery.base.facility import get_facility

        config = get_facility(facility)
        wiki_sites = config.get("wiki_sites", [])
    except Exception:
        return None

    for site in wiki_sites:
        site_url = site.get("url", "").rstrip("/")
        if site_url == current_base_url.rstrip("/"):
            return site.get("max_depth")

    return None


def _seed_portal_page(
    facility: str,
    portal_page: str,
    base_url: str,
    site_type: str,
) -> None:
    """Create the portal page as initial seed if it doesn't exist."""
    from imas_codex.discovery.wiki.scraper import canonical_page_id

    page_id = canonical_page_id(portal_page, facility)

    # Build URL based on site type
    if site_type == "twiki":
        if "/" not in portal_page:
            portal_page = f"Main/{portal_page}"
        url = f"{base_url}/bin/view/{portal_page}"
    elif site_type == "confluence":
        # portal_page is a space key (e.g. "IMP"), not a numeric page ID
        url = f"{base_url}/spaces/{portal_page}/overview"
    else:
        url = f"{base_url}/{urllib.parse.quote(portal_page, safe='/')}"

    with GraphClient() as gc:
        gc.query(
            """
            MERGE (wp:WikiPage {id: $id})
            ON CREATE SET wp.title = $title,
                          wp.url = $url,
                          wp.facility_id = $facility,
                          wp.status = $scanned,
                          wp.link_depth = 0,
                          wp.discovered_at = datetime()
            """,
            id=page_id,
            title=portal_page,
            url=url,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
        )


# =============================================================================
# Stats Query
# =============================================================================


def get_wiki_discovery_stats(facility: str) -> dict[str, int | float]:
    """Get wiki discovery statistics from graph.

    Returns stats needed for progress display:
    - total: Total wiki pages for this facility
    - scanned: Pages with status=scanned (awaiting scoring)
    - scored: Pages with status=scored (awaiting ingest or skipped)
    - ingested: Pages with status=ingested (final state)
    - pending_score: Same as scanned count (for progress display)
    - pending_ingest: Scored pages with score >= 0.5 awaiting ingestion
    - accumulated_cost: Total score_cost from all scored/ingested pages
    - skipped: Pages with status=skipped or score < 0.5
    """
    with GraphClient() as gc:
        # Get status counts
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WITH wp.status AS status
            RETURN status, count(*) AS count
            """,
            facility=facility,
        )

        stats: dict[str, int | float] = {
            "total": 0,
            "discovered": 0,
            "scanning": 0,
            "scanned": 0,
            "scoring": 0,
            "scored": 0,
            "ingesting": 0,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
        }

        for r in result:
            status = r["status"]
            count = r["count"]
            if status in stats:
                stats[status] = count
            stats["total"] += count

        # Get pending ingest count (scored pages with score >= 0.5)
        ingest_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scored AND wp.score >= 0.5
            RETURN count(wp) AS pending_ingest
            """,
            facility=facility,
            scored=WikiPageStatus.scored.value,
        )
        stats["pending_score"] = stats["scanned"]
        stats["pending_ingest"] = (
            ingest_result[0]["pending_ingest"] if ingest_result else 0
        )

        # Get accumulated cost from all pages, artifacts, and images
        cost_result = gc.query(
            """
            OPTIONAL MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.score_cost IS NOT NULL
            WITH sum(wp.score_cost) AS page_cost
            OPTIONAL MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE wa.score_cost IS NOT NULL
            WITH page_cost, sum(wa.score_cost) AS artifact_cost
            OPTIONAL MATCH (img:Image {facility_id: $facility})
            WHERE img.score_cost IS NOT NULL
            RETURN page_cost, artifact_cost, sum(img.score_cost) AS image_cost
            """,
            facility=facility,
        )
        page_cost = (
            cost_result[0]["page_cost"]
            if cost_result and cost_result[0]["page_cost"]
            else 0.0
        )
        artifact_cost = (
            cost_result[0]["artifact_cost"]
            if cost_result and cost_result[0]["artifact_cost"]
            else 0.0
        )
        image_cost = (
            cost_result[0]["image_cost"]
            if cost_result and cost_result[0]["image_cost"]
            else 0.0
        )
        stats["accumulated_cost"] = page_cost + artifact_cost + image_cost
        stats["accumulated_page_cost"] = page_cost
        stats["accumulated_artifact_cost"] = artifact_cost
        stats["accumulated_image_cost"] = image_cost

        # Add artifact stats (total + pending counts per worker)
        artifact_result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WITH wa.status AS status, wa.score AS score,
                 wa.artifact_type AS atype,
                 coalesce(wa.score_exempt, false) AS exempt
            RETURN status, score, atype, exempt, count(*) AS cnt
            """,
            facility=facility,
        )
        total_artifacts = 0
        pending_artifact_score = 0
        pending_artifact_ingest = 0
        scorable = SCORABLE_ARTIFACT_TYPES
        ingestable = INGESTABLE_ARTIFACT_TYPES
        for r in artifact_result:
            total_artifacts += r["cnt"]
            st = r["status"]
            atype = r["atype"]
            exempt = r["exempt"]
            if (
                st == WikiArtifactStatus.discovered.value
                and atype in scorable
                and not exempt
            ):
                pending_artifact_score += r["cnt"]
            elif st == WikiArtifactStatus.discovered.value and exempt:
                pending_artifact_ingest += r["cnt"]
            elif (
                st == WikiArtifactStatus.scored.value
                and atype in ingestable
                and r["score"] is not None
                and r["score"] >= 0.5
            ):
                pending_artifact_ingest += r["cnt"]

        # Count artifacts by terminal status
        artifacts_ingested = 0
        artifacts_scored = 0
        artifacts_failed = 0
        artifacts_deferred = 0
        artifacts_skipped = 0
        for r in artifact_result:
            st = r["status"]
            if st == WikiArtifactStatus.ingested.value:
                artifacts_ingested += r["cnt"]
            elif st == WikiArtifactStatus.scored.value:
                artifacts_scored += r["cnt"]
            elif st == "failed":
                artifacts_failed += r["cnt"]
            elif st == "deferred":
                artifacts_deferred += r["cnt"]
            elif st == "skipped":
                artifacts_skipped += r["cnt"]

        stats["total_artifacts"] = total_artifacts
        stats["artifacts_ingested"] = artifacts_ingested
        stats["artifacts_scored"] = artifacts_scored
        stats["artifacts_failed"] = artifacts_failed
        stats["artifacts_deferred"] = artifacts_deferred
        stats["artifacts_skipped"] = artifacts_skipped
        stats["pending_artifact_score"] = pending_artifact_score
        stats["pending_artifact_ingest"] = pending_artifact_ingest

        # Image node counts (created during page ingestion, scored by VLM)
        img_result = gc.query(
            """
            MATCH (img:Image {facility_id: $facility})
            RETURN
                count(CASE WHEN img.description IS NOT NULL THEN 1 END) AS scored,
                count(CASE WHEN img.description IS NULL
                              AND img.status IN ['ingested'] THEN 1 END) AS pending
            """,
            facility=facility,
        )
        if img_result:
            stats["images_scored"] = img_result[0]["scored"]
            stats["pending_image_score"] = img_result[0]["pending"]
        else:
            stats["images_scored"] = 0
            stats["pending_image_score"] = 0

        return stats
