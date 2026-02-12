"""Wiki discovery state management.

Shared state dataclass for parallel wiki discovery workers.
Tracks worker statistics, controls stopping conditions, and manages
shared async HTTP/SSH clients.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency at module level
# graph_ops.py imports nothing from state.py, so this is safe
def _get_graph_ops():
    from . import graph_ops

    return graph_ops


@dataclass
class WikiDiscoveryState:
    """Shared state for parallel wiki discovery."""

    facility: str
    site_type: str  # mediawiki, confluence, twiki
    base_url: str
    portal_page: str
    ssh_host: str | None = None

    # Service monitor for worker gating (set by parallel.py)
    service_monitor: Any = field(default=None, repr=False)

    # Authentication for HTTP-based access
    auth_type: str | None = None  # tequila, session, keycloak, basic, or None
    credential_service: str | None = None  # Keyring service for credentials

    # Shared async wiki client for native async HTTP (avoids re-auth per page)
    # Initialized lazily on first use, shared across all workers
    _async_wiki_client: Any = field(default=None, repr=False)
    _async_wiki_client_lock: Any = field(default=None, repr=False)

    # Shared Keycloak async client (for keycloak auth via oauth2-proxy)
    _keycloak_client: Any = field(default=None, repr=False)
    _keycloak_client_lock: Any = field(default=None, repr=False)

    # Shared HTTP Basic auth async client (for basic auth wikis like JET)
    _basic_auth_client: Any = field(default=None, repr=False)
    _basic_auth_client_lock: Any = field(default=None, repr=False)

    # Shared HTTP fetch semaphore — bounds total concurrent wiki HTTP requests
    # across ALL workers (score + ingest) to prevent httpx PoolTimeout.
    # Default 30 keeps us well under httpx pool max_connections=50.
    _http_fetch_semaphore: Any = field(default=None, repr=False)

    # SSH fetch semaphore — bounds concurrent SSH subprocess calls.
    # SSH multiplexes over ControlMaster, but too many concurrent ssh
    # processes can race during master establishment or create load.
    # Default 4 keeps footprint low on remote hosts.
    _ssh_fetch_semaphore: Any = field(default=None, repr=False)

    # Limits
    cost_limit: float = 10.0
    page_limit: int | None = None
    max_depth: int | None = None
    focus: str | None = None

    # Worker stats (simplified: score + ingest only)
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    ingest_stats: WorkerStats = field(default_factory=WorkerStats)
    artifact_stats: WorkerStats = field(default_factory=WorkerStats)
    artifact_score_stats: WorkerStats = field(default_factory=WorkerStats)
    image_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    scan_idle_count: int = 0
    score_idle_count: int = 0
    ingest_idle_count: int = 0
    artifact_idle_count: int = 0
    artifact_score_idle_count: int = 0
    image_idle_count: int = 0

    # SSH retry tracking
    ssh_retry_count: int = 0
    max_ssh_retries: int = 5
    ssh_error_message: str | None = None

    @property
    def http_fetch_semaphore(self) -> asyncio.Semaphore:
        """Shared semaphore bounding total concurrent wiki HTTP fetches.

        Lazily initialized because asyncio.Semaphore needs a running event loop.
        Shared across ALL score and ingest workers so total concurrency stays
        within the httpx connection pool capacity.
        """
        if self._http_fetch_semaphore is None:
            self._http_fetch_semaphore = asyncio.Semaphore(30)
        return self._http_fetch_semaphore

    @property
    def ssh_fetch_semaphore(self) -> asyncio.Semaphore:
        """Shared semaphore bounding concurrent SSH subprocess calls.

        SSH ControlMaster multiplexes over a single TCP connection, but
        too many concurrent ssh processes can race during establishment
        or create unnecessary load on remote hosts. Limit to 4.
        """
        if self._ssh_fetch_semaphore is None:
            self._ssh_fetch_semaphore = asyncio.Semaphore(4)
        return self._ssh_fetch_semaphore

    @property
    def effective_fetch_semaphore(self) -> asyncio.Semaphore:
        """Return SSH semaphore for SSH-based sites, HTTP semaphore otherwise."""
        if self.ssh_host and self.auth_type in (None, "none"):
            return self.ssh_fetch_semaphore
        return self.http_fetch_semaphore

    @property
    def total_cost(self) -> float:
        return (
            self.score_stats.cost
            + self.ingest_stats.cost
            + self.image_stats.cost
            + self.artifact_score_stats.cost
        )

    async def await_services(self) -> bool:
        """Block until all critical services are healthy.

        Returns True when services are ready, False if monitor was stopped.
        No-op (returns True immediately) when no service_monitor is set.
        """
        if self.service_monitor is None:
            return True
        return await self.service_monitor.await_services_ready()

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def page_limit_reached(self) -> bool:
        if self.page_limit is None:
            return False
        return self.score_stats.processed >= self.page_limit

    def should_stop(self) -> bool:
        """Check if ALL workers should terminate.

        Used by the main loop to determine when discovery is complete.

        When budget is exhausted, LLM-dependent workers (score, artifact_score,
        image) exit their loops immediately — their idle counts may stay at 0.
        We treat them as implicitly "done" so the main loop doesn't hang
        waiting for idle counts that can never increment.  I/O workers
        (ingest, artifact) continue draining their queues normally.
        """
        if self.stop_requested:
            return True

        budget_done = self.budget_exhausted

        # LLM workers: idle OR budget-stopped counts as "done"
        score_done = self.score_idle_count >= 3 or budget_done
        artifact_score_done = self.artifact_score_idle_count >= 3 or budget_done
        image_done = self.image_idle_count >= 3 or budget_done

        # I/O workers: must be genuinely idle
        all_idle = (
            self.scan_idle_count >= 3
            and score_done
            and self.ingest_idle_count >= 3
            and self.artifact_idle_count >= 3
            and artifact_score_done
            and image_done
        )
        if all_idle:
            # Check for remaining work.  When budget is exhausted only
            # check I/O queues — LLM-dependent pending work (scanned pages
            # needing scoring, images needing VLM) cannot be processed.
            graph_ops = _get_graph_ops()
            if budget_done:
                has_work = graph_ops.has_pending_ingest_work(
                    self.facility
                ) or graph_ops.has_pending_artifact_ingest_work(self.facility)
            else:
                has_work = (
                    graph_ops.has_pending_work(self.facility)
                    or graph_ops.has_pending_artifact_work(self.facility)
                    or graph_ops.has_pending_image_work(self.facility)
                )

            if has_work:
                # Reset only I/O worker idle counts when budget is exhausted
                # (LLM workers are already stopped and won't re-poll).
                # Reset all idle counts when budget is not exhausted.
                if budget_done:
                    self.ingest_idle_count = 0
                    self.artifact_idle_count = 0
                else:
                    self.scan_idle_count = 0
                    self.score_idle_count = 0
                    self.ingest_idle_count = 0
                    self.artifact_idle_count = 0
                    self.artifact_score_idle_count = 0
                    self.image_idle_count = 0
                return False
            return True
        return False

    def should_stop_scanning(self) -> bool:
        """Check if scan workers should stop.

        Scan workers continue even when budget is exhausted. They only stop
        when explicitly requested or when no pending work remains.
        """
        if self.stop_requested:
            return True
        # Only stop scanning when idle with no work
        if self.scan_idle_count >= 3 and not _get_graph_ops().has_pending_scan_work(
            self.facility
        ):
            return True
        return False

    def should_stop_scoring(self) -> bool:
        """Check if score workers should stop.

        Score workers stop when budget exhausted or page limit reached.
        """
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.page_limit_reached:
            return True
        return False

    def should_stop_ingesting(self) -> bool:
        """Check if ingest workers should stop.

        Ingest workers continue AFTER budget is exhausted to drain the
        ingest queue. This ensures all scorable content gets ingested
        before termination. They only stop when:
        1. Explicitly requested
        2. Idle for 3+ iterations with no pending ingest work AND
           score workers are also idle (no more scoring happening)
        """
        if self.stop_requested:
            return True
        # Continue even when budget exhausted - drain the ingest queue
        # BUT don't exit early if scoring is still running - pages may arrive soon
        if self.ingest_idle_count >= 3:
            # Only stop if scoring is also idle AND no pending ingest work
            scoring_done = self.score_idle_count >= 3 or self.budget_exhausted
            if scoring_done and not _get_graph_ops().has_pending_ingest_work(
                self.facility
            ):
                return True
        return False

    def should_stop_artifact_worker(self) -> bool:
        """Check if artifact ingest workers should stop.

        Artifact ingest workers continue until no pending scored artifacts remain.
        They stop when explicitly requested or idle for 3+ iterations.
        """
        if self.stop_requested:
            return True
        if (
            self.artifact_idle_count >= 3
            and not _get_graph_ops().has_pending_artifact_ingest_work(self.facility)
        ):
            return True
        return False

    def should_stop_artifact_scoring(self) -> bool:
        """Check if artifact score workers should stop.

        Artifact score workers stop when budget exhausted or no more discovered artifacts.
        """
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if (
            self.artifact_score_idle_count >= 3
            and not _get_graph_ops().has_pending_artifact_score_work(self.facility)
        ):
            return True
        return False

    def should_stop_image_scoring(self) -> bool:
        """Check if image score workers should stop.

        Image score workers respect the cost budget (VLM calls are expensive).
        They also wait for ingestion to start producing images before giving up,
        since images are only created during page ingestion and may not exist
        at the start of a run.

        Stop when:
        1. Explicitly requested
        2. Budget exhausted (VLM must respect budget like LLM workers)
        3. Idle for 3+ iterations AND ingestion is done AND no pending images
        """
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        # Wait for ingestion to finish before declaring no work.
        # Images only appear after pages are ingested, so the worker may
        # see an empty queue early in the run.
        ingestion_done = self.ingest_idle_count >= 3
        if (
            self.image_idle_count >= 3
            and ingestion_done
            and not _get_graph_ops().has_pending_image_work(self.facility)
        ):
            return True
        return False

    async def get_async_wiki_client(self):
        """Get shared AsyncMediaWikiClient for Tequila auth.

        Lazily initializes an async client that persists session cookies
        across all page fetches. Uses native async HTTP for better performance.
        """
        if self._async_wiki_client is None:
            from imas_codex.discovery.wiki.mediawiki import AsyncMediaWikiClient

            if self._async_wiki_client_lock is None:
                self._async_wiki_client_lock = asyncio.Lock()

            async with self._async_wiki_client_lock:
                if self._async_wiki_client is None:
                    self._async_wiki_client = AsyncMediaWikiClient(
                        base_url=self.base_url,
                        credential_service=self.credential_service
                        or f"{self.facility}-wiki",
                        verify_ssl=False,
                    )
                    # Pre-authenticate to warm up session
                    try:
                        await self._async_wiki_client.authenticate()
                        logger.info(
                            "Initialized shared AsyncMediaWikiClient with Tequila auth"
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to pre-authenticate AsyncMediaWikiClient: %s", e
                        )

        return self._async_wiki_client

    async def close_async_wiki_client(self):
        """Close the shared async wiki client."""
        if self._async_wiki_client is not None:
            try:
                await self._async_wiki_client.close()
            except Exception:
                pass
            self._async_wiki_client = None

    async def get_keycloak_client(self) -> Any:
        """Get shared AsyncKeycloakSession for Keycloak OIDC auth.

        Lazily initializes a Keycloak OIDC session that persists cookies
        across all page fetches. One login authenticates across all wiki
        sites on the same domain.

        Returns:
            Authenticated httpx.AsyncClient or None if auth fails.
        """
        if self._keycloak_client is None:
            from imas_codex.discovery.wiki.keycloak import AsyncKeycloakSession

            if self._keycloak_client_lock is None:
                self._keycloak_client_lock = asyncio.Lock()

            async with self._keycloak_client_lock:
                if self._keycloak_client is None:
                    aks = AsyncKeycloakSession(self.credential_service or self.facility)
                    try:
                        self._keycloak_client = await aks.login(f"{self.base_url}/")
                        logger.info(
                            "Initialized shared Keycloak session for %s",
                            self.credential_service,
                        )
                    except Exception as e:
                        logger.warning("Keycloak auth failed: %s", e)

        return self._keycloak_client

    async def close_keycloak_client(self):
        """Close the shared Keycloak client."""
        if self._keycloak_client is not None:
            try:
                await self._keycloak_client.aclose()
            except Exception:
                pass
            self._keycloak_client = None

    async def get_basic_auth_client(self) -> Any:
        """Get shared httpx.AsyncClient with HTTP Basic auth.

        For wikis that use standard HTTP Basic authentication (RFC 7617),
        e.g. JET wiki sites at wiki.jetdata.eu.

        Returns:
            httpx.AsyncClient with BasicAuth or None if credentials unavailable.
        """
        if self._basic_auth_client is None:
            import httpx

            from imas_codex.discovery.wiki.auth import CredentialManager

            if self._basic_auth_client_lock is None:
                self._basic_auth_client_lock = asyncio.Lock()

            async with self._basic_auth_client_lock:
                if self._basic_auth_client is None:
                    cred_mgr = CredentialManager()
                    service = self.credential_service or self.facility
                    creds = cred_mgr.get_credentials(service, prompt_if_missing=False)
                    if not creds:
                        logger.warning(
                            "No credentials for %s - cannot use HTTP Basic auth",
                            service,
                        )
                        return None
                    username, password = creds
                    self._basic_auth_client = httpx.AsyncClient(
                        auth=httpx.BasicAuth(username, password),
                        timeout=httpx.Timeout(60.0),
                        follow_redirects=True,
                        verify=False,
                        headers={
                            "User-Agent": "imas-codex/1.0 (IMAS Data Mapping Tool)",
                            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                        },
                        limits=httpx.Limits(
                            max_connections=20,
                            max_keepalive_connections=10,
                        ),
                    )
                    logger.info(
                        "Initialized shared HTTP Basic auth client for %s",
                        service,
                    )

        return self._basic_auth_client

    async def close_basic_auth_client(self):
        """Close the shared HTTP Basic auth client."""
        if self._basic_auth_client is not None:
            try:
                await self._basic_auth_client.aclose()
            except Exception:
                pass
            self._basic_auth_client = None
