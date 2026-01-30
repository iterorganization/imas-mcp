"""Integrated wiki discovery pipeline.

Complete workflow:
1. CRAWL - Fast link extraction, builds wiki graph structure
2. PREFETCH - Fetch page content and generate summaries
3. SCORE - Content-aware LLM evaluation, assigns interest scores
4. INGEST - Fetch and chunk high-score pages for search

This module is facility-agnostic - wiki configuration comes from facility YAML.

Example:
    from imas_codex.wiki.discovery import WikiDiscovery

    # Run complete pipeline
    discovery = WikiDiscovery("tcv", cost_limit_usd=10.0)
    await discovery.run()

    # Or run individual steps
    discovery.crawl()
    await discovery.prefetch()
    await discovery.score()
    await discovery.ingest()
"""

import logging
import subprocess
import time
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Suppress pydantic deprecation warnings from llama-index internals
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings(
    "ignore", message=".*model_fields.*", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message=".*model_computed_fields.*", category=DeprecationWarning
)
warnings.filterwarnings("ignore", message=".*__fields__.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*__fields_set__.*", category=DeprecationWarning
)
warnings.filterwarnings("ignore", message=".*__fields__.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*__fields_set__.*", category=DeprecationWarning
)

from rich.console import Console  # noqa: E402
from smolagents import CodeAgent, Tool  # noqa: E402

from imas_codex.agentic.agents import (  # noqa: E402
    create_litellm_model,
    get_model_for_task,
)
from imas_codex.agentic.prompt_loader import load_prompts  # noqa: E402
from imas_codex.graph import GraphClient  # noqa: E402
from imas_codex.wiki.progress import (  # noqa: E402
    CrawlProgressMonitor,
    ScoreProgressMonitor,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DiscoveryStats:
    """Statistics for discovery progress."""

    # Phase 1: Crawl
    pages_crawled: int = 0
    artifacts_found: int = 0
    links_found: int = 0
    max_depth_reached: int = 0
    frontier_size: int = 0

    # Phase 2: Score
    pages_scored: int = 0
    artifacts_scored: int = 0
    high_score_count: int = 0  # interest_score >= 0.7 (pages + artifacts)
    low_score_count: int = 0  # interest_score < 0.3 (pages + artifacts)
    page_high_score_count: int = 0  # Pages only
    page_low_score_count: int = 0  # Pages only
    artifact_high_score_count: int = 0  # Artifacts only
    artifact_low_score_count: int = 0  # Artifacts only

    # Phase 3: Ingest
    pages_ingested: int = 0
    chunks_created: int = 0

    # Cost tracking
    cost_spent_usd: float = 0.0
    cost_limit_usd: float = 10.0

    # Timing
    start_time: float = field(default_factory=time.time)
    phase: str = "idle"

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def elapsed_formatted(self) -> str:
        """Format elapsed time as human-readable string."""
        seconds = int(self.elapsed_seconds())
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, mins = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {mins}m {secs}s"
        days, hrs = divmod(hours, 24)
        return f"{days}d {hrs}h {mins}m {secs}s"

    def cost_remaining(self) -> float:
        return max(0, self.cost_limit_usd - self.cost_spent_usd)

    def budget_exhausted(self) -> bool:
        return self.cost_spent_usd >= self.cost_limit_usd


@dataclass
class WikiConfig:
    """Wiki configuration for a facility site.

    Supports multiple site types:
    - mediawiki: MediaWiki sites (SSH proxy or direct)
    - confluence: Atlassian Confluence (REST API)
    - generic: Generic HTML scraping
    """

    base_url: str
    portal_page: str
    facility_id: str
    site_type: str = "mediawiki"  # mediawiki, confluence, generic
    auth_type: str = "ssh_proxy"  # none, ssh_proxy, basic, session
    ssh_host: str | None = None  # for ssh_proxy auth
    credential_service: str | None = None  # keyring service name

    @classmethod
    def from_facility(cls, facility: str, site_index: int = 0) -> "WikiConfig":
        """Load wiki config from facility configuration.

        Args:
            facility: Facility identifier
            site_index: Index of wiki site in facility's wiki_sites list

        Returns:
            WikiConfig for the specified site
        """
        from imas_codex.discovery.facility import get_facility

        # Try to load from facility YAML first
        try:
            config = get_facility(facility)
            wiki_sites = config.get("wiki_sites", [])

            if wiki_sites and site_index < len(wiki_sites):
                site = wiki_sites[site_index]
                return cls(
                    base_url=site["url"],
                    portal_page=site.get("portal_page", ""),
                    facility_id=facility,
                    site_type=site.get("site_type", "mediawiki"),
                    auth_type=site.get("auth_type", "none"),
                    ssh_host=site.get("ssh_host") or config.get("ssh_host"),
                    credential_service=site.get("credential_service"),
                )
        except Exception:
            pass  # Fall back to hardcoded defaults

        # Default configurations per facility (legacy support)
        configs = {
            "tcv": {
                "base_url": "https://spcwiki.epfl.ch/wiki",
                "portal_page": "Portal:TCV",
                "site_type": "mediawiki",
                "auth_type": "ssh_proxy",
                "ssh_host": "tcv",
            },
            "iter": {
                "base_url": "https://confluence.iter.org",
                "portal_page": "IMP",
                "site_type": "confluence",
                "auth_type": "session",
                "credential_service": "iter-confluence",
            },
        }

        if facility not in configs:
            raise ValueError(
                f"Unknown facility: {facility}. Known: {list(configs.keys())}"
            )

        cfg = configs[facility]
        return cls(
            base_url=cfg["base_url"],
            portal_page=cfg["portal_page"],
            facility_id=facility,
            site_type=cfg.get("site_type", "mediawiki"),
            auth_type=cfg.get("auth_type", "none"),
            ssh_host=cfg.get("ssh_host"),
            credential_service=cfg.get("credential_service"),
        )

    @classmethod
    def list_sites(cls, facility: str) -> list["WikiConfig"]:
        """List all wiki sites configured for a facility.

        Args:
            facility: Facility identifier

        Returns:
            List of WikiConfig for all configured sites
        """
        from imas_codex.discovery.facility import get_facility

        sites = []

        try:
            config = get_facility(facility)
            wiki_sites = config.get("wiki_sites", [])

            if wiki_sites:
                for i, _site in enumerate(wiki_sites):
                    sites.append(cls.from_facility(facility, site_index=i))
                return sites
        except Exception:
            pass

        # Fall back to single default site from hardcoded configs
        try:
            sites.append(cls.from_facility(facility))
        except ValueError:
            pass

        return sites

    @property
    def requires_auth(self) -> bool:
        """Check if this site requires authentication."""
        return self.auth_type in ("basic", "session")

    @property
    def requires_ssh(self) -> bool:
        """Check if this site requires SSH proxy to access.

        Confluence sites use REST API directly (no SSH needed).
        MediaWiki sites may require SSH proxy if behind firewall.
        """
        return self.auth_type == "ssh_proxy" and self.ssh_host is not None


class WikiDiscovery:
    """Integrated wiki discovery pipeline.

    Provides both unified workflow (run()) and individual steps
    (crawl, prefetch, score, ingest) for flexible execution.
    """

    # File extensions that indicate artifacts (not wiki pages)
    ARTIFACT_EXTENSIONS = {
        # Documents
        ".pdf": "pdf",
        ".doc": "document",
        ".docx": "document",
        ".odt": "document",
        ".rtf": "document",
        # Presentations
        ".ppt": "presentation",
        ".pptx": "presentation",
        ".key": "presentation",
        # Spreadsheets
        ".xls": "spreadsheet",
        ".xlsx": "spreadsheet",
        ".csv": "spreadsheet",
        # Images
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "image",
        ".svg": "image",
        ".bmp": "image",
        # Notebooks
        ".nb": "notebook",
        ".ipynb": "notebook",
        # Data files
        ".mat": "data",
        ".hdf5": "data",
        ".h5": "data",
        ".nc": "data",
        ".npy": "data",
        # Archives
        ".zip": "archive",
        ".tar": "archive",
        ".gz": "archive",
        ".tgz": "archive",
        # Text
        ".txt": "document",
    }

    def __init__(
        self,
        facility: str,
        cost_limit_usd: float = 10.0,
        max_pages: int | None = None,
        max_depth: int | None = None,
        verbose: bool = False,
        model: str | None = None,
        focus: str | None = None,
    ):
        self.config = WikiConfig.from_facility(facility)
        self.stats = DiscoveryStats(cost_limit_usd=cost_limit_usd)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.verbose = verbose
        self.focus = focus
        # Model override - if None, get_model_for_task("discovery") is used
        self._model = model

        # Graph client for persistence
        self._gc: GraphClient | None = None

        # Cache for Confluence portal page ID resolution
        self._portal_page_id: str | None = None

        # Cache for Confluence page titles (page_id -> title)
        self._confluence_page_titles: dict[str, str] = {}

    @property
    def resolved_model(self) -> str:
        """Get the model to use for LLM operations."""
        if self._model:
            return self._model
        return get_model_for_task("discovery")

    def _get_gc(self) -> GraphClient:
        if self._gc is None:
            self._gc = GraphClient()
        return self._gc

    def _resolve_confluence_portal_page(self) -> str:
        """Resolve Confluence portal page name to page ID.

        For Confluence sites, the portal_page is typically a space key (e.g., 'IMP').
        We need to get the space homepage ID.

        Returns:
            Page ID of the portal/homepage
        """
        if self._portal_page_id is not None:
            return self._portal_page_id

        from imas_codex.wiki.confluence import ConfluenceClient

        try:
            client = ConfluenceClient(
                self.config.base_url,
                self.config.credential_service or "confluence",
            )

            if not client.authenticate():
                logger.error("Failed to authenticate with Confluence")
                # Fallback to using portal_page as-is
                self._portal_page_id = self.config.portal_page
                return self._portal_page_id

            # Get space homepage
            homepage = client.get_space_homepage(self.config.portal_page)
            if homepage:
                self._portal_page_id = homepage.id
                # Cache the title immediately
                self._confluence_page_titles[homepage.id] = homepage.title
                logger.info(
                    "Resolved portal page '%s' to ID: %s (%s)",
                    self.config.portal_page,
                    homepage.id,
                    homepage.title,
                )
            else:
                logger.warning(
                    "Could not resolve portal page '%s', using as-is",
                    self.config.portal_page,
                )
                self._portal_page_id = self.config.portal_page

            client.close()

        except Exception as e:
            logger.error("Error resolving portal page: %s", e)
            self._portal_page_id = self.config.portal_page

        return self._portal_page_id

    def _get_confluence_page_title(self, page_id: str) -> str:
        """Get the title for a Confluence page ID.

        Uses cached titles from previous API calls to avoid extra requests.
        Falls back to page ID if title not available.

        Args:
            page_id: Confluence page ID

        Returns:
            Page title or page ID if not found
        """
        if page_id in self._confluence_page_titles:
            return self._confluence_page_titles[page_id]

        # Title not in cache, return page ID as fallback
        # The title will be populated when the page is crawled
        return page_id

    def _classify_link(self, link: str) -> tuple[str, str | None]:
        """Classify a link as 'page' or 'artifact'.

        Returns:
            (link_type, artifact_type) where link_type is 'page' or 'artifact',
            and artifact_type is the ArtifactType enum value (or None for pages).
        """
        link_lower = link.lower()
        for ext, artifact_type in self.ARTIFACT_EXTENSIONS.items():
            if link_lower.endswith(ext):
                return ("artifact", artifact_type)
        return ("page", None)

    def _extract_filename(self, url: str) -> str:
        """Extract filename from a URL path."""
        decoded = urllib.parse.unquote(url)
        # Handle wiki/images/a/ab/filename.pdf format
        parts = decoded.split("/")
        return parts[-1] if parts else decoded

    # =========================================================================
    # Phase 1: CRAWL - Fast link extraction
    # =========================================================================

    def _extract_links_from_page(
        self, page_name: str
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Extract all internal wiki links from a page.

        Delegates to site-specific implementation based on site_type.

        Returns:
            (page_links, artifact_links) where:
            - page_links: list of page names to crawl
            - artifact_links: list of (url_path, artifact_type) tuples
        """
        if self.config.site_type == "confluence":
            return self._extract_links_from_confluence_page(page_name)
        else:
            return self._extract_links_from_mediawiki_page(page_name)

    def _extract_links_from_confluence_page(
        self, page_id: str
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Extract links from a Confluence page via REST API.

        Args:
            page_id: Confluence page ID (not title)

        Returns:
            (pCache the page title for later use
            self._confluence_page_titles[page_id] = page.title

            # age_links, artifact_links)
        """
        from imas_codex.wiki.confluence import ConfluenceClient

        page_links: list[str] = []
        artifact_links: list[tuple[str, str]] = []

        try:
            client = ConfluenceClient(
                self.config.base_url,
                self.config.credential_service or "confluence",
            )

            # Get page content with children and attachments expanded
            page = client.get_page_content(page_id)
            if not page:
                client.close()
                return [], []

            # Cache the page title for later use
            self._confluence_page_titles[page_id] = page.title

            # Extract child pages - use dedicated API to get ALL children (paginated)
            # The children.page expansion only returns first page of results (default 25)
            all_children = client.get_page_children(page_id)
            for child_id in all_children:
                page_links.append(child_id)
                # Fetch child title if not already cached (use lightweight API call)
                if child_id not in self._confluence_page_titles:
                    try:
                        basic_info = client.get_page_basic_info(child_id)
                        if basic_info:
                            self._confluence_page_titles[child_id] = basic_info[
                                0
                            ]  # title
                    except Exception as e:
                        logger.debug(
                            "Could not fetch title for child %s: %s", child_id, e
                        )

            # Extract attachments and classify by media type
            for att in page.attachments:
                media_type = att.get("mediaType", "")
                # Map media type to artifact type
                if "pdf" in media_type:
                    artifact_type = "pdf"
                elif "image" in media_type:
                    artifact_type = "image"
                elif "presentation" in media_type or "powerpoint" in media_type:
                    artifact_type = "presentation"
                elif "spreadsheet" in media_type or "excel" in media_type:
                    artifact_type = "spreadsheet"
                elif "document" in media_type or "word" in media_type:
                    artifact_type = "document"
                else:
                    # Fallback: use filename extension
                    filename = att.get("title", "").lower()
                    artifact_type = "document"  # default
                    for ext, atype in self.ARTIFACT_EXTENSIONS.items():
                        if filename.endswith(ext):
                            artifact_type = atype
                            break

                artifact_links.append((att["downloadUrl"], artifact_type))

            # Extract links from HTML content to find cross-references
            import re

            for match in re.finditer(r'href="([^"]+)"', page.content_html):
                url = match.group(1)
                if "/pages/viewpage.action?pageId=" in url:
                    # Extract page ID from viewpage URL
                    page_id_match = re.search(r"pageId=(\d+)", url)
                    if page_id_match:
                        linked_id = page_id_match.group(1)
                        if linked_id not in page_links:
                            page_links.append(linked_id)
                elif "/display/" in url:
                    # Extract page ID from display URL format: /display/SPACE/Page+Title
                    # We'll need to resolve this later, skip for now
                    pass

            client.close()

        except Exception as e:
            logger.warning("Error extracting Confluence links from %s: %s", page_id, e)

        return page_links, artifact_links

    def _extract_links_from_mediawiki_page(
        self, page_name: str
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Extract all internal wiki links from a MediaWiki page via SSH.

        Returns:
            (page_links, artifact_links) where:
            - page_links: list of page names to crawl
            - artifact_links: list of (url_path, artifact_type) tuples
        """
        if not self.config.ssh_host:
            logger.error("SSH host not configured for MediaWiki site")
            return [], []

        encoded = urllib.parse.quote(page_name, safe="")
        url = f"{self.config.base_url}/{encoded}"

        # Extract ALL hrefs that point to internal wiki paths (including images/)
        # Allow colons in path for images/a/ab/filename.pdf pattern
        cmd = f'''curl -sk "{url}" | grep -oP 'href="/wiki/[^"#]+' | sed 's|href="/wiki/||' | sort -u'''

        try:
            result = subprocess.run(
                ["ssh", self.config.ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return [], []

            page_links: list[str] = []
            artifact_links: list[tuple[str, str]] = []

            excluded_prefixes = (
                "Special:",
                "File:",
                "Talk:",
                "User_talk:",
                "Template:",
                "Category:",
                "Help:",
                "MediaWiki:",
                "index.php",
                "skins/",
                "opensearch",
            )
            # Only exclude code/style files, not document artifacts
            excluded_extensions = (".css", ".js", ".php")

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # Skip excluded prefixes
                if line.startswith(excluded_prefixes):
                    continue
                # Skip code/style files
                if any(line.lower().endswith(ext) for ext in excluded_extensions):
                    continue
                # Skip query strings
                if "?" in line or "&" in line:
                    continue

                # Decode URL encoding
                decoded = urllib.parse.unquote(line)

                # Classify as page or artifact
                link_type, artifact_type = self._classify_link(decoded)

                if link_type == "artifact" and artifact_type:
                    artifact_links.append((decoded, artifact_type))
                else:
                    page_links.append(decoded)

            return page_links, artifact_links

        except subprocess.TimeoutExpired:
            logger.warning("Timeout extracting links from %s", page_name)
            return [], []

    def _crawl_batch(
        self, pages: list[str]
    ) -> dict[str, tuple[list[str], list[tuple[str, str]]]]:
        """Crawl multiple pages in parallel.

        Returns:
            Dict mapping page -> (page_links, artifact_links)
        """
        results: dict[str, tuple[list[str], list[tuple[str, str]]]] = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._extract_links_from_page, page): page
                for page in pages
            }

            for future in as_completed(futures):
                page = futures[future]
                try:
                    page_links, artifact_links = future.result()
                    results[page] = (page_links, artifact_links)
                except Exception as e:
                    logger.warning("Error crawling %s: %s", page, e)
                    results[page] = ([], [])

        return results

    def crawl(self, monitor: CrawlProgressMonitor | None = None) -> int:
        """Crawl wiki and build link structure.

        Graph-driven: resumes from existing state. Already-crawled pages
        are loaded from the graph, and pending pages form the frontier.

        Creates WikiPage nodes for pages and WikiArtifact nodes for linked
        documents (PDFs, presentations, etc.).

        Args:
            monitor: Optional CrawlProgressMonitor for Rich display.

        Returns:
            Number of pages crawled in this session.
        """
        self.stats.phase = "CRAWL"
        gc = self._get_gc()

        # Load existing state from graph
        visited, frontier, depth_map = self._load_crawl_state(gc)
        known_artifacts = self._load_known_artifacts(gc)

        # If no frontier and no visited, start from portal
        if not frontier and not visited:
            # For Confluence, resolve portal page name to ID
            if self.config.site_type == "confluence":
                portal = self._resolve_confluence_portal_page()
            else:
                portal = self.config.portal_page
            frontier = {portal}
            depth_map = {portal: 0}

        # Track all page->page links for relationship creation
        all_page_links: dict[str, list[str]] = {}
        # Track all page->artifact links for relationship creation
        all_artifact_links: dict[str, list[str]] = {}
        session_crawled = 0

        # Update stats with loaded state
        self.stats.pages_crawled = len(visited)
        self.stats.frontier_size = len(frontier)
        if depth_map:
            self.stats.max_depth_reached = max(depth_map.values())

        # Crawl until frontier is empty or max_pages reached
        while frontier:
            if self.max_pages is not None and session_crawled >= self.max_pages:
                break

            # Get next batch from frontier
            batch = list(frontier)[:50]
            frontier -= set(batch)

            # Crawl batch - returns {page: (page_links, artifact_links)}
            results = self._crawl_batch(batch)

            for page, (page_links, artifact_links) in results.items():
                if page in visited:
                    if monitor:
                        monitor.update(page=page, skipped=True)
                    continue

                visited.add(page)
                current_depth = depth_map.get(page, 0)
                self.stats.max_depth_reached = max(
                    self.stats.max_depth_reached, current_depth
                )

                total_out_degree = len(page_links) + len(artifact_links)

                # Create WikiPage node with canonical ID
                from imas_codex.wiki.scraper import canonical_page_id

                page_id = canonical_page_id(page, self.config.facility_id)

                # For Confluence, fetch the actual page title
                # For MediaWiki, the page name is the title
                if self.config.site_type == "confluence":
                    page_title = self._get_confluence_page_title(page)
                    page_url = (
                        f"{self.config.base_url}/pages/viewpage.action?pageId={page}"
                    )
                else:
                    page_title = page
                    page_url = (
                        f"{self.config.base_url}/{urllib.parse.quote(page, safe='/')}"
                    )

                gc.query(
                    """
                    MERGE (wp:WikiPage {id: $id})
                    SET wp.title = $title,
                        wp.url = $url,
                        wp.status = 'discovered',
                        wp.facility_id = $facility_id,
                        wp.link_depth = $depth,
                        wp.out_degree = $out_degree,
                        wp.discovered_at = datetime()
                    WITH wp
                    MATCH (f:Facility {id: $facility_id})
                    MERGE (wp)-[:FACILITY_ID]->(f)
                    """,
                    id=page_id,
                    title=page_title,
                    url=page_url,
                    facility_id=self.config.facility_id,
                    depth=current_depth,
                    out_degree=total_out_degree,
                )

                # Track page links for later relationship creation
                all_page_links[page] = page_links

                # Add new page links to frontier
                for link in page_links:
                    if link not in visited and link not in frontier:
                        # Check max_depth only if it's set
                        if (
                            self.max_depth is None
                            or current_depth + 1 <= self.max_depth
                        ):
                            frontier.add(link)
                            depth_map[link] = current_depth + 1
                            self._persist_pending_page(gc, link, current_depth + 1)

                # Process artifact links - create WikiArtifact nodes
                artifact_ids_for_page = []
                for artifact_path, artifact_type in artifact_links:
                    artifact_id = self._persist_artifact(
                        gc, artifact_path, artifact_type, current_depth, known_artifacts
                    )
                    if artifact_id:
                        artifact_ids_for_page.append(artifact_id)

                if artifact_ids_for_page:
                    all_artifact_links[page] = artifact_ids_for_page

                session_crawled += 1
                self.stats.pages_crawled += 1
                self.stats.links_found += total_out_degree

                if monitor:
                    monitor.update(
                        page=page_title,  # Use title instead of ID for display
                        links_found=len(page_links),
                        artifacts_found=len(artifact_links),
                        frontier_size=len(frontier),
                        depth=current_depth,
                    )

            self.stats.frontier_size = len(frontier)

        # Create LINKS_TO relationships between pages
        self._create_page_link_relationships(all_page_links, gc)

        # Create LINKS_TO_ARTIFACT relationships
        self._create_artifact_link_relationships(all_artifact_links, gc)

        # Compute in_degree for all pages
        gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            OPTIONAL MATCH (wp)<-[:LINKS_TO]-(source)
            WITH wp, count(source) AS in_deg
            SET wp.in_degree = in_deg
            """,
            facility_id=self.config.facility_id,
        )

        # Compute in_degree for all artifacts
        gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility_id})
            OPTIONAL MATCH (wa)<-[:LINKS_TO_ARTIFACT]-(source)
            WITH wa, count(source) AS in_deg
            SET wa.in_degree = in_deg
            """,
            facility_id=self.config.facility_id,
        )

        return session_crawled

    def _load_known_artifacts(self, gc: GraphClient) -> set[str]:
        """Load known artifact IDs from graph."""
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility_id})
            RETURN wa.id AS id
            """,
            facility_id=self.config.facility_id,
        )
        return {r["id"] for r in result}

    def _persist_artifact(
        self,
        gc: GraphClient,
        artifact_path: str,
        artifact_type: str,
        linking_depth: int,
        known_artifacts: set[str],
    ) -> str | None:
        """Persist a WikiArtifact node.

        Returns:
            artifact_id if created/updated, None if error
        """
        filename = self._extract_filename(artifact_path)
        artifact_id = f"{self.config.facility_id}:{filename}"

        # For Confluence, artifact_path is already a full URL from the API
        # For MediaWiki, it's a relative path that needs the base URL
        if artifact_path.startswith("http"):
            url = artifact_path  # Already a full URL
        else:
            url = f"{self.config.base_url}/{urllib.parse.quote(artifact_path, safe='')}"

        is_new = artifact_id not in known_artifacts

        gc.query(
            """
            MERGE (wa:WikiArtifact {id: $id})
            ON CREATE SET wa.facility_id = $facility_id,
                          wa.url = $url,
                          wa.filename = $filename,
                          wa.artifact_type = $artifact_type,
                          wa.status = 'discovered',
                          wa.link_depth = $link_depth,
                          wa.discovered_at = datetime()
            ON MATCH SET wa.link_depth = CASE
                WHEN wa.link_depth IS NULL OR wa.link_depth > $link_depth
                THEN $link_depth ELSE wa.link_depth END
            WITH wa
            MATCH (f:Facility {id: $facility_id})
            MERGE (wa)-[:FACILITY_ID]->(f)
            """,
            id=artifact_id,
            facility_id=self.config.facility_id,
            url=url,
            filename=filename,
            artifact_type=artifact_type,
            link_depth=linking_depth,
        )

        if is_new:
            known_artifacts.add(artifact_id)
            self.stats.artifacts_found += 1

        return artifact_id

    def _load_crawl_state(
        self, gc: GraphClient
    ) -> tuple[set[str], set[str], dict[str, int]]:
        """Load crawl state from graph.

        Returns:
            (visited, frontier, depth_map)
        """
        # Get already-crawled pages
        crawled_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'discovered'})
            RETURN wp.title AS title, wp.link_depth AS depth
            """,
            facility_id=self.config.facility_id,
        )
        visited = {r["title"] for r in crawled_result}
        depth_map = {r["title"]: r["depth"] or 0 for r in crawled_result}

        # Get pending pages (discovered but not crawled)
        pending_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'pending_crawl'})
            RETURN wp.title AS title, wp.link_depth AS depth
            """,
            facility_id=self.config.facility_id,
        )
        frontier = {r["title"] for r in pending_result}
        for r in pending_result:
            depth_map[r["title"]] = r["depth"] or 0

        return visited, frontier, depth_map

    def _persist_pending_page(self, gc: GraphClient, page: str, depth: int) -> None:
        """Persist a pending page to the graph."""
        from imas_codex.wiki.scraper import canonical_page_id

        page_id = canonical_page_id(page, self.config.facility_id)
        gc.query(
            """
            MERGE (wp:WikiPage {id: $id})
            ON CREATE SET wp.title = $title,
                          wp.status = 'pending_crawl',
                          wp.facility_id = $facility_id,
                          wp.link_depth = $depth,
                          wp.discovered_at = datetime()
            """,
            id=page_id,
            title=page,
            facility_id=self.config.facility_id,
            depth=depth,
        )

    def _create_page_link_relationships(
        self, results: dict[str, list[str]], gc: GraphClient
    ) -> None:
        """Create LINKS_TO relationships between WikiPages in bulk."""
        from imas_codex.wiki.scraper import canonical_page_id

        for source_page, target_pages in results.items():
            if not target_pages:
                continue

            source_id = canonical_page_id(source_page, self.config.facility_id)
            target_ids = [
                canonical_page_id(t, self.config.facility_id) for t in target_pages
            ]

            gc.query(
                """
                MATCH (source:WikiPage {id: $source_id})
                UNWIND $target_ids AS target_id
                MATCH (target:WikiPage {id: target_id})
                MERGE (source)-[:LINKS_TO]->(target)
                """,
                source_id=source_id,
                target_ids=target_ids,
            )

    def _create_artifact_link_relationships(
        self, results: dict[str, list[str]], gc: GraphClient
    ) -> None:
        """Create LINKS_TO_ARTIFACT relationships from WikiPages to WikiArtifacts."""
        for source_page, artifact_ids in results.items():
            if not artifact_ids:
                continue

            source_id = f"{self.config.facility_id}:{source_page}"

            gc.query(
                """
                MATCH (source:WikiPage {id: $source_id})
                UNWIND $artifact_ids AS artifact_id
                MATCH (artifact:WikiArtifact {id: artifact_id})
                MERGE (source)-[:LINKS_TO_ARTIFACT]->(artifact)
                """,
                source_id=source_id,
                artifact_ids=artifact_ids,
            )

    # =========================================================================
    # Phase 2: SCORE - Agent evaluates graph metrics
    # =========================================================================

    def _get_scoring_tools(self) -> list[Tool]:
        """Get tools for the scoring agent."""
        gc = self._get_gc()
        facility_id = self.config.facility_id
        stats = self.stats

        class GetPagesToScoreTool(Tool):
            """Get crawled pages that need scoring."""

            name = "get_pages_to_score"
            description = "Get crawled pages needing scores. Returns id, title, depth, in_degree, out_degree."
            inputs = {
                "limit": {
                    "type": "integer",
                    "description": "Maximum pages to return (default 100)",
                    "nullable": True,
                },
            }
            output_type = "string"

            def forward(self, limit: int = 100) -> str:
                import json

                result = gc.query(
                    """
                    MATCH (wp:WikiPage {facility_id: $facility_id, status: 'discovered'})
                    RETURN wp.id AS id,
                           wp.title AS title,
                           wp.link_depth AS depth,
                           wp.in_degree AS in_degree,
                           wp.out_degree AS out_degree,
                           wp.preview_summary AS preview_summary,
                           wp.preview_fetch_error AS preview_fetch_error
                    ORDER BY wp.in_degree DESC
                    LIMIT $limit
                    """,
                    facility_id=facility_id,
                    limit=limit,
                )

                return json.dumps({"pages": result, "count": len(result)})

        class GetNeighborInfoTool(Tool):
            """Get info about pages that link to/from this page."""

            name = "get_neighbor_info"
            description = "Get pages that link to/from a specific page. Use to assess value from context."
            inputs = {
                "page_id": {
                    "type": "string",
                    "description": "Page ID to get neighbors for",
                },
            }
            output_type = "string"

            def forward(self, page_id: str) -> str:
                import json

                result = gc.query(
                    """
                    MATCH (wp:WikiPage {id: $page_id})
                    OPTIONAL MATCH (wp)-[:LINKS_TO]->(outgoing)
                    OPTIONAL MATCH (incoming)-[:LINKS_TO]->(wp)
                    WITH wp,
                         collect(DISTINCT {id: outgoing.id, title: outgoing.title, score: outgoing.interest_score}) AS out_links,
                         collect(DISTINCT {id: incoming.id, title: incoming.title, score: incoming.interest_score}) AS in_links
                    RETURN out_links, in_links
                    """,
                    page_id=page_id,
                )

                if result:
                    return json.dumps(result[0])
                return json.dumps({"out_links": [], "in_links": []})

        class UpdatePageScoresTool(Tool):
            """Update interest_score and metadata for pages."""

            name = "update_page_scores"
            description = "Update scores for pages. Pass JSON array: [{id, score, reasoning, skip_reason}]"
            inputs = {
                "scores_json": {
                    "type": "string",
                    "description": "JSON array of score objects",
                },
            }
            output_type = "string"

            def forward(self, scores_json: str) -> str:
                import json

                try:
                    scores = json.loads(scores_json)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON: {e}"})

                updated = 0
                for s in scores:
                    page_id = s.get("id")
                    score = s.get("score", 0.5)
                    reasoning = s.get("reasoning", "")
                    skip_reason = s.get("skip_reason")
                    page_type = s.get("page_type", "other")
                    is_physics = s.get("is_physics", False)
                    value_rating = s.get("value_rating", 0)

                    gc.query(
                        """
                        MATCH (wp:WikiPage {id: $id})
                        SET wp.interest_score = $score,
                            wp.score_reasoning = $reasoning,
                            wp.skip_reason = $skip_reason,
                            wp.page_type = $page_type,
                            wp.is_physics_content = $is_physics,
                            wp.value_rating = $value_rating,
                            wp.status = 'scored',
                            wp.scored_at = datetime()
                        """,
                        id=page_id,
                        score=score,
                        reasoning=reasoning,
                        skip_reason=skip_reason,
                        page_type=page_type,
                        is_physics=is_physics,
                        value_rating=value_rating,
                    )
                    updated += 1

                    # Track stats
                    if score >= 0.7:
                        stats.high_score_count += 1
                        stats.page_high_score_count += 1
                    elif score < 0.3:
                        stats.low_score_count += 1
                        stats.page_low_score_count += 1

                stats.pages_scored += updated
                return json.dumps({"updated": updated})

        class GetScoringProgressTool(Tool):
            """Get current scoring progress."""

            name = "get_scoring_progress"
            description = "Get current scoring progress and remaining budget."
            inputs = {}
            output_type = "string"

            def forward(self) -> str:
                import json

                result = gc.query(
                    """
                    MATCH (wp:WikiPage {facility_id: $facility_id})
                    RETURN wp.status AS status, count(*) AS count
                    """,
                    facility_id=facility_id,
                )

                return json.dumps(
                    {
                        "status_counts": {r["status"]: r["count"] for r in result},
                        "pages_scored": stats.pages_scored,
                        "cost_spent": stats.cost_spent_usd,
                        "cost_remaining": stats.cost_remaining(),
                    }
                )

        return [
            GetPagesToScoreTool(),
            GetNeighborInfoTool(),
            UpdatePageScoresTool(),
            GetScoringProgressTool(),
        ]

    def _get_artifact_scoring_tools(self) -> list[Tool]:
        """Get tools for scoring artifacts."""
        gc = self._get_gc()
        facility_id = self.config.facility_id
        stats = self.stats

        class GetArtifactsToScoreTool(Tool):
            """Get discovered artifacts that need scoring."""

            name = "get_artifacts_to_score"
            description = "Get artifacts needing scores. Returns id, filename, artifact_type, in_degree, link_depth."
            inputs = {
                "limit": {
                    "type": "integer",
                    "description": "Maximum artifacts to return (default 100)",
                    "nullable": True,
                },
            }
            output_type = "string"

            def forward(self, limit: int = 100) -> str:
                import json

                result = gc.query(
                    """
                    MATCH (wa:WikiArtifact {facility_id: $facility_id, status: 'discovered'})
                    WHERE wa.interest_score IS NULL
                    RETURN wa.id AS id,
                           wa.filename AS filename,
                           wa.artifact_type AS artifact_type,
                           wa.in_degree AS in_degree,
                           wa.link_depth AS link_depth
                    ORDER BY wa.in_degree DESC
                    LIMIT $limit
                    """,
                    facility_id=facility_id,
                    limit=limit,
                )

                return json.dumps({"artifacts": result, "count": len(result)})

        class UpdateArtifactScoresTool(Tool):
            """Update interest_score for artifacts."""

            name = "update_artifact_scores"
            description = "Update scores for artifacts. Pass JSON array: [{id, score, reasoning, skip_reason}]"
            inputs = {
                "scores_json": {
                    "type": "string",
                    "description": "JSON array of score objects",
                },
            }
            output_type = "string"

            def forward(self, scores_json: str) -> str:
                import json

                try:
                    scores = json.loads(scores_json)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON: {e}"})

                updated = 0
                for s in scores:
                    artifact_id = s.get("id")
                    score = s.get("score", 0.5)
                    reasoning = s.get("reasoning", "")
                    skip_reason = s.get("skip_reason")

                    # Determine status based on score
                    if score >= 0.5:
                        status = "scored"
                    else:
                        status = "skipped"

                    gc.query(
                        """
                        MATCH (wa:WikiArtifact {id: $id})
                        SET wa.interest_score = $score,
                            wa.score_reasoning = $reasoning,
                            wa.skip_reason = $skip_reason,
                            wa.status = $status,
                            wa.scored_at = datetime()
                        """,
                        id=artifact_id,
                        score=score,
                        reasoning=reasoning,
                        skip_reason=skip_reason,
                        status=status,
                    )
                    updated += 1

                    # Track stats
                    if score >= 0.7:
                        stats.high_score_count += 1
                        stats.artifact_high_score_count += 1
                    elif score < 0.3:
                        stats.low_score_count += 1
                        stats.artifact_low_score_count += 1

                stats.artifacts_scored += updated
                return json.dumps({"updated": updated})

        class GetArtifactContextTool(Tool):
            """Get pages that link to this artifact for context."""

            name = "get_artifact_context"
            description = (
                "Get pages that link to an artifact. Use to assess value from context."
            )
            inputs = {
                "artifact_id": {
                    "type": "string",
                    "description": "Artifact ID to get context for",
                },
            }
            output_type = "string"

            def forward(self, artifact_id: str) -> str:
                import json

                result = gc.query(
                    """
                    MATCH (wa:WikiArtifact {id: $artifact_id})
                    OPTIONAL MATCH (page:WikiPage)-[:LINKS_TO_ARTIFACT]->(wa)
                    WITH wa, collect(DISTINCT {
                        title: page.title,
                        score: page.interest_score,
                        in_degree: page.in_degree
                    }) AS linking_pages
                    RETURN linking_pages
                    """,
                    artifact_id=artifact_id,
                )

                if result:
                    return json.dumps(result[0])
                return json.dumps({"linking_pages": []})

        return [
            GetArtifactsToScoreTool(),
            UpdateArtifactScoresTool(),
            GetArtifactContextTool(),
        ]

    async def score(
        self,
        monitor: ScoreProgressMonitor | None = None,
        batch_size: int = 100,
    ) -> int:
        """Score pages using agent with graph metrics.

        Uses CLI-orchestrated batching with fresh agents to avoid context
        overflow. Each agent processes one batch, then a new agent is spawned.

        Args:
            monitor: Optional ScoreProgressMonitor for Rich display
            batch_size: Pages to fetch per agent iteration

        Returns:
            Number of pages scored in this session.
        """
        self.stats.phase = "SCORE"
        gc = self._get_gc()

        # Load prompt
        prompts = load_prompts()
        system_prompt = prompts.get("wiki/scorer")
        system_prompt_text = (
            system_prompt.content
            if system_prompt
            else self._get_default_scorer_prompt()
        )

        # Query graph for current state (resume from previous runs)
        state_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            WITH wp,
                 CASE WHEN wp.status = 'scored' THEN 1 ELSE 0 END AS scored,
                 CASE WHEN wp.interest_score >= 0.7 THEN 1 ELSE 0 END AS high,
                 CASE WHEN wp.interest_score IS NOT NULL AND wp.interest_score < 0.3 THEN 1 ELSE 0 END AS low
            RETURN count(*) AS total_pages,
                   sum(scored) AS already_scored,
                   sum(high) AS high_score,
                   sum(low) AS low_score,
                   count(*) - sum(scored) AS remaining
            """,
            facility_id=self.config.facility_id,
        )

        if state_result:
            state = state_result[0]
            total_pages = state["total_pages"]
            already_scored = state["already_scored"]
            remaining = state["remaining"]
            # Sync stats with graph state
            self.stats.pages_scored = already_scored
            self.stats.high_score_count = state["high_score"]
            self.stats.low_score_count = state["low_score"]
            self.stats.page_high_score_count = state["high_score"]
            self.stats.page_low_score_count = state["low_score"]
        else:
            total_pages = 0
            already_scored = 0
            remaining = 0

        # Get total artifacts to include in progress calculation
        artifact_count_result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility_id})
            WHERE wa.interest_score IS NULL OR wa.status IN ['scored', 'skipped']
            RETURN count(*) AS total_artifacts
            """,
            facility_id=self.config.facility_id,
        )
        total_artifacts = (
            artifact_count_result[0]["total_artifacts"] if artifact_count_result else 0
        )

        if monitor:
            # Set total to include both pages and artifacts from the start
            monitor.stats.total_pages = total_pages + total_artifacts
            monitor.stats.pages_scored = already_scored
            monitor.stats.high_score_count = self.stats.high_score_count
            monitor.stats.low_score_count = self.stats.low_score_count
            # Force a render update to show initial state
            if monitor._live:
                monitor._live.update(monitor._render())

        if remaining == 0:
            logger.info("All pages already scored")
            return self.stats.pages_scored

        session_scored = 0

        # Orchestration loop: spawn fresh agents until done or budget exhausted
        while True:
            # Check cost limit
            if self.stats.budget_exhausted():
                logger.info("Cost limit reached: $%.2f", self.stats.cost_spent_usd)
                break

            # Check page limit if set
            if self.max_pages and session_scored >= self.max_pages:
                logger.info("Page limit reached: %d", self.max_pages)
                break

            # Get batch of unscored pages
            pages_result = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility_id, status: 'discovered'})
                RETURN wp.id AS id,
                       wp.title AS title,
                       wp.link_depth AS depth,
                       wp.in_degree AS in_degree,
                       wp.out_degree AS out_degree
                ORDER BY wp.in_degree DESC
                LIMIT $limit
                """,
                facility_id=self.config.facility_id,
                limit=batch_size,
            )

            if not pages_result:
                logger.info("All pages scored")
                break

            # Create fresh agent for this batch
            model = self.resolved_model
            llm = create_litellm_model(
                model=model,
                temperature=0.3,
                max_tokens=8192,
            )

            tools = self._get_scoring_tools()
            agent = CodeAgent(
                tools=tools,
                model=llm,
                instructions=system_prompt_text,
                max_steps=20,
                name="wiki_scorer",
            )

            # Prepare batch info for agent
            import json

            batch_json = json.dumps(pages_result)
            task = f"""Score this batch of {len(pages_result)} wiki pages for {self.config.facility_id}.

For each page:
{batch_json}

Each page has: id, title, preview_summary, facility_id, in_degree, link_depth, preview_fetch_error

Assess each page and call update_page_scores with a JSON array:
[
  {{
    "id": "page_id",
    "score": 0.75,
    "page_type": "data_source",
    "is_physics": true,
    "value_rating": 8,
    "reasoning": "Brief explanation"
  }},
  ...
]

Scoring Guidelines:
- Data sources/databases: 0.7-1.0 (valuable even with low in_degree)
- Technical documentation: 0.6-0.8
- Code documentation: 0.6-0.8
- User guides/tutorials: 0.5-0.7
- Process/administrative: 0.3-0.5
- Meeting notes: 0.1-0.4

Important:
- Use preview_summary to understand page content
- Low in_degree does NOT mean low value for specialized content
- ITER Confluence pages may have lower in_degree by design
- Focus on CONTENT VALUE, not network topology
- If preview_summary is null or preview_fetch_error exists, score based on title only

Call update_page_scores with ALL pages in a single call."""

            batch_high = self.stats.high_score_count
            batch_low = self.stats.low_score_count
            batch_scored = 0

            try:
                await agent.run(task)
                # Count what was scored in this batch
                batch_scored = (
                    self.stats.pages_scored
                    - session_scored
                    - (
                        self.stats.pages_scored - session_scored - len(pages_result)
                        if self.stats.pages_scored > session_scored
                        else 0
                    )
                )
                # Estimate: assume all pages in batch were scored
                batch_scored = len(pages_result)

            except Exception as e:
                logger.error("Agent error on batch: %s", e)

            # Update session counter
            session_scored = self.stats.pages_scored

            # Update monitor
            if monitor:
                # Sample a page for display
                if pages_result:
                    sample = pages_result[0]
                    monitor.set_current(sample["title"], 0.5)
                monitor.add_batch(
                    scored=batch_scored,
                    high=self.stats.high_score_count - batch_high,
                    low=self.stats.low_score_count - batch_low,
                    cost=0.05,  # Estimated cost per batch
                )
                batch_high = self.stats.high_score_count
                batch_low = self.stats.low_score_count

            # Estimate cost (rough: ~$0.05 per batch with Sonnet)
            self.stats.cost_spent_usd += 0.05

        # Now score artifacts
        await self._score_artifacts(monitor, batch_size)

        return self.stats.pages_scored + self.stats.artifacts_scored

    async def _score_artifacts(
        self,
        monitor: ScoreProgressMonitor | None = None,
        batch_size: int = 100,
    ) -> int:
        """Score artifacts based on filename and linking pages."""
        gc = self._get_gc()

        # Get artifact scorer prompt
        artifact_prompt = self._get_artifact_scorer_prompt()

        # Get total unscored artifacts
        total_result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility_id})
            WHERE wa.interest_score IS NULL
            RETURN count(*) AS total
            """,
            facility_id=self.config.facility_id,
        )
        total_unscored = total_result[0]["total"] if total_result else 0

        if total_unscored == 0:
            # No new artifacts to score - check if we've previously scored any
            previously_scored_result = gc.query(
                """
                MATCH (wa:WikiArtifact {facility_id: $facility_id})
                WHERE wa.interest_score IS NOT NULL
                WITH wa,
                     CASE WHEN wa.interest_score >= 0.7 THEN 1 ELSE 0 END AS high,
                     CASE WHEN wa.interest_score IS NOT NULL AND wa.interest_score < 0.3 THEN 1 ELSE 0 END AS low
                RETURN count(*) AS total,
                       sum(high) AS high_count,
                       sum(low) AS low_count
                """,
                facility_id=self.config.facility_id,
            )

            if previously_scored_result and previously_scored_result[0]["total"] > 0:
                prev = previously_scored_result[0]
                # Update stats with previously-scored counts
                self.stats.artifacts_scored = prev["total"]
                self.stats.artifact_high_score_count = prev["high_count"] or 0
                self.stats.artifact_low_score_count = prev["low_count"] or 0
                # Also update combined counts
                self.stats.high_score_count += self.stats.artifact_high_score_count
                self.stats.low_score_count += self.stats.artifact_low_score_count
                logger.info(
                    "No new artifacts to score. Previously scored: %d (%d high, %d low)",
                    prev["total"],
                    prev["high_count"] or 0,
                    prev["low_count"] or 0,
                )
            else:
                logger.info("No artifacts to score")
            return self.stats.artifacts_scored

        while True:
            # Check cost limit
            if self.stats.budget_exhausted():
                logger.info("Cost limit reached during artifact scoring")
                break

            # Get batch of unscored artifacts
            artifacts_result = gc.query(
                """
                MATCH (wa:WikiArtifact {facility_id: $facility_id})
                WHERE wa.interest_score IS NULL
                RETURN wa.id AS id,
                       wa.filename AS filename,
                       wa.artifact_type AS artifact_type,
                       wa.in_degree AS in_degree,
                       wa.link_depth AS link_depth
                ORDER BY wa.in_degree DESC
                LIMIT $limit
                """,
                facility_id=self.config.facility_id,
                limit=batch_size,
            )

            if not artifacts_result:
                logger.info("All artifacts scored")
                break

            # Create fresh agent for this batch
            model = self.resolved_model
            llm = create_litellm_model(
                model=model,
                temperature=0.3,
                max_tokens=8192,
            )

            tools = self._get_artifact_scoring_tools()
            agent = CodeAgent(
                tools=tools,
                model=llm,
                instructions=artifact_prompt,
                max_steps=20,
                name="artifact_scorer",
            )

            import json

            batch_json = json.dumps(artifacts_result)
            task = f"""Score this batch of {len(artifacts_result)} wiki artifacts for {self.config.facility_id}.

Here are the artifacts with their metadata:
{batch_json}

For each artifact, compute interest_score (0.0-1.0) based on:
- filename keywords: manual, guide, Thomson, LIUQE = high; meeting, workshop = low
- artifact_type: pdf manuals = high, presentations = medium, images = low
- in_degree: >3 high value, 0 low value

Call update_artifact_scores with ALL artifacts in a single call.
Provide reasoning for each score."""

            try:
                await agent.run(task)
            except Exception as e:
                logger.error("Agent error on artifact batch: %s", e)

            # Update monitor
            if monitor and artifacts_result:
                sample = artifacts_result[0]
                monitor.set_current(sample["filename"], 0.5)
                monitor.add_batch(
                    scored=len(artifacts_result),
                    high=0,
                    low=0,
                    cost=0.05,
                )

            self.stats.cost_spent_usd += 0.05

        return self.stats.artifacts_scored

    def _get_artifact_scorer_prompt(self) -> str:
        """System prompt for artifact scoring agent."""
        return """You are scoring wiki artifacts (PDFs, presentations, documents) for a fusion research facility.

Your goal is to assign interest_score (0.0-1.0) based on filename and context.

## Scoring Guidelines

HIGH SCORE (0.7-1.0):
- Filename contains physics terms: Thomson, CXRS, LIUQE, equilibrium, MHD, diagnostic
- Filename contains data terms: manual, guide, calibration, nodes, signals
- Type: pdf with technical documentation
- in_degree > 3: Many pages reference this

MEDIUM SCORE (0.4-0.7):
- Technical presentations with useful content
- Code documentation, analysis tools
- in_degree 1-3

LOW SCORE (0.0-0.4):
- Filename contains: meeting, workshop, notes, draft, minutes
- Type: generic images without technical content
- in_degree = 0 AND no physics/data keywords

## Important

- PDFs with physics terms are almost always valuable
- Base scores on filename keywords and artifact_type
- Use get_artifact_context for ambiguous filenames
- Process all artifacts in a single update_artifact_scores call
- Provide reasoning for each score"""

    def _get_default_scorer_prompt(self) -> str:
        """Default system prompt for scoring agent."""
        return """You are scoring wiki pages for a fusion research facility.

Your goal is to assign interest_score (0.0-1.0) to each page based on measurable graph metrics.

## Scoring Guidelines

HIGH SCORE (0.7-1.0):
- Title contains physics terms: Thomson, CXRS, LIUQE, equilibrium, MHD, diagnostic
- Title contains data terms: nodes, signals, fields, parameters, calibration
- Leaf nodes (out_degree=0) with technical titles are DATA SOURCES - score HIGH
- in_degree > 5: Many pages link here - indicates importance
- Code documentation: DDJ, mdsopen, tcvget, TDI

MEDIUM SCORE (0.4-0.7):
- in_degree 1-5: Some references
- Administrative but useful content
- link_depth 3-5 with generic titles

LOW SCORE (0.0-0.4):
- Title contains: Meeting, Workshop, Mission, personal, ToDo, draft
- Old dated pages (2008, 2012) with no updates
- in_degree = 0 AND no physics/data keywords: True orphan pages

## Important

- LEAF NODES with physics terms are VALUABLE data sources, not low priority
- Low out_degree does NOT mean low value - it often means focused content
- ALWAYS provide reasoning for scores
- Skip pages with skip_reason if score < 0.5
- Process in batches of 20-50 pages
- Stop when all pages scored or budget exhausted"""

    # =========================================================================
    # PREFETCH - Fetch content and generate summaries
    # =========================================================================

    async def prefetch(
        self, batch_size: int = 50, include_scored: bool = False
    ) -> dict:
        """Prefetch page content and generate summaries for content-aware scoring.

        Args:
            batch_size: Pages per batch
            include_scored: Also prefetch already-scored pages

        Returns:
            Stats dict with fetched/summarized/failed counts
        """
        from imas_codex.wiki.prefetch import prefetch_pages

        self.stats.phase = "PREFETCH"
        return await prefetch_pages(
            facility_id=self.config.facility_id,
            batch_size=batch_size,
            max_pages=self.max_pages,
            include_scored=include_scored,
        )

    # =========================================================================
    # INGEST - Fetch and chunk high-score pages
    # =========================================================================

    async def ingest(
        self,
        min_score: float = 0.5,
        rate_limit: float = 0.5,
        content_type: str = "all",
    ) -> dict:
        """Ingest high-score pages and artifacts.

        Args:
            min_score: Minimum interest score threshold
            rate_limit: Seconds between requests
            content_type: 'all', 'pages', or 'artifacts'

        Returns:
            Stats dict with ingestion counts
        """
        from imas_codex.wiki import (
            WikiArtifactPipeline,
            WikiIngestionPipeline,
            get_pending_wiki_artifacts,
            get_pending_wiki_pages,
        )
        from imas_codex.wiki.pipeline import create_wiki_vector_index

        self.stats.phase = "INGEST"

        # Create vector index
        try:
            create_wiki_vector_index()
        except Exception:
            pass

        total_stats = {
            "pages": 0,
            "artifacts": 0,
            "chunks": 0,
        }

        # Ingest pages
        if content_type in ("all", "pages"):
            pending_pages = get_pending_wiki_pages(
                facility_id=self.config.facility_id,
                limit=self.max_pages,
                min_interest_score=min_score,
            )

            if pending_pages:
                pipeline = WikiIngestionPipeline(
                    facility_id=self.config.facility_id, use_rich=False
                )
                stats = await pipeline.ingest_from_graph(
                    limit=self.max_pages,
                    min_interest_score=min_score,
                    rate_limit=rate_limit,
                )
                total_stats["pages"] = stats.get("pages", 0)
                total_stats["chunks"] += stats.get("chunks", 0)
                self.stats.pages_ingested = total_stats["pages"]
                self.stats.chunks_created = total_stats["chunks"]

        # Ingest artifacts
        if content_type in ("all", "artifacts"):
            pending_artifacts = get_pending_wiki_artifacts(
                facility_id=self.config.facility_id,
                limit=self.max_pages,
                min_interest_score=min_score,
            )

            if pending_artifacts:
                artifact_pipeline = WikiArtifactPipeline(
                    facility_id=self.config.facility_id, use_rich=False
                )
                stats = await artifact_pipeline.ingest_from_graph(
                    limit=self.max_pages,
                    min_interest_score=min_score,
                    rate_limit=rate_limit,
                )
                total_stats["artifacts"] = stats.get("artifacts", 0)
                total_stats["chunks"] += stats.get("chunks", 0)
                self.stats.chunks_created = total_stats["chunks"]

        return total_stats

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def run(self) -> DiscoveryStats:
        """Run full discovery pipeline: crawl → prefetch → score → ingest."""
        console.print(f"[bold]Wiki Discovery: {self.config.facility_id}[/bold]")
        console.print(f"Portal: {self.config.portal_page}")
        console.print(f"Cost limit: ${self.stats.cost_limit_usd:.2f}")
        console.print()

        # Step 1: Crawl
        console.print("[cyan]Step 1: CRAWL[/cyan]")
        with CrawlProgressMonitor() as monitor:
            self.crawl(monitor)

        # Step 2: Prefetch content for content-aware scoring
        console.print("\n[cyan]Step 2: PREFETCH[/cyan]")
        prefetch_stats = await self.prefetch(batch_size=50, include_scored=False)
        console.print(
            f"  Fetched: {prefetch_stats['fetched']}, "
            f"Summarized: {prefetch_stats['summarized']}, "
            f"Failed: {prefetch_stats['failed']}"
        )

        # Step 3: Score with content-aware evaluation
        console.print("\n[cyan]Step 3: SCORE[/cyan]")
        with ScoreProgressMonitor(
            total=0,  # Will be set by score()
            cost_limit=self.stats.cost_limit_usd,
            facility=self.config.facility_id,
        ) as monitor:
            await self.score(monitor)
        console.print(
            f"  Scored {self.stats.pages_scored} pages ({self.stats.page_high_score_count} high, {self.stats.page_low_score_count} low) + "
            f"{self.stats.artifacts_scored} artifacts ({self.stats.artifact_high_score_count} high, {self.stats.artifact_low_score_count} low)"
        )

        # Step 4: Ingest high-score content
        console.print("\n[cyan]Step 4: INGEST[/cyan]")
        ingest_stats = await self.ingest(
            min_score=0.5, rate_limit=0.5, content_type="all"
        )
        console.print(
            f"  Ingested {ingest_stats['pages']} pages, "
            f"{ingest_stats['artifacts']} artifacts, "
            f"{ingest_stats['chunks']} chunks"
        )

        console.print(
            f"\n[green]Discovery complete in {self.stats.elapsed_formatted()}[/green]"
        )
        console.print(
            f"Crawled: {self.stats.pages_crawled} pages, {self.stats.artifacts_found} artifacts | "
            f"Scored: {self.stats.pages_scored + self.stats.artifacts_scored} total ({self.stats.high_score_count} high, {self.stats.low_score_count} low) | "
            f"Ingested: {self.stats.pages_ingested} pages, {self.stats.chunks_created} chunks"
        )
        console.print(f"Cost: ${self.stats.cost_spent_usd:.4f}")

        return self.stats

    def close(self) -> None:
        """Close graph connection."""
        if self._gc:
            self._gc.close()
            self._gc = None


async def run_wiki_discovery(
    facility: str = "tcv",
    cost_limit_usd: float = 10.0,
    max_pages: int | None = None,
    max_depth: int | None = None,
    verbose: bool = False,
    model: str | None = None,
    focus: str | None = None,
) -> dict:
    """Run wiki discovery and return stats as dict.

    Args:
        facility: Facility ID (e.g., "tcv")
        cost_limit_usd: Maximum cost budget
        max_pages: Maximum pages to crawl (None = unlimited)
        max_depth: Maximum link depth from portal
        verbose: Enable verbose output
        model: LLM model override (None = use config)
        focus: Optional focus for discovery (e.g., "equilibrium")

    Returns:
        Dictionary with discovery statistics
    """
    discovery = WikiDiscovery(
        facility=facility,
        cost_limit_usd=cost_limit_usd,
        max_pages=max_pages,
        max_depth=max_depth,
        verbose=verbose,
        model=model,
        focus=focus,
    )

    try:
        stats = await discovery.run()
        return {
            "pages_crawled": stats.pages_crawled,
            "artifacts_found": stats.artifacts_found,
            "links_found": stats.links_found,
            "pages_scored": stats.pages_scored,
            "high_score_count": stats.high_score_count,
            "low_score_count": stats.low_score_count,
            "pages_ingested": stats.pages_ingested,
            "cost_spent_usd": stats.cost_spent_usd,
            "elapsed_seconds": stats.elapsed_seconds(),
            "elapsed_formatted": stats.elapsed_formatted(),
        }
    finally:
        discovery.close()
