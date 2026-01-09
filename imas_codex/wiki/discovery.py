"""Three-phase wiki discovery pipeline.

Phase 1: CRAWL - Fast link extraction, builds wiki graph structure
Phase 2: SCORE - Agent evaluates graph metrics, assigns interest scores
Phase 3: INGEST - Fetch content for high-score pages, create chunks

This module is facility-agnostic - wiki configuration comes from facility YAML.

Example:
    from imas_codex.wiki.discovery import WikiDiscovery

    discovery = WikiDiscovery("epfl", cost_limit_usd=10.0)
    await discovery.run()
"""

import logging
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from imas_codex.agents.llm import get_llm, get_model_for_task
from imas_codex.agents.prompt_loader import load_prompts
from imas_codex.graph import GraphClient
from imas_codex.wiki.progress import (
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
    high_score_count: int = 0  # interest_score >= 0.7
    low_score_count: int = 0  # interest_score < 0.3

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
    """Wiki configuration for a facility."""

    base_url: str
    portal_page: str
    ssh_host: str
    facility_id: str

    @classmethod
    def from_facility(cls, facility: str) -> "WikiConfig":
        """Load wiki config from facility configuration."""
        # Default configurations per facility
        configs = {
            "epfl": {
                "base_url": "https://spcwiki.epfl.ch/wiki",
                "portal_page": "Portal:TCV",
                "ssh_host": "epfl",
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
            ssh_host=cfg["ssh_host"],
            facility_id=facility,
        )


class WikiDiscovery:
    """Three-phase wiki discovery pipeline."""

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
        max_depth: int = 10,
        verbose: bool = False,
    ):
        self.config = WikiConfig.from_facility(facility)
        self.stats = DiscoveryStats(cost_limit_usd=cost_limit_usd)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.verbose = verbose

        # Graph client for persistence
        self._gc: GraphClient | None = None

    def _get_gc(self) -> GraphClient:
        if self._gc is None:
            self._gc = GraphClient()
        return self._gc

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
        """Extract all internal wiki links from a page via SSH.

        Returns:
            (page_links, artifact_links) where:
            - page_links: list of page names to crawl
            - artifact_links: list of (url_path, artifact_type) tuples
        """
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

    def phase1_crawl(self, monitor: CrawlProgressMonitor | None = None) -> int:
        """Phase 1: Crawl wiki and build link structure.

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

                # Create WikiPage node
                page_id = f"{self.config.facility_id}:{page}"
                gc.query(
                    """
                    MERGE (wp:WikiPage {id: $id})
                    SET wp.title = $title,
                        wp.url = $url,
                        wp.status = 'crawled',
                        wp.facility_id = $facility_id,
                        wp.link_depth = $depth,
                        wp.out_degree = $out_degree,
                        wp.discovered_at = datetime()
                    WITH wp
                    MATCH (f:Facility {id: $facility_id})
                    MERGE (wp)-[:FACILITY_ID]->(f)
                    """,
                    id=page_id,
                    title=page,
                    url=f"{self.config.base_url}/{urllib.parse.quote(page, safe='')}",
                    facility_id=self.config.facility_id,
                    depth=current_depth,
                    out_degree=total_out_degree,
                )

                # Track page links for later relationship creation
                all_page_links[page] = page_links

                # Add new page links to frontier
                for link in page_links:
                    if link not in visited and link not in frontier:
                        if current_depth + 1 <= self.max_depth:
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
                        page=page,
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
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'crawled'})
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
        page_id = f"{self.config.facility_id}:{page}"
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
        for source_page, target_pages in results.items():
            if not target_pages:
                continue

            source_id = f"{self.config.facility_id}:{source_page}"
            target_ids = [f"{self.config.facility_id}:{t}" for t in target_pages]

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

    def _get_scoring_tools(self) -> list[FunctionTool]:
        """Get tools for the scoring agent."""
        gc = self._get_gc()
        facility_id = self.config.facility_id

        def get_pages_to_score(limit: int = 100) -> str:
            """Get crawled pages that need scoring, with graph metrics."""
            import json

            result = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility_id, status: 'crawled'})
                RETURN wp.id AS id,
                       wp.title AS title,
                       wp.link_depth AS depth,
                       wp.in_degree AS in_degree,
                       wp.out_degree AS out_degree
                ORDER BY wp.in_degree DESC
                LIMIT $limit
                """,
                facility_id=facility_id,
                limit=limit,
            )

            return json.dumps({"pages": result, "count": len(result)})

        def get_neighbor_info(page_id: str) -> str:
            """Get info about pages that link to/from this page."""
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

        def update_page_scores(scores_json: str) -> str:
            """Update interest_score for pages. Input: JSON array of {id, score, reasoning, skip_reason}."""
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

                # Determine status based on score
                if score >= 0.5:
                    status = "discovered"
                else:
                    status = "skipped"

                gc.query(
                    """
                    MATCH (wp:WikiPage {id: $id})
                    SET wp.interest_score = $score,
                        wp.score_reasoning = $reasoning,
                        wp.skip_reason = $skip_reason,
                        wp.status = $status,
                        wp.scored_at = datetime()
                    """,
                    id=page_id,
                    score=score,
                    reasoning=reasoning,
                    skip_reason=skip_reason,
                    status=status,
                )
                updated += 1

                # Track stats
                if score >= 0.7:
                    self.stats.high_score_count += 1
                elif score < 0.3:
                    self.stats.low_score_count += 1

            self.stats.pages_scored += updated
            return json.dumps({"updated": updated})

        def get_scoring_progress() -> str:
            """Get current scoring progress."""
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
                    "pages_scored": self.stats.pages_scored,
                    "cost_spent": self.stats.cost_spent_usd,
                    "cost_remaining": self.stats.cost_remaining(),
                }
            )

        return [
            FunctionTool.from_defaults(
                fn=get_pages_to_score,
                name="get_pages_to_score",
                description="Get crawled pages needing scores. Returns id, title, depth, in_degree, out_degree.",
            ),
            FunctionTool.from_defaults(
                fn=get_neighbor_info,
                name="get_neighbor_info",
                description="Get pages that link to/from a specific page. Use to assess value from context.",
            ),
            FunctionTool.from_defaults(
                fn=update_page_scores,
                name="update_page_scores",
                description="Update scores for pages. Pass JSON array: [{id, score, reasoning, skip_reason}]",
            ),
            FunctionTool.from_defaults(
                fn=get_scoring_progress,
                name="get_scoring_progress",
                description="Get current scoring progress and remaining budget.",
            ),
        ]

    async def phase2_score(
        self,
        monitor: ScoreProgressMonitor | None = None,
        batch_size: int = 100,
    ) -> int:
        """Phase 2: Score pages using agent with graph metrics.

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
        system_prompt = prompts.get("wiki-scorer")
        system_prompt_text = (
            system_prompt.content
            if system_prompt
            else self._get_default_scorer_prompt()
        )

        # Get total unscored pages for progress
        total_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'crawled'})
            RETURN count(*) AS total
            """,
            facility_id=self.config.facility_id,
        )
        total_unscored = total_result[0]["total"] if total_result else 0

        if monitor:
            monitor.stats.total_pages = total_unscored + self.stats.pages_scored

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
                MATCH (wp:WikiPage {facility_id: $facility_id, status: 'crawled'})
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
            model = get_model_for_task("discovery")
            llm = get_llm(model=model, temperature=0.3, max_tokens=8192)

            tools = self._get_scoring_tools()
            agent = ReActAgent(
                tools=tools,
                llm=llm,
                verbose=self.verbose,
                system_prompt=system_prompt_text,
                max_iterations=20,
            )

            # Prepare batch info for agent
            import json

            batch_json = json.dumps(pages_result)
            task = f"""Score this batch of {len(pages_result)} wiki pages for {self.config.facility_id}.

Here are the pages with their graph metrics:
{batch_json}

For each page, compute interest_score (0.0-1.0) based on:
- in_degree: >5 high value, 0 low value
- link_depth: ≤2 high value, >5 low value
- title keywords: Thomson, LIUQE, signals = high; Meeting, Workshop = low

Call update_page_scores with ALL pages in a single call.
Provide reasoning for each score."""

            batch_high = 0
            batch_low = 0
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

        return self.stats.pages_scored

    def _get_default_scorer_prompt(self) -> str:
        """Default system prompt for scoring agent."""
        return """You are scoring wiki pages for a fusion research facility.

Your goal is to assign interest_score (0.0-1.0) to each page based on measurable graph metrics.

## Scoring Guidelines

HIGH SCORE (0.7-1.0):
- in_degree > 5: Many pages link here - indicates importance
- Title contains: Thomson, CXRS, LIUQE, signals, nodes, calibration
- link_depth <= 2: Close to portal, central to documentation

MEDIUM SCORE (0.4-0.7):
- in_degree 1-5: Some references
- Technical content but not central
- link_depth 3-4

LOW SCORE (0.0-0.4):
- in_degree = 0: Orphan page, nobody references it
- Title contains: Meeting, Workshop, Mission, personal
- link_depth > 5: Far from main documentation

## Important

- ALWAYS provide reasoning for scores
- Skip pages with skip_reason if score < 0.5
- Use neighbor_info to check context when title is ambiguous
- Process in batches of 20-50 pages
- Stop when all pages scored or budget exhausted"""

    # =========================================================================
    # Phase 3: INGEST - Not implemented yet (placeholder)
    # =========================================================================

    async def phase3_ingest(self, progress: Progress | None = None) -> int:
        """Phase 3: Fetch and ingest high-score pages.

        Returns number of pages ingested.
        """
        self.stats.phase = "INGEST"
        # TODO: Implement full content fetching and chunking
        # For now, just return 0
        return 0

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def run(self) -> DiscoveryStats:
        """Run full three-phase discovery pipeline."""
        console.print(f"[bold]Wiki Discovery: {self.config.facility_id}[/bold]")
        console.print(f"Portal: {self.config.portal_page}")
        console.print(f"Cost limit: ${self.stats.cost_limit_usd:.2f}")
        console.print()

        # Phase 1: Crawl with integrated progress display
        console.print("[cyan]Phase 1: CRAWL[/cyan]")
        with CrawlProgressMonitor() as monitor:
            self.phase1_crawl(monitor)

        # Phase 2: Score
        console.print("\n[cyan]Phase 2: SCORE[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            scored = await self.phase2_score(progress)
        console.print(
            f"  Scored {scored} pages: {self.stats.high_score_count} high, {self.stats.low_score_count} low"
        )

        # Phase 3: Ingest (placeholder)
        # console.print("\n[cyan]Phase 3: INGEST[/cyan]")
        # ingested = await self.phase3_ingest(progress)
        # console.print(f"  Ingested {ingested} pages")

        console.print(
            f"\n[green]Discovery complete in {self.stats.elapsed_formatted()}[/green]"
        )
        console.print(
            f"Pages: {self.stats.pages_crawled}, Artifacts: {self.stats.artifacts_found}"
        )
        console.print(f"Cost: ${self.stats.cost_spent_usd:.4f}")

        return self.stats

    def close(self) -> None:
        """Close graph connection."""
        if self._gc:
            self._gc.close()
            self._gc = None


async def run_wiki_discovery(
    facility: str = "epfl",
    cost_limit_usd: float = 10.0,
    max_pages: int | None = None,
    max_depth: int = 10,
    verbose: bool = False,
) -> dict:
    """Run wiki discovery and return stats as dict.

    Args:
        facility: Facility ID (e.g., "epfl")
        cost_limit_usd: Maximum cost budget
        max_pages: Maximum pages to crawl (None = unlimited)
        max_depth: Maximum link depth from portal
        verbose: Enable verbose output

    Returns:
        Dictionary with discovery statistics
    """
    discovery = WikiDiscovery(
        facility=facility,
        cost_limit_usd=cost_limit_usd,
        max_pages=max_pages,
        max_depth=max_depth,
        verbose=verbose,
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
