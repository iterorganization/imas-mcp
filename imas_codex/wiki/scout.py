"""Wiki scout agent for discovering and evaluating wiki pages.

Uses a ReAct agent to intelligently crawl wiki pages, evaluate their value,
and queue high-value pages for ingestion while skipping low-value content.

The agent has tools to:
- Crawl wiki links from a starting page
- Fetch lightweight previews with pattern extraction
- Queue pages with graph relationships
- Monitor discovery budget

Example:
    from imas_codex.wiki.scout import run_wiki_discovery

    stats = await run_wiki_discovery(
        facility="epfl",
        start_page="Portal:TCV",
        cost_limit_usd=1.00,
    )
    print(f"Queued {stats['pages_queued']} pages")
"""

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from imas_codex.agents.llm import get_llm
from imas_codex.agents.prompt_loader import load_prompts
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Wiki base URL
WIKI_BASE_URL = "https://spcwiki.epfl.ch/wiki"

# Patterns for entity extraction
MDSPLUS_PATH_PATTERN = re.compile(
    r"\\\\?[A-Z_][A-Z_0-9]*::[A-Z_][A-Z_0-9:]*",
    re.IGNORECASE,
)

PHYSICS_DOMAIN_KEYWORDS = {
    "equilibrium": "equilibrium",
    "transport": "transport",
    "mhd": "mhd",
    "heating": "heating",
    "current drive": "current_drive",
    "fueling": "fueling",
    "profiles": "profiles",
    "edge": "edge",
    "divertor": "divertor",
    "wall": "wall",
}

DIAGNOSTIC_KEYWORDS = [
    "thomson",
    "cxrs",
    "ece",
    "magnetics",
    "fir",
    "bolometer",
    "interferometer",
    "polarimeter",
    "mse",
    "bes",
    "reflectometer",
    "langmuir",
    "spectrometer",
    "soft x-ray",
    "hard x-ray",
]

CODE_KEYWORDS = [
    "liuqe",
    "astra",
    "cronos",
    "jetto",
    "raptor",
    "chease",
    "efit",
    "equilibrium",
    "reconstruction",
]


@dataclass
class DiscoveryBudget:
    """Track discovery cost budget."""

    cost_limit_usd: float = 1.00
    cost_spent_usd: float = 0.0
    pages_queued: int = 0
    pages_skipped: int = 0
    pages_explored: int = 0
    start_time: float = field(default_factory=time.time)

    def add_cost(self, cost: float) -> None:
        self.cost_spent_usd += cost

    def remaining(self) -> float:
        return max(0, self.cost_limit_usd - self.cost_spent_usd)

    def exhausted(self) -> bool:
        return self.cost_spent_usd >= self.cost_limit_usd

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "cost_spent_usd": round(self.cost_spent_usd, 4),
            "cost_limit_usd": self.cost_limit_usd,
            "cost_remaining_usd": round(self.remaining(), 4),
            "pages_queued": self.pages_queued,
            "pages_skipped": self.pages_skipped,
            "pages_explored": self.pages_explored,
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "budget_exhausted": self.exhausted(),
        }


@dataclass
class PagePreview:
    """Lightweight preview of a wiki page."""

    page_name: str
    title: str
    size_bytes: int
    mdsplus_paths: list[str]
    physics_domains: list[str]
    diagnostics: list[str]
    codes: list[str]
    has_tables: bool = False
    error: str | None = None


@dataclass
class PageEvaluation:
    """Agent's evaluation of a wiki page."""

    page_name: str
    status: str  # "discovered" or "skipped"
    interest_score: float
    skip_reason: str | None = None
    diagnostics: list[str] = field(default_factory=list)
    physics_domains: list[str] = field(default_factory=list)
    tree_nodes: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)


# Global budget instance for tools to access
_current_budget: DiscoveryBudget | None = None


def _get_budget() -> DiscoveryBudget:
    """Get current discovery budget."""
    global _current_budget
    if _current_budget is None:
        _current_budget = DiscoveryBudget()
    return _current_budget


# =============================================================================
# Tool Functions
# =============================================================================


def _crawl_wiki_links(
    start_page: str,
    depth: int = 1,
    facility: str = "epfl",
    max_pages: int = 200,
) -> str:
    """
    Crawl wiki links starting from a page.

    Discovers internal wiki page names by following links.
    Filters out special pages (File:, Category:, Special:, etc.)

    Args:
        start_page: Page name to start from (e.g., "Portal:TCV", "Diagnostics")
        depth: How many levels deep to crawl (1 = direct links only)
        facility: SSH host alias
        max_pages: Maximum pages to discover (default: 200)

    Returns:
        JSON string with list of discovered page names
    """
    import json

    budget = _get_budget()

    discovered: set[str] = set()
    to_crawl = [(start_page, 0)]
    crawled: set[str] = set()

    while to_crawl and len(discovered) < max_pages:
        page, current_depth = to_crawl.pop(0)

        if page in crawled or current_depth > depth:
            continue

        crawled.add(page)
        budget.pages_explored += 1

        # Extract links via SSH curl + grep
        cmd = f"curl -sk '{WIKI_BASE_URL}/{page}' | grep -oP 'href=\"/wiki/[^\"]+\"' | sed 's|href=\"/wiki/||;s|\"||g' | sort -u"

        try:
            result = subprocess.run(
                ["ssh", facility, cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                links = result.stdout.strip().split("\n")
                for link in links:
                    link = link.strip()
                    # Filter out special pages and anchors
                    if (
                        link
                        and ":" not in link  # Skip File:, Category:, etc.
                        and "#" not in link  # Skip anchor links
                        and not link.startswith("Special")
                        and not link.startswith("%")  # Skip encoded special chars
                        and link not in discovered
                        and len(discovered) < max_pages
                    ):
                        discovered.add(link)
                        if current_depth < depth:
                            to_crawl.append((link, current_depth + 1))

        except subprocess.TimeoutExpired:
            logger.warning("Timeout crawling %s", page)
            continue

    logger.info("Crawled %d pages, discovered %d links", len(crawled), len(discovered))
    return json.dumps({"pages": sorted(discovered), "crawled": len(crawled)})


def _fetch_single_preview(page_name: str, facility: str = "epfl") -> dict:
    """Fetch preview for a single page."""
    import urllib.parse

    encoded = urllib.parse.quote(page_name, safe="")
    url = f"{WIKI_BASE_URL}/{encoded}"

    # Fetch via SSH - get first 10KB for preview
    fetch_script = f'''python3 -c "
import urllib.request, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    resp = urllib.request.urlopen('{url}', context=ctx, timeout=15)
    html = resp.read(10240).decode('utf-8', errors='ignore')
    print(html)
except Exception as e:
    print('ERROR:', str(e))
"'''

    try:
        result = subprocess.run(
            ["ssh", facility, fetch_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0 or result.stdout.startswith("ERROR:"):
            return {"page_name": page_name, "error": result.stderr or result.stdout}

        html = result.stdout
        size = len(html)

        # Extract title
        title_match = re.search(r"<title>([^<]+)</title>", html)
        title = (
            title_match.group(1).replace(" - SPCwiki", "") if title_match else page_name
        )

        # Extract MDSplus paths
        mdsplus_paths = list(set(MDSPLUS_PATH_PATTERN.findall(html)))[:10]

        # Detect physics domains
        html_lower = html.lower()
        physics_domains = [
            domain
            for keyword, domain in PHYSICS_DOMAIN_KEYWORDS.items()
            if keyword in html_lower
        ]

        # Detect diagnostics
        diagnostics = [d for d in DIAGNOSTIC_KEYWORDS if d in html_lower]

        # Detect codes
        codes = [c for c in CODE_KEYWORDS if c in html_lower]

        # Check for tables (signal tables are high value)
        has_tables = "<table" in html_lower

        return {
            "page_name": page_name,
            "title": title,
            "size_bytes": size,
            "mdsplus_paths": mdsplus_paths,
            "physics_domains": physics_domains,
            "diagnostics": diagnostics,
            "codes": codes,
            "has_tables": has_tables,
        }

    except subprocess.TimeoutExpired:
        return {"page_name": page_name, "error": "Timeout fetching preview"}


def _fetch_wiki_previews(
    page_names: list[str],
    facility: str = "epfl",
    max_workers: int = 10,
) -> str:
    """
    Fetch lightweight previews for multiple wiki pages concurrently.

    Gets title, size, and extracts patterns (MDSplus paths, physics domains,
    diagnostic names) from the first portion of each page.

    Args:
        page_names: List of page names to preview
        facility: SSH host alias
        max_workers: Number of concurrent SSH connections (default: 10)

    Returns:
        JSON string with preview data for each page
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    previews: list[dict] = []

    # Limit to avoid overwhelming the SSH connection
    pages_to_fetch = page_names[:100]  # Cap at 100 pages per call

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_preview, page, facility): page
            for page in pages_to_fetch
        }

        for future in as_completed(futures):
            try:
                preview = future.result()
                previews.append(preview)
            except Exception as e:
                page = futures[future]
                previews.append({"page_name": page, "error": str(e)})

    return json.dumps({"previews": previews, "count": len(previews)})


def _search_wiki_patterns(
    page_names: list[str],
    patterns: list[str],
    facility: str = "epfl",
) -> str:
    """
    Search wiki pages for patterns and return match counts.

    Much faster than fetching full previews - uses grep on remote server.
    Use this to quickly identify high-value pages before detailed analysis.

    Args:
        page_names: List of page names to search
        patterns: List of regex patterns to search for (case-insensitive)
        facility: SSH host alias

    Returns:
        JSON with per-page match counts for each pattern
    """
    import json
    import urllib.parse

    # Build a single SSH command that searches all pages
    # This is much faster than individual fetches
    results: list[dict] = []

    # Process in batches to avoid command line length limits
    batch_size = 20
    pattern_expr = "|".join(patterns)

    for i in range(0, len(page_names), batch_size):
        batch = page_names[i : i + batch_size]

        # Build search commands for each page in batch
        search_cmds = []
        for page in batch:
            encoded = urllib.parse.quote(page, safe="")
            url = f"{WIKI_BASE_URL}/{encoded}"
            # Count matches for each pattern
            cmd = f'count=$(curl -sk "{url}" | grep -ciE "{pattern_expr}"); echo "{page}:$count"'
            search_cmds.append(cmd)

        full_cmd = " && ".join(search_cmds)

        try:
            result = subprocess.run(
                ["ssh", facility, f"{{ {full_cmd}; }}"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if ":" in line:
                        page, count_str = line.rsplit(":", 1)
                        try:
                            count = int(count_str)
                        except ValueError:
                            count = 0
                        results.append({"page_name": page, "match_count": count})

        except subprocess.TimeoutExpired:
            logger.warning("Timeout searching batch starting at %d", i)
            for page in batch:
                results.append(
                    {"page_name": page, "match_count": 0, "error": "timeout"}
                )

    # Sort by match count descending
    results.sort(key=lambda x: x.get("match_count", 0), reverse=True)

    return json.dumps(
        {
            "results": results,
            "patterns": patterns,
            "total_pages": len(page_names),
            "pages_with_matches": sum(
                1 for r in results if r.get("match_count", 0) > 0
            ),
        }
    )


def _queue_wiki_pages(evaluations_json: str, facility: str = "epfl") -> str:
    """
    Queue evaluated wiki pages in the graph.

    Creates WikiPage nodes with status and relationships based on
    agent evaluations. High-value pages get status='discovered',
    low-value pages get status='skipped'.

    Args:
        evaluations_json: JSON array of page evaluations with:
            - page_name: str
            - status: "discovered" | "skipped"
            - interest_score: float (0.0-1.0)
            - skip_reason: str | null
            - diagnostics: list[str] (optional)
            - physics_domains: list[str] (optional)
            - tree_nodes: list[str] (optional)
            - codes: list[str] (optional)
        facility: Facility ID

    Returns:
        JSON string with queue statistics
    """
    import json

    evaluations = json.loads(evaluations_json)
    budget = _get_budget()

    queued = 0
    skipped = 0
    errors = []

    with GraphClient() as gc:
        for eval_data in evaluations:
            page_name = eval_data.get("page_name")
            status = eval_data.get("status", "discovered")
            interest_score = eval_data.get("interest_score", 0.5)
            skip_reason = eval_data.get("skip_reason")

            page_id = f"{facility}:{page_name}"
            url = f"{WIKI_BASE_URL}/{page_name}"

            try:
                # Create WikiPage node
                gc.query(
                    """
                    MERGE (wp:WikiPage {id: $id})
                    SET wp.facility_id = $facility_id,
                        wp.url = $url,
                        wp.title = $page_name,
                        wp.status = $status,
                        wp.interest_score = $interest_score,
                        wp.skip_reason = $skip_reason,
                        wp.discovered_at = datetime()
                    WITH wp
                    MATCH (f:Facility {id: $facility_id})
                    MERGE (wp)-[:FACILITY_ID]->(f)
                    """,
                    id=page_id,
                    facility_id=facility,
                    url=url,
                    page_name=page_name,
                    status=status,
                    interest_score=interest_score,
                    skip_reason=skip_reason,
                )

                # Create relationships for discovered pages
                if status == "discovered":
                    # Link to Diagnostics
                    for diag_name in eval_data.get("diagnostics", []):
                        gc.query(
                            """
                            MATCH (wp:WikiPage {id: $page_id})
                            MERGE (d:Diagnostic {name: $name, facility_id: $facility_id})
                            MERGE (wp)-[:MENTIONS_DIAGNOSTIC]->(d)
                            """,
                            page_id=page_id,
                            name=diag_name,
                            facility_id=facility,
                        )

                    # Link to PhysicsDomains
                    for domain in eval_data.get("physics_domains", []):
                        gc.query(
                            """
                            MATCH (wp:WikiPage {id: $page_id})
                            MERGE (pd:PhysicsDomain {name: $name})
                            MERGE (wp)-[:MENTIONS_DOMAIN]->(pd)
                            """,
                            page_id=page_id,
                            name=domain,
                        )

                    # Link to TreeNodes (if they exist)
                    for path in eval_data.get("tree_nodes", []):
                        gc.query(
                            """
                            MATCH (wp:WikiPage {id: $page_id})
                            MATCH (tn:TreeNode {path: $path})
                            MERGE (wp)-[:MENTIONS_TREE_NODE]->(tn)
                            """,
                            page_id=page_id,
                            path=path,
                        )

                    # Link to AnalysisCodes
                    for code_name in eval_data.get("codes", []):
                        gc.query(
                            """
                            MATCH (wp:WikiPage {id: $page_id})
                            MERGE (ac:AnalysisCode {name: $name, facility_id: $facility_id})
                            MERGE (wp)-[:MENTIONS_CODE]->(ac)
                            """,
                            page_id=page_id,
                            name=code_name,
                            facility_id=facility,
                        )

                    queued += 1
                    budget.pages_queued += 1
                else:
                    skipped += 1
                    budget.pages_skipped += 1

            except Exception as e:
                errors.append(f"{page_name}: {e}")
                logger.error("Failed to queue %s: %s", page_name, e)

    return json.dumps(
        {
            "queued": queued,
            "skipped": skipped,
            "errors": errors,
        }
    )


def _get_discovery_budget() -> str:
    """
    Get current discovery budget status.

    Returns remaining cost, pages processed, and whether budget is exhausted.
    Use this to decide when to stop discovery.

    Returns:
        JSON string with budget status
    """
    import json

    return json.dumps(_get_budget().to_dict())


# =============================================================================
# Agent Creation
# =============================================================================


def _get_wiki_schema() -> str:
    """
    Get WikiPage schema for valid JSON generation.

    Returns the LinkML-derived schema for WikiPage nodes including:
    - Required and optional properties
    - Valid enum values for status
    - Relationship types created

    Call this before queue_wiki_pages to ensure valid JSON.
    """
    import json

    from imas_codex.graph import get_schema

    schema = get_schema()

    # Get WikiPage enum values
    wp_status_enum = schema.get_enums().get("WikiPageStatus", [])

    # Get relationship types for WikiPage
    relationships = [
        "FACILITY_ID -> Facility",
        "MENTIONS_DIAGNOSTIC -> Diagnostic",
        "MENTIONS_DOMAIN -> PhysicsDomain",
        "MENTIONS_TREE_NODE -> TreeNode",
        "MENTIONS_CODE -> AnalysisCode",
    ]

    # Build simplified schema for agent
    evaluation_schema = {
        "description": "Schema for queue_wiki_pages evaluations",
        "required_fields": {
            "page_name": "string - Wiki page name",
            "status": f"string - One of: {wp_status_enum}",
            "interest_score": "float - 0.0-1.0, higher = more valuable",
        },
        "optional_fields": {
            "skip_reason": "string - Required if status='skipped'. Examples: event_or_mission, stub_or_empty, administrative, no_signal_data",
            "diagnostics": "string[] - Diagnostic names found (creates MENTIONS_DIAGNOSTIC relationships)",
            "physics_domains": "string[] - Physics domains: equilibrium, transport, mhd, profiles",
            "codes": "string[] - Analysis code names: liuqe, astra, cronos",
            "tree_nodes": "string[] - MDSplus paths found (optional)",
        },
        "relationships_created": relationships,
        "example": [
            {
                "page_name": "Thomson",
                "status": "discovered",
                "interest_score": 0.9,
                "diagnostics": ["thomson"],
                "physics_domains": ["profiles"],
            },
            {
                "page_name": "Missions_2025",
                "status": "skipped",
                "interest_score": 0.1,
                "skip_reason": "event_or_mission",
            },
        ],
    }

    return json.dumps(evaluation_schema, indent=2)


def get_wiki_scout_tools(facility: str = "epfl") -> list[FunctionTool]:
    """Get tools for the wiki scout agent."""

    def crawl_links(start_page: str, depth: int = 1, max_pages: int = 100) -> str:
        return _crawl_wiki_links(start_page, depth, facility, max_pages)

    def fetch_previews(page_names: list[str]) -> str:
        return _fetch_wiki_previews(page_names, facility)

    def search_patterns(page_names: list[str], patterns: list[str]) -> str:
        return _search_wiki_patterns(page_names, patterns, facility)

    def queue_pages(evaluations_json: str) -> str:
        return _queue_wiki_pages(evaluations_json, facility)

    return [
        FunctionTool.from_defaults(
            fn=_get_wiki_schema,
            name="get_wiki_schema",
            description="Get WikiPage schema from LinkML. Call FIRST to get valid field names, enum values, and JSON structure for queue_wiki_pages.",
        ),
        FunctionTool.from_defaults(
            fn=crawl_links,
            name="crawl_wiki_links",
            description="Discover wiki page names by crawling links from a starting page. Args: start_page (str), depth (int, default 1), max_pages (int, default 100). Returns list of page names found.",
        ),
        FunctionTool.from_defaults(
            fn=search_patterns,
            name="search_wiki_patterns",
            description="FAST: Search pages for patterns (regex). Returns match counts per page. Use patterns like: tcv_shot::|results::|magnetics:: for MDSplus paths, or diagnostic names. Much faster than fetch_previews.",
        ),
        FunctionTool.from_defaults(
            fn=fetch_previews,
            name="fetch_wiki_previews",
            description="SLOW: Fetch full previews for pages. Use only for detailed analysis of promising pages identified by search_patterns.",
        ),
        FunctionTool.from_defaults(
            fn=queue_pages,
            name="queue_wiki_pages",
            description="Queue evaluated pages in the graph. Pass JSON string with array of evaluations. Call get_wiki_schema first for valid structure.",
        ),
        FunctionTool.from_defaults(
            fn=_get_discovery_budget,
            name="get_discovery_budget",
            description="Get current discovery budget status including cost spent, pages processed, and remaining budget.",
        ),
    ]


def get_wiki_scout_agent(
    facility: str = "epfl",
    verbose: bool = False,
    model: str = "google/gemini-3-flash-preview",
) -> ReActAgent:
    """
    Create a wiki scout agent for discovering and evaluating pages.

    Args:
        facility: Facility ID for wiki access
        verbose: Enable verbose agent output
        model: LLM model to use

    Returns:
        Configured ReActAgent
    """
    prompts = load_prompts()
    system_prompt = prompts.get("wiki-scout")

    if system_prompt is None:
        raise ValueError("wiki-scout prompt not found")

    llm = get_llm(model=model, temperature=0.3, max_tokens=16384)
    tools = get_wiki_scout_tools(facility)

    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        system_prompt=system_prompt.content,
        max_iterations=20,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_wiki_discovery(
    facility: str = "epfl",
    start_page: str = "Portal:TCV",
    cost_limit_usd: float = 1.00,
    verbose: bool = False,
    model: str = "google/gemini-3-flash-preview",
) -> dict:
    """
    Run wiki discovery using the scout agent.

    The agent will:
    1. Start from the given portal page
    2. Discover wiki pages by crawling links
    3. Fetch previews and evaluate page value
    4. Queue high-value pages, skip low-value ones
    5. Continue until budget exhausted or no new pages

    Args:
        facility: Facility ID
        start_page: Page to start discovery from
        cost_limit_usd: Maximum cost budget
        verbose: Enable verbose agent output
        model: LLM model to use

    Returns:
        Discovery statistics dict
    """
    global _current_budget
    _current_budget = DiscoveryBudget(cost_limit_usd=cost_limit_usd)

    agent = get_wiki_scout_agent(facility=facility, verbose=verbose, model=model)

    task = f"""Discover and evaluate wiki pages for facility '{facility}'.

Start from: {start_page}
Cost budget: ${cost_limit_usd:.2f}

Instructions:
1. Crawl links from {start_page} with depth=1 to get initial pages
2. Fetch previews for all discovered pages (batch them efficiently)
3. Evaluate each page based on preview data
4. Queue high-value pages (diagnostics, signals, codes) with interest_score >= 0.5
5. Skip low-value pages (events, meetings, stubs) with appropriate skip_reason
6. If you find portal-like pages, crawl those too for more content
7. Check budget periodically and stop when exhausted
8. Report final statistics when done

Focus on pages with:
- MDSplus paths (highest value)
- Diagnostic documentation
- Signal tables
- Analysis code docs
- COCOS/sign conventions
"""

    try:
        response = await agent.run(task)
        logger.info("Agent response: %s", str(response)[:500])
    except Exception as e:
        logger.error("Agent error: %s", e)

    return _current_budget.to_dict()


__all__ = [
    "DiscoveryBudget",
    "PageEvaluation",
    "PagePreview",
    "get_wiki_scout_agent",
    "get_wiki_scout_tools",
    "run_wiki_discovery",
]
