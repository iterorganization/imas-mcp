"""Wiki scraper for TCV documentation via SSH.

Fetches wiki pages from EPFL SPC wiki (https://spcwiki.epfl.ch)
via SSH to bypass network restrictions. Extracts entity mentions
(MDSplus paths, IMAS paths, units, conventions) for graph linking.

The wiki is accessible from EPFL hosts without authentication.
Use -k flag to skip self-signed certificate verification.

Example:
    page = fetch_wiki_page("Thomson", facility="tcv")
    print(f"Found {len(page.mdsplus_paths)} MDSplus paths")
    print(f"Conventions: {page.conventions}")
"""

import hashlib
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

from .entity_extraction import (
    extract_conventions,
    extract_imas_paths,
    extract_mdsplus_paths,
    extract_units,
)

logger = logging.getLogger(__name__)

# Base URL for the TCV wiki
WIKI_BASE_URL = "https://spcwiki.epfl.ch/wiki"


# =============================================================================
# ID Normalization
# =============================================================================


def canonical_page_id(page_name: str, facility_id: str) -> str:
    """Generate canonical WikiPage ID.

    Ensures consistent ID format regardless of URL encoding.
    Uses decoded page name with special characters preserved.

    Args:
        page_name: Page name (may be URL-encoded or decoded)
        facility_id: Facility identifier (e.g., 'tcv')

    Returns:
        Canonical ID like 'tcv:Portal:TCV' or 'tcv:Thomson/DDJ'

    Examples:
        >>> canonical_page_id('Portal%3ATCV', 'tcv')
        'tcv:Portal:TCV'
        >>> canonical_page_id('Thomson/DDJ', 'tcv')
        'tcv:Thomson/DDJ'
    """
    import urllib.parse

    decoded = urllib.parse.unquote(page_name)
    return f"{facility_id}:{decoded}"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class WikiPage:
    """Scraped wiki page with structured content and extracted entities."""

    url: str
    title: str
    content_html: str
    content_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    mdsplus_paths: list[str] = field(default_factory=list)
    imas_paths: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    conventions: list[dict[str, str]] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """SHA256 hash of content for change detection (first 16 chars)."""
        return hashlib.sha256(self.content_html.encode()).hexdigest()[:16]

    @property
    def page_name(self) -> str:
        """Extract page name from URL (URL-decoded)."""
        import urllib.parse

        raw = self.url.split("/wiki/")[-1] if "/wiki/" in self.url else self.url
        return urllib.parse.unquote(raw)


# =============================================================================
# Wiki Fetching
# =============================================================================


def _fetch_confluence_page(
    page_id: str,
    facility: str,
    timeout: int = 30,
) -> WikiPage | None:
    """Fetch a Confluence page via REST API.

    Args:
        page_id: Confluence page ID
        facility: Facility identifier
        timeout: Request timeout

    Returns:
        WikiPage with content, or None if fetch failed
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.wiki.confluence import ConfluenceClient

    config = get_facility(facility)
    wiki_sites = config.get("wiki_sites", [])

    if not wiki_sites:
        logger.error("No wiki sites configured for facility: %s", facility)
        return None

    # Use first wiki site (typically the main one)
    site = wiki_sites[0]
    base_url = site["url"]
    credential_service = site.get("credential_service", "confluence")

    try:
        client = ConfluenceClient(base_url, credential_service, timeout=timeout)

        if not client.authenticate():
            logger.error("Failed to authenticate with Confluence")
            return None

        page = client.get_page_content(page_id)
        if not page:
            logger.error("Failed to fetch page %s", page_id)
            return None

        # Extract entities from content
        wiki_page = WikiPage(
            url=page.url,
            title=page.title,
            content_html=page.content_html,
            content_text=page.content_text,
            mdsplus_paths=extract_mdsplus_paths(page.content_text),
            imas_paths=extract_imas_paths(page.content_text),
            units=extract_units(page.content_text),
            conventions=extract_conventions(page.content_text),
        )

        client.close()
        return wiki_page

    except Exception as e:
        logger.error("Error fetching Confluence page %s: %s", page_id, e)
        return None


def fetch_wiki_page(
    page_name: str,
    facility: str = "tcv",
    timeout: int = 60,
    site_type: str = "mediawiki",
    auth_type: str | None = None,
    credential_service: str | None = None,
) -> WikiPage:
    """Fetch a wiki page.

    For MediaWiki with tequila auth: Fetches via HTTP with Tequila SSO.
    For MediaWiki with ssh_proxy: Fetches via SSH using urllib on remote host.
    For Confluence: Fetches via REST API.

    Args:
        page_name: Wiki page name (MediaWiki) or page ID (Confluence)
        facility: Facility identifier (e.g., "tcv", "iter")
        timeout: Request timeout in seconds
        site_type: Site type ("mediawiki" or "confluence")
        auth_type: Authentication type ("tequila", "ssh_proxy", "session", or None)
        credential_service: Keyring service name for credentials

    Returns:
        WikiPage with HTML content and extracted entities

    Raises:
        RuntimeError: If fetch fails
    """
    # Handle Confluence sites
    if site_type == "confluence":
        return _fetch_confluence_page(page_name, facility, timeout)

    # Handle MediaWiki with Tequila auth (direct HTTP)
    if auth_type == "tequila" or auth_type == "session":
        return _fetch_mediawiki_page_http(
            page_name=page_name,
            facility=facility,
            timeout=timeout,
            credential_service=credential_service or f"{facility}-wiki",
        )

    # Handle MediaWiki sites via SSH (fallback)
    return _fetch_mediawiki_page_ssh(page_name, facility, timeout)


def _fetch_mediawiki_page_http(
    page_name: str,
    facility: str,
    timeout: int,
    credential_service: str,
) -> WikiPage:
    """Fetch MediaWiki page via HTTP with Tequila authentication.

    Args:
        page_name: Wiki page name
        facility: Facility identifier
        timeout: Request timeout in seconds
        credential_service: Keyring service name for credentials

    Returns:
        WikiPage with HTML content and extracted entities

    Raises:
        RuntimeError: If fetch fails
    """
    from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

    client = MediaWikiClient(
        base_url=WIKI_BASE_URL,
        credential_service=credential_service,
        timeout=timeout,
        verify_ssl=False,  # Self-signed cert
    )

    try:
        if not client.authenticate():
            raise RuntimeError(f"Failed to authenticate to {WIKI_BASE_URL}")

        page = client.get_page(page_name)
        if page is None:
            raise RuntimeError(f"Failed to fetch page: {page_name}")

        # Extract entities from content
        mdsplus_paths = extract_mdsplus_paths(page.content_html)
        imas_paths = extract_imas_paths(page.content_html)
        units = extract_units(page.content_html)
        conventions = extract_conventions(page.content_html)

        logger.info(
            "Fetched %s via HTTP: %d chars, %d MDSplus paths, %d units, %d conventions",
            page_name,
            len(page.content_html),
            len(mdsplus_paths),
            len(units),
            len(conventions),
        )

        return WikiPage(
            url=page.url,
            title=page.title,
            content_html=page.content_html,
            content_text=page.content_text,
            mdsplus_paths=mdsplus_paths,
            imas_paths=imas_paths,
            units=units,
            conventions=conventions,
        )

    finally:
        client.close()


def _fetch_mediawiki_page_ssh(
    page_name: str,
    facility: str,
    timeout: int,
) -> WikiPage:
    """Fetch MediaWiki page via SSH proxy (fallback method).

    Args:
        page_name: Wiki page name
        facility: Facility identifier (SSH host)
        timeout: Request timeout in seconds

    Returns:
        WikiPage with HTML content and extracted entities

    Raises:
        RuntimeError: If fetch fails
    """
    # URL-encode page name to handle spaces and special characters
    # Use safe="/" to preserve subpage slashes (e.g., Thomson/DDJ)
    import urllib.parse

    encoded_page_name = urllib.parse.quote(page_name, safe="/")
    url = f"{WIKI_BASE_URL}/{encoded_page_name}"

    # Fetch via SSH with SSL verification disabled (self-signed cert)
    # Use urllib directly on the remote host
    fetch_script = f'''python3 -c "
import urllib.request
import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    html = urllib.request.urlopen('{url}', context=ctx, timeout=30).read().decode('utf-8', errors='ignore')
    print(html)
except Exception as e:
    print('ERROR:', str(e))
"'''

    logger.debug("Fetching wiki page via SSH: %s", url)

    try:
        result = subprocess.run(
            ["ssh", facility, fetch_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"SSH timeout after {timeout}s fetching {page_name}") from e

    if result.returncode != 0:
        raise RuntimeError(f"SSH failed: {result.stderr}")

    html = result.stdout

    if html.startswith("ERROR:"):
        raise RuntimeError(f"Fetch failed: {html}")

    # Extract title (simple regex)
    title_match = re.search(r"<title>([^<]+)</title>", html)
    title = title_match.group(1).replace(" - SPCwiki", "") if title_match else page_name

    # Extract text content for entity extraction
    # Simple approach: strip tags (for production, use BeautifulSoup)
    text_content = re.sub(r"<[^>]+>", " ", html)
    text_content = re.sub(r"\s+", " ", text_content)

    # Extract entities
    mdsplus_paths = extract_mdsplus_paths(html)
    imas_paths = extract_imas_paths(html)
    units = extract_units(html)
    conventions = extract_conventions(html)

    logger.info(
        "Fetched %s via SSH: %d chars, %d MDSplus paths, %d units, %d conventions",
        page_name,
        len(html),
        len(mdsplus_paths),
        len(units),
        len(conventions),
    )

    return WikiPage(
        url=url,
        title=title,
        content_html=html,
        content_text=text_content,
        mdsplus_paths=mdsplus_paths,
        imas_paths=imas_paths,
        units=units,
        conventions=conventions,
    )


def discover_wiki_pages(
    start_page: str = "Main_Page",
    facility: str = "tcv",
    max_pages: int = 100,
    rate_limit: float = 0.5,
) -> list[str]:
    """Discover wiki pages by scanning links.

    Graph-aware scanner that skips already-discovered pages and continues
    finding new pages. Subsequent runs will explore deeper into the wiki
    beyond the previous limit.

    Starts from a page and finds all internal wiki links.
    Filters out special pages (File:, Category:, Special:, etc.)

    Args:
        start_page: Page to start scanning from
        facility: SSH host alias
        max_pages: Maximum NEW pages to discover (skips existing)
        rate_limit: Minimum seconds between requests

    Returns:
        List of newly discovered page names (excludes already-queued pages)
    """
    from imas_codex.graph import GraphClient

    # Query graph for already-discovered pages
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            RETURN wp.title AS title
            """,
            facility_id=facility,
        )
        already_discovered = {r["title"] for r in result} if result else set()

    logger.info(
        "Skipping %d already-discovered pages from graph", len(already_discovered)
    )

    discovered: set[str] = already_discovered.copy()
    new_pages: set[str] = set()  # Track only newly found pages
    to_scan = [start_page]
    scanned: set[str] = set()

    # Priority pages to ensure we get the important ones
    priority_pages = [
        "Portal:TCV",
        "Diagnostics",
        "MDS",
        "Thomson",
        "CXRS",
        "ECE",
        "Magnetics",
        "Ion_Temperature_Nodes",
    ]

    for page in priority_pages:
        if page not in discovered:
            discovered.add(page)
            new_pages.add(page)

    while to_scan and len(new_pages) < max_pages:
        page = to_scan.pop(0)
        if page in scanned:
            continue

        scanned.add(page)

        # Extract links via SSH
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
                    # Filter out special pages
                    if (
                        link
                        and ":" not in link  # Skip File:, Category:, etc.
                        and not link.startswith("Special")
                        and link not in discovered
                    ):
                        discovered.add(link)
                        new_pages.add(link)
                        to_scan.append(link)

        except subprocess.TimeoutExpired:
            logger.warning("Timeout discovering from %s", page)
            continue

        # Rate limiting
        time.sleep(rate_limit)

        if len(new_pages) % 10 == 0 and len(new_pages) > 0:
            logger.info(
                "Discovered %d new pages (%d total in graph)...",
                len(new_pages),
                len(discovered),
            )

    # Return only NEW pages
    result = sorted(new_pages)[:max_pages]
    logger.info(
        "Discovery complete: %d new pages (graph had %d existing)",
        len(result),
        len(already_discovered),
    )
    return result


def get_priority_pages() -> list[str]:
    """Get list of high-priority wiki pages for initial ingestion.

    These pages contain structured MDSplus node tables and critical
    signal documentation.

    Returns:
        Ordered list of priority page names
    """
    return [
        # Primary diagnostics with signal tables
        "Ion_Temperature_Nodes",
        "Thomson",
        "CXRS",
        "ECE",
        "Magnetics",
        "FIR",
        "Bolometry",
        # Data system documentation
        "MDS",
        "MDS_commands",
        # Portals for discovery
        "Portal:TCV",
        "Diagnostics",
        "TCV_all_diagnostics",
        # Analysis codes
        "LIUQE",
        "ASTRA",
        "NBI",
        "DNBI",
        # Conventions and calibration
        "TCV_data_acquisition",
        "Personal_RESULTS_trees",
    ]
