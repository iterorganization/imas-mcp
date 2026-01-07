"""Wiki scraper for TCV documentation via SSH.

Fetches wiki pages from EPFL SPC wiki (https://spcwiki.epfl.ch)
via SSH to bypass network restrictions. Extracts entity mentions
(MDSplus paths, IMAS paths, units, conventions) for graph linking.

The wiki is accessible from EPFL hosts without authentication.
Use -k flag to skip self-signed certificate verification.

Example:
    page = fetch_wiki_page("Thomson", facility="epfl")
    print(f"Found {len(page.mdsplus_paths)} MDSplus paths")
    print(f"Conventions: {page.conventions}")
"""

import hashlib
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Base URL for the TCV wiki
WIKI_BASE_URL = "https://spcwiki.epfl.ch/wiki"

# =============================================================================
# Entity Extraction Patterns
# =============================================================================

# MDSplus path patterns: \TREE::NODE or \\TREE::NODE:PATH
MDSPLUS_PATH_PATTERN = re.compile(
    r"\\\\?[A-Z_][A-Z_0-9]*::[A-Z_][A-Z_0-9:]*",
    re.IGNORECASE,
)

# IMAS path patterns: ids/path/to/field or ids.path.to.field
IMAS_PATH_PATTERN = re.compile(
    r"\b(equilibrium|core_profiles|magnetics|thomson_scattering|"
    r"charge_exchange|ece|interferometer|bolometer|nbi|mhd|"
    r"edge_profiles|polarimeter)[./][a-z_0-9./]+",
    re.IGNORECASE,
)

# Physical units (common in fusion data)
UNIT_PATTERN = re.compile(
    r"\b(eV|keV|MeV|GeV|"  # Energy
    r"m\^?-?[123]|cm\^?-?[123]|mm|"  # Length/density
    r"Tesla|T|Wb|Weber|"  # Magnetic
    r"Amp(?:ere)?s?|A|MA|kA|"  # Current
    r"Ohm|ohm|Ω|"  # Resistance
    r"Volt|V|kV|"  # Voltage
    r"Watt|W|MW|kW|"  # Power
    r"m/s|m\.s\^-1|"  # Velocity
    r"rad|radian|degrees?|°|"  # Angle
    r"seconds?|s|ms|μs|ns|"  # Time
    r"Hz|kHz|MHz|GHz|"  # Frequency
    r"Pa|kPa|MPa|bar|mbar|Torr)\b",
    re.IGNORECASE,
)

# COCOS convention patterns
COCOS_PATTERN = re.compile(
    r"COCOS\s*(\d{1,2})|"  # COCOS 11, COCOS 3
    r"cocos\s*=\s*(\d{1,2})|"  # cocos=11
    r"coordinate\s+convention[s]?\s+(\d{1,2})",
    re.IGNORECASE,
)

# Sign convention patterns
SIGN_CONVENTION_PATTERN = re.compile(
    r"(positive|negative)\s+(clockwise|counter-?clockwise|outward|inward|"
    r"upward|downward|toroidal|poloidal|radial)|"
    r"(sign\s+convention)|"
    r"(I_?[pP]\s*[><]=?\s*0)|"  # Ip > 0
    r"(B_?[tTφ]\s*[><]=?\s*0)",  # Bt > 0
    re.IGNORECASE,
)


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
        """Extract page name from URL."""
        return self.url.split("/wiki/")[-1] if "/wiki/" in self.url else self.url


# =============================================================================
# Entity Extraction Functions
# =============================================================================


def extract_mdsplus_paths(text: str) -> list[str]:
    """Extract MDSplus paths from text.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of MDSplus paths found
    """
    matches = MDSPLUS_PATH_PATTERN.findall(text)
    # Normalize: ensure single backslash prefix, uppercase
    normalized = set()
    for m in matches:
        path = m.lstrip("\\")
        path = "\\" + path.upper()
        normalized.add(path)
    return sorted(normalized)


def extract_imas_paths(text: str) -> list[str]:
    """Extract IMAS data dictionary paths from text.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of IMAS paths found
    """
    matches = IMAS_PATH_PATTERN.findall(text)
    # Normalize: lowercase, use / separator
    normalized = set()
    for m in matches:
        path = m.lower().replace(".", "/")
        normalized.add(path)
    return sorted(normalized)


def extract_units(text: str) -> list[str]:
    """Extract physical units from text.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of unit symbols found
    """
    matches = UNIT_PATTERN.findall(text)
    return sorted(set(matches))


def extract_conventions(text: str) -> list[dict[str, str]]:
    """Extract sign and coordinate conventions from text.

    Args:
        text: Raw text content

    Returns:
        List of convention dicts with type, name, and description
    """
    conventions = []

    # Find COCOS references
    for match in COCOS_PATTERN.finditer(text):
        cocos_num = match.group(1) or match.group(2) or match.group(3)
        if cocos_num:
            conventions.append(
                {
                    "type": "cocos",
                    "name": f"COCOS {cocos_num}",
                    "cocos_index": int(cocos_num),
                    "context": text[max(0, match.start() - 50) : match.end() + 50],
                }
            )

    # Find sign conventions
    for match in SIGN_CONVENTION_PATTERN.finditer(text):
        matched_text = match.group(0)
        conventions.append(
            {
                "type": "sign",
                "name": matched_text.strip(),
                "context": text[max(0, match.start() - 50) : match.end() + 50],
            }
        )

    return conventions


# =============================================================================
# SSH-Based Wiki Fetching
# =============================================================================


def fetch_wiki_page(
    page_name: str,
    facility: str = "epfl",
    timeout: int = 60,
) -> WikiPage:
    """Fetch a wiki page via SSH.

    The EPFL wiki is accessible from EPFL hosts without authentication.
    Uses Python's urllib on the remote host to fetch HTML content.

    Args:
        page_name: Wiki page name (e.g., "Thomson", "Diagnostics")
        facility: SSH host alias (default: "epfl")
        timeout: SSH command timeout in seconds

    Returns:
        WikiPage with HTML content and extracted entities

    Raises:
        RuntimeError: If SSH command fails or times out
    """
    url = f"{WIKI_BASE_URL}/{page_name}"

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

    logger.debug("Fetching wiki page: %s", url)

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
        "Fetched %s: %d chars, %d MDSplus paths, %d units, %d conventions",
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
    facility: str = "epfl",
    max_pages: int = 100,
    rate_limit: float = 0.5,
) -> list[str]:
    """Discover wiki pages by crawling links.

    Starts from a page and finds all internal wiki links.
    Filters out special pages (File:, Category:, Special:, etc.)

    Args:
        start_page: Page to start crawling from
        facility: SSH host alias
        max_pages: Maximum pages to discover
        rate_limit: Minimum seconds between requests

    Returns:
        List of discovered page names
    """
    discovered: set[str] = set()
    to_crawl = [start_page]
    crawled: set[str] = set()

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

    while to_crawl and len(discovered) < max_pages:
        page = to_crawl.pop(0)
        if page in crawled:
            continue

        crawled.add(page)

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
                        to_crawl.append(link)

        except subprocess.TimeoutExpired:
            logger.warning("Timeout discovering from %s", page)
            continue

        # Rate limiting
        time.sleep(rate_limit)

        if len(discovered) % 10 == 0:
            logger.info("Discovered %d pages...", len(discovered))

    return sorted(discovered)[:max_pages]


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
