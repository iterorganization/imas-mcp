"""Site-specific adapters for wiki discovery.

This module provides a unified interface for discovering pages and artifacts
across different wiki platforms (MediaWiki, TWiki, Confluence).

The key insight is separating:
1. Discovery (listing what exists) - uses platform APIs, no content fetch needed
2. Ingestion (extracting value) - requires content fetch, platform-agnostic

Each adapter implements:
- bulk_discover_pages() - List all pages via platform API
- bulk_discover_artifacts() - List all files/attachments via platform API
"""

from __future__ import annotations

import logging
import re
import subprocess
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredPage:
    """A page discovered via bulk discovery."""

    name: str
    url: str | None = None
    namespace: str | None = None


@dataclass
class DiscoveredArtifact:
    """An artifact (file) discovered via bulk discovery."""

    filename: str
    url: str
    artifact_type: str
    size_bytes: int | None = None
    mime_type: str | None = None
    # Pages that link to this artifact (for scoring by association)
    linked_pages: list[str] = field(default_factory=list)


class WikiAdapter(ABC):
    """Base class for wiki platform adapters."""

    site_type: str

    @abstractmethod
    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via platform API.

        Args:
            facility: Facility ID
            base_url: Base URL of the wiki
            on_progress: Progress callback (message, stats)

        Returns:
            List of discovered pages
        """
        pass

    @abstractmethod
    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover all artifacts (files) via platform API.

        Args:
            facility: Facility ID
            base_url: Base URL of the wiki
            on_progress: Progress callback (message, stats)

        Returns:
            List of discovered artifacts with metadata
        """
        pass


class MediaWikiAdapter(WikiAdapter):
    """Adapter for MediaWiki sites.

    Page discovery: Special:AllPages
    Artifact discovery: list=allimages API

    Can use either:
    - SSH proxy for shell commands
    - HTTP client with Tequila auth
    """

    site_type = "mediawiki"

    def __init__(
        self,
        ssh_host: str | None = None,
        wiki_client: MediaWikiClient | None = None,
        credential_service: str | None = None,
    ):
        """Initialize MediaWiki adapter.

        Args:
            ssh_host: SSH host for proxied commands (mutually exclusive with wiki_client)
            wiki_client: Authenticated MediaWikiClient (for Tequila auth)
            credential_service: Keyring service name (used with wiki_client)
        """
        self.ssh_host = ssh_host
        self.wiki_client = wiki_client
        self.credential_service = credential_service

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via Special:AllPages.

        Uses SSH if available, otherwise HTTP with auth.
        """
        if self.ssh_host:
            return self._discover_pages_ssh(facility, base_url, on_progress)
        elif self.wiki_client:
            return self._discover_pages_http(facility, base_url, on_progress)
        else:
            logger.warning(
                "No SSH host or wiki client configured for MediaWiki adapter"
            )
            return []

    def _discover_pages_ssh(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover pages via SSH using Special:AllPages."""
        pages: list[DiscoveredPage] = []
        index_url = f"{base_url}/index.php?title=Special:AllPages"

        # First request to get page count
        cmd = f'curl -sk "{index_url}"'
        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return pages

            html = result.stdout

            # Extract page links from current range
            pages.extend(self._extract_page_links(html, base_url))

            # Find pagination links (from=NextPage)
            pagination_pattern = (
                r'href="[^"]*title=Special:AllPages[^"]*from=([^&"]+)[^"]*"'
            )
            ranges = re.findall(pagination_pattern, html)
            ranges = list(set(ranges))  # Deduplicate

            if on_progress:
                on_progress(f"found {len(ranges)} page ranges", None)

            # Fetch each range
            for i, from_page in enumerate(ranges):
                range_url = (
                    f"{base_url}/index.php?title=Special:AllPages&from={from_page}"
                )
                cmd = f'curl -sk "{range_url}"'

                result = subprocess.run(
                    ["ssh", self.ssh_host, cmd],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    pages.extend(self._extract_page_links(result.stdout, base_url))

                if on_progress:
                    on_progress(
                        f"range {i + 1}/{len(ranges)}: {len(pages)} pages", None
                    )

        except subprocess.TimeoutExpired:
            logger.warning("Timeout during bulk page discovery")
        except Exception as e:
            logger.warning(f"Error during bulk page discovery: {e}")

        # Deduplicate
        seen = set()
        unique_pages = []
        for page in pages:
            if page.name not in seen:
                seen.add(page.name)
                unique_pages.append(page)

        return unique_pages

    def _discover_pages_http(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover pages via HTTP with Tequila auth."""
        if not self.wiki_client:
            return []

        # Ensure client is authenticated before using session
        if hasattr(self.wiki_client, "authenticate"):
            try:
                self.wiki_client.authenticate()
            except Exception as e:
                logger.warning(f"Failed to authenticate wiki client: {e}")
                return []

        pages: list[DiscoveredPage] = []
        index_url = f"{base_url}/index.php?title=Special:AllPages"

        try:
            response = self.wiki_client.session.get(index_url, verify=False, timeout=60)
            if response.status_code != 200:
                return pages

            html = response.text
            pages.extend(self._extract_page_links(html, base_url))

            # Find pagination
            pagination_pattern = (
                r'href="[^"]*title=Special:AllPages[^"]*from=([^&"]+)[^"]*"'
            )
            ranges = list(set(re.findall(pagination_pattern, html)))

            if on_progress:
                on_progress(f"found {len(ranges)} page ranges", None)

            for i, from_page in enumerate(ranges):
                range_url = (
                    f"{base_url}/index.php?title=Special:AllPages&from={from_page}"
                )
                response = self.wiki_client.session.get(
                    range_url, verify=False, timeout=60
                )
                if response.status_code == 200:
                    pages.extend(self._extract_page_links(response.text, base_url))

                if on_progress:
                    on_progress(
                        f"range {i + 1}/{len(ranges)}: {len(pages)} pages", None
                    )

        except Exception as e:
            logger.warning(f"Error during HTTP page discovery: {e}")

        # Deduplicate
        seen = set()
        unique_pages = []
        for page in pages:
            if page.name not in seen:
                seen.add(page.name)
                unique_pages.append(page)

        return unique_pages

    def _extract_page_links(self, html: str, base_url: str) -> list[DiscoveredPage]:
        """Extract page links from Special:AllPages HTML."""
        pages = []

        # Links in allpagesform table: /wiki/Page_Name or index.php?title=Page_Name
        link_pattern = re.compile(
            r'<a[^>]+href="([^"]*(?:/wiki/|title=)([^"&]+))"[^>]*>([^<]+)</a>',
            re.IGNORECASE,
        )

        excluded_prefixes = (
            "Special:",
            "File:",
            "Talk:",
            "User_talk:",
            "Template:",
            "Category:",
            "Help:",
            "MediaWiki:",
            "User:",
            "Module:",
        )

        for match in link_pattern.finditer(html):
            href, page_name, _title = match.groups()

            # Decode URL encoding
            page_name = urllib.parse.unquote(page_name)

            # Skip excluded namespaces
            if page_name.startswith(excluded_prefixes):
                continue

            # Skip AllPages navigation links
            if "AllPages" in page_name or "Allpages" in page_name:
                continue

            # Construct full URL
            if href.startswith("/"):
                url = f"{base_url.rsplit('/', 1)[0]}{href}"
            else:
                url = f"{base_url}/index.php?title={urllib.parse.quote(page_name, safe='')}"

            pages.append(DiscoveredPage(name=page_name, url=url))

        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover all artifacts via MediaWiki API list=allimages.

        The API returns all uploaded files with metadata:
        - filename, url, size, mime type
        - No need to crawl pages!

        Example API call:
        /api.php?action=query&list=allimages&ailimit=500&aiprop=url|size|mime&format=json
        """
        if self.ssh_host:
            return self._discover_artifacts_ssh(facility, base_url, on_progress)
        elif self.wiki_client:
            return self._discover_artifacts_http(facility, base_url, on_progress)
        else:
            logger.warning("No SSH host or wiki client configured")
            return []

    def _discover_artifacts_ssh(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts via SSH using MediaWiki API."""
        import json
        from urllib.parse import urlparse

        artifacts: list[DiscoveredArtifact] = []

        # MediaWiki API endpoint - derive from base URL
        # base_url might be https://spcwiki.epfl.ch/wiki, but API is at https://spcwiki.epfl.ch/api.php
        parsed = urlparse(base_url)
        api_url = f"{parsed.scheme}://{parsed.netloc}/api.php"
        params = (
            "action=query&list=allimages&ailimit=500&aiprop=url|size|mime&format=json"
        )

        continue_token = None
        batch = 0

        while True:
            url = f"{api_url}?{params}"
            if continue_token:
                url += f"&aicontinue={continue_token}"

            cmd = f'curl -sk "{url}"'

            try:
                result = subprocess.run(
                    ["ssh", self.ssh_host, cmd],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    break

                data = json.loads(result.stdout)
                images = data.get("query", {}).get("allimages", [])

                for img in images:
                    artifact = self._parse_image_info(img, facility)
                    if artifact:
                        artifacts.append(artifact)

                batch += 1
                if on_progress:
                    on_progress(f"batch {batch}: {len(artifacts)} artifacts", None)

                # Check for continuation
                if "continue" in data:
                    continue_token = data["continue"].get("aicontinue")
                else:
                    break

            except subprocess.TimeoutExpired:
                logger.warning("Timeout during artifact discovery")
                break
            except json.JSONDecodeError:
                logger.warning("Invalid JSON response from MediaWiki API")
                break
            except Exception as e:
                logger.warning(f"Error during artifact discovery: {e}")
                break

        return artifacts

    def _discover_artifacts_http(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts via HTTP with Tequila auth.

        Tries MediaWiki API first, falls back to HTML scraping of Special:ListFiles
        for older MediaWiki versions that don't support list=allimages API.
        """
        if not self.wiki_client:
            logger.warning("No wiki_client provided for artifact discovery")
            return []

        # Ensure client is authenticated before using session
        if hasattr(self.wiki_client, "authenticate"):
            try:
                logger.debug("Authenticating wiki client for artifact discovery...")
                self.wiki_client.authenticate()
                logger.debug("Wiki client authenticated successfully")
            except Exception as e:
                logger.warning(f"Failed to authenticate wiki client: {e}")
                return []

        # Try API first
        artifacts = self._discover_artifacts_via_api(facility, base_url, on_progress)
        if artifacts:
            return artifacts

        # API didn't work, fall back to HTML scraping
        logger.debug(
            "API unavailable, falling back to HTML scraping of Special:ListFiles"
        )
        return self._discover_artifacts_via_html(facility, base_url, on_progress)

    def _discover_artifacts_via_api(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts via MediaWiki API (list=allimages)."""
        artifacts: list[DiscoveredArtifact] = []

        # MediaWiki API endpoint - use wiki path (common for proxied wikis)
        api_url = f"{base_url}/api.php"

        params = {
            "action": "query",
            "list": "allimages",
            "ailimit": 500,
            "aiprop": "url|size|mime",
            "format": "json",
        }

        continue_token = None
        batch = 0

        while True:
            if continue_token:
                params["aicontinue"] = continue_token

            try:
                logger.debug(f"Requesting artifacts from {api_url}")
                response = self.wiki_client.session.get(
                    api_url, params=params, verify=False, timeout=60
                )
                logger.debug(
                    f"Response status: {response.status_code}, URL: {response.url}"
                )
                if response.status_code != 200:
                    logger.warning(f"Non-200 response: {response.status_code}")
                    break

                # Check if response is HTML (API unavailable) instead of JSON
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    # Old MediaWiki versions don't support list=allimages API
                    # This is expected - HTML fallback will be used
                    logger.debug(
                        f"API returned HTML, falling back to HTML scraping. URL: {response.url}"
                    )
                    break

                data = response.json()
                images = data.get("query", {}).get("allimages", [])

                for img in images:
                    artifact = self._parse_image_info(img, facility)
                    if artifact:
                        artifacts.append(artifact)

                batch += 1
                if on_progress:
                    on_progress(f"batch {batch}: {len(artifacts)} artifacts", None)

                # Check for continuation
                if "continue" in data:
                    continue_token = data["continue"].get("aicontinue")
                else:
                    break

            except Exception as e:
                logger.warning(f"Error during HTTP artifact discovery: {e}")
                break

        return artifacts

    def _parse_image_info(self, img: dict, facility: str) -> DiscoveredArtifact | None:
        """Parse MediaWiki API image info into DiscoveredArtifact."""
        filename = img.get("name", "")
        url = img.get("url", "")
        size = img.get("size")
        mime = img.get("mime", "")

        if not filename or not url:
            return None

        # Determine artifact type from extension or mime type
        artifact_type = self._get_artifact_type(filename, mime)

        return DiscoveredArtifact(
            filename=filename,
            url=url,
            artifact_type=artifact_type,
            size_bytes=size,
            mime_type=mime,
        )

    def _discover_artifacts_via_html(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts by scraping Special:ListFiles page.

        Fallback for older MediaWiki installations (< 1.17) that don't support
        the list=allimages API action.
        """
        from urllib.parse import unquote, urljoin

        artifacts: list[DiscoveredArtifact] = []
        seen_filenames: set[str] = set()
        seen_offsets: set[str] = set()

        # Start with Special:ListFiles
        list_url = f"{base_url}/Special:ListFiles"
        batch = 0
        max_batches = 500  # Allow more batches for large wikis

        while list_url and batch < max_batches:
            try:
                response = self.wiki_client.session.get(
                    list_url, verify=False, timeout=60
                )
                if response.status_code != 200:
                    logger.warning(
                        f"Non-200 response from Special:ListFiles: {response.status_code}"
                    )
                    break

                html = response.text

                # Extract file info from File: page links
                # Pattern: href="/wiki/File:Filename.ext"
                file_pattern = re.compile(r'href="[^"]*File:([^"]+)"')
                for match in file_pattern.finditer(html):
                    filename = unquote(match.group(1))
                    if filename in seen_filenames:
                        continue
                    seen_filenames.add(filename)

                    # Find corresponding image URL
                    # Pattern: href="/wiki/images/X/XX/Filename.ext"
                    # This is the ACTUAL file URL, not the File: page
                    img_pattern = re.compile(
                        rf'href="([^"]*images/[^"]*{re.escape(filename)})"',
                        re.IGNORECASE,
                    )
                    img_match = img_pattern.search(html)
                    if img_match:
                        img_path = img_match.group(1)
                        url = urljoin(base_url, img_path)
                    else:
                        # Skip this artifact - we couldn't find the actual file URL
                        # The File: prefix URL is just a description page, not the file
                        logger.debug(
                            "Skipping artifact %s: could not find images/ URL",
                            filename,
                        )
                        continue

                    artifact = DiscoveredArtifact(
                        filename=filename,
                        url=url,
                        artifact_type=self._get_artifact_type(filename),
                    )
                    artifacts.append(artifact)

                batch += 1
                if on_progress:
                    on_progress(f"batch {batch}: {len(artifacts)} artifacts", None)

                # Find all offset links and pick one we haven't visited
                # Pattern: offset=TIMESTAMP (timestamps are numeric)
                offset_pattern = re.compile(
                    r'href="[^"]*Special:ListFiles[^"]*offset=(\d+)[^"]*"'
                )
                all_offsets = set(offset_pattern.findall(html))
                new_offsets = all_offsets - seen_offsets
                seen_offsets.update(all_offsets)

                if new_offsets:
                    # Pick the lowest (oldest) offset to traverse chronologically
                    next_offset = min(new_offsets)
                    list_url = f"{base_url}/index.php?title=Special:ListFiles&offset={next_offset}"
                else:
                    break

            except Exception as e:
                logger.warning(f"Error scraping Special:ListFiles: {e}")
                break

        logger.debug(f"Discovered {len(artifacts)} artifacts via HTML scraping")
        return artifacts

    def _get_artifact_type(self, filename: str, mime: str | None = None) -> str:
        """Get artifact type from filename extension or MIME type."""
        filename_lower = filename.lower()

        if filename_lower.endswith(".pdf"):
            return "pdf"
        if filename_lower.endswith((".doc", ".docx", ".odt", ".rtf")):
            return "document"
        if filename_lower.endswith((".ppt", ".pptx", ".key")):
            return "presentation"
        if filename_lower.endswith((".xls", ".xlsx", ".csv")):
            return "spreadsheet"
        if filename_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
            return "image"
        if filename_lower.endswith((".ipynb", ".nb")):
            return "notebook"
        if filename_lower.endswith((".h5", ".hdf5", ".mat", ".nc", ".cdf")):
            return "data"
        if filename_lower.endswith((".zip", ".tar", ".gz", ".tgz", ".7z")):
            return "archive"

        # Try MIME type
        if mime:
            if "pdf" in mime:
                return "pdf"
            if "image" in mime:
                return "image"
            if "spreadsheet" in mime or "excel" in mime:
                return "spreadsheet"
            if "presentation" in mime or "powerpoint" in mime:
                return "presentation"

        return "document"


class TWikiAdapter(WikiAdapter):
    """Adapter for TWiki sites.

    Page discovery: /bin/view/Web/ listing
    Artifact discovery: /pub/Web/ directory listing
    """

    site_type = "twiki"

    def __init__(self, ssh_host: str | None = None):
        """Initialize TWiki adapter.

        Args:
            ssh_host: SSH host for proxied commands
        """
        self.ssh_host = ssh_host

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via TWiki web listing."""
        if not self.ssh_host:
            logger.warning("TWiki adapter requires SSH host")
            return []

        pages: list[DiscoveredPage] = []

        # TWiki typically has webs like Main, Sandbox, etc.
        # List topics via index
        cmd = f'curl -sk "{base_url}/bin/view/Main" | grep -oP \'href="/twiki/bin/view/Main/[^"]+"\''

        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    # Extract topic name: href="/twiki/bin/view/Main/TopicName"
                    match = re.search(r'/twiki/bin/view/(\w+/\w+)"', line)
                    if match:
                        topic = match.group(1)
                        if not topic.startswith(("TWiki/", "Sandbox/")):
                            url = f"{base_url}/bin/view/{topic}"
                            pages.append(DiscoveredPage(name=topic, url=url))

            if on_progress:
                on_progress(f"discovered {len(pages)} pages", None)

        except Exception as e:
            logger.warning(f"Error during TWiki page discovery: {e}")

        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover all artifacts via TWiki pub directory listing."""
        if not self.ssh_host:
            logger.warning("TWiki adapter requires SSH host")
            return []

        artifacts: list[DiscoveredArtifact] = []

        # TWiki stores attachments in /pub/Web/Topic/filename
        # List via directory listing if available
        cmd = f'curl -sk "{base_url}/pub/Main/" | grep -oP \'href="[^"]+\\.(pdf|doc|docx|ppt|pptx|xls|xlsx|ipynb|h5|hdf5|mat)\'\''

        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    # Extract filename from href
                    match = re.search(r'href="([^"]+)"', line)
                    if match:
                        path = match.group(1)
                        filename = path.split("/")[-1]
                        if path.startswith("/"):
                            url = f"{base_url.rsplit('/', 1)[0]}{path}"
                        else:
                            url = f"{base_url}/pub/Main/{path}"

                        artifact_type = self._get_artifact_type(filename)
                        artifacts.append(
                            DiscoveredArtifact(
                                filename=filename,
                                url=url,
                                artifact_type=artifact_type,
                            )
                        )

            if on_progress:
                on_progress(f"discovered {len(artifacts)} artifacts", None)

        except Exception as e:
            logger.warning(f"Error during TWiki artifact discovery: {e}")

        return artifacts

    def _get_artifact_type(self, filename: str) -> str:
        """Get artifact type from filename."""
        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            return "pdf"
        if filename_lower.endswith((".doc", ".docx")):
            return "document"
        if filename_lower.endswith((".ppt", ".pptx")):
            return "presentation"
        if filename_lower.endswith((".xls", ".xlsx")):
            return "spreadsheet"
        if filename_lower.endswith(".ipynb"):
            return "notebook"
        if filename_lower.endswith((".h5", ".hdf5", ".mat")):
            return "data"
        return "document"


class ConfluenceAdapter(WikiAdapter):
    """Adapter for Confluence sites.

    Page discovery: GET /rest/api/content
    Artifact discovery: GET /rest/api/content/{pageId}/child/attachment
    """

    site_type = "confluence"

    def __init__(self, api_token: str | None = None, ssh_host: str | None = None):
        """Initialize Confluence adapter.

        Args:
            api_token: API token for REST API authentication
            ssh_host: SSH host for proxied commands (alternative to token)
        """
        self.api_token = api_token
        self.ssh_host = ssh_host

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via Confluence REST API."""
        import json

        pages: list[DiscoveredPage] = []

        # Confluence REST API endpoint
        api_url = f"{base_url}/rest/api/content"
        start = 0
        limit = 100

        while True:
            url = f"{api_url}?type=page&start={start}&limit={limit}&expand=space"

            if self.ssh_host:
                cmd = f'curl -sk "{url}"'
                try:
                    result = subprocess.run(
                        ["ssh", self.ssh_host, cmd],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode != 0:
                        break
                    data = json.loads(result.stdout)
                except Exception as e:
                    logger.warning(f"Error during Confluence discovery: {e}")
                    break
            else:
                # Direct HTTP (would need auth header)
                logger.warning("Confluence direct HTTP not implemented")
                break

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                page_id = page.get("id")
                title = page.get("title", "")
                space_key = page.get("space", {}).get("key", "")

                view_url = f"{base_url}/pages/viewpage.action?pageId={page_id}"
                pages.append(
                    DiscoveredPage(name=title, url=view_url, namespace=space_key)
                )

            if on_progress:
                on_progress(f"discovered {len(pages)} pages", None)

            # Check for more pages
            size = data.get("size", 0)
            if size < limit:
                break
            start += limit

        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover all attachments via Confluence REST API.

        Note: Requires iterating through pages to get attachments.
        For large wikis, this should be done in batches.
        """
        # For Confluence, we'd need to:
        # 1. List all pages
        # 2. For each page, GET /rest/api/content/{id}/child/attachment
        # This is O(n) API calls but still faster than parsing pages

        # For now, return empty - implement later if needed
        logger.info("Confluence artifact discovery not yet implemented")
        return []


def get_adapter(
    site_type: str,
    ssh_host: str | None = None,
    wiki_client: MediaWikiClient | None = None,
    credential_service: str | None = None,
    api_token: str | None = None,
) -> WikiAdapter:
    """Get the appropriate adapter for a wiki site type.

    Args:
        site_type: Type of wiki (mediawiki, twiki, confluence)
        ssh_host: SSH host for proxied commands
        wiki_client: Authenticated MediaWikiClient (for Tequila)
        credential_service: Keyring service name
        api_token: API token (for Confluence)

    Returns:
        WikiAdapter instance for the site type
    """
    if site_type == "mediawiki":
        return MediaWikiAdapter(
            ssh_host=ssh_host,
            wiki_client=wiki_client,
            credential_service=credential_service,
        )
    elif site_type == "twiki":
        return TWikiAdapter(ssh_host=ssh_host)
    elif site_type == "confluence":
        return ConfluenceAdapter(api_token=api_token, ssh_host=ssh_host)
    else:
        raise ValueError(f"Unknown site type: {site_type}")
