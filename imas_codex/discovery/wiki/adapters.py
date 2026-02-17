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
import shlex
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

    Page discovery: Special:AllPages (HTML scraping) or allpages API (JSON)
    Artifact discovery: list=allimages API

    Supports multiple auth backends:
    - SSH proxy for shell commands
    - HTTP client with Tequila auth (MediaWikiClient)
    - Pre-authenticated session (Keycloak OIDC, HTTP Basic, etc.)
    """

    site_type = "mediawiki"

    def __init__(
        self,
        ssh_host: str | None = None,
        wiki_client: MediaWikiClient | None = None,
        credential_service: str | None = None,
        session: Any = None,
    ):
        """Initialize MediaWiki adapter.

        Args:
            ssh_host: SSH host for proxied commands
            wiki_client: Authenticated MediaWikiClient (for Tequila auth)
            credential_service: Keyring service name (used with wiki_client)
            session: Pre-authenticated requests.Session (Keycloak, Basic auth, etc.)
        """
        self.ssh_host = ssh_host
        self.wiki_client = wiki_client
        self.credential_service = credential_service
        self.session = session

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via Special:AllPages or MediaWiki API.

        Priority: session (API) > wiki_client (HTML) > SSH (HTML).
        The API path (api.php?action=query&list=allpages) is preferred
        when a session is available (Keycloak, Basic auth) because it
        handles pagination reliably via JSON.
        """
        if self.session:
            return self._discover_pages_api(facility, base_url, on_progress)
        elif self.wiki_client:
            return self._discover_pages_http(facility, base_url, on_progress)
        elif self.ssh_host:
            return self._discover_pages_ssh(facility, base_url, on_progress)
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

    def _discover_pages_api(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover pages via MediaWiki allpages API with pre-authenticated session.

        Uses api.php?action=query&list=allpages which is the most reliable
        method. Works with any session that has valid auth cookies/headers
        (Keycloak OIDC, HTTP Basic, etc.).
        """
        import json as json_mod
        import urllib.parse

        if not self.session:
            return []

        all_pages: set[str] = set()
        apcontinue: str | None = None
        _SKIP_PREFIXES = (
            "Special:",
            "File:",
            "Talk:",
            "User:",
            "Template:",
            "Category:",
            "Help:",
            "MediaWiki:",
        )

        while True:
            api_url = (
                f"{base_url}/api.php?action=query&list=allpages&aplimit=500&format=json"
            )
            if apcontinue:
                api_url += f"&apcontinue={urllib.parse.quote(apcontinue)}"

            try:
                response = self.session.get(api_url, timeout=30)
                if response.status_code != 200:
                    logger.warning(
                        "API returned HTTP %d for %s", response.status_code, base_url
                    )
                    break
                data = json_mod.loads(response.text)
            except Exception as e:
                logger.warning("Error fetching API for %s: %s", base_url, e)
                break

            pages = data.get("query", {}).get("allpages", [])
            for page in pages:
                title = page.get("title", "")
                if title and not any(title.startswith(p) for p in _SKIP_PREFIXES):
                    all_pages.add(title)

            if on_progress:
                on_progress(f"{len(all_pages)} pages discovered", None)

            cont = data.get("continue", {})
            apcontinue = cont.get("apcontinue")
            if not apcontinue:
                break

        # Convert to DiscoveredPage objects
        discovered = []
        for title in sorted(all_pages):
            url = f"{base_url}/{urllib.parse.quote(title, safe='/')}"
            discovered.append(DiscoveredPage(name=title, url=url))

        return discovered

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
        if self.session:
            return self._discover_artifacts_via_session(facility, base_url, on_progress)
        elif self.wiki_client:
            return self._discover_artifacts_http(facility, base_url, on_progress)
        elif self.ssh_host:
            return self._discover_artifacts_ssh(facility, base_url, on_progress)
        else:
            logger.warning("No SSH host or wiki client configured")
            return []

    def _discover_artifacts_via_session(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts via pre-authenticated session (Keycloak, Basic auth).

        Uses the same allimages API + fileusage enrichment as the wiki_client
        path, but with the pre-authenticated requests.Session directly.
        """
        if not self.session:
            return []

        artifacts: list[DiscoveredArtifact] = []

        # Find working API URL
        parsed = urllib.parse.urlparse(base_url)
        api_candidates = [f"{base_url}/api.php"]
        root_api = f"{parsed.scheme}://{parsed.netloc}/api.php"
        if root_api != api_candidates[0]:
            api_candidates.append(root_api)

        api_url = None
        for candidate in api_candidates:
            try:
                probe = self.session.get(
                    candidate,
                    params={"action": "query", "meta": "siteinfo", "format": "json"},
                    verify=False,
                    timeout=15,
                )
                ct = probe.headers.get("content-type", "")
                if probe.status_code == 200 and "json" in ct:
                    api_url = candidate
                    break
            except Exception:
                continue

        if not api_url:
            logger.debug(
                "No working API URL found for session-based artifact discovery"
            )
            return []

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
                response = self.session.get(
                    api_url, params=params, verify=False, timeout=60
                )
                if response.status_code != 200:
                    break

                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
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

                if "continue" in data:
                    continue_token = data["continue"].get("aicontinue")
                else:
                    break

            except Exception as e:
                logger.warning(f"Error during session artifact discovery: {e}")
                break

        # Enrich artifacts with page links via prop=fileusage API
        if artifacts and api_url:
            self._enrich_artifact_page_links_http(artifacts, api_url, on_progress)

        return artifacts

    def _discover_artifacts_ssh(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts via SSH using MediaWiki API.

        Tries two API URL candidates since MediaWiki installations vary:
        1. {base_url}/api.php  (e.g. https://spcwiki.epfl.ch/wiki/api.php)
        2. {scheme}://{netloc}/api.php  (e.g. https://spcwiki.epfl.ch/api.php)

        SSH access bypasses SSO (e.g. Tequila) since curl runs from the
        facility's internal network.
        """
        import json
        from urllib.parse import urlparse

        artifacts: list[DiscoveredArtifact] = []

        # Build candidate API URLs - try wiki path first, then root
        parsed = urlparse(base_url)
        api_candidates = [f"{base_url}/api.php"]
        root_api = f"{parsed.scheme}://{parsed.netloc}/api.php"
        if root_api != api_candidates[0]:
            api_candidates.append(root_api)

        params = (
            "action=query&list=allimages&ailimit=500&aiprop=url|size|mime&format=json"
        )

        # Find working API URL
        api_url = None
        for candidate in api_candidates:
            test_url = f"{candidate}?action=query&meta=siteinfo&format=json"
            cmd = f'curl -sk "{test_url}"'
            try:
                result = subprocess.run(
                    ["ssh", self.ssh_host, cmd],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if "query" in data:
                        api_url = candidate
                        logger.debug("Using API URL: %s", api_url)
                        break
            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
                continue

        if not api_url:
            logger.warning(
                "Could not find working MediaWiki API via SSH (tried %s)",
                ", ".join(api_candidates),
            )
            return []

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

                # Guard against HTML responses (login pages, error pages)
                stdout = result.stdout.strip()
                if stdout.startswith("<!") or stdout.startswith("<html"):
                    logger.warning(
                        "SSH API returned HTML instead of JSON (possible auth redirect)"
                    )
                    break

                data = json.loads(stdout)
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

        # Enrich artifacts with page links via prop=images API
        if artifacts and api_url:
            self._enrich_artifact_page_links_ssh(artifacts, api_url, on_progress)

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
        """Discover artifacts via MediaWiki API (list=allimages).

        Tries two API URL candidates since MediaWiki installations vary:
        1. {base_url}/api.php  (e.g. https://spcwiki.epfl.ch/wiki/api.php)
        2. {scheme}://{netloc}/api.php  (e.g. https://spcwiki.epfl.ch/api.php)
        """
        artifacts: list[DiscoveredArtifact] = []

        # Build candidate API URLs
        parsed = urllib.parse.urlparse(base_url)
        api_candidates = [f"{base_url}/api.php"]
        root_api = f"{parsed.scheme}://{parsed.netloc}/api.php"
        if root_api != api_candidates[0]:
            api_candidates.append(root_api)

        # Find working API URL by probing siteinfo
        api_url = None
        for candidate in api_candidates:
            try:
                probe = self.wiki_client.session.get(
                    candidate,
                    params={"action": "query", "meta": "siteinfo", "format": "json"},
                    verify=False,
                    timeout=15,
                )
                ct = probe.headers.get("content-type", "")
                if probe.status_code == 200 and "json" in ct:
                    api_url = candidate
                    logger.debug("Using API URL: %s", api_url)
                    break
            except Exception:
                continue

        if not api_url:
            logger.debug(
                "No working API URL found (tried %s), will fall back to HTML",
                ", ".join(api_candidates),
            )
            return []

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

        # Enrich artifacts with page links via prop=fileusage API
        if artifacts and api_url:
            self._enrich_artifact_page_links_http(artifacts, api_url, on_progress)

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

    def _enrich_artifact_page_links_ssh(
        self,
        artifacts: list[DiscoveredArtifact],
        api_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Enrich artifacts with page links via SSH using prop=fileusage API.

        Queries which pages use each file by batching artifact filenames
        into prop=fileusage requests (50 files per batch). This is
        O(num_artifacts / 50) instead of O(all_pages * images_per_page).

        Args:
            artifacts: Discovered artifacts to enrich (modified in place)
            api_url: Working MediaWiki API URL
            on_progress: Optional progress callback
        """
        import json

        # Build case-insensitive filename -> artifact lookup
        artifact_by_name: dict[str, list[DiscoveredArtifact]] = {}
        for a in artifacts:
            key = a.filename.lower()
            artifact_by_name.setdefault(key, []).append(a)

        if not artifact_by_name:
            return

        # Build list of File: titles to query
        all_filenames = sorted(artifact_by_name.keys())
        file_titles = [f"File:{fn}" for fn in all_filenames]

        linked_count = 0
        batch = 0
        batch_size = 50  # MediaWiki titles limit per request

        for i in range(0, len(file_titles), batch_size):
            title_batch = file_titles[i : i + batch_size]
            titles_param = "|".join(title_batch)

            # Use prop=fileusage to find pages that embed these files
            continue_params: dict[str, str] = {}

            while True:
                params = {
                    "action": "query",
                    "titles": titles_param,
                    "prop": "fileusage",
                    "fulimit": "500",
                    "funamespace": "0",
                    "format": "json",
                    **continue_params,
                }
                param_str = "&".join(
                    f"{k}={urllib.parse.quote(str(v), safe='|:')}"
                    for k, v in params.items()
                )
                url = f"{api_url}?{param_str}"
                cmd = f'curl -sk "{url}"'

                try:
                    result = subprocess.run(
                        ["ssh", self.ssh_host, cmd],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode != 0:
                        logger.warning(
                            "fileusage API failed (rc=%d, batch %d): %s",
                            result.returncode,
                            i // batch_size,
                            result.stderr[:200] if result.stderr else "no stderr",
                        )
                        break

                    stdout = result.stdout.strip()
                    if stdout.startswith("<!") or stdout.startswith("<html"):
                        logger.warning(
                            "fileusage API returned HTML instead of JSON "
                            "(batch %d, likely auth redirect). "
                            "Artifact-page linking will be incomplete.",
                            i // batch_size,
                        )
                        break

                    data = json.loads(stdout)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "fileusage API timed out (batch %d)", i // batch_size
                    )
                    break
                except json.JSONDecodeError as e:
                    logger.warning(
                        "fileusage API returned invalid JSON (batch %d): %s",
                        i // batch_size,
                        e,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "fileusage API error (batch %d): %s", i // batch_size, e
                    )
                    break

                pages = data.get("query", {}).get("pages", {})
                for page_data in pages.values():
                    file_title = page_data.get("title", "")
                    fname = file_title.removeprefix("File:").lower()
                    for usage in page_data.get("fileusage", []):
                        page_title = usage.get("title", "")
                        if fname in artifact_by_name and page_title:
                            for artifact in artifact_by_name[fname]:
                                if page_title not in artifact.linked_pages:
                                    artifact.linked_pages.append(page_title)
                                    linked_count += 1

                batch += 1

                if "continue" in data:
                    continue_params = {
                        k: v for k, v in data["continue"].items() if k != "continue"
                    }
                else:
                    break

            if on_progress and (i // batch_size) % 5 == 0:
                on_progress(
                    f"enriching page links: batch {i // batch_size + 1}/"
                    f"{(len(file_titles) + batch_size - 1) // batch_size}, "
                    f"{linked_count} links",
                    None,
                )

        logger.info(
            "Enriched %d artifact-page links across %d API batches",
            linked_count,
            batch,
        )

    def _enrich_artifact_page_links_http(
        self,
        artifacts: list[DiscoveredArtifact],
        api_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Enrich artifacts with page links via HTTP using prop=fileusage API.

        Queries which pages use each file by batching artifact filenames
        into prop=fileusage requests (50 files per batch). This is
        O(num_artifacts / 50) instead of O(all_pages * images_per_page).

        Args:
            artifacts: Discovered artifacts to enrich (modified in place)
            api_url: Working MediaWiki API URL
            on_progress: Optional progress callback
        """
        if not self.wiki_client and not self.session:
            return

        session = self.session or (
            self.wiki_client.session if self.wiki_client else None
        )
        if not session:
            return

        # Build case-insensitive filename -> artifact lookup
        artifact_by_name: dict[str, list[DiscoveredArtifact]] = {}
        for a in artifacts:
            key = a.filename.lower()
            artifact_by_name.setdefault(key, []).append(a)

        if not artifact_by_name:
            return

        # Build list of File: titles to query
        all_filenames = sorted(artifact_by_name.keys())
        file_titles = [f"File:{fn}" for fn in all_filenames]

        linked_count = 0
        batch = 0
        batch_size = 50  # MediaWiki titles limit per request

        for i in range(0, len(file_titles), batch_size):
            title_batch = file_titles[i : i + batch_size]
            titles_param = "|".join(title_batch)

            continue_params: dict[str, str] = {}

            while True:
                params: dict[str, str | int] = {
                    "action": "query",
                    "titles": titles_param,
                    "prop": "fileusage",
                    "fulimit": 500,
                    "funamespace": 0,
                    "format": "json",
                    **continue_params,
                }

                try:
                    response = session.get(
                        api_url, params=params, verify=False, timeout=60
                    )
                    if response.status_code != 200:
                        break

                    content_type = response.headers.get("content-type", "")
                    if "text/html" in content_type:
                        break

                    data = response.json()
                except Exception:
                    break

                pages = data.get("query", {}).get("pages", {})
                for page_data in pages.values():
                    file_title = page_data.get("title", "")
                    fname = file_title.removeprefix("File:").lower()
                    for usage in page_data.get("fileusage", []):
                        page_title = usage.get("title", "")
                        if fname in artifact_by_name and page_title:
                            for artifact in artifact_by_name[fname]:
                                if page_title not in artifact.linked_pages:
                                    artifact.linked_pages.append(page_title)
                                    linked_count += 1

                batch += 1

                if "continue" in data:
                    continue_params = {
                        k: v for k, v in data["continue"].items() if k != "continue"
                    }
                else:
                    break

            if on_progress and (i // batch_size) % 5 == 0:
                on_progress(
                    f"enriching page links: batch {i // batch_size + 1}/"
                    f"{(len(file_titles) + batch_size - 1) // batch_size}, "
                    f"{linked_count} links",
                    None,
                )

        logger.info(
            "Enriched %d artifact-page links across %d API batches",
            linked_count,
            batch,
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

        # Start with Special:ListFiles (limit=500 to match API batch size)
        list_url = f"{base_url}/Special:ListFiles?limit=500"
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
                    list_url = f"{base_url}/index.php?title=Special:ListFiles&limit=500&offset={next_offset}"
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
        if filename_lower.endswith(".json"):
            return "json"
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
    """Adapter for live TWiki sites served via Apache/CGI.

    Discovers pages by fetching /bin/view/<Web>/WebTopicList for each configured
    web, then extracting topic links. Supports SSH-proxied access with proxy
    bypass for sites behind corporate firewalls.

    Page discovery: WebTopicList page per web
    Artifact discovery: /pub/<Web>/ directory listing
    """

    site_type = "twiki"

    def __init__(
        self,
        ssh_host: str | None = None,
        webs: list[str] | None = None,
        base_url: str | None = None,
        pub_path: str | None = None,
    ):
        """Initialize TWiki adapter.

        Args:
            ssh_host: SSH host for proxied commands
            webs: TWiki webs to discover (default: ["Main"])
            base_url: Base URL of TWiki installation (e.g. http://host/twiki)
            pub_path: Absolute path to TWiki pub/ directory on the server.
                When set, artifact discovery uses fd over SSH to scan
                pub/<web>/ for files, which is instant and provides
                inherent topic→artifact linkage from the directory structure.
        """
        self.ssh_host = ssh_host
        self.webs = webs or ["Main"]
        self._base_url = base_url
        self._pub_path = pub_path.rstrip("/") if pub_path else None

    def _ssh_curl(self, url: str, timeout: int = 30) -> str | None:
        """Fetch URL via SSH-proxied curl with proxy bypass.

        Args:
            url: URL to fetch
            timeout: Command timeout in seconds

        Returns:
            Response body text, or None on failure
        """
        if not self.ssh_host:
            logger.warning("TWiki adapter requires SSH host")
            return None

        cmd = f'curl -s --noproxy "*" --max-time {timeout} "{url}"'
        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=timeout + 15,
            )
            if result.returncode != 0:
                logger.warning(
                    "SSH curl failed for %s (exit %d)", url, result.returncode
                )
                return None
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning("SSH curl timed out for %s", url)
            return None
        except Exception as e:
            logger.warning("SSH curl error for %s: %s", url, e)
            return None

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages from configured TWiki webs via WebTopicList.

        For each web, fetches the WebTopicList page which contains links to
        every topic. Extracts topic names from href attributes.

        Args:
            facility: Facility ID
            base_url: TWiki base URL (e.g. http://host/twiki)
            on_progress: Progress callback

        Returns:
            List of discovered pages across all webs
        """
        if not self.ssh_host:
            logger.warning("TWiki adapter requires SSH host")
            return []

        effective_url = self._base_url or base_url
        pages: list[DiscoveredPage] = []

        for web in self.webs:
            topic_list_url = f"{effective_url}/bin/view/{web}/WebTopicList"
            logger.info("Fetching topic list for web '%s' from %s", web, topic_list_url)

            html = self._ssh_curl(topic_list_url)
            if not html:
                logger.warning("Failed to fetch WebTopicList for web '%s'", web)
                continue

            # Extract topic links: href="/twiki/bin/view/<Web>/<Topic>"
            pattern = rf'href="/twiki/bin/view/{re.escape(web)}/([^"]+)"'
            seen: set[str] = set()
            for match in re.finditer(pattern, html):
                topic_name = match.group(1)
                # Skip utility pages
                if topic_name in (
                    "WebTopicList",
                    "WebIndex",
                    "WebRss",
                    "WebAtom",
                    "WebNotify",
                    "WebChanges",
                    "WebSearch",
                    "WebSearchAdvanced",
                    "WebLeftBar",
                    "WebTopBar",
                    "WebPreferences",
                ):
                    continue
                if topic_name in seen:
                    continue
                seen.add(topic_name)

                page_name = f"{web}/{topic_name}"
                page_url = f"{effective_url}/bin/view/{web}/{topic_name}"
                pages.append(DiscoveredPage(name=page_name, url=page_url))

            logger.info("Discovered %d topics in web '%s'", len(seen), web)
            if on_progress:
                on_progress(f"web {web}: {len(seen)} topics", None)

        if on_progress:
            on_progress(f"discovered {len(pages)} pages total", None)

        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts via fd scanning of the pub/ directory tree.

        When pub_path is configured, uses a single SSH call with fd to list
        all artifact files across all webs. The directory structure
        pub/<web>/<topic>/<filename> provides inherent topic→artifact linkage.

        Falls back to HTTP directory listing when pub_path is not available.

        Args:
            facility: Facility ID
            base_url: TWiki base URL (used for building artifact URLs)
            on_progress: Progress callback

        Returns:
            List of discovered artifacts with page linkage
        """
        if not self.ssh_host:
            logger.warning("TWiki adapter requires SSH host")
            return []

        if self._pub_path:
            return self._discover_artifacts_via_fd(base_url, on_progress)

        logger.info("No pub_path configured, skipping artifact discovery")
        return []

    def _discover_artifacts_via_fd(
        self,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts by scanning the pub/ filesystem via fd over SSH.

        Single SSH call using fd to find all artifact files. The TWiki pub/
        directory structure (pub/<web>/<topic>/<filename>) maps each file
        to its parent topic for page linkage.

        Args:
            base_url: TWiki base URL (for building web-accessible URLs)
            on_progress: Progress callback

        Returns:
            List of discovered artifacts with topic linkage
        """
        effective_url = (self._base_url or base_url).rstrip("/")
        artifacts: list[DiscoveredArtifact] = []

        for web in self.webs:
            web_pub_path = f"{self._pub_path}/{web}"
            ext_args = " ".join(f"-e {ext.lstrip('.')}" for ext in _ARTIFACT_EXTENSIONS)
            cmd = f"fd {ext_args} . {shlex.quote(web_pub_path)}"

            if on_progress:
                on_progress(f"scanning {web_pub_path} via fd", None)

            logger.info(
                "Running fd artifact scan: %s via %s", web_pub_path, self.ssh_host
            )

            try:
                result = subprocess.run(
                    ["ssh", self.ssh_host, cmd],
                    capture_output=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                logger.warning("fd scan timed out for %s", web_pub_path)
                continue
            except Exception as e:
                logger.warning("fd scan failed for %s: %s", web_pub_path, e)
                continue

            if result.returncode not in (0, 1):
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                if stderr:
                    logger.warning("fd scan stderr: %s", stderr[:200])
                continue

            output = result.stdout.decode("utf-8", errors="replace")
            for line in output.strip().split("\n"):
                if not line:
                    continue

                filepath = line.strip()
                # Extract topic and filename from pub/<web>/<topic>/<filename>
                rel_path = filepath.replace(f"{self._pub_path}/{web}/", "")
                parts = rel_path.split("/")
                if len(parts) < 2:
                    continue

                topic_name = parts[0]
                filename = parts[-1]
                artifact_type = _get_artifact_type_from_filename(filename)

                # Build web-accessible URL: /twiki/pub/<web>/<topic>/<filename>
                url_path = f"/twiki/pub/{web}/{rel_path}"
                parsed = urllib.parse.urlparse(effective_url)
                artifact_url = f"{parsed.scheme}://{parsed.netloc}{url_path}"

                artifact = DiscoveredArtifact(
                    filename=filename,
                    url=artifact_url,
                    artifact_type=artifact_type,
                )
                artifact.linked_pages.append(f"{web}/{topic_name}")
                artifacts.append(artifact)

            logger.info(
                "fd scan complete for web '%s': %d artifacts", web, len(artifacts)
            )

        if on_progress:
            on_progress(f"discovered {len(artifacts)} artifacts", None)

        return artifacts


def _fetch_html_direct(url: str, timeout: float = 10.0) -> str | None:
    """Fetch HTML content via direct HTTP request.

    Tries a quick direct connection first. This succeeds when:
    - Running on a machine with VPN access (e.g., laptop)
    - A SOCKS/port-forward tunnel makes the URL reachable locally

    Uses a short timeout to fail fast when the URL isn't directly reachable.

    Args:
        url: URL to fetch
        timeout: Connection timeout in seconds (kept short for fast fallback)

    Returns:
        HTML content string, or None on failure
    """
    try:
        import httpx

        with httpx.Client(verify=False, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.debug("Direct HTTP failed for %s: %s", url, e)
        return None


# SOCKS proxy port for laptop tunnel (iter → laptop VPN → internet)
_SOCKS_PORT = 19080
_socks_tunnel_checked = False
_socks_tunnel_available = False


def _ensure_socks_tunnel() -> bool:
    """Ensure SOCKS tunnel to laptop is available.

    Establishes a SOCKS5 tunnel via the laptop host which has VPN access.
    Uses SSH ControlMaster for connection reuse.

    Returns:
        True if tunnel is available, False otherwise
    """
    global _socks_tunnel_checked, _socks_tunnel_available

    if _socks_tunnel_checked:
        return _socks_tunnel_available

    _socks_tunnel_checked = True

    # First check if SOCKS port is already bound locally
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", _SOCKS_PORT))
        sock.close()
        if result == 0:
            # Port is in use - assume our tunnel is running
            _socks_tunnel_available = True
            logger.debug("SOCKS tunnel already active on port %d", _SOCKS_PORT)
            return True
    except Exception:
        pass

    # Port not bound - try to establish tunnel via laptop
    # Check if laptop host is reachable (reverse tunnel must be active)
    try:
        result = subprocess.run(
            ["ssh", "-O", "check", "laptop"],
            capture_output=True,
            timeout=5,
        )
        laptop_reachable = result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        laptop_reachable = False

    if not laptop_reachable:
        # Try to establish connection (will fail if reverse tunnel not active)
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-f", "-N", "laptop"],
                capture_output=True,
                timeout=15,
            )
            laptop_reachable = result.returncode == 0
        except Exception:
            laptop_reachable = False

    if not laptop_reachable:
        logger.debug("Laptop reverse tunnel not available")
        _socks_tunnel_available = False
        return False

    # Start SOCKS tunnel
    try:
        result = subprocess.run(
            ["ssh", "-D", str(_SOCKS_PORT), "-f", "-N", "laptop"],
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            _socks_tunnel_available = True
            logger.info("Started SOCKS tunnel on port %d via laptop", _SOCKS_PORT)
            return True
        elif b"Address already in use" in result.stderr:
            # Another process using the port, assume it's our tunnel
            _socks_tunnel_available = True
            return True
    except Exception as e:
        logger.debug("Failed to start SOCKS tunnel: %s", e)

    _socks_tunnel_available = False
    return False


def _fetch_html_via_socks(url: str, timeout: int = 15) -> str | None:
    """Fetch HTML via SOCKS proxy through laptop.

    Routes traffic: iter → localhost:SOCKS_PORT → laptop → VPN → target.
    Much faster than SSH double-hop (~1.5s vs ~17s).

    Args:
        url: URL to fetch
        timeout: Curl timeout in seconds

    Returns:
        HTML content string, or None on failure
    """
    if not _ensure_socks_tunnel():
        return None

    cmd = [
        "curl",
        "-sk",
        "--socks5-hostname",
        f"localhost:{_SOCKS_PORT}",
        "--connect-timeout",
        str(timeout),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 5)
        if result.returncode != 0:
            logger.debug("SOCKS curl failed for %s (exit %d)", url, result.returncode)
            return None
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        logger.debug("SOCKS curl timed out for %s", url)
        return None
    except Exception as e:
        logger.debug("SOCKS curl error for %s: %s", url, e)
        return None


def _fetch_html_via_ssh(url: str, ssh_host: str, timeout: int = 60) -> str | None:
    """Fetch HTML content via SSH-proxied curl.

    Used as fallback when SOCKS tunnel is unavailable. Routes traffic
    via SSH to the facility host which then makes the HTTP request.
    Slower than SOCKS (~17s vs ~1.5s) due to SSH double-hop.

    Args:
        url: URL to fetch
        ssh_host: SSH host to proxy through
        timeout: Command timeout in seconds (default 60s to accommodate
                 multi-hop SSH chains like iter->laptop->facility)

    Returns:
        HTML content string, or None on failure
    """
    cmd = f'curl -sk --noproxy "*" --connect-timeout 20 "{url}"'
    try:
        result = subprocess.run(
            ["ssh", ssh_host, cmd],
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning("SSH curl failed for %s (exit %d)", url, result.returncode)
            return None
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        logger.warning("SSH curl timed out for %s via %s", url, ssh_host)
        return None
    except Exception as e:
        logger.warning("SSH curl error for %s: %s", url, e)
        return None


def _fetch_html(
    url: str,
    ssh_host: str | None = None,
    access_method: str = "direct",
    timeout: float = 10.0,
) -> str | None:
    """Fetch HTML with automatic strategy selection.

    Strategy depends on access_method:
    - "direct": Only try direct HTTP (for auth-protected sites)
    - "vpn" / "tunnel": Try direct, then SOCKS proxy, then SSH fallback

    Args:
        url: URL to fetch
        ssh_host: SSH host for fallback proxy access
        access_method: "direct", "vpn", or "tunnel"
        timeout: Direct HTTP timeout in seconds

    Returns:
        HTML content string, or None if all methods fail
    """
    needs_proxy = access_method in ("vpn", "tunnel")

    # For VPN/tunnel sites, use a short timeout for direct — the server is
    # almost certainly unreachable from the workstation.  Saves ~10s per page.
    direct_timeout = 2.0 if needs_proxy else timeout
    html = _fetch_html_direct(url, timeout=direct_timeout)
    if html is not None:
        return html

    # For VPN/tunnel-protected sites, try proxy methods
    if needs_proxy:
        # Try SOCKS proxy via laptop — fast path when on iter
        html = _fetch_html_via_socks(url)
        if html is not None:
            return html

        # Fall back to SSH proxy if configured — slow but reliable
        if ssh_host:
            logger.debug(
                "SOCKS unavailable, falling back to SSH proxy via %s", ssh_host
            )
            return _fetch_html_via_ssh(url, ssh_host)

    return None


class TWikiStaticAdapter(WikiAdapter):
    """Adapter for static TWiki HTML exports.

    Used for pre-rendered TWiki exports served as static HTML (e.g., JT-60SA).
    These exports include utility pages like WebTopicList.html that provide
    a complete manifest of all topics without requiring crawling.

    Page discovery: WebTopicList.html (bullet list of all topics)
    Artifact discovery: Parse topic pages for linked files
    """

    site_type = "twiki_static"

    def __init__(
        self,
        base_url: str | None = None,
        ssh_host: str | None = None,
        access_method: str = "direct",
        pub_path: str | None = None,
    ):
        """Initialize TWiki static adapter.

        Args:
            base_url: Base URL of the static TWiki export
            ssh_host: SSH host for proxied access (only used when access_method="vpn")
            access_method: "direct" (auth-protected) or "vpn" (requires proxy)
            pub_path: Absolute path to the static export's resource directory.
                When set, artifact discovery uses fd over SSH instead of
                curl+rg page scraping.
        """
        self._base_url = base_url
        self._ssh_host = ssh_host
        self._access_method = access_method
        self._pub_path = pub_path.rstrip("/") if pub_path else None

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via WebTopicList.html.

        TWiki static exports include a WebTopicList.html page that contains
        a bullet list of all topics. This provides complete coverage without
        requiring crawling.

        When ssh_host is configured, fetches via SSH-proxied curl instead
        of direct HTTP (needed when the URL is only reachable from the
        facility host).

        Args:
            facility: Facility ID
            base_url: Base URL of the static TWiki export
            on_progress: Progress callback (message, stats)

        Returns:
            List of discovered pages
        """
        from bs4 import BeautifulSoup

        pages: list[DiscoveredPage] = []
        effective_base_url = base_url or self._base_url
        if not effective_base_url:
            logger.warning("No base URL configured for TWiki static adapter")
            return []

        # Normalize URL by stripping trailing slash
        effective_base_url = effective_base_url.rstrip("/")

        # WebTopicList.html contains a bullet list of all topics
        topic_list_url = f"{effective_base_url}/WebTopicList.html"

        try:
            if on_progress:
                on_progress("fetching WebTopicList.html", None)

            # Fetch HTML: try direct first, fall back to SSH proxy
            html_text = _fetch_html(
                topic_list_url,
                ssh_host=self._ssh_host,
                access_method=self._access_method,
            )
            if html_text is None:
                logger.warning("Failed to fetch WebTopicList.html")
                return []

            soup = BeautifulSoup(html_text, "html.parser")

            # TWiki exports WebTopicList as a <ul> with one <li><a> per topic
            # Links are like: <a href="TopicName.html">TopicName</a>
            for link in soup.select("ul li a"):
                href = link.get("href", "")
                name = link.get_text(strip=True)

                # Skip utility pages (Web* pages are TWiki system pages)
                if name.startswith("Web"):
                    continue

                # Skip external links
                if href.startswith(("http://", "https://")):
                    if not href.startswith(effective_base_url):
                        continue

                # Build full URL for the topic
                if href.endswith(".html"):
                    if href.startswith("/"):
                        # Absolute path
                        url = f"{effective_base_url.rsplit('/', 1)[0]}{href}"
                    elif href.startswith("http"):
                        url = href
                    else:
                        # Relative path
                        url = f"{effective_base_url}/{href}"
                    pages.append(DiscoveredPage(name=name, url=url))

            if on_progress:
                on_progress(f"discovered {len(pages)} pages", None)

        except Exception as e:
            logger.warning(f"Error during TWiki static page discovery: {e}")

        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts by scanning topic pages for linked files.

        When SSH is available, uses a single SSH command with server-side
        curl + rg to scan all pages in one shot (seconds instead of minutes).
        Falls back to per-page fetching when SSH is not available.

        Args:
            facility: Facility ID
            base_url: Base URL of the static TWiki export
            on_progress: Progress callback (message, stats)

        Returns:
            List of discovered artifacts
        """
        effective_base_url = (base_url or self._base_url or "").rstrip("/")
        if not effective_base_url:
            return []

        # Fastest path: fd scan of the resource directory on disk
        if self._pub_path and self._ssh_host:
            return self._discover_artifacts_via_fd(effective_base_url, on_progress)

        # Fast path: single SSH command with server-side curl + rg
        if self._ssh_host:
            pages = self.bulk_discover_pages(facility, base_url, on_progress=None)
            if not pages:
                return []
            if on_progress:
                on_progress(
                    f"scanning {len(pages)} pages via SSH+rg (single connection)", None
                )
            return _discover_artifacts_via_ssh_rg(
                pages, effective_base_url, self._ssh_host, on_progress
            )

        # Slow fallback: per-page HTTP fetch (no SSH available)
        pages = self.bulk_discover_pages(facility, base_url, on_progress=None)
        if not pages:
            return []
        return self._discover_artifacts_per_page(pages, effective_base_url, on_progress)

    def _discover_artifacts_via_fd(
        self,
        effective_base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts by scanning the resource directory via fd over SSH.

        The static export stores resources in rsrc/<web>/<topic>/<filename>,
        which maps directly to topic linkage.

        Args:
            effective_base_url: Base URL for building web-accessible URLs
            on_progress: Progress callback

        Returns:
            List of discovered artifacts with topic linkage
        """
        ext_args = " ".join(f"-e {ext.lstrip('.')}" for ext in _ARTIFACT_EXTENSIONS)
        cmd = f"fd {ext_args} . {shlex.quote(self._pub_path)}"

        if on_progress:
            on_progress(f"scanning {self._pub_path} via fd", None)

        logger.info(
            "Running fd artifact scan: %s via %s", self._pub_path, self._ssh_host
        )

        try:
            result = subprocess.run(
                ["ssh", self._ssh_host, cmd],
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            logger.warning("fd scan timed out for %s", self._pub_path)
            return []
        except Exception as e:
            logger.warning("fd scan failed for %s: %s", self._pub_path, e)
            return []

        if result.returncode not in (0, 1):
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            if stderr:
                logger.warning("fd scan stderr: %s", stderr[:200])
            return []

        output = result.stdout.decode("utf-8", errors="replace")
        artifacts: list[DiscoveredArtifact] = []

        for line in output.strip().split("\n"):
            if not line:
                continue

            filepath = line.strip()
            # Extract web/topic/filename from rsrc/<web>/<topic>/<filename>
            rel_path = filepath.replace(f"{self._pub_path}/", "")
            parts = rel_path.split("/")
            if len(parts) < 2:
                continue

            # First part is the web name, second is the topic
            topic_name = parts[1] if len(parts) >= 3 else parts[0]
            filename = parts[-1]
            artifact_type = _get_artifact_type_from_filename(filename)

            # Build web-accessible URL: base_url/rsrc/<rel_path>
            artifact_url = f"{effective_base_url}/rsrc/{rel_path}"

            artifact = DiscoveredArtifact(
                filename=filename,
                url=artifact_url,
                artifact_type=artifact_type,
            )
            artifact.linked_pages.append(topic_name)
            artifacts.append(artifact)

        if on_progress:
            on_progress(f"discovered {len(artifacts)} artifacts", None)

        logger.info(
            "fd scan complete: %d artifacts from %s", len(artifacts), self._pub_path
        )
        return artifacts

    def _discover_artifacts_per_page(
        self,
        pages: list[DiscoveredPage],
        effective_base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Fallback: fetch each page individually to find artifact links."""
        from bs4 import BeautifulSoup

        artifacts: list[DiscoveredArtifact] = []
        seen_urls: set[str] = set()

        if on_progress:
            on_progress(f"scanning {len(pages)} pages for artifacts", None)

        try:
            for i, page in enumerate(pages):
                if not page.url:
                    continue

                html_text = _fetch_html(
                    page.url,
                    ssh_host=self._ssh_host,
                    access_method=self._access_method,
                )
                if html_text is None:
                    continue

                soup = BeautifulSoup(html_text, "html.parser")

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    href_lower = href.lower()

                    if any(href_lower.endswith(ext) for ext in _ARTIFACT_EXTENSIONS):
                        if href.startswith("http"):
                            artifact_url = href
                        elif href.startswith("/"):
                            artifact_url = (
                                f"{effective_base_url.rsplit('/', 1)[0]}{href}"
                            )
                        else:
                            artifact_url = f"{effective_base_url}/{href}"

                        if artifact_url in seen_urls:
                            continue
                        seen_urls.add(artifact_url)

                        filename = artifact_url.split("/")[-1]
                        artifact_type = self._get_artifact_type(filename)

                        artifact = DiscoveredArtifact(
                            filename=filename,
                            url=artifact_url,
                            artifact_type=artifact_type,
                        )
                        artifact.linked_pages.append(page.name)
                        artifacts.append(artifact)

                if on_progress and (i + 1) % 5 == 0:
                    on_progress(
                        f"scanned {i + 1}/{len(pages)} pages, "
                        f"found {len(artifacts)} artifacts",
                        None,
                    )

            if on_progress:
                on_progress(f"discovered {len(artifacts)} artifacts", None)

        except Exception as e:
            logger.warning(f"Error during TWiki static artifact discovery: {e}")

        return artifacts

    def _get_artifact_type(self, filename: str) -> str:
        """Get artifact type from filename."""
        return _get_artifact_type_from_filename(filename)


# =============================================================================
# Shared artifact discovery helpers
# =============================================================================

_ARTIFACT_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".ipynb",
    ".h5",
    ".hdf5",
    ".mat",
)


def _discover_artifacts_via_ssh_rg(
    pages: list[DiscoveredPage],
    base_url: str,
    ssh_host: str,
    on_progress: Callable[[str, Any], None] | None = None,
    timeout: int = 300,
) -> list[DiscoveredArtifact]:
    """Discover artifacts across all pages using a single SSH command.

    Pipes page name/URL pairs into a server-side script that curls each
    page locally (loopback — near-instant) and extracts artifact links
    via rg. This replaces N individual SSH+curl calls with 1 SSH call.

    Performance: ~5-10s for 272 pages vs ~5-15 minutes with per-page SSH.

    Args:
        pages: List of discovered pages with URLs
        base_url: Base URL for resolving relative hrefs
        ssh_host: SSH host that can reach the URLs locally
        on_progress: Progress callback
        timeout: SSH command timeout in seconds

    Returns:
        List of discovered artifacts with page linkage
    """
    if not pages:
        return []

    # Build tab-separated input: name\turl
    page_lines = []
    for page in pages:
        if page.url:
            page_lines.append(f"{page.name}\t{page.url}")

    if not page_lines:
        return []

    page_input = "\n".join(page_lines)

    # Server-side script: read name/url pairs from stdin, curl each locally,
    # extract artifact hrefs. rg is used for fast regex; grep -oP as fallback.
    # Each match is prefixed with PAGE:name for page→artifact mapping.
    remote_script = r"""
while IFS=$'\t' read -r name url; do
  html=$(curl -sk --noproxy '*' --max-time 5 "$url" 2>/dev/null) || continue
  matches=$(echo "$html" | rg -oNI 'href="[^"]*\.(pdf|doc|docx|ppt|pptx|xls|xlsx|ipynb|h5|hdf5|mat)(\?[^"]*)?(\#[^"]*)?"' 2>/dev/null \
         || echo "$html" | grep -oP 'href="[^"]*\.(pdf|doc|docx|ppt|pptx|xls|xlsx|ipynb|h5|hdf5|mat)(\?[^"]*)?(\#[^"]*)?"' 2>/dev/null)
  if [ -n "$matches" ]; then
    while IFS= read -r match; do
      echo "PAGE:${name}	${match}"
    done <<< "$matches"
  fi
done
"""

    # Combine: page data followed by script, using heredoc to separate them
    # We pipe the page list as stdin data and embed the script in the command
    ssh_command = f"bash -c {shlex.quote(remote_script)}"

    logger.info(
        "Running SSH+rg artifact scan: %d pages via %s", len(page_lines), ssh_host
    )

    try:
        result = subprocess.run(
            ["ssh", ssh_host, ssh_command],
            input=page_input.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "SSH+rg artifact scan timed out after %ds for %s", timeout, ssh_host
        )
        return []
    except Exception as e:
        logger.warning("SSH+rg artifact scan failed: %s", e)
        return []

    if result.returncode not in (0, 1):  # rg returns 1 for no matches
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if stderr:
            logger.warning("SSH+rg artifact scan stderr: %s", stderr[:200])

    # Parse output: "PAGE:TopicName\thref="path/to/file.pdf""
    output = result.stdout.decode("utf-8", errors="replace")
    artifacts: list[DiscoveredArtifact] = []
    seen_urls: set[str] = set()
    parsed_base = urllib.parse.urlparse(base_url)

    for line in output.strip().split("\n"):
        if not line or "\t" not in line:
            continue

        page_part, href_part = line.split("\t", 1)
        page_name = page_part.removeprefix("PAGE:")

        # Extract URL from href="..."
        match = re.search(r'href="([^"]+)"', href_part)
        if not match:
            continue
        href = match.group(1)

        # Strip query/fragment for deduplication
        href_clean = re.sub(r"[?#].*$", "", href)

        # Resolve to absolute URL
        if href_clean.startswith("http"):
            artifact_url = href_clean
        elif href_clean.startswith("/"):
            artifact_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href_clean}"
        else:
            artifact_url = f"{base_url}/{href_clean}"

        if artifact_url in seen_urls:
            # Still add page linkage to existing artifact
            for a in artifacts:
                if a.url == artifact_url and page_name not in a.linked_pages:
                    a.linked_pages.append(page_name)
                    break
            continue
        seen_urls.add(artifact_url)

        filename = artifact_url.split("/")[-1]
        artifact_type = _get_artifact_type_from_filename(filename)

        artifact = DiscoveredArtifact(
            filename=filename,
            url=artifact_url,
            artifact_type=artifact_type,
        )
        artifact.linked_pages.append(page_name)
        artifacts.append(artifact)

    if on_progress:
        on_progress(f"discovered {len(artifacts)} artifacts", None)

    logger.info(
        "SSH+rg artifact scan complete: %d artifacts from %d pages",
        len(artifacts),
        len(page_lines),
    )
    return artifacts


def _get_artifact_type_from_filename(filename: str) -> str:
    """Get artifact type from filename extension.

    Shared utility for all adapters that discover artifacts.
    Returns semantic type names matching ArtifactType enum values.
    """
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return "pdf"
    if filename_lower.endswith((".doc", ".docx")):
        return "document"
    if filename_lower.endswith((".ppt", ".pptx")):
        return "presentation"
    if filename_lower.endswith((".xls", ".xlsx")):
        return "spreadsheet"
    if filename_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
        return "image"
    if filename_lower.endswith(".ipynb"):
        return "notebook"
    if filename_lower.endswith(".json"):
        return "json"
    if filename_lower.endswith((".h5", ".hdf5", ".mat")):
        return "data"
    return "document"


class StaticHtmlAdapter(WikiAdapter):
    """Adapter for static HTML documentation sites.

    Discovers pages by breadth-first crawling from a portal page,
    following same-origin HTML links. Excludes paths that belong to
    other wiki_sites entries (e.g., /twiki_html/ paths when crawling
    the server landing page).

    Page discovery: BFS crawl from portal page
    Artifact discovery: Extract PDF/document links during crawl
    """

    site_type = "static_html"

    # Depth limit for BFS crawl (primary control)
    DEFAULT_MAX_DEPTH = 3
    # Maximum pages to crawl (secondary safety net)
    DEFAULT_MAX_PAGES = 500

    def __init__(
        self,
        base_url: str | None = None,
        exclude_prefixes: list[str] | None = None,
        ssh_host: str | None = None,
        access_method: str = "direct",
        max_depth: int | None = None,
        max_pages: int | None = None,
    ):
        """Initialize static HTML adapter.

        Args:
            base_url: Base URL of the static site
            exclude_prefixes: URL path prefixes to exclude from crawling
                (e.g., ["/twiki_html"] to avoid overlapping with a TWiki site)
            ssh_host: SSH host for proxied access (only used when access_method="vpn")
            access_method: "direct" (auth-protected) or "vpn" (requires proxy)
            max_depth: Maximum BFS depth from portal page (default: 3).
                Pages at depth 0 are seed URLs, depth 1 are directly linked
                from seeds, etc. This is the primary crawl control — it keeps
                discovery focused near the portal page.
            max_pages: Maximum pages to crawl (default: 500). Secondary safety
                net to prevent runaway BFS even within the depth limit.
        """
        self._base_url = base_url
        self._exclude_prefixes = exclude_prefixes or []
        self._ssh_host = ssh_host
        self._access_method = access_method
        self._max_depth = max_depth if max_depth is not None else self.DEFAULT_MAX_DEPTH
        self._max_pages = max_pages or self.DEFAULT_MAX_PAGES
        self._cached_pages: list[DiscoveredPage] | None = None

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover pages by BFS crawling from the portal page.

        Follows same-origin links to .html pages, respecting
        exclude_prefixes to avoid overlapping with other wiki sites.

        When ssh_host is configured, fetches via SSH-proxied curl instead
        of direct HTTP (needed when the URL is only reachable from the
        facility host).

        Args:
            facility: Facility ID
            base_url: Base URL (portal page URL)
            on_progress: Progress callback

        Returns:
            List of discovered pages
        """
        from bs4 import BeautifulSoup

        effective_base_url = base_url or self._base_url
        if not effective_base_url:
            logger.warning("No base URL configured for static HTML adapter")
            return []

        effective_base_url = effective_base_url.rstrip("/")
        parsed_base = urllib.parse.urlparse(effective_base_url)
        origin = f"{parsed_base.scheme}://{parsed_base.netloc}"

        # BFS state — queue tracks (url, depth) tuples
        seen_urls: set[str] = set()
        queue: list[tuple[str, int]] = []
        pages: list[DiscoveredPage] = []

        # Seed with common entry points at depth 0
        portal_candidates = [
            f"{effective_base_url}/index.html",
            f"{effective_base_url}/index-en.html",
            effective_base_url,
        ]
        for url in portal_candidates:
            if url not in seen_urls:
                queue.append((url, 0))
                seen_urls.add(url)

        # Log access method (only for VPN sites that need proxying)
        if self._access_method == "vpn":
            if _ensure_socks_tunnel():
                logger.info("Using SOCKS proxy for BFS crawl")
            elif self._ssh_host:
                logger.info("Using SSH proxy via %s for BFS crawl", self._ssh_host)

        max_depth_reached = False
        try:
            while queue and len(pages) < self._max_pages:
                current_url, depth = queue.pop(0)

                html_text = _fetch_html(
                    current_url,
                    ssh_host=self._ssh_host,
                    access_method=self._access_method,
                )
                if html_text is None:
                    continue

                # Extract page name from URL path
                parsed = urllib.parse.urlparse(current_url)
                path = parsed.path.rstrip("/")
                name = path.split("/")[-1] if path else "index.html"

                pages.append(DiscoveredPage(name=name, url=current_url))

                # Only follow links if we haven't reached max depth
                if depth >= self._max_depth:
                    max_depth_reached = True
                    continue

                # Parse links
                soup = BeautifulSoup(html_text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]

                    # Resolve relative URLs
                    full_url = urllib.parse.urljoin(current_url, href)

                    # Strip fragment
                    full_url = urllib.parse.urldefrag(full_url)[0]

                    # Same origin only
                    parsed_link = urllib.parse.urlparse(full_url)
                    if f"{parsed_link.scheme}://{parsed_link.netloc}" != origin:
                        continue

                    # Must be HTML (not PDF, images, etc.)
                    link_path = parsed_link.path.lower()
                    if not (
                        link_path.endswith(".html")
                        or link_path.endswith(".htm")
                        or link_path.endswith("/")
                        or "." not in link_path.split("/")[-1]
                    ):
                        continue

                    # Exclude paths belonging to other wiki sites
                    if any(
                        parsed_link.path.startswith(prefix)
                        for prefix in self._exclude_prefixes
                    ):
                        continue

                    # Skip CGI/dynamic paths
                    if "/cgi/" in parsed_link.path or "/cgi-bin/" in parsed_link.path:
                        continue

                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        queue.append((full_url, depth + 1))

                if on_progress and len(pages) % 10 == 0:
                    on_progress(
                        f"crawled {len(pages)} pages (depth {depth}), "
                        f"{len(queue)} queued",
                        None,
                    )

            if len(pages) >= self._max_pages:
                logger.info(
                    "BFS crawl stopped at max_pages=%d (queue had %d remaining)",
                    self._max_pages,
                    len(queue),
                )
            if max_depth_reached:
                logger.info(
                    "BFS crawl reached max_depth=%d, discovered %d pages",
                    self._max_depth,
                    len(pages),
                )

        except Exception as e:
            logger.warning(f"Error during static HTML discovery: {e}")

        if on_progress:
            on_progress(f"discovered {len(pages)} pages", None)

        # Cache for reuse in bulk_discover_artifacts (avoids re-crawling)
        self._cached_pages = pages
        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts by scanning discovered pages for file links.

        When SSH is available, uses a single SSH command with server-side
        curl + rg to scan all pages in one shot. Falls back to per-page
        fetching when SSH is not available.

        Args:
            facility: Facility ID
            base_url: Base URL of the static site
            on_progress: Progress callback (message, stats)

        Returns:
            List of discovered artifacts
        """
        effective_base_url = (base_url or self._base_url or "").rstrip("/")
        if not effective_base_url:
            return []

        # Use cached pages from prior bulk_discover_pages call if available
        pages = self._cached_pages or self.bulk_discover_pages(
            facility, base_url, on_progress=None
        )
        if not pages:
            return []

        # Fast path: single SSH command with server-side curl + rg
        if self._ssh_host:
            if on_progress:
                on_progress(
                    f"scanning {len(pages)} pages via SSH+rg (single connection)", None
                )
            return _discover_artifacts_via_ssh_rg(
                pages, effective_base_url, self._ssh_host, on_progress
            )

        # Slow fallback: per-page HTTP fetch (no SSH available)
        return self._discover_artifacts_per_page(pages, effective_base_url, on_progress)

    def _discover_artifacts_per_page(
        self,
        pages: list[DiscoveredPage],
        effective_base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Fallback: fetch each page individually to find artifact links."""
        from bs4 import BeautifulSoup

        artifacts: list[DiscoveredArtifact] = []
        seen_urls: set[str] = set()

        if on_progress:
            on_progress(f"scanning {len(pages)} pages for artifacts", None)

        try:
            for i, page in enumerate(pages):
                if not page.url:
                    continue

                html_text = _fetch_html(
                    page.url,
                    ssh_host=self._ssh_host,
                    access_method=self._access_method,
                )
                if html_text is None:
                    continue

                soup = BeautifulSoup(html_text, "html.parser")

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    href_lower = href.lower()

                    if any(href_lower.endswith(ext) for ext in _ARTIFACT_EXTENSIONS):
                        if href.startswith("http"):
                            artifact_url = href
                        elif href.startswith("/"):
                            parsed_base = urllib.parse.urlparse(effective_base_url)
                            artifact_url = (
                                f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                            )
                        else:
                            artifact_url = urllib.parse.urljoin(page.url, href)

                        if artifact_url in seen_urls:
                            continue
                        seen_urls.add(artifact_url)

                        filename = artifact_url.split("/")[-1]
                        artifact_type = _get_artifact_type_from_filename(filename)

                        artifact = DiscoveredArtifact(
                            filename=filename,
                            url=artifact_url,
                            artifact_type=artifact_type,
                        )
                        artifact.linked_pages.append(page.name)
                        artifacts.append(artifact)

                if on_progress and (i + 1) % 5 == 0:
                    on_progress(
                        f"scanned {i + 1}/{len(pages)} pages, "
                        f"found {len(artifacts)} artifacts",
                        None,
                    )

            if on_progress:
                on_progress(f"discovered {len(artifacts)} artifacts", None)

        except Exception as e:
            logger.warning(f"Error during static HTML artifact discovery: {e}")

        return artifacts


class ConfluenceAdapter(WikiAdapter):
    """Adapter for Confluence sites.

    Page discovery: GET /rest/api/content?spaceKey=X
    Artifact discovery: GET /rest/api/content/{pageId}/child/attachment
    """

    site_type = "confluence"

    def __init__(
        self,
        api_token: str | None = None,
        ssh_host: str | None = None,
        session: Any = None,
        space_key: str | None = None,
    ):
        """Initialize Confluence adapter.

        Args:
            api_token: API token for REST API authentication
            ssh_host: SSH host for proxied commands (alternative to token)
            session: Authenticated requests.Session for direct HTTP
            space_key: Confluence space key to scope discovery (e.g. "IMP").
                When set, only pages from this space are discovered.
                Without this, ALL pages from the entire Confluence site are returned.
        """
        self.api_token = api_token
        self.ssh_host = ssh_host
        self.session = session
        self.space_key = space_key

    def _fetch_page_batch(
        self,
        api_url: str,
        start: int,
        limit: int,
    ) -> dict[str, Any] | None:
        """Fetch a batch of pages from the Confluence REST API.

        Args:
            api_url: Base API content URL
            start: Pagination offset
            limit: Page size

        Returns:
            JSON response dict or None on failure
        """
        import json

        params = f"type=page&start={start}&limit={limit}&expand=space"
        if self.space_key:
            params += f"&spaceKey={self.space_key}"
        url = f"{api_url}?{params}"

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
                    logger.warning(
                        "SSH curl failed (rc=%d): %s",
                        result.returncode,
                        result.stderr[:200],
                    )
                    return None
                return json.loads(result.stdout)
            except Exception as e:
                logger.warning("Error during Confluence SSH discovery: %s", e)
                return None

        if self.session:
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning("Error during Confluence HTTP discovery: %s", e)
                return None

        logger.warning(
            "Confluence discovery requires either ssh_host or an authenticated session"
        )
        return None

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all pages via Confluence REST API."""
        pages: list[DiscoveredPage] = []

        # Confluence REST API endpoint
        api_url = f"{base_url}/rest/api/content"
        start = 0
        limit = 100

        while True:
            data = self._fetch_page_batch(api_url, start, limit)
            if data is None:
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

    def _fetch_json(self, url: str) -> dict[str, Any] | None:
        """Fetch JSON from Confluence REST API.

        Uses the same auth strategy as _fetch_page_batch: SSH proxy
        or authenticated session.

        Args:
            url: Full API URL

        Returns:
            Parsed JSON dict, or None on failure
        """
        import json

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
                    return None
                return json.loads(result.stdout)
            except Exception:
                return None

        if self.session:
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception:
                return None

        return None

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover all attachments via Confluence REST API.

        Iterates through all pages and fetches attachments for each
        using GET /rest/api/content/{id}/child/attachment.

        Args:
            facility: Facility ID
            base_url: Confluence base URL
            on_progress: Progress callback

        Returns:
            List of discovered artifacts
        """
        # Step 1: Get all page IDs (reuse _fetch_page_batch)
        api_url = f"{base_url}/rest/api/content"
        page_ids: list[tuple[str, str]] = []  # (page_id, title)
        start = 0
        limit = 100

        if on_progress:
            on_progress("listing pages for attachment scan", None)

        while True:
            data = self._fetch_page_batch(api_url, start, limit)
            if data is None:
                break
            results = data.get("results", [])
            if not results:
                break
            for page in results:
                pid = page.get("id")
                title = page.get("title", "")
                if pid:
                    page_ids.append((pid, title))
            if data.get("size", 0) < limit:
                break
            start += limit

        if not page_ids:
            logger.info("No pages found for Confluence artifact discovery")
            return []

        logger.info("Scanning %d pages for attachments in %s", len(page_ids), base_url)

        # Step 2: Fetch attachments for each page
        artifacts: list[DiscoveredArtifact] = []
        seen_urls: set[str] = set()

        for i, (page_id, title) in enumerate(page_ids):
            att_url = (
                f"{base_url}/rest/api/content/{page_id}/child/attachment?limit=100"
            )
            data = self._fetch_json(att_url)
            if data is None:
                continue

            for att in data.get("results", []):
                att_title = att.get("title", "")
                download = att.get("_links", {}).get("download", "")
                if not download:
                    continue

                artifact_url = f"{base_url}{download}"
                if artifact_url in seen_urls:
                    # Add page linkage to existing artifact
                    for a in artifacts:
                        if a.url == artifact_url and title not in a.linked_pages:
                            a.linked_pages.append(title)
                            break
                    continue
                seen_urls.add(artifact_url)

                media_type = att.get("metadata", {}).get("mediaType", "")
                size = att.get("extensions", {}).get("fileSize", 0)
                artifact_type = _get_artifact_type_from_filename(att_title)

                artifact = DiscoveredArtifact(
                    filename=att_title,
                    url=artifact_url,
                    artifact_type=artifact_type,
                    size_bytes=size if size else None,
                    mime_type=media_type if media_type else None,
                )
                artifact.linked_pages.append(title)
                artifacts.append(artifact)

            if on_progress and (i + 1) % 50 == 0:
                on_progress(
                    f"scanned {i + 1}/{len(page_ids)} pages, "
                    f"found {len(artifacts)} attachments",
                    None,
                )

        if on_progress:
            on_progress(f"discovered {len(artifacts)} attachments", None)

        logger.info(
            "Confluence attachment discovery: %d artifacts from %d pages",
            len(artifacts),
            len(page_ids),
        )
        return artifacts


class TWikiRawAdapter(WikiAdapter):
    """Adapter for raw TWiki data directories accessed via SSH.

    Used when TWiki content exists as raw .txt markup files on the
    filesystem (accessible via SSH) but is not served via HTTP. This is
    common for TWiki "webs" that were never exported to static HTML.

    Page discovery: List *.txt files in data/<web>/ directory via SSH
    Artifact discovery: List files in pub/<web>/ directory via SSH
    """

    site_type = "twiki_raw"

    def __init__(
        self,
        ssh_host: str,
        data_path: str,
        pub_path: str | None = None,
        web_name: str = "Main",
        exclude_patterns: list[str] | None = None,
    ):
        """Initialize TWiki raw adapter.

        Args:
            ssh_host: SSH host for filesystem access
            data_path: Absolute path to TWiki data/<web>/ directory
            pub_path: Absolute path to TWiki pub/<web>/ directory (for artifacts)
            web_name: TWiki web name (e.g., "Main", "Code")
            exclude_patterns: Regex patterns for topic names to skip.
                Defaults to excluding system pages, watchlists, and TWiki internals.
        """
        self._ssh_host = ssh_host
        self._data_path = data_path.rstrip("/")
        self._pub_path = pub_path.rstrip("/") if pub_path else None
        self._web_name = web_name
        self._exclude_patterns = exclude_patterns or [
            r"^Web",  # TWiki system pages (WebHome, WebTopicList, etc.)
            r"Watchlist$",  # User watchlists
            r"^TWiki",  # TWiki admin pages
            r"Bookmarks$",  # User bookmarks
            r"Template$",  # Form templates
        ]
        self._compiled_excludes = [re.compile(p) for p in self._exclude_patterns]

    def _should_skip(self, topic_name: str) -> bool:
        """Check if a topic should be excluded from discovery."""
        return any(pat.search(topic_name) for pat in self._compiled_excludes)

    def bulk_discover_pages(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredPage]:
        """Discover all topics by listing .txt files via SSH.

        Args:
            facility: Facility ID
            base_url: Not used for filesystem access (kept for interface compat)
            on_progress: Progress callback

        Returns:
            List of discovered pages with ssh:// URLs for content retrieval
        """
        if on_progress:
            on_progress(f"listing {self._data_path}/*.txt via SSH", None)

        try:
            # Use find instead of ls *.txt to avoid ARG_MAX with many files
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ClearAllForwardings=yes",
                    self._ssh_host,
                    f'find {shlex.quote(self._data_path)} -maxdepth 1 -name "*.txt" -type f 2>/dev/null',
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "Failed to list TWiki data dir %s via %s",
                    self._data_path,
                    self._ssh_host,
                )
                return []

            filenames = (
                result.stdout.decode("utf-8", errors="replace").strip().split("\n")
            )
            filenames = [f.strip() for f in filenames if f.strip()]

        except subprocess.TimeoutExpired:
            logger.warning("Timeout listing %s via SSH", self._data_path)
            return []
        except Exception as e:
            logger.warning("Error listing TWiki data dir: %s", e)
            return []

        pages: list[DiscoveredPage] = []
        for filepath in filenames:
            # Extract topic name from /path/to/TopicName.txt
            name = filepath.rsplit("/", 1)[-1].removesuffix(".txt")

            if self._should_skip(name):
                continue

            # URL encodes the SSH file path for later retrieval
            url = f"ssh://{self._ssh_host}{filepath}"
            pages.append(DiscoveredPage(name=name, url=url, namespace=self._web_name))

        if on_progress:
            on_progress(f"discovered {len(pages)} pages", None)

        logger.info(
            "TWiki raw discovery: %d topics from %s (excluded %d)",
            len(pages),
            self._data_path,
            len(filenames) - len(pages),
        )
        return pages

    def bulk_discover_artifacts(
        self,
        facility: str,
        base_url: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> list[DiscoveredArtifact]:
        """Discover artifacts by listing pub directory via SSH.

        TWiki stores attachments in pub/<web>/<topic>/<filename>.
        Scans the pub directory tree for common artifact types.

        Args:
            facility: Facility ID
            base_url: Not used for filesystem access
            on_progress: Progress callback

        Returns:
            List of discovered artifacts
        """
        if not self._pub_path:
            logger.info(
                "No pub_path configured for TWiki raw adapter, skipping artifacts"
            )
            return []

        artifact_extensions = (
            ".pdf",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            ".ipynb",
            ".json",
            ".h5",
            ".hdf5",
            ".mat",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
        )

        if on_progress:
            on_progress(f"scanning {self._pub_path} for artifacts via SSH", None)

        try:
            # Use find to list all files in the pub directory
            ext_args = " -o ".join(f'-name "*{ext}"' for ext in artifact_extensions)
            cmd = f"find {self._pub_path} -type f \\( {ext_args} \\) 2>/dev/null"
            result = subprocess.run(
                ["ssh", self._ssh_host, cmd],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning("Failed to scan pub dir %s", self._pub_path)
                return []

            files = result.stdout.decode("utf-8", errors="replace").strip().split("\n")
            files = [f.strip() for f in files if f.strip()]

        except subprocess.TimeoutExpired:
            logger.warning("Timeout scanning pub dir %s", self._pub_path)
            return []
        except Exception as e:
            logger.warning("Error scanning pub dir: %s", e)
            return []

        artifacts: list[DiscoveredArtifact] = []
        for filepath in files:
            filename = filepath.rsplit("/", 1)[-1]
            artifact_type = _get_artifact_type_from_filename(filename)

            # Extract topic name from pub/<web>/<topic>/<filename>
            parts = filepath.replace(self._pub_path + "/", "").split("/")
            topic_name = parts[0] if len(parts) > 1 else ""

            artifact = DiscoveredArtifact(
                filename=filename,
                url=f"ssh://{self._ssh_host}{filepath}",
                artifact_type=artifact_type,
            )
            if topic_name:
                artifact.linked_pages.append(topic_name)
            artifacts.append(artifact)

        if on_progress:
            on_progress(f"discovered {len(artifacts)} artifacts", None)

        return artifacts


def fetch_twiki_raw_content(ssh_host: str, filepath: str) -> str | None:
    """Fetch raw TWiki markup content from a file via SSH.

    Used during ingestion to read .txt files from TWiki data directories.

    Args:
        ssh_host: SSH host
        filepath: Absolute path to the .txt file on the remote host

    Returns:
        Raw TWiki markup content, or None on failure
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ClearAllForwardings=yes",
                ssh_host,
                f"cat {shlex.quote(filepath)}",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="replace")
        logger.warning("Failed to read %s via SSH (rc=%d)", filepath, result.returncode)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Timeout reading %s via SSH", filepath)
        return None
    except Exception as e:
        logger.warning("Error reading TWiki file %s: %s", filepath, e)
        return None


def get_adapter(
    site_type: str,
    ssh_host: str | None = None,
    wiki_client: MediaWikiClient | None = None,
    credential_service: str | None = None,
    api_token: str | None = None,
    base_url: str | None = None,
    access_method: str = "direct",
    data_path: str | None = None,
    pub_path: str | None = None,
    web_name: str = "Main",
    exclude_patterns: list[str] | None = None,
    webs: list[str] | None = None,
    session: Any = None,
    exclude_prefixes: list[str] | None = None,
    max_depth: int | None = None,
    space_key: str | None = None,
) -> WikiAdapter:
    """Get the appropriate adapter for a wiki site type.

    Args:
        site_type: Type of wiki (mediawiki, twiki, twiki_static, twiki_raw,
            confluence, static_html)
        ssh_host: SSH host for proxied commands
        wiki_client: Authenticated MediaWikiClient (for Tequila)
        credential_service: Keyring service name
        api_token: API token (for Confluence)
        base_url: Base URL (for static sites)
        access_method: Access method ("direct" or "vpn")
        data_path: TWiki data directory path (for twiki_raw)
        pub_path: TWiki pub directory path (for twiki_raw)
        web_name: TWiki web name (for twiki_raw, default "Main")
        exclude_patterns: Topic name exclude patterns (for twiki_raw)
        webs: TWiki web names to discover (for twiki, default ["Main"])
        session: Pre-authenticated requests.Session (Keycloak, Basic auth)
        exclude_prefixes: URL path prefixes to exclude (for static_html)
        max_depth: BFS depth limit (for static_html, default 3)
        space_key: Confluence space key to scope discovery (for confluence)

    Returns:
        WikiAdapter instance for the site type
    """
    if site_type == "mediawiki":
        return MediaWikiAdapter(
            ssh_host=ssh_host,
            wiki_client=wiki_client,
            credential_service=credential_service,
            session=session,
        )
    elif site_type == "twiki":
        return TWikiAdapter(
            ssh_host=ssh_host, webs=webs, base_url=base_url, pub_path=pub_path
        )
    elif site_type == "twiki_static":
        return TWikiStaticAdapter(
            base_url=base_url,
            ssh_host=ssh_host,
            access_method=access_method,
            pub_path=pub_path,
        )
    elif site_type == "twiki_raw":
        if not ssh_host:
            raise ValueError("twiki_raw requires ssh_host")
        if not data_path:
            raise ValueError("twiki_raw requires data_path")
        return TWikiRawAdapter(
            ssh_host=ssh_host,
            data_path=data_path,
            pub_path=pub_path,
            web_name=web_name,
            exclude_patterns=exclude_patterns,
        )
    elif site_type == "static_html":
        return StaticHtmlAdapter(
            base_url=base_url,
            ssh_host=ssh_host,
            access_method=access_method,
            exclude_prefixes=exclude_prefixes,
            max_depth=max_depth,
        )
    elif site_type == "confluence":
        return ConfluenceAdapter(
            api_token=api_token,
            ssh_host=ssh_host,
            session=session,
            space_key=space_key,
        )
    else:
        raise ValueError(f"Unknown site type: {site_type}")
