"""Confluence REST API client for wiki scraping.

Provides authenticated access to Atlassian Confluence instances using
the REST API. Supports session-based authentication with cookie persistence.

The Confluence REST API is preferred over HTML scraping because:
- Structured JSON responses (no HTML parsing needed)
- Pagination support for large spaces
- Attachment metadata and download
- Better rate limiting handling

Example:
    from imas_codex.wiki.confluence import ConfluenceClient

    client = ConfluenceClient(
        base_url="https://confluence.iter.org",
        credential_service="iter-confluence",
    )

    # Authenticate (uses keyring credentials)
    client.authenticate()

    # Get space overview
    space = client.get_space("IMP")

    # List pages in space
    pages = client.get_space_pages("IMP", limit=100)

    # Get page content
    content = client.get_page_content(page_id)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from imas_codex.wiki.auth import CredentialManager, require_credentials

logger = logging.getLogger(__name__)

# Default request timeout
DEFAULT_TIMEOUT = 30

# Rate limiting: minimum seconds between requests
RATE_LIMIT_DELAY = 0.5


@dataclass
class ConfluencePage:
    """A Confluence page with content and metadata."""

    id: str
    title: str
    space_key: str
    url: str
    content_html: str = ""
    content_text: str = ""
    version: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    ancestors: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def page_name(self) -> str:
        """Page name for compatibility with WikiPage interface."""
        return self.title


@dataclass
class ConfluenceSpace:
    """A Confluence space."""

    key: str
    name: str
    description: str = ""
    url: str = ""
    page_count: int = 0


class ConfluenceClient:
    """REST API client for Atlassian Confluence.

    Handles authentication, session management, and API calls.
    Uses session cookies stored in keyring for persistence.
    """

    def __init__(
        self,
        base_url: str,
        credential_service: str,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize Confluence client.

        Args:
            base_url: Confluence base URL (e.g., "https://confluence.iter.org")
            credential_service: Keyring service name for credentials
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.credential_service = credential_service
        self.timeout = timeout

        self._session: requests.Session | None = None
        self._creds = CredentialManager()
        self._last_request_time = 0.0
        self._authenticated = False

    @property
    def api_url(self) -> str:
        """REST API base URL."""
        return f"{self.base_url}/rest/api"

    def _get_session(self) -> requests.Session:
        """Get or create requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )

            # Try to restore session cookies from keyring
            cookies = self._creds.get_session(self.credential_service)
            if cookies:
                # Clear existing cookies to avoid duplicates
                self._session.cookies.clear()
                self._session.cookies.update(cookies)
                logger.debug("Restored session cookies from keyring")

        return self._session

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Make an API request with rate limiting.

        Args:
            method: HTTP method
            endpoint: API endpoint (relative to api_url)
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.HTTPError: On API errors
        """
        self._rate_limit()
        session = self._get_session()

        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)

        logger.debug("%s %s", method, url)
        response = session.request(method, url, **kwargs)

        # Handle authentication errors
        if response.status_code == 401:
            logger.warning("Authentication required or session expired")
            self._authenticated = False
            raise requests.HTTPError("Authentication required", response=response)

        response.raise_for_status()
        return response

    def authenticate(self, force: bool = False) -> bool:
        """Authenticate with Confluence.

        Attempts to use existing session, falls back to credentials.

        Args:
            force: Force re-authentication even if session exists

        Returns:
            True if authenticated successfully
        """
        session = self._get_session()

        # Check if existing session is valid
        if not force and self._authenticated:
            return True

        if not force:
            try:
                # Test session with a simple API call
                response = session.get(
                    f"{self.api_url}/user/current",
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    user = response.json()
                    logger.info(
                        "Session valid for user: %s",
                        user.get("displayName", user.get("username")),
                    )
                    self._authenticated = True
                    return True
            except Exception as e:
                logger.debug("Session check failed: %s", e)

        # Need to authenticate with credentials
        username, password = require_credentials(self.credential_service)

        # Confluence uses session-based auth via login page
        # Try basic auth first (works for API tokens)
        session.auth = (username, password)

        try:
            response = session.get(
                f"{self.api_url}/user/current",
                timeout=self.timeout,
            )
            if response.status_code == 200:
                user = response.json()
                logger.info(
                    "Authenticated as: %s",
                    user.get("displayName", user.get("username")),
                )

                # Save session cookies (clear duplicates first)
                # Convert to dict to deduplicate cookies by name
                cookie_dict = {}
                for cookie in session.cookies:
                    cookie_dict[cookie.name] = cookie.value

                self._creds.set_session(
                    self.credential_service,
                    cookie_dict,
                )
                self._authenticated = True
                return True
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                logger.error("Authentication failed: invalid credentials")
            else:
                logger.error("Authentication failed: %s", e)
            return False

        return False

    def get_space(self, space_key: str) -> ConfluenceSpace | None:
        """Get space information.

        Args:
            space_key: Space key (e.g., "IMP")

        Returns:
            ConfluenceSpace or None if not found
        """
        if not self.authenticate():
            return None

        try:
            response = self._request("GET", f"space/{space_key}")
            data = response.json()

            return ConfluenceSpace(
                key=data["key"],
                name=data["name"],
                description=data.get("description", {})
                .get("plain", {})
                .get("value", ""),
                url=f"{self.base_url}/spaces/{space_key}/overview",
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning("Space not found: %s", space_key)
            else:
                logger.error("Failed to get space: %s", e)
            return None

    def get_space_pages(
        self,
        space_key: str,
        limit: int = 100,
        start: int = 0,
        expand: str = "ancestors,version",
    ) -> list[ConfluencePage]:
        """Get pages in a space.

        Args:
            space_key: Space key
            limit: Maximum pages to return
            start: Starting index for pagination
            expand: Fields to expand in response

        Returns:
            List of ConfluencePage objects
        """
        if not self.authenticate():
            return []

        pages = []
        try:
            response = self._request(
                "GET",
                "content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "limit": min(limit, 100),  # API max is 100
                    "start": start,
                    "expand": expand,
                },
            )
            data = response.json()

            for item in data.get("results", []):
                page = ConfluencePage(
                    id=item["id"],
                    title=item["title"],
                    space_key=space_key,
                    url=f"{self.base_url}{item['_links']['webui']}",
                    version=item.get("version", {}).get("number", 1),
                    ancestors=[a["id"] for a in item.get("ancestors", [])],
                )
                pages.append(page)

            logger.info("Retrieved %d pages from space %s", len(pages), space_key)

        except requests.HTTPError as e:
            logger.error("Failed to get space pages: %s", e)

        return pages

    def get_page_basic_info(
        self,
        page_id: str,
    ) -> tuple[str, str] | None:
        """Get basic page info (title and space key) without fetching content.

        This is much faster than get_page_content for just getting titles.

        Args:
            page_id: Page ID

        Returns:
            (title, space_key) tuple or None if not found
        """
        if not self.authenticate():
            return None

        try:
            response = self._request(
                "GET",
                f"content/{page_id}",
                params={"expand": "space"},
            )
            data = response.json()

            title = data.get("title", "")
            space_key = data.get("space", {}).get("key", "")

            return (title, space_key)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning("Page not found: %s", page_id)
            else:
                logger.error("Failed to get page info: %s", e)
            return None

    def get_page_children(
        self,
        page_id: str,
        limit: int = 100,
    ) -> list[str]:
        """Get ALL child page IDs with pagination.

        The children.page expansion in get_page_content only returns
        the first page of results (default 25). This method paginates
        through all children.

        Args:
            page_id: Parent page ID
            limit: Results per page (max 100)

        Returns:
            List of all child page IDs
        """
        if not self.authenticate():
            return []

        all_children = []
        start = 0

        while True:
            try:
                response = self._request(
                    "GET",
                    f"content/{page_id}/child/page",
                    params={
                        "limit": min(limit, 100),
                        "start": start,
                    },
                )
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for child in results:
                    all_children.append(child["id"])

                # Check if there are more results
                if len(results) < limit or data.get("size", 0) < limit:
                    break

                start += limit

            except requests.HTTPError as e:
                logger.error("Failed to get children for page %s: %s", page_id, e)
                break

        return all_children

    def get_page_content(
        self,
        page_id: str,
        expand: str = "body.storage,version,ancestors,children.page,children.attachment",
    ) -> ConfluencePage | None:
        """Get full page content.

        Args:
            page_id: Page ID
            expand: Fields to expand

        Returns:
            ConfluencePage with content or None
        """
        if not self.authenticate():
            return None

        try:
            response = self._request(
                "GET",
                f"content/{page_id}",
                params={"expand": expand},
            )
            data = response.json()

            # Extract HTML content
            content_html = data.get("body", {}).get("storage", {}).get("value", "")

            # Simple HTML to text conversion
            content_text = re.sub(r"<[^>]+>", " ", content_html)
            content_text = re.sub(r"\s+", " ", content_text).strip()

            # Get space key from ancestors or _expandable
            space_key = ""
            if data.get("ancestors"):
                # Space is typically in the URL
                pass
            space_data = data.get("space", {})
            if isinstance(space_data, dict):
                space_key = space_data.get("key", "")

            # Extract attachments
            attachments = []
            children = data.get("children", {})
            if isinstance(children, dict):
                attachment_data = children.get("attachment", {})
                if isinstance(attachment_data, dict):
                    for att in attachment_data.get("results", []):
                        attachments.append(
                            {
                                "id": att["id"],
                                "title": att["title"],
                                "mediaType": att.get("metadata", {}).get(
                                    "mediaType", ""
                                ),
                                "downloadUrl": f"{self.base_url}{att['_links']['download']}",
                            }
                        )

            # Extract child page IDs
            child_pages = []
            if isinstance(children, dict):
                page_children = children.get("page", {})
                if isinstance(page_children, dict):
                    for child in page_children.get("results", []):
                        child_pages.append(child["id"])

            return ConfluencePage(
                id=data["id"],
                title=data["title"],
                space_key=space_key,
                url=f"{self.base_url}{data['_links']['webui']}",
                content_html=content_html,
                content_text=content_text,
                version=data.get("version", {}).get("number", 1),
                created_at=data.get("history", {}).get("createdDate"),
                updated_at=data.get("version", {}).get("when"),
                ancestors=[a["id"] for a in data.get("ancestors", [])],
                children=child_pages,
                attachments=attachments,
            )

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning("Page not found: %s", page_id)
            else:
                logger.error("Failed to get page content: %s", e)
            return None

    def search(
        self,
        query: str,
        space_key: str | None = None,
        limit: int = 50,
    ) -> list[ConfluencePage]:
        """Search for pages using CQL.

        Args:
            query: Search query (CQL or text)
            space_key: Limit to specific space
            limit: Maximum results

        Returns:
            List of matching pages
        """
        if not self.authenticate():
            return []

        # Build CQL query
        cql_parts = [f'text ~ "{query}"']
        if space_key:
            cql_parts.append(f'space = "{space_key}"')
        cql = " AND ".join(cql_parts)

        pages = []
        try:
            response = self._request(
                "GET",
                "content/search",
                params={
                    "cql": cql,
                    "limit": limit,
                },
            )
            data = response.json()

            for item in data.get("results", []):
                page = ConfluencePage(
                    id=item["id"],
                    title=item["title"],
                    space_key=item.get("space", {}).get("key", ""),
                    url=f"{self.base_url}{item['_links']['webui']}",
                )
                pages.append(page)

            logger.info("Search returned %d results", len(pages))

        except requests.HTTPError as e:
            logger.error("Search failed: %s", e)

        return pages

    def get_space_homepage(self, space_key: str) -> ConfluencePage | None:
        """Get the homepage of a space.

        Args:
            space_key: Space key

        Returns:
            Homepage ConfluencePage or None
        """
        if not self.authenticate():
            return None

        try:
            response = self._request(
                "GET",
                f"space/{space_key}",
                params={"expand": "homepage"},
            )
            data = response.json()

            homepage = data.get("homepage")
            if homepage:
                return self.get_page_content(homepage["id"])

        except requests.HTTPError as e:
            logger.error("Failed to get space homepage: %s", e)

        return None

    def close(self) -> None:
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self) -> "ConfluenceClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


def detect_site_type(url: str) -> str:
    """Detect wiki site type from URL.

    Args:
        url: Site URL

    Returns:
        Site type: "confluence", "mediawiki", or "generic"
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Confluence indicators
    if "confluence" in parsed.netloc.lower():
        return "confluence"
    if "/wiki/" in path and "/spaces/" in path:
        return "confluence"
    if "/rest/api" in path:
        return "confluence"

    # MediaWiki indicators
    if "/wiki/" in path and "index.php" not in path:
        # Could be either - check for MediaWiki-specific paths
        if "Special:" in url or "Portal:" in url:
            return "mediawiki"

    # Check for MediaWiki API
    if "/api.php" in path or "/w/api.php" in path:
        return "mediawiki"

    # Default to generic
    return "generic"
