"""MediaWiki client with Tequila (EPFL SSO) authentication.

Provides HTTP access to MediaWiki sites that use EPFL Tequila SSO.
Session cookies are stored in system keyring for persistence.

Tequila Authentication Flow:
1. Request wiki page → redirects to tequila.epfl.ch
2. POST username/password to Tequila login form
3. Tequila validates and redirects back with session cookie
4. Session cookies cached in keyring for future requests (TTL: 24 hours)

Example:
    from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

    client = MediaWikiClient(
        base_url="https://spcwiki.epfl.ch/wiki",
        credential_service="tcv",
    )

    if client.authenticate():
        page = client.get_page("Portal:TCV")
        print(page.content_html)
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
import requests
import urllib3

from imas_codex.discovery.wiki.auth import CredentialManager, require_credentials

if TYPE_CHECKING:
    from requests import Response

# Suppress InsecureRequestWarning when verify_ssl=False (intended behavior)
# This is set after imports because it modifies urllib3 behavior at runtime
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Session TTL for wiki: 24 hours (Tequila sessions last ~8 hours)
MEDIAWIKI_SESSION_TTL = 24 * 60 * 60

# Request timeout - increased from 30s to 60s for large wiki pages
DEFAULT_TIMEOUT = 60

# Rate limiting: minimum seconds between requests
RATE_LIMIT_DELAY = 0.3


@dataclass
class MediaWikiPage:
    """A MediaWiki page with content and metadata."""

    title: str
    url: str
    content_html: str
    content_text: str = ""
    page_id: int | None = None
    revision_id: int | None = None
    last_modified: str | None = None
    categories: list[str] | None = None

    @property
    def page_name(self) -> str:
        """Page name for compatibility with WikiPage interface."""
        return self.title


class TequilaAuthError(Exception):
    """Tequila authentication failed."""

    pass


class MediaWikiClient:
    """HTTP client for MediaWiki sites with Tequila (EPFL SSO) authentication.

    Handles the Tequila SSO redirect flow:
    1. Access wiki → redirect to tequila.epfl.ch
    2. Submit credentials to Tequila login form
    3. Follow redirects back to wiki with session
    4. Cache session cookies in keyring

    Attributes:
        base_url: MediaWiki base URL (e.g., "https://spcwiki.epfl.ch/wiki")
        credential_service: Keyring service name for credentials
        timeout: Request timeout in seconds
    """

    # Tequila SSO endpoints
    TEQUILA_HOST = "tequila.epfl.ch"
    TEQUILA_LOGIN_URL = "https://tequila.epfl.ch/cgi-bin/tequila/login"

    def __init__(
        self,
        base_url: str,
        credential_service: str,
        timeout: int = DEFAULT_TIMEOUT,
        verify_ssl: bool = False,  # SPC wiki has self-signed cert
    ) -> None:
        """Initialize MediaWiki client.

        Args:
            base_url: MediaWiki base URL (e.g., "https://spcwiki.epfl.ch/wiki")
            credential_service: Keyring service name for credentials
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        import threading

        self.base_url = base_url.rstrip("/")
        self.credential_service = credential_service
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        self._session: requests.Session | None = None
        self._creds = CredentialManager()
        self._last_request_time = 0.0
        self._authenticated = False
        self._redirect_retry_attempted = False  # Track retry to prevent recursion
        # Reentrant lock for session operations (requests.Session is not thread-safe)
        # Uses RLock so get_page() can call _authenticate_impl() while holding the lock
        self._lock = threading.RLock()

    def _get_session(self) -> requests.Session:
        """Get or create requests session with restored cookies.

        Configures session for optimal performance:
        - gzip/deflate compression (70-80% size reduction)
        - Keep-alive connections (reuse TCP connections)
        - Connection pooling via HTTPAdapter
        """
        if self._session is None:
            self._session = requests.Session()

            # Performance: Configure connection pooling
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            # Retry strategy for transient errors
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,  # Number of connection pools
                pool_maxsize=20,  # Connections per pool
            )
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)

            self._session.headers.update(
                {
                    "User-Agent": "imas-codex/1.0 (IMAS Data Mapping Tool)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    # Performance: Enable gzip/deflate compression
                    "Accept-Encoding": "gzip, deflate",
                    # Performance: Keep connections alive
                    "Connection": "keep-alive",
                }
            )

            # Restore session cookies from keyring
            cookies = self._creds.get_session(self.credential_service)
            if cookies:
                self._session.cookies.clear()
                self._session.cookies.update(cookies)
                logger.debug("Restored session cookies from keyring")

        return self._session

    @property
    def session(self) -> requests.Session:
        """Public access to the authenticated session."""
        return self._get_session()

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _is_tequila_redirect(self, response: Response) -> bool:
        """Check if response is a Tequila login redirect."""
        # Check if we were redirected to Tequila
        return self.TEQUILA_HOST in response.url

    def _extract_tequila_params(self, html: str, url: str) -> dict[str, str]:
        """Extract hidden form parameters from Tequila login page.

        Args:
            html: Tequila login page HTML
            url: Current URL (contains requestkey)

        Returns:
            Dict of form parameters needed for login POST
        """
        params = {}

        # Extract requestkey from URL
        requestkey_match = re.search(r"requestkey=([^&]+)", url)
        if requestkey_match:
            params["requestkey"] = requestkey_match.group(1)

        # Extract any hidden form fields
        hidden_pattern = re.compile(
            r'<input[^>]*type=["\']hidden["\'][^>]*name=["\']([^"\']+)["\'][^>]*value=["\']([^"\']*)["\']',
            re.IGNORECASE,
        )
        for match in hidden_pattern.finditer(html):
            params[match.group(1)] = match.group(2)

        # Also try reversed order (value before name)
        hidden_pattern2 = re.compile(
            r'<input[^>]*value=["\']([^"\']*)["\'][^>]*name=["\']([^"\']+)["\'][^>]*type=["\']hidden["\']',
            re.IGNORECASE,
        )
        for match in hidden_pattern2.finditer(html):
            params[match.group(2)] = match.group(1)

        return params

    def _perform_tequila_login(
        self,
        login_page_html: str,
        login_url: str,
        username: str,
        password: str,
    ) -> Response:
        """Submit Tequila login form and follow redirects.

        Args:
            login_page_html: HTML of Tequila login page
            login_url: URL of Tequila login page
            username: EPFL username
            password: EPFL password

        Returns:
            Final response after authentication

        Raises:
            TequilaAuthError: If login fails
        """
        session = self._get_session()

        # Extract form parameters
        params = self._extract_tequila_params(login_page_html, login_url)

        # Add credentials and submit button
        params["username"] = username
        params["password"] = password
        params["login"] = ""  # Submit button field

        logger.debug("Submitting Tequila login form to %s", self.TEQUILA_LOGIN_URL)

        # POST to Tequila
        # Note: Use verify=False because the redirect chain ends at spcwiki
        # which has a self-signed certificate
        response = session.post(
            self.TEQUILA_LOGIN_URL,
            data=params,
            timeout=self.timeout,
            verify=False,  # Redirect back to spcwiki needs this
            allow_redirects=True,
        )

        # Check if login failed (still on Tequila page)
        if self.TEQUILA_HOST in response.url:
            # Check for error message in response
            if (
                "Invalid username" in response.text
                or "Invalid password" in response.text
            ):
                raise TequilaAuthError("Invalid username or password")
            if "form" in response.text.lower() and "login" in response.text.lower():
                raise TequilaAuthError("Login failed - still on login page")

        return response

    def authenticate(self, force: bool = False) -> bool:
        """Authenticate with Tequila SSO.

        Attempts to use existing session, falls back to fresh login.
        Thread-safe: uses lock to prevent concurrent authentication races.

        Args:
            force: Force re-authentication even if session exists

        Returns:
            True if authenticated successfully
        """
        # Quick check without lock for already-authenticated case
        if not force and self._authenticated:
            return True

        # All other operations need the lock (session is not thread-safe)
        with self._lock:
            # Double-check after acquiring lock
            if not force and self._authenticated:
                return True

            return self._authenticate_impl(force)

    def _authenticate_impl(self, force: bool) -> bool:
        """Internal authentication implementation (must be called with lock held)."""
        session = self._get_session()

        if not force:
            try:
                # Test session with a simple page request
                test_url = self.base_url
                response = session.get(
                    test_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    allow_redirects=True,
                )

                # If we don't get redirected to Tequila, session is valid
                if not self._is_tequila_redirect(response):
                    logger.debug("Session valid (no Tequila redirect)")
                    self._authenticated = True
                    return True

                logger.debug("Session expired, need to re-authenticate")
                # Clear stale cookies to prevent redirect loops during fresh auth
                session.cookies.clear()
                self._creds.delete_session(self.credential_service)

            except Exception as e:
                logger.debug("Session check failed: %s", e)
                # Clear cookies on failure to ensure clean slate
                session.cookies.clear()

        # Need to authenticate with credentials
        username, password = require_credentials(self.credential_service)

        try:
            # Access wiki to trigger Tequila redirect
            response = session.get(
                self.base_url,
                timeout=self.timeout,
                verify=self.verify_ssl,
                allow_redirects=True,
            )

            if not self._is_tequila_redirect(response):
                # No auth needed? Unusual but possible
                logger.info("No Tequila redirect - site may be public")
                self._authenticated = True
                return True

            # Perform Tequila login
            response = self._perform_tequila_login(
                login_page_html=response.text,
                login_url=response.url,
                username=username,
                password=password,
            )

            # Verify we're back on the wiki
            if self._is_tequila_redirect(response):
                logger.error("Still on Tequila after login - authentication failed")
                return False

            logger.info("Authenticated successfully via Tequila")

            # Save session cookies
            cookie_dict = {cookie.name: cookie.value for cookie in session.cookies}
            self._creds.set_session(
                self.credential_service,
                cookie_dict,
                ttl=MEDIAWIKI_SESSION_TTL,
            )

            self._authenticated = True
            return True

        except TequilaAuthError as e:
            logger.error("Tequila authentication failed: %s", e)
            return False
        except requests.TooManyRedirects:
            # Redirect loop during auth - only retry once
            if self._redirect_retry_attempted:
                logger.error(
                    "Redirect loop persists after retry - check Tequila credentials and network"
                )
                return False

            logger.warning(
                "Redirect loop during authentication - clearing cookies and retrying once"
            )
            self._redirect_retry_attempted = True
            session.cookies.clear()
            self._creds.delete_session(self.credential_service)

            # Create fresh session to avoid stale cookie state
            self._session = None
            session = self._get_session()

            # Retry authentication once with fresh session
            try:
                response = session.get(
                    self.base_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    allow_redirects=True,
                )
                if self._is_tequila_redirect(response):
                    response = self._perform_tequila_login(
                        login_page_html=response.text,
                        login_url=response.url,
                        username=username,
                        password=password,
                    )
                    if not self._is_tequila_redirect(response):
                        cookie_dict = {
                            cookie.name: cookie.value for cookie in session.cookies
                        }
                        self._creds.set_session(
                            self.credential_service,
                            cookie_dict,
                            ttl=MEDIAWIKI_SESSION_TTL,
                        )
                        self._authenticated = True
                        self._redirect_retry_attempted = False  # Reset on success
                        logger.info("Re-authenticated successfully after redirect loop")
                        return True
            except requests.TooManyRedirects:
                logger.error(
                    "Redirect loop persists after retry - Tequila service may be misconfigured"
                )
            except Exception as retry_e:
                logger.error("Retry after redirect loop failed: %s", retry_e)
            return False
        except requests.RequestException as e:
            logger.error("Request failed during authentication: %s", e)
            return False

    def get_page(self, page_name: str) -> MediaWikiPage | None:
        """Fetch a MediaWiki page.

        Thread-safe: uses lock to prevent concurrent session access.

        Args:
            page_name: Page name (e.g., "Portal:TCV", "Thomson/DDJ")

        Returns:
            MediaWikiPage or None if fetch failed
        """
        if not self._authenticated:
            if not self.authenticate():
                return None

        self._rate_limit()

        # URL-encode page name (preserve slashes for subpages)
        from urllib.parse import quote

        encoded_name = quote(page_name, safe="/")
        url = f"{self.base_url}/{encoded_name}"

        # All session operations need the lock (not thread-safe)
        with self._lock:
            session = self._get_session()
            try:
                response = session.get(
                    url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    allow_redirects=True,
                )

                # Check if session expired mid-request
                if self._is_tequila_redirect(response):
                    logger.warning("Session expired during request, re-authenticating")
                    self._authenticated = False
                    # Note: authenticate() will acquire lock again, but we support reentrant calls
                    if not self._authenticate_impl(force=True):
                        return None
                    # Retry request
                    response = session.get(
                        url,
                        timeout=self.timeout,
                        verify=self.verify_ssl,
                        allow_redirects=True,
                    )

                response.raise_for_status()

                # Extract title
                title_match = re.search(r"<title>([^<]+)</title>", response.text)
                title = (
                    title_match.group(1).replace(" - SPCwiki", "")
                    if title_match
                    else page_name
                )

                # Extract text content (strip tags for entity extraction)
                text_content = re.sub(r"<[^>]+>", " ", response.text)
                text_content = re.sub(r"\s+", " ", text_content)

                return MediaWikiPage(
                    title=title,
                    url=url,
                    content_html=response.text,
                    content_text=text_content,
                )

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.debug("Page not found: %s", page_name)
                else:
                    logger.error("HTTP error fetching %s: %s", page_name, e)
                return None
            except requests.TooManyRedirects:
                # Redirect loop - session cookies are likely corrupt
                # Clear cookies and force re-authentication on next request
                logger.warning(
                    "Redirect loop for %s - clearing session and retrying", page_name
                )
                self._authenticated = False
                session.cookies.clear()
                # Clear cached session from keyring
                self._creds.delete_session(self.credential_service)
                # Retry once with fresh auth
                if self._authenticate_impl(force=True):
                    try:
                        response = session.get(
                            url,
                            timeout=self.timeout,
                            verify=self.verify_ssl,
                            allow_redirects=True,
                        )
                        if not self._is_tequila_redirect(response):
                            response.raise_for_status()
                            title_match = re.search(
                                r"<title>([^<]+)</title>", response.text
                            )
                            title = (
                                title_match.group(1).replace(" - SPCwiki", "")
                                if title_match
                                else page_name
                            )
                            text_content = re.sub(r"<[^>]+>", " ", response.text)
                            text_content = re.sub(r"\s+", " ", text_content)
                            return MediaWikiPage(
                                title=title,
                                url=url,
                                content_html=response.text,
                                content_text=text_content,
                            )
                    except Exception as retry_e:
                        logger.error(
                            "Retry after redirect loop failed for %s: %s",
                            page_name,
                            retry_e,
                        )
                return None
            except requests.RequestException as e:
                logger.warning("Request failed for %s: %s", page_name, e)
                return None

    def get_page_links(self, page_name: str) -> list[str]:
        """Extract internal wiki links from a page.

        Args:
            page_name: Page name to extract links from

        Returns:
            List of linked page names
        """
        page = self.get_page(page_name)
        if not page:
            return []

        links = set()
        parsed_base = urlparse(self.base_url)
        wiki_path = parsed_base.path  # e.g., "/wiki"

        # Find internal wiki links
        link_pattern = re.compile(
            rf'<a[^>]*href=["\']({re.escape(wiki_path)}/([^"\'#]+))["\']',
            re.IGNORECASE,
        )

        for match in link_pattern.finditer(page.content_html):
            page_path = match.group(2)
            # Skip special pages, images, etc.
            if not any(
                page_path.startswith(s)
                for s in ("Special:", "File:", "Image:", "User:", "Talk:", "Category:")
            ):
                from urllib.parse import unquote

                links.add(unquote(page_path))

        # Also look for relative links
        relative_pattern = re.compile(
            r'<a[^>]*href=["\'](?!\w+://)["\']?/wiki/([^"\'#]+)["\']',
            re.IGNORECASE,
        )
        for match in relative_pattern.finditer(page.content_html):
            page_path = match.group(1)
            if not any(
                page_path.startswith(s)
                for s in ("Special:", "File:", "Image:", "User:", "Talk:", "Category:")
            ):
                from urllib.parse import unquote

                links.add(unquote(page_path))

        return list(links)

    def close(self) -> None:
        """Close the session."""
        if self._session is not None:
            self._session.close()
            self._session = None
            self._authenticated = False


def get_mediawiki_client(facility: str, site_index: int = 0) -> MediaWikiClient:
    """Create a MediaWiki client for a facility.

    Args:
        facility: Facility identifier (e.g., "tcv")
        site_index: Index of wiki site in facility config

    Returns:
        Configured MediaWikiClient

    Raises:
        ValueError: If facility has no wiki sites or invalid index
    """
    from imas_codex.discovery.wiki.config import WikiConfig

    config = WikiConfig.from_facility(facility, site_index)

    if config.site_type != "mediawiki":
        raise ValueError(f"Site type {config.site_type} is not mediawiki")

    return MediaWikiClient(
        base_url=config.base_url,
        credential_service=config.credential_service or f"{facility}-wiki",
        verify_ssl=False,  # Most MediaWiki sites have self-signed certs
    )


class AsyncMediaWikiClient:
    """Async HTTP client for MediaWiki sites with Tequila (EPFL SSO) authentication.

    This is the async version of MediaWikiClient using httpx.AsyncClient
    for native async HTTP operations instead of blocking requests.

    Handles the Tequila SSO redirect flow:
    1. Access wiki → redirect to tequila.epfl.ch
    2. Submit credentials to Tequila login form
    3. Follow redirects back to wiki with session
    4. Cache session cookies in keyring

    Example:
        async with AsyncMediaWikiClient(
            base_url="https://spcwiki.epfl.ch/wiki",
            credential_service="tcv",
        ) as client:
            if await client.authenticate():
                page = await client.get_page("Portal:TCV")
                print(page.content_html)

    Attributes:
        base_url: MediaWiki base URL (e.g., "https://spcwiki.epfl.ch/wiki")
        credential_service: Keyring service name for credentials
        timeout: Request timeout in seconds
    """

    # Tequila SSO endpoints
    TEQUILA_HOST = "tequila.epfl.ch"
    TEQUILA_LOGIN_URL = "https://tequila.epfl.ch/cgi-bin/tequila/login"

    def __init__(
        self,
        base_url: str,
        credential_service: str,
        timeout: float = DEFAULT_TIMEOUT,
        verify_ssl: bool = False,  # SPC wiki has self-signed cert
    ) -> None:
        """Initialize async MediaWiki client.

        Args:
            base_url: MediaWiki base URL (e.g., "https://spcwiki.epfl.ch/wiki")
            credential_service: Keyring service name for credentials
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        import asyncio

        self.base_url = base_url.rstrip("/")
        self.credential_service = credential_service
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        self._client: httpx.AsyncClient | None = None
        self._creds = CredentialManager()
        self._last_request_time = 0.0
        self._authenticated = False
        self._redirect_retry_attempted = False
        # Async lock for session operations
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> AsyncMediaWikiClient:
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client with restored cookies.

        Configures client for optimal performance:
        - gzip/deflate compression
        - HTTP/2 support
        - Connection pooling
        """
        import asyncio

        import httpx

        if self._client is None:
            # Restore session cookies from keyring (non-blocking)
            cookies = await asyncio.to_thread(
                self._creds.get_session, self.credential_service
            )
            cookie_jar = httpx.Cookies()
            if cookies:
                for name, value in cookies.items():
                    cookie_jar.set(name, value)
                logger.debug("Restored session cookies from keyring")

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                verify=self.verify_ssl,
                cookies=cookie_jar,
                headers={
                    "User-Agent": "imas-codex/1.0 (IMAS Data Mapping Tool)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                },
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                ),
            )

        return self._client

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        import asyncio

        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _is_tequila_redirect(self, response: httpx.Response) -> bool:
        """Check if response is a Tequila login redirect."""
        return self.TEQUILA_HOST in str(response.url)

    def _extract_tequila_params(self, html: str, url: str) -> dict[str, str]:
        """Extract hidden form parameters from Tequila login page."""
        params = {}

        # Extract requestkey from URL
        requestkey_match = re.search(r"requestkey=([^&]+)", url)
        if requestkey_match:
            params["requestkey"] = requestkey_match.group(1)

        # Extract any hidden form fields
        hidden_pattern = re.compile(
            r'<input[^>]*type=["\']hidden["\'][^>]*name=["\']([^"\']+)["\'][^>]*value=["\']([^"\']*)["\']',
            re.IGNORECASE,
        )
        for match in hidden_pattern.finditer(html):
            params[match.group(1)] = match.group(2)

        # Also try reversed order (value before name)
        hidden_pattern2 = re.compile(
            r'<input[^>]*value=["\']([^"\']*)["\'][^>]*name=["\']([^"\']+)["\'][^>]*type=["\']hidden["\']',
            re.IGNORECASE,
        )
        for match in hidden_pattern2.finditer(html):
            params[match.group(2)] = match.group(1)

        return params

    async def _perform_tequila_login(
        self,
        login_page_html: str,
        login_url: str,
        username: str,
        password: str,
    ) -> httpx.Response:
        """Submit Tequila login form and follow redirects.

        Args:
            login_page_html: HTML of Tequila login page
            login_url: URL of Tequila login page
            username: EPFL username
            password: EPFL password

        Returns:
            Final response after authentication

        Raises:
            TequilaAuthError: If login fails
        """
        client = await self._get_client()

        # Extract form parameters
        params = self._extract_tequila_params(login_page_html, login_url)

        # Add credentials and submit button
        params["username"] = username
        params["password"] = password
        params["login"] = ""  # Submit button field

        logger.debug("Submitting Tequila login form to %s", self.TEQUILA_LOGIN_URL)

        # POST to Tequila
        response = await client.post(
            self.TEQUILA_LOGIN_URL,
            data=params,
        )

        # Check if login failed (still on Tequila page)
        if self.TEQUILA_HOST in str(response.url):
            if (
                "Invalid username" in response.text
                or "Invalid password" in response.text
            ):
                raise TequilaAuthError("Invalid username or password")
            if "form" in response.text.lower() and "login" in response.text.lower():
                raise TequilaAuthError("Login failed - still on login page")

        return response

    async def authenticate(self, force: bool = False) -> bool:
        """Authenticate with Tequila SSO.

        Attempts to use existing session, falls back to fresh login.
        Thread-safe: uses async lock to prevent concurrent authentication races.

        Args:
            force: Force re-authentication even if session exists

        Returns:
            True if authenticated successfully
        """
        # Quick check without lock for already-authenticated case
        if not force and self._authenticated:
            return True

        # All other operations need the lock
        async with self._lock:
            # Double-check after acquiring lock
            if not force and self._authenticated:
                return True

            return await self._authenticate_impl(force)

    async def _authenticate_impl(self, force: bool) -> bool:
        """Internal authentication implementation (must be called with lock held)."""
        import asyncio

        import httpx

        client = await self._get_client()

        if not force:
            try:
                # Test session with a simple page request
                response = await client.get(self.base_url)

                # If we don't get redirected to Tequila, session is valid
                if not self._is_tequila_redirect(response):
                    logger.debug("Session valid (no Tequila redirect)")
                    self._authenticated = True
                    return True

                logger.debug("Session expired, need to re-authenticate")
                # Clear stale cookies
                client.cookies.clear()
                self._creds.delete_session(self.credential_service)

            except Exception as e:
                logger.debug("Session check failed: %s", e)
                client.cookies.clear()

        # Need to authenticate with credentials
        username, password = require_credentials(self.credential_service)

        try:
            # Access wiki to trigger Tequila redirect
            response = await client.get(self.base_url)

            if not self._is_tequila_redirect(response):
                # No auth needed? Unusual but possible
                logger.info("No Tequila redirect - site may be public")
                self._authenticated = True
                return True

            # Perform Tequila login
            response = await self._perform_tequila_login(
                login_page_html=response.text,
                login_url=str(response.url),
                username=username,
                password=password,
            )

            # Verify we're back on the wiki
            if self._is_tequila_redirect(response):
                logger.error("Still on Tequila after login - authentication failed")
                return False

            logger.info("Authenticated successfully via Tequila")

            # Save session cookies (non-blocking)
            cookie_dict = dict(client.cookies.items())
            await asyncio.to_thread(
                self._creds.set_session,
                self.credential_service,
                cookie_dict,
                MEDIAWIKI_SESSION_TTL,
            )

            self._authenticated = True
            return True

        except TequilaAuthError as e:
            logger.error("Tequila authentication failed: %s", e)
            return False
        except httpx.TooManyRedirects:
            # Redirect loop during auth - only retry once
            if self._redirect_retry_attempted:
                logger.error(
                    "Redirect loop persists after retry - check Tequila credentials"
                )
                return False

            logger.warning(
                "Redirect loop during authentication - clearing cookies and retrying"
            )
            self._redirect_retry_attempted = True
            client.cookies.clear()
            await asyncio.to_thread(self._creds.delete_session, self.credential_service)

            # Create fresh client
            if self._client:
                await self._client.aclose()
            self._client = None
            client = await self._get_client()

            # Retry authentication once with fresh client
            try:
                response = await client.get(self.base_url)
                if self._is_tequila_redirect(response):
                    response = await self._perform_tequila_login(
                        login_page_html=response.text,
                        login_url=str(response.url),
                        username=username,
                        password=password,
                    )
                    if not self._is_tequila_redirect(response):
                        cookie_dict = dict(client.cookies.items())
                        await asyncio.to_thread(
                            self._creds.set_session,
                            self.credential_service,
                            cookie_dict,
                            MEDIAWIKI_SESSION_TTL,
                        )
                        self._authenticated = True
                        self._redirect_retry_attempted = False
                        logger.info("Re-authenticated successfully after redirect loop")
                        return True
            except httpx.TooManyRedirects:
                logger.error("Redirect loop persists after retry")
            except Exception as retry_e:
                logger.error("Retry after redirect loop failed: %s", retry_e)
            return False
        except httpx.RequestError as e:
            logger.error("Request failed during authentication: %s", e)
            return False

    async def get_page(self, page_name: str) -> MediaWikiPage | None:
        """Fetch a MediaWiki page.

        Uses httpx AsyncClient for concurrent requests with connection pooling.
        Includes a backstop timeout to prevent indefinite hangs.

        Args:
            page_name: Page name (e.g., "Portal:TCV", "Thomson/DDJ")

        Returns:
            MediaWikiPage or None if fetch failed
        """
        import asyncio

        import httpx

        if not self._authenticated:
            if not await self.authenticate():
                return None

        await self._rate_limit()

        # URL-encode page name (preserve slashes for subpages)
        from urllib.parse import quote

        encoded_name = quote(page_name, safe="/")
        url = f"{self.base_url}/{encoded_name}"

        # Backstop timeout (2x the client timeout) in case httpx timeout fails
        backstop_timeout = self.timeout * 2

        client = await self._get_client()
        try:
            # Wrap request in wait_for as backstop against hanging connections
            response = await asyncio.wait_for(client.get(url), timeout=backstop_timeout)

            # Check if session expired mid-request
            if self._is_tequila_redirect(response):
                logger.warning("Session expired during request, re-authenticating")
                self._authenticated = False
                async with self._lock:
                    if not await self._authenticate_impl(force=True):
                        return None
                # Retry request with backstop timeout
                response = await asyncio.wait_for(
                    client.get(url), timeout=backstop_timeout
                )

            response.raise_for_status()

            # Extract title
            title_match = re.search(r"<title>([^<]+)</title>", response.text)
            title = (
                title_match.group(1).replace(" - SPCwiki", "")
                if title_match
                else page_name
            )

            # Extract text content (strip tags for entity extraction)
            text_content = re.sub(r"<[^>]+>", " ", response.text)
            text_content = re.sub(r"\s+", " ", text_content)

            return MediaWikiPage(
                title=title,
                url=url,
                content_html=response.text,
                content_text=text_content,
            )

        except TimeoutError:
            logger.warning(
                "Request timed out for %s after %.1fs", page_name, backstop_timeout
            )
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("Page not found: %s", page_name)
            else:
                logger.error("HTTP error fetching %s: %s", page_name, e)
            return None
        except httpx.TooManyRedirects:
            logger.warning(
                "Redirect loop for %s - clearing session and retrying", page_name
            )
            self._authenticated = False
            client.cookies.clear()
            await asyncio.to_thread(self._creds.delete_session, self.credential_service)
            async with self._lock:
                if await self._authenticate_impl(force=True):
                    try:
                        response = await asyncio.wait_for(
                            client.get(url), timeout=backstop_timeout
                        )
                        if not self._is_tequila_redirect(response):
                            response.raise_for_status()
                            title_match = re.search(
                                r"<title>([^<]+)</title>", response.text
                            )
                            title = (
                                title_match.group(1).replace(" - SPCwiki", "")
                                if title_match
                                else page_name
                            )
                            text_content = re.sub(r"<[^>]+>", " ", response.text)
                            text_content = re.sub(r"\s+", " ", text_content)
                            return MediaWikiPage(
                                title=title,
                                url=url,
                                content_html=response.text,
                                content_text=text_content,
                            )
                    except Exception as retry_e:
                        logger.error(
                            "Retry after redirect loop failed for %s: %s",
                            page_name,
                            retry_e,
                        )
            return None
        except httpx.RequestError as e:
            logger.warning("Request failed for %s: %s", page_name, e)
            return None

    async def close(self) -> None:
        """Close the async client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._authenticated = False


async def get_async_mediawiki_client(
    facility: str, site_index: int = 0
) -> AsyncMediaWikiClient:
    """Create an async MediaWiki client for a facility.

    Args:
        facility: Facility identifier (e.g., "tcv")
        site_index: Index of wiki site in facility config

    Returns:
        Configured AsyncMediaWikiClient

    Raises:
        ValueError: If facility has no wiki sites or invalid index
    """
    from imas_codex.discovery.wiki.config import WikiConfig

    config = WikiConfig.from_facility(facility, site_index)

    if config.site_type != "mediawiki":
        raise ValueError(f"Site type {config.site_type} is not mediawiki")

    return AsyncMediaWikiClient(
        base_url=config.base_url,
        credential_service=config.credential_service or f"{facility}-wiki",
        verify_ssl=False,
    )
