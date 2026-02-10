"""Keycloak OIDC authentication for wiki sites behind oauth2-proxy.

JET Data Centre wikis use Keycloak (OIDC) via oauth2-proxy. The flow:
1. GET wiki URL → 302 to oauth2-proxy → 302 to Keycloak login form
2. POST username/password to Keycloak form action
3. Keycloak redirects back with auth code → oauth2-proxy sets session cookie
4. All subsequent requests use the session cookie

One login authenticates across all wiki sites on the same domain.

Usage:
    from imas_codex.discovery.wiki.keycloak import KeycloakSession

    # Sync (requests)
    ks = KeycloakSession("jet")
    session = ks.login("https://wiki.jetdata.eu/pog/")
    r = session.get("https://wiki.jetdata.eu/ia/api.php?action=query&...")

    # Async (httpx)
    aks = AsyncKeycloakSession("jet")
    client = await aks.login("https://wiki.jetdata.eu/pog/")
    r = await client.get("https://wiki.jetdata.eu/ia/api.php?action=query&...")
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import httpx
import requests

from imas_codex.discovery.wiki.auth import CredentialManager

logger = logging.getLogger(__name__)


class KeycloakSession:
    """Synchronous Keycloak OIDC authentication via requests.Session."""

    def __init__(self, credential_service: str):
        self.credential_service = credential_service
        self._session: requests.Session | None = None

    def login(self, start_url: str) -> requests.Session:
        """Authenticate via Keycloak form login.

        Args:
            start_url: Any URL behind the oauth2-proxy that triggers auth.

        Returns:
            Authenticated requests.Session with session cookies.

        Raises:
            RuntimeError: If authentication fails.
        """
        cred_mgr = CredentialManager()
        creds = cred_mgr.get_credentials(
            self.credential_service, prompt_if_missing=False
        )
        if not creds:
            raise RuntimeError(
                f"No credentials for {self.credential_service}. "
                f"Set with: imas-codex credentials set {self.credential_service}"
            )

        username, password = creds

        s = requests.Session()
        s.verify = False

        # Step 1: GET the wiki URL - follows redirects to Keycloak login form
        r = s.get(start_url, allow_redirects=True, timeout=30)

        if "/auth/realms/" not in r.url:
            # No Keycloak redirect - maybe already authenticated or no auth needed
            if r.status_code == 200:
                logger.debug(
                    "No Keycloak redirect for %s - may not need auth", start_url
                )
                self._session = s
                return s
            raise RuntimeError(
                f"Unexpected response from {start_url}: HTTP {r.status_code}"
            )

        # Step 2: Extract form action from Keycloak login page
        action_match = re.search(r'action="([^"]+)"', r.text)
        if not action_match:
            raise RuntimeError("Could not find Keycloak login form action")

        login_url = action_match.group(1).replace("&amp;", "&")

        # Make absolute URL if relative
        if login_url.startswith("/"):
            parsed = urlparse(r.url)
            login_url = f"{parsed.scheme}://{parsed.netloc}{login_url}"

        # Step 3: POST credentials to Keycloak
        r2 = s.post(
            login_url,
            data={"username": username, "password": password},
            allow_redirects=True,
            timeout=30,
        )

        # Verify auth succeeded - should be back on the wiki with 200
        if r2.status_code != 200:
            raise RuntimeError(
                f"Keycloak login failed: HTTP {r2.status_code} at {r2.url}"
            )

        # Check if we're still on the login page (wrong credentials)
        if "/auth/realms/" in r2.url and "login" in r2.url:
            raise RuntimeError(
                f"Keycloak login failed for user {username} - check credentials"
            )

        logger.info("Keycloak auth successful for %s", self.credential_service)
        self._session = s
        return s

    @property
    def session(self) -> requests.Session:
        """Get the authenticated session (must call login() first)."""
        if self._session is None:
            raise RuntimeError("Not authenticated - call login() first")
        return self._session

    def close(self):
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None


class KeycloakWikiClient:
    """Wrapper providing MediaWikiClient-compatible interface for Keycloak auth.

    The MediaWikiAdapter needs ``wiki_client.session`` and optionally
    ``wiki_client.authenticate()``.  This wraps :class:`KeycloakSession`
    to provide that interface.
    """

    def __init__(self, base_url: str, credential_service: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.credential_service = credential_service
        self._ks = KeycloakSession(credential_service)
        self._authenticated = False

    def authenticate(self) -> bool:
        """Authenticate via Keycloak and return True on success."""
        if self._authenticated:
            return True
        try:
            self._ks.login(f"{self.base_url}/")
            self._authenticated = True
            return True
        except Exception as e:
            logger.warning("Keycloak auth failed for %s: %s", self.base_url, e)
            return False

    @property
    def session(self) -> requests.Session:
        """Return the authenticated requests.Session."""
        if not self._authenticated:
            self.authenticate()
        return self._ks.session

    def close(self) -> None:
        """Close the session."""
        self._ks.close()
        self._authenticated = False


class AsyncKeycloakSession:
    """Async Keycloak OIDC authentication via httpx.AsyncClient."""

    def __init__(self, credential_service: str):
        self.credential_service = credential_service
        self._client: httpx.AsyncClient | None = None

    async def login(self, start_url: str) -> httpx.AsyncClient:
        """Authenticate via Keycloak form login (async).

        Args:
            start_url: Any URL behind the oauth2-proxy that triggers auth.

        Returns:
            Authenticated httpx.AsyncClient with session cookies.

        Raises:
            RuntimeError: If authentication fails.
        """
        cred_mgr = CredentialManager()
        creds = cred_mgr.get_credentials(
            self.credential_service, prompt_if_missing=False
        )
        if not creds:
            raise RuntimeError(
                f"No credentials for {self.credential_service}. "
                f"Set with: imas-codex credentials set {self.credential_service}"
            )

        username, password = creds

        client = httpx.AsyncClient(timeout=30.0, follow_redirects=True, verify=False)

        # Step 1: GET the wiki URL - follows redirects to Keycloak login form
        r = await client.get(start_url)

        if "/auth/realms/" not in str(r.url):
            if r.status_code == 200:
                logger.debug("No Keycloak redirect - may not need auth")
                self._client = client
                return client
            raise RuntimeError(
                f"Unexpected response from {start_url}: HTTP {r.status_code}"
            )

        # Step 2: Extract form action from Keycloak login page
        action_match = re.search(r'action="([^"]+)"', r.text)
        if not action_match:
            raise RuntimeError("Could not find Keycloak login form action")

        login_url = action_match.group(1).replace("&amp;", "&")

        # Make absolute URL if relative
        if login_url.startswith("/"):
            parsed = urlparse(str(r.url))
            login_url = f"{parsed.scheme}://{parsed.netloc}{login_url}"

        # Step 3: POST credentials to Keycloak
        r2 = await client.post(
            login_url,
            data={"username": username, "password": password},
        )

        if r2.status_code != 200:
            raise RuntimeError(
                f"Keycloak login failed: HTTP {r2.status_code} at {r2.url}"
            )

        if "/auth/realms/" in str(r2.url) and "login" in str(r2.url):
            raise RuntimeError(
                f"Keycloak login failed for user {username} - check credentials"
            )

        logger.info("Keycloak auth successful for %s", self.credential_service)
        self._client = client
        return client

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the authenticated client (must call login() first)."""
        if self._client is None:
            raise RuntimeError("Not authenticated - call login() first")
        return self._client

    async def close(self):
        """Close the async client."""
        if self._client:
            await self._client.aclose()
            self._client = None
