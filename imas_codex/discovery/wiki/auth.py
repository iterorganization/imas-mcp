"""Credential management for wiki authentication.

Provides secure credential storage using system keyring with fallbacks
to environment variables and interactive prompts.

Credential lookup order:
1. System keyring (GNOME Keyring, macOS Keychain, Windows Credential Locker)
2. Environment variables (FACILITY_SITE_USERNAME, FACILITY_SITE_PASSWORD)
3. Interactive prompt (if terminal available)

Session cookies are also stored in keyring for persistence across runs.

Example:
    from imas_codex.discovery.wiki.auth import CredentialManager

    creds = CredentialManager()

    # Store credentials (one-time setup)
    creds.set_credentials("iter-confluence", "username", "password")

    # Retrieve credentials
    username, password = creds.get_credentials("iter-confluence")

    # Store session cookies after login
    creds.set_session("iter-confluence", cookies_dict)

    # Retrieve session for subsequent requests
    session = creds.get_session("iter-confluence")
"""

import getpass
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Service name prefix for all imas-codex credentials
SERVICE_PREFIX = "imas-codex"

# Session TTL in seconds (default: 8 hours)
DEFAULT_SESSION_TTL = 8 * 60 * 60


@dataclass
class WikiSiteConfig:
    """Configuration for a wiki site."""

    url: str
    portal_page: str
    site_type: str  # mediawiki, confluence, twiki_static, static_html
    auth_type: str  # none, tequila, session, basic
    access_method: str = "direct"  # direct, vpn (preferred network route)
    ssh_available: bool = False  # SSH access possible (for scp, tunneling)
    credential_service: str | None = None  # keyring service name
    ssh_host: str | None = None  # SSH host for tunnel/proxy access


def _is_wsl() -> bool:
    """Detect if running in Windows Subsystem for Linux."""
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


class CredentialManager:
    """Manage credentials using system keyring with fallbacks.

    Uses the system keyring for secure storage:
    - Linux: GNOME Keyring (via SecretService D-Bus API)
    - macOS: Keychain
    - Windows: Credential Locker

    Falls back to environment variables for CI/automation,
    and interactive prompts for first-time setup.

    Credentials entered via interactive prompts are cached in-memory
    (class-level) so they persist across CredentialManager instances
    within the same process, avoiding re-prompting on headless servers
    where keyring is unavailable.
    """

    # Timeout for keyring operations (seconds)
    KEYRING_TIMEOUT = 3

    # Class-level in-memory credential cache: {site: (username, password)}
    # Persists across instances within the same process
    _memory_cache: dict[str, tuple[str, str]] = {}

    def __init__(self) -> None:
        """Initialize credential manager."""
        self._keyring_available = self._check_keyring()

    def _check_keyring(self) -> bool:
        """Check if keyring is available and functional.

        Returns False (disables keyring) in these cases:
        - No DISPLAY and no DBUS_SESSION_BUS_ADDRESS (headless server)
        - Keyring backend check times out
        - PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring (explicit disable)
        """
        # Check for explicit disable via environment
        if os.environ.get("PYTHON_KEYRING_BACKEND") == "keyring.backends.null.Keyring":
            logger.info("Keyring explicitly disabled via PYTHON_KEYRING_BACKEND")
            return False

        # Skip on headless Linux without D-Bus (but not WSL - it has GUI support)
        if sys.platform == "linux" and not _is_wsl():
            if not os.environ.get("DISPLAY") and not os.environ.get(
                "DBUS_SESSION_BUS_ADDRESS"
            ):
                logger.info(
                    "Headless Linux detected (no DISPLAY/DBUS) - using env vars."
                )
                return False

        import concurrent.futures

        def _do_check() -> bool:
            import keyring

            # Test with a dummy operation
            backend = keyring.get_keyring()
            logger.debug("Keyring backend: %s", backend)
            return True

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_check)
                result = future.result(timeout=self.KEYRING_TIMEOUT)
                return result
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Keyring check timed out after %ds - D-Bus/SecretService may be unresponsive. "
                "Using environment variables instead.",
                self.KEYRING_TIMEOUT,
            )
            if _is_wsl():
                logger.warning(
                    "WSL detected: If GNOME Keyring prompts for unlock password, "
                    "you can reset it with: rm -rf ~/.local/share/keyrings/*"
                )
            return False
        except Exception as e:
            logger.warning("Keyring not available: %s", e)
            return False

    def _service_name(self, site: str) -> str:
        """Generate full service name for keyring."""
        return f"{SERVICE_PREFIX}/{site}"

    def _env_var_name(self, site: str, key: str) -> str:
        """Generate environment variable name.

        Converts site name to uppercase with underscores.
        e.g., "iter-confluence" + "username" -> "ITER_CONFLUENCE_USERNAME"
        """
        site_upper = site.upper().replace("-", "_").replace("/", "_")
        return f"{site_upper}_{key.upper()}"

    def _keyring_op(self, func, *args, **kwargs):
        """Run a keyring operation with timeout.

        Prevents blocking when GNOME Keyring prompts for unlock password.

        Args:
            func: Function to call (e.g., keyring.get_password)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func, or None on timeout/error
        """
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=self.KEYRING_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Keyring operation timed out after %ds - may need unlock. "
                "See: imas-codex credentials status",
                self.KEYRING_TIMEOUT,
            )
            return None
        except Exception as e:
            logger.debug("Keyring operation failed: %s", e)
            return None

    def set_credentials(
        self,
        site: str,
        username: str,
        password: str,
    ) -> bool:
        """Store credentials in keyring.

        Args:
            site: Site identifier (e.g., "iter-confluence")
            username: Username
            password: Password

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._keyring_available:
            logger.error("Keyring not available. Cannot store credentials.")
            return False

        import keyring

        service = self._service_name(site)
        try:
            # Store username and password as JSON
            creds = json.dumps({"username": username, "password": password})

            # Use timeout wrapper to avoid blocking on unlock prompt
            def _set():
                keyring.set_password(service, "credentials", creds)
                return True

            result = self._keyring_op(_set)
            if result:
                logger.info("Stored credentials for %s", site)
                return True
            else:
                logger.error(
                    "Failed to store credentials - keyring may need unlock. "
                    "Run: imas-codex credentials status"
                )
                return False
        except Exception as e:
            # Provide detailed error with setup instructions
            error_msg = str(e)
            logger.error("Failed to store credentials: %s", error_msg)

            # Check for common keyring backend issues
            if "No recommended backend" in error_msg or "keyrings.alt" in error_msg:
                env_user = self._env_var_name(site, "username")
                env_pass = self._env_var_name(site, "password")
                logger.error(
                    "\n"
                    "Keyring backend not configured. Setup options:\n"
                    "\n"
                    "Option 1: Install a keyring backend (recommended for desktop)\n"
                    "  Linux:   sudo apt install gnome-keyring  # or libsecret\n"
                    "  macOS:   Built-in Keychain should work automatically\n"
                    "  Windows: Built-in Credential Locker should work automatically\n"
                    "\n"
                    "Option 2: Use file-based backend (for servers/containers)\n"
                    "  pip install keyrings.alt\n"
                    "  Then create ~/.config/python_keyring/keyringrc.cfg:\n"
                    "    [backend]\n"
                    "    default-keyring=keyrings.alt.file.PlaintextKeyring\n"
                    "\n"
                    "Option 3: Use environment variables (for CI/automation)\n"
                    f"  export {env_user}=your_username\n"
                    f"  export {env_pass}=your_password\n"
                )
            return False

    def get_credentials(
        self,
        site: str,
        prompt_if_missing: bool = True,
    ) -> tuple[str, str] | None:
        """Retrieve credentials with fallback chain.

        Lookup order:
        1. System keyring
        2. Environment variables
        3. Interactive prompt (if prompt_if_missing=True)

        Args:
            site: Site identifier (e.g., "iter-confluence")
            prompt_if_missing: Whether to prompt interactively if not found

        Returns:
            Tuple of (username, password) or None if not available
        """
        # 1. Try in-memory cache (fastest, works across instances)
        if site in self._memory_cache:
            logger.debug("Retrieved credentials from memory cache for %s", site)
            return self._memory_cache[site]

        # 2. Try keyring (with timeout to avoid blocking on unlock prompt)
        if self._keyring_available:
            import keyring

            service = self._service_name(site)

            def _get():
                return keyring.get_password(service, "credentials")

            creds_json = self._keyring_op(_get)
            if creds_json:
                try:
                    creds = json.loads(creds_json)
                    logger.debug("Retrieved credentials from keyring for %s", site)
                    result = creds["username"], creds["password"]
                    # Populate memory cache for future lookups
                    self._memory_cache[site] = result
                    return result
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug("Invalid credentials JSON: %s", e)

        # 3. Try environment variables
        username_var = self._env_var_name(site, "username")
        password_var = self._env_var_name(site, "password")
        username = os.environ.get(username_var)
        password = os.environ.get(password_var)
        if username and password:
            logger.debug("Retrieved credentials from environment for %s", site)
            result = username, password
            self._memory_cache[site] = result
            return result

        # 4. Interactive prompt
        if prompt_if_missing and sys.stdin.isatty():
            return self._prompt_credentials(site)

        return None

    def _prompt_credentials(self, site: str) -> tuple[str, str] | None:
        """Prompt user for credentials interactively.

        Args:
            site: Site identifier for display

        Returns:
            Tuple of (username, password) or None if cancelled
        """
        print(f"\nCredentials required for: {site}")
        print("(These will be stored in your system keyring for future use)\n")

        try:
            username = input("Username: ").strip()
            if not username:
                return None
            password = getpass.getpass("Password: ")
            if not password:
                return None

            # Offer to save to keyring
            save = input("Save to keyring? [Y/n]: ").strip().lower()
            if save != "n":
                self.set_credentials(site, username, password)

            # Always cache in memory for this process
            result = username, password
            self._memory_cache[site] = result
            return result
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return None

    def delete_credentials(self, site: str) -> bool:
        """Delete stored credentials.

        Args:
            site: Site identifier

        Returns:
            True if deleted, False otherwise
        """
        if not self._keyring_available:
            return False

        import keyring

        service = self._service_name(site)
        try:
            keyring.delete_password(service, "credentials")
            logger.info("Deleted credentials for %s", site)
            return True
        except keyring.errors.PasswordDeleteError:
            logger.debug("No credentials found for %s", site)
            return False
        except Exception as e:
            logger.error("Failed to delete credentials: %s", e)
            return False

    def set_session(
        self,
        site: str,
        cookies: dict[str, str],
        ttl: int = DEFAULT_SESSION_TTL,
    ) -> bool:
        """Store session cookies in keyring.

        Args:
            site: Site identifier
            cookies: Cookie dict from requests.Session
            ttl: Time-to-live in seconds (default: 8 hours)

        Returns:
            True if stored successfully
        """
        if not self._keyring_available:
            return False

        import keyring

        service = self._service_name(site)
        session_data = {
            "cookies": cookies,
            "expires_at": time.time() + ttl,
        }
        try:
            keyring.set_password(service, "session", json.dumps(session_data))
            logger.debug("Stored session for %s (TTL: %ds)", site, ttl)
            return True
        except Exception as e:
            logger.error("Failed to store session: %s", e)
            return False

    def get_session(self, site: str) -> dict[str, str] | None:
        """Retrieve valid session cookies.

        Args:
            site: Site identifier

        Returns:
            Cookie dict if session is valid, None otherwise
        """
        if not self._keyring_available:
            return None

        import keyring

        service = self._service_name(site)
        try:
            session_json = keyring.get_password(service, "session")
            if not session_json:
                return None

            session_data = json.loads(session_json)
            if time.time() > session_data.get("expires_at", 0):
                logger.debug("Session expired for %s", site)
                self.delete_session(site)
                return None

            logger.debug("Retrieved valid session for %s", site)
            return session_data["cookies"]
        except Exception as e:
            logger.debug("Session lookup failed: %s", e)
            return None

    def delete_session(self, site: str) -> bool:
        """Delete stored session.

        Args:
            site: Site identifier

        Returns:
            True if deleted
        """
        if not self._keyring_available:
            return False

        import keyring

        service = self._service_name(site)
        try:
            keyring.delete_password(service, "session")
            return True
        except Exception:
            return False

    def has_credentials(self, site: str) -> bool:
        """Check if credentials exist for a site.

        Args:
            site: Site identifier

        Returns:
            True if credentials are available (memory cache, keyring, or env)
        """
        # Check in-memory cache first
        if site in self._memory_cache:
            return True

        # Check keyring (with timeout to avoid blocking on unlock prompt)
        if self._keyring_available:
            import keyring

            service = self._service_name(site)

            def _check():
                return keyring.get_password(service, "credentials")

            result = self._keyring_op(_check)
            if result:
                return True

        # Check environment
        username_var = self._env_var_name(site, "username")
        password_var = self._env_var_name(site, "password")
        return bool(os.environ.get(username_var) and os.environ.get(password_var))

    def list_sites(self) -> list[str]:
        """List all sites with stored credentials.

        Note: This only works reliably on some keyring backends.

        Returns:
            List of site identifiers
        """
        # This is backend-dependent and may not work everywhere
        # For now, return empty list - CLI can track known sites separately
        return []


def require_credentials(site: str) -> tuple[str, str]:
    """Get credentials or exit with helpful message.

    Use this in CLI commands that require authentication.

    Args:
        site: Site identifier

    Returns:
        Tuple of (username, password)

    Raises:
        SystemExit: If credentials not available
    """
    creds = CredentialManager()

    result = creds.get_credentials(site, prompt_if_missing=True)
    if result is None:
        print(f"\n❌ Credentials required for: {site}")
        print("\nTo set up credentials, run:")
        print(f"  imas-codex credentials set {site}")
        print("\nOr set environment variables:")
        env_user = creds._env_var_name(site, "username")
        env_pass = creds._env_var_name(site, "password")
        print(f"  export {env_user}=your_username")
        print(f"  export {env_pass}=your_password")
        raise SystemExit(1)

    return result
