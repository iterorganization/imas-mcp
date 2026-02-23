"""Confluence REST API client for wiki scraping.

Provides authenticated access to Atlassian Confluence instances using
the REST API. Supports session-based authentication with cookie persistence.

The Confluence REST API is preferred over HTML scraping because:
- Structured JSON responses (no HTML parsing needed)
- Pagination support for large spaces
- Attachment metadata and download
- Better rate limiting handling

Example:
    from imas_codex.discovery.wiki.confluence import ConfluenceClient

    client = ConfluenceClient(
        base_url="https://confluence.iter.org",
        credential_service="iter",
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

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from imas_codex.discovery.wiki.auth import CredentialManager, require_credentials

logger = logging.getLogger(__name__)

# Default request timeout
DEFAULT_TIMEOUT = 30

# Rate limiting: minimum seconds between requests
RATE_LIMIT_DELAY = 0.5

# Content-Type header for HTML form submissions (SSO flow)
_FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


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

        Supports multiple authentication methods:
        1. Existing session cookies (fast path)
        2. Azure AD SAML SSO via F5 BIG-IP APM
        3. HTTP Basic Auth fallback (simple instances)

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
                # Test session — don't follow redirects so we can detect SSO
                response = session.get(
                    f"{self.api_url}/user/current",
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                if response.status_code == 200:
                    try:
                        user = response.json()
                        logger.info(
                            "Session valid for user: %s",
                            user.get("displayName", user.get("username")),
                        )
                        self._authenticated = True
                        return True
                    except (json.JSONDecodeError, ValueError):
                        logger.debug("Session returned non-JSON — SSO redirect likely")
                elif response.status_code in (301, 302, 303):
                    logger.debug("Session expired (got redirect)")
            except Exception as e:
                logger.debug("Session check failed: %s", e)

        # Need to authenticate with credentials
        username, password = require_credentials(self.credential_service)

        # Try Azure AD SSO authentication first
        # (handles F5 BIG-IP APM → Azure AD → SAML → Confluence flow)
        if self._authenticate_sso(username, password):
            return True

        # Fallback: HTTP Basic Auth (simple Confluence instances without SSO)
        session.auth = (username, password)
        try:
            response = session.get(
                f"{self.api_url}/user/current",
                timeout=self.timeout,
                allow_redirects=False,
            )
            if response.status_code == 200:
                try:
                    user = response.json()
                    logger.info(
                        "Authenticated as: %s",
                        user.get("displayName", user.get("username")),
                    )
                    cookie_dict = {c.name: c.value for c in session.cookies}
                    self._creds.set_session(
                        self.credential_service,
                        cookie_dict,
                    )
                    self._authenticated = True
                    return True
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Basic auth returned non-JSON — SSO blocking access")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                logger.error("Authentication failed: invalid credentials")
            else:
                logger.error("Authentication failed: %s", e)
        except Exception as e:
            logger.error("Authentication failed: %s", e)

        # Clear basic auth so it doesn't interfere with subsequent requests
        session.auth = None
        return False

    def _authenticate_sso(self, username: str, password: str) -> bool:
        """Authenticate via Azure AD SAML SSO through F5 BIG-IP APM.

        Flow:
        1. Navigate to Confluence → F5 redirects to Azure AD
        2. Extract login config ($Config) from Azure AD page
        3. Call GetCredentialType to validate user and get updated flowToken
        4. POST credentials to Azure AD login endpoint
        5. Handle KMSI (Keep Me Signed In) prompt if shown
        6. Follow SAML assertion redirects back through F5
        7. Verify authenticated API access and save session cookies

        Returns:
            True if SSO authentication succeeded
        """
        session = self._get_session()
        session.auth = None  # Clear any basic auth

        logger.info("Attempting Azure AD SSO for %s", self.base_url)

        # Step 1: Navigate to Confluence → triggers F5 → Azure AD redirect
        try:
            response = session.get(
                f"{self.base_url}/",
                timeout=60,
                allow_redirects=True,
            )
        except Exception as e:
            logger.debug("SSO initial navigation failed: %s", e)
            return False

        if "login.microsoftonline.com" not in response.url:
            logger.debug(
                "SSO did not redirect to Azure AD (url: %s)", response.url[:100]
            )
            return False

        # Step 2: Extract $Config JSON from Azure AD login page
        config = self._extract_azure_config(response.text)
        if config is None:
            logger.warning("Could not extract Azure AD config from login page")
            return False

        flow_token = config.get("sFT", "")
        ctx = config.get("sCtx", "")
        url_post = config.get("urlPost", "")
        canary = config.get("canary", "")

        if not flow_token or not ctx or not url_post:
            logger.warning("Missing required Azure AD login parameters")
            return False

        # Username must be a full email (Azure AD UPN)
        if "@" not in username:
            logger.error(
                "Azure AD SSO requires an email address as username, "
                "got '%s'. Update credentials with: "
                "imas-codex credentials set %s --username your.name@iter.org",
                username,
                self.credential_service,
            )
            return False

        # Step 3: Call GetCredentialType to validate user and refresh flowToken
        gct_url = config.get(
            "urlGetCredentialType",
            "https://login.microsoftonline.com/common/GetCredentialType?mkt=en-US",
        )
        gct_data = {
            "username": username,
            "isOtherIdpSupported": True,
            "checkPhones": False,
            "isRemoteNGCSupported": True,
            "isCookieBannerShown": False,
            "isFidoSupported": True,
            "forceotclogin": False,
            "isExternalFederationDisallowed": False,
            "isRemoteConnectSupported": False,
            "federationFlags": 0,
            "isSignup": False,
            "flowToken": flow_token,
        }

        try:
            gct_resp = session.post(gct_url, json=gct_data, timeout=30)
            gct_result = gct_resp.json()

            if_exists = gct_result.get("IfExistsResult", -1)
            if if_exists == 1:
                logger.error(
                    "Azure AD: User '%s' does not exist in this tenant. "
                    "Check your email format (e.g., firstname.lastname@iter.org). "
                    "Update with: imas-codex credentials set %s --username your.name@iter.org",
                    username,
                    self.credential_service,
                )
                return False

            # Update flowToken from GetCredentialType response
            updated_ft = gct_result.get("FlowToken")
            if updated_ft:
                flow_token = updated_ft
                logger.debug("Updated flowToken from GetCredentialType")

            # Check for federation redirect
            fed_url = gct_result.get("Credentials", {}).get("FederationRedirectUrl")
            if fed_url:
                logger.info("User is federated, redirect: %s", fed_url[:100])
                # Federated users need to authenticate at their IdP
                # For now, try the standard Azure AD flow — federation may be
                # handled server-side via the updated credential type info

        except Exception as e:
            logger.debug("GetCredentialType failed (non-fatal): %s", e)

        # Step 4: POST credentials to Azure AD
        login_url = f"https://login.microsoftonline.com{url_post}"
        logger.debug("Posting credentials to Azure AD for %s", username)

        post_data = {
            "i13": "0",
            "login": username,
            "loginfmt": username,
            "type": "11",
            "LoginOptions": "3",
            "lrt": "",
            "lrtPartition": "",
            "hisRegion": "",
            "hisScaleUnit": "",
            "passwd": password,
            "ps": "2",
            "psRNGCDefaultType": "",
            "psRNGCEntropy": "",
            "psRNGCSLK": "",
            "canary": canary,
            "ctx": ctx,
            "hpgrequestid": "",
            "flowToken": flow_token,
            "PPSX": "",
            "NewUser": "1",
            "FoundMSAs": "",
            "fspost": "0",
            "i21": "0",
            "CookieDisclosure": "0",
            "IsFidoSupported": "1",
            "isSignupPost": "0",
            "i19": "2326",
        }

        try:
            response = session.post(
                login_url,
                data=post_data,
                timeout=60,
                allow_redirects=False,
                headers=_FORM_HEADERS,
            )
        except Exception as e:
            logger.warning("SSO credential POST failed: %s", e)
            return False

        # Step 5: Follow auto-submit forms (device auth, SAML assertions)
        # and handle interactive pages (TFA, KMSI) in a unified loop.
        for _step in range(20):
            # Handle redirects
            if response.status_code in (301, 302, 303):
                redirect_url = response.headers.get("Location", "")
                if redirect_url.startswith("/"):
                    parsed = urlparse(response.url)
                    redirect_url = f"{parsed.scheme}://{parsed.netloc}{redirect_url}"
                logger.debug("Following redirect → %s", redirect_url[:100])
                response = session.get(
                    redirect_url,
                    timeout=60,
                    allow_redirects=False,
                )
                continue

            if response.status_code != 200:
                break

            # Check for Azure AD error page
            resp_config = self._extract_azure_config(response.text)
            if resp_config:
                err_code = resp_config.get("sErrorCode")
                if err_code:
                    logger.error(
                        "Azure AD error %s: %s",
                        err_code,
                        resp_config.get("sErrTxt", ""),
                    )
                    return False

                # Handle MFA (Two-Factor Authentication)
                pgid = resp_config.get("pgid", "")
                if pgid == "ConvergedTFA":
                    response = self._handle_mfa(session, resp_config)
                    if response is None:
                        return False
                    continue

                # Handle KMSI ("Stay signed in?")
                if pgid == "KmsiInterrupt":
                    response = self._handle_kmsi(session, resp_config)
                    if response is None:
                        return False
                    continue

            # Try auto-submit form (device auth, SAML assertion relay)
            action, fields = self._extract_auto_submit_form(response.text, response.url)
            if action:
                logger.debug("Submitting form → %s", action[:80])
                try:
                    response = session.post(
                        action,
                        data=fields,
                        timeout=60,
                        allow_redirects=False,
                        headers=_FORM_HEADERS,
                    )
                except Exception as e:
                    logger.warning("Form submission failed: %s", e)
                    return False
                continue

            # No more forms/redirects — we should be on Confluence now
            break

        # Step 7: Verify authentication
        try:
            verify = session.get(
                f"{self.api_url}/user/current",
                timeout=self.timeout,
                allow_redirects=False,
            )
            if verify.status_code == 200:
                try:
                    user = verify.json()
                    logger.info(
                        "SSO authentication successful: %s",
                        user.get("displayName", user.get("username", "unknown")),
                    )
                    cookie_dict = {c.name: c.value for c in session.cookies}
                    self._creds.set_session(self.credential_service, cookie_dict)
                    self._authenticated = True
                    return True
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "SSO produced non-JSON API response — auth may have failed"
                    )
            else:
                logger.warning(
                    "SSO verification returned status %d", verify.status_code
                )
        except Exception as e:
            logger.warning("SSO verification request failed: %s", e)

        return False

    def _handle_mfa(
        self, session: requests.Session, config: dict
    ) -> requests.Response | None:
        """Handle Azure AD Multi-Factor Authentication.

        Supports PhoneAppOTP (authenticator TOTP code) and
        PhoneAppNotification (push notification with polling).

        Returns the response after MFA completion, or None on failure.
        """
        flow_token = config.get("sFT", "")
        ctx = config.get("sCtx", "")
        url_begin = self._resolve_azure_url(
            config.get(
                "urlBeginAuth",
                "https://login.microsoftonline.com/common/SAS/BeginAuth",
            )
        )
        url_end = self._resolve_azure_url(
            config.get(
                "urlEndAuth",
                "https://login.microsoftonline.com/common/SAS/EndAuth",
            )
        )
        url_process = self._resolve_azure_url(
            config.get(
                "urlPost",
                "https://login.microsoftonline.com/common/SAS/ProcessAuth",
            )
        )

        proofs = config.get("arrUserProofs", [])
        default_method = None
        for p in proofs:
            if p.get("isDefault"):
                default_method = p.get("authMethodId")
                break
        if not default_method and proofs:
            default_method = proofs[0].get("authMethodId")

        logger.info("MFA required (method: %s)", default_method)

        # Step 1: Begin MFA authentication
        begin_data = {
            "AuthMethodId": default_method,
            "Method": "BeginAuth",
            "ctx": ctx,
            "flowToken": flow_token,
        }
        try:
            begin_resp = session.post(url_begin, json=begin_data, timeout=30)
            begin_result = begin_resp.json()
        except Exception as e:
            logger.error("MFA BeginAuth failed: %s", e)
            return None

        if not begin_result.get("Success"):
            logger.error("MFA BeginAuth rejected: %s", begin_result.get("Message", ""))
            return None

        mfa_flow_token = begin_result.get("FlowToken", flow_token)
        mfa_ctx = begin_result.get("Ctx", ctx)
        session_id = begin_result.get("SessionId", "")

        # Step 2: Get OTP code from user
        if default_method in ("PhoneAppOTP", "OneWaySMS"):
            otp_code = self._prompt_mfa_code()
            if not otp_code:
                logger.error("MFA: No OTP code provided")
                return None
        elif default_method == "PhoneAppNotification":
            # Push notification — poll for approval
            logger.info("MFA: Check your authenticator app for approval")
            otp_code = None
            max_polls = config.get("iMaxPollAttempts", 12)
            for poll in range(max_polls):
                time.sleep(5)
                poll_data = {
                    "AuthMethodId": default_method,
                    "Method": "EndAuth",
                    "SessionId": session_id,
                    "FlowToken": mfa_flow_token,
                    "Ctx": mfa_ctx,
                    "PollCount": poll + 1,
                }
                try:
                    poll_resp = session.post(url_end, json=poll_data, timeout=30)
                    poll_result = poll_resp.json()
                except Exception as e:
                    logger.warning("MFA poll %d failed: %s", poll + 1, e)
                    continue

                if poll_result.get("Success"):
                    mfa_flow_token = poll_result.get("FlowToken", mfa_flow_token)
                    mfa_ctx = poll_result.get("Ctx", mfa_ctx)
                    logger.info("MFA push approved")
                    break
                if poll_result.get("Retry", True) is False:
                    logger.error("MFA push denied")
                    return None
            else:
                logger.error("MFA push approval timed out")
                return None
        else:
            logger.error("Unsupported MFA method: %s", default_method)
            return None

        # Step 3: End MFA verification (for OTP methods)
        if otp_code is not None:
            end_data = {
                "AuthMethodId": default_method,
                "Method": "EndAuth",
                "SessionId": session_id,
                "FlowToken": mfa_flow_token,
                "Ctx": mfa_ctx,
                "AdditionalAuthData": otp_code,
            }
            try:
                end_resp = session.post(url_end, json=end_data, timeout=30)
                end_result = end_resp.json()
            except Exception as e:
                logger.error("MFA EndAuth failed: %s", e)
                return None

            if not end_result.get("Success"):
                logger.error(
                    "MFA verification failed: %s",
                    end_result.get("Message", "Invalid code"),
                )
                return None

            mfa_flow_token = end_result.get("FlowToken", mfa_flow_token)
            mfa_ctx = end_result.get("Ctx", mfa_ctx)

        # Step 4: Process MFA completion
        process_data = {
            "type": 19,
            "GeneralVerify": False,
            "request": mfa_ctx,
            "mfaLastPollStart": "",
            "mfaLastPollEnd": "",
            "mfaPollingInterval": "",
            "flowToken": mfa_flow_token,
            "canary": config.get("canary", ""),
            "login": "",
            "hpgrequestid": "",
        }
        try:
            response = session.post(
                url_process,
                data=process_data,
                timeout=60,
                allow_redirects=False,
                headers=_FORM_HEADERS,
            )
            logger.debug("MFA ProcessAuth → status %d", response.status_code)
            return response
        except Exception as e:
            logger.error("MFA ProcessAuth failed: %s", e)
            return None

    def _prompt_mfa_code(self) -> str | None:
        """Prompt user for MFA verification code.

        Returns:
            OTP code string, or None if unable to prompt.
        """
        import sys

        if not sys.stdin.isatty():
            logger.error(
                "MFA code required but no interactive terminal. "
                "Run interactively or use a session with valid cookies."
            )
            return None

        try:
            code = input("\nMFA verification code from authenticator app: ").strip()
            return code if code else None
        except (EOFError, KeyboardInterrupt):
            return None

    @staticmethod
    def _resolve_azure_url(url: str) -> str:
        """Resolve a possibly-relative Azure AD URL to absolute."""
        if url and url.startswith("/"):
            return f"https://login.microsoftonline.com{url}"
        return url

    def _handle_kmsi(
        self, session: requests.Session, config: dict
    ) -> requests.Response | None:
        """Handle Azure AD KMSI (Keep Me Signed In) prompt.

        Automatically accepts "Stay signed in?" to get longer sessions.

        Returns the response after KMSI, or None on failure.
        """
        flow_token = config.get("sFT", "")
        ctx = config.get("sCtx", "")
        canary = config.get("canary", "")
        url_post = self._resolve_azure_url(config.get("urlPost", ""))

        if not url_post:
            logger.warning("KMSI page missing urlPost")
            return None

        # Accept KMSI — value "0" means "Yes, stay signed in"
        kmsi_data = {
            "LoginOptions": "1",
            "type": "28",
            "ctx": ctx,
            "hpgrequestid": "",
            "flowToken": flow_token,
            "canary": canary,
            "i19": "2326",
        }

        try:
            response = session.post(
                url_post,
                data=kmsi_data,
                timeout=60,
                allow_redirects=False,
                headers=_FORM_HEADERS,
            )
            logger.debug("KMSI submitted → status %d", response.status_code)
            return response
        except Exception as e:
            logger.warning("KMSI submission failed: %s", e)
            return None

    def _extract_azure_config(self, html: str) -> dict | None:
        """Extract $Config JSON from Azure AD login page.

        Azure AD embeds authentication parameters as a JavaScript variable:
        $Config={...};

        Uses bracket-depth counting to handle nested objects.
        """
        idx = html.find("$Config=")
        if idx == -1:
            return None

        json_start = idx + len("$Config=")
        if json_start >= len(html) or html[json_start] != "{":
            return None

        # Find matching closing brace with depth tracking
        depth = 0
        i = json_start
        in_string = False
        while i < len(html):
            c = html[i]
            if in_string:
                if c == "\\":
                    i += 1  # Skip escaped character
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1

        if depth != 0:
            return None

        try:
            return json.loads(html[json_start : i + 1])
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse Azure AD config: %s", e)
            return None

    def _extract_auto_submit_form(
        self, html: str, current_url: str
    ) -> tuple[str | None, dict]:
        """Extract action URL and fields from an auto-submit HTML form.

        Detects forms with JavaScript auto-submit patterns used in SAML
        SSO flows (assertion forwarding, KMSI prompts, etc.).
        """
        html_lower = html.lower()
        if "<form" not in html_lower:
            return None, {}

        has_autosubmit = any(
            p in html_lower
            for p in [
                "document.forms[0].submit()",
                ".submit()",
                "onload",
            ]
        )
        if not has_autosubmit:
            return None, {}

        form_match = re.search(
            r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if not form_match:
            return None, {}

        action = form_match.group(1).replace("&amp;", "&")
        body = form_match.group(2)

        fields = dict(
            re.findall(
                r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
                body,
                re.IGNORECASE,
            )
        )
        if not fields:
            return None, {}

        # Resolve relative URLs
        if action.startswith("/"):
            parsed = urlparse(current_url)
            action = f"{parsed.scheme}://{parsed.netloc}{action}"
        elif not action.startswith("http"):
            from urllib.parse import urljoin

            action = urljoin(current_url, action)

        return action, fields

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
