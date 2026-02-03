"""
User enrichment for facility discovery.

Extracts user information from home directory paths and enriches with
GECOS data via getent passwd. Creates FacilityUser nodes and links
them to Person nodes for cross-facility identification.

Design:
- Run during scan phase for each discovered home directory
- Uses get_user_info.py remote script with cascading fallbacks
- Handles facility-specific name formats (ITER: "Last First", EPFL: "First Last")
- Continuous deduplication via Person node matching

Cross-facility linking strategies:
1. ORCID (if available via config or public sources)
2. Normalized name matching (given_name + family_name)
3. Email matching (if discoverable)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.facility import get_facility
from imas_codex.remote.executor import run_python_script

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Patterns for extracting username from home directory paths
# Priority order: more specific patterns first
HOME_PATH_PATTERNS = [
    re.compile(
        r"^/home/ITER/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE
    ),  # /home/ITER/username
    re.compile(r"^/home/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /home/username
    re.compile(r"^/users/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /users/username
    re.compile(r"^/u/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /u/username
    re.compile(r"^/work/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /work/username
]


@dataclass
class UserInfo:
    """Extracted user information."""

    username: str
    name: str  # Full name from GECOS
    given_name: str | None = None
    family_name: str | None = None
    home_path: str | None = None
    source: str = "getent"  # getent, passwd, id


def extract_username_from_path(path: str, facility_id: str | None = None) -> str | None:
    """Extract username from a home directory path.

    Checks facility-specific pattern first (from config), then falls back
    to standard patterns.

    Args:
        path: Directory path (e.g., /home/ITER/dubrovm/codes)
        facility_id: Optional facility ID to check for custom home_path_pattern

    Returns:
        Username if pattern matches, None otherwise
    """
    # Check facility-specific pattern first
    if facility_id:
        try:
            config = get_facility(facility_id)
            user_info = config.get("user_info", {})
            custom_pattern = user_info.get("home_path_pattern")
            if custom_pattern:
                compiled = re.compile(custom_pattern, re.IGNORECASE)
                match = compiled.match(path)
                if match:
                    return match.group(1)
        except (ValueError, KeyError):
            pass

    # Fall back to standard patterns
    for pattern in HOME_PATH_PATTERNS:
        match = pattern.match(path)
        if match:
            return match.group(1)
    return None


def get_name_parser(facility_id: str):
    """Get the appropriate name parser for a facility from config.

    Reads user_info.name_format and user_info.gecos_suffix_pattern from
    the facility YAML config exposed via get_facility().

    Args:
        facility_id: Facility identifier

    Returns:
        Name parser function that returns (given_name, family_name) tuple
    """
    # Read parsing config from facility YAML
    try:
        config = get_facility(facility_id)
        user_info = config.get("user_info", {})
    except (ValueError, KeyError):
        user_info = {}

    name_format = user_info.get("name_format", "first_last")
    suffix_pattern = user_info.get("gecos_suffix_pattern")

    # Compile suffix pattern if provided
    suffix_regex = re.compile(suffix_pattern) if suffix_pattern else None

    def parser(gecos: str) -> tuple[str | None, str | None]:
        """Parse GECOS name field based on facility config."""
        if not gecos:
            return None, None

        text = gecos.strip()

        # Apply suffix pattern if configured (e.g., strip " EXT" from ITER names)
        if suffix_regex:
            text = suffix_regex.sub("", text).strip()

        # Parse based on configured format
        if name_format == "last_first":
            # ITER-style: "Last First" -> (given, family)
            parts = text.split(None, 1)
            if len(parts) == 2:
                return parts[1], parts[0]  # given=second, family=first
        else:
            # Standard: "First Last" -> (given, family)
            parts = text.rsplit(None, 1)
            if len(parts) == 2:
                return parts[0], parts[1]  # given=first, family=last

        return None, None

    return parser


def fetch_user_info(
    facility: str,
    usernames: list[str],
    timeout: int = 30,
) -> dict[str, UserInfo]:
    """Fetch user info for multiple usernames via remote script.

    Args:
        facility: Facility identifier for SSH/local execution
        usernames: List of usernames to look up
        timeout: Command timeout in seconds

    Returns:
        Dict mapping username -> UserInfo
    """
    if not usernames:
        return {}

    # Resolve ssh_host from facility config
    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except ValueError:
        ssh_host = facility

    input_data = {"usernames": usernames}

    try:
        output = run_python_script(
            "get_user_info.py",
            input_data=input_data,
            ssh_host=ssh_host,
            timeout=timeout,
        )
    except Exception as e:
        logger.warning(f"User info fetch failed for {facility}: {e}")
        return {}

    # Parse JSON output
    try:
        if "[stderr]:" in output:
            output = output.split("[stderr]:")[0].strip()
        data = json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse user info output: {e}")
        return {}

    # Get name parser for this facility
    parse_name = get_name_parser(facility)

    # Convert to UserInfo objects
    results: dict[str, UserInfo] = {}
    for user_data in data.get("users", []):
        username = user_data.get("username")
        if not username:
            continue

        name = user_data.get("name", "")
        given_name, family_name = parse_name(name)

        results[username] = UserInfo(
            username=username,
            name=name,
            given_name=given_name,
            family_name=family_name,
            home_path=user_data.get("home"),
            source=user_data.get("source", "getent"),
        )

    # Log errors
    for error in data.get("errors", []):
        logger.debug(f"User lookup error: {error}")

    return results


def enrich_users_from_paths(
    facility: str,
    paths: list[str],
    gc: Any | None = None,
) -> list[dict]:
    """Extract and enrich users from a list of paths.

    This is the main entry point for user enrichment during scan phase.
    Extracts unique usernames from paths, fetches their info, and returns
    FacilityUser-compatible dicts.

    Args:
        facility: Facility identifier
        paths: List of directory paths to extract users from
        gc: Optional GraphClient instance (skips users already in graph)

    Returns:
        List of dicts suitable for add_to_graph("FacilityUser", ...)
    """
    # Extract unique usernames from paths
    usernames: set[str] = set()
    username_to_paths: dict[str, list[str]] = {}

    for path in paths:
        username = extract_username_from_path(path, facility)
        if username:
            usernames.add(username)
            if username not in username_to_paths:
                username_to_paths[username] = []
            username_to_paths[username].append(path)

    if not usernames:
        return []

    # Skip users already in the graph (optimization for incremental discovery)
    if gc is not None:
        try:
            user_ids = [f"{facility}:{u}" for u in usernames]
            existing = gc.query(
                "UNWIND $ids AS id MATCH (u:FacilityUser {id: id}) RETURN u.username AS username",
                ids=user_ids,
            )
            existing_usernames = {r["username"] for r in existing}
            usernames = usernames - existing_usernames
            if not usernames:
                return []
        except Exception:
            pass  # Continue with all usernames if query fails

    logger.debug(f"Enriching {len(usernames)} new users for {facility}")

    # Fetch user info
    user_info = fetch_user_info(facility, list(usernames))

    # Build FacilityUser dicts
    now = datetime.now(UTC).isoformat()
    facility_users = []

    for username in usernames:
        info = user_info.get(username)

        user_dict = {
            "id": f"{facility}:{username}",
            "facility_id": facility,
            "username": username,
            "discovered_at": now,
        }

        if info:
            user_dict["name"] = info.name
            user_dict["given_name"] = info.given_name
            user_dict["family_name"] = info.family_name
            # home_path_id for OWNS relationship (facility:path format)
            if info.home_path:
                user_dict["home_path_id"] = f"{facility}:{info.home_path}"
            user_dict["enriched_at"] = now

        facility_users.append(user_dict)

    return facility_users


def normalize_name(given_name: str | None, family_name: str | None) -> str | None:
    """Create normalized name key for deduplication.

    Args:
        given_name: First name
        family_name: Last name

    Returns:
        Normalized key (lowercase, no diacritics) or None
    """
    if not given_name or not family_name:
        return None

    # Lowercase and normalize
    given = given_name.lower().strip()
    family = family_name.lower().strip()

    # Remove common diacritics (basic ASCII folding)
    # For full support, use unicodedata.normalize + ascii encoding
    import unicodedata

    given = unicodedata.normalize("NFKD", given).encode("ascii", "ignore").decode()
    family = unicodedata.normalize("NFKD", family).encode("ascii", "ignore").decode()

    return f"{given}|{family}"


def find_or_create_person(
    facility_user: dict,
) -> dict | None:
    """Find or create a Person node for cross-facility linking.

    Uses continuous deduplication: if a Person with matching name or
    ORCID exists, return it; otherwise create a new one.

    Args:
        facility_user: FacilityUser dict with name info

    Returns:
        Person dict suitable for add_to_graph, or None if no name info
    """
    given_name = facility_user.get("given_name")
    family_name = facility_user.get("family_name")
    name = facility_user.get("name", "").strip()

    if not name and not (given_name and family_name):
        return None

    # Build Person dict
    # Use normalized name as ID for deduplication
    normalized = normalize_name(given_name, family_name)
    if normalized:
        person_id = f"person:{normalized}"
    else:
        # Fallback to full name hash
        import hashlib

        name_hash = hashlib.sha256(name.encode()).hexdigest()[:12]
        person_id = f"person:{name_hash}"

    person = {
        "id": person_id,
        "name": name or f"{given_name} {family_name}",
        "given_name": given_name,
        "family_name": family_name,
    }

    return person
