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
from typing import TYPE_CHECKING

from imas_codex.discovery.facility import get_facility
from imas_codex.remote.executor import run_python_script

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Patterns for extracting username from home directory paths
HOME_PATH_PATTERNS = [
    re.compile(r"^/home/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /home/username
    re.compile(r"^/users/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /users/username
    re.compile(r"^/u/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /u/username
    re.compile(r"^/work/([a-z0-9_-]+)(?:/|$)", re.IGNORECASE),  # /work/username
]

# ITER uses "Last First [EXT]" format
ITER_NAME_PATTERN = re.compile(r"^(\S+)\s+(.+?)(?:\s+EXT)?$")

# Standard "First Last" format (EPFL, most others)
STANDARD_NAME_PATTERN = re.compile(r"^(.+?)\s+(\S+)$")


@dataclass
class UserInfo:
    """Extracted user information."""

    username: str
    name: str  # Full name from GECOS
    given_name: str | None = None
    family_name: str | None = None
    home_path: str | None = None
    source: str = "getent"  # getent, passwd, id


def extract_username_from_path(path: str) -> str | None:
    """Extract username from a home directory path.

    Args:
        path: Directory path (e.g., /home/dubrovm/codes)

    Returns:
        Username if pattern matches, None otherwise
    """
    for pattern in HOME_PATH_PATTERNS:
        match = pattern.match(path)
        if match:
            return match.group(1)
    return None


def parse_name_iter(gecos: str) -> tuple[str | None, str | None]:
    """Parse ITER-style name: "Last First [EXT]".

    Args:
        gecos: GECOS field value (e.g., "Dubrov Maksim EXT")

    Returns:
        (given_name, family_name) tuple
    """
    if not gecos:
        return None, None

    match = ITER_NAME_PATTERN.match(gecos.strip())
    if match:
        family_name = match.group(1)
        given_name = match.group(2)
        return given_name, family_name

    return None, None


def parse_name_standard(gecos: str) -> tuple[str | None, str | None]:
    """Parse standard name: "First Last".

    Args:
        gecos: GECOS field value (e.g., "Alessandro Balestri")

    Returns:
        (given_name, family_name) tuple
    """
    if not gecos:
        return None, None

    match = STANDARD_NAME_PATTERN.match(gecos.strip())
    if match:
        given_name = match.group(1)
        family_name = match.group(2)
        return given_name, family_name

    return None, None


def get_name_parser(facility_id: str):
    """Get the appropriate name parser for a facility.

    Args:
        facility_id: Facility identifier

    Returns:
        Name parser function
    """
    # ITER uses "Last First" format
    if facility_id == "iter":
        return parse_name_iter
    # Most facilities use "First Last"
    return parse_name_standard


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
) -> list[dict]:
    """Extract and enrich users from a list of paths.

    This is the main entry point for user enrichment during scan phase.
    Extracts unique usernames from paths, fetches their info, and returns
    FacilityUser-compatible dicts.

    Args:
        facility: Facility identifier
        paths: List of directory paths to extract users from

    Returns:
        List of dicts suitable for add_to_graph("FacilityUser", ...)
    """
    # Extract unique usernames from paths
    usernames: set[str] = set()
    username_to_paths: dict[str, list[str]] = {}

    for path in paths:
        username = extract_username_from_path(path)
        if username:
            usernames.add(username)
            if username not in username_to_paths:
                username_to_paths[username] = []
            username_to_paths[username].append(path)

    if not usernames:
        return []

    logger.info(f"Enriching {len(usernames)} users for {facility}")

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
            user_dict["home_path"] = info.home_path
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
