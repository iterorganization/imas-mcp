"""ORCID integration for researcher identity verification.

ORCID (Open Researcher and Contributor ID) provides persistent identifiers
for researchers, enabling cross-facility identity matching even when
emails and usernames vary.

API Documentation: https://info.orcid.org/documentation/api-tutorials/
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class ORCIDClient:
    """Client for ORCID public API v3.0."""

    BASE_URL = "https://pub.orcid.org/v3.0"

    def __init__(self, timeout: float = 10.0):
        """Initialize ORCID client.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "IMAS-Codex/1.0",
        }

    async def search_by_email(self, email: str) -> str | None:
        """Search ORCID by email address.

        Args:
            email: Email address to search

        Returns:
            ORCID ID (e.g., "0000-0002-1825-0097") or None
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/search",
                    params={"q": f"email:{email}"},
                    headers=self.headers,
                )
                response.raise_for_status()

                data = response.json()
                results = data.get("result", [])

                if results:
                    # Return first match (most relevant)
                    orcid_path = results[0].get("orcid-identifier", {}).get("path")
                    logger.debug(f"Found ORCID {orcid_path} for email {email}")
                    return orcid_path

                logger.debug(f"No ORCID found for email {email}")
                return None

        except httpx.HTTPError as e:
            logger.warning(f"ORCID API error for email {email}: {e}")
            return None

    async def search_by_name(self, given_name: str, family_name: str) -> list[dict]:
        """Search ORCID by researcher name.

        Args:
            given_name: Given/first name
            family_name: Family/last name

        Returns:
            List of matches with ORCID IDs and metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                query = f"given-names:{given_name} AND family-name:{family_name}"
                response = await client.get(
                    f"{self.BASE_URL}/search",
                    params={"q": query, "rows": 10},
                    headers=self.headers,
                )
                response.raise_for_status()

                data = response.json()
                results = data.get("result", [])

                matches = []
                for result in results:
                    orcid_id = result.get("orcid-identifier", {}).get("path")
                    if orcid_id:
                        matches.append(
                            {
                                "orcid": orcid_id,
                                "given_name": given_name,
                                "family_name": family_name,
                            }
                        )

                logger.debug(
                    f"Found {len(matches)} ORCID matches for {given_name} {family_name}"
                )
                return matches

        except httpx.HTTPError as e:
            logger.warning(f"ORCID API error for name {given_name} {family_name}: {e}")
            return []

    async def get_record(self, orcid_id: str) -> dict | None:
        """Fetch full ORCID record.

        Args:
            orcid_id: ORCID ID (e.g., "0000-0002-1825-0097")

        Returns:
            Record metadata or None
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/{orcid_id}/record",
                    headers=self.headers,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.warning(f"ORCID API error for ID {orcid_id}: {e}")
            return None


async def enrich_person_with_orcid(
    email: str | None = None,
    given_name: str | None = None,
    family_name: str | None = None,
) -> str | None:
    """Lookup ORCID ID for a person.

    Priority:
    1. Email search (most reliable)
    2. Name search (may have multiple matches)

    Args:
        email: Email address
        given_name: Given/first name
        family_name: Family/last name

    Returns:
        ORCID ID or None

    Example:
        >>> import asyncio
        >>> orcid = asyncio.run(enrich_person_with_orcid(
        ...     email="researcher@example.org",
        ...     given_name="Jane",
        ...     family_name="Smith"
        ... ))
        >>> print(orcid)  # "0000-0002-1825-0097" or None
    """
    client = ORCIDClient()

    # Try email first (most accurate)
    if email:
        orcid = await client.search_by_email(email)
        if orcid:
            return orcid

    # Fall back to name search
    if given_name and family_name:
        matches = await client.search_by_name(given_name, family_name)
        if matches:
            # Return first match (ambiguous if multiple)
            # TODO: Add disambiguation logic or manual confirmation
            logger.info(
                f"Found {len(matches)} ORCID matches for {given_name} {family_name}"
            )
            return matches[0]["orcid"]

    return None
