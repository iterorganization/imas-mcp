"""ORCID integration for researcher identity verification.

ORCID (Open Researcher and Contributor ID) provides persistent identifiers
for researchers, enabling cross-facility identity matching even when
emails and usernames vary.

Supports phonetic matching for non-Latin names (e.g., Japanese katakana)
via pykakasi (Japanese→romaji) and jellyfish (Soundex).

API Documentation: https://info.orcid.org/documentation/api-tutorials/
"""

from __future__ import annotations

import logging
import unicodedata

import httpx

logger = logging.getLogger(__name__)

# Lazy-loaded phonetic matching dependencies
_kakasi = None
_jellyfish = None


def _get_kakasi():
    """Lazy load pykakasi for Japanese→romaji conversion."""
    global _kakasi
    if _kakasi is None:
        import pykakasi

        _kakasi = pykakasi.kakasi()
    return _kakasi


def _get_jellyfish():
    """Lazy load jellyfish for phonetic matching."""
    global _jellyfish
    if _jellyfish is None:
        import jellyfish

        _jellyfish = jellyfish
    return _jellyfish


def _contains_non_latin(text: str) -> bool:
    """Check if text contains non-Latin characters (CJK, etc.)."""
    for char in text:
        if unicodedata.category(char).startswith("L"):
            # Letter category - check if Latin
            name = unicodedata.name(char, "")
            if not name.startswith("LATIN"):
                return True
    return False


def _transliterate_to_romaji(text: str) -> str:
    """Convert Japanese text to romaji using Hepburn romanization."""
    kks = _get_kakasi()
    return "".join([item["hepburn"] for item in kks.convert(text)])


def soundex_match(name1: str, name2: str) -> bool:
    """Check if two names match phonetically using Soundex.

    Soundex reduces names to a phonetic code (e.g., "Simon" -> "S550").
    Useful for matching transliterated names like "saimon" == "Simon".
    """
    jf = _get_jellyfish()
    return jf.soundex(name1.lower()) == jf.soundex(name2.lower())


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
                results = data.get("result") or []

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
    2. Direct name search
    3. Phonetic name search (for non-Latin scripts like Japanese katakana)

    For non-Latin names, transliterates to romaji and uses Soundex matching
    to find phonetically equivalent names in ORCID.

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
            logger.info(
                f"Found {len(matches)} ORCID matches for {given_name} {family_name}"
            )
            return matches[0]["orcid"]

        # No direct match - try phonetic matching for non-Latin names
        if _contains_non_latin(given_name) or _contains_non_latin(family_name):
            orcid = await _phonetic_orcid_search(client, given_name, family_name)
            if orcid:
                return orcid

    return None


async def _phonetic_orcid_search(
    client: ORCIDClient,
    given_name: str,
    family_name: str,
) -> str | None:
    """Search ORCID using phonetic matching for non-Latin names.

    Converts names to romaji and uses Soundex to find phonetic matches.

    Args:
        client: ORCID API client
        given_name: Given name (may be non-Latin)
        family_name: Family name (may be non-Latin)

    Returns:
        ORCID ID if a phonetic match is found, None otherwise
    """
    # Transliterate to romaji
    romaji_given = _transliterate_to_romaji(given_name)
    romaji_family = _transliterate_to_romaji(family_name)

    logger.debug(
        f"Phonetic search: {given_name} {family_name} -> {romaji_given} {romaji_family}"
    )

    # Search ORCID with romanized names
    matches = await client.search_by_name(romaji_given, romaji_family)

    if matches:
        logger.info(
            f"Found {len(matches)} phonetic ORCID matches for "
            f"{romaji_given} {romaji_family}"
        )
        return matches[0]["orcid"]

    # If still no matches, try broader search with just family name
    # and filter by Soundex on given name
    broad_matches = await _broad_phonetic_search(client, romaji_given, romaji_family)
    if broad_matches:
        return broad_matches[0]

    return None


def _romaji_to_wildcard(romaji: str) -> str:
    """Convert romaji to a wildcard pattern for ORCID search.

    Strategy: Extract key consonant clusters that are likely preserved
    across transliteration variants.

    E.g., "makkintosshu" -> "*ntosh*" (extracts "ntosh" from the middle)
         "saimon" -> "*imon*" (captures "-imon" ending)

    Args:
        romaji: Romanized Japanese name

    Returns:
        Wildcard pattern for ORCID search
    """
    import re

    romaji = romaji.lower()

    # Remove doubled consonants (common in Japanese transliteration)
    # makkintosshu -> makintoshu
    simplified = re.sub(r"(.)\1+", r"\1", romaji)

    # Remove trailing 'u' (Japanese adds vowels after consonants)
    # makintoshu -> makintosh
    if simplified.endswith("u") and len(simplified) > 3:
        simplified = simplified[:-1]

    # For longer names, extract a middle portion with consonant clusters
    if len(simplified) >= 6:
        # Find consonant+vowel+consonant patterns (most distinctive)
        # Take chars from position 40% to 90% of the string
        start = len(simplified) * 2 // 5
        end = len(simplified) * 9 // 10
        core = simplified[start:end]
        if len(core) >= 3:
            return f"*{core}*"

    # For shorter names, use vowel+consonant ending
    if len(simplified) >= 4:
        return f"*{simplified[-4:]}*"

    return f"*{simplified}*"


async def _broad_phonetic_search(
    client: ORCIDClient,
    romaji_given: str,
    romaji_family: str,
) -> list[str]:
    """Broader phonetic search using wildcards and Soundex filtering.

    Uses wildcard ORCID queries derived from romaji, then filters
    candidates by Soundex match.
    """
    try:
        async with httpx.AsyncClient(timeout=client.timeout) as http_client:
            # Convert romaji to wildcard patterns
            given_pattern = _romaji_to_wildcard(romaji_given)
            family_pattern = _romaji_to_wildcard(romaji_family)

            # Search with wildcards
            query = f"given-names:{given_pattern} AND family-name:{family_pattern}"
            logger.debug(f"Wildcard ORCID search: {query}")

            response = await http_client.get(
                f"{client.BASE_URL}/search",
                params={"q": query, "rows": 20},
                headers=client.headers,
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("result") or []

            if not results:
                logger.debug(f"No wildcard matches for {query}")
                return []

            matches = []
            for result in results:
                orcid_id = result.get("orcid-identifier", {}).get("path")
                if not orcid_id:
                    continue

                # Fetch the record to get full name
                record = await client.get_record(orcid_id)
                if not record:
                    continue

                # Extract registered name from record
                person = record.get("person", {})
                name_data = person.get("name", {})
                reg_given = name_data.get("given-names", {}).get("value", "")
                reg_family = name_data.get("family-name", {}).get("value", "")

                if not reg_given or not reg_family:
                    continue

                # Check Soundex match on both names
                if soundex_match(romaji_given, reg_given) and soundex_match(
                    romaji_family, reg_family
                ):
                    logger.info(
                        f"Phonetic match: {romaji_given} {romaji_family} "
                        f"-> {reg_given} {reg_family} ({orcid_id})"
                    )
                    matches.append(orcid_id)

            return matches

    except httpx.HTTPError as e:
        logger.warning(f"Broad phonetic search failed: {e}")
        return []
