"""Physics domain definitions for IDS categorization."""

from pathlib import Path

# LinkML schema defining the PhysicsDomain enum values
DOMAINS_SCHEMA = Path(__file__).parent / "domains.yaml"

# Version-independent IDS to physics domain mappings (covers all DD versions)
IDS_DOMAINS_FILE = Path(__file__).parent / "ids_domains.json"

__all__ = ["DOMAINS_SCHEMA", "IDS_DOMAINS_FILE"]
