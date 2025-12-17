"""Physics domain definitions for IDS categorization."""

from pathlib import Path

DOMAINS_SCHEMA = Path(__file__).parent / "domains.yaml"

__all__ = ["DOMAINS_SCHEMA"]
