"""Lifecycle extractor for IMAS data dictionary transformation."""

import xml.etree.ElementTree as ET
from typing import Any

from imas_mcp.core.extractors.base import BaseExtractor


class LifecycleExtractor(BaseExtractor):
    """Extract lifecycle and version information."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
        """Extract lifecycle metadata."""
        lifecycle_data = {}

        # Check for lifecycle status
        lifecycle_status = elem.get("lifecycle_status")
        if lifecycle_status:
            lifecycle_data["lifecycle"] = lifecycle_status

        # Check for lifecycle version
        lifecycle_version = elem.get("lifecycle_version")
        if lifecycle_version:
            lifecycle_data["lifecycle_version"] = lifecycle_version

        return lifecycle_data
