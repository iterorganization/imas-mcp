"""Lifecycle and physics context extractors."""

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


class PhysicsExtractor(BaseExtractor):
    """Extract physics context and domain information."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
        """Extract physics context from element."""
        physics_data = {}

        # Infer physics context
        physics_context_str = self._infer_element_physics_context(elem)
        if physics_context_str:
            physics_data["physics_context"] = {
                "domain": physics_context_str,
                "phenomena": [physics_context_str],
                "typical_values": {},
            }

        # Extract timebase information
        timebase = elem.get("timebasepath")
        if timebase:
            physics_data["temporal_context"] = timebase

        return physics_data

    def _infer_element_physics_context(self, elem: ET.Element) -> str | None:
        """Infer physics context from element name and path."""
        elem_name = elem.get("name", "").lower()
        path = elem.get("path", "").lower()

        # Physics domain mapping
        physics_patterns = {
            "equilibrium": {
                "magnetic_field": [
                    "b_tor",
                    "b_pol",
                    "b0",
                    "vacuum_toroidal_field",
                    "magnetic",
                ],
                "geometry": [
                    "r0",
                    "geometric_axis",
                    "boundary",
                    "outline",
                    "elongation",
                ],
                "flux_surfaces": ["psi", "psi_norm", "rho_tor", "flux"],
                "global_quantities": ["ip", "beta", "li", "volume", "area"],
            },
            "core_profiles": {
                "kinetic": ["density", "temperature", "pressure", "electrons", "ion"],
                "transport": ["d_eff", "v_eff", "flux", "conductivity"],
                "rotation": ["rotation", "omega", "velocity"],
            },
        }

        # Get patterns for current IDS
        ids_patterns = physics_patterns.get(self.context.ids_name, {})

        # Check element name and path against patterns
        for domain, patterns in ids_patterns.items():
            if any(pattern in elem_name or pattern in path for pattern in patterns):
                return domain

        # Fallback coordinate-based inference
        coord1 = elem.get("coordinate1")
        if coord1 == "time":
            return "temporal_evolution"
        elif coord1 in ["rho_tor_norm", "psi_norm"]:
            return "radial_profile"
        elif coord1 in ["r", "z"]:
            return "spatial_geometry"

        return None
