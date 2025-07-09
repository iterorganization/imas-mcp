"""Relationship extraction with composable analyzers and performance optimizations."""

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from collections import defaultdict

from .base import BaseExtractor


class RelationshipAnalyzer:
    """Base class for relationship analysis strategies."""

    def __init__(self, context):
        self.context = context
        # Initialize indexes for efficient lookups
        self._path_index: Dict[str, ET.Element] = {}
        self._coordinate_index: Dict[str, List[ET.Element]] = defaultdict(list)
        self._units_index: Dict[str, List[ET.Element]] = defaultdict(list)
        self._built_indexes = False

    def _build_indexes(self):
        """Build indexes once for efficient relationship lookups."""
        if self._built_indexes:
            return

        self._path_index = {}
        self._coordinate_index = defaultdict(list)
        self._units_index = defaultdict(list)

        # Single traversal to build all indexes
        for elem in self.context.ids_elem.iter():
            # Path index
            path = elem.get("path")
            if path:
                self._path_index[path] = elem

            # Coordinate indexes
            coord1 = elem.get("coordinate1")
            coord2 = elem.get("coordinate2")
            if coord1:
                self._coordinate_index[coord1].append(elem)
            if coord2:
                self._coordinate_index[coord2].append(elem)

            # Units index
            units = elem.get("units")
            if units:
                self._units_index[units].append(elem)

        self._built_indexes = True

    def find_relationships(self, elem: ET.Element, current_path: str) -> List[str]:
        """Find relationships for an element."""
        self._build_indexes()
        raise NotImplementedError


class CrossIdsAnalyzer(RelationshipAnalyzer):
    """Analyze cross-IDS references."""

    def find_relationships(self, elem: ET.Element, current_path: str) -> List[str]:
        """Extract cross-IDS references."""
        cross_refs = []

        # Check coordinates for IDS references
        coord1 = elem.get("coordinate1")
        coord2 = elem.get("coordinate2")

        for coord in [coord1, coord2]:
            if coord and coord.startswith("IDS:"):
                cross_refs.append(coord)

        # Check structure references
        structure_ref = elem.get("structure_reference")
        if structure_ref and ":" in structure_ref:
            cross_refs.append(f"structure:{structure_ref}")

        # Extract from documentation
        documentation = elem.get("documentation", "") or elem.text or ""
        if "IDS:" in documentation:
            ids_patterns = re.findall(r"IDS:[\w/]+", documentation)
            cross_refs.extend(ids_patterns)

        return cross_refs


class HierarchicalAnalyzer(RelationshipAnalyzer):
    """Analyze hierarchical (parent-child-sibling) relationships."""

    def find_relationships(self, elem: ET.Element, current_path: str) -> List[str]:
        """Extract hierarchical relationships using indexes."""
        self._build_indexes()
        relationships = []

        if not current_path:
            return relationships

        path_parts = current_path.split("/")

        # Parent relationship
        if len(path_parts) > 1:
            parent_path = "/".join(path_parts[:-1])
            if parent_path in self._path_index:
                relationships.append(parent_path)

            # Sibling relationships
            siblings = self._find_siblings(parent_path, current_path)
            relationships.extend(siblings[:5])

        # Child relationships
        children = self._find_children(current_path)
        relationships.extend(children[:5])

        return relationships

    def _find_siblings(self, parent_path: str, current_path: str) -> List[str]:
        """Find sibling paths using index - O(n) single pass instead of O(n²)."""
        siblings = []
        parent_depth = len(parent_path.split("/"))

        for path in self._path_index.keys():
            if path == current_path:
                continue

            path_parts = path.split("/")
            if len(path_parts) == parent_depth + 1:  # Same depth as current
                path_parent = "/".join(path_parts[:-1])
                if path_parent == parent_path:
                    siblings.append(path)

        return siblings

    def _find_children(self, parent_path: str) -> List[str]:
        """Find direct child paths using index."""
        children = []
        parent_depth = len(parent_path.split("/"))

        for path in self._path_index.keys():
            if path.startswith(parent_path + "/"):
                path_parts = path.split("/")
                if len(path_parts) == parent_depth + 1:  # Direct child
                    children.append(path)

        return children


class PhysicsAnalyzer(RelationshipAnalyzer):
    """Analyze physics-domain relationships."""

    def find_relationships(self, elem: ET.Element, current_path: str) -> List[str]:
        """Extract physics-based relationships using indexes."""
        self._build_indexes()
        relationships = []

        elem_name = elem.get("name", "")
        units = elem.get("units", "")

        # Skip error fields to reduce noise
        if "error" in elem_name.lower():
            return relationships

        # Group by physics concepts
        physics_groups = {
            "magnetic_field": ["b_field", "magnetic", "flux", "psi"],
            "geometry": ["r", "z", "radius", "height", "position"],
            "pressure": ["pressure", "temperature", "density"],
            "current": ["current", "j_", "conductivity"],
            "profiles": ["profile", "rho_", "norm"],
        }

        # Find physics domain
        current_domain = self._classify_physics_domain(
            elem_name, current_path, physics_groups
        )

        if current_domain:
            domain_paths = self._find_physics_domain_paths(
                current_domain, physics_groups[current_domain], current_path
            )
            relationships.extend(domain_paths[:5])

        # Unit-based relationships using index
        if units and units not in ["1", "", "mixed"]:
            unit_paths = self._find_same_unit_paths(units, current_path)
            relationships.extend(unit_paths[:3])

        return relationships

    def _classify_physics_domain(
        self, elem_name: str, current_path: str, groups: Dict[str, List[str]]
    ) -> Optional[str]:
        """Classify element into physics domain."""
        elem_name_lower = elem_name.lower()
        path_lower = current_path.lower()

        for domain, keywords in groups.items():
            if any(
                keyword in elem_name_lower or keyword in path_lower
                for keyword in keywords
            ):
                return domain

        return None

    def _find_physics_domain_paths(
        self, domain: str, keywords: List[str], current_path: str
    ) -> List[str]:
        """Find paths in same physics domain using index."""
        domain_paths = []

        for path, elem in self._path_index.items():
            if path == current_path:
                continue

            name = elem.get("name", "").lower()
            path_lower = path.lower()

            for keyword in keywords:
                if keyword in name or keyword in path_lower:
                    domain_paths.append(path)
                    break

        return list(set(domain_paths))

    def _find_same_unit_paths(self, units: str, current_path: str) -> List[str]:
        """Find paths with same units using index."""
        same_unit_paths = []

        if units in self._units_index:
            related_elements = self._units_index[units]
            for elem in related_elements:
                path = elem.get("path", "")
                if path and path != current_path:
                    same_unit_paths.append(path)

        return same_unit_paths


class CoordinateAnalyzer(RelationshipAnalyzer):
    """Analyze coordinate-based relationships."""

    def find_relationships(self, elem: ET.Element, current_path: str) -> List[str]:
        """Extract coordinate-based relationships using indexes."""
        self._build_indexes()
        relationships = []

        coord1 = elem.get("coordinate1")
        coord2 = elem.get("coordinate2")

        # Skip noisy coordinates
        if self._should_skip_coordinate(coord1) or self._should_skip_coordinate(coord2):
            return relationships

        for coord in [coord1, coord2]:
            if coord and coord in self._coordinate_index:
                related_elements = self._coordinate_index[coord]
                for related_elem in related_elements[:5]:  # Limit results
                    related_path = related_elem.get("path")
                    if related_path and related_path != current_path:
                        relationships.append(related_path)

        return relationships

    def _should_skip_coordinate(self, coord: Optional[str]) -> bool:
        """Check if coordinate should be skipped."""
        if not coord:
            return False
        return coord.startswith("ids_properties")


class RelationshipExtractor(BaseExtractor):
    """Composable relationship extractor using multiple analyzers with performance optimizations."""

    def __init__(self, context):
        super().__init__(context)

        # Build indexes once for efficient lookups
        self._path_index: Dict[str, ET.Element] = {}
        self._coordinate_index: Dict[str, List[ET.Element]] = defaultdict(list)
        self._units_index: Dict[str, List[ET.Element]] = defaultdict(list)
        self._built_indexes = False

        # Initialize analyzers
        self.analyzers = [
            CrossIdsAnalyzer(context),
            HierarchicalAnalyzer(context),
            PhysicsAnalyzer(context),
            CoordinateAnalyzer(context),
        ]

    def _build_indexes(self):
        """Build indexes once for efficient relationship lookups."""
        if self._built_indexes:
            return

        self._path_index = {}
        self._coordinate_index = defaultdict(list)
        self._units_index = defaultdict(list)

        # Single traversal to build all indexes
        for elem in self.context.ids_elem.iter():
            # Path index
            path = elem.get("path")
            if path:
                self._path_index[path] = elem

            # Coordinate indexes
            coord1 = elem.get("coordinate1")
            coord2 = elem.get("coordinate2")
            if coord1:
                self._coordinate_index[coord1].append(elem)
            if coord2:
                self._coordinate_index[coord2].append(elem)

            # Units index
            units = elem.get("units")
            if units:
                self._units_index[units].append(elem)

        self._built_indexes = True

        # Share indexes with analyzers
        for analyzer in self.analyzers:
            analyzer._path_index = self._path_index
            analyzer._coordinate_index = self._coordinate_index
            analyzer._units_index = self._units_index
            analyzer._built_indexes = True

    def extract(self, elem: ET.Element) -> Dict[str, Any]:
        """Extract relationships using all analyzers with performance optimizations."""
        self._build_indexes()

        relationship_data = {}

        current_path = elem.get("path", "")
        if not current_path:
            return relationship_data

        # Collect relationships from all analyzers
        all_relationships = []
        categorized_relationships = {
            "cross_ids": [],
            "hierarchical": [],
            "physics_related": [],
            "coordinates": [],
        }

        for i, analyzer in enumerate(self.analyzers):
            try:
                relationships = analyzer.find_relationships(elem, current_path)

                # Categorize by analyzer type
                if isinstance(analyzer, CrossIdsAnalyzer):
                    categorized_relationships["cross_ids"].extend(relationships)
                elif isinstance(analyzer, HierarchicalAnalyzer):
                    categorized_relationships["hierarchical"].extend(relationships)
                elif isinstance(analyzer, PhysicsAnalyzer):
                    categorized_relationships["physics_related"].extend(relationships)
                elif isinstance(analyzer, CoordinateAnalyzer):
                    categorized_relationships["coordinates"].extend(relationships)

                all_relationships.extend(relationships)

            except Exception as e:
                print(f"Warning: {analyzer.__class__.__name__} failed: {e}")

        # Clean and prioritize relationships
        cleaned_relationships = self._clean_and_prioritize(
            all_relationships, current_path
        )

        if cleaned_relationships:
            relationship_data["related_paths"] = cleaned_relationships[:15]

        # Add categorized relationships if any category has content (with filtering)
        filtered_categorized = {}
        for category, paths in categorized_relationships.items():
            filtered_paths = [
                p for p in paths if not self._should_filter_relationship(p)
            ]
            if filtered_paths:
                filtered_categorized[category] = filtered_paths

        if any(filtered_categorized.values()):
            relationship_data["relationships"] = filtered_categorized

        # Generate usage examples
        usage_examples = self._generate_usage_examples(elem)
        if usage_examples:
            relationship_data["usage_examples"] = usage_examples

        return relationship_data

    def _clean_and_prioritize(
        self, relationships: List[str], current_path: str
    ) -> List[str]:
        """Clean and prioritize relationships with filtering."""
        if not relationships:
            return []

        # Remove duplicates and self-references
        seen = set()
        unique_relationships = []

        for rel in relationships:
            if rel not in seen and rel != current_path:
                # Apply filtering to exclude unwanted paths
                if not self._should_filter_relationship(rel):
                    seen.add(rel)
                    unique_relationships.append(rel)

        # Sort by priority
        def priority(rel_path: str) -> int:
            if rel_path.startswith("IDS:"):
                return 1  # Cross-IDS highest priority
            elif "/" in rel_path and len(rel_path.split("/")) == len(
                current_path.split("/")
            ):
                return 2  # Siblings
            elif rel_path in current_path or current_path in rel_path:
                return 3  # Hierarchical
            else:
                return 4  # Others

        unique_relationships.sort(key=priority)
        return unique_relationships

    def _should_filter_relationship(self, path: str) -> bool:
        """Filter out unwanted relationship paths."""
        # Filter excluded patterns (ids_properties, code)
        for pattern in ["ids_properties", "code"]:
            if pattern in path:
                return True

        # Filter GGD entries
        if (
            "ggd" in path.lower()
            or "/ggd/" in path.lower()
            or "grids_ggd" in path.lower()
            or path.lower().startswith("grids_ggd")
            or "/grids_ggd/" in path.lower()
        ):
            return True

        # Filter error fields
        if (
            "_error_" in path
            or path.endswith("_error_upper")
            or path.endswith("_error_lower")
            or path.endswith("_error_index")
            or "error_upper" in path
            or "error_lower" in path
            or "error_index" in path
        ):
            return True

        return False

    def _generate_usage_examples(self, elem: ET.Element) -> List[Dict[str, str]]:
        """Generate basic usage examples."""
        examples = []

        elem_name = elem.get("name", "")
        data_type = elem.get("data_type", "")
        coord1 = elem.get("coordinate1", "")
        path = elem.get("path", elem_name)

        # Generate examples based on element characteristics
        if coord1 == "time" and "FLT_1D" in data_type:
            examples.append(
                {
                    "scenario": "Time evolution access",
                    "code": f"time_trace = ids.{path}[:]",
                    "notes": "Access time evolution of quantity",
                }
            )
        elif coord1 == "rho_tor_norm":
            examples.append(
                {
                    "scenario": "Profile access",
                    "code": f"profile = ids.{path}[time_idx, :]",
                    "notes": "Access radial profile at specific time",
                }
            )

        return examples
