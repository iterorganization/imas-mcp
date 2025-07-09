"""Shared XML processing utilities for IMAS data dictionary.

This module provides common utilities for processing IMAS XML data dictionary
files, including hierarchical documentation building and tree traversal utilities.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional


class DocumentationBuilder:
    """Utilities for building hierarchical documentation from XML elements."""

    @staticmethod
    def build_hierarchical_documentation(documentation_parts: Dict[str, str]) -> str:
        """Build LLM-optimized hierarchical documentation from path-based parts.

        Creates documentation optimized for Large Language Model comprehension by:
        1. Leading with structured metadata for quick parsing
        2. Providing clear semantic relationships
        3. Using explicit labels rather than implicit markdown structure
        4. Placing most relevant context first
        5. Including relationship and type information

        Args:
            documentation_parts: Dictionary where keys are hierarchical paths
                and values are the documentation strings for each node.

        Returns:
            A structured string optimized for LLM understanding with explicit
            semantic relationships and clear information hierarchy.
        """
        if not documentation_parts:
            return ""

        # Find the deepest (leaf) path
        paths_by_depth = sorted(documentation_parts.keys(), key=lambda x: x.count("/"))
        if not paths_by_depth:
            return ""

        deepest_path = paths_by_depth[-1]
        leaf_doc = documentation_parts.get(deepest_path, "")

        # Build LLM-optimized structure
        doc_sections = []

        # 1. STRUCTURED METADATA BLOCK (Easy for LLMs to parse)
        path_parts = deepest_path.split("/")
        ids_name = path_parts[0] if path_parts else ""

        metadata_block = [
            "=== IMAS DATA DICTIONARY ENTRY ===",
            f"FULL_PATH: {deepest_path}",
            f"IDS: {ids_name}",
            f"HIERARCHY_DEPTH: {len(path_parts) - 1}",
            f"TOTAL_CONTEXT_LEVELS: {len(documentation_parts)}",
        ]

        if len(path_parts) > 1:
            metadata_block.append(f"PARENT_CONTAINER: {'/'.join(path_parts[:-1])}")
            metadata_block.append(f"FIELD_NAME: {path_parts[-1]}")
            metadata_block.append("IS_LEAF_FIELD: True")
        else:
            metadata_block.append("IS_ROOT_IDS: True")

        # Add path components for easy parsing
        if len(path_parts) > 1:
            metadata_block.append(f"PATH_COMPONENTS: {' -> '.join(path_parts)}")

        doc_sections.append("\n".join(metadata_block))

        # 2. PRIMARY DESCRIPTION (Most important info first)
        if leaf_doc:
            doc_sections.append(f"PRIMARY_DESCRIPTION: {leaf_doc}")

        # 3. CONTEXTUAL HIERARCHY (Explicit semantic relationships)
        remaining_paths = paths_by_depth[:-1]
        if remaining_paths:
            hierarchy_lines = ["=== CONTEXTUAL HIERARCHY ==="]

            for path_key in remaining_paths:
                doc = documentation_parts[path_key]
                if doc:
                    path_parts_ctx = path_key.split("/")
                    level_name = "ROOT_IDS" if len(path_parts_ctx) == 1 else "CONTAINER"
                    hierarchy_lines.append(f"{level_name}[{path_key}]: {doc}")

            doc_sections.append("\n".join(hierarchy_lines))

        return "\n\n".join(doc_sections)

    @staticmethod
    def collect_documentation_hierarchy(
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Dict[str, str]:
        """Collect documentation from element hierarchy up to IDS root.

        Walks up the XML tree from the given element to collect documentation
        from each parent element, building a mapping of paths to documentation.

        Args:
            elem: The XML element to start from (leaf element)
            ids_elem: The IDS root element to stop at
            ids_name: Name of the IDS
            parent_map: Mapping of child elements to their parents for efficient traversal

        Returns:
            Dictionary mapping hierarchical paths to their documentation strings.
            Includes documentation from the element itself and all parent elements
            up to the IDS root. All nodes in the path are included even if they
            don't have documentation.
        """
        documentation_parts = {}

        # First, build the complete path from root to element
        path_elements = []
        current = elem

        # Collect all elements with names from leaf to root
        while current is not None and current != ids_elem:
            name = current.get("name")
            if name:
                path_elements.insert(
                    0, current
                )  # Insert at beginning to build root-to-leaf
            current = parent_map.get(current)

        # Now build documentation for each level that has documentation
        for i, element in enumerate(path_elements):
            # Build the path up to this element
            path_parts = [e.get("name") for e in path_elements[: i + 1]]
            full_path = "/".join([ids_name] + path_parts)

            # Add documentation if this element has it
            doc = element.get("documentation")
            if doc:
                documentation_parts[full_path] = doc

        # Add IDS documentation
        ids_doc = ids_elem.get("documentation")
        if ids_doc:
            documentation_parts[ids_name] = ids_doc

        return documentation_parts


class XmlTreeUtils:
    """Common XML tree traversal utilities."""

    @staticmethod
    def build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
        """Build parent map for efficient tree traversal.

        Creates a mapping from child elements to their parent elements,
        enabling efficient upward traversal of the XML tree.

        Args:
            root: Root element of the XML tree

        Returns:
            Dictionary mapping child elements to their parent elements
        """
        return {child: parent for parent in root.iter() for child in parent}

    @staticmethod
    def build_element_path(
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Optional[str]:
        """Build full IMAS path for XML element.

        Constructs the full hierarchical path for an XML element by walking
        up the tree to the IDS root.

        Args:
            elem: The XML element to build path for
            ids_elem: The IDS root element
            ids_name: Name of the IDS
            parent_map: Mapping of child elements to their parents

        Returns:
            Full path string in format "ids_name/parent/child" or None if
            no valid path can be constructed.
        """
        path_parts = []
        current = elem

        # Walk up the tree to build path using parent map
        while current is not None and current != ids_elem:
            name = current.get("name")
            if name:
                path_parts.insert(0, name)
            current = parent_map.get(current)

        if not path_parts:
            return None

        return f"{ids_name}/{'/'.join(path_parts)}"
