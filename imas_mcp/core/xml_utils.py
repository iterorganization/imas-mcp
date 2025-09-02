"""Shared XML processing utilities for IMAS data dictionary.

This module provides common utilities for processing IMAS XML data dictionary
files, including hierarchical documentation building and tree traversal utilities.
"""

import xml.etree.ElementTree as ET


class DocumentationBuilder:
    """Utilities for building hierarchical documentation from XML elements."""

    @staticmethod
    def _normalize_sentence_punctuation(text: str) -> str:
        """Add missing punctuation to ensure proper sentence structure.

        Args:
            text: Raw documentation text

        Returns:
            Text with proper sentence-ending punctuation
        """
        text = text.strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    @staticmethod
    def build_hierarchical_documentation(documentation_parts: dict[str, str]) -> str:
        """Build semantic documentation optimized for sentence transformers.

        Creates natural language documentation by combining the primary field
        description with hierarchical context from parent elements. Optimized
        for semantic embedding generation rather than structured parsing.

        Args:
            documentation_parts: Dictionary where keys are hierarchical paths
                and values are the documentation strings for each node.

        Returns:
            Natural language documentation combining primary description with
            hierarchical context for optimal semantic understanding.
        """
        if not documentation_parts:
            return ""

        # Find the deepest (leaf) path - this is the primary element
        paths_by_depth = sorted(documentation_parts.keys(), key=lambda x: x.count("/"))
        if not paths_by_depth:
            return ""

        deepest_path = paths_by_depth[-1]
        leaf_doc = documentation_parts.get(deepest_path, "")

        # Build natural language description
        doc_parts = []

        # Primary description first (most important for embeddings)
        if leaf_doc:
            doc_parts.append(
                DocumentationBuilder._normalize_sentence_punctuation(leaf_doc)
            )

        # Add hierarchical context as natural language
        remaining_paths = paths_by_depth[:-1]  # Exclude the leaf path
        if remaining_paths:
            context_descriptions = []

            for path_key in remaining_paths:
                parent_doc = documentation_parts.get(path_key)
                if parent_doc:
                    normalized_doc = (
                        DocumentationBuilder._normalize_sentence_punctuation(parent_doc)
                    )

                    # Determine context level naturally
                    if "/" not in path_key:
                        # Root IDS level
                        context_descriptions.append(
                            f"Within {path_key} IDS: {normalized_doc}"
                        )
                    else:
                        # Container level
                        container_name = path_key.split("/")[-1]
                        context_descriptions.append(
                            f"Within {container_name} container: {normalized_doc}"
                        )

            # Add context descriptions
            doc_parts.extend(context_descriptions)

        return " ".join(doc_parts)

    @staticmethod
    def collect_documentation_hierarchy(
        elem: ET.Element,
        ids_elem: ET.Element,
        ids_name: str,
        parent_map: dict[ET.Element, ET.Element],
    ) -> dict[str, str]:
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
    def build_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
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
        parent_map: dict[ET.Element, ET.Element],
    ) -> str | None:
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
