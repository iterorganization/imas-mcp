"""Identifier schema extractor for processing doc_identifier references."""

import logging
import xml.etree.ElementTree as ET
from typing import Any

from imas_mcp.core.data_model import IdentifierOption, IdentifierSchema
from imas_mcp.core.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class IdentifierExtractor(BaseExtractor):
    """Extract identifier schema information from doc_identifier references."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
        """Extract identifier schema information if element has doc_identifier."""
        result = {}

        doc_identifier = elem.get("doc_identifier")
        if doc_identifier:
            schema = self._parse_identifier_schema(doc_identifier)
            if schema:
                result["identifier_schema"] = schema.model_dump()

        return result

    def _parse_identifier_schema(self, doc_identifier: str) -> IdentifierSchema | None:
        """Parse an identifier schema from doc_identifier path.

        Args:
            doc_identifier: The schema path (e.g., 'equilibrium/equilibrium_profiles_2d_identifier.xml')

        Returns:
            IdentifierSchema object or None if not accessible
        """
        try:
            # Get the schema using the data dictionary accessor
            if hasattr(self.context.dd_accessor, "get_schema"):
                schema_tree = self.context.dd_accessor.get_schema(doc_identifier)
                if schema_tree is not None:
                    return self._extract_schema_data(schema_tree, doc_identifier)
                else:
                    logger.debug(f"Schema not accessible: {doc_identifier}")
                    return None
            else:
                logger.debug(f"get_schema method not available for: {doc_identifier}")
                return None

        except Exception as e:
            logger.debug(f"Error parsing identifier schema {doc_identifier}: {e}")
            return None

    def _extract_schema_data(
        self, schema_tree: ET.ElementTree, schema_path: str
    ) -> IdentifierSchema:
        """Extract data from a schema XML tree.

        Args:
            schema_tree: The parsed XML tree of the schema
            schema_path: The original schema path

        Returns:
            IdentifierSchema with extracted data
        """
        root = schema_tree.getroot()

        # Extract main documentation
        documentation = root.get("documentation") or root.text or ""
        if documentation:
            documentation = documentation.strip()

        # Extract enumeration options
        options = self._extract_identifier_options(root)

        # Extract additional metadata
        metadata = {}
        for attr_name, attr_value in root.attrib.items():
            if attr_name not in ["documentation"]:
                metadata[attr_name] = attr_value

        return IdentifierSchema(
            schema_path=schema_path,
            documentation=documentation,
            options=options,
            metadata=metadata,
        )

    def _extract_identifier_options(self, root: ET.Element) -> list[IdentifierOption]:
        """Extract identifier options from the schema root.

        This method looks for the IMAS identifier pattern with <int> elements
        containing name/description attributes and integer values as text.

        Args:
            root: The root element of the schema XML

        Returns:
            List of IdentifierOption objects
        """
        options = []

        # Look for IMAS standard pattern: <int name="..." description="...">value</int>
        for elem in root.iter("int"):
            option = self._extract_imas_int_option(elem)
            if option:
                options.append(option)

        # If no standard pattern found, look for other enumeration patterns
        if not options:
            options = self._extract_alternative_patterns(root)

        return sorted(options, key=lambda x: x.index)

    def _extract_imas_int_option(self, elem: ET.Element) -> IdentifierOption | None:
        """Extract identifier option from IMAS <int> element.

        Args:
            elem: XML <int> element with name/description attributes

        Returns:
            IdentifierOption or None if extraction fails
        """
        try:
            name = elem.get("name", "")
            description = elem.get("description", "")

            # Get index from element text
            index_text = elem.text or "0"
            try:
                index = int(index_text.strip())
            except ValueError:
                logger.debug(
                    f"Could not parse index '{index_text}' for option '{name}'"
                )
                index = 0

            if name:  # Only create option if we have a name
                return IdentifierOption(
                    name=name.strip(), index=index, description=description.strip()
                )

        except Exception as e:
            logger.debug(f"Error extracting IMAS int option: {e}")

        return None

    def _is_identifier_option(self, elem: ET.Element) -> bool:
        """Check if an element represents an identifier option.

        Args:
            elem: XML element to check

        Returns:
            True if element appears to be an identifier option
        """
        # Look for elements that have name, index, and description children
        children = list(elem)
        child_names = {child.get("name") for child in children}

        # Standard IMAS identifier pattern
        required_fields = {"name", "index", "description"}
        return required_fields.issubset(child_names)

    def _extract_single_option(self, elem: ET.Element) -> IdentifierOption | None:
        """Extract a single identifier option from an element.

        Args:
            elem: XML element containing identifier option

        Returns:
            IdentifierOption or None if extraction fails
        """
        try:
            name = ""
            index = 0
            description = ""

            # Extract name, index, and description from children
            for child in elem:
                child_name = child.get("name")
                if child_name == "name":
                    name = child.get("value") or child.text or ""
                elif child_name == "index":
                    index_text = child.get("value") or child.text or "0"
                    try:
                        index = int(index_text)
                    except ValueError:
                        index = 0
                elif child_name == "description":
                    description = child.get("value") or child.text or ""

            if name:  # Only create option if we have a name
                return IdentifierOption(
                    name=name.strip(), index=index, description=description.strip()
                )

        except Exception as e:
            logger.debug(f"Error extracting identifier option: {e}")

        return None

    def _extract_alternative_patterns(self, root: ET.Element) -> list[IdentifierOption]:
        """Extract identifier options using alternative patterns.

        This method handles cases where the schema doesn't follow the standard
        <int> pattern but still contains enumeration information.

        Args:
            root: The root element of the schema XML

        Returns:
            List of IdentifierOption objects
        """
        options = []

        # Pattern 1: Look for standard identifier structures (name/index/description children)
        for elem in root.iter():
            if self._is_identifier_option(elem):
                option = self._extract_single_option(elem)
                if option:
                    options.append(option)

        # Pattern 2: Direct attributes or text content with enumeration values
        if not options:
            for elem in root.iter():
                name = elem.get("name")
                if name and elem.get("value"):
                    try:
                        index = int(elem.get("index", "0"))
                        description = elem.get("description") or elem.text or ""
                        options.append(
                            IdentifierOption(
                                name=name, index=index, description=description.strip()
                            )
                        )
                    except ValueError:
                        continue

        # Pattern 3: Look for enumeration-like structures in documentation
        if not options:
            doc = root.get("documentation") or ""
            if doc:
                options = self._parse_documentation_enums(doc)

        return options

    def _parse_documentation_enums(self, documentation: str) -> list[IdentifierOption]:
        """Parse enumeration values from documentation text.

        This method looks for common patterns in documentation that indicate
        enumeration values, such as "ID=1: value1; ID=2: value2" etc.

        Args:
            documentation: Documentation text to parse

        Returns:
            List of IdentifierOption objects
        """
        options = []

        # Common patterns to look for
        import re

        # Pattern: "ID=1: description; ID=2: description"
        pattern1 = r"ID\s*=\s*(\d+)\s*:\s*([^;]+)"
        matches = re.findall(pattern1, documentation, re.IGNORECASE)

        for match in matches:
            try:
                index = int(match[0])
                description = match[1].strip()
                # Use first word of description as name
                name = description.split()[0] if description else f"option_{index}"
                options.append(
                    IdentifierOption(name=name, index=index, description=description)
                )
            except (ValueError, IndexError):
                continue

        # Pattern: "1: description, 2: description"
        if not options:
            pattern2 = r"(\d+)\s*:\s*([^,;]+)"
            matches = re.findall(pattern2, documentation)

            for match in matches:
                try:
                    index = int(match[0])
                    description = match[1].strip()
                    name = description.split()[0] if description else f"option_{index}"
                    options.append(
                        IdentifierOption(
                            name=name, index=index, description=description
                        )
                    )
                except (ValueError, IndexError):
                    continue

        return options
