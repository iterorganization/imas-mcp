"""Relationship extractor for finding connections between IMAS data paths."""

import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from imas_mcp.core.extractors.base import BaseExtractor


class RelationshipExtractor(BaseExtractor):
    """Extract relationships between data paths and across IDS."""

    def extract(self, elem: ET.Element) -> dict[str, Any]:
        """Extract relationship information for this element."""
        relationships = {}

        # Extract relationships from documentation
        doc_relationships = self._extract_relationships_from_documentation(elem)
        if doc_relationships:
            relationships.update(doc_relationships)

        # Extract cross-IDS references
        cross_ids_refs = self._extract_cross_ids_references(elem)
        if cross_ids_refs:
            relationships["cross_ids"] = cross_ids_refs

        # Extract physics relationships
        physics_related = self._extract_physics_relationships(elem)
        if physics_related:
            relationships["physics_related"] = physics_related

        # Extract coordinate relationships
        coord_relationships = self._extract_coordinate_relationships(elem)
        if coord_relationships:
            relationships["coordinate_related"] = coord_relationships

        # Extract unit relationships (same units = related)
        unit_relationships = self._extract_unit_relationships(elem)
        if unit_relationships:
            relationships["unit_related"] = unit_relationships

        return {"relationships": relationships} if relationships else {}

    def _extract_relationships_from_documentation(
        self, elem: ET.Element
    ) -> dict[str, list[str]]:
        """Extract relationships mentioned in documentation text."""
        documentation = elem.get("documentation", "")
        if not documentation:
            return {}

        relationships = {}

        # Look for explicit IDS references in documentation
        ids_pattern = r"\b([a-z_]+)(?:\s+IDS|_ids)\b"
        ids_matches = re.findall(ids_pattern, documentation.lower())
        if ids_matches:
            # Filter out self-references and common words
            filtered_matches = [
                match
                for match in ids_matches
                if match != self.context.ids_name.lower()
                and match not in ["this", "the", "an", "a", "and", "or", "for"]
                and len(match) > 2
            ]
            if filtered_matches:
                relationships["documentation_refs"] = list(set(filtered_matches))

        # Look for path references (words ending in specific patterns)
        path_pattern = r"\b([a-z_]+(?:_[a-z_]+)*)/([a-z_]+(?:[/_][a-z_]+)*)\b"
        path_matches = re.findall(path_pattern, documentation.lower())
        if path_matches:
            full_paths = [f"{ids}/{path}" for ids, path in path_matches]
            relationships["path_refs"] = full_paths

        # Look for quantity references (common physics quantities)
        quantity_patterns = [
            r"\bdensity\b",
            r"\btemperature\b",
            r"\bpressure\b",
            r"\bmagnetic\s+field\b",
            r"\bcurrent\b",
            r"\bflux\b",
            r"\bprofile\b",
            r"\bvelocity\b",
            r"\benergy\b",
        ]
        found_quantities = []
        for pattern in quantity_patterns:
            if re.search(pattern, documentation.lower()):
                found_quantities.append(pattern.strip("\\b"))
        if found_quantities:
            relationships["quantity_refs"] = found_quantities

        return relationships

    def _extract_cross_ids_references(self, elem: ET.Element) -> list[str]:
        """Extract references to other IDS from various XML attributes."""
        cross_refs = []

        # Check for explicit IDS references in XML attributes
        for _attr_name, attr_value in elem.attrib.items():
            if attr_value and isinstance(attr_value, str):
                # Look for IDS names in attribute values
                ids_pattern = r"\b([a-z_]+)(?:\s*/|\s+|$)"
                matches = re.findall(ids_pattern, attr_value.lower())
                for match in matches:
                    if (
                        match != self.context.ids_name.lower()
                        and len(match) > 3
                        and "_" in match  # Likely IDS names have underscores
                        and match not in cross_refs
                    ):
                        cross_refs.append(match)

        # Check for timebasepath references (points to other IDS)
        timebase = elem.get("timebasepath", "")
        if timebase and "/" in timebase:
            timebase_ids = timebase.split("/")[0]
            if timebase_ids != self.context.ids_name and timebase_ids not in cross_refs:
                cross_refs.append(timebase_ids)

        return cross_refs

    def _extract_physics_relationships(self, elem: ET.Element) -> list[str]:
        """Extract physics-based relationships."""
        physics_related = []

        elem_name = elem.get("name", "").lower()
        documentation = elem.get("documentation", "").lower()

        # Define physics relationship patterns
        physics_patterns = {
            "kinetic_profiles": [
                "density",
                "temperature",
                "pressure",
                "n_e",
                "t_e",
                "t_i",
            ],
            "magnetic_quantities": [
                "b_field",
                "b_tor",
                "b_pol",
                "psi",
                "flux",
                "current",
            ],
            "transport": ["diffusivity", "conductivity", "flux", "gradient"],
            "geometry": ["r", "z", "rho", "phi", "radius", "outline", "boundary"],
            "temporal": ["time", "frequency", "period", "evolution"],
        }

        # Find which physics category this element belongs to
        element_categories = []
        for category, patterns in physics_patterns.items():
            if any(
                pattern in elem_name or pattern in documentation for pattern in patterns
            ):
                element_categories.append(category)

        # Elements in the same physics category are related
        if element_categories:
            physics_related.extend(element_categories)

        return physics_related

    def _extract_coordinate_relationships(self, elem: ET.Element) -> list[str]:
        """Extract relationships based on coordinate systems."""
        coord_related = []

        # Check coordinate attributes
        coord1 = elem.get("coordinate1", "")
        coord2 = elem.get("coordinate2", "")

        if coord1:
            coord_related.append(f"coord1:{coord1}")
        if coord2:
            coord_related.append(f"coord2:{coord2}")

        # Parse coordinate descriptions
        coordinates = elem.get("coordinates", "")
        if coordinates and isinstance(coordinates, str):
            # Extract coordinate patterns
            coord_patterns = re.findall(r"([a-z_]+(?:\([^)]+\))?)", coordinates.lower())
            for pattern in coord_patterns:
                if pattern not in coord_related:
                    coord_related.append(f"coord:{pattern}")

        return coord_related

    def _extract_unit_relationships(self, elem: ET.Element) -> list[str]:
        """Extract relationships based on units."""
        units = elem.get("units", "")
        if not units or units in ["", "1", "-"]:
            return []

        # Elements with the same units are related
        return [f"units:{units}"]

    def extract_all_relationships(
        self, ids_data: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Extract cross-IDS relationships from all paths with performance optimization."""
        all_relationships = {
            "cross_ids_map": {},
            "physics_concept_map": {},
            "unit_families": {},
            "coordinate_families": {},
        }

        # Collect paths efficiently
        all_paths = []
        paths_by_ids = {}

        for ids_name, ids_info in ids_data.items():
            paths = list(ids_info.get("paths", {}).keys())
            paths_by_ids[ids_name] = paths
            all_paths.extend(paths)

        print(f"Processing {len(all_paths)} total paths from {len(ids_data)} IDS")

        # Group by units with better logic
        unit_groups = {}

        for _ids_name, ids_info in ids_data.items():
            paths_dict = ids_info.get("paths", {})

            for path, path_data in paths_dict.items():
                # Group by units
                units = path_data.get("units", "")
                if units and units not in ["", "1", "-"]:
                    if units not in unit_groups:
                        unit_groups[units] = []
                    unit_groups[units].append(path)

        # Build unit families
        print(f"Unit groups found: {len(unit_groups)}")

        all_relationships["unit_families"] = {}
        for unit, paths in unit_groups.items():
            if len(paths) >= 2:  # At least 2 paths share this unit
                all_relationships["unit_families"][unit] = {
                    "base_unit": unit,
                    "paths_using": paths,  # Include all paths
                    "conversion_factors": {},
                }

        # Build cross-IDS relationships using embedding-based semantic analysis
        cross_ids_relationships = {}

        # Initialize sentence transformer for semantic similarity
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Use same model as search

        # Collect path descriptions for embedding
        path_descriptions = {}
        path_metadata = {}

        for ids_name, ids_info in ids_data.items():
            paths_dict = ids_info.get("paths", {})

            for path, path_data in paths_dict.items():
                if "/" not in path or len(path.split("/")) < 3:
                    continue

                # Skip generic structural/metadata fields that have standardized documentation
                # These don't represent actual semantic content for relationships
                path_parts = path.split("/")
                last_component = path_parts[-1]

                # Skip common metadata fields that have generic documentation
                if last_component in ["name", "index", "description", "identifier"]:
                    continue

                # Skip other generic structural fields
                if any(
                    generic in last_component
                    for generic in [
                        "_index",
                        "_name",
                        "_description",
                        "_identifier",
                        "grid_index",
                        "species_index",
                        "element_index",
                    ]
                ):
                    continue

                # Build semantic description using complete documentation, excluding units/metadata
                documentation = path_data.get("documentation", "")
                physics_context = path_data.get("physics_context", "")

                # Use complete documentation string, excluding only cross-reference sections
                complete_doc = ""
                if documentation:
                    # Remove "Within X IDS:" sections which are cross-references, not semantic content
                    within_pattern = re.compile(
                        r"\.\s*Within\s+\w+\s+(IDS|container)\s*:.*?(?=\.|$)",
                        re.IGNORECASE | re.DOTALL,
                    )

                    # Remove cross-reference sections but keep all other documentation
                    cleaned_doc = within_pattern.sub("", documentation).strip()
                    complete_doc = re.sub(r"\s+", " ", cleaned_doc).strip()

                # Extract meaningful path components (excluding generic IDS prefix and metadata fields)
                path_parts = path.split("/")
                if len(path_parts) >= 2:
                    # Skip the IDS name (first part) and focus on meaningful path components
                    meaningful_parts = path_parts[1:]  # Skip IDS name

                    # Filter out generic structural terms but keep domain-specific terms
                    filtered_parts = []
                    for part in meaningful_parts:
                        # Skip very generic terms but keep physics/domain terms
                        if part not in [
                            "time_slice",
                            "profiles_1d",
                            "profiles_2d",
                            "global_quantities",
                        ]:
                            filtered_parts.append(part)

                    # Create semantic path context
                    path_context = " ".join(filtered_parts).replace("_", " ")
                else:
                    path_context = ""

                # Create semantic description combining documentation and meaningful path context
                description_parts = []
                if complete_doc:
                    description_parts.append(complete_doc)
                if path_context:
                    description_parts.append(f"Context: {path_context}")

                semantic_description = (
                    ". ".join(description_parts)
                    if description_parts
                    else "Generic data field"
                )

                path_descriptions[path] = semantic_description
                path_metadata[path] = {
                    "ids": ids_name,
                    "documentation": documentation,
                    "physics_context": physics_context,
                    "related_paths": path_data.get("related_paths", []),
                }

        print(f"Generating embeddings for {len(path_descriptions)} paths...")

        # Generate embeddings for all path descriptions
        descriptions_list = list(path_descriptions.values())
        paths_list = list(path_descriptions.keys())

        embeddings = model.encode(
            descriptions_list,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
            show_progress_bar=False,
        )

        print("Computing semantic similarities...")

        # Compute semantic similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Extract cross-IDS relationships based on semantic similarity
        semantic_threshold = 0.7  # Adjusted threshold for complete documentation
        max_relationships_per_ids = 15  # Reduced to focus on strongest relationships

        # Store detailed relationship information including source paths and scores
        detailed_relationships = {}

        for i, source_path in enumerate(paths_list):
            source_ids = path_metadata[source_path]["ids"]

            # Find semantically similar paths from other IDS
            similarities = similarity_matrix[i]

            # Get indices sorted by similarity (excluding self)
            similar_indices = np.argsort(similarities)[::-1]

            for j in similar_indices:
                if i == j:  # Skip self-similarity
                    continue

                target_path = paths_list[j]
                target_ids = path_metadata[target_path]["ids"]
                similarity_score = similarities[j]

                # Only include cross-IDS relationships above threshold
                if target_ids != source_ids and similarity_score >= semantic_threshold:
                    # Skip if semantic descriptions are too short or generic
                    source_desc = path_descriptions[source_path]
                    target_desc = path_descriptions[target_path]

                    if (
                        len(source_desc) < 20
                        or len(target_desc) < 20
                        or source_desc == "Generic data field"
                        or target_desc == "Generic data field"
                    ):
                        continue

                    # Initialize the source IDS entry if needed
                    if source_ids not in detailed_relationships:
                        detailed_relationships[source_ids] = []

                    # Store detailed relationship information
                    detailed_relationships[source_ids].append(
                        {
                            "target_path": target_path,
                            "source_path": source_path,
                            "similarity_score": float(similarity_score),
                            "relationship_type": "semantic",
                            "source_description": source_desc[:100] + "..."
                            if len(source_desc) > 100
                            else source_desc,
                            "target_description": target_desc[:100] + "..."
                            if len(target_desc) > 100
                            else target_desc,
                        }
                    )

                    # Limit to prevent explosion
                    if (
                        len(detailed_relationships[source_ids])
                        >= max_relationships_per_ids
                    ):
                        break

            # Also include explicitly related paths from documentation
            explicit_related = path_metadata[source_path].get("related_paths", [])
            for related_path in explicit_related:
                if (
                    related_path in path_metadata
                    and path_metadata[related_path]["ids"] != source_ids
                ):
                    if source_ids not in detailed_relationships:
                        detailed_relationships[source_ids] = []

                    # Add explicit relationships
                    detailed_relationships[source_ids].append(
                        {
                            "target_path": related_path,
                            "source_path": source_path,
                            "similarity_score": 1.0,  # Explicit relationships get max score
                            "relationship_type": "explicit",
                        }
                    )

        # Convert detailed relationships back to simple format for backward compatibility
        # but also store the detailed version
        cross_ids_relationships = {}
        for ids_name, relationships in detailed_relationships.items():
            # Remove duplicates by target path, keeping highest scoring
            unique_relationships = {}
            for rel in relationships:
                target = rel["target_path"]
                if (
                    target not in unique_relationships
                    or rel["similarity_score"]
                    > unique_relationships[target]["similarity_score"]
                ):
                    unique_relationships[target] = rel

            # Limit and sort by similarity score
            sorted_relationships = sorted(
                unique_relationships.values(),
                key=lambda x: x["similarity_score"],
                reverse=True,
            )
            cross_ids_relationships[ids_name] = [
                rel["target_path"]
                for rel in sorted_relationships[:max_relationships_per_ids]
            ]

        # Store both formats in the results
        all_relationships["cross_ids_map"] = cross_ids_relationships
        all_relationships["detailed_cross_ids_map"] = detailed_relationships

        total_relationships = sum(
            len(paths) for paths in cross_ids_relationships.values()
        )
        print(
            f"Generated cross-IDS relationships: {len(cross_ids_relationships)} IDS pairs, {total_relationships} total connections"
        )
        print(f"Generated {len(unit_groups)} unit families")

        return all_relationships
