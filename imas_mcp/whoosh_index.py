"""
IMAS Data Dictionary indexing functionality using Whoosh.

This module provides tools for creating, managing, and searching a Whoosh index
of IMAS Data Dictionary entries. It allows for efficient querying of documentation
and metadata from the IMAS Data Dictionary.
"""

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Type, Union

import pydantic
import whoosh
import whoosh.analysis
import whoosh.fields
import whoosh.index
import whoosh.qparser
import whoosh.query
import whoosh.searching
import whoosh.writing

from imas_mcp.search_result import DataDictionaryEntry, SearchResult

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class WhooshIndex:
    """Index class for creating and managing a Whoosh index for IMAS DD entries."""

    dirname: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "resources"
        / "index_data"
    )
    indexname: Optional[str] = field(default=None)
    schema: Optional[whoosh.fields.Schema] = field(default=None, repr=False)
    document_model: Type[DataDictionaryEntry] = field(default=DataDictionaryEntry)
    skip_unit_parsing: bool = True

    _writer: whoosh.writing.IndexWriter | None = field(
        init=False, default=None, repr=False
    )
    _unit_error: dict[str, list[str]] = field(
        init=False, repr=False, default_factory=dict
    )

    @property
    def _validation_context(self) -> Dict[str, Any]:
        """Build validation context for Pydantic models."""
        return {
            "skip_unit_parsing": self.skip_unit_parsing,
        }

    def __len__(self) -> int:
        """Return the number of documents in the Whoosh index."""
        with self._index.searcher() as searcher:
            doc_count = searcher.doc_count()
            assert isinstance(doc_count, int), "Document count should be an integer"
            return doc_count

    @property
    def resolved_schema(self) -> whoosh.fields.Schema:
        """Return resolved Whoosh Schema."""
        if self.schema is None:
            self.schema = self._get_schema()
        return self.schema

    def _get_schema(self) -> whoosh.fields.Schema:
        """Return the Whoosh schema with extended JSON fields."""
        if self.schema is not None:
            return self.schema
        return whoosh.fields.Schema(
            path=whoosh.fields.ID(
                stored=True, unique=True
            ),  # The full IDS path as unique ID
            documentation=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Documentation content
            units=whoosh.fields.KEYWORD(stored=True),  # Units of the field
            ids_name=whoosh.fields.ID(stored=True),  # The root IDS
            # Extended fields from JSON data
            coordinates=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Coordinate information
            lifecycle=whoosh.fields.KEYWORD(stored=True),  # Lifecycle status
            data_type=whoosh.fields.KEYWORD(stored=True),  # Data type
            physics_context=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Physics context
            related_paths=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Related paths as searchable text
            usage_examples=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Usage examples as searchable text
            validation_rules=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Validation rules as text
            relationships=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Relationships as searchable text
            introduced_after=whoosh.fields.KEYWORD(stored=True),  # Version introduced
            coordinate1=whoosh.fields.KEYWORD(stored=True),  # Primary coordinate
            coordinate2=whoosh.fields.KEYWORD(stored=True),  # Secondary coordinate
            timebase=whoosh.fields.KEYWORD(stored=True),  # Timebase information
            type=whoosh.fields.KEYWORD(stored=True),  # Type field
        )

    @cached_property
    def _index(self) -> whoosh.index.FileIndex:
        """Return cached Whoosh index."""
        return self._get_index()

    def _get_index(self) -> whoosh.index.FileIndex:
        """Return the Whoosh index"""
        if not self.dirname.exists():
            # Create the directory if it doesn't exist
            self.dirname.mkdir(parents=True, exist_ok=True)
        if whoosh.index.exists_in(self.dirname, indexname=self.indexname):
            # Update the schema and return the existing index
            index = whoosh.index.open_dir(self.dirname, self.indexname)
            self.schema = index.schema
            return index
        # Create whoosh index
        return whoosh.index.create_in(
            self.dirname, self.resolved_schema, self.indexname
        )

    def _validate_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document against the Pydantic model.

        Args:
            document: Dictionary containing fields to be indexed.

        Returns:
            Validated dictionary with properly formatted fields.

        Raises:
            pydantic.ValidationError: If document does not match the model schema.
        """
        validated_document = self.document_model.model_validate(
            document, context=self._validation_context
        ).model_dump()
        assert isinstance(validated_document, dict)
        return validated_document

    @contextmanager
    def writer(self) -> Generator[whoosh.writing.IndexWriter, None, None]:
        """Yield a Whoosh index writer for batch add operations."""
        self._writer = None
        self._unit_error = {}
        try:
            self._writer = self._index.writer(procs=4, limitmb=256, multisegment=True)
            yield self._writer
            for unit, paths in self._unit_error.items():
                logger.error(
                    f"Unit error: {len(paths)} paths define unit as '{unit}' {paths}"
                )
        finally:
            if self._writer is not None:
                self._writer.commit()
                self._writer = None

    def __add__(self, document: Dict[str, Any]) -> "WhooshIndex":
        """Add a document to the Whoosh index.

        Args:
            document: Dictionary containing fields to be indexed.
                      Keys must match the schema field names.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If _writer is None.
            pydantic.ValidationError: If document does not match the schema.

        Examples:
            >>> with index.writer():
            ...     # Add a single document
            ...     index = index + {"path": "/path/to/doc", "content": "Example content"}
            ...
            ...     # Chain multiple adds
            ...     index = index + doc1 + doc2 + doc3
        """
        if self._writer is None:
            raise ValueError(
                "Writer is not initialized. Use 'with index.writer():' context."
            )
        try:
            validated_doc = self._validate_document(document)
        except pydantic.ValidationError as error:
            error_text = str(error)
            if "units" in error_text:  # log UndefinedUnitError and continue
                # append unit error for logging
                units = document.get("units")
                if (
                    units and units not in self._unit_error
                ):  # Check if units is not None
                    self._unit_error[units] = []
                if units:  # Check if units is not None
                    self._unit_error[units].append(
                        str(document.get("path"))
                    )  # Ensure path is str
                validated_doc = document  # type: ignore # Allow assignment if units error
                pass
            else:
                logging.error(f"Pydantic validation error: {error}")
                raise

        self._writer.update_document(**validated_doc)  # upsert
        return self

    def __iadd__(self, document: Dict[str, Any]) -> "WhooshIndex":
        """Add a document to the Whoosh index in-place (+=).

        Args:
            document: Dictionary containing fields to be indexed. Keys must match the schema
                      field names.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If _writer is None.
            pydantic.ValidationError: If document does not match the schema.

        Examples:
            >>> with index.writer():
            ...     # Add a single document in-place
            ...     index += {"path": "/path/to/doc", "documentation": "Example content"}
            ...
            ...     # Multiple operations
            ...     for doc in documents:
            ...         index += doc
        """
        return self.__add__(document)

    def add_document(self, document: Dict[str, Any]) -> None:
        """Add a single document to the index using the writer context.

        Args:
            document: Dictionary containing fields to be indexed.
        """
        with self.writer():
            self += document

    def add_document_batch(self, documents: list[Dict[str, Any]]) -> None:
        """Add a batch of documents to the index using the writer context.

        Args:
            documents: A list of dictionaries, each containing fields to be indexed.
        """
        with self.writer():
            for doc in documents:
                self += doc

    @contextmanager
    def searcher(self) -> Generator[whoosh.searching.Searcher, None, None]:
        """Yield a Whoosh index searcher for querying the index."""
        with self._index.searcher() as searcher:
            yield searcher

    def search_by_keywords(
        self,
        query_str: str = None,
        keywords: list[str] = None,
        page_size: int = 10,
        page: int = 1,
        fuzzy: bool = False,
        search_fields: Optional[list[str]] = None,
        sort_by: Optional[Union[str, list[str]]] = None,
        sort_reverse: bool = False,
        exploratory_mode: bool = True,
    ) -> list[SearchResult]:
        """
        Search the index for paths matching the given keywords.

        Enhanced to support both string queries and list-based multi-keyword searches
        with semantic relevance ranking and intelligent filtering.

        Args:
            query_str: Natural language query string (use this OR keywords, not both)
            keywords: List of keywords for enhanced multi-keyword search
            page_size: Maximum number of results per page.
            page: Page number to retrieve (1-based).
            fuzzy: Enable fuzzy term matching (e.g., "temperture~" for "temperature").
            search_fields: List of fields to search in.
            sort_by: Field name or list of field names to sort by.
            sort_reverse: Whether to reverse the sort order.
            exploratory_mode: If True, uses OR logic between keywords for broader exploration.

        Returns:
            List of SearchResult objects.
        """
        # Handle both string and list inputs
        if keywords is not None and query_str is not None:
            raise ValueError("Provide either query_str or keywords, not both")

        if keywords is not None:
            # Enhanced multi-keyword search with semantic ranking
            return self._search_multi_keywords(
                keywords, page_size, page, fuzzy, search_fields
            )

        if query_str is None:
            return []

        # Original string-based search logic
        results = []

        if search_fields is None:
            search_fields = [
                "documentation",
                "coordinates",
                "physics_context",
                "related_paths",
                "usage_examples",
                "validation_rules",
                "relationships",
            ]

        # If fuzzy is True and query_str does not contain ~, append ~ to each keyword
        if fuzzy and "~" not in query_str:
            query_str = " ".join(keyword + "~" for keyword in query_str.split())

        # Enable fuzzy if query_str contains ~ character
        if "~" in query_str:
            fuzzy = True

        # Handle exploratory mode for multi-keyword queries
        if exploratory_mode and not any(
            op in query_str.upper() for op in ["AND", "OR", "NOT", ":"]
        ):
            # Convert space-separated keywords to OR query for exploration
            keywords_from_str = query_str.split()
            if len(keywords_from_str) > 1:
                query_str = " OR ".join(keywords_from_str)

        with self.searcher() as searcher:
            parser = whoosh.qparser.MultifieldParser(
                search_fields, self.resolved_schema
            )

            if fuzzy:
                parser.add_plugin(whoosh.qparser.FuzzyTermPlugin())

            parsed_query = parser.parse(query_str)

            # Use search_page for pagination. pagenum is 1-based.
            search_results = searcher.search_page(
                parsed_query,
                pagenum=page,
                pagelen=page_size,
                sortedby=sort_by,
                reverse=sort_reverse,
            )
            for hit in search_results:
                results.append(SearchResult.from_hit(hit))
        return results

    def _search_multi_keywords(
        self,
        keywords: list[str],
        page_size: int = 10,
        page: int = 1,
        fuzzy: bool = False,
        search_fields: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Internal method for enhanced multi-keyword search with semantic relevance.

        Prioritizes semantic relevance over exact multi-keyword matching.
        """
        if not keywords:
            return []

        if search_fields is None:
            search_fields = [
                "documentation",
                "coordinates",
                "physics_context",
                "related_paths",
                "usage_examples",
                "validation_rules",
                "relationships",
            ]

        # Strategy: Get top results for each keyword, then apply semantic filtering
        all_results = []
        keyword_weights = {}

        # Calculate keyword weights based on specificity (inverse frequency)
        for keyword in keywords:
            # Get total count for this keyword to estimate specificity
            temp_results = self.search_by_keywords(
                query_str=keyword,
                page_size=1,
                fuzzy=fuzzy,
                search_fields=search_fields,
                exploratory_mode=True,
            )
            # More specific keywords (fewer results) get higher weight
            keyword_weights[keyword] = (
                1.0 / (len(temp_results) + 1) if temp_results else 1.0
            )

        # Get the best results for each keyword
        for keyword in keywords:
            results = self.search_by_keywords(
                query_str=keyword,
                page_size=max(
                    5, page_size // len(keywords) + 2
                ),  # Distribute results per keyword
                page=1,
                fuzzy=fuzzy,
                search_fields=search_fields,
                exploratory_mode=True,
            )

            # Apply keyword-specific weight and semantic filtering
            weight = keyword_weights[keyword]
            for result in results:
                # Advanced semantic filtering for physics searches
                path_lower = result.path.lower()
                doc_lower = result.documentation.lower()

                # Define different types of terms
                structural_terms = [
                    "grid",
                    "coordinate",
                    "volume_element",
                    "mesh",
                    "node",
                    "index",
                    "geometry",
                ]
                measurement_terms = [
                    "profile",
                    "data",
                    "measurement",
                    "sensor",
                    "diagnostic",
                    "detector",
                ]

                # Check if this is a structural path
                is_structural = any(
                    struct_term in path_lower for struct_term in structural_terms
                )

                if is_structural:
                    # For structural paths, be very strict about relevance
                    # Only keep if the documentation explicitly discusses measurements/profiles of our keywords
                    has_measurement_context = any(
                        meas_term in doc_lower for meas_term in measurement_terms
                    )

                    # Or if the documentation specifically mentions temperature/density measurements (not just "plasma volume")
                    specific_physics_context = any(
                        kw in doc_lower
                        for kw in ["temperature", "density", "current"]
                        if kw in keywords
                    )

                    if not (has_measurement_context and specific_physics_context):
                        continue

                # Apply semantic scoring
                semantic_score = result.score * weight

                # Higher weights for matches in leaf nodes or immediate parent
                path_parts = result.path.split("/")

                # Check if keywords match in the leaf node (last part)
                leaf_matches = any(
                    kw.lower() in path_parts[-1].lower() for kw in keywords
                )
                if leaf_matches:
                    semantic_score *= 2.0  # Strong boost for leaf node matches

                # Check if keywords match in the second-to-last part (immediate parent)
                elif len(path_parts) > 1:
                    parent_matches = any(
                        kw.lower() in path_parts[-2].lower() for kw in keywords
                    )
                    if parent_matches:
                        semantic_score *= 1.5  # Good boost for parent node matches

                # Boost if the path contains physics-relevant terms
                physics_terms = [
                    "profile",
                    "temperature",
                    "density",
                    "current",
                    "field",
                    "electron",
                    "ion",
                ]
                if any(term in path_lower for term in physics_terms):
                    semantic_score *= 1.3

                # Boost if documentation is rich and relevant
                if len(result.documentation) > 50 and any(
                    kw in doc_lower for kw in keywords
                ):
                    semantic_score *= 1.2

                # Create enhanced result
                enhanced_result = SearchResult(
                    path=result.path,
                    score=semantic_score,
                    documentation=result.documentation,
                    units=result.units,
                    ids_name=result.ids_name,
                    highlights=result.highlights,
                    coordinates=result.coordinates,
                    lifecycle=result.lifecycle,
                    data_type=result.data_type,
                    physics_context=result.physics_context,
                    related_paths=result.related_paths,
                    usage_examples=result.usage_examples,
                    validation_rules=result.validation_rules,
                    relationships=result.relationships,
                    introduced_after=result.introduced_after,
                    coordinate1=result.coordinate1,
                    coordinate2=result.coordinate2,
                    timebase=result.timebase,
                    type=result.type,
                )
                all_results.append(enhanced_result)

        # Remove duplicates while preserving the best score for each path
        path_best_results = {}
        for result in all_results:
            if (
                result.path not in path_best_results
                or result.score > path_best_results[result.path].score
            ):
                path_best_results[result.path] = result

        # Sort by semantic score and apply pagination
        final_results = list(path_best_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return final_results[start_idx:end_idx]

    def search_by_exact_path(self, path_value: str) -> Optional[SearchResult]:
        """Return documentation and associated metadata via an exact IDS path lookup.

        Args:
            path_value: The exact path of the document to retrieve.

        Returns:
            A SearchResult object if found, otherwise None.
        """
        with self.searcher() as searcher:
            document_fields = searcher.document(path=path_value)
            if document_fields:
                return SearchResult.from_document(document_fields)
            return None

    def search_by_path_prefix(
        self,
        path_prefix: str,
        page_size: int = 10,
        page: int = 1,
        sort_by: Optional[Union[str, list[str]]] = None,
        sort_reverse: bool = False,
    ) -> list[SearchResult]:
        """Return all entries matching a given IDS path prefix.

        Args:
            path_prefix: The prefix of the path to search for (e.g., "core_profiles/profiles_1d").
            page_size: Maximum number of results per page.
            page: Page number to retrieve (1-based).
            sort_by: Field name or list of field names to sort_by.
            sort_reverse: Whether to reverse the sort order.

        Returns:
            A list of SearchResult objects.
        """
        results = []
        query = whoosh.query.Prefix("path", path_prefix)

        with self.searcher() as searcher:
            # Use search_page for pagination. pagenum is 1-based.
            search_results = searcher.search_page(
                query,
                pagenum=page,
                pagelen=page_size,
                sortedby=sort_by,
                reverse=sort_reverse,
            )
            for hit in search_results:
                results.append(SearchResult.from_hit(hit))
        return results

    def filter_search_results(
        self,
        search_results: list[SearchResult],
        filters: Dict[str, Any],
        regex: bool = True,
    ) -> list[SearchResult]:
        """
        Filter a list of SearchResult objects based on exact field values and/or
        pattern matching. When regex=True (default), filtering automatically uses
        regex when special characters are detected in filter values. When regex=False,
        only exact string matching is performed regardless of special characters.
        Regex searches are case-insensitive.

        Args:
            search_results: A list of SearchResult objects to filter.
            filters: A dictionary where keys are field names (str) present in
                     SearchResult (e.g., "ids", "units", "path", "documentation")
                     and values are the desired values for those fields.
            regex: When True (default), enables automatic regex pattern detection
                   and matching with case-insensitive searches. When False, disables
                   regex matching and uses only exact string comparison.

        Returns:
            A new list containing only the SearchResult objects that match
            all the specified filters.

        Examples:
            >>> # Assume 'index' is an instance of WhooshIndex
            >>> # First, get some initial results
            >>> initial_results = index.search_by_keywords("temperature")
            >>>
            >>> # Filter with automatic regex detection (default behavior, regex=True)
            >>> filtered_by_ids = index.filter_search_results(
            ...     initial_results, {"ids": "core_profiles"}
            ... )
            >>>
            >>> # Exact string matching only (regex=False)
            >>> exact_path = index.filter_search_results(
            ...     initial_results,
            ...     {"path": "core_profiles/profiles_1d"},
            ...     regex=False
            ... )
            >>>
            >>> # Enable regex pattern matching (regex=True, default)
            >>> prefix_filtered = index.filter_search_results(
            ...     initial_results,
            ...     {"path": r"^core_profiles/profiles_1d"}
            ... )
            >>>
            >>> # Documentation regex matching (regex=True, case-insensitive)
            >>> doc_pattern = index.filter_search_results(
            ...     initial_results,
            ...     {"documentation": r".*temperature.*"}
            ... )
            >>>
            >>> # Force exact matching even with special chars (regex=False)
            >>> literal_match = index.filter_search_results(
            ...     initial_results,
            ...     {"path": "core_profiles.*"},  # Matches literal ".*"
            ...     regex=False
            ... )
            >>>
            >>> # Complex regex patterns (regex=True, case-insensitive)
            >>> complex_pattern = index.filter_search_results(
            ...     initial_results,
            ...     {"path": r".*(temperature|density)$", "documentation": r".*ion.*"}
            ... )
        """
        if not filters:
            return search_results

        filtered_list = []

        for result in search_results:
            match = True

            for field_name, filter_value in filters.items():
                field_value = getattr(result, field_name)
                filter_str = str(filter_value)

                # When regex=True, enable regex matching detection
                # When regex=False, disable regex matching (use exact matching only)
                # Check if the string would be different when escaped
                # If it's different, it contains regex special characters
                if regex and re.escape(filter_str) != filter_str:
                    # Use regex matching
                    try:
                        if not re.search(filter_str, str(field_value), re.IGNORECASE):
                            match = False
                            break
                    except re.error as e:
                        logger.warning(
                            f"Invalid regex pattern '{filter_str}' for field '{field_name}': {e}. "
                            f"Using literal string matching."
                        )
                        # Do not match if regex fails
                        match = False
                        break
                else:
                    # Use exact string matching for simple strings
                    if field_value != filter_value:
                        match = False
                        break

            if match:
                filtered_list.append(result)

        return filtered_list


if __name__ == "__main__":  # pragma: no cover
    index = WhooshIndex(indexname="test_index")

    index.add_document_batch(
        [
            {
                "path": "pf_active/coil/location",
                "documentation": "coil position in the machine",
                "units": "m/s",
            },
            {
                "path": "pf_active/coil/name",
                "documentation": "coil name",
                "units": "none",
            },
        ]
    )

    with index.searcher() as searcher:
        for doc in searcher.documents():
            print(f"Document found: {doc}")

    # examples moved to data_dictionary_index
    print(1, index.search_by_keywords("name"))
    print(2, index.search_by_exact_path("pf_active/coil/name"))
    print(3, index.search_by_path_prefix("pf_active/coil"))
    print(3, index.search_by_path_prefix("pf_active/coil"))
    with index.searcher() as searcher:
        for doc in searcher.documents():
            print(f"Document found: {doc}")

    # examples moved to data_dictionary_index
    print(1, index.search_by_keywords("name"))
    print(2, index.search_by_exact_path("pf_active/coil/name"))
    print(3, index.search_by_path_prefix("pf_active/coil"))
    print(3, index.search_by_path_prefix("pf_active/coil"))
