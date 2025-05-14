from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Set, Type, TypeVar, Union


import pydantic
import whoosh
import whoosh.analysis
import whoosh.fields
import whoosh.index
import whoosh.qparser
import whoosh.writing
import whoosh.searching  # Added import
import whoosh.query  # Added import

from imas_mcp_server import pint


# Base model for document validation
class IndexableDocument(pydantic.BaseModel):
    """Base model for documents that can be indexed in WhooshIndex."""

    class Config:
        extra = "forbid"  # Prevent additional fields not in schema
        validate_assignment = True


class DataDictionaryEntry(IndexableDocument):
    """IMAS Data Dictionary document model for validating Whoosh documents."""

    path: str
    documentation: str
    units: str = ""

    ids: Optional[str] = None
    path_segments: Optional[str] = None

    @pydantic.field_validator("units", mode="after")
    @classmethod
    def parse_units(cls, units: str) -> str:
        """Return units formatted as custom UDUNITS."""
        return f"{pint.Unit(units):~F}"

    @pydantic.model_validator(mode="after")
    def update_fields(self) -> "DataDictionaryEntry":
        """Update unset fields."""
        if self.ids is None:
            self.ids = self.path.split("/")[0]
        if self.path_segments is None:  # Updated to use self.path_segments
            self.path_segments = " ".join(self.path.split("/"))
        return self


class SearchResult(pydantic.BaseModel):
    """Model for storing a single search result from Whoosh."""

    path: str
    score: float
    documentation: str
    units: str
    ids: str
    highlights: str = ""

    @classmethod
    def from_hit(cls, hit: whoosh.searching.Hit) -> "SearchResult":
        """Create a SearchResult instance from a Whoosh Hit object."""
        return cls(
            path=hit["path"],
            score=hit.score if hit.score is not None else 0.0,
            documentation=hit.get("documentation", ""),
            units=hit.get("units", ""),
            ids=hit.get("ids", ""),
            highlights=hit.highlights("documentation", ""),
        )

    @classmethod
    def from_document(cls, document: Dict[str, Any]) -> "SearchResult":
        """Create a SearchResult instance from a Whoosh document dictionary."""
        return cls(
            path=document["path"],
            score=1.0,  # Exact match, so score is 1.0
            documentation=document.get("documentation", ""),
            units=document.get("units", ""),
            ids=document.get("ids", ""),
            highlights="",  # No highlights for direct document retrieval
        )

    def __str__(self) -> str:
        """Return a string representation of the SearchResult."""
        doc_preview = (
            self.documentation[:100] + "..."
            if len(self.documentation) > 100
            else self.documentation
        )
        lines = [
            f"Path: {self.path}",
            f"  Score: {self.score:.4f}",
            f"  IDS: {self.ids if self.ids else 'N/A'}",
            f"  Units: {self.units if self.units else 'N/A'}",
            f"  Documentation: {doc_preview}",
        ]
        if self.highlights:  # Check if highlights string is not empty
            lines.append(f"  Highlights: {self.highlights}")
        return "\\n".join(lines)  # Corrected newline character

    class Config:
        extra = "forbid"
        validate_assignment = True


# Type variable for generic model support
T = TypeVar("T", bound=IndexableDocument)


@dataclass
class WhooshIndex(Generic[T]):
    """Index class for creating and managing a Whoosh index for IMAS DD entries."""

    indexname: str = "_MAIN"
    dirname: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "index"
    )
    schema: Optional[whoosh.fields.Schema] = field(default=None, repr=False)
    document_model: Type[T] = field(default=DataDictionaryEntry)  # type: ignore

    _index: whoosh.index.FileIndex = field(init=False, repr=False)
    _writer: Optional[whoosh.writing.IndexWriter] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize Whoosh schema and index."""
        self.schema = self._get_schema()
        self._index = self._get_index()

    def __len__(self) -> int:
        """Return the number of documents in the Whoosh index."""
        with self._index.searcher() as searcher:
            return searcher.doc_count()

    def _get_schema(self) -> whoosh.fields.Schema:
        """Return the Whoosh schema."""
        if self.schema is not None:
            return self.schema
        return whoosh.fields.Schema(
            path=whoosh.fields.ID(
                stored=True, unique=True
            ),  # The full IDS path as unique ID
            documentation=whoosh.fields.TEXT(
                stored=True, analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Documentation content
            units=whoosh.fields.KEYWORD(stored=True),  # Units of the documentation
            ids=whoosh.fields.ID(stored=True),  # The root IDS
            path_segments=whoosh.fields.TEXT(  # Renamed from segments
                analyzer=whoosh.analysis.StemmingAnalyzer()
            ),  # Individual IDS path segments
        )

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
        return whoosh.index.create_in(self.dirname, self.schema, self.indexname)

    def _validate_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document against the Pydantic model.

        Args:
            document: Dictionary containing fields to be indexed.

        Returns:
            Validated dictionary with properly formatted fields.

        Raises:
            pydantic.ValidationError: If document does not match the model schema.
        """
        validated = self.document_model(**document)
        return validated.model_dump()

    @contextmanager
    def writer(self):
        """Yield a Whoosh index writer for batch add operations."""
        self._writer = self._index.writer(procs=4, limitmb=256, multisegment=True)
        yield self._writer
        assert self._writer is not None, "Writer is not initialized."
        self._writer.commit()
        self._writer = None

    def __add__(self, document: Dict[str, Any]) -> "WhooshIndex[T]":
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
        validated_doc = self._validate_document(document)
        self._writer.update_document(**validated_doc)  # upsert
        return self

    def __iadd__(self, document: Dict[str, Any]) -> "WhooshIndex[T]":
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

    def add_document(self, document: Dict[str, Any]):
        """Add a single document to the index using the writer context.\

        Args:
            document: Dictionary containing fields to be indexed.
        """
        with self.writer():
            self += document

    def add_document_batch(self, documents: list[Dict[str, Any]]):
        """Add a batch of documents to the index using the writer context.\

        Args:
            documents: A list of dictionaries, each containing fields to be indexed.
        """
        with self.writer():
            for doc in documents:
                self += doc

    @contextmanager
    def searcher(self):
        """Yield a Whoosh index searcher for querying the index."""
        with self._index.searcher() as searcher:
            yield searcher

    def ids_set(self) -> Set[str]:
        """Return the set of IDS names from the index."""
        with self.searcher() as searcher:
            return {
                doc["ids"]
                for doc in searcher.documents()
                if "ids" in doc and doc["ids"] is not None
            }

    def search_by_keywords(
        self,
        query_str: str,
        page_size: int = 10,
        page: int = 1,
        enable_fuzzy: bool = False,
        search_fields: Optional[list[str]] = None,
        sort_by: Optional[Union[str, list[str]]] = None,
        sort_reverse: bool = False,
    ) -> list[SearchResult]:
        """
        Search the index for paths matching the given keywords.

        Wildcards (e.g., 'term*', 't?rm') are generally supported by default
        in the query string as long as the field's analysis chain is compatible.

        Args:
            query_str: Natural language query.
                Can include:
                - Field prefixes (e.g., "documentation:plasma ids:core_profiles")
                - Field-specific boosts (e.g., "documentation^2.0 plasma ^0.5")
                - Wildcards (e.g., "doc* current?")
                - Boolean operators (e.g., "density AND NOT temperature")
                - Phrases (e.g., "\"ion temperature\"")
            page_size: Maximum number of results per page.
            page: Page number to retrieve (1-based).
            enable_fuzzy: Enable fuzzy term matching (e.g., "temperture~" for "temperature").
            search_fields: List of fields to search in. Defaults to ["documentation", "path_segments"].
            sort_by: Field name or list of field names to sort by.
            sort_reverse: Whether to reverse the sort order.

        Returns:
            List of SearchResult objects.

        Examples:
            >>> # Basic keyword search
            >>> index.search_by_keywords("plasma current")

            >>> # Search with field prefix and pagination
            >>> index.search_by_keywords("documentation:ion", page_size=5, page=2)

            >>> # Search with wildcard
            >>> index.search_by_keywords("core_profiles/prof*")

            >>> # Search with fuzzy matching enabled
            >>> index.search_by_keywords("electrn densty", enable_fuzzy=True)

            >>> # Search with field boosting in the query string
            >>> index.search_by_keywords("documentation^3.0 equilibrium ^0.5 reconstruction")

            >>> # Complex query with boolean operators, phrases, and boosts
            >>> index.search_by_keywords('ids:summary AND (documentation:"ion temperature"^2.0 OR :elect*)')

            >>> # Search specific fields and sort results
            >>> index.search_by_keywords("data", search_fields=["documentation"], sort_by="path", sort_reverse=True)
        """
        results = []

        if search_fields is None:
            search_fields = ["documentation", "path_segments"]

        with self.searcher() as searcher:
            parser = whoosh.qparser.MultifieldParser(search_fields, self.schema)

            if enable_fuzzy:
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
        self, search_results: list[SearchResult], filters: Dict[str, Any]
    ) -> list[SearchResult]:
        """
        Filter a list of SearchResult objects based on exact field values.

        Args:
            search_results: A list of SearchResult objects to filter.
            filters: A dictionary where keys are field names (str) present in
                     SearchResult (e.g., "ids", "units", "") and values
                     are the desired exact values for those fields.

        Returns:
            A new list containing only the SearchResult objects that match
            all specified filter criteria.

        Examples:
            >>> # Assume 'index' is an instance of WhooshIndex
            >>> # First, get some initial results
            >>> initial_results = index.search_by_keywords("temperature")
            >>>
            >>> # Filter for results specifically from the 'core_profiles' IDS
            >>> filtered_by_ids = index.filter_search_results(
            ...     initial_results, {"ids": "core_profiles"}
            ... )
            >>>
            >>> # Further filter the already filtered results by units
            >>> specific_units_results = index.filter_search_results(
            ...     filtered_by_ids, {"units": "eV"}
            ... )
            >>>
            >>> # Alternatively, filter by multiple criteria at once from initial results
            >>> multi_criteria_results = index.filter_search_results(
            ...     initial_results, {"ids": "core_profiles", "units": "eV"}
            ... )
        """
        if not filters:
            return search_results  # Return original list if no filters are applied

        filtered_list = []
        for result in search_results:
            match = True
            for field_name, filter_value in filters.items():
                # Check if the SearchResult object has the attribute and if it matches
                if (
                    not hasattr(result, field_name)
                    or getattr(result, field_name) != filter_value
                ):
                    match = False
                    break  # No need to check other filters for this result
            if match:
                filtered_list.append(result)
        return filtered_list


if __name__ == "__main__":  # pragma: no cover

    index = WhooshIndex(indexname="test_index")

    index.add_document(
        {
            "path": "pf_active/coil/location",
            "documentation": "coil position in the machine",
            "units": "m/s",
        }
    )

    with index.searcher() as searcher:
        for doc in searcher.documents():
            print(f"Document found: {doc}")

    print(1, index.search_by_keywords("name"))
    print(2, index.search_by_exact_path("pf_active/coil/name"))
    print(3, index.search_by_path_prefix("pf_active/coil"))
