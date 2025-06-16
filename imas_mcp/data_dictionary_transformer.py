"""
Handles the creation and querying of a semantic vector index for IMAS Data Dictionary.
"""

import dataclasses
import datetime
import hashlib
import logging
import shutil
import time
from typing import Any, Dict, List, Optional, Set, Union

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api import ClientAPI  # Changed import for ClientAPI
from chromadb.types import Metadata
from pydantic import ValidationError

from imas_mcp.whoosh_index import DataDictionaryEntry, SearchResult
from imas_mcp.data_dictionary_index import DataDictionaryBase  # Renamed

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Or your preferred default
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DataDictionaryTransformer(DataDictionaryBase):  # Renamed
    """
    Manages a vector database for semantic search on IMAS Data Dictionary entries
    using ChromaDB and sentence transformers.
    """

    index_name_prefix: str = "semantic_dd_transformer"
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL

    _client: Optional[ClientAPI] = dataclasses.field(
        init=False, default=None
    )  # Changed type to ClientAPI
    _collection: Optional[chromadb.Collection] = dataclasses.field(
        init=False, default=None
    )
    _embedding_function: Optional[
        embedding_functions.SentenceTransformerEmbeddingFunction
    ] = dataclasses.field(init=False, default=None)
    _path_to_details_cache: Dict[str, SearchResult] = dataclasses.field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Initializes the ChromaDB client, embedding function, and collection.
        Also loads the path-to-details cache if the index is already built.
        """
        super().__post_init__()  # This creates self.dirname

        try:
            if not isinstance(self.embedding_model_name, str):
                raise ValueError("embedding_model_name must be a string")
            self._embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
            )
            logger.info(
                f"Initialized embedding function with model: {self.embedding_model_name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize SentenceTransformerEmbeddingFunction: {e}. "
                "Ensure 'sentence-transformers' is installed ('pip install sentence-transformers')"
            )
            raise

        chroma_db_dir = self.dirname / self.full_index_name
        chroma_db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChromaDB persistent path: {chroma_db_dir}")

        self._client = chromadb.PersistentClient(path=str(chroma_db_dir))

        # Sanitize collection name for ChromaDB
        raw_collection_name = self.full_index_name
        # Rule 1: Replace invalid characters
        sanitized_name = "".join(
            c if c.isalnum() or c in ("_", "-", ".") else "_"
            for c in raw_collection_name
        )
        # Rule 2: Cannot contain '..'
        while ".." in sanitized_name:
            sanitized_name = sanitized_name.replace(
                "..", "._"
            )  # Replace with valid sequence
        # Rule 3 & 4: Length 3-63, start/end with alphanumeric
        if not (
            3 <= len(sanitized_name) <= 63
            and sanitized_name[0].isalnum()
            and sanitized_name[-1].isalnum()
        ):
            hashed_name = hashlib.md5(raw_collection_name.encode("utf-8")).hexdigest()
            # Truncate hash to fit "col_" prefix and ensure total length <= 63
            # Max hash part length = 63 - 4 (for "col_") = 59
            final_collection_name = "col_" + hashed_name[:59]

            # Ensure start/end alphanumeric after hashing/prefixing (though hash is hex)
            if not final_collection_name[0].isalnum():  # Should not happen with "col_"
                final_collection_name = "a" + final_collection_name[1:]
            if not final_collection_name[-1].isalnum():  # Hash is hex, so this is fine
                pass  # No change needed for typical hash

            # Ensure length again after potential modifications (unlikely here)
            final_collection_name = final_collection_name[:63]
            if len(final_collection_name) < 3:  # Pad if somehow too short
                final_collection_name = (final_collection_name + "xxx")[:3]

            logger.warning(
                f"Original collection name '{raw_collection_name}' was "
                f"sanitized/hashed to '{final_collection_name}' due to ChromaDB constraints."
            )
        else:
            final_collection_name = sanitized_name

        # Ensure final_collection_name is valid one last time (mostly for length if original was short)
        if len(final_collection_name) < 3:
            final_collection_name = (final_collection_name + "pad")[:3]
        if len(final_collection_name) > 63:
            final_collection_name = final_collection_name[:63]

        try:
            if self._embedding_function is None:
                raise ValueError(
                    "Embedding function not initialized before creating collection."
                )
            if self._client is None:  # Should be initialized by now
                raise ValueError("ChromaDB client not initialized.")

            self._collection = self._client.get_or_create_collection(
                name=final_collection_name,
                embedding_function=self._embedding_function,  # type: ignore
                # Chroma's type hints for embedding_function might be too generic.
                # If SentenceTransformerEmbeddingFunction is compatible, this ignore is okay.
                metadata={"hnsw:space": "cosine"},  # Example, use if relevant
            )
            logger.info(
                f"ChromaDB collection '{final_collection_name}' loaded/created."
            )
        except Exception as e:
            logger.error(
                f"Failed to get or create ChromaDB collection '{final_collection_name}': {e}"
            )
            raise

        if self._is_index_built():
            self._load_path_to_details_cache()
        else:
            logger.info(
                f"Index {self.full_index_name} (collection: {self._collection.name if self._collection else 'N/A'}) "
                "is not built or is empty. Call build_index()."
            )

    def _is_index_built(self) -> bool:
        """Checks if the ChromaDB collection exists and has entries."""
        if self._collection is None:
            return False
        try:
            return int(self._collection.count()) > 0  # Cast to int
        except Exception as e:
            logger.error(f"Error checking collection count: {e}")
            return False

    def _get_path_cache_filename_suffix(self) -> str:
        return "_path_to_details_cache.json"

    def _load_path_to_details_cache(self) -> None:
        """Loads the path-to-details cache from a JSON file or reconstructs it from ChromaDB."""
        logger.info(f"Loading path-to-details cache for {self.full_index_name}...")

        loaded_data = self._load_metadata(self._get_path_cache_filename_suffix())
        if loaded_data and isinstance(loaded_data, dict):
            try:
                temp_cache: Dict[str, SearchResult] = {}
                for path, details_dict in loaded_data.items():
                    if isinstance(details_dict, dict):
                        # Reconstruct SearchResult, handling potential missing fields gracefully
                        temp_cache[path] = SearchResult(
                            path=details_dict.get("path", path),
                            documentation=details_dict.get("documentation", ""),
                            units=details_dict.get("units", "none"),
                            score=details_dict.get("score"),
                            ids_name=details_dict.get("ids_name"),
                        )
                    else:
                        logger.warning(
                            f"Skipping malformed item in cache for path {path}: {details_dict}"
                        )
                self._path_to_details_cache = temp_cache
                logger.info(
                    f"Loaded {len(self._path_to_details_cache)} items into path cache from JSON."
                )
                return
            except TypeError as e:
                logger.warning(
                    f"Error reconstructing SearchResult from cache ({e}). Cache might be malformed."
                )
            except Exception as e:  # Catch any other unexpected errors
                logger.warning(
                    f"Could not load path cache from JSON due to an unexpected error: {e}. Will try to rebuild."
                )
        else:
            logger.info("Path cache JSON not found or invalid format.")

        logger.info("Attempting to reconstruct path cache from ChromaDB metadata.")
        self._path_to_details_cache = {}
        if self._collection and self._collection.count() > 0:
            try:
                # Fetch all items with their metadata
                results = self._collection.get(include=["metadatas"])
                if results and results["ids"] and results["metadatas"]:
                    for i, item_id in enumerate(results["ids"]):
                        meta = results["metadatas"][i]
                        if meta:  # Ensure metadata is not None
                            # item_id is the path
                            path_val = meta.get("path", item_id)
                            doc_val = meta.get("documentation_snippet", "")
                            units_val = meta.get("units", "none")
                            ids_name_val = meta.get("ids_name")

                            self._path_to_details_cache[item_id] = SearchResult(
                                path=str(path_val) if path_val is not None else item_id,
                                documentation=str(doc_val)
                                if doc_val is not None
                                else "",
                                units=str(units_val)
                                if units_val is not None
                                else "none",
                                ids_name=str(ids_name_val)
                                if ids_name_val is not None
                                else None,
                                # Score is not typically stored in Chroma metadata directly for all items
                                score=None,  # Explicitly set score as it's required by SearchResult
                            )
                        else:
                            logger.warning(
                                f"Missing metadata for item ID: {item_id} during cache reconstruction."
                            )
                    logger.info(
                        f"Reconstructed {len(self._path_to_details_cache)} items for path cache from ChromaDB."
                    )
            except Exception as e:
                logger.error(f"Error reconstructing path cache from ChromaDB: {e}")

        if not self._path_to_details_cache:
            logger.warning("Path cache is empty after attempting load/reconstruction.")

    def _save_path_to_details_cache(self) -> None:
        """Saves the current path-to-details cache to a JSON file."""
        logger.info(f"Saving path-to-details cache for {self.full_index_name}...")
        serializable_cache = {}
        for path, result_obj in self._path_to_details_cache.items():
            if isinstance(result_obj, SearchResult):
                # Pydantic models have .model_dump() for serialization
                serializable_cache[path] = result_obj.model_dump(exclude_none=True)
            else:
                logger.warning(
                    f"Item for path '{path}' is not a SearchResult object. Skipping."
                )

        if serializable_cache:
            self._save_metadata(
                self._get_path_cache_filename_suffix(), serializable_cache
            )
            logger.info(f"Saved {len(serializable_cache)} items to path cache JSON.")
        else:
            logger.warning("Path cache was empty or non-serializable. Not saving.")

    def build_index(self) -> None:
        """
        Builds the semantic search index by processing data dictionary entries,
        embedding their documentation, and adding them to ChromaDB.
        """
        if self._is_index_built() and self._path_to_details_cache:
            logger.info(
                f"Semantic index '{self.full_index_name}' is already built with {len(self)} items. "
                "Skipping build. Call delete_index() first to rebuild."
            )
            return

        if self._collection is None:
            logger.error("ChromaDB collection is not initialized. Cannot build index.")
            # Attempt to re-initialize, or raise a more specific error
            # This state should ideally be prevented by __post_init__
            raise RuntimeError("ChromaDB collection not available for building index.")

        logger.info(f"Building semantic index: {self.full_index_name}")
        start_build_time = time.time()

        documents_to_embed: List[str] = []
        metadatas_for_chroma: List[Metadata] = []  # Changed type to List[Metadata]
        ids_for_chroma: List[str] = []

        self._path_to_details_cache.clear()
        processed_count = 0
        batch_size = 100  # Adjust batch size as needed for performance

        all_ids_names: Set[str] = set()

        for entry_dict in self._get_document_entries_from_dd():
            try:
                entry = DataDictionaryEntry(**entry_dict)
                # if not current_dd_version and entry.dd_version: # Removed dd_version access
                #     current_dd_version = entry.dd_version # Removed dd_version access
                if entry.ids_name:
                    all_ids_names.add(entry.ids_name)

            except ValidationError as e:
                logger.warning(
                    f"Skipping entry due to validation error: {e}. Entry: {entry_dict}"
                )
                continue

            # Use path as the unique ID for ChromaDB and cache
            doc_id = entry.path

            # Prepare data for ChromaDB
            # Document for embedding: typically the main text content
            documents_to_embed.append(
                entry.documentation if entry.documentation else ""
            )

            # Metadata for ChromaDB (store relevant, queryable, or filterable fields)
            # Keep metadata light for Chroma; full details are in _path_to_details_cache
            chroma_meta: Dict[
                str, Union[str, int, float, bool, None]
            ] = {  # Ensure values match ChromaDB's Metadata type
                "path": entry.path,
                "units": entry.units if entry.units else "none",
                "ids_name": entry.ids_name if entry.ids_name else "unknown",
                # dd_version: entry.dd_version if entry.dd_version else "unknown", # Removed dd_version access
                # Add a snippet of documentation if full doc is too long for direct meta
                "documentation_snippet": (
                    entry.documentation[:200] + "..."
                    if entry.documentation
                    and len(entry.documentation)
                    > 200  # Check if documentation is not None
                    else entry.documentation
                )
                if entry.documentation  # Check if documentation is not None
                else "",
            }
            metadatas_for_chroma.append(chroma_meta)
            ids_for_chroma.append(doc_id)

            # Populate the path-to-details cache
            self._path_to_details_cache[doc_id] = SearchResult(
                path=entry.path,
                documentation=entry.documentation
                if entry.documentation
                else "",  # Ensure documentation is not None
                units=entry.units
                if entry.units
                else "none",  # Ensure units is not None
                ids_name=entry.ids_name
                if entry.ids_name
                else "unknown",  # Ensure ids_name is not None
                score=None,
            )
            processed_count += 1

            if processed_count % batch_size == 0:
                if documents_to_embed:  # Ensure there's something to add
                    self._collection.add(
                        ids=ids_for_chroma,
                        documents=documents_to_embed,
                        metadatas=metadatas_for_chroma,
                    )
                    logger.info(
                        f"Added batch of {len(ids_for_chroma)} items to ChromaDB. Total: {processed_count}"
                    )
                documents_to_embed, metadatas_for_chroma, ids_for_chroma = (
                    [],
                    [],
                    [],
                )  # Reset for next batch

        # Add any remaining items
        if documents_to_embed:
            self._collection.add(
                ids=ids_for_chroma,
                documents=documents_to_embed,
                metadatas=metadatas_for_chroma,
            )
            logger.info(
                f"Added final batch of {len(ids_for_chroma)} items to ChromaDB. Total: {processed_count}"
            )

        if processed_count > 0:
            self._save_path_to_details_cache()
            # Save build metadata (e.g., build time, number of items, version)
            build_metadata = {
                "build_timestamp": datetime.datetime.utcnow().isoformat(),
                # "data_dictionary_version": current_dd_version,  # Removed dd_version access
                "ids_names_in_index": sorted(list(all_ids_names)),
                "total_items_indexed": processed_count,
                "embedding_model_name": self.embedding_model_name,
                "index_name": self.full_index_name,
                "build_duration_seconds": time.time() - start_build_time,
            }
            self._save_build_metadata(build_metadata)  # Uses method from base
            logger.info(
                f"Successfully built semantic index '{self.full_index_name}' with {processed_count} items "
                f"in {time.time() - start_build_time:.2f} seconds."
            )
        else:
            logger.warning(
                f"No documents were processed or added to the index '{self.full_index_name}'."
            )

    def __len__(self) -> int:
        """Returns the number of items in the index."""
        if self._collection:
            try:
                return int(self._collection.count())  # Cast to int
            except Exception as e:
                logger.error(f"Error getting collection count for __len__: {e}")
                return 0  # Or raise
        return 0

    def search_by_exact_path(self, path: str) -> Optional[SearchResult]:
        """
        Retrieves a document by its exact path from the cache.

        Args:
            path: The exact path of the document to retrieve.

        Returns:
            A SearchResult object if found, otherwise None.
        """
        return self._path_to_details_cache.get(path)

    def search_by_keywords(
        self,
        query_str: str,  # Matches base class
        limit: int = 10,  # Matches base class
        ids_name_filter: Optional[List[str]] = None,  # Matches base class
        dd_version_filter: Optional[str] = None,  # Matches base class
    ) -> List[SearchResult]:
        """
        Performs a semantic search for documents matching the given keywords.

        Args:
            query_str: The search query string.
            limit: The maximum number of results to return.
            ids_name_filter: Optional list of IDS names to filter results by.
            dd_version_filter: Optional data dictionary version to filter by.

        Returns:
            A list of SearchResult objects matching the query, ordered by relevance.
        """
        if not self._is_index_built() or self._collection is None:
            logger.warning(
                f"Index '{self.full_index_name}' is not built or collection is unavailable. Cannot search."
            )
            return []

        if not query_str:  # Check query_str instead of keywords
            return []

        where_filter: Dict[str, Any] = {}
        if ids_name_filter:
            if len(ids_name_filter) == 1:
                where_filter["ids_name"] = ids_name_filter[0]
            else:
                where_filter["ids_name"] = {"$in": ids_name_filter}

        if dd_version_filter:
            if where_filter:
                if "$and" not in where_filter:
                    existing_filter_value = where_filter.pop("ids_name", None)
                    if existing_filter_value is not None:
                        where_filter["$and"] = [{"ids_name": existing_filter_value}]
                    else:
                        where_filter["$and"] = []
                elif not isinstance(where_filter["$and"], list):
                    logger.warning(
                        "where_filter['$and'] was not a list, reinitializing."
                    )
                    where_filter["$and"] = []
                where_filter["$and"].append({"dd_version": dd_version_filter})
            else:
                where_filter["dd_version"] = dd_version_filter

        try:
            query_results = self._collection.query(
                query_texts=[query_str],  # Use query_str
                n_results=limit,
                where=where_filter if where_filter else None,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Error during ChromaDB query: {e}")
            return []

        search_results: List[SearchResult] = []
        if query_results and query_results["ids"] and query_results["ids"][0]:
            result_ids = query_results["ids"][0]
            distances_list = query_results.get("distances")
            distances = (
                distances_list[0]
                if distances_list
                and len(distances_list) > 0
                and distances_list[0] is not None
                else [None] * len(result_ids)
            )

            for i, doc_id in enumerate(result_ids):
                cached_item = self._path_to_details_cache.get(doc_id)
                if cached_item:
                    current_distance = distances[i]
                    score = (
                        (1.0 - current_distance)
                        if isinstance(current_distance, (float, int))
                        else None
                    )
                    search_results.append(
                        SearchResult(
                            path=cached_item.path,
                            documentation=cached_item.documentation,
                            units=cached_item.units,
                            ids_name=cached_item.ids_name,
                            # dd_version=cached_item.dd_version, # Removed
                            score=score,
                        )
                    )
                else:
                    logger.warning(
                        f"Document ID '{doc_id}' from search results not found in path cache."
                    )

        return search_results

    def delete_index(self) -> None:
        """
        Deletes the ChromaDB collection, its persistent storage directory,
        and associated metadata files.
        """
        logger.info(f"Attempting to delete index: {self.full_index_name}")

        # 1. Delete ChromaDB collection
        if self._client and self._collection:
            try:
                logger.info(f"Deleting ChromaDB collection: {self._collection.name}")
                self._client.delete_collection(name=self._collection.name)
                logger.info(
                    f"Successfully deleted ChromaDB collection: {self._collection.name}"
                )
            except Exception as e:  # Catch specific chromadb errors if possible
                logger.error(
                    f"Error deleting ChromaDB collection '{self._collection.name}': {e}"
                )

        # 2. Delete the ChromaDB persistent storage directory
        chroma_db_dir = self.dirname / self.full_index_name
        if chroma_db_dir.exists():
            try:
                shutil.rmtree(chroma_db_dir)
                logger.info(f"Successfully deleted ChromaDB directory: {chroma_db_dir}")
            except OSError as e:
                logger.error(
                    f"Error deleting Chroma DB directory '{chroma_db_dir}': {e}"
                )

        # 3. Delete specific metadata/cache files managed by this class
        path_cache_file = (
            self.dirname
            / f"{self.full_index_name}{self._get_path_cache_filename_suffix()}"
        )  # Corrected f-string
        if path_cache_file.exists():
            try:
                path_cache_file.unlink()
                logger.info(f"Deleted path cache file: {path_cache_file}")
            except OSError as e:
                logger.error(f"Error deleting path cache file '{path_cache_file}': {e}")

        # 4. Call super to delete common metadata files (like _build_metadata.json)
        super().delete_index()

        # 5. Reset internal state
        self._collection = None
        # self._client = None # Client might be reusable if other collections exist for it,
        # but for a dedicated dir per index, it's safer to clear.
        # If PersistentClient is path-specific, it's fine.
        self._path_to_details_cache.clear()
        logger.info(f"Index '{self.full_index_name}' deleted and state reset.")

    def get_indexed_ids_names(self) -> List[str]:
        """
        Retrieves a list of unique IDS names present in the current index.
        This relies on the build metadata.
        """
        metadata = self._load_build_metadata()
        if metadata and "ids_names_in_index" in metadata:
            ids_names = metadata["ids_names_in_index"]
            if isinstance(ids_names, list):
                return ids_names
            logger.warning("'ids_names_in_index' in metadata is not a list.")
        logger.warning("Build metadata not found or does not contain IDS names list.")
        # Fallback: try to derive from cache if metadata is missing (less efficient)
        if self._path_to_details_cache:
            ids_names_set: Set[str] = set()
            for item in self._path_to_details_cache.values():
                if item.ids_name:
                    ids_names_set.add(item.ids_name)
            if ids_names_set:
                logger.info("Derived IDS names from cache as fallback.")
                return sorted(list(ids_names_set))
        return []

    def get_indexed_dd_versions(self) -> List[str]:
        """
        Retrieves a list of unique Data Dictionary versions present in the current index.
        This relies on the build metadata.
        """
        # Assuming 'data_dictionary_version' in build_metadata stores a single version string
        # or a list if multiple DDs were somehow indexed together (unlikely for this design).
        # metadata = self._load_build_metadata() # Unused variable
        # if metadata and "data_dictionary_version" in metadata: # Removed dd_version access
        #     # dd_version_val = metadata["data_dictionary_version"] # Removed
        #     # if isinstance(dd_version_val, str): # Removed
        #     #     return [dd_version_val] # Removed
        #     # if isinstance(dd_version_val, list): # Removed
        #     #     return [str(v) for v in dd_version_val if isinstance(v, (str, int, float))] # Removed
        #     # logger.warning( # Removed
        #     #     "'data_dictionary_version' in metadata is not a string or list." # Removed
        #     # ) # Removed
        #     # Return an empty list or handle as appropriate if dd_version is no longer stored
        #     # For now, let's assume we might still want to get this from metadata if it exists
        #     # but it's not part of SearchResult anymore.
        #     # This method might need rethinking if dd_version is truly gone.
        #     # For now, let's make it return an empty list as SearchResult doesn't hold it anymore.
        #     pass # Let's re-evaluate this method's purpose later if needed.

        logger.warning(
            "Build metadata not found or 'data_dictionary_version' key is missing (expected as dd_version is removed)."
        )
        return []
