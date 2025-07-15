import logging
from dataclasses import InitVar, dataclass
from typing import Final, Literal

from imas_mcp.data_dictionary_index import DataDictionaryIndex
from imas_mcp.whoosh_index import WhooshIndex

IndexPrefixT = Literal["lexicographic"]

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class LexicographicSearch(WhooshIndex, DataDictionaryIndex):
    """Specialized search tools using Whoosh index for IMAS Data Dictionary entries."""

    INDEX_PREFIX: Final[IndexPrefixT] = "lexicographic"

    auto_build: InitVar[bool] = True

    def __post_init__(self, auto_build: bool) -> None:
        super().__post_init__()
        if auto_build:
            current_count = len(self)
            if current_count == 0:
                logger.info("Index is empty, starting build process")
                self.build_index()
            else:
                logger.info(
                    f"Index already exists with {current_count} documents, skipping build"
                )
        else:
            logger.info("Auto-build disabled, index will not be built automatically")

    @property
    def index_prefix(self) -> IndexPrefixT:
        """Return the type of resource."""
        return self.INDEX_PREFIX

    def build_index(self):
        """Build the lexicographic search index."""
        batch_count = 0
        total_documents = 0

        # Pass use_rich parameter to get clean progress display
        use_rich_flag = self.use_rich if self.use_rich is not None else True
        for document_batch in self._get_document_batch(use_rich=use_rich_flag):
            batch_count += 1
            batch_size = len(document_batch)
            total_documents += batch_size
            self.add_document_batch(document_batch)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Use a smaller subset for testing to avoid hanging on large datasets
    index = LexicographicSearch(auto_build=False, ids_set={"pf_active"})

    index.build_index()
    print(index.search_by_exact_path("pf_active/coil/name"))
