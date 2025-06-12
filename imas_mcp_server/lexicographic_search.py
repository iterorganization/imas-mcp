import logging
from dataclasses import InitVar, dataclass
from typing import Final, Literal

from imas_mcp_server.data_dictionary_index import DataDictionaryIndex
from imas_mcp_server.whoosh_index import WhooshIndex


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
        if auto_build and len(self) == 0:
            self.build_index()

    @property
    def index_prefix(self) -> IndexPrefixT:
        """Return the type of resource."""
        return self.INDEX_PREFIX

    def build_index(self):
        """Build the lexicographic search index."""
        for document_batch in self._get_document_batch():
            self.add_document_batch(document_batch)
        logger.info("Index building completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Use a smaller subset for testing to avoid hanging on large datasets
    index = LexicographicSearch()

    # index.build_index()
    print(index.search_by_exact_path("pf_active/coil/name"))
