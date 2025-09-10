"""Embeddings lifecycle management for the IMAS MCP server (synchronous version).

This refactored implementation performs embedding initialization *synchronously*
at construction time. The previous asynchronous background task model has been
removed to simplify downstream dependency assumptions – components that require
embeddings can now rely on them being fully available after `Embeddings` is
instantiated.

Key changes from async version:
* No status state-machine / background task
* Initialization happens in `__post_init__` (blocking)
* Status concept removed (always implicitly ready once constructed)
* `encode_texts` never triggers implicit (duplicate) initialization
"""

import hashlib
import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np

from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder
from imas_mcp.search.document_store import DocumentStore

logger = logging.getLogger(__name__)


EmbeddingStatus = Literal["ready"]  # Kept for type compatibility in public APIs


@dataclass
class Embeddings:
    """Manage synchronous embedding initialization and cache status.

    Responsibilities (synchronous mode):
    - Perform blocking initialization during construction
    - Provide expected cache file path & existence check
    - Expose model name & embeddings matrix for search components
    - Always report status ``"ready"`` once instance exists
    """

    document_store: DocumentStore
    ids_set: set[str] | None = None
    use_rich: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    load_embeddings: bool = True  # if True, build document embeddings in __post_init__ (was defer_build False)

    # Internal state
    _normalize_embeddings: bool = True
    _use_half_precision: bool = False

    # Blocking work (executed in thread via to_thread)
    # Internal embedding state
    _encoder: Encoder | None = field(init=False, default=None, repr=False)
    _embeddings: np.ndarray | None = field(init=False, default=None, repr=False)
    _path_ids: list[str] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:  # noqa: D401 - initialization orchestrator
        """Create encoder/config; optionally build document embeddings immediately.

        Renamed flag: ``defer_build`` -> ``load_embeddings`` (inverted semantics).
        If ``load_embeddings`` is False, only the encoder is constructed and
        embeddings are built lazily upon first access that requires them.
        """
        config = EncoderConfig(
            model_name=self.model_name,
            ids_set=self.ids_set,
            use_rich=self.use_rich,
            normalize_embeddings=self._normalize_embeddings,
            use_half_precision=self._use_half_precision,
        )
        self._encoder = Encoder(config=config)
        if self.load_embeddings:
            self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load or generate document embeddings."""
        if self._embeddings is not None:
            return
        documents = self.document_store.get_all_documents()
        if not documents:
            logger.info("No documents found; embeddings matrix will be empty")
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            self._path_ids = []
            return
        texts = [d.embedding_text for d in documents]
        identifiers = [d.metadata.path_id for d in documents]
        cache_key = self._encoder.config.generate_cache_key()  # type: ignore[union-attr]
        embeddings, path_ids, was_cached = self._encoder.build_document_embeddings(  # type: ignore[union-attr]
            texts=texts,
            identifiers=identifiers,
            cache_key=cache_key,
            force_rebuild=False,
            source_data_dir=self.document_store._data_dir,
        )
        self._embeddings = embeddings
        self._path_ids = path_ids
        status = "loaded from cache" if was_cached else "generated"
        logger.info(
            f"Embeddings {status}: {embeddings.shape[0]} documents, dim={embeddings.shape[1] if embeddings.size else 0}"
        )

    # Cache helpers -----------------------------------------------------
    def cache_filename(self) -> str:
        """Derive expected cache filename using same scheme as EmbeddingManager."""
        model_name_sanitized = self.model_name.split("/")[-1].replace("-", "_")
        config_parts: list[str] = [
            f"norm_{self._normalize_embeddings}",
            f"half_{self._use_half_precision}",
        ]
        if self.ids_set:
            config_parts.append(f"ids_{'_'.join(sorted(self.ids_set))}")
        if self.ids_set:  # filtered dataset -> hashed filename
            config_str = "_".join(config_parts)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            return f".{model_name_sanitized}_{config_hash}.pkl"
        return f".{model_name_sanitized}.pkl"

    def cache_path(self) -> Path:
        """Return absolute path to expected cache file (under resources)."""
        base = resources.files("imas_mcp") / "resources" / "embeddings"
        embeddings_dir = Path(str(base))
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        return embeddings_dir / self.cache_filename()

    def cache_exists(self) -> bool:
        """True if expected cache file exists on disk."""
        return self.cache_path().exists()

    # Convenience for health/readiness endpoints retained via constant property
    @property
    def effective_status(self) -> EmbeddingStatus:  # pragma: no cover - trivial
        return "ready"

    # Accessors for semantic search
    def get_embeddings_matrix(self) -> np.ndarray:
        if self._embeddings is None:
            # Lazy build if deferred
            if not self.load_embeddings:
                self._load_embeddings()
            else:
                raise RuntimeError("Embeddings not yet initialized (unexpected)")
        assert self._embeddings is not None  # for type checkers
        return self._embeddings

    def get_path_ids(self) -> list[str]:
        if self._embeddings is None:
            return []
        return self._path_ids

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not self._encoder:  # pragma: no cover - defensive
            raise RuntimeError("Encoder missing (unexpected in sync mode)")
        return self._encoder.embed_texts(texts)  # type: ignore[union-attr]

    @property
    def encoder_config(self) -> EncoderConfig:
        if not self._encoder:  # pragma: no cover - defensive
            raise RuntimeError("Encoder not initialized")
        return self._encoder.config  # type: ignore[return-value]

    @property
    def is_built(self) -> bool:
        """Return True if document embeddings have been built (matrix available)."""
        return self._embeddings is not None and self._embeddings.size > 0

    # Public explicit trigger -------------------------------------------------
    def materialize_embeddings(self) -> None:
        """Explicitly trigger embedding load/generation if not already built.

        Public method renamed from ``load_embeddings`` to avoid name collision
        with new boolean configuration flag ``load_embeddings``.
        """
        self._load_embeddings()


# Registry removed previously; synchronous Embeddings must be explicitly constructed and injected.
